"""
agents/multi_search.py
Multi Search Layer — executes DuckDuckGo + Bing in parallel.

- No single point of failure: if one provider fails, others continue
- Merges and deduplicates results by URL
- Returns structured URLCandidate list
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from core.config import get_settings
from core.state import ResearchState, URLCandidate
from agents.source_authority import authority_score, classify_source

logger = logging.getLogger(__name__)

# Domains that are high-value for business intelligence
_DOMAIN_SCORES = {
    "linkedin.com": 0.85,
    "crunchbase.com": 0.85,
    "bloomberg.com": 0.9,
    "reuters.com": 0.9,
    "techcrunch.com": 0.85,
    "forbes.com": 0.85,
    "businesswire.com": 0.85,
    "prnewswire.com": 0.8,
    "g2.com": 0.85,
    "trustpilot.com": 0.85,
    "glassdoor.com": 0.82,
    "reddit.com": 0.78,
    "clutch.co": 0.80,
    "capterra.com": 0.80,
    "twitter.com": 0.70,
    "x.com": 0.70,
    "wikipedia.org": 0.7,
    "sec.gov": 0.9,
    "wsj.com": 0.9,
    "ft.com": 0.9,
    "tracxn.com": 0.80,
    "zaubacorp.com": 0.82,
    "tofler.in": 0.82,
    "ambitionbox.com": 0.75,
    "builtwith.com": 0.80,
    "wappalyzer.com": 0.80,
}

# Domains to skip (ads, trackers, irrelevant aggregators)
_SKIP_DOMAINS = {
    "google.com", "bing.com", "yahoo.com", "duckduckgo.com",
    "amazon.com", "ebay.com", "instagram.com",
    "youtube.com", "tiktok.com",
}


def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _domain_score(domain: str, company_website: str = "") -> float:
    return authority_score(domain, company_website)


def _should_skip(url: str, domain: str) -> bool:
    if not url.startswith("http"):
        return True
    for skip in _SKIP_DOMAINS:
        if skip in domain:
            return True
    return False


# ── DuckDuckGo Search ─────────────────────────────────────────────────────────

async def _search_duckduckgo(queries: list[str], max_results_per_query: int = 5, company_website: str = "") -> list[URLCandidate]:
    """Search DuckDuckGo using the ddgs package."""
    results = []
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("[MultiSearch] ddgs not installed — skipping DuckDuckGo")
            return []

    # Run synchronously in thread to not block event loop
    def _run_ddgs():
        candidates = []
        ddgs = DDGS()
        for query in queries[:20]:  # balanced coverage across specialized workers
            try:
                hits = ddgs.text(query, max_results=max_results_per_query)
                for rank, hit in enumerate(hits or []):
                    url = hit.get("href") or hit.get("url", "")
                    domain = _get_domain(url)
                    if not url or _should_skip(url, domain):
                        continue
                    candidates.append(URLCandidate(
                        url=url,
                        domain=domain,
                        title=hit.get("title", ""),
                        snippet=hit.get("body", ""),
                        provider="duckduckgo",
                        rank=rank,
                        domain_score=_domain_score(url, company_website),
                        authority_score=authority_score(url, company_website),
                        source_category=classify_source(url, company_website),
                    ))
            except Exception as e:
                logger.warning(f"[MultiSearch] DuckDuckGo query failed: '{query}': {e}")
        return candidates

    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(None, _run_ddgs)
        logger.info(f"[MultiSearch] DuckDuckGo returned {len(results)} candidates")
    except Exception as e:
        logger.warning(f"[MultiSearch] DuckDuckGo failed: {e}")

    return results


# ── Bing Search via httpx ─────────────────────────────────────────────────────

async def _search_bing_httpx(queries: list[str], max_results_per_query: int = 5, company_website: str = "") -> list[URLCandidate]:
    """Search Bing by scraping the results page via httpx."""
    results = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=15,
        headers=headers,
    ) as client:
        for query in queries[:15]:  # balanced coverage across specialized workers
            try:
                resp = await client.get(
                    "https://www.bing.com/search",
                    params={"q": query, "count": max_results_per_query * 2},
                )
                # Parse Bing results
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "lxml")

                rank = 0
                for li in soup.select("li.b_algo")[:max_results_per_query]:
                    a_tag = li.select_one("h2 a")
                    snippet_tag = li.select_one(".b_caption p")
                    if not a_tag:
                        continue
                    url = a_tag.get("href", "")
                    domain = _get_domain(url)
                    if not url or _should_skip(url, domain):
                        continue
                    results.append(URLCandidate(
                        url=url,
                        domain=domain,
                        title=a_tag.get_text(strip=True),
                        snippet=snippet_tag.get_text(strip=True) if snippet_tag else "",
                        provider="bing",
                        rank=rank,
                        domain_score=_domain_score(url, company_website),
                        authority_score=authority_score(url, company_website),
                        source_category=classify_source(url, company_website),
                    ))
                    rank += 1

                await asyncio.sleep(0.5)  # polite delay between requests
            except Exception as e:
                logger.warning(f"[MultiSearch] Bing query failed: '{query}': {e}")

    logger.info(f"[MultiSearch] Bing returned {len(results)} candidates")
    return results


# ── Merge & Deduplicate ───────────────────────────────────────────────────────

def _merge_and_deduplicate(
    all_results: list[URLCandidate],
    max_urls: int = 50
) -> list[URLCandidate]:
    """Deduplicate by URL, boost score if found by multiple providers, sort by domain_score."""
    seen: dict[str, URLCandidate] = {}
    provider_counts: dict[str, int] = {}

    for candidate in all_results:
        url = candidate["url"]
        if url in seen:
            # Multi-provider boost
            provider_counts[url] = provider_counts.get(url, 1) + 1
            existing = seen[url]
            existing["domain_score"] = min(1.0, existing["domain_score"] + 0.1)
        else:
            seen[url] = candidate
            provider_counts[url] = 1

    # Sort by domain_score descending
    deduped = sorted(seen.values(), key=lambda x: x["domain_score"], reverse=True)
    return deduped[:max_urls]



def _resolve_company_profile(company: str, profile: dict, candidates: list[URLCandidate]) -> dict:
    """Resolve official website and LinkedIn company URL from ranked search results."""
    resolved = dict(profile or {})
    company_tokens = [t for t in re.findall(r"[a-z0-9]+", company.lower()) if len(t) > 2]
    excluded = {
        "linkedin.com", "crunchbase.com", "zaubacorp.com", "tofler.in", "tracxn.com",
        "glassdoor.com", "ambitionbox.com", "trustpilot.com", "g2.com", "clutch.co",
        "wikipedia.org", "facebook.com", "instagram.com", "youtube.com",
    }

    for candidate in candidates:
        url = candidate.get("url", "")
        domain = candidate.get("domain", "").lower()
        category = candidate.get("source_category") or classify_source(url, resolved.get("website", ""))
        haystack = f"{url} {candidate.get('title', '')} {candidate.get('snippet', '')}".lower()
        token_matches = sum(token in haystack for token in company_tokens)
        required_matches = max(1, (len(company_tokens) + 1) // 2)
        if (
            category == "LINKEDIN_COMPANY_PAGE"
            and not resolved.get("linkedin_company_page")
            and token_matches >= required_matches
        ):
            resolved["linkedin_company_page"] = url
        if resolved.get("website"):
            continue
        if category != "UNKNOWN" or any(blocked in domain for blocked in excluded):
            continue
        domain_matches = sum(token in domain for token in company_tokens)
        if company_tokens and domain_matches >= required_matches:
            resolved["website"] = f"https://{domain}"

    return resolved

# ── LangGraph node ────────────────────────────────────────────────────────────


def _balanced_worker_queries(worker_queries: dict[str, list[str]], fallback: list[str]) -> list[str]:
    """Interleave worker queries so no specialist is starved by provider caps."""
    if not worker_queries:
        return fallback
    balanced = []
    max_len = max((len(queries) for queries in worker_queries.values()), default=0)
    for index in range(max_len):
        for queries in worker_queries.values():
            if index < len(queries):
                balanced.append(queries[index])
    return list(dict.fromkeys(balanced + fallback))


def run_multi_search(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: Multi Search Layer."""
    t0 = time.time()
    company = state["company"]
    search_queries = _balanced_worker_queries(
        state.get("worker_queries", {}), state.get("search_queries", [])
    )
    settings = get_settings()
    profile = state.get("company_profile", {}) or {}
    company_website = profile.get("website") or state.get("card_info", {}).get("website", "")

    logger.info(f"[MultiSearch] Searching for '{company}' with {len(search_queries)} queries")

    async def _run_parallel():
        # Run DuckDuckGo and Bing in parallel
        ddg_task = _search_duckduckgo(search_queries, max_results_per_query=5, company_website=company_website)
        bing_task = _search_bing_httpx(search_queries, max_results_per_query=5, company_website=company_website)
        ddg_results, bing_results = await asyncio.gather(ddg_task, bing_task, return_exceptions=True)

        all_results = []
        if isinstance(ddg_results, list):
            all_results.extend(ddg_results)
        else:
            logger.warning(f"[MultiSearch] DuckDuckGo failed: {ddg_results}")

        if isinstance(bing_results, list):
            all_results.extend(bing_results)
        else:
            logger.warning(f"[MultiSearch] Bing failed: {bing_results}")

        return all_results

    try:
        all_candidates = asyncio.run(_run_parallel())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        all_candidates = loop.run_until_complete(_run_parallel())

    # Also add the company website if known from card
    card_info = state.get("card_info", {})
    website = profile.get("website") or card_info.get("website", "")
    if website:
        domain = _get_domain(website)
        all_candidates.insert(0, URLCandidate(
            url=website,
            domain=domain,
            title=f"{company} Official Website",
            snippet="",
            provider="card_info",
            rank=0,
            domain_score=1.0,  # highest priority
            authority_score=1.0,
            source_category="OFFICIAL_WEBSITE",
        ))

    merged = _merge_and_deduplicate(all_candidates, max_urls=settings.max_urls_to_fetch)
    resolved_profile = _resolve_company_profile(company, profile, merged)
    resolved_website = resolved_profile.get("website", company_website)
    for candidate in merged:
        candidate["source_category"] = classify_source(candidate["url"], resolved_website)
        candidate["authority_score"] = authority_score(candidate["url"], resolved_website)
        candidate["domain_score"] = candidate["authority_score"]
    merged.sort(key=lambda x: x.get("authority_score", 0), reverse=True)

    elapsed = round(time.time() - t0, 2)
    logger.info(f"[MultiSearch] Collected {len(merged)} unique URLs in {elapsed}s")

    return {
        "url_candidates": merged,
        "company_profile": resolved_profile,
        "status": "search_done",
        "progress_pct": 20,
        "node_timings": {"multi_search": elapsed},
        "log": [f"[MultiSearch] {len(merged)} unique URLs found for {company}"],
    }
