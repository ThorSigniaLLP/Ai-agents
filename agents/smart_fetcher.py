"""
agents/smart_fetcher.py
Smart Content Fetcher — 3-level fallback strategy.

Level 1 (default): httpx + trafilatura + readability-lxml
Level 2 (fallback): Playwright headless browser
Level 3 (last resort): browser-use AI agent

Never crashes on failure — skips the page and continues.
Retries with different User-Agent on 403.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse, quote_plus

import httpx
import trafilatura

from core.config import get_settings
from core.state import ResearchState, URLCandidate, PageResult
from agents.source_authority import authority_score, classify_source
from core.db_tracer import trace_fetched_pages

logger = logging.getLogger(__name__)

_SOURCE_TYPE_MAP = {
    "linkedin.com": "linkedin",
    "zaubacorp.com": "registry",
    "mca.gov.in": "registry",
    "tofler.in": "registry",
    "tracxn.com": "registry",
    "ambitionbox.com": "review",
    "builtwith.com": "technology",
    "wappalyzer.com": "technology",
    "crunchbase.com": "crunchbase",
    "sec.gov": "sec",
    "bloomberg.com": "news",
    "reuters.com": "news",
    "techcrunch.com": "news",
    "forbes.com": "news",
    "businesswire.com": "news",
    "prnewswire.com": "news",
    "g2.com": "review",
    "trustpilot.com": "review",
    "glassdoor.com": "review",
    "reddit.com": "social",
    "wikipedia.org": "wiki",
    "wsj.com": "news",
    "ft.com": "news",
}

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
]


def _classify_source(domain: str) -> str:
    for key, src in _SOURCE_TYPE_MAP.items():
        if key in domain:
            return src
    return "company_site"


def _company_slug(company: str) -> str:
    """Convert company name to URL-friendly slug."""
    return re.sub(r"[^a-z0-9]+", "-", company.lower()).strip("-")


def _inject_priority_urls(company: str, existing_urls: set[str]) -> list[URLCandidate]:
    """Inject known high-value source URLs that search may miss."""
    slug = _company_slug(company)
    q = quote_plus(company)
    candidates = []
    priority_sources = [
        # Reviews & pain points
        (f"https://www.g2.com/search#q={q}&scope=products", "g2.com", "review", 0.85),
        (f"https://www.glassdoor.com/Search/results.htm?keyword={q}", "glassdoor.com", "review", 0.82),
        (f"https://www.trustpilot.com/search?query={q}", "trustpilot.com", "review", 0.80),
        (f"https://clutch.co/search?query={q}", "clutch.co", "review", 0.80),
        # Firmographics
        (f"https://www.crunchbase.com/search/organizations/field/organizations/facet_ids/{slug}", "crunchbase.com", "crunchbase", 0.85),
        (f"https://tracxn.com/d/companies/{slug}", "tracxn.com", "registry", 0.80),
        # Social signals
        (f"https://www.reddit.com/search/?q={q}+company+problems+issues", "reddit.com", "social", 0.78),
    ]
    for url, domain, src_type, score in priority_sources:
        if url not in existing_urls:
            candidates.append(URLCandidate(
                url=url,
                domain=domain,
                title=f"{company} — {src_type}",
                snippet="",
                provider="priority_injection",
                rank=0,
                domain_score=score,
                authority_score=score,
                source_category=classify_source(url, ""),
            ))
    return candidates


def _extract_published_date(html: str) -> str:
    """Try to extract publication date from HTML meta tags."""
    patterns = [
        r'<meta[^>]+(?:property|name)=["\'](?:article:published_time|datePublished|pubdate)["\'][^>]+content=["\']([\d\-T:Z+]+)',
        r'"datePublished"\s*:\s*"([\d\-T:Z+]+)"',
        r'"publishedAt"\s*:\s*"([\d\-T:Z+]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return match.group(1)[:10]  # Return YYYY-MM-DD
    return ""


async def _fetch_level1_httpx(url: str, timeout: int = 20, user_agent_idx: int = 0) -> Optional[tuple[str, str, str]]:
    """Level 1: httpx + trafilatura + readability. Returns (text, title, published_date) or None."""
    ua = _USER_AGENTS[user_agent_idx % len(_USER_AGENTS)]
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        ) as client:
            resp = await client.get(url)
            if resp.status_code == 403:
                return None  # Signal for retry with different UA
            resp.raise_for_status()
            html = resp.text

            # Extract date before stripping HTML
            published_date = _extract_published_date(html)

            # Try trafilatura (best quality)
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_links=False,
                no_fallback=False,
                favor_precision=True,
                output_format="markdown",
            )

            if text and len(text) > 200:
                # Try to get title
                title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else url
                return text, title, published_date

            # Fallback: readability-lxml
            try:
                from readability import Document
                doc = Document(html)
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(doc.summary(), "lxml")
                text = soup.get_text(separator="\n", strip=True)
                if text and len(text) > 200:
                    return text, doc.title(), published_date
            except Exception:
                pass

            # Last fallback: BeautifulSoup strip
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "lxml")
                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                if text and len(text) > 200:
                    return text, soup.title.string if soup.title else url, published_date
            except Exception:
                pass

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403 and user_agent_idx == 0:
            return None  # Retry with different UA
        logger.debug(f"[SmartFetcher] HTTP {e.response.status_code} for {url}")
    except Exception as e:
        logger.debug(f"[SmartFetcher] Level 1 failed for {url}: {type(e).__name__}")

    return None


async def _fetch_level2_playwright(url: str, timeout: int = 30) -> Optional[tuple[str, str, str]]:
    """Level 2: Playwright headless browser."""
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=_USER_AGENTS[0],
                viewport={"width": 1280, "height": 720},
            )
            page = await context.new_page()
            await page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)  # Let JS render

            html = await page.content()
            title = await page.title()
            published_date = _extract_published_date(html)

            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
                output_format="markdown",
            )
            await browser.close()

            if text and len(text) > 200:
                return text, title, published_date

    except Exception as e:
        logger.debug(f"[SmartFetcher] Level 2 Playwright failed for {url}: {e}")

    return None


async def _fetch_single_page(url_obj: URLCandidate, timeout: int, company_website: str = "") -> Optional[PageResult]:
    """Fetch a single URL using the 3-level strategy."""
    url = url_obj["url"]
    domain = url_obj["domain"]

    # Level 1: httpx (try twice with different User-Agents)
    result = await _fetch_level1_httpx(url, timeout=timeout, user_agent_idx=0)
    if result is None:
        # Retry with different UA (handles 403s)
        result = await _fetch_level1_httpx(url, timeout=timeout, user_agent_idx=1)

    fetch_method = "httpx"

    # Level 2: Playwright (for JS-heavy pages)
    if result is None:
        result = await _fetch_level2_playwright(url, timeout=timeout)
        fetch_method = "playwright"

    if result is None:
        logger.debug(f"[SmartFetcher] All levels failed for {url}")
        return None

    text, title, published_date = result

    return PageResult(
        url=url,
        title=title[:200],
        content=text[:15000],  # Cap at 15k chars per page
        domain=domain,
        source_type=_classify_source(domain),
        published_date=published_date,
        fetch_method=fetch_method,
        timestamp=datetime.now(timezone.utc).isoformat(),
        relevance_score=url_obj["domain_score"],
        authority_score=url_obj.get("authority_score", authority_score(url, company_website)),
        source_category=url_obj.get("source_category", classify_source(url, company_website)),
        page_type=url_obj.get("page_type", url_obj.get("source_category", "UNKNOWN")),
        page_authority_score=url_obj.get("page_authority_score", 1.0),
    )


def run_smart_fetcher(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: Smart Content Fetcher."""
    t0 = time.time()
    company = state["company"]
    url_candidates: list[URLCandidate] = state.get("url_candidates", [])
    settings = get_settings()
    profile = state.get("company_profile", {}) or {}
    company_website = profile.get("website") or state.get("card_info", {}).get("website", "")

    # Inject priority source URLs (G2, Glassdoor, Crunchbase, Reddit)
    existing_urls = {c["url"] for c in url_candidates}
    priority_urls = _inject_priority_urls(company, existing_urls)

    # ── Pin the EXACT company website from card_info as the FIRST URL fetched ──
    # This prevents domain confusion (e.g. "green view medical" → gvhcol.com)
    card = state.get("card_info", {}) or {}
    pinned_website = company_website or card.get("website", "")
    card_email = card.get("email", "")
    card_email_domain = card_email.split("@")[-1] if "@" in card_email else ""
    pinned_urls: list[URLCandidate] = []
    
    if pinned_website:
        # Clean and normalize to https:// scheme
        if not pinned_website.startswith("http"):
            pinned_website = f"https://{pinned_website}"
        domain = urlparse(pinned_website).netloc.replace("www.", "")
        if domain:  # Valid domain extracted
            for path in ["/", "/about", "/services", "/about-us", "/contact"]:
                url = f"https://{domain}{path}"
                if url not in existing_urls:
                    pinned_urls.append(URLCandidate(
                        url=url,
                        domain=domain,
                        title=f"{company} — {path.strip('/') or 'home'}",
                        snippet="",
                        provider="card_info_pin",
                        rank=0,
                        domain_score=1.0,
                        authority_score=1.0,
                        source_category="OFFICIAL_WEBSITE",
                        page_type="HOME" if path == "/" else "ABOUT",
                        page_authority_score=1.0,
                    ))
            logger.info("[SmartFetcher] Pinned %d URLs from known domain: %s", len(pinned_urls), domain)
    
    # Prepend: pinned website first, then other priority URLs, then search candidates
    all_candidates = pinned_urls + priority_urls + url_candidates

    # Fetch top 30 URLs per iteration
    urls_to_fetch = all_candidates[:30]
    logger.info(f"[SmartFetcher] Fetching {len(urls_to_fetch)} URLs for {company} ({len(pinned_urls)} pinned, {len(priority_urls)} priority injected)")

    async def _run():
        tasks = [_fetch_single_page(url_obj, settings.page_load_timeout, company_website) for url_obj in urls_to_fetch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pages = []
        for r in results:
            if isinstance(r, dict) and "url" in r:
                pages.append(r)
        return pages

    try:
        pages = asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        pages = loop.run_until_complete(_run())

    elapsed = round(time.time() - t0, 2)
    success_count = len(pages)
    
    # Save fetched pages to db
    trace_fetched_pages(state.get("job_id", ""), pages)
    
    logger.info(f"[SmartFetcher] Fetched {success_count}/{len(urls_to_fetch)} pages in {elapsed}s")

    return {
        "raw_pages": pages,
        "sources_used": [p["url"] for p in pages],
        "status": "fetching_done",
        "progress_pct": 40,
        "node_timings": {"smart_fetcher": elapsed},
        "log": [f"[SmartFetcher] {success_count} pages fetched for {company}"],
    }
