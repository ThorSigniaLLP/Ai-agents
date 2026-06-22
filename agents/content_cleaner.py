"""
agents/content_cleaner.py
Content Cleaner — validates pages belong to the target company, chunks documents,
and attaches rich metadata.

Chunk size: 3000-5000 characters
Metadata per chunk: url, title, domain, published_date, source_type
Strips: nav, ads, cookie banners, headers, footers (via trafilatura favor_precision)
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any
from urllib.parse import urlparse

from core.state import ResearchState, PageResult, EvidenceChunk, URLCandidate
from agents.source_authority import combined_score, classify_source
from core.db_tracer import trace_evidence_chunks

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 3500    # characters
_OVERLAP = 200        # overlap between chunks


def _chunk_text(text: str) -> list[str]:
    """Split into overlapping chunks, preserving paragraph boundaries."""
    if len(text) <= _CHUNK_SIZE:
        return [text]

    # Try to split on paragraph boundaries
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > _CHUNK_SIZE and current:
            chunks.append(current.strip())
            # Overlap: keep last _OVERLAP chars
            current = current[-_OVERLAP:] + "\n\n" + para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    # If still empty (no paragraph breaks), fall back to char-based chunking
    if not chunks:
        start = 0
        while start < len(text):
            chunks.append(text[start:start + _CHUNK_SIZE])
            start += _CHUNK_SIZE - _OVERLAP

    return chunks


def _score_relevance(chunk: str, company: str) -> float:
    """Enhanced relevance scoring — pain points and B2B signals weighted highest."""
    company_lower = company.lower()
    chunk_lower = chunk.lower()

    score = 0.0

    # Company name present
    if company_lower in chunk_lower:
        score += 0.35

    identity_kws = [
        "about", "industry", "headquarters", "located", "founded", "founder",
        "service", "solution", "product", "employee", "company size", "legal",
        "registered", "director", "technology", "platform", "software", "cloud",
        "complaint", "review", "issue", "problem", "revenue", "capital", "competitor",
        "growth", "expansion", "hiring", "pain point",
    ]
    hits = sum(1 for kw in identity_kws if kw in chunk_lower)
    score += min(hits * 0.04, 0.45)

    return round(min(score, 1.0), 3)


def _build_entity_aliases(company: str, card: dict) -> list[str]:
    """Build a list of name/domain tokens that must appear in a page for it to be relevant."""
    aliases = []

    # Company name tokens (split into significant words, drop short stop words)
    stopwords = {"pvt", "ltd", "limited", "private", "inc", "llc", "the", "and", "of", "a", "an", "for"}
    words = [w.lower() for w in re.split(r"[\s\-_]+", company) if len(w) > 2 and w.lower() not in stopwords]
    aliases.extend(words)

    # Known website domain
    website = card.get("website", "")
    if website:
        domain = urlparse(website if website.startswith("http") else f"https://{website}").netloc
        domain_root = domain.replace("www.", "").split(".")[0]  # e.g. "gvhcol" from "gvhcol.com"
        if len(domain_root) > 2:
            aliases.append(domain_root.lower())

    # Email domain root
    email = card.get("email", "")
    if "@" in email:
        email_domain_root = email.split("@")[-1].split(".")[0].lower()
        if len(email_domain_root) > 2:
            aliases.append(email_domain_root)

    return list(set(aliases))


def _validate_page_entity(page: PageResult, company: str, card: dict, known_domain: str) -> tuple[bool, float]:
    """
    Validate that a fetched page is actually about the target company.
    
    Returns (is_valid, confidence_multiplier).
    - Pinned official website pages always pass with multiplier 1.0.
    - Pages that mention company name/domain pass with multiplier based on confidence.
    - Generic/unrelated pages are filtered out (multiplier 0.0 = reject).
    """
    page_domain = page.get("domain", "").lower().replace("www.", "")
    url = page.get("url", "")
    title = (page.get("title", "") or "").lower()
    content = (page.get("content", "") or "").lower()[:3000]  # Check first 3000 chars only

    # ── 1. Pinned domain: always valid, highest confidence ──
    if known_domain and known_domain.lower().replace("www.", "") in page_domain:
        return True, 1.0

    # ── 2. Review/news/registry sites: valid but pass through with reduced confidence ──
    # These sites talk ABOUT companies, not belonging to them
    third_party_domains = {
        "linkedin.com", "glassdoor.com", "g2.com", "trustpilot.com", "clutch.co",
        "crunchbase.com", "tracxn.com", "zaubacorp.com", "tofler.in",
        "ambitionbox.com", "reddit.com", "wikipedia.org", "bloomberg.com",
        "reuters.com", "techcrunch.com", "forbes.com", "businesswire.com",
        "builtwith.com", "wappalyzer.com", "mojeek.com", "google.com", "yandex.com"
    }
    is_third_party = any(d in page_domain for d in third_party_domains)

    # ── 3. Wrong Official Domain check ──
    # If we know the target's exact website, and this is NOT a third-party aggregator,
    # then another official-looking domain (e.g. meltchocolates.com when we want .co.in) is likely a conflict.
    if known_domain and not is_third_party and known_domain.lower().replace("www.", "") not in page_domain:
        logger.debug("[ContentCleaner] ⚠ Rejecting wrong official domain: %s (expected %s)", page_domain, known_domain)
        return False, 0.0

    # Build entity tokens to validate presence in content
    aliases = _build_entity_aliases(company, card)

    # Check if any alias token appears in title or content
    found_in_title = any(alias in title for alias in aliases)
    found_in_content = sum(1 for alias in aliases if alias in content)

    if is_third_party:
        # Third-party sites must mention at least one company token somewhere
        if found_in_title or found_in_content >= 1:
            confidence = min(0.5 + found_in_content * 0.1, 1.0)
            return True, confidence
        else:
            # Third-party page about a completely different company
            logger.debug("[ContentCleaner] ⚠ Rejecting third-party page (no company mention): %s", url[:80])
            return False, 0.0

    # ── 3. Unknown/other domain: strict validation ──
    # Must mention company name tokens in both title and content
    if found_in_title and found_in_content >= 2:
        return True, 0.85
    elif found_in_content >= 3:
        return True, 0.7
    elif found_in_content >= 1:
        # Marginal — keep but heavily penalize
        return True, 0.35
    else:
        logger.debug("[ContentCleaner] ⚠ Rejecting irrelevant page: %s | title=%s", url[:60], title[:60])
        return False, 0.0


def run_content_cleaner(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: Content Cleaner with entity validation."""
    t0 = time.time()
    company = state["company"]
    card = state.get("card_info", {}) or {}
    raw_pages: list[PageResult] = state.get("raw_pages", [])
    url_candidates: list[URLCandidate] = state.get("url_candidates", [])

    # Resolve known domain for pinned-page bypass
    profile = state.get("company_profile", {}) or {}
    known_website = profile.get("website") or card.get("website", "")
    known_domain = ""
    if known_website:
        raw = known_website if known_website.startswith("http") else f"https://{known_website}"
        known_domain = urlparse(raw).netloc.replace("www.", "")

    logger.info(
        "[ContentCleaner] Processing %d pages and %d snippets for '%s' (known domain: %s)",
        len(raw_pages), len(url_candidates), company, known_domain or "UNKNOWN"
    )

    all_chunks: list[EvidenceChunk] = []
    rejected_pages = 0

    for page in raw_pages:
        content = page.get("content", "")
        if not content or len(content) < 100:
            continue

        # ── Entity validation: is this page actually about our target company? ──
        is_valid, confidence_mult = _validate_page_entity(page, company, card, known_domain)
        if not is_valid:
            rejected_pages += 1
            continue

        url = page.get("url", "")
        domain = page.get("domain", "")
        source_type = page.get("source_type", "unknown")
        published_date = page.get("published_date", "")
        authority = page.get("authority_score", 0.5)
        category = page.get("source_category", "UNKNOWN")
        page_type = page.get("page_type", category)
        page_authority = page.get("page_authority_score", 1.0) * confidence_mult

        # Split into chunks
        text_chunks = _chunk_text(content)

        for i, chunk in enumerate(text_chunks):
            relevance = _score_relevance(chunk, company)
            score = combined_score(relevance, authority) * page_authority
            if score < 0.05:
                continue
            all_chunks.append(EvidenceChunk(
                url=url,
                domain=domain,
                chunk=chunk.strip(),
                source_type=source_type,
                published_date=published_date,
                chunk_index=i,
                rerank_score=score,
                authority_score=authority,
                source_category=category,
                page_type=page_type,
                page_authority_score=page_authority,
            ))

    # Inject search snippets as chunks to capture data from blocked sites (ZaubaCorp, Glassdoor, G2)
    # Snippets are pre-validated by the search engine so we use a lighter check
    aliases = _build_entity_aliases(company, card)
    snippet_accepted = 0
    snippet_rejected = 0

    for cand in url_candidates:
        snippet = cand.get("snippet", "")
        if not snippet or len(snippet) < 30:
            continue

        # Validate snippet mentions target company (at least 1 alias token)
        snippet_lower = (cand.get("title", "") + " " + snippet).lower()
        if not any(alias in snippet_lower for alias in aliases):
            snippet_rejected += 1
            continue

        snippet_accepted += 1
        chunk_text = f"Title: {cand.get('title', '')}\nSnippet: {snippet}"
        relevance = _score_relevance(chunk_text, company)
        authority = cand.get("authority_score", 0.5)
        score = combined_score(relevance, authority)

        all_chunks.append(EvidenceChunk(
            url=cand.get("url", ""),
            domain=cand.get("domain", ""),
            chunk=chunk_text,
            source_type=classify_source(cand.get("url", ""), ""),
            published_date="",
            chunk_index=0,
            rerank_score=score + 0.1,  # Boost snippets: highly dense context
            authority_score=authority,
            source_category=cand.get("source_category", "UNKNOWN"),
            page_type="SEARCH_SNIPPET",
            page_authority_score=cand.get("page_authority_score", 1.0),
        ))

    # Sort by initial relevance, keep top 80 for reranker
    all_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_chunks = all_chunks[:80]

    elapsed = round(time.time() - t0, 2)

    # Save chunks to db
    trace_evidence_chunks(state.get("job_id", ""), top_chunks)

    logger.info(
        "[ContentCleaner] Produced %d chunks | Rejected %d pages (wrong entity) | "
        "Snippets: %d accepted, %d rejected | %.2fs",
        len(top_chunks), rejected_pages, snippet_accepted, snippet_rejected, elapsed
    )

    return {
        "evidence_chunks": top_chunks,
        "status": "cleaning_done",
        "progress_pct": 50,
        "node_timings": {"content_cleaner": elapsed},
        "log": [
            f"[ContentCleaner] {len(top_chunks)} chunks | "
            f"{rejected_pages} pages rejected (wrong entity) | "
            f"{snippet_accepted}/{snippet_accepted + snippet_rejected} snippets validated"
        ],
    }
