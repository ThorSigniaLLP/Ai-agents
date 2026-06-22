"""Prioritize resolved identity URLs without inventing deterministic company slugs."""
from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

from agents.source_authority import authority_score, classify_source
from agents.website_mapper import PAGE_AUTHORITY_SCORES, classify_page
from core.state import ResearchState, URLCandidate
from core.db_tracer import trace_search_candidates


def run_url_graph_builder(state: ResearchState) -> dict[str, Any]:
    started = time.time()
    profile = state.get("company_profile", {})
    website = profile.get("website", "")
    candidates = list(state.get("url_candidates", []))

    if website:
        candidates.append(URLCandidate(
            url=website,
            domain=urlparse(website).netloc.replace("www.", ""),
            title=f"{state['company']} Official Website",
            snippet="",
            provider="company_resolver",
            rank=0,
            domain_score=1.0,
            authority_score=1.0,
            source_category="OFFICIAL_WEBSITE",
            page_type="HOME",
            page_authority_score=1.0,
        ))

    unique: dict[str, URLCandidate] = {}
    for candidate in candidates:
        url = candidate.get("url", "").rstrip("/")
        if not url:
            continue
        category = classify_source(url, website)
        candidate["source_category"] = category
        candidate["authority_score"] = authority_score(url, website)
        if category == "OFFICIAL_WEBSITE":
            page_type = classify_page(url, website)
            candidate["page_type"] = page_type
            candidate["page_authority_score"] = PAGE_AUTHORITY_SCORES[page_type]
        else:
            candidate.setdefault("page_type", category)
            candidate.setdefault("page_authority_score", 1.0)
        candidate["domain_score"] = candidate["authority_score"] * candidate["page_authority_score"]
        current = unique.get(url)
        if current is None or candidate["domain_score"] > current["domain_score"]:
            unique[url] = candidate

    final = sorted(unique.values(), key=lambda item: item["domain_score"], reverse=True)[:50]
    
    # Save candidates to db
    trace_search_candidates(state.get("job_id", ""), final)
    return {
        "url_candidates": final,
        "status": "url_graph_built",
        "progress_pct": 25,
        "node_timings": {"url_graph": round(time.time() - started, 2)},
        "log": [f"[URLGraph] {len(final)} grounded URLs queued"],
    }
