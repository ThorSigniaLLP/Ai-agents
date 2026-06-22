"""
agents/temporal_verifier.py
Temporal Verifier — prefer newer facts, flag stale data.

If a fact comes from an article older than 2 years, it adds a "stale data" warning.
It also updates the progress and logs for the graph.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from dateutil.parser import parse as parse_date

from core.state import ResearchState, VerifiedFact

logger = logging.getLogger(__name__)


def _is_stale(date_str: str) -> bool:
    """Check if the date is older than 2 years from today."""
    if not date_str:
        return False
    try:
        pub_date = parse_date(date_str)
        # Handle naive vs aware datetime
        if pub_date.tzinfo is None:
            now = datetime.now()
        else:
            now = datetime.now(pub_date.tzinfo)
            
        diff_days = (now - pub_date).days
        return diff_days > (365 * 2)
    except Exception:
        return False


def run_temporal_verifier(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: Temporal Verifier."""
    t0 = time.time()
    company = state["company"]
    verified_facts: dict[str, VerifiedFact] = state.get("verified_facts", {})
    evidence_items = state.get("evidence_items", [])

    if not verified_facts:
        logger.warning("[TemporalVerifier] No verified facts to process")
        return {
            "status": "temporal_verification_done",
            "progress_pct": 75,
            "node_timings": {"temporal_verifier": 0},
            "log": ["[TemporalVerifier] Skipped — no facts"],
        }

    # Map URLs to their published_date from evidence items
    url_to_date = {}
    for item in evidence_items:
        url = item["source_url"]
        date_str = item.get("date", "")
        if url and date_str:
            # If multiple dates for same URL somehow, keep the newest
            if url in url_to_date:
                try:
                    d1 = parse_date(url_to_date[url])
                    d2 = parse_date(date_str)
                    if d2 > d1:
                        url_to_date[url] = date_str
                except Exception:
                    pass
            else:
                url_to_date[url] = date_str

    stale_count = 0
    for field, fact in verified_facts.items():
        # Check if all sources for this fact are stale
        sources = fact.get("sources", [])
        if not sources:
            continue

        stale_sources = []
        for src in sources:
            date_str = url_to_date.get(src, "")
            if _is_stale(date_str):
                stale_sources.append(src)

        dated_sources = []
        for src in sources:
            date_str = url_to_date.get(src, "")
            if not date_str:
                continue
            try:
                dated_sources.append((src, parse_date(date_str)))
            except Exception:
                pass

        if dated_sources:
            newest_src, newest_date = max(dated_sources, key=lambda x: x[1])
            materially_old = [src for src, dt in dated_sources if (newest_date - dt).days > 365]
            if materially_old:
                fact["sources"] = list(dict.fromkeys([newest_src] + [s for s in sources if s not in materially_old]))
                fact["note"] = "Newest evidence preferred; older conflicting/stale sources were de-prioritized"

        # If ALL sources for this fact are stale, warn
        if len(stale_sources) == len(sources) and len(sources) > 0:
            fact["note"] = "WARNING: Data may be stale (>2 years old)"
            stale_count += 1

    elapsed = round(time.time() - t0, 2)
    logger.info(f"[TemporalVerifier] Found {stale_count} stale facts in {elapsed}s")

    return {
        "verified_facts": verified_facts,  # Updated inline
        "status": "temporal_verification_done",
        "progress_pct": 75,
        "node_timings": {"temporal_verifier": elapsed},
        "log": [f"[TemporalVerifier] Flagged {stale_count} stale facts"],
    }
