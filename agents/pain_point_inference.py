"""
Evidence-backed pain point inference.

Pain points are inferred only from strong supporting evidence categories. Weak
context can remain in the graph, but it cannot become a company pain point.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from agents.source_authority import REVIEW_CATEGORIES, WEAK_CONTEXT_CATEGORIES
from core.state import EvidenceItem, ResearchState, VerifiedFact

logger = logging.getLogger(__name__)

ALLOWED_PAIN_FIELDS = {
    "pain_points",
    "risks",
    "recent_news",
    "technology",
    "hiring",
    "growth",
}

ALLOWED_PAIN_CATEGORIES = REVIEW_CATEGORIES | {
    "NEWS",
    "TECH_STACK",
    "OFFICIAL_WEBSITE",
    "LINKEDIN_COMPANY_PAGE",
}


def _theme_key(text: str) -> str:
    lowered = text.lower()
    themes = {
        "technology_gap": ["manual", "spreadsheet", "crm", "automation", "legacy", "cloud", "ai"],
        "negative_news": ["layoff", "lawsuit", "controversy", "breach", "outage", "loss"],
        "customer_complaints": ["complaint", "poor", "bad", "delay", "support", "refund"],
        "employee_complaints": ["glassdoor", "ambitionbox", "work life", "management", "salary"],
        "growth_bottleneck": ["expansion", "growth", "scale", "capacity", "bottleneck"],
        "hiring": ["hiring", "job", "recruit", "talent", "headcount", "vacancy"],
    }
    for key, words in themes.items():
        if any(word in lowered for word in words):
            return key
    return "general"


def _eligible(item: EvidenceItem) -> bool:
    category = item.get("category", "")
    if category in WEAK_CONTEXT_CATEGORIES:
        return False
    if category and category not in ALLOWED_PAIN_CATEGORIES:
        return False
    return item.get("field") in ALLOWED_PAIN_FIELDS


def run_pain_point_inference(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: infer supported pain points and drop unsupported ones."""
    t0 = time.time()
    evidence_items: list[EvidenceItem] = state.get("evidence_items", [])
    verified_facts: dict[str, VerifiedFact] = state.get("verified_facts", {})

    grouped: dict[str, list[EvidenceItem]] = {}
    for item in evidence_items:
        if not _eligible(item):
            continue
        if item.get("verification_status") != "SUPPORTED":
            continue
        theme = _theme_key(item.get("fact_candidate", ""))
        if theme == "general":
            continue
        grouped.setdefault(theme, []).append(item)

    inferred = []
    rejected = list(state.get("rejected_facts", []))
    for theme, items in grouped.items():
        unique_sources = {i.get("source_url", "") for i in items if i.get("source_url")}
        if len(items) < 2 or len(unique_sources) < 2:
            for item in items:
                rejected.append({
                    "field": "pain_points",
                    "claim": item.get("fact_candidate", ""),
                    "reason": "UNKNOWN: fewer than two supporting EvidenceItems",
                })
            continue

        best_items = sorted(
            items,
            key=lambda x: (x.get("confidence", 0), x.get("authority_score", 0)),
            reverse=True,
        )[:3]
        claim = best_items[0].get("fact_candidate", "")
        inferred.append({
            "theme": theme,
            "pain_point": claim,
            "supporting_evidence": [
                {
                    "fact": item.get("fact_candidate", ""),
                    "source_url": item.get("source_url", ""),
                    "authority_score": item.get("authority_score", 0),
                    "category": item.get("category", ""),
                }
                for item in best_items
            ],
        })

    if inferred:
        verified_facts["pain_points"] = VerifiedFact(
            value=[p["pain_point"] for p in inferred],
            sources=list({e["source_url"] for p in inferred for e in p["supporting_evidence"]}),
            confidence=0.85,
            verification_status="SUPPORTED",
            note="Inferred only from pain-point-eligible evidence with at least two supporting sources.",
        )
    elif "pain_points" in verified_facts:
        rejected.append({
            "field": "pain_points",
            "claim": verified_facts["pain_points"].get("value"),
            "reason": "UNKNOWN: pain point inference lacked two strong supporting EvidenceItems",
        })
        verified_facts.pop("pain_points", None)

    business_analysis = state.get("business_analysis", {})
    business_analysis["inferred_pain_points"] = inferred
    business_analysis["key_pain_points"] = [p["pain_point"] for p in inferred]

    elapsed = round(time.time() - t0, 2)
    logger.info(f"[PainPointInference] Inferred {len(inferred)} supported pain points")

    return {
        "verified_facts": verified_facts,
        "business_analysis": business_analysis,
        "rejected_facts": rejected,
        "status": "pain_points_inferred",
        "progress_pct": 80,
        "node_timings": {"pain_point_inference": elapsed},
        "log": [f"[PainPointInference] {len(inferred)} pain points passed evidence threshold"],
    }

