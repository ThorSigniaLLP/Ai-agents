"""Deterministic validation gates for final CompanyProfile fields."""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from typing import Any

from core.state import EvidenceItem, ResearchState

UNKNOWN = "UNKNOWN"


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _confidence(items: list[EvidenceItem]) -> float:
    if not items:
        return 0.0
    return min(1.0, sum(item["authority_score"] * item["relevance_score"] for item in items) / len(items))


def _best_group(items: list[EvidenceItem]) -> tuple[Any, list[EvidenceItem]]:
    groups: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in items:
        groups[_norm(item["value"])].append(item)
    if not groups:
        return UNKNOWN, []
    group = max(groups.values(), key=lambda values: (len({v["source_url"] for v in values}), _confidence(values)))
    return group[0]["value"], group


def _scalar(items: list[EvidenceItem], min_sources: int = 1, min_confidence: float = 0.5, required_page_type: str | None = None) -> str:
    value, support = _best_group(items)
    if value == UNKNOWN:
        return UNKNOWN
    if required_page_type and not all(item["page_type"] == required_page_type for item in support):
        return UNKNOWN
    if len({item["source_url"] for item in support}) < min_sources:
        return UNKNOWN
    if _confidence(support) < min_confidence:
        return UNKNOWN
    return str(value)


def _list_values(items: list[EvidenceItem], allowed_page_types: set[str] | None = None, min_score: float = 0.3) -> list[str]:
    accepted = []
    seen = set()
    for item in sorted(items, key=lambda value: value["authority_score"] * value["relevance_score"], reverse=True):
        if allowed_page_types and item["page_type"] not in allowed_page_types:
            continue
        if item["authority_score"] * item["relevance_score"] < min_score:
            continue
        key = _norm(item["value"])
        if key and key not in seen:
            seen.add(key)
            accepted.append(str(item["value"]))
    return accepted


def _collect_json_list(items: list[EvidenceItem], field: str) -> list:
    """Collect JSON-encoded values from evidence items of a given field."""
    results = []
    for item in items:
        if item.get("field") != field:
            continue
        try:
            val = json.loads(item["value"])
            if isinstance(val, list):
                results.extend(val)
            elif isinstance(val, dict):
                results.append(val)
        except Exception:
            if item["value"] and item["value"] != UNKNOWN:
                results.append(item["value"])
    return results


def _collect_json_dict(items: list[EvidenceItem], field: str) -> dict:
    """Merge JSON-encoded dicts from evidence items of a given field."""
    merged = {}
    for item in items:
        if item.get("field") != field:
            continue
        try:
            val = json.loads(item["value"])
            if isinstance(val, dict):
                # Merge non-null values
                for k, v in val.items():
                    if v and v not in ("null", "none", None, UNKNOWN, "UNKNOWN"):
                        merged[k] = v
        except Exception:
            pass
    return merged


def run_profile_validator(state: ResearchState) -> dict[str, Any]:
    started = time.time()
    seed = state.get("company_profile", {})
    grouped: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in state.get("evidence_items", []):
        grouped[item["field"]].append(item)

    candidate_urls = state.get("url_candidates", [])
    card = state.get("card_info", {}) or {}

    # Website: prefer card_info.website (already verified), fall back to seed, never erase it
    card_website = card.get("website", "")
    if card_website and not card_website.startswith("http"):
        card_website = f"https://{card_website}"
    website = card_website or seed.get("website") or UNKNOWN

    # LinkedIn: keep if it looks valid
    linkedin = seed.get("linkedin_company_page") or UNKNOWN
    if linkedin != UNKNOWN and "linkedin.com/company/" not in linkedin.lower():
        linkedin = UNKNOWN

    # ── Identity fields (relaxed validation) ─────────────────────────────────
    profile: dict[str, Any] = {
        "company_name": state["company"],
        "website": website,
        "linkedin_company_page": linkedin,
        "headquarters": _scalar(grouped["headquarters"], min_sources=1, min_confidence=0.15),
        "industry": _scalar(grouped["industry"], min_sources=1, min_confidence=0.15),
        "overview": _scalar(grouped["overview"], min_sources=1, min_confidence=0.15),
        "founders": _list_values(grouped["founders"], min_score=0.1),
        "services": _list_values(grouped["services"], min_score=0.1),
        "technologies": _list_values(grouped["technologies"], min_score=0.1),
        "employee_count": _scalar(grouped["employee_count"], min_sources=1, min_confidence=0.15),
        "legal_entity": _scalar(grouped["legal_entity"], min_sources=1, min_confidence=0.15),
    }

    # ── Extended B2B intelligence fields (relaxed validation) ─────────────────

    # Firmographics
    profile["founded_year"] = _scalar(grouped["founded_year"], min_sources=1, min_confidence=0.1)
    profile["revenue"] = _scalar(grouped["revenue"], min_sources=1, min_confidence=0.1)
    profile["competitors"] = _list_values(grouped["competitors"], min_score=0.1)

    # Pain points — collect as structured dicts
    pain_points = _collect_json_list(grouped["pain_points"], "pain_points")
    profile["pain_points"] = pain_points if pain_points else []

    # Growth signals — merge into one dict
    growth = _collect_json_dict(grouped["growth_signals"], "growth_signals")
    if not growth:
        growth = {"job_postings": [], "recent_news": [], "expansion_signals": [], "hiring_trend": UNKNOWN}
    profile["growth_signals"] = growth

    # Disconnection signals — plain list
    profile["disconnection_signals"] = [
        item["value"] for item in grouped["disconnection_signals"]
        if item.get("value") and item["value"] != UNKNOWN
    ]

    # Tech stack — merge all tech_stack evidence dicts
    tech = _collect_json_dict(grouped["tech_stack"], "tech_stack")
    if not tech:
        tech = {"crm": UNKNOWN, "erp": UNKNOWN, "marketing_tools": [], "development_stack": [], "cloud_provider": UNKNOWN}
    # Ensure list fields are lists
    for list_field in ("marketing_tools", "development_stack"):
        if not isinstance(tech.get(list_field), list):
            tech[list_field] = [tech[list_field]] if tech.get(list_field) else []
    profile["tech_stack"] = tech

    # Pitch opportunities — plain list
    profile["pitch_opportunities"] = [
        item["value"] for item in grouped["pitch_opportunities"]
        if item.get("value") and item["value"] != UNKNOWN
    ]

    return {
        "company_profile": profile,
        "verified_facts": profile,
        "status": "profile_validated",
        "progress_pct": 92,
        "node_timings": {"profile_validator": round(time.time() - started, 2)},
        "log": ["[Validator] Company profile validation complete with B2B intelligence fields"],
    }
