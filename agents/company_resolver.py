"""Resolve the company identity surface before any extraction."""
from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urlparse

from core.state import ResearchState


def _clean_url(value: str) -> str:
    if not value:
        return ""
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    parsed = urlparse(value)
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def run_company_resolver(state: ResearchState) -> dict[str, Any]:
    """Seed identity candidates and authoritative identity queries only."""
    started = time.time()
    company = state["company"].strip()
    card = state.get("card_info", {})
    website = _clean_url(card.get("website", ""))
    linkedin = card.get("linkedin", "")
    if linkedin and "/company/" not in linkedin.lower():
        linkedin = ""

    profile = {
        "company_name": company,
        "website": website,
        "linkedin_company_page": linkedin,
        "headquarters": "UNKNOWN",
        "industry": "UNKNOWN",
        "founders": [],
        "services": [],
        "technologies": [],
        "employee_count": "UNKNOWN",
        "legal_entity": "UNKNOWN",
        "country": "",
        "aliases": [company],
    }
    queries = [
        f"{company} official website",
        f"site:linkedin.com/company {company}",
        f"{company} legal entity registered company",
        f"{company} headquarters industry",
        f"{company} founders",
        f"{company} MCA company master data",
        f"{company} ZaubaCorp",
        f"{company} Tofler",
    ]
    if website:
        domain = urlparse(website).netloc
        queries.extend([f"site:{domain} about", f"site:{domain} services", f"site:{domain} team"])

    return {
        "company_profile": profile,
        "search_queries": list(dict.fromkeys(queries + state.get("search_queries", []))),
        "status": "company_resolution_started",
        "progress_pct": 12,
        "node_timings": {"company_resolver": round(time.time() - started, 2)},
        "log": [f"[CompanyResolver] Identity discovery prepared for {company}"],
    }
