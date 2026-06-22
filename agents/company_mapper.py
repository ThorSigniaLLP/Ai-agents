"""
Company Mapper Agent.

Runs immediately after planning to establish the company identity surface and
seed authoritative discovery targets before broader search happens.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any
from urllib.parse import urlparse

from core.state import ResearchState

logger = logging.getLogger(__name__)


def _slug(company: str, sep: str = "-") -> str:
    cleaned = re.sub(r"[^a-z0-9]+", sep, company.lower()).strip(sep)
    return cleaned


def _clean_website(url: str) -> str:
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def run_company_mapper(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: Build an initial CompanyProfile and authoritative queries."""
    t0 = time.time()
    company = state["company"]
    card_info = state.get("card_info", {})

    website = _clean_website(card_info.get("website", ""))
    linkedin = card_info.get("linkedin") or f"https://www.linkedin.com/company/{_slug(company, '-')}/"

    company_profile = {
        "name": company,
        "website": website,
        "linkedin": linkedin,
        "industry": card_info.get("industry", ""),
        "headquarters": card_info.get("address", ""),
        "employees": "",
        "founders": [],
        "entity_type": "",
        "legal_entity": "",
        "registry_records": [],
    }

    mapper_queries = [
        f"{company} official website",
        f"site:linkedin.com/company {company}",
        f"{company} LinkedIn company page",
        f"{company} industry headquarters employee count founders",
        f"{company} legal entity registered company",
        f"{company} zaubacorp",
        f"{company} MCA company master data",
        f"{company} tofler",
        f"{company} tracxn",
    ]
    if website:
        mapper_queries.extend([
            f"site:{urlparse(website).netloc} {company} about",
            f"site:{urlparse(website).netloc} {company} services",
            f"site:{urlparse(website).netloc} {company} team",
        ])

    existing_queries = state.get("search_queries", [])
    search_queries = list(dict.fromkeys(mapper_queries + existing_queries))

    elapsed = round(time.time() - t0, 2)
    logger.info(f"[CompanyMapper] Seeded profile and {len(mapper_queries)} mapper queries for {company}")

    return {
        "company_profile": company_profile,
        "search_queries": search_queries,
        "status": "company_mapped",
        "progress_pct": 12,
        "node_timings": {"company_mapper": elapsed},
        "log": [f"[CompanyMapper] Company identity seeds prepared for {company}"],
    }

