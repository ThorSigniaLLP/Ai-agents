"""Deep B2B intelligence specialist query planning."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from core.state import ResearchState


def run_specialized_workers(state: ResearchState) -> dict[str, Any]:
    started = time.time()
    company = state["company"]
    year = datetime.now().year
    website = state.get("company_profile", {}).get("website", "")
    domain = urlparse(website).netloc.replace("www.", "") if website else ""
    site = f"site:{domain}" if domain else company

    workers = {
        "identity": [
            f"{company} official website",
            f"site:linkedin.com/company {company}",
            f"{company} headquarters location industry",
        ],
        "website": [
            f"{site} about",
            f"{site} services solutions products",
            f"{site} case studies clients",
            f"{site} team founders leadership",
            f"{site} contact",
            f"{site} careers jobs",
        ],
        "registry": [
            f"{company} MCA company master data",
            f"{company} ZaubaCorp registered company",
            f"{company} Tofler legal entity directors",
            f"{company} opencorporates registration",
        ],
        "linkedin": [
            f"site:linkedin.com/company {company} employees headquarters",
            f"{company} LinkedIn company page employees",
            f"{company} LinkedIn jobs openings hiring {year}",
        ],
        "technology": [
            f"{company} BuiltWith technology stack",
            f"{company} Wappalyzer CRM ERP tools",
            f"{company} software platform tools used internally",
            f"{company} SAP Oracle Salesforce HubSpot integration",
        ],
        "pain_points": [
            f"{company} G2 reviews complaints rating problems",
            f"{company} Glassdoor reviews employee issues",
            f"{company} Trustpilot Clutch customer complaints",
            f"{company} Reddit problems issues discussion",
            f"{company} Twitter complaints customers",
            f"{company} customer support issues slow response",
            f"{company} negative reviews alternatives",
        ],
        "growth_signals": [
            f"{company} news announcement {year}",
            f"{company} latest news expansion",
            f"{company} new office hiring announcement",
            f"{company} funding raise investment {year}",
            f"{company} partnership deal launch",
            f"{company} job postings engineer manager",
        ],
        "competitors": [
            f"{company} competitors alternatives comparison",
            f"{company} vs alternatives market",
            f"alternatives to {company} best similar tools",
        ],
        "firmographics": [
            f"{company} founded year history established",
            f"{company} revenue annual turnover",
            f"{company} crunchbase company profile funding",
            f"{company} tracxn company intelligence",
        ],
    }

    flat = [query for group in workers.values() for query in group]
    # Merge with existing planner queries, preserving order and deduping
    existing = state.get("search_queries", [])
    all_queries = list(dict.fromkeys(flat + existing))

    return {
        "worker_queries": workers,
        "search_queries": all_queries,
        "status": "workers_planned",
        "progress_pct": 15,
        "node_timings": {"specialized_workers": round(time.time() - started, 2)},
        "log": [f"[Workers] 9 specialist workers contributed {len(flat)} queries for {company}"],
    }
