"""Deterministic planner for deep B2B company intelligence research."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from core.state import ResearchState
from core.db_tracer import trace_job_start


def run_planner(state: ResearchState) -> dict[str, Any]:
    started = time.time()
    company = state["company"]
    job_id = state.get("job_id", "")
    year = datetime.now().year
    trace_job_start(job_id, company)

    questions = [
        f"What is the official website of {company}?",
        f"What is the verified LinkedIn company page for {company}?",
        f"What legal entity corresponds to {company}?",
        f"Where is {company} headquartered?",
        f"Which industry and services does {company} explicitly claim?",
        f"Who founded {company} and when was it established?",
        f"What is the employee count reported on LinkedIn?",
        f"What technologies and tools does {company} use?",
        f"What are customers saying about {company} on G2, Glassdoor, Reddit?",
        f"What are the latest news and announcements from {company}?",
        f"What job openings does {company} have right now?",
        f"Who are {company}'s main competitors?",
        f"What is {company}'s revenue or funding status?",
        f"What are the pain points and complaints about {company}?",
        f"What growth signals indicate {company} is expanding?",
    ]

    queries = [
        # Identity
        f"{company} official website",
        f"site:linkedin.com/company {company}",
        f"{company} MCA ZaubaCorp Tofler legal entity registered",
        f"{company} headquarters location city country",
        f"{company} industry services overview about",
        f"{company} founders CEO leadership team",
        # Firmographics
        f"{company} employee count size LinkedIn",
        f"{company} founded year established history",
        f"{company} revenue funding investors",
        f"{company} crunchbase tracxn company profile",
        # Tech stack
        f"{company} BuiltWith Wappalyzer technology stack",
        f"{company} CRM ERP tools software used",
        # Pain points and reviews
        f"{company} reviews complaints problems customers",
        f"{company} G2 reviews rating issues",
        f"{company} Glassdoor reviews employee feedback",
        f"{company} Reddit discussion problems issues",
        f"{company} Trustpilot Clutch reviews",
        # Growth and news
        f"{company} news announcement {year}",
        f"{company} latest news expansion hiring",
        f"{company} job openings hiring LinkedIn jobs",
        f"{company} new office expansion market",
        # Competitive intelligence
        f"{company} competitors alternatives comparison",
        f"{company} vs alternatives disadvantages problems",
    ]

    return {
        "sub_questions": questions,
        "search_queries": list(dict.fromkeys(queries)),  # dedupe preserving order
        "status": "planning_done",
        "progress_pct": 8,
        "node_timings": {"planner": round(time.time() - started, 2)},
        "log": [f"[Planner] Deep B2B research plan created for {company} — {len(queries)} queries"],
    }
