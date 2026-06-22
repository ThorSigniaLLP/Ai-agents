"""
agents/sales_strategist.py
Maps company intelligence to sales strategy:
- Best messaging angle
- Key product benefits to highlight
- Objection handlers
- Best contact approach

Uses Gemini 2.5 Flash for strategic reasoning.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from litellm import completion as litellm_completion
from core.llm_router import completion_with_fallback

from core.config import get_settings
from core.state import ResearchState

logger = logging.getLogger(__name__)

_STRATEGIST_PROMPT = """\
You are a sales strategy consultant. Given a target company profile and business analysis,
craft a precise sales strategy for reaching out to this company.

Company: {company}
Contact: {contact_name} ({contact_title})

Company Profile:
{facts_summary}

Business Analysis:
- Market Position: {market_position}
- Growth Trajectory: {growth_trajectory}
- Pain Points: {pain_points}
- Opportunities: {opportunities}

Our Product/Service Context: B2B sales intelligence and outreach tool.

Based ONLY on the above information, provide:
1. The single best messaging angle (what resonates most with their current situation)
2. Top 3 specific benefits to highlight (tied to their actual pain points)
3. Top 2 likely objections and how to handle them
4. Best contact approach (email? LinkedIn? call? sequence?)

Return ONLY valid JSON:
{{
  "messaging_angle": "...",
  "key_benefits": ["...", "...", "..."],
  "objection_handlers": ["Objection: ... Response: ...", "..."],
  "best_contact_approach": "..."
}}
"""


def run_sales_strategist(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: Sales Strategist Agent."""
    t0 = time.time()
    company = state["company"]
    card_info = state.get("card_info", {})
    verified_facts = state.get("verified_facts", {})
    business_analysis = state.get("business_analysis", {})
    settings = get_settings()

    def _get_val(field: str, default: str = "Unknown") -> str:
        f = verified_facts.get(field, {})
        if isinstance(f, dict):
            v = f.get("value")
            if f.get("verification_status") == "SUPPORTED" and v:
                return str(v)
        return default

    facts_summary = "\n".join([
        f"  Industry: {_get_val('industry')}",
        f"  Overview: {_get_val('overview')}",
        f"  Products/Services: {_get_val('products_services')}",
        f"  Competitors: {_get_val('competitors')}",
        f"  Recent News: {_get_val('recent_news')}",
    ])

    prompt = _STRATEGIST_PROMPT.format(
        company=company,
        contact_name=card_info.get("name", "Unknown"),
        contact_title=card_info.get("job_title", "Unknown"),
        facts_summary=facts_summary,
        market_position=business_analysis.get("market_position", "Unknown"),
        growth_trajectory=business_analysis.get("growth_trajectory", "Unknown"),
        pain_points=str(business_analysis.get("key_pain_points", [])),
        opportunities=str(business_analysis.get("opportunities", [])),
    )

    try:
        response, target = completion_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            settings=settings,
            temperature=0.2,
            timeout=45,
            response_format={"type": "json_object"}
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        strategy = json.loads(raw)
    except Exception as e:
        logger.warning(f"[SalesStrategist] Failed: {e} — using fallback")
        supported_pain_points = business_analysis.get("key_pain_points", [])
        strategy = {
            "messaging_angle": (
                f"Address {supported_pain_points[0]}" if supported_pain_points
                else f"Explore fit with {company} based on its verified company profile"
            ),
            "key_benefits": [],
            "objection_handlers": [],
            "best_contact_approach": "Email first, then LinkedIn follow-up",
        }

    elapsed = round(time.time() - t0, 2)
    logger.info(f"[SalesStrategist] Strategy crafted for {company} in {elapsed}s")

    return {
        "sales_strategy": strategy,
        "status": "strategy_done",
        "progress_pct": 88,
        "node_timings": {"sales_strategist": elapsed},
        "log": [f"[SalesStrategist] Sales strategy generated for {company}"],
    }
