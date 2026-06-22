"""
agents/business_analyst.py
Analyzes verified company facts to produce business intelligence:
- Market position
- Growth trajectory
- Pain points
- Opportunities
- Risks

Uses Groq Llama 3.3 70B (fast + strong reasoning) grounded in verified facts only.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from core.config import get_settings
from core.state import ResearchState
from core.llm_router import completion_with_fallback

logger = logging.getLogger(__name__)

_ANALYST_PROMPT = """\
You are a senior business intelligence analyst compiling a Comprehensive Intelligence Dossier. 
Based ONLY on the verified company data below, produce a deep analysis of the target company's business problems.

Company: {company}
Verified Data:
{facts_summary}

Produce analysis for (based ONLY on the data above — no assumptions, include [Source: URL] if applicable):
1. Market Position: How do they stand vs competitors?
2. Growth Trajectory: Are they growing, stable, or declining? (cite specific evidence)
3. Key Pain Points: Use ONLY the verified inferred pain points. Do not create new pain points.
4. Opportunities: What growth opportunities exist?
5. Risks: What are the main competitor threats or macroeconomic headwinds?

Return ONLY valid JSON:
{{
  "market_position": "...",
  "growth_trajectory": "...",
  "key_pain_points": [],
  "opportunities": ["...", "..."],
  "risks": ["...", "..."]
}}
"""


def _facts_to_summary(verified_facts: dict) -> str:
    lines = []
    for field, fact in verified_facts.items():
        if isinstance(fact, dict):
            val = fact.get("value")
            status = fact.get("verification_status")
            sources = fact.get("sources", [])
            if val and status == "SUPPORTED":
                src_str = ", ".join(sources[:2])
                lines.append(f"  [{field.upper()}] {val} (Sources: {src_str})")
    return "\n".join(lines) or "  (no verified facts available)"


def run_business_analyst(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: Business Analyst Agent."""
    t0 = time.time()
    company = state["company"]
    verified_facts = state.get("verified_facts", {})
    settings = get_settings()

    facts_summary = _facts_to_summary(verified_facts)
    inferred = state.get("business_analysis", {}).get("inferred_pain_points", [])
    if inferred:
        inferred_lines = "\n".join(f"  [INFERRED_PAIN_POINT] {p.get('pain_point')}" for p in inferred)
        facts_summary = f"{facts_summary}\n{inferred_lines}"
    prompt = _ANALYST_PROMPT.format(company=company, facts_summary=facts_summary)

    try:
        response, target = completion_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            settings=settings,
            temperature=0.1,
            timeout=45,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        analysis = json.loads(raw)
        if "pain_points" in verified_facts:
            analysis["key_pain_points"] = verified_facts["pain_points"].get("value", [])
        else:
            analysis["key_pain_points"] = []
    except Exception as e:
        logger.warning(f"[BusinessAnalyst] Failed: {e} — using fallback")
        analysis = {
            "market_position": "Unable to determine — insufficient data",
            "growth_trajectory": "Unable to determine — insufficient data",
            "key_pain_points": verified_facts.get("pain_points", {}).get("value", []),
            "opportunities": [],
            "risks": [],
        }

    elapsed = round(time.time() - t0, 2)
    logger.info(f"[BusinessAnalyst] Analysis completed for {company} in {elapsed}s")

    return {
        "business_analysis": analysis,
        "status": "analysis_done",
        "progress_pct": 82,
        "node_timings": {"business_analyst": elapsed},
        "log": [f"[BusinessAnalyst] Business intelligence analysis completed for {company}"],
    }
