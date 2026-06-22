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
You are a senior B2B investigative intelligence analyst compiling a Comprehensive Intelligence Dossier. 
Based ONLY on the raw web evidence chunks below, produce a deep, structured analysis of the target company.

Company: {company}
Verified Facts (Summary):
{facts_summary}

Raw Web Evidence:
{chunks_text}

Produce a detailed markdown report analyzing the company's business challenges, explicitly categorizing them using these lenses:
1. GAPS – What’s Missing (e.g., no case studies, missing website features, poor digital presence)
2. CONSTRAINTS – What Limits Them (e.g., small team size, geographic concentration, dependency on a single vendor, lack of funding)
3. FRICTION – What Makes Things Harder (e.g., poor email open rates, lack of pricing transparency, manual processes)
4. PRESSURE – Stress or Urgency Signals (e.g., recent leadership changes, hiring sprees, aggressive competitor threats)
5. INFERENCES – Hidden Problems (Logical deductions from the observed data)

After the categorizations, provide a final table "Extracted Business Challenges & Pain Points" with 4 columns:
| # | Challenge / Pain Point | Evidence (Direct Quote or Logical Inference) | Solvability Lens |

Output ONLY the raw markdown report. Do not use code blocks around the entire output.
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
    chunks = state.get("evidence_chunks", [])
    # Grab the top 15 chunks based on rerank score
    top_chunks = sorted(chunks, key=lambda c: c.get("rerank_score", 0), reverse=True)[:15]
    
    chunks_text = "\n\n".join(
        f"[Source: {c.get('url', 'UNKNOWN')}]\n{c.get('chunk', '')[:1000]}"
        for c in top_chunks
    )
    if not chunks_text:
        chunks_text = "No raw web evidence available."

    prompt = _ANALYST_PROMPT.format(
        company=company, 
        facts_summary=facts_summary,
        chunks_text=chunks_text
    )

    try:
        response, target = completion_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            settings=settings,
            temperature=0.1,
            timeout=60,
        )
        report = response.choices[0].message.content.strip()
        analysis = {"deep_analysis_report": report}
    except Exception as e:
        logger.warning(f"[BusinessAnalyst] Failed: {e} — using fallback")
        analysis = {"deep_analysis_report": "Unable to generate deep analysis report due to an error or timeout."}

    elapsed = round(time.time() - t0, 2)
    logger.info(f"[BusinessAnalyst] Deep Analysis completed for {company} in {elapsed}s")

    return {
        "business_analysis": analysis,
        "status": "analysis_done",
        "progress_pct": 82,
        "node_timings": {"business_analyst": elapsed},
        "log": [f"[BusinessAnalyst] Deep Business intelligence analysis completed for {company}"],

    }
