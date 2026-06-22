"""
agents/fact_extractor.py
Fact Extractor — single model extraction with mandatory citations.

Replaces the 5-model majority voting approach.
Primary: groq/qwen/qwen3-32b
Fallback: groq/llama-3.3-70b-versatile

Every extracted fact MUST include source_urls.
Facts without citations are discarded.
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
from core.state import ResearchState, EvidenceItem

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a structured data extractor. Your job is to synthesize evidence into structured company facts.

CRITICAL RULES:
1. You MUST ONLY use information from the provided evidence items.
2. You MUST include source_urls for every field you populate.
3. If a field has no evidence, return null.
4. Never guess, infer, or use your training knowledge.
5. Return ONLY valid JSON.
"""

_EXTRACTION_PROMPT = """\
Synthesize the following evidence items about "{company}" into a structured company profile.

EVIDENCE ITEMS:
---
{evidence_text}
---

Extract the following fields (use null if no evidence found):

{{
  "overview": null,           // What the company does (1-2 sentences max)
  "industry": null,           // Industry sector
  "website": null,           // Official company website
  "linkedin": null,          // LinkedIn company page only
  "hq_location": null,        // Headquarters city/country
  "founding_year": null,      // Year founded
  "employee_count": null,     // Number of employees
  "founders": [],             // Named founders
  "legal_entity": null,       // Registered legal company name
  "entity_type": null,        // Pvt Ltd, LLC, PLC, etc.
  "registration_status": null,// Registry status
  "directors": [],            // Registry-listed directors
  "registered_capital": null, // Authorized/paid-up capital
  "revenue": null,            // Annual revenue or ARR
  "ceo": null,                // CEO name
  "products_services": [],    // List of main products/services
  "competitors": [],          // List of named competitors
  "funding": null,            // Total funding or last round
  "investors": [],            // Notable investors
  "recent_news": [],          // Recent significant news items
  "pain_point_signals": [],   // Evidence-backed signals only; final pain points are inferred later
  "risks": [],                // Key risks and threats
  "technology_stack": [],     // Technologies, software, platforms used
  "growth_signals": [],       // Positive growth indicators
  "source_urls": {{           // MANDATORY: source URLs for each populated field
    "overview": [],
    "pain_point_signals": [],
    "revenue": [],
    "competitors": [],
    "recent_news": []
  }}
}}

IMPORTANT: Do not directly output inferred pain points. Only extract explicitly stated signals such as employee/customer complaints, negative news, technology gaps, hiring patterns, manual processes, or growth bottlenecks.
"""


def _build_evidence_text(evidence_items: list[EvidenceItem], max_chars: int = 15000) -> str:
    """Format evidence items into a readable block for the LLM."""
    lines = []
    total = 0
    items_by_field: dict[str, list] = {}

    # Group by field
    for item in evidence_items:
        field = item.get("field", "other")
        items_by_field.setdefault(field, []).append(item)

    # Pain points first (highest priority)
    priority_order = ["pain_points", "recent_news", "risks", "overview",
                      "revenue", "competitors", "technology", "growth", "leadership"]

    for field in priority_order:
        if field not in items_by_field:
            continue
        for item in items_by_field[field]:
            line = (
                f"[{item['field'].upper()}] Source: {item['source_url']}\n"
                f"Date: {item.get('date', 'unknown')}\n"
                f"Claim: {item['fact_candidate']}\n"
                f"Context: {item['paragraph'][:400]}\n"
            )
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)

    return "\n---\n".join(lines) if lines else "(no evidence available)"


def run_fact_extractor(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: Fact Extractor."""
    t0 = time.time()
    company = state["company"]
    evidence_items: list[EvidenceItem] = state.get("evidence_items", [])
    settings = get_settings()

    if not evidence_items:
        logger.warning("[FactExtractor] No evidence items — skipping extraction")
        return {
            "extracted_facts": {},
            "extraction_errors": ["No evidence items available for extraction"],
            "status": "extraction_skipped",
            "progress_pct": 65,
            "node_timings": {"fact_extractor": 0},
            "log": ["[FactExtractor] Skipped — no evidence found"],
        }

    evidence_text = _build_evidence_text(evidence_items)
    prompt = _EXTRACTION_PROMPT.format(company=company, evidence_text=evidence_text)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    extracted = {}
    errors = []

    # Primary: configured models via router
    try:
        response, target = completion_with_fallback(
            messages=messages,
            settings=settings,
            temperature=0.0,
            timeout=60,
        )
        raw = response.choices[0].message.content.strip()
        # Strip <think> tags (Qwen3 chain-of-thought)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1:
            extracted = json.loads(raw[start:end])
        logger.info(f"[FactExtractor] Model {target.model} succeeded")

    except Exception as e:
        logger.error(f"[FactExtractor] Models failed: {e}")
        errors.append(str(e))

    elapsed = round(time.time() - t0, 2)
    signals = extracted.get("pain_point_signals", [])
    logger.info(f"[FactExtractor] Extracted {len(extracted)} fields, {len(signals)} pain-point signals in {elapsed}s")

    return {
        "extracted_facts": extracted,
        "extraction_errors": errors,
        "status": "extraction_done",
        "progress_pct": 65,
        "node_timings": {"fact_extractor": elapsed},
        "log": [f"[FactExtractor] {len(extracted)} fields extracted | {len(signals)} pain-point signals"],
    }
