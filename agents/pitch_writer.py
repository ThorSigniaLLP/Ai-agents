"""
agents/pitch_writer.py
Generates personalized outreach content:
  - Personalized email (subject + body)
  - LinkedIn connection message
  - Follow-up email

Rules:
  - Reference specific recent company news/facts from research
  - No generic language ("I hope this email finds you well")
  - First sentence must reference something specific about the company
  - Keep LinkedIn message under 300 characters
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

_PITCH_PROMPT = """\
You are an expert B2B sales copywriter. Write hyper-personalized outreach for a prospect.

Prospect Details:
  Name: {contact_name}
  Title: {contact_title}
  Company: {company}

Company Intelligence (VERIFIED FACTS — use these specifically):
  Overview: {overview}
  Recent News: {recent_news}
  Products: {products}
  Strategic Priorities: {priorities}

Sales Strategy:
  Messaging Angle: {messaging_angle}
  Key Benefits: {key_benefits}

RULES:
1. Open with a SPECIFIC reference to recent company news or a verified fact — not generic praise
2. Be concise — email body max 120 words
3. LinkedIn message MUST be under 280 characters
4. Never use: "I hope this email finds you well", "I wanted to reach out", "synergies"
5. Use the prospect's first name only
6. Reference the messaging angle naturally

Return ONLY valid JSON:
{{
  "email_subject": "...",
  "email_body": "...",
  "linkedin_message": "...",
  "follow_up_message": "...",
  "talking_points": ["...", "...", "..."]
}}
"""


def run_pitch_writer(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: Pitch Writer Agent."""
    t0 = time.time()
    company = state["company"]
    card_info = state.get("card_info", {})
    verified_facts = state.get("verified_facts", {})
    business_analysis = state.get("business_analysis", {})
    sales_strategy = state.get("sales_strategy", {})
    settings = get_settings()

    def _get_val(field: str, default: str = "") -> str:
        f = verified_facts.get(field, {})
        if isinstance(f, dict):
            v = f.get("value")
            if f.get("verification_status") == "SUPPORTED" and v:
                if isinstance(v, list):
                    return "; ".join(str(i) for i in v[:3])
                return str(v)
        return default

    prompt = _PITCH_PROMPT.format(
        contact_name=card_info.get("name", "there"),
        contact_title=card_info.get("job_title", ""),
        company=company,
        overview=_get_val("overview", "a leading company in their space"),
        recent_news=_get_val("recent_news", "their recent growth initiatives"),
        products=_get_val("products_services", "their core solutions"),
        priorities=_get_val("strategic_priorities", "expanding their market presence"),
        messaging_angle=sales_strategy.get("messaging_angle", ""),
        key_benefits="; ".join(sales_strategy.get("key_benefits", [])),
    )

    try:
        response, target = completion_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            settings=settings,
            temperature=0.4,
            timeout=45,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        # Strip <think> tags if model outputs chain-of-thought
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        pitch = json.loads(raw)
    except Exception as e:
        logger.error(f"[PitchWriter] Models failed: {e}")
            pitch = {
                "email_subject": f"Quick question about {company}'s growth",
                "email_body": f"Hi {card_info.get('name', 'there')},\n\nI noticed {company}'s recent work and wanted to connect about how we might help.\n\nWould love a quick chat.\n\nBest,",
                "linkedin_message": f"Hi, noticed {company}'s recent work — think we could help. Worth a 15-min chat?",
                "follow_up_message": "Following up on my previous message...",
                "talking_points": [],
            }

    elapsed = round(time.time() - t0, 2)
    logger.info(f"[PitchWriter] Pitch generated for {company} in {elapsed}s")

    return {
        "pitch": pitch,
        "status": "pitch_done",
        "progress_pct": 95,
        "node_timings": {"pitch_writer": elapsed},
        "log": [f"[PitchWriter] Personalized pitch generated for {card_info.get('name', company)}"],
    }
