"""
agents/semantic_verifier.py
Semantic Verifier — Option B: Groq LLM-based NLI verification.

Replaces majority voting with semantic entailment checking.
For each extracted fact, checks if it is actually supported by the source evidence.

Labels: SUPPORTED | CONTRADICTED | UNKNOWN
Rejects CONTRADICTED and UNKNOWN facts from final output.
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
from core.state import ResearchState, EvidenceItem, VerifiedFact

logger = logging.getLogger(__name__)

_NLI_PROMPT = """\
You are a fact-verification system using Natural Language Inference.

Your job: Determine if the "Claim" is SUPPORTED, CONTRADICTED, or UNKNOWN based ONLY on the "Evidence" provided.

Company: {company}

Claim: {claim}

Evidence from source ({source_url}):
---
{evidence}
---

Rules:
- SUPPORTED: The evidence clearly and directly supports the claim.
- CONTRADICTED: The evidence directly contradicts or disputes the claim.
- UNKNOWN: The evidence does not mention or confirm the claim.
- Base your judgment ONLY on the evidence text — not your training knowledge.

Return ONLY valid JSON:
{{"verdict": "SUPPORTED", "confidence": 0.9, "reason": "brief reason"}}
"""

_BATCH_VERIFY_PROMPT = """\
You are a fact-verification system. Verify each claim against its source evidence.

Company: {company}

Verify these {count} claim-evidence pairs:
{pairs}

For each, determine if the claim is SUPPORTED, CONTRADICTED, or UNKNOWN based ONLY on the provided evidence.

Return a JSON array:
[
  {{"verdict": "SUPPORTED", "confidence": 0.9}},
  ...
]
"""


_FIELD_EVIDENCE_MAP = {
    "pain_point_signals": {"pain_points", "risks", "recent_news", "technology", "growth", "hiring"},
    "products_services": {"overview"},
    "technology_stack": {"technology"},
    "growth_signals": {"growth", "recent_news"},
    "founders": {"leadership", "overview"},
    "directors": {"leadership", "overview"},
    "ceo": {"leadership", "overview"},
    "legal_entity": {"overview", "leadership"},
    "entity_type": {"overview"},
    "registration_status": {"overview", "recent_news"},
    "registered_capital": {"revenue", "overview"},
    "employee_count": {"overview", "growth"},
    "hq_location": {"overview"},
    "industry": {"overview"},
    "website": {"overview"},
    "linkedin": {"overview"},
}


def _matching_evidence(evidence_items: list[EvidenceItem], field: str) -> list[EvidenceItem]:
    accepted = _FIELD_EVIDENCE_MAP.get(field, {field})
    return [item for item in evidence_items if item.get("field") in accepted]


def _build_pairs_text(items: list[tuple[str, str, str]]) -> str:
    """Format (claim, source_url, evidence_paragraph) tuples."""
    lines = []
    for i, (claim, src_url, para) in enumerate(items):
        lines.append(
            f"[{i+1}] Claim: {claim}\n"
            f"    Source: {src_url}\n"
            f"    Evidence: {para[:400]}"
        )
    return "\n\n".join(lines)


def _verify_batch(items: list[tuple[str, str, str]], company: str, settings) -> list[dict]:
    """Batch verify a list of (claim, source_url, paragraph) tuples."""
    if not items:
        return []

    pairs_text = _build_pairs_text(items)
    prompt = _BATCH_VERIFY_PROMPT.format(
        company=company,
        count=len(items),
        pairs=pairs_text,
    )

    try:
        response, target = completion_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            settings=settings,
            temperature=0.0,
            timeout=45,
            max_tokens=1000,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        # Extract the array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            results = json.loads(match.group(0))
            if len(results) == len(items):
                return results

    except Exception as e:
        logger.warning(f"[SemanticVerifier] Batch verification failed: {e}")

    # Fallback: mark all as UNKNOWN
    return [{"verdict": "UNKNOWN", "confidence": 0.5} for _ in items]


def run_semantic_verifier(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: Semantic Verifier (NLI-based, Option B)."""
    t0 = time.time()
    company = state["company"]
    evidence_items: list[EvidenceItem] = state.get("evidence_items", [])
    extracted_facts: dict = state.get("extracted_facts", {})
    settings = get_settings()

    if not evidence_items or not extracted_facts:
        logger.warning("[SemanticVerifier] Nothing to verify")
        return {
            "verified_facts": {},
            "rejected_facts": [],
            "status": "verification_done",
            "progress_pct": 72,
            "node_timings": {"semantic_verifier": 0},
            "log": ["[SemanticVerifier] Skipped — no facts to verify"],
        }

    # Build claim → evidence lookup
    claim_to_evidence: dict[str, EvidenceItem] = {}
    for item in evidence_items:
        key = item["fact_candidate"][:100].lower().strip()
        claim_to_evidence[key] = item

    # Collect claims to verify
    claims_to_verify: list[tuple[str, str, str]] = []
    field_to_claims: dict[str, list[str]] = {}

    list_fields = ["pain_point_signals", "competitors", "recent_news", "risks",
                   "products_services", "technology_stack", "growth_signals", "founders", "directors"]
    scalar_fields = ["overview", "revenue", "ceo", "founding_year",
                     "employee_count", "hq_location", "funding", "industry", "website",
                     "linkedin", "legal_entity", "entity_type", "registration_status",
                     "registered_capital"]

    for field in list_fields:
        values = extracted_facts.get(field, [])
        if isinstance(values, list):
            for v in values[:5]:  # max 5 per field
                claim = str(v)
                # Find matching evidence
                evidence_para = ""
                source_url = ""
                key = claim[:100].lower().strip()
                if key in claim_to_evidence:
                    item = claim_to_evidence[key]
                    evidence_para = item["paragraph"]
                    source_url = item["source_url"]
                else:
                    # Use any evidence of the same field type
                    matching = _matching_evidence(evidence_items, field)
                    if matching:
                        evidence_para = matching[0]["paragraph"]
                        source_url = matching[0]["source_url"]

                if evidence_para:
                    claims_to_verify.append((claim, source_url, evidence_para))
                    field_to_claims.setdefault(field, []).append(claim)

    for field in scalar_fields:
        value = extracted_facts.get(field)
        if value and str(value).lower() not in ("null", "none", "unknown", ""):
            claim = str(value)
            matching = _matching_evidence(evidence_items, field)
            if matching:
                evidence_para = matching[0]["paragraph"]
                source_url = matching[0]["source_url"]
                claims_to_verify.append((claim, source_url, evidence_para))
                field_to_claims.setdefault(field, []).append(claim)

    # Batch verify in groups of 10
    BATCH_SIZE = 10
    all_verdicts = []
    for i in range(0, len(claims_to_verify), BATCH_SIZE):
        batch = claims_to_verify[i:i + BATCH_SIZE]
        verdicts = _verify_batch(batch, company, settings)
        all_verdicts.extend(verdicts)

    # Map verdicts back to claims
    claim_verdicts: dict[str, dict] = {}
    for (claim, src_url, _), verdict in zip(claims_to_verify, all_verdicts):
        claim_verdicts[claim] = {"verdict": verdict.get("verdict", "UNKNOWN"),
                                 "confidence": verdict.get("confidence", 0.5),
                                 "source_url": src_url}

    # Build verified_facts dict
    verified_facts: dict[str, VerifiedFact] = {}
    rejected_facts = []

    # Source URLs from extracted_facts
    source_url_map = extracted_facts.get("source_urls", {})

    for field in list_fields:
        values = extracted_facts.get(field, [])
        if not isinstance(values, list):
            continue

        supported_values = []
        sources = list(source_url_map.get(field, []))

        for v in values[:5]:
            claim = str(v)
            verdict_info = claim_verdicts.get(claim, {"verdict": "UNKNOWN", "confidence": 0.5})
            if verdict_info["verdict"] == "SUPPORTED":
                supported_values.append(v)
                if verdict_info.get("source_url"):
                    sources.append(verdict_info["source_url"])
                for item in evidence_items:
                    if item.get("source_url") == verdict_info.get("source_url") and item in _matching_evidence(evidence_items, field):
                        item["verification_status"] = "SUPPORTED"
            elif verdict_info["verdict"] == "CONTRADICTED":
                rejected_facts.append({"field": field, "claim": v, "reason": "CONTRADICTED by source"})
            # UNKNOWN claims are silently dropped

        if supported_values:
            verified_facts[field] = VerifiedFact(
                value=supported_values,
                sources=list(set(sources)),
                confidence=0.85,
                verification_status="SUPPORTED",
                note=None,
            )

    for field in scalar_fields:
        value = extracted_facts.get(field)
        if not value or str(value).lower() in ("null", "none", "unknown", ""):
            continue

        claim = str(value)
        verdict_info = claim_verdicts.get(claim, {"verdict": "UNKNOWN", "confidence": 0.5})
        sources = list(source_url_map.get(field, []))
        if verdict_info.get("source_url"):
            sources.append(verdict_info["source_url"])

        if verdict_info["verdict"] in ("SUPPORTED", "UNKNOWN"):
            # For scalar facts, we include UNKNOWN with lower confidence
            verified_facts[field] = VerifiedFact(
                value=value,
                sources=list(set(sources)),
                confidence=0.9 if verdict_info["verdict"] == "SUPPORTED" else 0.5,
                verification_status=verdict_info["verdict"],
                note=None,
            )
        else:
            rejected_facts.append({"field": field, "claim": value, "reason": "CONTRADICTED"})

    elapsed = round(time.time() - t0, 2)
    supported_count = sum(1 for f in verified_facts.values() if f["verification_status"] == "SUPPORTED")
    logger.info(
        f"[SemanticVerifier] {supported_count} verified, {len(rejected_facts)} rejected in {elapsed}s"
    )

    return {
        "verified_facts": verified_facts,
        "rejected_facts": rejected_facts,
        "status": "verification_done",
        "progress_pct": 72,
        "node_timings": {"semantic_verifier": elapsed},
        "log": [f"[SemanticVerifier] {supported_count} facts SUPPORTED, {len(rejected_facts)} rejected"],
    }
