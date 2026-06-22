"""
agents/evidence_graph.py
Evidence Graph Builder — converts top-ranked chunks into structured EvidenceItems.

Each EvidenceItem has:
- fact_candidate: extracted claim
- source_url, source_domain: provenance
- paragraph: exact source text
- date: publication date
- confidence: based on domain_score + rerank_score
- verification_status: "PENDING" (set by semantic verifier later)
- field: which structured field this supports
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
from core.state import ResearchState, EvidenceChunk, EvidenceItem
from agents.source_authority import WEAK_CONTEXT_CATEGORIES

logger = logging.getLogger(__name__)

_EVIDENCE_EXTRACT_PROMPT = """\
You are a structured evidence extractor. Extract specific factual claims from the text below about the company "{company}".

Source URL: {source_url}
Published Date: {date}

Text:
---
{chunk}
---

Extract all factual claims you find. For each claim, specify which field it belongs to:
- "pain_points": challenges, problems, complaints, bottlenecks
- "revenue": revenue, ARR, funding amounts, valuations
- "competitors": named competitors, market threats
- "technology": tech stack, software used, platforms
- "leadership": CEO, executives, leadership changes
- "overview": what the company does, products, services
- "recent_news": announcements, acquisitions, launches, layoffs
- "growth": growth metrics, expansion, market share
- "risks": regulatory, competitive, operational risks

RULES:
- Only extract claims EXPLICITLY stated in the text above
- Never invent or infer facts not clearly present
- Each claim must be a single, specific sentence
- Do NOT extract company pain points from individual LinkedIn posts, tweets, blogs, or thought-leadership content
- If no relevant claims found, return empty array

Return ONLY valid JSON:
{{
  "claims": [
    {{"field": "pain_points", "claim": "exact claim from text", "paragraph": "exact source sentence"}},
    ...
  ]
}}
"""


def _extract_claims_from_chunk(
    chunk: EvidenceChunk,
    company: str,
    settings,
) -> list[EvidenceItem]:
    """Extract structured claims from a single chunk using the LLM."""
    prompt = _EVIDENCE_EXTRACT_PROMPT.format(
        company=company,
        source_url=chunk["url"],
        date=chunk.get("published_date", "unknown"),
        chunk=chunk["chunk"][:3000],
    )

    try:
        response, target = completion_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            settings=settings,
            temperature=0.0,
            timeout=20,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        claims = data.get("claims", [])

        evidence_items = []
        authority = chunk.get("authority_score", 0.5)
        category = chunk.get("source_category", "UNKNOWN")
        base_confidence = min(0.5 + chunk.get("rerank_score", 0.5) * 0.4, 0.9)
        if category in WEAK_CONTEXT_CATEGORIES:
            base_confidence = min(base_confidence, 0.30)

        for claim in claims:
            if not claim.get("claim") or not claim.get("field"):
                continue
            if claim.get("field") == "pain_points" and category in WEAK_CONTEXT_CATEGORIES:
                continue
            evidence_items.append(EvidenceItem(
                fact=claim["claim"],
                fact_candidate=claim["claim"],
                source_url=chunk["url"],
                source_domain=chunk.get("domain", ""),
                paragraph=claim.get("paragraph", chunk["chunk"][:500]),
                date=chunk.get("published_date", ""),
                publication_date=chunk.get("published_date", ""),
                authority_score=authority,
                relevance_score=chunk.get("rerank_score", 0.0),
                confidence=base_confidence,
                verification_status="PENDING",
                field=claim["field"],
                category=category,
            ))

        return evidence_items

    except Exception as e:
        logger.debug(f"[EvidenceGraph] Claim extraction failed for {chunk['url']}: {e}")
        return []


def run_evidence_graph_builder(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: Evidence Graph Builder."""
    t0 = time.time()
    company = state["company"]
    evidence_chunks: list[EvidenceChunk] = state.get("evidence_chunks", [])
    settings = get_settings()

    if not evidence_chunks:
        logger.warning("[EvidenceGraph] No chunks to process")
        return {
            "evidence_items": [],
            "status": "evidence_graph_built",
            "progress_pct": 58,
            "node_timings": {"evidence_graph": 0},
            "log": ["[EvidenceGraph] No chunks — empty evidence graph"],
        }

    logger.info(f"[EvidenceGraph] Building evidence graph from {len(evidence_chunks)} chunks")

    all_items: list[EvidenceItem] = []
    # Process top 20 chunks (the best ones after reranking)
    for chunk in evidence_chunks[:20]:
        items = _extract_claims_from_chunk(chunk, company, settings)
        all_items.extend(items)

    # Deduplicate: remove near-identical claims
    seen_claims = set()
    unique_items = []
    for item in all_items:
        claim_key = item["fact_candidate"][:100].lower().strip()
        if claim_key not in seen_claims:
            seen_claims.add(claim_key)
            unique_items.append(item)

    elapsed = round(time.time() - t0, 2)
    logger.info(f"[EvidenceGraph] Built {len(unique_items)} unique evidence items in {elapsed}s")

    # Group by field for observability
    field_counts = {}
    for item in unique_items:
        field_counts[item["field"]] = field_counts.get(item["field"], 0) + 1
    logger.info(f"[EvidenceGraph] Field distribution: {field_counts}")

    return {
        "evidence_items": unique_items,
        "status": "evidence_graph_built",
        "progress_pct": 58,
        "node_timings": {"evidence_graph": elapsed},
        "log": [f"[EvidenceGraph] {len(unique_items)} evidence items | fields: {field_counts}"],
    }


def check_evidence_loop(state: ResearchState) -> str:
    """
    Conditional edge: decide whether to loop back for more research or proceed.
    Returns: "refine_search" | "proceed"
    """
    evidence_items: list[EvidenceItem] = state.get("evidence_items", [])
    retry_count = state.get("retry_count", 0)
    settings = get_settings()

    min_required = settings.min_evidence_items
    max_retries = settings.max_retry_count

    # Count items per critical field. Pain points are optional here because they
    # are inferred later and require at least two supporting EvidenceItems.
    identity = [e for e in evidence_items if e["field"] in ("overview", "leadership", "technology", "recent_news", "growth")]

    sufficient = (
        len(evidence_items) >= min_required and
        len(identity) >= 1
    )

    if sufficient or retry_count >= max_retries:
        logger.info(f"[EvidenceGraph] Proceeding with {len(evidence_items)} items (retry={retry_count})")
        return "proceed"
    else:
        logger.info(f"[EvidenceGraph] Insufficient evidence ({len(evidence_items)} items) — refining search (retry {retry_count + 1}/{max_retries})")
        return "refine_search"
