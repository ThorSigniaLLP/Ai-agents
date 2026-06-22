"""
agents/reranker.py
CrossEncoder Reranker with LLM fallback.

Scores each evidence chunk against the company research query using
Groq's fast Llama 3.1 8B model. No heavy local model download required.

Keeps top 30 chunks. Discards the rest.
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
from core.state import ResearchState, EvidenceChunk
from agents.source_authority import combined_score

logger = logging.getLogger(__name__)

_RERANK_PROMPT = """\
You are a relevance scorer for a business intelligence research system.

Target Company: {company}
Research Goal: Verify company identity, official website, industry, headquarters, services, founders, legal entity, employee count, and technologies.

Rate the following text chunk on a relevance scale from 0.0 to 1.0:
- 1.0 = Highly relevant (directly discusses company challenges, financials, strategy, complaints)
- 0.5 = Somewhat relevant (mentions company or industry)
- 0.0 = Not relevant (generic content, ads, unrelated)

Text chunk:
---
{chunk}
---

Return ONLY a JSON object: {{"score": 0.0}}
"""

_CROSS_ENCODER = None
_CROSS_ENCODER_FAILED = False


def _cross_encoder_rerank(chunks: list[EvidenceChunk], company: str) -> list[float] | None:
    """Use BAAI/bge-reranker-v2-m3 when sentence-transformers is installed."""
    global _CROSS_ENCODER, _CROSS_ENCODER_FAILED
    if _CROSS_ENCODER_FAILED or not chunks:
        return None
    try:
        if _CROSS_ENCODER is None:
            from sentence_transformers import CrossEncoder
            _CROSS_ENCODER = CrossEncoder("BAAI/bge-reranker-v2-m3")
        query = (
            f"Company identity evidence for {company}: official website, industry, headquarters, services, "
            "founders, legal entity, employee count, and detected technology stack"
        )
        pairs = [(query, c["chunk"][:1200]) for c in chunks]
        raw_scores = _CROSS_ENCODER.predict(pairs)
        scores = [float(s) for s in raw_scores]
        if not scores:
            return None
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return [0.5 for _ in scores]
        return [(score - lo) / (hi - lo) for score in scores]
    except Exception as e:
        _CROSS_ENCODER_FAILED = True
        logger.warning(f"[Reranker] CrossEncoder unavailable, falling back to LLM: {e}")
        return None

_BATCH_RERANK_PROMPT = """\
You are a relevance scorer for business intelligence research.

Target Company: {company}
Research Goal: Verify company identity and explicit profile facts only.

Rate each of these {count} text chunks on a scale from 0.0 to 1.0:
- 1.0 = Directly discusses company's challenges, problems, financials, or strategy
- 0.5 = Mentions company or is generally relevant to the industry
- 0.0 = Irrelevant, generic, or just navigation/ads

Chunks:
{chunks_text}

Return ONLY a JSON array of scores in the same order: [0.8, 0.3, ...]
"""


def _batch_rerank(chunks: list[EvidenceChunk], company: str, settings) -> list[float]:
    """Score all chunks in one LLM batch call for efficiency."""
    if not chunks:
        return []

    # Build numbered chunk list
    chunks_text = "\n\n".join(
        f"[{i+1}] {c['chunk'][:500]}..."
        for i, c in enumerate(chunks)
    )

    prompt = _BATCH_RERANK_PROMPT.format(
        company=company,
        count=len(chunks),
        chunks_text=chunks_text,
    )

    try:
        response, target = completion_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            settings=settings,
            temperature=0.0,
            timeout=30,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        # Extract the JSON array
        match = re.search(r"\[[\d\s.,]+\]", raw)
        if match:
            scores = json.loads(match.group(0))
            if len(scores) == len(chunks):
                return [float(s) for s in scores]
    except Exception as e:
        logger.warning(f"[Reranker] Batch LLM rerank failed: {e}")

    # Fallback: return existing rerank scores
    return [c["rerank_score"] for c in chunks]


def run_reranker(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: LLM-based Reranker."""
    t0 = time.time()
    company = state["company"]
    evidence_chunks: list[EvidenceChunk] = state.get("evidence_chunks", [])
    settings = get_settings()

    if not evidence_chunks:
        logger.warning("[Reranker] No evidence chunks to rerank")
        return {
            "evidence_chunks": [],
            "status": "reranking_done",
            "progress_pct": 55,
            "node_timings": {"reranker": 0},
            "log": ["[Reranker] No chunks to rerank"],
        }

    logger.info(f"[Reranker] Reranking {len(evidence_chunks)} chunks for {company}")

    all_scores = _cross_encoder_rerank(evidence_chunks, company)

    if all_scores is None:
        # Process in batches of 15 to stay within token limits
        BATCH_SIZE = 15
        all_scores = []
        for batch_start in range(0, len(evidence_chunks), BATCH_SIZE):
            batch = evidence_chunks[batch_start:batch_start + BATCH_SIZE]
            scores = _batch_rerank(batch, company, settings)
            all_scores.extend(scores)

    # Apply new rerank scores multiplied by source authority
    for chunk, score in zip(evidence_chunks, all_scores):
        chunk["rerank_score"] = combined_score(float(score), chunk.get("authority_score", 0.5)) * chunk.get("page_authority_score", 1.0)

    # Sort by rerank score and keep top 30
    evidence_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_chunks = evidence_chunks[:30]

    elapsed = round(time.time() - t0, 2)
    logger.info(f"[Reranker] Kept top {len(top_chunks)} chunks after reranking in {elapsed}s")

    return {
        "evidence_chunks": top_chunks,  # Replace with re-ranked subset
        "status": "reranking_done",
        "progress_pct": 55,
        "node_timings": {"reranker": elapsed},
        "log": [f"[Reranker] {len(top_chunks)} top chunks selected after LLM reranking"],
    }
