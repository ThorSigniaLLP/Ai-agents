"""
agents/content_cleaner.py
Content Cleaner — chunks documents and attaches rich metadata.

Chunk size: 3000-5000 characters
Metadata per chunk: url, title, domain, published_date, source_type
Strips: nav, ads, cookie banners, headers, footers (via trafilatura favor_precision)
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from core.state import ResearchState, PageResult, EvidenceChunk
from agents.source_authority import combined_score
from core.db_tracer import trace_evidence_chunks

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 3500    # characters
_OVERLAP = 200        # overlap between chunks


def _chunk_text(text: str) -> list[str]:
    """Split into overlapping chunks, preserving paragraph boundaries."""
    if len(text) <= _CHUNK_SIZE:
        return [text]

    # Try to split on paragraph boundaries
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > _CHUNK_SIZE and current:
            chunks.append(current.strip())
            # Overlap: keep last _OVERLAP chars
            current = current[-_OVERLAP:] + "\n\n" + para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    # If still empty (no paragraph breaks), fall back to char-based chunking
    if not chunks:
        start = 0
        while start < len(text):
            chunks.append(text[start:start + _CHUNK_SIZE])
            start += _CHUNK_SIZE - _OVERLAP

    return chunks


def _score_relevance(chunk: str, company: str) -> float:
    """Enhanced relevance scoring — pain points weighted highest."""
    company_lower = company.lower()
    chunk_lower = chunk.lower()

    score = 0.0

    # Company name present
    if company_lower in chunk_lower:
        score += 0.35

    identity_kws = [
        "about", "industry", "headquarters", "located", "founded", "founder",
        "service", "solution", "product", "employee", "company size", "legal",
        "registered", "director", "technology", "platform", "software", "cloud",
    ]
    hits = sum(1 for kw in identity_kws if kw in chunk_lower)
    score += min(hits * 0.04, 0.45)

    return round(min(score, 1.0), 3)


def run_content_cleaner(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: Content Cleaner."""
    t0 = time.time()
    company = state["company"]
    raw_pages: list[PageResult] = state.get("raw_pages", [])

    logger.info(f"[ContentCleaner] Processing {len(raw_pages)} pages for {company}")

    all_chunks: list[EvidenceChunk] = []

    for page in raw_pages:
        content = page.get("content", "")
        if not content or len(content) < 100:
            continue

        url = page.get("url", "")
        domain = page.get("domain", "")
        source_type = page.get("source_type", "unknown")
        published_date = page.get("published_date", "")
        authority = page.get("authority_score", 0.5)
        category = page.get("source_category", "UNKNOWN")
        page_type = page.get("page_type", category)
        page_authority = page.get("page_authority_score", 1.0)

        # Split into chunks
        text_chunks = _chunk_text(content)

        for i, chunk in enumerate(text_chunks):
            relevance = _score_relevance(chunk, company)
            score = combined_score(relevance, authority) * page_authority
            if score < 0.05:
                continue
            all_chunks.append(EvidenceChunk(
                url=url,
                domain=domain,
                chunk=chunk.strip(),
                source_type=source_type,
                published_date=published_date,
                chunk_index=i,
                rerank_score=score,  # Initial authority-adjusted score; reranker will update this
                authority_score=authority,
                source_category=category,
                page_type=page_type,
                page_authority_score=page_authority,
            ))

    # Sort by initial relevance, keep top 80 for reranker
    all_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_chunks = all_chunks[:80]

    elapsed = round(time.time() - t0, 2)
    
    # Save chunks to db
    trace_evidence_chunks(state.get("job_id", ""), top_chunks)
    
    logger.info(f"[ContentCleaner] Produced {len(top_chunks)} chunks from {len(raw_pages)} pages in {elapsed}s")

    return {
        "evidence_chunks": top_chunks,
        "status": "cleaning_done",
        "progress_pct": 50,
        "node_timings": {"content_cleaner": elapsed},
        "log": [f"[ContentCleaner] {len(top_chunks)} evidence chunks from {len(raw_pages)} pages"],
    }
