"""
diagnose_db.py
Diagnose what was stored in the DB for the last research job.
Pinpoints whether the issue is: crawler, entity validation, LLM extraction.

Usage:
    python diagnose_db.py                          # Last job
    python diagnose_db.py "supai infotech"         # Latest job for this company
"""
import asyncio
import sys
import textwrap
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()


async def diagnose(company_filter: str = ""):
    from core.database import init_db
    from core.models import (
        ResearchJob, FetchedPage, EvidenceChunkModel,
        ExtractedItem, PipelineError
    )
    from tortoise import Tortoise

    await init_db()

    # ── Find the latest job ──────────────────────────────────────────────────
    q = ResearchJob.all().order_by("-created_at")
    if company_filter:
        q = q.filter(company_name__icontains=company_filter)
    job = await q.first()

    if not job:
        print(f"❌ No job found{f' for \"{company_filter}\"' if company_filter else ''}")
        await Tortoise.close_connections()
        return

    print(f"\n{'='*65}")
    print(f"  JOB: {job.job_id[:16]}... | Company: {job.company_name}")
    print(f"  Status: {job.status} | Created: {job.created_at}")
    print(f"{'='*65}\n")

    # -- STAGE 1: Fetched Pages -----------------------------------------------
    pages = await FetchedPage.filter(job=job).all()
    print(f"-- STAGE 1: CRAWLER -- Fetched Pages ({len(pages)} total) --")
    if not pages:
        print("  [!] NO PAGES STORED -- DB writes are failing (asyncio event loop issue?)")
    else:
        empty_pages = [p for p in pages if not p.content_text or len(p.content_text) < 100]
        good_pages = [p for p in pages if p.content_text and len(p.content_text) >= 100]
        print(f"  [OK] Good pages (>100 chars content): {len(good_pages)}")
        print(f"  [!!] Empty/failed pages:              {len(empty_pages)}")
        print("\n  Top 5 pages by content length:")
        sorted_pages = sorted(good_pages, key=lambda p: len(p.content_text or ""), reverse=True)[:5]
        for p in sorted_pages:
            print(f"    [{len(p.content_text or ''):>6} chars] {p.url[:80]}")
        if empty_pages:
            print("\n  [!] Empty/failed page URLs:")
            for p in empty_pages[:5]:
                print(f"    {p.url[:80]}")
    print()

    # -- STAGE 2: Evidence Chunks ---------------------------------------------
    chunks = await EvidenceChunkModel.filter(job=job).all()
    print(f"-- STAGE 2: CONTENT CLEANER -- Evidence Chunks ({len(chunks)} total) --")
    if not chunks:
        print("  [!] NO CHUNKS STORED -- Entity validation may be too strict, or DB issue")
    else:
        scores = [c.rerank_score or 0.0 for c in chunks]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        high = sum(1 for s in scores if s >= 0.5)
        med = sum(1 for s in scores if 0.2 <= s < 0.5)
        low = sum(1 for s in scores if s < 0.2)
        print(f"  Average rerank score: {avg_score:.3f}")
        print(f"  High (>=0.5): {high}  |  Medium (0.2-0.5): {med}  |  Low (<0.2): {low}")
        print(f"\n  Top 5 chunks (most relevant):")
        sorted_chunks = sorted(chunks, key=lambda c: c.rerank_score or 0, reverse=True)[:5]
        for c in sorted_chunks:
            preview = (c.chunk_text or "")[:120].replace("\n", " ")
            print(f"    [score={c.rerank_score:.3f}] {c.url[:50]} | {preview}...")

        # Check for pain-point relevant chunks
        pp_chunks = [c for c in chunks if any(
            kw in (c.chunk_text or "").lower()
            for kw in ["complaint", "problem", "issue", "review", "poor", "slow", "difficult", "disappointing"]
        )]
        print(f"\n  Pain-point relevant chunks: {len(pp_chunks)}")
        if pp_chunks:
            for c in pp_chunks[:3]:
                preview = (c.chunk_text or "")[:150].replace("\n", " ")
                print(f"    {preview}...")
    print()

    # -- STAGE 3: Extracted Items ---------------------------------------------
    items = await ExtractedItem.filter(job=job).all()
    print(f"-- STAGE 3: LLM EXTRACTION -- Extracted Items ({len(items)} total) --")
    if not items:
        print("  [!] NO ITEMS STORED -- LLM extraction failed or DB issue")
    else:
        by_field = defaultdict(list)
        for it in items:
            by_field[it.field].append(it.value)

        print(f"  Fields extracted: {list(by_field.keys())}")
        print(f"\n  Details per field:")
        for field, values in sorted(by_field.items()):
            print(f"\n  [{field}] ({len(values)} items)")
            for v in values[:3]:
                val_preview = str(v)[:120]
                print(f"    -> {val_preview}")
            if len(values) > 3:
                print(f"    ... and {len(values)-3} more")
    print()

    # -- STAGE 4: Pipeline Errors ---------------------------------------------
    errors = await PipelineError.filter(job=job).all()
    print(f"-- STAGE 4: PIPELINE ERRORS ({len(errors)}) --")
    if not errors:
        print("  [OK] No pipeline errors recorded")
    else:
        for e in errors:
            print(f"  [ERR] [{e.node_name}] {e.error_message[:120]}")
    print()

    # -- DIAGNOSIS CONCLUSION -------------------------------------------------
    print("-- DIAGNOSIS --")
    if not pages:
        print("  [RED] ROOT CAUSE: DB writes failing -- no data stored from any stage")
    elif len(good_pages) < 5:
        print(f"  [RED] ROOT CAUSE: Crawler only got {len(good_pages)} usable pages -- scraping is failing")
        print("       Check: anti-bot blocking, timeouts, trafilatura discards")
    elif not chunks:
        print("  [RED] ROOT CAUSE: Content cleaner produced 0 chunks -- entity validation too strict")
        print("       Check: _validate_page_entity() and _build_entity_aliases()")
    elif len(chunks) < 5:
        print(f"  [YEL] PARTIAL: Only {len(chunks)} chunks -- validation filtering too aggressively")
    elif not items:
        print("  [RED] ROOT CAUSE: LLM extraction returned 0 items -- TOML parsing or token issue")
    elif len(by_field) < 4:
        print(f"  [YEL] PARTIAL: LLM only extracted {len(by_field)} fields -- increase max_tokens or check prompts")
    else:
        print(f"  [OK] Pipeline looks healthy -- {len(pages)} pages -> {len(chunks)} chunks -> {len(items)} evidence items")

    await Tortoise.close_connections()


if __name__ == "__main__":
    company = sys.argv[1] if len(sys.argv) > 1 else ""
    asyncio.run(diagnose(company))
