"""
core/db_tracer.py
Async-safe database tracer.

Works both:
- Outside uvicorn (test_research.py) — uses asyncio.run()
- Inside uvicorn event loop — uses nest_asyncio or thread-based execution
"""
import logging
import asyncio
import threading
from datetime import datetime, timezone
from core.database import init_db, close_db
from core.models import (
    ResearchJob, NodeLog, SearchCandidate, FetchedPage, 
    EvidenceChunkModel, ExtractedItem, PipelineError
)

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine safely regardless of whether an event loop is running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside uvicorn/FastAPI — run in a separate thread with its own loop
        result = [None]
        exception = [None]

        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result[0] = new_loop.run_until_complete(coro)
            except Exception as e:
                exception[0] = e
            finally:
                new_loop.close()

        t = threading.Thread(target=run_in_thread, daemon=True)
        t.start()
        t.join(timeout=15)  # Max 15s wait for DB write
        if exception[0]:
            raise exception[0]
        return result[0]
    else:
        # No running loop — use asyncio.run() directly
        return asyncio.run(coro)


async def _trace_job_start(job_id: str, company: str):
    await init_db()
    try:
        await ResearchJob.create(job_id=job_id, company_name=company, status="running")
    except Exception as e:
        logger.error(f"[Tracer] Error saving job: {e}")
    finally:
        await close_db()

def trace_job_start(job_id: str, company: str):
    try:
        _run_async(_trace_job_start(job_id, company))
    except Exception as e:
        logger.warning("[Tracer] trace_job_start failed silently: %s", e)


async def _trace_node_timing(job_id: str, node_name: str, duration: float):
    await init_db()
    try:
        job = await ResearchJob.get_or_none(job_id=job_id)
        if job:
            await NodeLog.create(job=job, node_name=node_name, duration_seconds=duration)
    except Exception as e:
        logger.error(f"[Tracer] Error saving node log: {e}")
    finally:
        await close_db()

def trace_node_timing(job_id: str, node_name: str, duration: float):
    try:
        _run_async(_trace_node_timing(job_id, node_name, duration))
    except Exception as e:
        logger.warning("[Tracer] trace_node_timing failed silently: %s", e)


async def _trace_search_candidates(job_id: str, candidates: list[dict]):
    await init_db()
    try:
        job = await ResearchJob.get_or_none(job_id=job_id)
        if job:
            for c in candidates:
                await SearchCandidate.create(
                    job=job,
                    url=c.get("url"),
                    domain=c.get("domain"),
                    title=c.get("title"),
                    provider=c.get("provider", "unknown"),
                    domain_score=c.get("domain_score", 0.0)
                )
    except Exception as e:
        logger.error(f"[Tracer] Error saving candidates: {e}")
    finally:
        await close_db()

def trace_search_candidates(job_id: str, candidates: list[dict]):
    try:
        _run_async(_trace_search_candidates(job_id, candidates))
    except Exception as e:
        logger.warning("[Tracer] trace_search_candidates failed silently: %s", e)


async def _trace_fetched_pages(job_id: str, pages: list[dict]):
    await init_db()
    try:
        job = await ResearchJob.get_or_none(job_id=job_id)
        if job:
            for p in pages:
                await FetchedPage.create(
                    job=job,
                    url=p.get("url"),
                    title=p.get("title"),
                    content_text=p.get("content", "")[:50000],
                    fetch_method=p.get("fetch_method", "unknown"),
                    status_code=p.get("status_code", 200)
                )
    except Exception as e:
        logger.error(f"[Tracer] Error saving pages: {e}")
    finally:
        await close_db()

def trace_fetched_pages(job_id: str, pages: list[dict]):
    try:
        _run_async(_trace_fetched_pages(job_id, pages))
    except Exception as e:
        logger.warning("[Tracer] trace_fetched_pages failed silently: %s", e)


async def _trace_evidence_chunks(job_id: str, chunks: list[dict]):
    await init_db()
    try:
        job = await ResearchJob.get_or_none(job_id=job_id)
        if job:
            for c in chunks:
                await EvidenceChunkModel.create(
                    job=job,
                    url=c.get("url"),
                    chunk_index=c.get("chunk_index", 0),
                    chunk_text=c.get("chunk", ""),
                    rerank_score=c.get("rerank_score", 0.0)
                )
    except Exception as e:
        logger.error(f"[Tracer] Error saving chunks: {e}")
    finally:
        await close_db()

def trace_evidence_chunks(job_id: str, chunks: list[dict]):
    try:
        _run_async(_trace_evidence_chunks(job_id, chunks))
    except Exception as e:
        logger.warning("[Tracer] trace_evidence_chunks failed silently: %s", e)


async def _trace_extracted_items(job_id: str, items: list):
    await init_db()
    try:
        job = await ResearchJob.get_or_none(job_id=job_id)
        if job:
            for it in items:
                await ExtractedItem.create(
                    job=job,
                    field=it.get("field"),
                    value=str(it.get("value")),
                    source_url=it.get("source_url")
                )
    except Exception as e:
        logger.error(f"[Tracer] Error saving extracted items: {e}")
    finally:
        await close_db()

def trace_extracted_items(job_id: str, items: list):
    try:
        _run_async(_trace_extracted_items(job_id, items))
    except Exception as e:
        logger.warning("[Tracer] trace_extracted_items failed silently: %s", e)


async def _trace_error(job_id: str, node: str, err: str):
    await init_db()
    try:
        job = await ResearchJob.get_or_none(job_id=job_id)
        if job:
            await PipelineError.create(job=job, node_name=node, error_message=str(err))
    except Exception as e:
        logger.error(f"[Tracer] Error saving pipeline error: {e}")
    finally:
        await close_db()

def trace_error(job_id: str, node: str, err: str):
    try:
        _run_async(_trace_error(job_id, node, err))
    except Exception as e:
        logger.warning("[Tracer] trace_error failed silently: %s", e)


async def _trace_job_complete(job_id: str, status: str):
    await init_db()
    try:
        job = await ResearchJob.get_or_none(job_id=job_id)
        if job:
            job.status = status
            job.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
            await job.save(update_fields=["status", "completed_at"])
    except Exception as e:
        logger.error(f"[Tracer] Error completing job: {e}")
    finally:
        await close_db()

def trace_job_complete(job_id: str, status: str):
    try:
        _run_async(_trace_job_complete(job_id, status))
    except Exception as e:
        logger.warning("[Tracer] trace_job_complete failed silently: %s", e)
