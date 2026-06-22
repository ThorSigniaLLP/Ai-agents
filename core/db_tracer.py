import logging
import asyncio
from datetime import datetime, timezone
from core.database import init_db, close_db
from core.models import (
    ResearchJob, NodeLog, SearchCandidate, FetchedPage, 
    EvidenceChunkModel, ExtractedItem, PipelineError
)

logger = logging.getLogger(__name__)

async def _trace_job_start(job_id: str, company: str):
    await init_db()
    try:
        await ResearchJob.create(job_id=job_id, company_name=company, status="running")
    except Exception as e:
        logger.error(f"[Tracer] Error saving job: {e}")
    finally:
        await close_db()

def trace_job_start(job_id: str, company: str):
    asyncio.run(_trace_job_start(job_id, company))

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
    asyncio.run(_trace_node_timing(job_id, node_name, duration))

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
    asyncio.run(_trace_search_candidates(job_id, candidates))

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
    asyncio.run(_trace_fetched_pages(job_id, pages))

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
    asyncio.run(_trace_evidence_chunks(job_id, chunks))

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
    asyncio.run(_trace_extracted_items(job_id, items))

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
    asyncio.run(_trace_error(job_id, node, err))

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
    asyncio.run(_trace_job_complete(job_id, status))
