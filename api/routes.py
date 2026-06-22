"""
api/routes.py
FastAPI route definitions.

Endpoints:
  POST /research          — Start research job (async background task)
  GET  /research/{job_id} — Poll job status / get result
  GET  /research/{job_id}/stream — SSE streaming progress
  GET  /health            — Health check
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import (
    ResearchRequest, ResearchJobResponse,
    ResearchResultResponse, HealthResponse,
)
from core.state import ResearchState

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory job store (replace with Redis/DB for production)
_jobs: dict[str, dict] = {}


# ── Background task ───────────────────────────────────────────────────────────

def _run_research_job(job_id: str, company: str, card_info: dict):
    """Background task: runs the full LangGraph research pipeline."""
    try:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["progress_pct"] = 5

        from graph.research_graph import get_graph
        from core.state import ResearchState

        graph = get_graph()

        from graph.research_graph import initial_research_state
        initial_state = initial_research_state(company, card_info, job_id)

        config = {"configurable": {"thread_id": job_id}}

        # Stream intermediate state updates
        for chunk in graph.stream(initial_state, config=config, stream_mode="values"):
            progress = chunk.get("progress_pct", 0)
            status = chunk.get("status", "running")
            log_entries = chunk.get("log", [])

            _jobs[job_id]["progress_pct"] = progress
            _jobs[job_id]["status_detail"] = status
            if log_entries:
                _jobs[job_id].setdefault("logs", []).extend(log_entries)

        # Final result
        from graph.research_graph import get_graph as _get_graph
        final_state = graph.get_state(config)
        final_output = final_state.values.get("final_output", {})

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["progress_pct"] = 100
        _jobs[job_id]["result"] = final_output

    except Exception as e:
        logger.error(f"[Routes] Research job {job_id} failed: {e}", exc_info=True)
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/research", response_model=ResearchJobResponse, status_code=202)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Start an autonomous company research job.
    Returns immediately with a job_id. Poll /research/{job_id} for results.
    """
    job_id = str(uuid.uuid4())
    company = request.company.strip()
    card_info = request.card_info.model_dump() if request.card_info else {"company": company}

    if not company:
        raise HTTPException(status_code=400, detail="Company name is required")

    _jobs[job_id] = {
        "job_id": job_id,
        "company": company,
        "status": "queued",
        "progress_pct": 0,
        "result": None,
        "error": None,
        "logs": [],
    }

    background_tasks.add_task(_run_research_job, job_id, company, card_info)

    logger.info(f"[Routes] Research job queued: {job_id} for {company}")

    return ResearchJobResponse(
        job_id=job_id,
        company=company,
        status="queued",
        message=f"Research started for {company}. Poll /research/{job_id} for results.",
    )


@router.get("/research/{job_id}", response_model=ResearchResultResponse)
async def get_research_result(job_id: str):
    """Poll research job status and result."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return ResearchResultResponse(
        job_id=job_id,
        company=job.get("company", ""),
        status=job.get("status", "unknown"),
        progress_pct=job.get("progress_pct", 0),
        result=job.get("result"),
        error=job.get("error"),
    )


@router.get("/research/{job_id}/stream")
async def stream_research_progress(job_id: str):
    """
    Server-Sent Events (SSE) endpoint for real-time research progress.
    Connect and receive live updates as the research progresses.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        last_log_count = 0
        while True:
            job = _jobs.get(job_id, {})
            status = job.get("status", "unknown")
            progress = job.get("progress_pct", 0)
            logs = job.get("logs", [])
            new_logs = logs[last_log_count:]
            last_log_count = len(logs)

            event_data = json.dumps({
                "job_id": job_id,
                "status": status,
                "progress_pct": progress,
                "log": new_logs,
                "result": job.get("result") if status == "completed" else None,
                "error": job.get("error"),
            })
            yield f"data: {event_data}\n\n"

            if status in ("completed", "failed"):
                break

            await asyncio.sleep(1.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        from graph.research_graph import get_graph
        get_graph()
        graph_ready = True
    except Exception:
        graph_ready = False

    return HealthResponse(
        status="ok",
        version="1.0.0",
        graph_ready=graph_ready,
    )


@router.get("/research/{job_id}/logs")
async def get_research_logs(job_id: str):
    """Get research log entries for a job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"job_id": job_id, "logs": job.get("logs", [])}
