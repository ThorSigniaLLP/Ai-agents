"""
api/schemas.py
FastAPI request/response Pydantic models.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class CardInfo(BaseModel):
    name: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    email: Optional[str] = None
    mobile: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    notes: Optional[str] = None


class ResearchRequest(BaseModel):
    company: str = Field(..., description="Company name to research")
    card_info: Optional[CardInfo] = Field(default=None, description="Business card details")
    options: Optional[dict] = Field(default=None, description="Optional overrides (max_iterations, etc.)")


class ResearchJobResponse(BaseModel):
    job_id: str
    company: str
    status: str
    message: str


class ResearchResultResponse(BaseModel):
    job_id: str
    company: str
    status: str
    progress_pct: int = 0
    result: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    graph_ready: bool
