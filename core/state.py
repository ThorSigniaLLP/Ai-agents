"""Typed state for the company identity research graph."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from typing_extensions import NotRequired, TypedDict


def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts — used as reducer for node_timings in parallel branches."""
    return {**a, **b}


def _last_value(a: Any, b: Any) -> Any:
    """Keep the last-written value — used for 'status' written by parallel branches."""
    return b


def _max_value(a: int | float, b: int | float) -> int | float:
    """Keep the higher value — used for 'progress_pct' written by parallel branches."""
    return max(a, b)


class URLCandidate(TypedDict):
    url: str
    domain: str
    title: str
    snippet: str
    provider: str
    rank: int
    domain_score: float
    authority_score: NotRequired[float]
    source_category: NotRequired[str]
    page_type: NotRequired[str]
    page_authority_score: NotRequired[float]


class PageResult(TypedDict):
    url: str
    title: str
    content: str
    domain: str
    source_type: str
    published_date: str
    fetch_method: str
    timestamp: str
    relevance_score: float
    authority_score: NotRequired[float]
    source_category: NotRequired[str]
    page_type: NotRequired[str]
    page_authority_score: NotRequired[float]


class EvidenceChunk(TypedDict):
    url: str
    domain: str
    chunk: str
    source_type: str
    published_date: str
    chunk_index: int
    rerank_score: float
    authority_score: NotRequired[float]
    source_category: NotRequired[str]
    page_type: NotRequired[str]
    page_authority_score: NotRequired[float]


class EvidenceItem(TypedDict):
    field: str
    value: Any
    source_url: str
    source_domain: str
    page_type: str
    authority_score: float
    relevance_score: float
    publication_date: str


class CompanyProfile(TypedDict):
    company_name: str
    website: str
    linkedin_company_page: str
    headquarters: str
    industry: str
    overview: NotRequired[str]
    founders: list[str]
    services: list[str]
    technologies: list[str]
    employee_count: str
    legal_entity: str
    # Extended B2B intelligence fields
    founded_year: NotRequired[str]
    revenue: NotRequired[str]
    competitors: NotRequired[list[str]]
    pain_points: NotRequired[list[dict]]        # [{issue, severity, source, quote, frequency}]
    growth_signals: NotRequired[dict]           # {job_postings, recent_news, expansion_signals, hiring_trend}
    disconnection_signals: NotRequired[list[str]]
    tech_stack: NotRequired[dict]               # {crm, erp, marketing_tools, development_stack, cloud_provider}
    pitch_opportunities: NotRequired[list[str]]
    deep_analysis_report: NotRequired[str]
    country: NotRequired[str]
    aliases: NotRequired[list[str]]


class VerifiedFact(TypedDict):
    value: Any
    sources: list[str]
    confidence: float
    verification_status: str
    note: Optional[str]


class ResearchState(TypedDict):
    company: str
    card_info: dict
    job_id: str
    sub_questions: list[str]
    search_queries: list[str]
    company_profile: CompanyProfile
    worker_queries: dict[str, list[str]]
    url_candidates: list[URLCandidate]
    raw_pages: list[PageResult]
    evidence_chunks: list[EvidenceChunk]
    evidence_items: list[EvidenceItem]
    iteration_count: int
    retry_count: int
    evidence_sufficient: bool
    coverage_gaps: list[str]
    extracted_facts: dict[str, Any]
    extraction_errors: Annotated[list[str], operator.add]
    verified_facts: dict[str, VerifiedFact]
    rejected_facts: list[dict]
    business_analysis: dict[str, Any]
    final_output: dict
    sources_used: list[str]
    node_timings: Annotated[dict[str, float], _merge_dicts]
    errors: Annotated[list[str], operator.add]
    status: Annotated[str, _last_value]
    progress_pct: Annotated[int, _max_value]
    log: Annotated[list[str], operator.add]
