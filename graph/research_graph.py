"""Grounded company identity graph.

Flow:
  planner -> company_resolver -> specialized_workers -> multi_search -> url_graph
  -> website_mapper -> smart_fetcher -> content_cleaner -> reranker
  -> specialized_extractors -> profile_validator -> assemble_output
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.company_resolver import run_company_resolver
from agents.content_cleaner import run_content_cleaner
from agents.multi_search import run_multi_search
from agents.planner import run_planner
from agents.profile_validator import run_profile_validator
from agents.reranker import run_reranker
from agents.smart_fetcher import run_smart_fetcher
from agents.specialized_extractors import run_specialized_extractors
from agents.specialized_workers import run_specialized_workers
from agents.url_graph import run_url_graph_builder
from agents.website_mapper import run_website_mapper
from agents.business_analyst import run_business_analyst
from core.state import ResearchState
from core.db_tracer import trace_job_complete

logger = logging.getLogger(__name__)


def assemble_final_output(state: ResearchState) -> dict[str, Any]:
    """Return exactly the validated CompanyProfile and no analysis fields."""
    started = time.time()
    profile = state.get("company_profile", {})
    
    # Inject the deep analysis report into the profile
    analysis = state.get("business_analysis", {})
    if "deep_analysis_report" in analysis:
        profile["deep_analysis_report"] = analysis["deep_analysis_report"]
    
    trace_job_complete(state.get("job_id", ""), "completed")
    
    logger.info("[Graph] Company understanding completed for %s", state["company"])
    return {
        "final_output": profile,
        "status": "done",
        "progress_pct": 100,
        "node_timings": {"assemble_output": round(time.time() - started, 2)},
        "log": [f"[Graph] Validated company profile completed for {state['company']}"],
    }


def build_research_graph(use_memory_checkpointer: bool = True):
    builder = StateGraph(ResearchState)
    builder.add_node("planner", run_planner)
    builder.add_node("company_resolver", run_company_resolver)
    builder.add_node("specialized_workers", run_specialized_workers)
    builder.add_node("multi_search", run_multi_search)
    builder.add_node("url_graph", run_url_graph_builder)
    builder.add_node("website_mapper", run_website_mapper)
    builder.add_node("smart_fetcher", run_smart_fetcher)
    builder.add_node("content_cleaner", run_content_cleaner)
    builder.add_node("reranker", run_reranker)
    builder.add_node("specialized_extractors", run_specialized_extractors)
    builder.add_node("profile_validator", run_profile_validator)
    builder.add_node("business_analyst", run_business_analyst)
    builder.add_node("assemble_output", assemble_final_output)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "company_resolver")
    builder.add_edge("company_resolver", "specialized_workers")
    builder.add_edge("specialized_workers", "multi_search")
    builder.add_edge("multi_search", "url_graph")
    builder.add_edge("url_graph", "website_mapper")
    builder.add_edge("website_mapper", "smart_fetcher")
    builder.add_edge("smart_fetcher", "content_cleaner")
    builder.add_edge("content_cleaner", "reranker")
    builder.add_edge("reranker", "specialized_extractors")

    # ── Parallel branch: profile_validator + business_analyst run simultaneously ──
    builder.add_edge("specialized_extractors", "profile_validator")
    builder.add_edge("specialized_extractors", "business_analyst")
    # Both branches converge at assemble_output (LangGraph waits for both)
    builder.add_edge("profile_validator", "assemble_output")
    builder.add_edge("business_analyst", "assemble_output")
    builder.add_edge("assemble_output", END)

    if use_memory_checkpointer:
        return builder.compile(checkpointer=MemorySaver())
    return builder.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_research_graph()
    return _graph


def initial_research_state(company: str, card_info: dict, job_id: str | None = None) -> ResearchState:
    return ResearchState(
        company=company,
        card_info=card_info,
        job_id=job_id or str(uuid.uuid4()),
        sub_questions=[],
        search_queries=[],
        company_profile={
            "company_name": company,
            "website": "",
            "linkedin_company_page": "",
            "headquarters": "UNKNOWN",
            "industry": "UNKNOWN",
            "overview": "UNKNOWN",
            "founders": [],
            "services": [],
            "technologies": [],
            "employee_count": "UNKNOWN",
            "legal_entity": "UNKNOWN",
            # Extended B2B intelligence fields
            "founded_year": "UNKNOWN",
            "revenue": "UNKNOWN",
            "competitors": [],
            "pain_points": [],
            "growth_signals": {
                "job_postings": [],
                "recent_news": [],
                "expansion_signals": [],
                "hiring_trend": "UNKNOWN",
            },
            "disconnection_signals": [],
            "tech_stack": {
                "crm": "UNKNOWN",
                "erp": "UNKNOWN",
                "marketing_tools": [],
                "development_stack": [],
                "cloud_provider": "UNKNOWN",
            },
            "pitch_opportunities": [],
        },
        worker_queries={},
        url_candidates=[],
        raw_pages=[],
        evidence_chunks=[],
        evidence_items=[],
        iteration_count=1,
        retry_count=0,
        evidence_sufficient=False,
        coverage_gaps=[],
        extracted_facts={},
        extraction_errors=[],
        verified_facts={},
        rejected_facts=[],
        business_analysis={},
        final_output={},
        sources_used=[],
        node_timings={},
        errors=[],
        status="starting",
        progress_pct=0,
        log=[],
    )


def run_research_sync(company: str, card_info: dict) -> dict:
    state = initial_research_state(company, card_info)
    config = {"configurable": {"thread_id": state["job_id"]}}
    result = get_graph().invoke(state, config=config)
    return result.get("final_output", {})
