"""
test_research.py
Quick end-to-end test script.

Usage:
  python test_research.py

Or test a specific company:
  python test_research.py "Salesforce"
"""

__test__ = False  # Manual live-network script; exclude from pytest collection.
import sys
import json
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

def test_sync(company: str = "Salesforce"):
    """Test the full pipeline synchronously."""
    from graph.research_graph import run_research_sync

    print(f"\n{'='*60}")
    print(f"Researching: {company}")
    print(f"{'='*60}\n")

    card_info = {
        "company": company,
        "name": "John Smith",
        "job_title": "VP of Sales",
        "email": "john@example.com",
        "website": "",
    }

    result = run_research_sync(company, card_info)

    print("\nRESEARCH RESULT:")
    print(json.dumps(result, indent=2, default=str))

    expected = {
        "company_name", "website", "linkedin_company_page", "headquarters",
        "industry", "founders", "services", "technologies",
        "employee_count", "legal_entity",
    }
    print(f"\nExact profile schema: {set(result) == expected}")



def test_planner_only(company: str = "Salesforce"):
    """Quick test of just the planner."""
    from agents.planner import run_planner

    state = {
        "company": company,
        "card_info": {"company": company},
        "job_id": "test-001",
        "sub_questions": [],
        "search_queries": [],
        "raw_pages": [],
        "evidence_chunks": [],
        "iteration_count": 0,
        "evidence_sufficient": False,
        "coverage_gaps": [],
        "model_extractions": {},
        "extraction_errors": [],
        "verified_facts": {},
        "disputed_facts": [],
        "judged_facts": {},
        "business_analysis": {},
        "sales_strategy": {},
        "pitch": {},
        "final_output": {},
        "sources_used": [],
        "errors": [],
        "status": "starting",
        "progress_pct": 0,
        "log": [],
    }
    result = run_planner(state)
    print("Sub-questions:", result["sub_questions"])
    print("Search queries:", result["search_queries"])


if __name__ == "__main__":
    company = sys.argv[1] if len(sys.argv) > 1 else "Salesforce"
    mode = sys.argv[2] if len(sys.argv) > 2 else "full"

    if mode == "planner":
        test_planner_only(company)
    else:
        test_sync(company)
