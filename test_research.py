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

def test_sync(company: str = "Salesforce", website: str = "", email: str = "", address: str = ""):
    """Test the full pipeline synchronously."""
    from graph.research_graph import run_research_sync

    print(f"\n{'='*60}")
    print(f"Researching: {company}")
    if website:
        print(f"Website:     {website}")
    if email:
        print(f"Email:       {email}")
    if address:
        print(f"Address:     {address}")
    print(f"{'='*60}\n")

    card_info = {
        "company": company,
        "name": "",
        "job_title": "",
        "email": email,
        "website": website,
        "address": address,
    }

    result = run_research_sync(company, card_info)

    print("\nRESEARCH RESULT:")
    result_json = json.dumps(result, indent=2, default=str)
    print(result_json)
    
    with open("report.json", "w", encoding="utf-8") as f:
        f.write(result_json)
    print("\n[+] Full report saved to report.json in the current directory!")

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
    import argparse
    parser = argparse.ArgumentParser(description="Test the research pipeline")
    parser.add_argument("company", nargs="?", default="Salesforce", help="Company name to research")
    parser.add_argument("--website", "-w", default="", help="Known company website (e.g. gvhcol.com)")
    parser.add_argument("--email", "-e", default="", help="Contact email (used to derive domain)")
    parser.add_argument("--address", "-a", default="", help="Company address")
    parser.add_argument("--mode", "-m", default="full", choices=["full", "planner"], help="Test mode")
    args = parser.parse_args()

    if args.mode == "planner":
        test_planner_only(args.company)
    else:
        test_sync(args.company, website=args.website, email=args.email, address=args.address)

