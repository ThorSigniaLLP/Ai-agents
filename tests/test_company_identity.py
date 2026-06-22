"""Focused tests for authority, page mapping, and profile validation gates."""
from agents.profile_validator import run_profile_validator
from agents.source_authority import authority_score, classify_source
from agents.website_mapper import PAGE_AUTHORITY_SCORES, classify_page


def _item(field, value, url, page_type, authority=1.0, relevance=0.9):
    return {
        "field": field,
        "value": value,
        "source_url": url,
        "source_domain": url.split("/")[2],
        "page_type": page_type,
        "authority_score": authority,
        "relevance_score": relevance,
        "publication_date": "",
    }


def _state(items):
    return {
        "company": "Acme",
        "company_profile": {
            "company_name": "Acme",
            "website": "https://acme.example",
            "linkedin_company_page": "https://linkedin.com/company/acme",
        },
        "url_candidates": [
            {"url": "https://acme.example", "source_category": "OFFICIAL_WEBSITE"},
            {"url": "https://linkedin.com/company/acme", "source_category": "LINKEDIN_COMPANY_PAGE"},
        ],
        "evidence_items": items,
    }


def test_source_authority_bans_personal_posts():
    assert authority_score("https://linkedin.com/posts/person_abc") == 0.1
    assert authority_score("https://medium.com/@person/opinion") == 0.1
    assert authority_score("https://twitter.com/person/status/1") == 0.05
    assert classify_source("https://linkedin.com/company/acme") == "LINKEDIN_COMPANY_PAGE"


def test_page_authority_and_classification():
    assert classify_page("https://acme.example/about", "https://acme.example") == "ABOUT"
    assert classify_page("https://acme.example/refund-policy", "https://acme.example") == "REFUND"
    assert PAGE_AUTHORITY_SCORES["ABOUT"] == 1.0
    assert PAGE_AUTHORITY_SCORES["REFUND"] == 0.01


def test_industry_and_headquarters_require_two_sources():
    result = run_profile_validator(_state([
        _item("industry", "Software", "https://acme.example/about", "ABOUT"),
        _item("headquarters", "Pune, India", "https://acme.example/about", "ABOUT"),
    ]))["company_profile"]
    assert result["industry"] == "UNKNOWN"
    assert result["headquarters"] == "UNKNOWN"


def test_field_specific_validation_accepts_grounded_values():
    result = run_profile_validator(_state([
        _item("industry", "Software", "https://acme.example/about", "ABOUT"),
        _item("industry", "Software", "https://linkedin.com/company/acme", "LINKEDIN_COMPANY_PAGE", 0.9),
        _item("headquarters", "Pune, India", "https://acme.example/about", "ABOUT"),
        _item("headquarters", "Pune, India", "https://zaubacorp.com/acme", "ZAUBACORP", 0.98),
        _item("services", "Cloud consulting", "https://acme.example/services", "SERVICES"),
        _item("technologies", "HubSpot", "https://builtwith.com/acme.example", "TECH_STACK", 0.8),
        _item("employee_count", "51-200", "https://linkedin.com/company/acme", "LINKEDIN_COMPANY_PAGE", 0.9),
        _item("legal_entity", "Acme Private Limited", "https://zaubacorp.com/acme", "ZAUBACORP", 0.98),
    ]))["company_profile"]
    assert result["industry"] == "Software"
    assert result["headquarters"] == "Pune, India"
    assert result["services"] == ["Cloud consulting"]
    assert result["technologies"] == ["HubSpot"]
    assert result["employee_count"] == "51-200"
    assert result["legal_entity"] == "Acme Private Limited"


def test_low_authority_pages_cannot_supply_services():
    result = run_profile_validator(_state([
        _item("services", "Imaginary service", "https://acme.example/refund", "REFUND"),
    ]))["company_profile"]
    assert result["services"] == []
