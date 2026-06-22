"""
Source authority scoring for company intelligence research.

The pipeline treats relevance and authority separately, then combines them when
ranking evidence. This prevents weak context such as personal social posts from
overpowering company-specific records and reviews.
"""
from __future__ import annotations

from urllib.parse import urlparse


AUTHORITY_SCORES = {
    "OFFICIAL_WEBSITE": 1.0,
    "MCA": 0.98,
    "ZAUBACORP": 0.98,
    "TOFLER": 0.95,
    "TRACXN": 0.90,
    "LINKEDIN_COMPANY_PAGE": 0.90,
    "CRUNCHBASE": 0.90,
    "GLASSDOOR": 0.80,
    "AMBITIONBOX": 0.80,
    "TRUSTPILOT": 0.75,
    "G2": 0.75,
    "CLUTCH": 0.75,
    "NEWS": 0.70,
    "TECH_STACK": 0.70,
    "BLOGS": 0.10,
    "INDIVIDUAL_LINKEDIN_POSTS": 0.10,
    "RANDOM_SOCIAL_MEDIA": 0.05,
    "UNKNOWN": 0.50,
}

REVIEW_CATEGORIES = {"GLASSDOOR", "AMBITIONBOX", "TRUSTPILOT", "G2", "CLUTCH"}
REGISTRY_CATEGORIES = {"MCA", "ZAUBACORP", "TOFLER", "TRACXN"}
WEAK_CONTEXT_CATEGORIES = {"INDIVIDUAL_LINKEDIN_POSTS", "RANDOM_SOCIAL_MEDIA", "BLOGS"}


def normalize_domain(url_or_domain: str) -> str:
    """Return a lowercase domain without the www prefix."""
    if not url_or_domain:
        return ""
    parsed = urlparse(url_or_domain if "://" in url_or_domain else f"https://{url_or_domain}")
    return parsed.netloc.lower().replace("www.", "")


def classify_source(url: str, company_website: str = "") -> str:
    """Classify a URL into the authority hierarchy."""
    domain = normalize_domain(url)
    path = urlparse(url).path.lower()
    company_domain = normalize_domain(company_website)

    if company_domain and domain == company_domain:
        return "OFFICIAL_WEBSITE"
    if "mca.gov.in" in domain:
        return "MCA"
    if "zaubacorp.com" in domain:
        return "ZAUBACORP"
    if "tofler.in" in domain:
        return "TOFLER"
    if "tracxn.com" in domain:
        return "TRACXN"
    if "linkedin.com" in domain:
        if path.startswith("/company/") or "/company/" in path:
            return "LINKEDIN_COMPANY_PAGE"
        return "INDIVIDUAL_LINKEDIN_POSTS"
    if "crunchbase.com" in domain:
        return "CRUNCHBASE"
    if "glassdoor." in domain:
        return "GLASSDOOR"
    if "ambitionbox.com" in domain:
        return "AMBITIONBOX"
    if "trustpilot.com" in domain:
        return "TRUSTPILOT"
    if "g2.com" in domain:
        return "G2"
    if "clutch.co" in domain:
        return "CLUTCH"
    if "builtwith.com" in domain or "wappalyzer.com" in domain:
        return "TECH_STACK"
    if any(d in domain for d in ("twitter.com", "x.com", "facebook.com", "instagram.com", "tiktok.com", "reddit.com")):
        return "RANDOM_SOCIAL_MEDIA"
    if any(d in domain for d in ("reuters.com", "bloomberg.com", "techcrunch.com", "businesswire.com", "prnewswire.com", "forbes.com", "wsj.com", "ft.com")):
        return "NEWS"
    if "blog" in domain or "/blog" in path or "medium.com" in domain:
        return "BLOGS"
    return "UNKNOWN"


def authority_score(url: str, company_website: str = "") -> float:
    return AUTHORITY_SCORES.get(classify_source(url, company_website), AUTHORITY_SCORES["UNKNOWN"])


def combined_score(relevance_score: float, authority: float) -> float:
    """Reranking rule: final score = relevance * authority."""
    return round(max(0.0, min(1.0, relevance_score)) * max(0.0, min(1.0, authority)), 4)

