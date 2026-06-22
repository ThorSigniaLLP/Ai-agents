"""Discover and classify pages on the resolved official website."""
from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from core.state import ResearchState, URLCandidate

logger = logging.getLogger(__name__)

PAGE_AUTHORITY_SCORES = {
    "HOME": 1.0,
    "ABOUT": 1.0,
    "SERVICES": 1.0,
    "SOLUTIONS": 1.0,
    "CASE_STUDIES": 0.95,
    "TEAM": 0.90,
    "BLOG": 0.40,
    "CONTACT": 0.30,
    "PRIVACY": 0.05,
    "REFUND": 0.01,
    "TERMS": 0.01,
    "COOKIE": 0.01,
    "SHIPPING": 0.01,
    "OTHER": 0.20,
}


def classify_page(url: str, website: str = "") -> str:
    path = urlparse(url).path.lower().strip("/")
    if not path or url.rstrip("/") == website.rstrip("/"):
        return "HOME"
    rules = [
        ("CASE_STUDIES", ("case-stud", "customers", "success-stor")),
        ("SERVICES", ("service", "capabilit")),
        ("SOLUTIONS", ("solution",)),
        ("ABOUT", ("about", "company", "who-we-are")),
        ("TEAM", ("team", "leadership", "people", "founder")),
        ("BLOG", ("blog", "insight", "article", "resources")),
        ("CONTACT", ("contact", "location")),
        ("PRIVACY", ("privacy",)),
        ("REFUND", ("refund", "return")),
        ("TERMS", ("terms", "conditions")),
        ("COOKIE", ("cookie",)),
        ("SHIPPING", ("shipping", "delivery")),
    ]
    for page_type, tokens in rules:
        if any(token in path for token in tokens):
            return page_type
    return "OTHER"


def _candidate(url: str, company: str, page_type: str) -> URLCandidate:
    domain = urlparse(url).netloc.replace("www.", "")
    return URLCandidate(
        url=url,
        domain=domain,
        title=f"{company} {page_type.replace('_', ' ').title()}",
        snippet="",
        provider="website_mapper",
        rank=0,
        domain_score=PAGE_AUTHORITY_SCORES[page_type],
        authority_score=1.0,
        source_category="OFFICIAL_WEBSITE",
        page_type=page_type,
        page_authority_score=PAGE_AUTHORITY_SCORES[page_type],
    )


def run_website_mapper(state: ResearchState) -> dict[str, Any]:
    started = time.time()
    company = state["company"]
    website = state.get("company_profile", {}).get("website", "")
    existing = state.get("url_candidates", [])
    if not website:
        return {
            "url_candidates": existing,
            "status": "website_mapping_skipped",
            "progress_pct": 28,
            "node_timings": {"website_mapper": 0},
            "log": ["[WebsiteMapper] No verified website candidate available"],
        }

    discovered = {website.rstrip("/"): _candidate(website.rstrip("/"), company, "HOME")}
    official_domain = urlparse(website).netloc.replace("www.", "")
    try:
        response = httpx.get(website, follow_redirects=True, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        for anchor in soup.select("a[href]"):
            url = urljoin(str(response.url), anchor.get("href", "")).split("#", 1)[0].rstrip("/")
            if not url.startswith(("http://", "https://")):
                continue
            if urlparse(url).netloc.replace("www.", "") != official_domain:
                continue
            page_type = classify_page(url, website)
            if page_type == "OTHER":
                continue
            discovered.setdefault(url, _candidate(url, company, page_type))
    except Exception as exc:
        logger.warning("[WebsiteMapper] Homepage discovery failed for %s: %s", website, exc)

    merged = {item["url"].rstrip("/"): item for item in existing}
    merged.update(discovered)
    urls = sorted(
        merged.values(),
        key=lambda item: item.get("authority_score", 0.5) * item.get("page_authority_score", 1.0),
        reverse=True,
    )
    return {
        "url_candidates": urls,
        "status": "website_mapped",
        "progress_pct": 28,
        "node_timings": {"website_mapper": round(time.time() - started, 2)},
        "log": [f"[WebsiteMapper] Classified {len(discovered)} official pages"],
    }
