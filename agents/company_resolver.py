"""Resolve the company identity surface before any extraction."""
from __future__ import annotations

import re
import time
import logging
from typing import Any
from urllib.parse import urlparse, quote_plus

import primp
from core.state import ResearchState
from core.llm_router import completion_with_fallback
from core.config import get_settings

logger = logging.getLogger(__name__)


def _clean_url(value: str) -> str:
    if not value:
        return ""
    value = value.strip()
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    parsed = urlparse(value)
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _domain_from_email(email: str) -> str:
    """Extract company domain from email address."""
    if "@" in email:
        return email.split("@")[-1].strip()
    return ""


def run_company_resolver(state: ResearchState) -> dict[str, Any]:
    """Seed identity candidates and perform 2-stage legal name discovery."""
    started = time.time()
    company = state["company"].strip()
    card = state.get("card_info", {}) or {}

    # Resolve known website — prefer explicit card_info.website, fallback to email domain
    website = _clean_url(card.get("website", ""))
    email = card.get("email", "")
    email_domain = _domain_from_email(email) if email else ""

    # If no website but have email domain, build likely website URL
    if not website and email_domain:
        website = f"https://{email_domain}"

    linkedin = card.get("linkedin", "")
    if linkedin and "/company/" not in linkedin.lower():
        linkedin = ""

    address = card.get("address", "")
    phone = card.get("mobile", "") or card.get("phone", "")

    # Build identity anchor — used to disambiguate company from similarly-named entities
    identity_hints = [company]
    if email_domain:
        identity_hints.append(email_domain)
    if address:
        # Extract city/region from address for geographic disambiguation
        city = address.split(",")[0].strip() if "," in address else address.strip()[:30]
        identity_hints.append(city)
        
    # ── Stage 1: Legal Name Pre-Discovery ──
    legal_entity = "UNKNOWN"
    if website:
        try:
            logger.info(f"[CompanyResolver] Pre-fetching {website} for legal entity discovery")
            client = primp.Client(timeout=10, impersonate="chrome_120")
            resp = client.get(website)
            if resp.status_code == 200:
                html = resp.text
                import readability
                from bs4 import BeautifulSoup
                
                doc = readability.Document(html)
                soup = BeautifulSoup(doc.summary(), "lxml")
                main_text = soup.get_text(separator=" ", strip=True)
                
                raw_soup = BeautifulSoup(html, "lxml")
                footer_text = ""
                for footer in raw_soup.find_all(['footer', 'small']) + raw_soup.find_all(class_=re.compile(r'footer|copyright|legal', re.I)):
                    footer_text += " " + footer.get_text(separator=" ", strip=True)
                    
                combined_text = (main_text[:4000] + "\n\n--- FOOTER & LEGAL ---\n\n" + footer_text[:3000]).strip()
                
                prompt = f"""Extract the exact registered legal entity name of this company from the website text.
Company Brand: {company}
Website: {website}

Rules:
- Look for copyright notices (e.g. "© 2024 [Name]"), footer links, or "About Us" text.
- The legal name usually contains Ltd, Limited, LLC, Inc, Pvt, Private, GmbH, etc.
- Return ONLY the legal name string. If absolutely not found, return EXACTLY: UNKNOWN

Website Text:
{combined_text[:7000]}"""
                
                response, _ = completion_with_fallback(
                    messages=[{"role": "user", "content": prompt}],
                    settings=get_settings(),
                    temperature=0.0,
                    timeout=30,
                    max_tokens=50
                )
                raw_legal = response.choices[0].message.content
                if raw_legal:
                    raw_legal = raw_legal.strip().strip('"\'')
                    if raw_legal.upper() != "UNKNOWN" and len(raw_legal) > 3:
                        legal_entity = raw_legal
                        identity_hints.append(legal_entity)
                        logger.info(f"[CompanyResolver] Discovered legal entity: {legal_entity}")
        except Exception as e:
            logger.warning(f"[CompanyResolver] Legal name discovery failed: {e}")

    profile = {
        "company_name": company,
        "website": website,
        "linkedin_company_page": linkedin,
        "headquarters": "UNKNOWN",
        "industry": "UNKNOWN",
        "founders": [],
        "services": [],
        "technologies": [],
        "employee_count": "UNKNOWN",
        "legal_entity": legal_entity,
        "country": "",
        "aliases": [company],
    }

    # ── Build search queries — always anchor with known domain to prevent mismatch ──
    q = quote_plus(company)
    queries = []

    if website:
        # Primary: site: queries against the KNOWN domain — highest precision
        domain = urlparse(website).netloc.replace("www.", "")
        queries.extend([
            f"site:{domain} about",
            f"site:{domain} services",
            f"site:{domain} team",
            f"site:{domain}",
        ])

    if email_domain and email_domain not in (website or ""):
        queries.append(f"site:{email_domain}")

    # General identity queries — include address/email to help find the right entity
    city = ""
    if address:
        # Extract city/region from address for geographic disambiguation
        city = address.split(",")[0].strip() if "," in address else address.strip()[:30]
    
    city_suffix = f' "{city}"' if city else ''

    queries.extend([
        f'"{company}" official website{city_suffix}',
        f'site:linkedin.com/company "{company}"{city_suffix}',
        f'"{company}" headquarters industry{city_suffix}',
        f'"{company}" founders{city_suffix}',
    ])
    
    # Precise registry queries using discovered legal name
    if legal_entity != "UNKNOWN":
        queries.extend([
            f'"{legal_entity}" legal entity registered company{city_suffix}',
            f'"{legal_entity}" MCA company master data',
            f'"{legal_entity}" ZaubaCorp',
            f'"{legal_entity}" Tofler',
            f'"{legal_entity}" Crunchbase',
        ])
    else:
        queries.extend([
            f'"{company}" legal entity registered company{city_suffix}',
            f'"{company}" MCA company master data{city_suffix}',
            f'"{company}" ZaubaCorp{city_suffix}',
            f'"{company}" Tofler{city_suffix}',
            f'"{company}" Crunchbase{city_suffix}',
        ])

    # If full address supplied, add one explicit full-address query
    if address:
        queries.append(f'"{company}" {address[:50]}')

    # B2B intelligence queries
    queries.extend([
        f'"{company}" services products solutions{city_suffix}',
        f'"{company}" competitors alternatives{city_suffix}',
        f'"{company}" customer reviews complaints{city_suffix}',
        f'"{company}" growth revenue expansion{city_suffix}',
        f'"{company}" tech stack technology{city_suffix}',
    ])

    return {
        "company_profile": profile,
        "search_queries": list(dict.fromkeys(queries + state.get("search_queries", []))),
        "status": "company_resolution_started",
        "progress_pct": 12,
        "node_timings": {"company_resolver": round(time.time() - started, 2)},
        "log": [
            f"[CompanyResolver] Identity discovery for '{company}'"
            + (f" anchored to domain: {urlparse(website).netloc}" if website else "")
        ],
    }
