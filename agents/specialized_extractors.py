"""Deep B2B intelligence extractors for company research."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable
from urllib.parse import urlparse

from litellm import completion as litellm_completion
from core.llm_router import completion_with_fallback

from core.config import get_settings
from core.state import EvidenceChunk, EvidenceItem, ResearchState
from core.db_tracer import trace_extracted_items

logger = logging.getLogger(__name__)

HIGH_AUTHORITY_PAGES = {"HOME", "ABOUT", "SERVICES", "SOLUTIONS", "CASE_STUDIES", "TEAM"}
IDENTITY_SOURCES = {"OFFICIAL_WEBSITE", "MCA", "ZAUBACORP", "TOFLER", "LINKEDIN_COMPANY_PAGE", "CRUNCHBASE"}
REVIEW_SOURCES = {"g2.com", "glassdoor.com", "trustpilot.com", "clutch.co", "capterra.com", "ambitionbox.com", "reddit.com"}
NEWS_SOURCES = {"techcrunch.com", "bloomberg.com", "reuters.com", "forbes.com", "businesswire.com", "prnewswire.com",
                "economictimes.indiatimes.com", "moneycontrol.com", "yourstory.com"}


def _allowed(chunk: EvidenceChunk, categories: set[str], page_types: set[str] | None = None) -> bool:
    category = chunk.get("source_category", "UNKNOWN")
    if category not in categories:
        return False
    if category == "OFFICIAL_WEBSITE" and page_types is not None:
        return chunk.get("page_type", "OTHER") in page_types
    return True


def _is_review_chunk(chunk: EvidenceChunk) -> bool:
    domain = chunk.get("domain", "")
    return any(r in domain for r in REVIEW_SOURCES)


def _is_news_chunk(chunk: EvidenceChunk) -> bool:
    domain = chunk.get("domain", "")
    return any(n in domain for n in NEWS_SOURCES)


def _llm_extract(prompt: str, settings=None, max_tokens: int = 1200) -> dict | list | None:
    """Call LLM with fallback and return parsed JSON."""
    if settings is None:
        settings = get_settings()
    try:
        response, target = completion_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            settings=settings,
            temperature=0.0,
            timeout=60,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        return json.loads(raw)
    except Exception as exc:
        logger.warning("[Extractor] LLM call failed: %s", exc)
        return None


# ── Original field-level extractor (kept for identity fields) ──────────────────

def _extract_values(
    state: ResearchState,
    field: str,
    instruction: str,
    predicate: Callable[[EvidenceChunk], bool],
) -> list[EvidenceItem]:
    chunks = [chunk for chunk in state.get("evidence_chunks", []) if predicate(chunk)][:4]
    if not chunks:
        return []

    indexed = []
    by_id: dict[str, EvidenceChunk] = {}
    for index, chunk in enumerate(chunks, 1):
        source_id = f"S{index}"
        by_id[source_id] = chunk
        indexed.append(
            f"[{source_id}] URL: {chunk['url']}\n"
            f"Page type: {chunk.get('page_type', 'UNKNOWN')}\n"
            f"Text: {chunk['chunk'][:1000]}"
        )

    prompt = f"""You extract one company profile field from supplied evidence.
Company: {state['company']}
Field: {field}
Instruction: {instruction}

Rules:
- Use only explicit text in the supplied sources.
- Do not infer, generalize, or use prior knowledge.
- Return UNKNOWN by returning an empty items array.
- Every value must cite exactly one supplied source_id.
- Personal posts, social media, blogs, motivational content, and thought leadership are forbidden.

Evidence:
{chr(10).join(indexed)}

Return only JSON: {{"items": [{{"value": "exact concise value", "source_id": "S1"}}]}}
"""
    time.sleep(1)  # rate limit mitigation
    result = _llm_extract(prompt)
    if not result:
        return []

    evidence: list[EvidenceItem] = []
    seen = set()
    for item in result.get("items", []):
        value = item.get("value")
        source = by_id.get(item.get("source_id", ""))
        if not value or str(value).strip().upper() == "UNKNOWN" or source is None:
            continue
        key = (str(value).strip().lower(), source["url"])
        if key in seen:
            continue
        seen.add(key)
        evidence.append(EvidenceItem(
            field=field,
            value=str(value).strip(),
            source_url=source["url"],
            source_domain=urlparse(source["url"]).netloc.replace("www.", ""),
            page_type=source.get("page_type", source.get("source_category", "UNKNOWN")),
            authority_score=float(source.get("authority_score", 0.5)),
            relevance_score=float(source.get("rerank_score", 0.0)),
            publication_date=source.get("published_date", ""),
        ))
    return evidence


# ── Identity extractors (existing) ────────────────────────────────────────────

def overview_extractor(state: ResearchState) -> list[EvidenceItem]:
    return _extract_values(
        state, "industry", "Extract the company's explicitly stated industry classification.",
        lambda chunk: _allowed(chunk, IDENTITY_SOURCES, {"HOME", "ABOUT"}),
    )


def service_extractor(state: ResearchState) -> list[EvidenceItem]:
    return _extract_values(
        state, "services", "Extract each explicitly offered service or solution as a separate item.",
        lambda chunk: _allowed(chunk, {"OFFICIAL_WEBSITE"}, {"HOME", "ABOUT", "SERVICES", "SOLUTIONS", "CASE_STUDIES"}),
    )


def technology_extractor(state: ResearchState) -> list[EvidenceItem]:
    return _extract_values(
        state, "technologies", "Extract each technology detected as used by the company as a separate item.",
        lambda chunk: chunk.get("source_category") == "TECH_STACK" or chunk.get("source_type") == "technology",
    )


def leadership_extractor(state: ResearchState) -> list[EvidenceItem]:
    return _extract_values(
        state, "founders", "Extract only people explicitly identified as founders or co-founders.",
        lambda chunk: _allowed(chunk, {"OFFICIAL_WEBSITE", "MCA", "ZAUBACORP", "TOFLER", "CRUNCHBASE"}, {"ABOUT", "TEAM"}),
    )


def employee_count_extractor(state: ResearchState) -> list[EvidenceItem]:
    return _extract_values(
        state, "employee_count", "Extract the employee count or company-size range shown on the LinkedIn company page.",
        lambda chunk: chunk.get("source_category") == "LINKEDIN_COMPANY_PAGE"
            or "linkedin" in chunk.get("domain", ""),
    )


def legal_entity_extractor(state: ResearchState) -> list[EvidenceItem]:
    return _extract_values(
        state, "legal_entity", "Extract the exact registered legal entity name.",
        lambda chunk: chunk.get("source_category") in {"MCA", "ZAUBACORP", "TOFLER"},
    )


def headquarters_extractor(state: ResearchState) -> list[EvidenceItem]:
    return _extract_values(
        state, "headquarters", "Extract the explicitly stated headquarters location, including city, state and country.",
        lambda chunk: _allowed(chunk, IDENTITY_SOURCES, {"HOME", "ABOUT"})
            or "linkedin" in chunk.get("domain", "")
            or "crunchbase" in chunk.get("domain", ""),
    )


# ── NEW: Firmographics extractor ───────────────────────────────────────────────

def firmographics_extractor(state: ResearchState) -> list[EvidenceItem]:
    """Extract founded_year, revenue, and competitors from all available chunks."""
    chunks = state.get("evidence_chunks", [])
    if not chunks:
        return []

    # Use top 5 firmographic-relevant chunks
    relevant = [c for c in chunks if any(
        kw in c.get("domain", "") + c.get("chunk", "").lower()
        for kw in ["crunchbase", "tracxn", "founded", "established", "revenue", "funding", "employees"]
    )][:5]
    if not relevant:
        relevant = chunks[:3]

    indexed = [
        f"[S{i+1}] URL: {c['url']}\nText: {c['chunk'][:800]}"
        for i, c in enumerate(relevant)
    ]
    by_id = {f"S{i+1}": c for i, c in enumerate(relevant)}

    prompt = f"""Extract firmographic data about "{state['company']}" from the evidence below.

Evidence:
{chr(10).join(indexed)}

Rules:
- Use ONLY what is explicitly stated in the sources.
- Return null for fields not found.
- For competitors, list only named companies explicitly mentioned as competitors or alternatives.

Return ONLY valid JSON:
{{
  "founded_year": "YYYY or null",
  "revenue": "e.g. $5M ARR or null",
  "competitors": ["CompetitorA", "CompetitorB"],
  "source_id": "S1"
}}
"""
    result = _llm_extract(prompt, max_tokens=600)
    if not result:
        return []

    evidence = []
    src_id = result.get("source_id", "S1")
    source = by_id.get(src_id, relevant[0] if relevant else None)
    src_url = source["url"] if source else ""
    src_domain = urlparse(src_url).netloc.replace("www.", "") if src_url else ""
    auth = float(source.get("authority_score", 0.5)) if source else 0.5
    rel = float(source.get("rerank_score", 0.5)) if source else 0.5

    for field in ["founded_year", "revenue"]:
        val = result.get(field)
        if val and str(val).lower() not in ("null", "none", "unknown", ""):
            evidence.append(EvidenceItem(
                field=field, value=str(val), source_url=src_url,
                source_domain=src_domain, page_type="FIRMOGRAPHICS",
                authority_score=auth, relevance_score=rel, publication_date="",
            ))

    for comp in result.get("competitors", []):
        if comp and str(comp).strip():
            evidence.append(EvidenceItem(
                field="competitors", value=str(comp).strip(), source_url=src_url,
                source_domain=src_domain, page_type="FIRMOGRAPHICS",
                authority_score=auth, relevance_score=rel, publication_date="",
            ))
    return evidence


# ── NEW: Pain points extractor ─────────────────────────────────────────────────

def pain_points_extractor(state: ResearchState) -> list[EvidenceItem]:
    """Extract structured pain points from review & social sources."""
    chunks = state.get("evidence_chunks", [])
    review_chunks = [c for c in chunks if _is_review_chunk(c)][:6]
    if not review_chunks:
        # Fall back to any chunk mentioning complaints
        review_chunks = [c for c in chunks if any(
            kw in c.get("chunk", "").lower()
            for kw in ["complaint", "problem", "issue", "review", "slow", "difficult", "expensive", "bug"]
        )][:4]
    if not review_chunks:
        return []

    indexed = [
        f"[S{i+1}] Source: {c['url']}\nText: {c['chunk'][:1000]}"
        for i, c in enumerate(review_chunks)
    ]

    prompt = f"""You are extracting customer pain points and complaints about "{state['company']}" from review sites and social media.

Evidence:
{chr(10).join(indexed)}

Rules:
- Only extract explicitly stated complaints, issues, or problems — not praise.
- Each issue must be directly quoted or paraphrased from the evidence.
- severity: "high" (many complaints or critical), "medium" (notable), "low" (minor)
- frequency: how often mentioned (e.g. "mentioned in 3 reviews", "single mention")
- If no genuine complaints found, return empty array.

Return ONLY valid JSON:
{{
  "pain_points": [
    {{
      "issue": "concise issue description",
      "severity": "high|medium|low",
      "source": "domain name or platform",
      "quote": "exact or near-exact quote from text",
      "frequency": "e.g. mentioned in 4 reviews"
    }}
  ]
}}
"""
    result = _llm_extract(prompt, max_tokens=1500)
    if not result:
        return []

    evidence = []
    for pp in result.get("pain_points", []):
        if not pp.get("issue"):
            continue
        # Store as a special JSON EvidenceItem
        evidence.append(EvidenceItem(
            field="pain_points",
            value=json.dumps(pp),
            source_url=review_chunks[0]["url"],
            source_domain=review_chunks[0].get("domain", ""),
            page_type="REVIEW",
            authority_score=0.75,
            relevance_score=0.8,
            publication_date="",
        ))
    return evidence


# ── NEW: Growth signals extractor ─────────────────────────────────────────────

def growth_signals_extractor(state: ResearchState) -> list[EvidenceItem]:
    """Extract growth signals: job postings, news, expansion."""
    chunks = state.get("evidence_chunks", [])
    news_chunks = [c for c in chunks if _is_news_chunk(c)]
    job_chunks = [c for c in chunks if "linkedin" in c.get("domain", "") or "job" in c.get("url", "").lower()]
    growth_chunks = news_chunks[:4] + job_chunks[:3]
    if not growth_chunks:
        growth_chunks = [c for c in chunks if any(
            kw in c.get("chunk", "").lower()
            for kw in ["expand", "hire", "launch", "announce", "raise", "partner", "open", "new office"]
        )][:5]
    if not growth_chunks:
        return []

    indexed = [
        f"[S{i+1}] Source: {c['url']}\nDate: {c.get('published_date', 'unknown')}\nText: {c['chunk'][:800]}"
        for i, c in enumerate(growth_chunks)
    ]

    prompt = f"""Extract growth signals and job postings for "{state['company']}" from the evidence below.
Focus on: recent news, new hires, expansions, funding, partnerships, product launches.

Evidence:
{chr(10).join(indexed)}

Rules:
- Only include explicitly stated facts from the evidence.
- recent_news: list of news headlines (max 5 most recent)
- job_postings: list of specific job titles mentioned
- expansion_signals: list of specific expansion events mentioned
- hiring_trend: "accelerating" | "stable" | "declining" | "UNKNOWN" based on evidence
- disconnection_signals: list of negative signals (layoffs, leadership exits, product failures)

Return ONLY valid JSON:
{{
  "job_postings": [],
  "recent_news": [],
  "expansion_signals": [],
  "hiring_trend": "UNKNOWN",
  "disconnection_signals": []
}}
"""
    result = _llm_extract(prompt, max_tokens=1000)
    if not result:
        return []

    src_url = growth_chunks[0]["url"] if growth_chunks else ""
    src_domain = growth_chunks[0].get("domain", "") if growth_chunks else ""

    evidence = []
    # Store growth_signals as one evidence item (JSON blob)
    growth_data = {
        "job_postings": result.get("job_postings", []),
        "recent_news": result.get("recent_news", []),
        "expansion_signals": result.get("expansion_signals", []),
        "hiring_trend": result.get("hiring_trend", "UNKNOWN"),
    }
    evidence.append(EvidenceItem(
        field="growth_signals",
        value=json.dumps(growth_data),
        source_url=src_url, source_domain=src_domain,
        page_type="NEWS", authority_score=0.75, relevance_score=0.8, publication_date="",
    ))

    for signal in result.get("disconnection_signals", []):
        if signal and str(signal).strip():
            evidence.append(EvidenceItem(
                field="disconnection_signals", value=str(signal).strip(),
                source_url=src_url, source_domain=src_domain,
                page_type="NEWS", authority_score=0.75, relevance_score=0.8, publication_date="",
            ))
    return evidence


# ── NEW: Tech stack extractor ──────────────────────────────────────────────────

def tech_stack_extractor(state: ResearchState) -> list[EvidenceItem]:
    """Extract CRM/ERP/cloud/marketing tech stack from BuiltWith, Wappalyzer, official site."""
    chunks = state.get("evidence_chunks", [])
    tech_chunks = [c for c in chunks if any(
        kw in c.get("domain", "").lower() + c.get("url", "").lower()
        for kw in ["builtwith", "wappalyzer", "stackshare", "techstack"]
    )]
    # Also include official website chunks that mention tech
    website_chunks = [c for c in chunks if c.get("source_category") == "OFFICIAL_WEBSITE"
                      and any(kw in c.get("chunk", "").lower()
                              for kw in ["salesforce", "hubspot", "sap", "oracle", "aws", "azure", "gcp", "crm", "erp"])]
    all_tech = (tech_chunks + website_chunks)[:5]
    if not all_tech:
        return []

    indexed = [
        f"[S{i+1}] Source: {c['url']}\nText: {c['chunk'][:800]}"
        for i, c in enumerate(all_tech)
    ]

    prompt = f"""Extract the technology stack used by "{state['company']}" from the evidence.

Evidence:
{chr(10).join(indexed)}

Rules:
- Only include tools explicitly mentioned in the evidence.
- Use null if not found.
- development_stack: list of languages, frameworks, databases mentioned.

Return ONLY valid JSON:
{{
  "crm": "e.g. Salesforce or null",
  "erp": "e.g. SAP or null",
  "marketing_tools": [],
  "development_stack": [],
  "cloud_provider": "e.g. AWS or null"
}}
"""
    result = _llm_extract(prompt, max_tokens=600)
    if not result:
        return []

    src_url = all_tech[0]["url"] if all_tech else ""
    src_domain = all_tech[0].get("domain", "") if all_tech else ""

    evidence = [EvidenceItem(
        field="tech_stack",
        value=json.dumps(result),
        source_url=src_url, source_domain=src_domain,
        page_type="TECH_STACK", authority_score=0.75, relevance_score=0.7, publication_date="",
    )]
    return evidence


# ── NEW: Pitch opportunities synthesizer ──────────────────────────────────────

def pitch_opportunities_extractor(state: ResearchState) -> list[EvidenceItem]:
    """Synthesize 3-5 pitch opportunities from all collected intelligence."""
    # Gather all collected evidence
    evidence_items = state.get("evidence_items", [])
    profile = state.get("company_profile", {})

    # Build a compact summary of what we know
    known = []
    for item in evidence_items:
        if item.get("field") in ("pain_points", "growth_signals", "tech_stack", "services"):
            known.append(f"[{item['field']}] {str(item['value'])[:300]}")

    if not known:
        return []

    prompt = f"""Based on the intelligence gathered about "{state['company']}", generate 3-5 specific, actionable B2B sales pitch opportunities.

Company Intelligence:
{chr(10).join(known[:15])}

Rules:
- Each pitch must be grounded in the actual evidence above.
- Pitch opportunities should be specific "hooks" a sales rep can use.
- Format: brief, punchy, action-oriented sentence.
- If insufficient data, return fewer pitches.

Return ONLY valid JSON:
{{
  "pitch_opportunities": [
    "Offer X to solve their Y pain point",
    "..."
  ]
}}
"""
    result = _llm_extract(prompt, max_tokens=500)
    if not result:
        return []

    evidence = []
    for pitch in result.get("pitch_opportunities", []):
        if pitch and str(pitch).strip():
            evidence.append(EvidenceItem(
                field="pitch_opportunities", value=str(pitch).strip(),
                source_url="synthesized", source_domain="synthesized",
                page_type="SYNTHESIS", authority_score=0.7, relevance_score=0.7, publication_date="",
            ))
    return evidence


# ── Main node ─────────────────────────────────────────────────────────────────

def run_specialized_extractors(state: ResearchState) -> dict[str, Any]:
    started = time.time()
    settings = get_settings()

    extractors = [
        # Identity (existing)
        overview_extractor,
        service_extractor,
        technology_extractor,
        leadership_extractor,
        employee_count_extractor,
        legal_entity_extractor,
        headquarters_extractor,
        # New deep B2B extractors
        firmographics_extractor,
        pain_points_extractor,
        growth_signals_extractor,
        tech_stack_extractor,
    ]

    evidence: list[EvidenceItem] = []
    for extractor in extractors:
        try:
            items = extractor(state)
            evidence.extend(items)
            logger.info("[Extractors] %s → %d items", extractor.__name__, len(items))
        except Exception as exc:
            logger.warning("[Extractors] %s failed: %s", extractor.__name__, exc)

    # Pitch synthesis runs AFTER all other evidence is collected
    try:
        # Temporarily inject current evidence into state for pitch synthesizer
        state_copy = dict(state)
        state_copy["evidence_items"] = evidence
        pitch_items = pitch_opportunities_extractor(state_copy)
        evidence.extend(pitch_items)
        logger.info("[Extractors] pitch_opportunities_extractor → %d items", len(pitch_items))
    except Exception as exc:
        logger.warning("[Extractors] pitch_opportunities_extractor failed: %s", exc)

    trace_extracted_items(state.get("job_id", ""), evidence)

    return {
        "evidence_items": evidence,
        "status": "specialized_extraction_done",
        "progress_pct": 78,
        "node_timings": {"specialized_extractors": round(time.time() - started, 2)},
        "log": [f"[Extractors] Produced {len(evidence)} evidence items (identity + B2B intelligence)"],
    }
