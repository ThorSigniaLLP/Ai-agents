"""Deep B2B intelligence extractors using async chunked TOML Map-Reduce."""
from __future__ import annotations

import asyncio
import logging
import re
import time
import tomllib
from typing import Any
from urllib.parse import urlparse

from core.llm_router import acompletion_with_fallback
from core.config import get_settings
from core.state import EvidenceChunk, EvidenceItem, ResearchState
from core.db_tracer import trace_extracted_items

logger = logging.getLogger(__name__)


def _build_entity_context(state: ResearchState) -> str:
    """Build a short entity anchor block from card_info to prevent hallucination."""
    card = state.get("card_info", {}) or {}
    parts = []
    company = state.get("company", "")
    if company:
        parts.append(f"Company name: {company}")
    if card.get("website"):
        parts.append(f"Known website: {card['website']}")
    if card.get("address"):
        parts.append(f"Address: {card['address']}")
    if card.get("email"):
        parts.append(f"Email domain: @{card['email'].split('@')[-1]}" if "@" in card.get("email", "") else f"Email: {card['email']}")
    if card.get("mobile"):
        parts.append(f"Phone: {card['mobile']}")
    if not parts:
        return f"Target company: {company}"
    return "Entity anchor (use ONLY this company, ignore similar names):\n" + "\n".join(f"  - {p}" for p in parts)


async def _async_toml_extract(prompt: str, settings=None, max_tokens: int = 1500) -> dict | None:
    """Call LLM asynchronously and parse returned TOML, with repair for truncated strings."""
    if settings is None:
        settings = get_settings()
    try:
        response, target = await acompletion_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            settings=settings,
            temperature=0.0,
            timeout=90,
            max_tokens=max_tokens,
            response_format={"type": "text"},
        )
        raw = response.choices[0].message.content
        if not raw:
            return None
        raw = raw.strip()
        # Extract innermost TOML content between backticks (greedy interior extraction)
        m = re.search(r"```(?:toml)?[ \t]*\n?(.*?)\n?```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
        else:
            # No closing backtick (truncated) - strip opening fence if present
            raw = re.sub(r"^```(?:toml)?[ \t]*\n?", "", raw).strip()

        # ── TOML repair: close any unterminated string on the last line ──
        lines = raw.splitlines()
        if lines:
            last = lines[-1]
            # Count unescaped quotes in the last line
            quote_count = last.count('"') - last.count('\\"')
            if quote_count % 2 == 1:
                # Odd number of quotes — string is unterminated; close it and truncate gracefully
                lines[-1] = last + '"'
                # Drop any partially written lines after the first array that isn't closed
            raw = "\n".join(lines)

        # Parse TOML
        try:
            return tomllib.loads(raw)
        except Exception as parse_err:
            logger.warning("[Extractor] TOML parse failed: %s. Raw: %s...", parse_err, raw[:150])
            return None
    except Exception as exc:
        logger.warning("[Extractor] async LLM call failed: %s", exc)
        return None

def _create_evidence(field: str, value: Any, source: EvidenceChunk | None) -> EvidenceItem:
    src_url = source["url"] if source else "synthesized"
    src_domain = source.get("domain", "") if source else "synthesized"
    auth = float(source.get("authority_score", 0.5)) if source else 0.5
    rel = float(source.get("rerank_score", 0.5)) if source else 0.5
    page_type = source.get("page_type", "UNKNOWN") if source else "SYNTHESIZED"
    
    return EvidenceItem(
        field=field,
        value=value,
        source_url=src_url,
        source_domain=src_domain,
        page_type=page_type,
        authority_score=auth,
        relevance_score=rel,
        publication_date=source.get("published_date", "") if source else "",
    )

async def extract_firmographics(state: ResearchState) -> list[EvidenceItem]:
    chunks = state.get("evidence_chunks", [])
    entity_ctx = _build_entity_context(state)
    relevant = [c for c in chunks if any(
        kw in c.get("domain", "") + c.get("chunk", "").lower()
        for kw in ["crunchbase", "tracxn", "zauba", "tofler", "mca", "founded", "established", "revenue", "funding", "capital", "financial", "competitor", "alternative", "vs", "about", "headquarters", "employee"]
    )][:8]
    if not relevant:
        relevant = chunks[:4]

    indexed = [f"[S{i+1}] URL: {c['url']}\nText: {c['chunk'][:1200]}" for i, c in enumerate(relevant)]
    by_id = {f"S{i+1}": c for i, c in enumerate(relevant)}

    prompt = f"""You are extracting B2B intelligence. Focus ONLY on the exact company below. Do NOT confuse with similarly-named companies.

{entity_ctx}

Extract core firmographics from the evidence below. Use TOML.

Evidence:
{chr(10).join(indexed)}

Rules:
- Use explicit facts only. Exclude if not found (omit field entirely rather than writing UNKNOWN).
- competitors must be named companies mentioned as direct competitors or alternatives.
- services must be a comprehensive list of explicitly offered services/products.
- founders must be an array of named individuals.
- Provide source_id (e.g. "S1") for the primary source.

Output EXACTLY this TOML format:
```toml
source_id = "S1"
overview = "2-3 sentence overview of what the company does..."
headquarters = "City, State, Country"
industry = "Industry Name"
legal_entity = "Full Registered Name"
employee_count = "e.g. 50-200"
founded_year = "YYYY"
revenue = "e.g. $5M ARR or Paid Up Capital: INR X"
competitors = ["CompA", "CompB"]
services = ["ServiceA", "ServiceB"]
founders = ["Name1", "Name2"]
```"""
    result = await _async_toml_extract(prompt, max_tokens=1800)
    if not result:
        return []

    source = by_id.get(result.get("source_id", "S1"), relevant[0] if relevant else None)
    evidence = []
    for field in ["overview", "headquarters", "industry", "legal_entity", "employee_count", "founded_year", "revenue"]:
        val = result.get(field)
        if val and str(val).upper() not in ("UNKNOWN", "NONE", "NULL", ""):
            evidence.append(_create_evidence(field, str(val), source))
            
    for arr_field in ["competitors", "services", "founders"]:
        arr = result.get(arr_field, [])
        if isinstance(arr, list):
            for item in arr:
                if item and str(item).strip():
                    evidence.append(_create_evidence(arr_field, str(item).strip(), source))
    return evidence


async def extract_pain_points(state: ResearchState) -> list[EvidenceItem]:
    chunks = state.get("evidence_chunks", [])
    entity_ctx = _build_entity_context(state)
    relevant = [c for c in chunks if any(
        kw in c.get("domain", "") + c.get("chunk", "").lower()
        for kw in ["limitation", "challenge", "downside", "lack of", "missing", "delay", "inefficiency", "bottleneck", "outdated", "expensive", "poor", "downtime", "problem", "issue", "struggle", "gap", "fail", "complaint"]
    )][:8]
    if not relevant:
        return []

    indexed = [f"[S{i+1}] Source: {c['url']}\nText: {c['chunk'][:1200]}" for i, c in enumerate(relevant)]
    by_id = {f"S{i+1}": c for i, c in enumerate(relevant)}

    prompt = f"""You are a B2B sales analyst identifying BUSINESS pain points and operational challenges. Focus ONLY on the exact company below. Do NOT include complaints about other companies or general employee grievances.

{entity_ctx}

Evidence:
{chr(10).join(indexed)}

Rules:
- Extract explicitly stated BUSINESS pain points, tech gaps, operational inefficiencies, or market challenges faced by THIS company's customers or by the company itself.
- Examples: "Relies on outdated manual processes", "Struggles with customer retention", "Lacks modern ERP integration", "High turnover affecting service delivery".
- Do NOT extract generic employee reviews (e.g., "bad management") unless they indicate a systemic business/service failure that impacts customers or scalability.
- Severity: "high", "medium", "low".
- If no business pain points are found, return: pain_points = []

Output EXACTLY this TOML format (use [[pain_points]] array-of-tables for multiple entries):
```toml
[[pain_points]]
source_id = "S1"
issue = "Short description"
severity = "high"
quote = "Exact or near-exact quote from source"
frequency = "e.g. mentioned in 3 reviews"
```"""
    result = await _async_toml_extract(prompt, max_tokens=1000)
    if not result:
        return []

    evidence = []
    for pp in result.get("pain_points", []):
        if not pp.get("issue"): continue
        source = by_id.get(pp.get("source_id", "S1"), relevant[0] if relevant else None)
        import json
        evidence.append(_create_evidence("pain_points", json.dumps(pp), source))
    return evidence


async def extract_growth_signals(state: ResearchState) -> list[EvidenceItem]:
    chunks = state.get("evidence_chunks", [])
    entity_ctx = _build_entity_context(state)
    relevant = [c for c in chunks if any(
        kw in c.get("domain", "") + c.get("chunk", "").lower()
        for kw in ["news", "hire", "hiring", "job", "expand", "announce", "raise", "partner", "launch", "office", "layoff", "cagr", "growth"]
    )][:8]
    if not relevant:
        return []

    indexed = [f"[S{i+1}] Source: {c['url']}\nText: {c['chunk'][:1200]}" for i, c in enumerate(relevant)]
    by_id = {f"S{i+1}": c for i, c in enumerate(relevant)}

    prompt = f"""You are extracting growth and business signals. Focus ONLY on the exact company below.

{entity_ctx}

Evidence:
{chr(10).join(indexed)}

Rules:
- Extract only signals directly attributed to this company.
- Do NOT include signals from other companies.
- If no signals found for a field, use an empty array [].

Output EXACTLY this TOML format:
```toml
source_id = "S1"
hiring_trend = "accelerating | stable | declining | UNKNOWN"
job_postings = ["Title 1", "Title 2"]
recent_news = ["Headline 1", "Headline 2"]
expansion_signals = ["Expansion detail 1"]
disconnection_signals = ["Layoffs, leadership exits, etc."]
```"""
    result = await _async_toml_extract(prompt, max_tokens=800)
    if not result:
        return []

    source = by_id.get(result.get("source_id", "S1"), relevant[0] if relevant else None)
    import json
    growth_data = {
        "job_postings": result.get("job_postings", []),
        "recent_news": result.get("recent_news", []),
        "expansion_signals": result.get("expansion_signals", []),
        "hiring_trend": result.get("hiring_trend", "UNKNOWN"),
    }
    evidence = [_create_evidence("growth_signals", json.dumps(growth_data), source)]
    
    for signal in result.get("disconnection_signals", []):
        if signal and str(signal).strip():
            evidence.append(_create_evidence("disconnection_signals", str(signal).strip(), source))
    return evidence


async def extract_tech_stack(state: ResearchState) -> list[EvidenceItem]:
    chunks = state.get("evidence_chunks", [])
    relevant = [c for c in chunks if any(
        kw in c.get("domain", "") + c.get("chunk", "").lower()
        for kw in ["builtwith", "wappalyzer", "techstack", "salesforce", "sap", "aws", "azure", "cloud", "crm", "erp"]
    )][:6]
    if not relevant:
        relevant = chunks[:3]

    indexed = [f"[S{i+1}] Source: {c['url']}\nText: {c['chunk'][:1200]}" for i, c in enumerate(relevant)]
    by_id = {f"S{i+1}": c for i, c in enumerate(relevant)}

    prompt = f"""Extract the technology stack for "{state['company']}". Use TOML.

Evidence:
{chr(10).join(indexed)}

Output EXACTLY this TOML format:
```toml
source_id = "S1"
crm = "e.g. Salesforce or UNKNOWN"
erp = "e.g. SAP or UNKNOWN"
cloud_provider = "e.g. AWS or UNKNOWN"
marketing_tools = ["Tool 1", "Tool 2"]
development_stack = ["Lang 1", "DB 1"]
```"""
    result = await _async_toml_extract(prompt, max_tokens=600)
    if not result:
        return []

    source = by_id.get(result.get("source_id", "S1"), relevant[0] if relevant else None)
    import json
    evidence = [_create_evidence("tech_stack", json.dumps(result), source)]
    return evidence


async def extract_pitch_opportunities(state: ResearchState, prior_evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    known = []
    for item in prior_evidence:
        if item.get("field") in ("pain_points", "growth_signals", "tech_stack", "services", "overview"):
            known.append(f"[{item['field']}] {str(item['value'])[:400]}")
    if not known:
        return []

    prompt = f"""Generate 3-5 specific B2B sales pitch opportunities for "{state['company']}" based on this intelligence. Use TOML.

Intelligence:
{chr(10).join(known[:20])}

Output EXACTLY this TOML format:
```toml
pitch_opportunities = [
  "Offer X to solve their Y pain point",
  "Partner on Z to accelerate their expansion"
]
```"""
    result = await _async_toml_extract(prompt, max_tokens=600)
    if not result:
        return []

    evidence = []
    for pitch in result.get("pitch_opportunities", []):
        if pitch and str(pitch).strip():
            evidence.append(_create_evidence("pitch_opportunities", str(pitch).strip(), None))
    return evidence


def run_specialized_extractors(state: ResearchState) -> dict[str, Any]:
    started = time.time()
    
    async def _run_all():
        # Execute 4 primary macro-extractors concurrently
        results = await asyncio.gather(
            extract_firmographics(state),
            extract_pain_points(state),
            extract_growth_signals(state),
            extract_tech_stack(state),
            return_exceptions=True
        )
        
        evidence = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.warning("[Extractors] Macro step %d failed: %s", i+1, res)
            elif res:
                evidence.extend(res)
                
        # Pitch synthesis depends on the primary evidence
        try:
            pitch_items = await extract_pitch_opportunities(state, evidence)
            evidence.extend(pitch_items)
        except Exception as exc:
            logger.warning("[Extractors] Pitch synthesis failed: %s", exc)
            
        return evidence

    # Run the asyncio event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we are already inside an event loop (e.g. uvicorn), create a task or use nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
            evidence = loop.run_until_complete(_run_all())
        else:
            evidence = asyncio.run(_run_all())
    except RuntimeError:
        evidence = asyncio.run(_run_all())

    trace_extracted_items(state.get("job_id", ""), evidence)

    logger.info("[Extractors] Multi-step TOML Map-Reduce complete. Produced %d items.", len(evidence))
    return {
        "evidence_items": evidence,
        "status": "specialized_extraction_done",
        "progress_pct": 78,
        "node_timings": {"specialized_extractors": round(time.time() - started, 2)},
        "log": [f"[Extractors] Produced {len(evidence)} evidence items via parallel TOML extraction"],
    }
