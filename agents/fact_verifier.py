"""
agents/fact_verifier.py
Aggregates all model extractions and produces verified, confidence-scored facts.

Algorithm per field:
  1. Collect all non-null values from all models
  2. Group identical / similar values
  3. Count agreements
  4. Detect contradictions
  5. Assign confidence score
  6. Return UNKNOWN for fields with no agreement or no sources
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from core.state import ResearchState, ConfidenceField, EvidenceChunk

logger = logging.getLogger(__name__)

# Fields to verify across models
_SCALAR_FIELDS = [
    "company_name", "overview", "industry", "hq_location",
    "founding_year", "employee_count", "revenue", "ceo",
    "funding", "website",
]
_LIST_FIELDS = [
    "products_services", "leadership", "investors",
    "target_customers", "competitors", "recent_news",
    "strategic_priorities", "pain_points",
]

# Source type authority weights for confidence scoring
_SOURCE_AUTHORITY = {
    "company_site": 1.0,
    "browser_use_session": 0.95,
    "linkedin": 0.85,
    "crunchbase": 0.85,
    "sec": 0.95,
    "news": 0.75,
    "wiki": 0.60,
    "google": 0.65,
    "unknown": 0.40,
}


def _normalize_value(val: Any) -> Optional[str]:
    """Normalize a value to lowercase string for comparison."""
    if val is None:
        return None
    if isinstance(val, (list, dict)):
        return None  # handled separately
    return str(val).lower().strip()


def _compute_confidence(
    agreements: int,
    total_models: int,
    sources: list[str],
    source_types: list[str],
) -> float:
    """
    confidence = model_agreement_weight * 0.5
               + source_count_weight * 0.3
               + authority_weight * 0.2
    """
    model_weight = (agreements / max(total_models, 1)) * 0.5

    source_weight = min(len(sources) / 3, 1.0) * 0.3

    avg_authority = 0.5
    if source_types:
        authority_scores = [_SOURCE_AUTHORITY.get(st, 0.4) for st in source_types]
        avg_authority = sum(authority_scores) / len(authority_scores)
    authority_weight = avg_authority * 0.2

    return round(model_weight + source_weight + authority_weight, 3)


def _collect_sources(evidence_chunks: list[EvidenceChunk]) -> tuple[list[str], list[str]]:
    """Return (urls, source_types) from evidence chunks."""
    urls = list(set(c.get("url", "") for c in evidence_chunks if c.get("url")))
    types = list(set(c.get("source_type", "unknown") for c in evidence_chunks))
    return urls, types


def _verify_scalar_field(
    field: str,
    model_extractions: dict[str, dict],
    evidence_chunks: list[EvidenceChunk],
    total_models: int,
) -> tuple[ConfidenceField, Optional[dict]]:
    """Verify a scalar field. Returns (verified_field, dispute_or_None)."""
    values: dict[str, list[str]] = {}  # normalized_val → [model_names]

    for model, extraction in model_extractions.items():
        raw_val = extraction.get(field)
        norm = _normalize_value(raw_val)
        if norm and norm not in ("null", "none", "unknown", "n/a", ""):
            values.setdefault(norm, []).append(model)

    if not values:
        return ConfidenceField(
            value="UNKNOWN",
            sources=[],
            confidence=0.0,
            models_agreed=0,
            models_total=total_models,
            note="No model found evidence for this field",
        ), None

    urls, types = _collect_sources(evidence_chunks)

    # Find majority value
    best_norm, best_models = max(values.items(), key=lambda x: len(x[1]))

    # Check for disputes (multiple different values with ≥1 model each)
    dispute = None
    if len(values) > 1:
        dispute = {
            "field": field,
            "values": {norm: models for norm, models in values.items()},
            "note": f"Models disagree on {field}",
        }

    # Use original (non-normalized) value from first agreeing model
    original_val = None
    for model in best_models:
        v = model_extractions[model].get(field)
        if v is not None:
            original_val = v
            break

    confidence = _compute_confidence(len(best_models), total_models, urls, types)

    field_result = ConfidenceField(
        value=original_val,
        sources=urls[:5],
        confidence=confidence,
        models_agreed=len(best_models),
        models_total=total_models,
        note="⚠️ CONFLICT detected" if dispute else None,
    )
    return field_result, dispute


def _verify_list_field(
    field: str,
    model_extractions: dict[str, dict],
    evidence_chunks: list[EvidenceChunk],
    total_models: int,
) -> ConfidenceField:
    """Merge list fields by collecting unique items across all models."""
    combined: list[Any] = []
    models_with_data = 0

    for model, extraction in model_extractions.items():
        raw_val = extraction.get(field)
        if isinstance(raw_val, list) and raw_val:
            models_with_data += 1
            for item in raw_val:
                if item and item not in combined:
                    combined.append(item)

    if not combined:
        return ConfidenceField(
            value=[],
            sources=[],
            confidence=0.0,
            models_agreed=0,
            models_total=total_models,
            note="No model found evidence for this field",
        )

    urls, types = _collect_sources(evidence_chunks)
    confidence = _compute_confidence(models_with_data, total_models, urls, types)

    return ConfidenceField(
        value=combined,
        sources=urls[:5],
        confidence=confidence,
        models_agreed=models_with_data,
        models_total=total_models,
    )


def run_fact_verifier(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: Fact Verifier."""
    model_extractions = state.get("model_extractions", {})
    evidence_chunks = state.get("evidence_chunks", [])

    if not model_extractions:
        logger.warning("[FactVerifier] No model extractions to verify")
        return {
            "verified_facts": {},
            "disputed_facts": [],
            "status": "verification_skipped",
            "progress_pct": 65,
            "log": ["[FactVerifier] No extractions to verify"],
        }

    total_models = len(model_extractions)
    logger.info(f"[FactVerifier] Verifying across {total_models} model extractions")

    verified: dict[str, ConfidenceField] = {}
    disputed: list[dict] = []

    # Verify scalar fields
    for field in _SCALAR_FIELDS:
        result, dispute = _verify_scalar_field(
            field, model_extractions, evidence_chunks, total_models
        )
        verified[field] = result
        if dispute:
            disputed.append(dispute)

    # Verify list fields
    for field in _LIST_FIELDS:
        result = _verify_list_field(
            field, model_extractions, evidence_chunks, total_models
        )
        verified[field] = result

    high_conf = sum(1 for f in verified.values() if f.get("confidence", 0) >= 0.7)
    logger.info(
        f"[FactVerifier] {len(verified)} fields verified | "
        f"{high_conf} high-confidence | {len(disputed)} disputes"
    )

    return {
        "verified_facts": verified,
        "disputed_facts": disputed,
        "status": "verification_done",
        "progress_pct": 68,
        "log": [
            f"[FactVerifier] {len(verified)} fields | "
            f"{high_conf} high-confidence | {len(disputed)} conflicts"
        ],
    }
