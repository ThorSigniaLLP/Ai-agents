"""
core/config.py
Centralized configuration for company identity research.
"""
from __future__ import annotations

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM API Keys ──────────────────────────────────────────────────────────
    amazon_bedrock: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    cerebras_api_key: str = ""
    mistral_api_key: str = ""

    # ── Search API Keys (optional) ────────────────────────────────────────────
    brave_search_api_key: Optional[str] = None
    bing_search_api_key: Optional[str] = None    # for Bing Web Search API (optional)
    serpapi_key: Optional[str] = None

    # ── Research Config ───────────────────────────────────────────────────────
    max_research_iterations: int = 3
    max_retry_count: int = 3           # self-healing loop max retries
    min_evidence_items: int = 5        # minimum EvidenceItems before proceeding
    max_urls_to_fetch: int = 50        # top N from URL graph
    max_pages_per_query: int = 5
    page_load_timeout: int = 25
    browser_headless: bool = True

    # ── Model Config ──────────────────────────────────────────────────────────
    primary_model: str = "mistral/mistral-large-latest"
    extraction_model: str = "mistral/mistral-large-latest"
    fast_model: str = "mistral/mistral-large-latest"
    mistral_fallback_model: str = "mistral/mistral-large-latest"
    gemini_fallback_model: str = "gemini/gemini-2.5-flash"
    groq_fallback_model: str = "groq/llama-3.1-8b-instant"
    cerebras_fallback_model: str = "cerebras/gpt-oss-120b"
    openrouter_fallback_model: str = "openrouter/nvidia/nemotron-3-super-120b-a12b:free"

    # ── FastAPI ───────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8002
    cors_origins: list[str] = ["*"]

    # ── LangGraph ─────────────────────────────────────────────────────────────
    checkpoint_db_path: str = "research_checkpoints.db"
    
    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql://postgres:password@localhost:5432/research_db"


# Singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# ── Model Registry (for reference) ───────────────────────────────────────────
MODEL_REGISTRY: dict[str, dict] = {
    "llama-3.3-70b": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "role": "primary",
        "max_tokens": 8192,
    },
    "qwen3-32b": {
        "provider": "groq",
        "model_id": "qwen/qwen3-32b",
        "role": "extractor",
        "max_tokens": 8192,
    },
    "llama-3.1-8b": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "role": "fast",
        "max_tokens": 4096,
    },
    "nemotron-120b": {
        "provider": "openrouter",
        "model_id": "nvidia/nemotron-3-super-120b-a12b:free",
        "role": "fallback",
        "max_tokens": 4096,
    },
}

EXTRACTOR_MODELS = [k for k, v in MODEL_REGISTRY.items() if v["role"] == "extractor"]
