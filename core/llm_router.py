"""Provider-aware LiteLLM routing for structured company extraction."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from litellm import completion as litellm_completion
from litellm import acompletion as litellm_acompletion

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelTarget:
    provider: str
    model: str
    api_key: str


def extraction_targets(settings: Settings | None = None) -> list[ModelTarget]:
    """Return configured models in primary-to-fallback order."""
    settings = settings or get_settings()
    candidates = [
        ModelTarget("cerebras", settings.fast_model, settings.cerebras_api_key),
        ModelTarget("mistral", settings.mistral_fallback_model, settings.mistral_api_key),
        ModelTarget("gemini", settings.gemini_fallback_model, settings.gemini_api_key),
        ModelTarget("groq", settings.groq_fallback_model, settings.groq_api_key),
        ModelTarget("openrouter", settings.openrouter_fallback_model, settings.openrouter_api_key),
    ]
    return [target for target in candidates if target.api_key and target.model]


def completion_with_fallback(
    *,
    messages: list[dict[str, str]],
    settings: Settings | None = None,
    timeout: int = 45,
    max_tokens: int = 800,
    response_format: dict[str, Any] | None = None,
    temperature: float = 0.0,
) -> tuple[Any, ModelTarget]:
    """Try Cerebras, Mistral, Gemini, Groq, then OpenRouter without failing on one provider outage."""
    settings = settings or get_settings()
    targets = extraction_targets(settings)
    if not targets:
        raise RuntimeError("No extraction provider API key is configured")

    failures = []
    for target in targets:
        kwargs: dict[str, Any] = {
            "model": target.model,
            "api_key": target.api_key,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout,
            "max_tokens": max_tokens,
            "num_retries": 1,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if target.provider == "gemini":
            kwargs["reasoning_effort"] = "none"
        try:
            response = litellm_completion(**kwargs)
            logger.info("[LLMRouter] %s succeeded with %s", target.provider, target.model)
            return response, target
        except Exception as exc:
            failures.append(f"{target.provider}: {type(exc).__name__}")
            logger.warning(
                "[LLMRouter] %s failed with %s; trying next provider",
                target.provider,
                type(exc).__name__,
            )

    raise RuntimeError("All extraction providers failed (" + ", ".join(failures) + ")")


async def acompletion_with_fallback(
    *,
    messages: list[dict[str, str]],
    settings: Settings | None = None,
    timeout: int = 45,
    max_tokens: int = 800,
    response_format: dict[str, Any] | None = None,
    temperature: float = 0.0,
) -> tuple[Any, ModelTarget]:
    """Async: Try Cerebras, Mistral, Gemini, Groq, then OpenRouter without failing on one provider outage."""
    settings = settings or get_settings()
    targets = extraction_targets(settings)
    if not targets:
        raise RuntimeError("No extraction provider API key is configured")

    failures = []
    for target in targets:
        kwargs: dict[str, Any] = {
            "model": target.model,
            "api_key": target.api_key,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout,
            "max_tokens": max_tokens,
            "num_retries": 1,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if target.provider == "gemini":
            kwargs["reasoning_effort"] = "none"
        try:
            response = await litellm_acompletion(**kwargs)
            logger.info("[LLMRouter] async %s succeeded with %s", target.provider, target.model)
            return response, target
        except Exception as exc:
            failures.append(f"{target.provider}: {type(exc).__name__}")
            logger.warning(
                "[LLMRouter] async %s failed with %s; trying next provider",
                target.provider,
                type(exc).__name__,
            )

    raise RuntimeError("All async extraction providers failed (" + ", ".join(failures) + ")")
