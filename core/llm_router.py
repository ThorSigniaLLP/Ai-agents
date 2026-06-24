"""Provider-aware LiteLLM routing for structured company extraction."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from litellm import completion as litellm_completion
from litellm import acompletion as litellm_acompletion
from litellm.exceptions import RateLimitError

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# How long to wait (seconds) between retries on a rate-limited provider
_RATE_LIMIT_RETRY_DELAYS = [5, 15, 30]  # 3 retries: 5s → 15s → 30s


@dataclass(frozen=True)
class ModelTarget:
    provider: str
    model: str
    api_key: str


def extraction_targets(settings: Settings | None = None) -> list[ModelTarget]:
    """Return configured models in primary-to-fallback order."""
    settings = settings or get_settings()
    candidates = [
        ModelTarget("mistral", settings.primary_model, settings.mistral_api_key),
        ModelTarget("cerebras", settings.fast_model, settings.cerebras_api_key),
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
    """Try providers in order. On rate-limit, wait and retry the same provider before falling back."""
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
            "num_retries": 0,  # We handle retries ourselves below
        }
        if response_format:
            kwargs["response_format"] = response_format
        if target.provider == "gemini":
            kwargs["reasoning_effort"] = "none"

        # Attempt with rate-limit retries on the same provider
        for attempt, delay in enumerate([0] + _RATE_LIMIT_RETRY_DELAYS):
            if delay > 0:
                logger.info(
                    "[LLMRouter] %s rate-limited — waiting %ds before retry %d/%d",
                    target.provider, delay, attempt, len(_RATE_LIMIT_RETRY_DELAYS),
                )
                time.sleep(delay)
            try:
                response = litellm_completion(**kwargs)
                logger.info("[LLMRouter] %s succeeded with %s", target.provider, target.model)
                return response, target
            except RateLimitError:
                if attempt < len(_RATE_LIMIT_RETRY_DELAYS):
                    continue  # Will retry with backoff
                # All retries exhausted — fall through to next provider
                logger.warning(
                    "[LLMRouter] %s exhausted all rate-limit retries; trying next provider",
                    target.provider,
                )
                failures.append(f"{target.provider}: RateLimitError (retried {len(_RATE_LIMIT_RETRY_DELAYS)}x)")
                break
            except Exception as exc:
                # Non-rate-limit error: fall through immediately
                failures.append(f"{target.provider}: {type(exc).__name__}")
                logger.warning(
                    "[LLMRouter] %s failed with %s; trying next provider",
                    target.provider,
                    type(exc).__name__,
                )
                break

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
    """Async: Try providers in order. On rate-limit, wait and retry the same provider before falling back."""
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
            "num_retries": 0,  # We handle retries ourselves below
        }
        if response_format:
            kwargs["response_format"] = response_format
        if target.provider == "gemini":
            kwargs["reasoning_effort"] = "none"

        # Attempt with rate-limit retries on the same provider
        for attempt, delay in enumerate([0] + _RATE_LIMIT_RETRY_DELAYS):
            if delay > 0:
                logger.info(
                    "[LLMRouter] async %s rate-limited — waiting %ds before retry %d/%d",
                    target.provider, delay, attempt, len(_RATE_LIMIT_RETRY_DELAYS),
                )
                await asyncio.sleep(delay)
            try:
                response = await litellm_acompletion(**kwargs)
                logger.info("[LLMRouter] async %s succeeded with %s", target.provider, target.model)
                return response, target
            except RateLimitError:
                if attempt < len(_RATE_LIMIT_RETRY_DELAYS):
                    continue  # Will retry with backoff
                # All retries exhausted — fall through to next provider
                logger.warning(
                    "[LLMRouter] async %s exhausted all rate-limit retries; trying next provider",
                    target.provider,
                )
                failures.append(f"{target.provider}: RateLimitError (retried {len(_RATE_LIMIT_RETRY_DELAYS)}x)")
                break
            except Exception as exc:
                # Non-rate-limit error: fall through immediately
                failures.append(f"{target.provider}: {type(exc).__name__}")
                logger.warning(
                    "[LLMRouter] async %s failed with %s; trying next provider",
                    target.provider,
                    type(exc).__name__,
                )
                break

    raise RuntimeError("All async extraction providers failed (" + ", ".join(failures) + ")")
