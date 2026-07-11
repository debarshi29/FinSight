from __future__ import annotations

import asyncio
import logging

import openai

from core.config import settings

_log = logging.getLogger(__name__)

_RETRY_DELAYS = [1.0, 2.0, 4.0]  # seconds between retries on rate-limit


def get_groq_async_client() -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
    )


def _get_fallback_client() -> openai.AsyncOpenAI | None:
    if not settings.fallback_api_key or not settings.fallback_base_url:
        return None
    return openai.AsyncOpenAI(
        api_key=settings.fallback_api_key,
        base_url=settings.fallback_base_url,
    )


async def _call(
    client: openai.AsyncOpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    response_format: dict | None,
) -> str:
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format
    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


async def chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    response_format: dict | None = None,
) -> str:
    # Use the fallback endpoint as primary if configured; Groq is the reserve.
    fallback_client = _get_fallback_client()
    if fallback_client and settings.fallback_model:
        primary_client = fallback_client
        primary_model = settings.fallback_model
        reserve_client: openai.AsyncOpenAI | None = get_groq_async_client()
        reserve_model: str = settings.groq_model
    else:
        primary_client = get_groq_async_client()
        primary_model = settings.groq_model
        reserve_client = None
        reserve_model = ""

    last_exc: Exception | None = None
    for attempt, delay in enumerate([0.0] + _RETRY_DELAYS):
        if delay:
            _log.warning("llm.primary_retry attempt=%d delay=%.1f", attempt, delay)
            await asyncio.sleep(delay)
        try:
            return await _call(
                primary_client, primary_model, messages, temperature, max_tokens, response_format
            )
        except openai.RateLimitError as exc:
            last_exc = exc
        except openai.APIStatusError as exc:
            if exc.status_code == 503:
                last_exc = exc
            else:
                raise

    # Primary exhausted — try reserve
    if reserve_client:
        _log.warning("llm.using_reserve reason=%s", str(last_exc)[:120])
        try:
            return await _call(
                reserve_client, reserve_model, messages, temperature, max_tokens, response_format
            )
        except Exception as exc:
            _log.error("llm.reserve_failed error=%s", str(exc)[:120])
            raise RuntimeError("Both LLM endpoints failed") from exc

    raise RuntimeError(f"LLM failed after {len(_RETRY_DELAYS) + 1} attempts") from last_exc
