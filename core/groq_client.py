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
    primary_client = get_groq_async_client()
    primary_model = model or settings.groq_model

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

    # Primary exhausted — try fallback if configured
    fallback_client = _get_fallback_client()
    if fallback_client and settings.fallback_model:
        _log.warning("llm.using_fallback reason=%s", str(last_exc)[:120])
        try:
            return await _call(
                fallback_client,
                settings.fallback_model,
                messages,
                temperature,
                max_tokens,
                response_format,
            )
        except Exception as exc:
            _log.error("llm.fallback_failed error=%s", str(exc)[:120])
            raise RuntimeError("Both primary and fallback LLM failed") from exc

    raise RuntimeError(f"Primary LLM failed after {len(_RETRY_DELAYS) + 1} attempts") from last_exc
