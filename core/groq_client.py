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


async def chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    response_format: dict | None = None,
) -> str:
    client = get_groq_async_client()
    kwargs: dict = {
        "model": model or settings.groq_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    last_exc: Exception | None = None
    for attempt, delay in enumerate([0.0] + _RETRY_DELAYS):
        if delay:
            _log.warning("groq.rate_limit_retry", attempt=attempt, delay=delay)
            await asyncio.sleep(delay)
        try:
            response = await client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except openai.RateLimitError as exc:
            last_exc = exc
        except openai.APIStatusError as exc:
            if exc.status_code == 503:
                last_exc = exc
            else:
                raise

    raise RuntimeError(f"Groq API failed after {len(_RETRY_DELAYS) + 1} attempts") from last_exc
