from __future__ import annotations

import asyncio
import logging

import openai

from core.config import settings

_log = logging.getLogger(__name__)

_RETRY_DELAYS = [1.0, 2.0, 4.0]  # seconds between retries on rate-limit

# Singleton clients — created once, reused across all requests.
# Explicit timeouts prevent the default 600-second hang on a slow endpoint.
_primary_client: openai.AsyncOpenAI | None = None
_primary_model: str = ""
_reserve_client: openai.AsyncOpenAI | None = None
_reserve_model: str = ""


def _init_clients() -> None:
    global _primary_client, _primary_model, _reserve_client, _reserve_model

    if settings.fallback_api_key and settings.fallback_model and settings.fallback_base_url:
        # Fallback endpoint is primary; Groq is the reserve.
        _primary_client = openai.AsyncOpenAI(
            api_key=settings.fallback_api_key,
            base_url=settings.fallback_base_url,
            timeout=45.0,
        )
        _primary_model = settings.fallback_model
        _reserve_client = openai.AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            timeout=60.0,
        )
        _reserve_model = settings.groq_model
    else:
        _primary_client = openai.AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            timeout=60.0,
        )
        _primary_model = settings.groq_model
        _reserve_client = None
        _reserve_model = ""


def _get_clients() -> tuple[openai.AsyncOpenAI, str, openai.AsyncOpenAI | None, str]:
    global _primary_client
    if _primary_client is None:
        _init_clients()
    return _primary_client, _primary_model, _reserve_client, _reserve_model  # type: ignore[return-value]


# Kept for backward compatibility — callers that imported these functions directly.
def get_groq_async_client() -> openai.AsyncOpenAI:
    _, _, reserve, _ = _get_clients()
    if reserve:
        return reserve
    primary, _, _, _ = _get_clients()
    return primary


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
    primary_client, primary_model, reserve_client, reserve_model = _get_clients()

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
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            _log.warning("llm.primary_timeout attempt=%d error=%s", attempt, str(exc)[:80])
            last_exc = exc
            break  # don't retry timeouts — go straight to reserve

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


async def chat_completion_hedged(
    messages: list[dict],
    hedge_after: float = 8.0,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    response_format: dict | None = None,
) -> str:
    """Hedged LLM call: fire primary, then if it hasn't responded within
    `hedge_after` seconds, concurrently fire the reserve and take whichever
    wins. Keeps p95 latency at ~hedge_after + winner_latency instead of
    primary_timeout + reserve_latency. Falls back to plain chat_completion
    when no reserve client is configured."""
    primary_client, primary_model, reserve_client, reserve_model = _get_clients()

    if not reserve_client:
        # No reserve configured — behave identically to chat_completion
        return await chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    primary_task = asyncio.create_task(
        _call(primary_client, primary_model, messages, temperature, max_tokens, response_format)
    )

    # Wait up to hedge_after seconds for the primary to finish cleanly
    done, _ = await asyncio.wait({primary_task}, timeout=hedge_after)
    if done and not primary_task.exception():
        return primary_task.result()

    # Primary is slow or already failed — fire the reserve concurrently
    _log.warning("llm.hedging primary_slow=True hedge_after=%.1f", hedge_after)
    reserve_task = asyncio.create_task(
        _call(reserve_client, reserve_model, messages, temperature, max_tokens, response_format)
    )

    done, pending = await asyncio.wait(
        {primary_task, reserve_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for t in pending:
        t.cancel()

    winner = next(iter(done))
    if winner.exception():
        # Winner raised — try the cancelled task's result if it completed
        for t in pending:
            try:
                return await t
            except Exception:
                pass
        raise RuntimeError("Both LLM endpoints failed in hedged call") from winner.exception()
    return winner.result()
