from __future__ import annotations

import openai

from core.config import settings


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
    kwargs = {
        "model": model or settings.groq_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""
