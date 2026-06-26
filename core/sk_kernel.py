from __future__ import annotations

import openai
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from core.config import settings

_kernel: sk.Kernel | None = None


def get_kernel() -> sk.Kernel:
    global _kernel
    if _kernel is None:
        _kernel = _build_kernel()
    return _kernel


def _build_kernel() -> sk.Kernel:
    kernel = sk.Kernel()

    async_client = openai.AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
    )

    chat_service = OpenAIChatCompletion(
        ai_model_id=settings.groq_model,
        async_client=async_client,
    )
    kernel.add_service(chat_service)
    return kernel


def reset_kernel() -> None:
    global _kernel
    _kernel = None
