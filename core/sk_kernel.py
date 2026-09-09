from __future__ import annotations

import openai
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelFunctionFromPrompt

from core.config import settings
from core.prompts import PLANNER_PROMPT, SYNTHESIZER_PROMPT

# ── Prompt templates registered as SK semantic functions ──────────────────────
# The Planner and Synthesizer are genuine semantic functions: the kernel
# manages prompt rendering, LLM dispatch, and result wrapping — not raw
# chat_completion calls. The template text lives in ``core/prompts.py`` so it
# can be shared with framework-agnostic callers; re-exported here for callers
# that still import it from this module.

_PLANNER_PROMPT = PLANNER_PROMPT
_SYNTHESIZER_PROMPT = SYNTHESIZER_PROMPT

__all__ = [
    "PLANNER_PROMPT",
    "SYNTHESIZER_PROMPT",
    "get_kernel",
    "get_fallback_kernel",
    "reset_kernel",
]

_kernel: sk.Kernel | None = None
_fallback_kernel: sk.Kernel | None = None


def get_kernel() -> sk.Kernel:
    global _kernel
    if _kernel is None:
        _kernel = _build_kernel()
    return _kernel


def get_fallback_kernel() -> sk.Kernel | None:
    """Return a reserve kernel (Groq) for when the primary endpoint fails."""
    if not settings.groq_api_key:
        return None
    global _fallback_kernel
    if _fallback_kernel is None:
        _fallback_kernel = _build_kernel(use_fallback=True)
    return _fallback_kernel


def reset_kernel() -> None:
    global _kernel, _fallback_kernel
    _kernel = None
    _fallback_kernel = None


def _build_kernel(use_fallback: bool = False) -> sk.Kernel:
    kernel = sk.Kernel()

    # When use_fallback=False we still prefer the fallback endpoint as primary
    # if it is configured; Groq becomes the reserve kernel (use_fallback=True).
    if use_fallback:
        client = openai.AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )
        model = settings.groq_model
    elif settings.fallback_api_key and settings.fallback_model and settings.fallback_base_url:
        client = openai.AsyncOpenAI(
            api_key=settings.fallback_api_key,
            base_url=settings.fallback_base_url,
        )
        model = settings.fallback_model
    else:
        client = openai.AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )
        model = settings.groq_model

    chat_service = OpenAIChatCompletion(
        ai_model_id=model,
        async_client=client,
    )
    kernel.add_service(chat_service)

    # ── Native plugins — agents registered as SK plugin functions ────────────
    # Imported here to avoid circular imports at module load time.
    from agents.analyst import AnalystPlugin
    from agents.auditor import AuditorPlugin
    from agents.comparator import ComparatorPlugin
    from agents.retriever import RetrieverPlugin

    kernel.add_plugin(RetrieverPlugin(), plugin_name="Retriever")
    kernel.add_plugin(AnalystPlugin(), plugin_name="Analyst")
    kernel.add_plugin(AuditorPlugin(), plugin_name="Auditor")
    kernel.add_plugin(ComparatorPlugin(), plugin_name="Comparator")

    # ── Semantic functions — prompt templates managed by the kernel ───────────
    # PlannerAgent: kernel renders the template, dispatches to Groq, returns result.
    kernel.add_function(
        plugin_name="Planner",
        function=KernelFunctionFromPrompt(
            function_name="decompose",
            prompt=_PLANNER_PROMPT,
        ),
    )

    # SynthesizerAgent: a semantic function per the spec — not a native plugin.
    # The kernel fills in {{$verified_claims}}, {{$uncertain_claims}}, etc.
    # from KernelArguments before dispatching to Groq.
    kernel.add_function(
        plugin_name="Synthesizer",
        function=KernelFunctionFromPrompt(
            function_name="synthesize",
            prompt=_SYNTHESIZER_PROMPT,
        ),
    )

    return kernel
