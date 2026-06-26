from __future__ import annotations

import openai
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelFunctionFromPrompt

from core.config import settings

# ── Prompt templates registered as SK semantic functions ──────────────────────
# The Planner and Synthesizer are genuine semantic functions: the kernel
# manages prompt rendering, LLM dispatch, and result wrapping — not raw
# chat_completion calls.

_PLANNER_PROMPT = """You are a financial analysis planning agent.
Decompose the user task into an ordered list of 2-6 specific retrieval subtasks.
Each subtask must be a precise, searchable query targeting a specific financial
metric, time period, or company.

Output ONLY valid JSON — a list of strings. No markdown, no explanation.

Example:
Input: "Compare Infosys and TCS operating margins FY2022-2024"
Output: ["Infosys operating margin FY2022", "Infosys operating margin FY2023",
"Infosys operating margin FY2024", "TCS operating margin FY2022",
"TCS operating margin FY2023", "TCS operating margin FY2024"]

User Task: {{$user_task}}"""

_SYNTHESIZER_PROMPT = """You are a financial report writer.
Assemble a structured, professional analysis report from the evidence below.

Query: {{$query}}
Task ID: {{$task_id}}

Verified Claims (include unmarked):
{{$verified_claims}}

Uncertain Claims (prefix each with [UNCERTAIN]):
{{$uncertain_claims}}

Comparative Analysis:
{{$comparison}}

Rules:
- Do NOT include any claim absent from the lists above
- Every figure must reference its source document
- Structure: Executive Summary → Key Findings → Comparative Analysis → Risk Flags
- Write in clear, professional financial analyst prose
- Output plain text, not JSON"""

_kernel: sk.Kernel | None = None


def get_kernel() -> sk.Kernel:
    global _kernel
    if _kernel is None:
        _kernel = _build_kernel()
    return _kernel


def _build_kernel() -> sk.Kernel:
    kernel = sk.Kernel()

    # ── LLM service via Groq's OpenAI-compatible endpoint ────────────────────
    async_client = openai.AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
    )
    chat_service = OpenAIChatCompletion(
        ai_model_id=settings.groq_model,
        async_client=async_client,
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


def reset_kernel() -> None:
    """Reset singleton — useful in tests and after config changes."""
    global _kernel
    _kernel = None
