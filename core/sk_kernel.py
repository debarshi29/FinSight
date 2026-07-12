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

PLANNER_PROMPT = """You are a financial analysis planning agent.
Decompose the user task into an ordered list of 2-6 specific retrieval subtasks.
Each subtask must be a precise, searchable query targeting a specific financial
metric, time period, or company.

Output ONLY valid JSON — a list of strings. No markdown, no explanation.

Example:
Input: "Compare Infosys and TCS operating margins FY2022-2024"
Output: ["Infosys operating margin FY2022", "Infosys operating margin FY2023",
"Infosys operating margin FY2024", "TCS operating margin FY2022",
"TCS operating margin FY2023", "TCS operating margin FY2024"]

User Task: {user_task}"""

SYNTHESIZER_PROMPT = """You are a financial report writer.
Assemble a structured, professional analysis report from the evidence below.

Query: {query}
Task ID: {task_id}

Verified Claims (include unmarked):
{verified_claims}

Uncertain Claims (prefix each with [UNCERTAIN]):
{uncertain_claims}

Comparative Analysis:
{comparison}

Rules:
- Do NOT include any figure, claim, or statistic absent from the lists above
- Do NOT perform arithmetic, derive new numbers, or infer values — only report what is explicitly stated
- If a company's data is missing or marked N/A, state that it is unavailable rather than estimating
- Every figure must cite its source document and page in parentheses
- Use these exact section headers: ## Executive Summary, ## Key Findings, ## Comparative Analysis, ## Risk Flags
- Use bullet points (- ) for lists of findings
- Mark uncertain claims with [UNCERTAIN] inline
- Write concise professional prose — 3-5 sentences per section
- Output GitHub-flavoured Markdown, NOT JSON and NOT plain text"""

# Legacy SK template aliases kept so nothing importing the old names breaks
_PLANNER_PROMPT = PLANNER_PROMPT
_SYNTHESIZER_PROMPT = SYNTHESIZER_PROMPT

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
