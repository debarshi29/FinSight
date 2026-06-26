from __future__ import annotations

import json
from typing import Any

import structlog
from semantic_kernel import KernelArguments

from core.models import AuditedClaim
from core.sk_kernel import get_kernel

log = structlog.get_logger()


async def synthesize_report(
    query: str,
    verified_claims: list[AuditedClaim],
    uncertain_claims: list[AuditedClaim],
    comparison: dict[str, Any],
    task_id: str,
) -> str:
    """
    SynthesizerAgent: invokes the Synthesizer.synthesize SK semantic function.

    Per the spec, the Synthesizer is a semantic function — a prompt template
    registered to the kernel via KernelFunctionFromPrompt. The kernel fills in
    {{$verified_claims}}, {{$uncertain_claims}}, {{$comparison}}, {{$query}},
    and {{$task_id}} from KernelArguments before dispatching to Groq.

    Only verified and uncertain claims reach this function — unverifiable claims
    are blocked at the AuditorAgent before this call.
    """
    if not verified_claims and not uncertain_claims:
        return (
            "Insufficient evidence found. No claims could be verified from the "
            "available filings. This may indicate the requested information is "
            "not present in the ingested documents."
        )

    kernel = get_kernel()

    result = await kernel.invoke(
        plugin_name="Synthesizer",
        function_name="synthesize",
        arguments=KernelArguments(
            query=query,
            task_id=task_id,
            verified_claims=json.dumps([c.to_dict() for c in verified_claims], indent=2)[:3000],
            uncertain_claims=json.dumps([c.to_dict() for c in uncertain_claims], indent=2)[:1000],
            comparison=json.dumps(comparison, indent=2)[:1500],
        ),
    )

    return str(result)
