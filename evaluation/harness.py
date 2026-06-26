from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import structlog

log = structlog.get_logger()

API_BASE = "http://localhost:8000"
RESULTS_DIR = Path("evaluation/results")


async def run_query(client: httpx.AsyncClient, query: str) -> dict:
    try:
        response = await client.post(
            f"{API_BASE}/query",
            json={"query": query},
            timeout=120.0,
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def score_result(result: dict, test_case: dict) -> dict:
    if "error" in result:
        return {"passed": False, "reason": result["error"]}

    verified = result.get("verified_claims", [])
    uncertain = result.get("uncertain_claims", [])
    audit_log = result.get("audit_log", {})
    blocked = audit_log.get("blocked_unverifiable", [])

    expected_behavior = test_case.get("expected_behavior", "")
    if expected_behavior in ("unverifiable", "unverifiable_or_insufficient_evidence"):
        passed = len(verified) == 0 or len(blocked) > 0
        reason = f"verified={len(verified)}, blocked={len(blocked)}"
    elif expected_behavior == "uncertain_or_unverifiable":
        passed = len(uncertain) > 0 or len(blocked) > 0
        reason = f"uncertain={len(uncertain)}, blocked={len(blocked)}"
    else:
        passed = len(verified) > 0
        reason = f"verified={len(verified)}"

    return {
        "passed": passed,
        "reason": reason,
        "verified_count": len(verified),
        "uncertain_count": len(uncertain),
        "blocked_count": len(blocked),
        "latency_ms": audit_log.get("latency_ms", 0),
    }


async def run_harness(query_file: str | Path) -> dict:
    query_file = Path(query_file)
    test_cases = json.loads(query_file.read_text())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    passed = 0

    async with httpx.AsyncClient() as client:
        for tc in test_cases:
            log.info("harness.running", id=tc["id"], query=tc["query"][:60])
            result = await run_query(client, tc["query"])
            score = score_result(result, tc)
            if score["passed"]:
                passed += 1

            results.append(
                {
                    "test_id": tc["id"],
                    "query": tc["query"],
                    "score": score,
                    "result_summary": {
                        "verified": len(result.get("verified_claims", [])),
                        "uncertain": len(result.get("uncertain_claims", [])),
                    },
                }
            )
            log.info(
                "harness.result",
                id=tc["id"],
                passed=score["passed"],
                reason=score["reason"],
            )

    summary = {
        "total": len(test_cases),
        "passed": passed,
        "pass_rate": round(passed / len(test_cases), 2) if test_cases else 0,
        "results": results,
    }

    out_path = RESULTS_DIR / f"harness_{int(time.time())}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("harness.complete", pass_rate=summary["pass_rate"], out=str(out_path))
    return summary


if __name__ == "__main__":
    import sys

    query_file = sys.argv[1] if len(sys.argv) > 1 else "evaluation/queries/happy_path.json"
    asyncio.run(run_harness(query_file))
