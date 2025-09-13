from __future__ import annotations

import json
from typing import Any, Dict


SUMMARY_SCHEMA = "{""summary"": str}"


def build_summary(*, sdk, params: Dict[str, Any], per_coder_counts: Dict[str, Dict[str, int]], integrated_counts: Dict[str, int], attempts: int = 1, timeout_s: float = 30.0) -> str:
    system = (
        "You produce a concise academic-style summary of analysis settings and counts. "
        "Keep it clear, neutral, and factual. JSON only."
    )
    user = json.dumps({
        "settings": params,
        "per_coder_counts": per_coder_counts,
        "integrated_counts": integrated_counts,
        "schema": SUMMARY_SCHEMA,
    }, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=SUMMARY_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return data.get("summary", "")

