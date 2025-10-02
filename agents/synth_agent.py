from __future__ import annotations

import json
from typing import Any, Dict, List

SYNTH_INCIDENT_SCHEMA = "{\"incident_patterns\": [{\"label\": str, \"representative_segments\": [str], \"comparative_note\": str}]}"
SYNTH_CATEGORY_SCHEMA = "{\"categories\": [{\"name\": str, \"synthesis_note\": str, \"combined_properties\": [str], \"supporting_segments\": [str]}]}"
SYNTH_MEMO_SCHEMA = "{\"memo_digest\": [{\"focus\": str, \"cross_coder_insight\": str, \"unresolved_tensions\": [str]}]}"
SYNTH_SUMMARY_SCHEMA = "{\"comparative_summary\": str}"


def synthesize_incident_patterns(
    *,
    sdk,
    per_coder_incidents: List[List[Dict[str, Any]]],
    analysis_mode: str = "classic",
    theoretical_framework: str = "",
    attempts: int = 2,
    timeout_s: float = 90.0,
) -> Dict[str, Any]:
    system = (
        "Integrate incident-level constant comparative findings across coders. Identify convergent labels, note where comparisons diverge, and capture a comparative note that links back to data. "
        "Return ONLY JSON per schema."
    )
    flat: List[Dict[str, Any]] = []
    for coder_list in per_coder_incidents:
        for note in coder_list:
            labels = note.get("labels") or []
            for label in labels:
                flat.append(
                    {
                        "label": str(label),
                        "transcript": note.get("transcript"),
                        "segment_number": note.get("segment_number"),
                        "memo": note.get("analytic_memo", ""),
                    }
                )
    payload = {
        "analysis_mode": analysis_mode,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "incident_notes": flat,
        "schema": SYNTH_INCIDENT_SCHEMA,
    }
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=SYNTH_INCIDENT_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    patterns = data.get("incident_patterns") or []
    if not patterns:
        # simple fallback: deduplicate labels
        seen: Dict[str, Dict[str, Any]] = {}
        for row in flat:
            key = f"{row.get('label','').strip().lower()}"
            if not key:
                continue
            entry = seen.setdefault(
                key,
                {
                    "label": row.get("label", ""),
                    "representative_segments": [],
                    "comparative_note": row.get("memo", ""),
                },
            )
            seg = f"{row.get('transcript','')}#{row.get('segment_number','')}"
            if seg not in entry["representative_segments"]:
                entry["representative_segments"].append(seg)
        patterns = list(seen.values())
    return {"incident_patterns": patterns}


def synthesize_category_matrix(
    *,
    sdk,
    per_coder_categories: List[List[Dict[str, Any]]],
    analysis_mode: str = "classic",
    theoretical_framework: str = "",
    attempts: int = 2,
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    system = (
        "Integrate comparative categories across coders. Merge similar categories, describe combined properties, and highlight how coders' comparisons align or differ. "
        "Return ONLY JSON per schema."
    )
    cats_in: List[Dict[str, Any]] = []
    for clist in per_coder_categories:
        for cat in clist:
            cats_in.append(
                {
                    "name": cat.get("name"),
                    "defining_properties": cat.get("defining_properties", []),
                    "comparative_insights": cat.get("comparative_insights", []),
                    "supporting_segments": cat.get("supporting_segments", []),
                }
            )
    payload = {
        "analysis_mode": analysis_mode,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "categories": cats_in,
        "schema": SYNTH_CATEGORY_SCHEMA,
    }
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=SYNTH_CATEGORY_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    cats = data.get("categories") or []
    return {"categories": cats}


def synthesize_memo_digest(
    *,
    sdk,
    per_coder_memos: List[List[Dict[str, Any]]],
    analysis_mode: str = "classic",
    theoretical_framework: str = "",
    attempts: int = 2,
    timeout_s: float = 90.0,
) -> Dict[str, Any]:
    system = (
        "Combine comparative memos from all coders. Surface shared insights, tensions, and unanswered questions that should inform further comparison or sampling. "
        "Return ONLY JSON per schema."
    )
    memos_in: List[Dict[str, Any]] = []
    for mlist in per_coder_memos:
        for memo in mlist:
            memos_in.append(
                {
                    "focus": memo.get("focus"),
                    "insights": memo.get("insights"),
                    "comparisons_made": memo.get("comparisons_made", []),
                    "questions": memo.get("questions", []),
                    "next_steps": memo.get("next_steps", []),
                }
            )
    payload = {
        "analysis_mode": analysis_mode,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "memos": memos_in,
        "schema": SYNTH_MEMO_SCHEMA,
    }
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=SYNTH_MEMO_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    digest = data.get("memo_digest") or []
    return {"memo_digest": digest}


def synthesize_cca_summary(
    *,
    sdk,
    per_coder_syntheses: List[Dict[str, Any]],
    analysis_mode: str = "classic",
    theoretical_framework: str = "",
    attempts: int = 2,
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    system = (
        "Integrate the coders' comparative summaries into a single unified paragraph. "
        "Highlight the key relationships or contrasts among the major categories without introducing bullet lists. "
        "Return ONLY JSON per schema."
    )
    payload = {
        "analysis_mode": analysis_mode,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "syntheses": per_coder_syntheses,
        "schema": SYNTH_SUMMARY_SCHEMA,
    }
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=SYNTH_SUMMARY_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return {"comparative_summary": data.get("comparative_summary", "")}
