from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

INCIDENT_SCHEMA = "{\"incident_notes\": [{\"transcript\": str, \"segment_number\": int, \"labels\": [str], \"comparison_notes\": [{\"focus\": str, \"similarities\": str, \"differences\": str}], \"analytic_memo\": str}]}"
CATEGORY_SCHEMA = "{\"comparative_categories\": [{\"name\": str, \"defining_properties\": [str], \"comparative_insights\": [str], \"supporting_segments\": [{\"transcript\": str, \"segment_number\": int, \"labels\": [str]}]}]}"
MEMO_SCHEMA = "{\"comparative_memos\": [{\"focus\": str, \"comparisons_made\": [str], \"insights\": str, \"questions\": [str], \"next_steps\": [str]}]}"
SYNTHESIS_SCHEMA = "{\"comparative_summary\": str}"


def _mode_instruction(analysis_mode: str, phase: str) -> str:
    if analysis_mode == "constructionist":
        return f"Apply the provided theoretical framework rigorously during {phase}. "
    if analysis_mode == "interpretive":
        return f"Use the theoretical framework as a sensitizing device during {phase}, but privilege data-driven insights. "
    return f"Conduct {phase} in a classic constant-comparative manner with no imposed framework. "


def run_incident_coding(
    *,
    sdk,
    segments: List[Dict[str, Any]],
    study_background: str,
    analysis_mode: str,
    theoretical_framework: str,
    transcript_summaries: Optional[Dict[str, str]] = None,
    prior_incidents: Optional[List[Dict[str, Any]]] = None,
    attempts: int = 2,
    timeout_s: float = 90.0,
) -> List[Dict[str, Any]]:
    system = (
        "You are executing constant comparative incident coding. For each segment, identify analytic incident labels (2–6 words) and articulate how they compare with prior incidents. "
        "Explicitly discuss similarities and differences and capture an analytic memo that justifies any refinements or new distinctions. "
        + _mode_instruction(analysis_mode, "incident comparison")
        + "Return ONLY JSON following the supplied schema."
    )
    items = [
        {
            "transcript": s.get("transcript"),
            "segment_number": s.get("segment_number"),
            "text": s.get("text", ""),
        }
        for s in segments
    ]
    payload: Dict[str, Any] = {
        "study_background": study_background,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "segments": items,
        "prior_incidents": prior_incidents or [],
        "instructions": (
            "Produce labelled incidents, comparison notes (similarities and differences versus prior incidents), and a concise analytic memo per segment."
        ),
        "schema": INCIDENT_SCHEMA,
    }
    if transcript_summaries:
        payload["transcript_summaries"] = transcript_summaries
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=INCIDENT_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return data.get("incident_notes") or []


def run_category_comparison(
    *,
    sdk,
    incident_notes: List[Dict[str, Any]],
    max_categories: int,
    study_background: str,
    analysis_mode: str,
    theoretical_framework: str,
    transcript_summaries: Optional[Dict[str, str]] = None,
    attempts: int = 2,
    timeout_s: float = 120.0,
) -> List[Dict[str, Any]]:
    system = (
        "You are advancing constant comparative analysis from incidents to provisional categories. Group incidents into analytic categories, specify defining properties, and articulate comparative insights that show how categories differ or overlap. "
        "Each category must list the supporting segments (transcript + segment number + labels) that anchor it. "
        + _mode_instruction(analysis_mode, "category comparison")
        + "Return ONLY JSON following the supplied schema."
    )
    limit_clause = (
        "Limit the final set to <= max_categories categories if the value is > 0 while ensuring analytic sufficiency. "
        if isinstance(max_categories, int) and max_categories > 0
        else "Let the data determine the number of categories; do not force an arbitrary count. "
    )
    payload = {
        "study_background": study_background,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "incident_notes": incident_notes,
        "transcript_summaries": transcript_summaries or {},
        "max_categories": int(max_categories) if isinstance(max_categories, int) else 0,
        "instructions": (
            limit_clause
            + " Compare incidents iteratively; name categories, define their properties, and highlight contrasts or boundary cases."
        ),
        "schema": CATEGORY_SCHEMA,
    }
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=CATEGORY_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return data.get("comparative_categories") or []


def run_comparative_memos(
    *,
    sdk,
    incident_notes: List[Dict[str, Any]],
    comparative_categories: List[Dict[str, Any]],
    study_background: str,
    analysis_mode: str,
    theoretical_framework: str,
    transcript_summaries: Optional[Dict[str, str]] = None,
    attempts: int = 2,
    timeout_s: float = 120.0,
) -> List[Dict[str, Any]]:
    system = (
        "Generate constant-comparative analytic memos. For each memo, indicate the focus (category, property, relationship, or process), list the comparisons considered, synthesize insights, and note remaining questions or sampling needs. "
        + _mode_instruction(analysis_mode, "memoing")
        + "Return ONLY JSON following the supplied schema."
    )
    payload = {
        "study_background": study_background,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "incident_notes": incident_notes,
        "comparative_categories": comparative_categories,
        "transcript_summaries": transcript_summaries or {},
        "instructions": (
            "Produce 3–6 rich memos capturing ongoing comparisons, theoretical leverage, unresolved tensions, and next steps."
        ),
        "schema": MEMO_SCHEMA,
    }
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=MEMO_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return data.get("comparative_memos") or []


def run_cca_synthesis(
    *,
    sdk,
    incident_notes: List[Dict[str, Any]],
    comparative_categories: List[Dict[str, Any]],
    comparative_memos: List[Dict[str, Any]],
    cac_enabled: bool,
    study_background: str,
    analysis_mode: str,
    theoretical_framework: str,
    transcript_summaries: Optional[Dict[str, str]] = None,
    attempts: int = 2,
    timeout_s: float = 150.0,
) -> Dict[str, Any]:
    system = (
        "Integrate the constant comparative analysis into a single concise narrative paragraph. "
        "Explain how the major comparative categories relate, interact, reinforce, or contrast with one another. "
        "If CAC is enabled, weave in any data-supported condition–action–consequence relationships naturally within the same paragraph. "
        + _mode_instruction(analysis_mode, "theoretical integration")
        + "Return ONLY JSON following the supplied schema."
    )
    payload = {
        "study_background": study_background,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "cac_enabled": bool(cac_enabled),
        "incident_notes": incident_notes,
        "comparative_categories": comparative_categories,
        "comparative_memos": comparative_memos,
        "transcript_summaries": transcript_summaries or {},
        "instructions": (
            "Write a single-paragraph comparative summary focusing on interactions, contrasts, or hierarchies among the major categories, grounding each claim in comparative insights."
        ),
        "schema": SYNTHESIS_SCHEMA,
    }
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=SYNTHESIS_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return {"comparative_summary": data.get("comparative_summary", "")}


SUMMARY_SCHEMA = "{\"summary\": str}"
_TOKEN_RE = re.compile(r"\S+")


def _limit_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    tokens = list(_TOKEN_RE.finditer(text))
    if len(tokens) <= max_tokens:
        return text
    cutoff = tokens[max_tokens - 1].end()
    return text[:cutoff].rstrip()


def summarize_transcript(
    *,
    sdk,
    transcript_name: str,
    text: str,
    attempts: int = 1,
    timeout_s: float = 120.0,
    max_tokens: int = 2000,
) -> str:
    system = (
        "You produce a comprehensive multi-paragraph academic-style summary of an interview transcript. "
        "Capture key themes, trajectories, tensions, and contextual nuances. Maintain neutrality and avoid speculation. JSON only."
    )
    user = json.dumps(
        {
            "transcript": transcript_name,
            "text": text,
            "instructions": "Write a comprehensive, multi-paragraph summary suitable for qualitative analysis background.",
            "schema": SUMMARY_SCHEMA,
        },
        ensure_ascii=False,
    )
    data = sdk.run_json(
        system,
        user,
        schema_hint=SUMMARY_SCHEMA,
        attempts=attempts,
        timeout_s=timeout_s,
    )
    summary = data.get("summary", "")
    return _limit_tokens(summary, max_tokens)


REFINE_CTX_SCHEMA = "{\"study_background\": str, \"theoretical_framework\": str}"


def refine_context(
    *,
    sdk,
    study_background: str,
    theoretical_framework: str,
    analysis_mode: str,
    attempts: int = 1,
    timeout_s: float = 45.0,
) -> Dict[str, str]:
    system = (
        "You rewrite provided academic context to optimize clarity, flow, and coherence while preserving meaning. "
        "Return polished text suitable for use in qualitative analysis prompts. JSON only."
    )
    user = json.dumps(
        {
            "analysis_mode": analysis_mode,
            "study_background": study_background or "",
            "theoretical_framework": theoretical_framework or "",
            "instructions": "Polish for clarity and cohesion. Preserve intent and terminology; avoid adding new claims.",
            "schema": REFINE_CTX_SCHEMA,
        },
        ensure_ascii=False,
    )
    data = sdk.run_json(system, user, schema_hint=REFINE_CTX_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    sb = data.get("study_background") or study_background or ""
    tf = data.get("theoretical_framework") or theoretical_framework or ""
    return {"study_background": sb, "theoretical_framework": tf}
