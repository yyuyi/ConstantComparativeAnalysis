from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Optional


# Open coding now requests character spans for quotes per code; we will slice the exact substring server-side.
OPEN_SCHEMA = (
    "{\"open_codes\": ["
    "{\"transcript\": str, \"segment_number\": int, \"codes\": ["
    "{\"code\": str, \"span\": {\"start\": int, \"end\": int}}]}]}"
)


def run_open_coding(*, sdk, segments: List[Dict[str, Any]], study_background: str, analysis_mode: str, theoretical_framework: str, attempts: int = 2, timeout_s: float = 60.0) -> List[Dict[str, Any]]:
    system = (
        "You are a grounded-theory expert performing high-quality OPEN CODING. "
        "For each segment, produce a reasonable number of concise, human-like codes (2–6 words) that capture actions, meanings, or conditions. "
        "Avoid generic words (e.g., the, and), names (e.g., interviewer), or single words. Prefer short gerund phrases or noun phrases. "
        "For each code, also return a SPAN specifying a VERBATIM substring (1–3 sentences) of the provided segment text that best supports the code. "
        "Return SPAN as character indices relative to the EXACT segment text provided: 0-based start (inclusive) and end (exclusive). "
        "Strict rules: choose a contiguous substring; do NOT paraphrase; do NOT add ellipses, brackets, or annotations; do NOT include leading or trailing quotation marks. "
        "Prefer interviewee speech; exclude interviewer prompts. "
        f"Analysis mode: {analysis_mode}. "
        + ("Apply the theoretical framework consistently in your open coding. " if analysis_mode == "constructionist" else "Apply the theoretical framework suggestively in your open coding, if and only if it deems relevant and useful. " if analysis_mode == "interpretive" else "Use classic grounded-theory (no framework). ")
        + "Return ONLY JSON in the described schema."
    )
    # Compact segment payload
    items = [
        {"transcript": s.get("transcript"), "segment_number": s.get("segment_number"), "text": s.get("text", "")} for s in segments
    ]
    user = json.dumps(
        {
            "study_background": study_background,
            "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
            "segments": items,
            "instructions": "2-6 word codes; provide a span {start,end} per code (0-based, end exclusive) referencing the segment text; prefer interviewee speech; JSON only.",
            "schema": OPEN_SCHEMA,
        },
        ensure_ascii=False,
    )

    data = sdk.run_json(system, user, schema_hint=OPEN_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return data.get("open_codes") or []


AXIAL_SCHEMA = "{""categories"": [{""name"": str, ""description"": str, ""members"": [{""transcript"": str, ""segment_number"": int}]}]}"


def run_axial_coding(
    *,
    sdk,
    open_codes: List[Dict[str, Any]],
    max_categories: int,
    study_background: str,
    analysis_mode: str,
    theoretical_framework: str,
    transcript_summaries: Optional[Dict[str, str]] = None,
    attempts: int = 2,
    timeout_s: float = 60.0,
) -> List[Dict[str, Any]]:
    # Guidance about max categories depends on whether a limit is specified (>0) or auto (<=0)
    limit_clause = (
        "Cluster open codes into ≤ max_categories clear, analytic categories with concise but informative names and full-paragraph descriptions. "
        if (isinstance(max_categories, int) and max_categories > 0)
        else "Determine a reasonable number of distinct categories solely driven by the data; do not force a fixed count. Provide concise but informative names and full-paragraph descriptions. "
    )
    system = (
        "You are a grounded-theory expert performing AXIAL CODING. "
        + limit_clause
        + "You may use the provided per-code sample quotes and transcript summaries AS BACKGROUND to improve clustering quality and naming, but DO NOT include quotes in the output. "
        + "Do NOT include supporting quotes; only provide category names, descriptions, and members as {transcript (with extension), segment_number}. "
        + f"Analysis mode: {analysis_mode}. "
        + ("Apply the theoretical framework consistently in your axial coding. " if analysis_mode == "constructionist" else "Apply the theoretical framework suggestively in your axial coding, if and only if it deems relevant and useful. " if analysis_mode == "interpretive" else "Use classic grounded-theory (no framework). ")
        + "Return ONLY JSON in the described schema."
    )
    items = [
        {
            "id": idx,
            "code": oc.get("code"),
            "transcript": oc.get("transcript"),
            "segment_number": oc.get("segment_number"),
            # Include optional micro-context if present (sample_quote), but schema remains the same on output
            "sample_quote": oc.get("sample_quote"),
        }
        for idx, oc in enumerate(open_codes)
    ]
    user = json.dumps(
        {
            "study_background": study_background,
            "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
            "max_categories": int(max_categories) if isinstance(max_categories, int) else 0,
            "open_codes": items,
            "transcript_summaries": transcript_summaries or {},
            "schema": AXIAL_SCHEMA,
        },
        ensure_ascii=False,
    )

    data = sdk.run_json(system, user, schema_hint=AXIAL_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return data.get("categories") or []


SELECTIVE_SCHEMA = "{""core_story"": str, ""supporting_quotes"": [str]}"


def run_selective_coding(
    *,
    sdk,
    categories: List[Dict[str, Any]],
    cac_enabled: bool,
    study_background: str,
    analysis_mode: str,
    theoretical_framework: str,
    transcript_summaries: Optional[Dict[str, str]] = None,
    attempts: int = 2,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    system = (
        "You are a grounded-theory expert performing SELECTIVE CODING. "
        "Write a multi-paragraph CORE STORY in a clear academic style that integrates the categories, "
        "providing as much detail as necessary to meet best academic standards. "
        "If CAC is enabled, explicitly structure around Condition-Action-Consequence when possible, "
        "but only apply CAC reasoning when such relationships genuinely exist. "
        "Use only the provided quotes (no new quotes). "
        "You may also use the provided transcript summaries as background context to ground the narrative. "
        f"Analysis mode: {analysis_mode}. "
        + ("Apply the theoretical framework consistently in your selective coding. " if analysis_mode == "constructionist" else "Apply the theoretical framework suggestively in your selective coding, if and only if it deems relevant and useful. " if analysis_mode == "interpretive" else "Use classic grounded-theory (no framework). ")
        + "Return ONLY JSON in the described schema. "
        f"CAC enabled: {cac_enabled}. "
    )
    cats = [
        {"name": c.get("name"), "description": c.get("description"), "supporting_quotes": (c.get("supporting_quotes") or [])[:5]}
        for c in categories
    ]
    user = json.dumps(
        {
            "study_background": study_background,
            "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
            "cac_enabled": bool(cac_enabled),
            "categories": cats,
            "transcript_summaries": transcript_summaries or {},
            "schema": SELECTIVE_SCHEMA,
        },
        ensure_ascii=False,
    )
    data = sdk.run_json(system, user, schema_hint=SELECTIVE_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return {"core_story": data.get("core_story", ""), "supporting_quotes": data.get("supporting_quotes", [])}


# Comprehensive transcript summarization used as background for axial and selective phases
SUMMARY_SCHEMA = "{""summary"": str}"


def summarize_transcript(*, sdk, transcript_name: str, text: str, attempts: int = 1, timeout_s: float = 120.0) -> str:
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
    data = sdk.run_json(system, user, schema_hint=SUMMARY_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return data.get("summary", "")


# Refinement of study background and theoretical framework to optimize clarity and flow
REFINE_CTX_SCHEMA = "{""study_background"": str, ""theoretical_framework"": str}"


def refine_context(*, sdk, study_background: str, theoretical_framework: str, analysis_mode: str, attempts: int = 1, timeout_s: float = 45.0) -> Dict[str, str]:
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
