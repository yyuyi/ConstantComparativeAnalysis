from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple, Optional


# Open coding returns codes only (no quotes).
OPEN_SCHEMA = "{\"open_codes\": [{\"transcript\": str, \"segment_number\": int, \"codes\": [str]}]}"


def run_open_coding(
    *,
    sdk,
    segments: List[Dict[str, Any]],
    study_background: str,
    analysis_mode: str,
    theoretical_framework: str,
    transcript_summaries: Optional[Dict[str, str]] = None,
    attempts: int = 2,
    timeout_s: float = 60.0,
) -> List[Dict[str, Any]]:
    system = (
        "You are a grounded-theory expert performing high-quality OPEN CODING. "
        "For each segment, produce human-like codes (2–6 words) that capture actions, meanings, or conditions. "
        "You are not limited in the number of codes you create—generate as many as needed to achieve thorough coverage of the concepts in each segment—but ensure each code is unique and avoid duplicates. "
        "Avoid generic words (e.g., the, and), names (e.g., interviewer), or single words. Prefer short gerund phrases or noun phrases. "
        "Return only the codes; do NOT provide supporting quotes or commentary. "
        "Prefer interviewee speech; exclude interviewer prompts. "
        f"Analysis mode: {analysis_mode}. "
        + ("Apply the theoretical framework consistently in your open coding. " if analysis_mode == "constructionist" else "Apply the theoretical framework suggestively in your open coding, if and only if it deems relevant and useful. " if analysis_mode == "interpretive" else "Use classic grounded-theory (no framework). ")
        + "Return ONLY JSON in the described schema."
    )
    # Compact segment payload
    items = [
        {"transcript": s.get("transcript"), "segment_number": s.get("segment_number"), "text": s.get("text", "")} for s in segments
    ]
    payload = {
        "study_background": study_background,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "segments": items,
        "instructions": "2-6 word codes per segment; no quotes or explanations; JSON only.",
        "schema": OPEN_SCHEMA,
    }
    if transcript_summaries:
        payload["transcript_summaries"] = transcript_summaries
    user = json.dumps(payload, ensure_ascii=False)

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
        "Write a multi-paragraph CORE STORY in a clear academic style. "
        "A CORE STORY is an emergent theory: a coherent, explanatory account that integrates "
        "all major categories and subcategories derived from OPEN and AXIAL coding "
        "The account must be entirely grounded in the coded transcripts—do not import external "
        "concepts, literature, or top-down assumptions. "
        "There are only two exceptions: One is when the user enables the CAC option, in which "
        "case the CORE STORY will consider potential condition-action-control relationships between the categories. "
        "The other is when the user selects the interpretive or constructionist analysis mode, in which case "
        "the CORE STORY will consider the theoretical framework provided by the user. "
        "Requirements: "
        "1. Identify the CORE CATEGORY (the central organizing idea) and explain why it is core "
        "(e.g., its reach, explanatory power, or frequency across participants), using evidence "
        "drawn from the coded data. "
        "2. Integrate all relevant categories by clarifying their relationships "
        "to the core category (e.g., association, contrast, sequence, enabling/limiting relation), "
        "without forcing any specific schema. "
        "3. Support every major analytic claim with data-based evidence from OPEN CODES and "
        "the categories identified in AXIAL coding. "
        "Include negative/deviant or contrasting cases when they refine or bound the theory. "
        "3. Maintain analytic abstraction: name concepts clearly, but trace each abstraction back "
        "to the underlying categories and open codes that warrant it. "
        "4. Specify scope conditions and boundaries of the theory as indicated by the data "
        "(e.g., when, where, for whom it seems to hold), and note important variations. "
        "5. Do not introduce new categories that were not previously identified; if a higher-level "
        "integration label is proposed, explicitly link it to existing categories/open codes. "
        "6. Keep the tone precise and non-speculative; avoid claims not supported by the coded data. "
        "7. Length: Use as much detail as necessary for a rigorous, cohesive theoretical narrative. "
        "8. Output: prose only (no bullet lists unless used briefly for clarity). "              
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
_TOKEN_RE = re.compile(r"\S+")


def _limit_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    tokens = list(_TOKEN_RE.finditer(text))
    if len(tokens) <= max_tokens:
        return text
    cutoff = tokens[max_tokens - 1].end()
    return text[:cutoff].rstrip()


def summarize_transcript(*, sdk, transcript_name: str, text: str, attempts: int = 1, timeout_s: float = 120.0, max_tokens: int = 2000) -> str:
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
