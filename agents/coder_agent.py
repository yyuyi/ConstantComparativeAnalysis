from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


OPEN_SCHEMA = "{""open_codes"": [{""transcript"": str, ""segment_number"": int, ""codes"": [str]}]}"


def run_open_coding(*, sdk, segments: List[Dict[str, Any]], study_background: str, analysis_mode: str, theoretical_framework: str, attempts: int = 2, timeout_s: float = 60.0) -> List[Dict[str, Any]]:
    system = (
        "You are a grounded-theory expert performing high-quality OPEN CODING. "
        "For each segment, produce a reasonable number of concise, human-like codes (2–6 words) that capture actions, meanings, or conditions. "
        "Avoid generic words (e.g., the, and), names (e.g., interviewer), or single words. Prefer short gerund phrases or noun phrases. "
        f"Analysis mode: {analysis_mode}. "
        + ("Apply the theoretical framework consistently in your open coding. " if analysis_mode == "constructionist" else "Apply the theoretical framework suggestively in your open coding, if and only if it deems relevant and useful. " if analysis_mode == "interpretive" else "Use classic grounded-theory (no framework). ")
        + "Return ONLY JSON in the described schema."
    )
    # Compact segment payload
    items = [{"transcript": s.get("transcript"), "segment_number": s.get("segment_number"), "text": s.get("text", "")} for s in segments]
    user = json.dumps({
        "study_background": study_background,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "segments": items,
        "instructions": "2-6 word codes, no stopwords, no interviewer content; JSON only.",
        "schema": OPEN_SCHEMA,
    }, ensure_ascii=False)

    data = sdk.run_json(system, user, schema_hint=OPEN_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return data.get("open_codes") or []


AXIAL_SCHEMA = "{""categories"": [{""name"": str, ""description"": str, ""members"": [{""transcript"": str, ""segment_number"": int}]}]}"


def run_axial_coding(*, sdk, open_codes: List[Dict[str, Any]], max_categories: int, study_background: str, analysis_mode: str, theoretical_framework: str, attempts: int = 2, timeout_s: float = 60.0) -> List[Dict[str, Any]]:
    system = (
        "You are a grounded-theory expert performing AXIAL CODING. "
        "Cluster open codes into ≤ max_categories clear, analytic categories with concise but informative names and full-paragraph descriptions. "
        "Do NOT include supporting quotes; only provide category names, descriptions, and members as {transcript (with extension), segment_number}. "
        f"Analysis mode: {analysis_mode}. "
        + ("Apply the theoretical framework consistently in your axial coding. " if analysis_mode == "constructionist" else "Apply the theoretical framework suggestively in your axial coding, if and only if it deems relevant and useful. " if analysis_mode == "interpretive" else "Use classic grounded-theory (no framework). ")
        + "Return ONLY JSON in the described schema."    
    )
    items = [{
        "id": idx,
        "code": oc.get("code"),
        "transcript": oc.get("transcript"),
        "segment_number": oc.get("segment_number"),
    } for idx, oc in enumerate(open_codes)]
    user = json.dumps({
        "study_background": study_background,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "max_categories": int(max_categories),
        "open_codes": items,
        "schema": AXIAL_SCHEMA,
    }, ensure_ascii=False)

    data = sdk.run_json(system, user, schema_hint=AXIAL_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return data.get("categories") or []


SELECTIVE_SCHEMA = "{""core_story"": str, ""supporting_quotes"": [str]}"


def run_selective_coding(*, sdk, categories: List[Dict[str, Any]], cac_enabled: bool, study_background: str, analysis_mode: str, theoretical_framework: str, attempts: int = 2, timeout_s: float = 60.0) -> Dict[str, Any]:
    system = (
        "You are a grounded-theory expert performing SELECTIVE CODING. "
        "Write a multi-paragraph CORE STORY in a clear academic style that integrates the categories, "
        "providing as much detail as necessary to meet best academic standards. "
        "If CAC is enabled, explicitly structure around Condition-Action-Consequence when possible, "
        "but only apply CAC reasoning when such relationships genuinely exist. "
        "Use only the provided quotes (no new quotes). "
        f"Analysis mode: {analysis_mode}. "
        + ("Apply the theoretical framework consistently in your selective coding. " if analysis_mode == "constructionist" else "Apply the theoretical framework suggestively in your selective coding, if and only if it deems relevant and useful. " if analysis_mode == "interpretive" else "Use classic grounded-theory (no framework). ")
        + "Return ONLY JSON in the described schema. "
        f"CAC enabled: {cac_enabled}. "
    )
    cats = [{"name": c.get("name"), "description": c.get("description"), "supporting_quotes": (c.get("supporting_quotes") or [])[:5]} for c in categories]
    user = json.dumps({
        "study_background": study_background,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "cac_enabled": bool(cac_enabled),
        "categories": cats,
        "schema": SELECTIVE_SCHEMA,
    }, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=SELECTIVE_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return {"core_story": data.get("core_story", ""), "supporting_quotes": data.get("supporting_quotes", [])}
