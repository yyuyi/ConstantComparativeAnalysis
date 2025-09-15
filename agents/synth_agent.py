from __future__ import annotations

import json
from typing import Any, Dict, List


SYNTH_OPEN_SCHEMA = "{""open_codes"": [str]}"
SYNTH_CATS_SCHEMA = "{""categories"": [{""name"": str, ""description"": str, ""supporting_quotes"": [str]}]}"
SYNTH_CORE_SCHEMA = "{""core_story"": str}"


def synthesize_open_codes(*, sdk, per_coder_codes: List[List[Dict[str, Any]]], analysis_mode: str = "classic", theoretical_framework: str = "", attempts: int = 2, timeout_s: float = 60.0) -> Dict[str, Any]:
    """Integrate open codes across coders into a simple list of strings.

    Returns { "open_codes": ["integrated open code", ...] }
    """
    system = (
        "You are an experienced grounded-theory expert integrating OPEN CODES across multiple coders. "
        "Carefully examine all open codes, integrate them, remove duplicates, and merge semantically similar codes into distinct human-like phrases (2–6 words each). "
        'Return ONLY JSON: {"open_codes": [str]}.'
    )
    flat: List[str] = []
    for coder_list in per_coder_codes:
        for oc in coder_list:
            code = str(oc.get("code", "")).strip()
            if code:
                flat.append(code)
    user = json.dumps({
        "open_codes": flat,
        "schema": SYNTH_OPEN_SCHEMA,
    }, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=SYNTH_OPEN_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    open_codes = data.get("open_codes") or []
    if not open_codes:
        # Fallback: case-insensitive dedup + sort
        seen = {}
        for c in flat:
            key = c.lower()
            if key not in seen:
                seen[key] = c
        open_codes = sorted(seen.values(), key=lambda x: x.lower())
    return {"open_codes": open_codes}


def synthesize_categories(*, sdk, per_coder_categories: List[List[Dict[str, Any]]], analysis_mode: str = "classic", theoretical_framework: str = "", attempts: int = 2, timeout_s: float = 60.0) -> List[Dict[str, Any]]:
    system = (
        "You are an experienced grounded-theory expert integrating categories across multiple coders. "
        "Comprehensively examine all categories, remove duplicates, integrate similar ones into a single category, and assign concise, informative names. "
        "Ensure the final set of categories is unique, distinct, and faithfully represents the original inputs. "
        "Provide a full paragraph description for each category and include only the supporting quotes drawn directly from the inputs. "
        f"Analysis mode: {analysis_mode}. "
        + ("Apply the theoretical framework consistently in your open coding. " if analysis_mode == "constructionist" else "Apply the theoretical framework suggestively in your open coding, if and only if it deems relevant and useful. " if analysis_mode == "interpretive" else "Use classic grounded-theory (no framework). ")
        + "Return ONLY JSON in the described schema."
    )
    cats_in = []
    for clist in per_coder_categories:
        for c in clist:
            cats_in.append({
                "name": c.get("name"),
                "description": c.get("description"),
                "supporting_quotes": (c.get("supporting_quotes") or [])[:5],
            })
    user = json.dumps({"categories": cats_in, "schema": SYNTH_CATS_SCHEMA}, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=SYNTH_CATS_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return data.get("categories") or []


def synthesize_core_story(*, sdk, per_coder_stories: List[Dict[str, Any]], cac_enabled: bool, analysis_mode: str = "classic", theoretical_framework: str = "", attempts: int = 2, timeout_s: float = 60.0) -> Dict[str, Any]:
    system = (
        "You are tasked with synthesizing core stories from all coders into a single integrated CORE STORY. "
        "The CORE STORY must be multi-paragraph, coherent, and written in a clear academic style, providing as much detail as necessary to meet best academic standards. "
        "Carefully examine the core stories from all coders and integrate them into a unified, logically organized narrative. "
        "If CAC is enabled, examine potential Condition-Action-Consequence relationships between categories, but apply CAC reasoning only when such relationships genuinely exist. "
        f"Analysis mode: {analysis_mode}. "
        + ("Apply the theoretical framework consistently in your selective coding. " if analysis_mode == "constructionist" else "Apply the theoretical framework suggestively in your selective coding, if and only if it deems relevant and useful. " if analysis_mode == "interpretive" else "Use classic grounded-theory (no framework). ")
        + "Return ONLY JSON in the described schema. "
        f"CAC enabled: {cac_enabled}. "
    )
    user = json.dumps({
        "cac_enabled": bool(cac_enabled),
        "analysis_mode": analysis_mode,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "stories": [s.get("core_story", "") for s in per_coder_stories],
        "schema": SYNTH_CORE_SCHEMA,
    }, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=SYNTH_CORE_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return {"core_story": data.get("core_story", "")}
