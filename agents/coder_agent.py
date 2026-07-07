from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

INCIDENT_SCHEMA = "{\"incident_notes\": [{\"transcript\": str, \"segment_number\": int, \"labels\": [str], \"comparison_notes\": [{\"focus\": str, \"similarities\": str, \"differences\": str}], \"analytic_memo\": str}]}"
CATEGORY_SCHEMA = "{\"comparative_categories\": [{\"name\": str, \"defining_properties\": [str], \"comparative_insights\": [str], \"supporting_segments\": [{\"transcript\": str, \"segment_number\": int, \"labels\": [str]}], \"boundary_or_negative_cases\": [{\"transcript\": str, \"segment_number\": int, \"case_summary\": str, \"category_implication\": str}], \"no_boundary_case_reason\": str}]}"
MEMO_SCHEMA = "{\"comparative_memos\": [{\"focus\": str, \"comparisons_made\": [str], \"insights\": str, \"questions\": [str], \"next_steps\": [str]}]}"
SYNTHESIS_SCHEMA = "{\"comparative_summary\": str}"
RECONCILIATION_SCHEMA = "{\"label_clusters\": [{\"name\": str, \"member_labels\": [str], \"representative_segments\": [str], \"comparative_note\": str}], \"global_comparisons\": [{\"transcript\": str, \"segment_number\": int, \"refined_labels\": [str], \"comparison_summary\": str, \"boundary_or_negative_case\": str}], \"audit_note\": str}"


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
    comparison_context: Optional[Dict[str, Any]] = None,
    attempts: int = 2,
    timeout_s: float = 90.0,
) -> List[Dict[str, Any]]:
    system = (
        "You are executing constant comparative incident coding. For each segment, identify analytic incident labels (2–6 words) and articulate how they compare with prior incidents. "
        "Explicitly discuss similarities and differences and capture an analytic memo that justifies any refinements or new distinctions. "
        "When comparison_context is supplied, use recent_prior for continuity, retrieved_prior for full-history analogues or contrasts, and anchor_prior for early anchors or boundary cases. "
        "Do not force a comparison where no meaningful prior case exists; state that the segment is a baseline or new distinction. "
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
            "Produce labelled incidents, comparison notes (similarities and differences versus prior incidents), and a concise analytic memo per segment. "
            "Make visible whether the most important comparison came from recent, retrieved, or anchor prior incidents."
        ),
        "schema": INCIDENT_SCHEMA,
    }
    if comparison_context:
        payload["comparison_context"] = comparison_context
    if transcript_summaries:
        payload["transcript_summaries"] = transcript_summaries
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=INCIDENT_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return data.get("incident_notes") or []


def _clip(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0].rstrip()


def _compact_incident_notes(incident_notes: List[Dict[str, Any]], *, memo_chars: int = 320) -> List[Dict[str, Any]]:
    if incident_notes:
        memo_chars = min(memo_chars, max(120, 90000 // max(1, len(incident_notes))))
    compact: List[Dict[str, Any]] = []
    for note in incident_notes:
        labels = [str(lbl).strip() for lbl in (note.get("labels") or []) if str(lbl).strip()]
        comparison_notes = note.get("comparison_notes") or []
        compact.append(
            {
                "transcript": note.get("transcript"),
                "segment_number": note.get("segment_number"),
                "labels": labels[:8],
                "analytic_memo": _clip(note.get("analytic_memo", ""), memo_chars),
                "comparison_note_count": len(comparison_notes) if isinstance(comparison_notes, list) else 0,
            }
        )
    return compact


def _sample_evenly(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return items[:1]
    step = (len(items) - 1) / (limit - 1)
    indexes: List[int] = []
    for idx in range(limit):
        candidate = round(idx * step)
        if not indexes or candidate != indexes[-1]:
            indexes.append(candidate)
    while len(indexes) < limit:
        for candidate in range(len(items)):
            if candidate not in indexes:
                indexes.append(candidate)
                break
    return [items[idx] for idx in sorted(indexes[:limit])]


def _category_incident_packet(incident_notes: List[Dict[str, Any]], *, limit: int = 320) -> List[Dict[str, Any]]:
    sampled = _sample_evenly(incident_notes, limit)
    memo_chars = 260 if len(sampled) > 180 else 360
    packet = _compact_incident_notes(sampled, memo_chars=memo_chars)
    for item, source in zip(packet, sampled):
        if source.get("refined_labels"):
            item["refined_labels"] = [str(lbl).strip() for lbl in (source.get("refined_labels") or []) if str(lbl).strip()][:8]
        if source.get("label_clusters"):
            item["label_clusters"] = [str(lbl).strip() for lbl in (source.get("label_clusters") or []) if str(lbl).strip()][:8]
        if source.get("global_comparison_summary"):
            item["global_comparison_summary"] = _clip(source.get("global_comparison_summary", ""), 260)
        if source.get("boundary_or_negative_case"):
            item["boundary_or_negative_case"] = _clip(source.get("boundary_or_negative_case", ""), 260)
    return packet


def _iter_batches(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    if size <= 0:
        return [items]
    return [items[i:i + size] for i in range(0, len(items), size)]


def _category_key(cat: Dict[str, Any]) -> str:
    name = re.sub(r"[^a-z0-9]+", " ", str(cat.get("name") or "").lower()).strip()
    if name:
        return name
    props = " ".join(str(item) for item in (cat.get("defining_properties") or [])[:2])
    return re.sub(r"[^a-z0-9]+", " ", props.lower()).strip()


def _dedupe_categories(categories: List[Dict[str, Any]], *, limit: int = 18) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for cat in categories:
        if not isinstance(cat, dict):
            continue
        key = _category_key(cat)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(cat)
        if limit > 0 and len(out) >= limit:
            break
    return out


def _run_category_comparison_once(
    *,
    sdk,
    incident_notes: List[Dict[str, Any]],
    max_categories: int,
    study_background: str,
    analysis_mode: str,
    theoretical_framework: str,
    transcript_summaries: Optional[Dict[str, str]],
    attempts: int,
    timeout_s: float,
    packet_note: str = "",
) -> List[Dict[str, Any]]:
    system = (
        "You are advancing constant comparative analysis from incidents to provisional categories. Group incidents into analytic categories, specify defining properties, and articulate comparative insights that show how categories differ or overlap. "
        "Each category must list the supporting segments (transcript + segment number + labels) that anchor it. "
        "Each category must also list at least one boundary_or_negative_case that complicates, limits, or differentiates the category; if no credible boundary case exists, state no_boundary_case_reason explicitly. "
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
        "incident_notes": _category_incident_packet(incident_notes),
        "transcript_summaries": transcript_summaries or {},
        "max_categories": int(max_categories) if isinstance(max_categories, int) else 0,
        "packet_note": packet_note,
        "instructions": (
            limit_clause
            + " Compare incidents iteratively; name categories, define their properties, and highlight contrasts or boundary cases. "
            + "Do not collapse distinct mechanisms into a broad bucket merely because they all affect the same high-level construct. "
            + "For each category, include boundary_or_negative_cases with transcript and segment_number, or explain why none were found."
        ),
        "schema": CATEGORY_SCHEMA,
    }
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=CATEGORY_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return data.get("comparative_categories") or []


def _integrate_provisional_categories(
    *,
    sdk,
    provisional_categories: List[Dict[str, Any]],
    max_categories: int,
    study_background: str,
    analysis_mode: str,
    theoretical_framework: str,
    transcript_summaries: Optional[Dict[str, str]],
    attempts: int,
    timeout_s: float,
) -> List[Dict[str, Any]]:
    if not provisional_categories:
        return []
    system = (
        "You are consolidating provisional constant-comparative categories produced from batches of the same dataset. "
        "Merge substantively similar categories, preserve distinct mechanisms and boundary cases, and keep concrete transcript/segment support. "
        + _mode_instruction(analysis_mode, "cross-batch category consolidation")
        + "Return ONLY JSON following the supplied schema."
    )
    limit_clause = (
        "Limit the final set to <= max_categories categories if the value is > 0 while ensuring analytic sufficiency. "
        if isinstance(max_categories, int) and max_categories > 0
        else "Prefer a mid-grained final set, often 5-10 categories when the data support that granularity. "
    )
    compact_categories = []
    for idx, cat in enumerate(provisional_categories, start=1):
        compact_categories.append(
            {
                "batch_category_id": idx,
                "name": cat.get("name"),
                "defining_properties": (cat.get("defining_properties") or [])[:8],
                "comparative_insights": (cat.get("comparative_insights") or [])[:8],
                "supporting_segments": (cat.get("supporting_segments") or [])[:10],
                "boundary_or_negative_cases": (cat.get("boundary_or_negative_cases") or [])[:4],
                "no_boundary_case_reason": cat.get("no_boundary_case_reason", ""),
            }
        )
    payload = {
        "study_background": study_background,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "provisional_categories": compact_categories,
        "transcript_summaries": transcript_summaries or {},
        "max_categories": int(max_categories) if isinstance(max_categories, int) else 0,
        "instructions": (
            limit_clause
            + " Consolidate across batches without inventing new domains. Use only supplied provisional categories and their segment references. "
            + "For each final category, carry forward representative supporting_segments and boundary_or_negative_cases."
        ),
        "schema": CATEGORY_SCHEMA,
    }
    data = sdk.run_json(
        system,
        json.dumps(payload, ensure_ascii=False),
        schema_hint=CATEGORY_SCHEMA,
        attempts=attempts,
        timeout_s=timeout_s,
    )
    cats = data.get("comparative_categories") or []
    return _dedupe_categories(cats, limit=max_categories if isinstance(max_categories, int) and max_categories > 0 else 14)


def run_incident_reconciliation(
    *,
    sdk,
    incident_notes: List[Dict[str, Any]],
    study_background: str,
    analysis_mode: str,
    theoretical_framework: str,
    transcript_summaries: Optional[Dict[str, str]] = None,
    attempts: int = 1,
    timeout_s: float = 180.0,
) -> Dict[str, Any]:
    system = (
        "You are conducting a second-pass global constant-comparative reconciliation. "
        "Review first-pass incident notes across the whole dataset, not only recent local context. "
        "Merge semantically similar incident labels into higher-level label clusters, identify early-late and cross-transcript comparisons, and flag boundary or negative cases. "
        "Do not rewrite the analysis from scratch; preserve traceability to transcript and segment identifiers. "
        + _mode_instruction(analysis_mode, "global incident reconciliation")
        + "Return ONLY JSON following the supplied schema."
    )
    payload = {
        "study_background": study_background,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "incident_notes": _compact_incident_notes(incident_notes),
        "transcript_summaries": transcript_summaries or {},
        "instructions": (
            "Create label clusters and global comparisons that make full-dataset constant comparison visible. "
            "global_comparisons should include only segments where a meaningful cross-dataset refinement, contrast, or boundary/negative case is found."
        ),
        "schema": RECONCILIATION_SCHEMA,
    }
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=RECONCILIATION_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return {
        "label_clusters": data.get("label_clusters") or [],
        "global_comparisons": data.get("global_comparisons") or [],
        "audit_note": data.get("audit_note", ""),
    }


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
    if not incident_notes:
        return []
    batch_size = 260
    direct_limit = 360
    if len(incident_notes) <= direct_limit:
        return _run_category_comparison_once(
            sdk=sdk,
            incident_notes=incident_notes,
            max_categories=max_categories,
            study_background=study_background,
            analysis_mode=analysis_mode,
            theoretical_framework=theoretical_framework,
            transcript_summaries=transcript_summaries,
            attempts=attempts,
            timeout_s=timeout_s,
            packet_note=f"Direct category pass over {len(incident_notes)} incident notes.",
        )

    provisional: List[Dict[str, Any]] = []
    batches = _iter_batches(incident_notes, batch_size)
    provisional_limit = 8 if not isinstance(max_categories, int) or max_categories <= 0 else min(max_categories, 8)
    for idx, batch in enumerate(batches, start=1):
        cats = _run_category_comparison_once(
            sdk=sdk,
            incident_notes=batch,
            max_categories=provisional_limit,
            study_background=study_background,
            analysis_mode=analysis_mode,
            theoretical_framework=theoretical_framework,
            transcript_summaries=transcript_summaries,
            attempts=attempts,
            timeout_s=timeout_s,
            packet_note=f"Batch {idx} of {len(batches)}; create provisional categories for this subset only.",
        )
        for cat in cats:
            if isinstance(cat, dict):
                clone = dict(cat)
                clone["source_batch"] = idx
                provisional.append(clone)
    provisional = _dedupe_categories(provisional, limit=36)
    return _integrate_provisional_categories(
        sdk=sdk,
        provisional_categories=provisional,
        max_categories=max_categories,
        study_background=study_background,
        analysis_mode=analysis_mode,
        theoretical_framework=theoretical_framework,
        transcript_summaries=transcript_summaries,
        attempts=attempts,
        timeout_s=timeout_s,
    )


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
        "incident_notes": _category_incident_packet(incident_notes, limit=260),
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
        "incident_notes": _category_incident_packet(incident_notes, limit=220),
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
