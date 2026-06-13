from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


EVALUATION_SCHEMA = "{\"overall_score\": float, \"criteria\": [{\"name\": str, \"score\": int, \"rationale\": str, \"evidence\": [str], \"recommendations\": [str]}], \"audit_notes\": [str]}"


def _clip(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0].rstrip()


def _compact_incidents(incident_notes: List[Dict[str, Any]], *, limit: int = 120) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    def _add(note: Dict[str, Any]) -> None:
        if len(selected) >= limit:
            return
        try:
            key = (str(note.get("transcript")), int(note.get("segment_number") or 0))
        except Exception:
            key = (str(note.get("transcript")), len(selected))
        if key in seen:
            return
        selected.append(note)
        seen.add(key)

    for note in incident_notes[: max(20, limit // 2)]:
        _add(note)
    for note in incident_notes:
        if note.get("global_comparison_summary") or note.get("boundary_or_negative_case"):
            _add(note)
    for note in incident_notes[-20:]:
        _add(note)

    compact: List[Dict[str, Any]] = []
    for note in selected:
        compact.append(
            {
                "transcript": note.get("transcript"),
                "segment_number": note.get("segment_number"),
                "labels": [str(lbl).strip() for lbl in (note.get("labels") or []) if str(lbl).strip()][:8],
                "comparison_note_count": len(note.get("comparison_notes") or []),
                "has_global_comparison": bool(note.get("global_comparison_summary")),
                "memo": _clip(note.get("analytic_memo", ""), 240),
            }
        )
    return compact


def _compact_categories(categories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for cat in categories:
        compact.append(
            {
                "name": cat.get("name"),
                "defining_properties": cat.get("defining_properties", [])[:8],
                "comparative_insights": cat.get("comparative_insights", [])[:8],
                "supporting_segment_count": len(cat.get("supporting_segments") or []),
                "supporting_quote_count": len(cat.get("supporting_quotes") or []),
                "supporting_quote_evidence_count": len(cat.get("supporting_quote_evidence") or []),
                "boundary_case_count": len(cat.get("boundary_or_negative_cases") or []),
                "no_boundary_case_reason": cat.get("no_boundary_case_reason", ""),
                "quote_audit": cat.get("quote_audit", {}),
            }
        )
    return compact


def _compact_memos(memos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "focus": memo.get("focus"),
            "comparisons_made": memo.get("comparisons_made", [])[:8],
            "insights": _clip(memo.get("insights", ""), 360),
            "questions": memo.get("questions", [])[:6],
            "next_steps": memo.get("next_steps", [])[:6],
        }
        for memo in memos
    ]


def run_reflective_evaluation(
    *,
    sdk,
    coder_id: str,
    incident_notes: List[Dict[str, Any]],
    comparative_categories: List[Dict[str, Any]],
    comparative_memos: List[Dict[str, Any]],
    synthesis: Dict[str, Any],
    reconciliation: Optional[Dict[str, Any]] = None,
    attempts: int = 1,
    timeout_s: float = 150.0,
) -> Dict[str, Any]:
    system = (
        "You are a reflective qualitative-methods evaluation agent for constant comparative analysis. "
        "Audit the analysis quality and coherence without rewriting the analysis. "
        "Score each criterion from 1 to 5, where 1 is poor, 3 is adequate/mixed, and 5 is strong. "
        "Focus on process fidelity, analytic consistency, traceability, comparison coverage, category distinctiveness, and synthesis coherence. "
        "Return ONLY JSON following the supplied schema."
    )
    payload = {
        "coder_id": coder_id,
        "counts": {
            "incident_notes": len(incident_notes),
            "incident_labels": sum(len(note.get("labels") or []) for note in incident_notes),
            "categories": len(comparative_categories),
            "memos": len(comparative_memos),
            "label_clusters": len((reconciliation or {}).get("label_clusters") or []),
            "global_comparisons": len((reconciliation or {}).get("global_comparisons") or []),
        },
        "incident_notes_sample": _compact_incidents(incident_notes),
        "categories": _compact_categories(comparative_categories),
        "memos": _compact_memos(comparative_memos),
        "synthesis": synthesis,
        "reconciliation_summary": {
            "label_clusters": (reconciliation or {}).get("label_clusters", [])[:30],
            "audit_note": (reconciliation or {}).get("audit_note", ""),
        },
        "instructions": (
            "Evaluate the analytic artifact as an audit trail. Emphasize whether claims can be traced to incidents/quotes, "
            "whether quotes are clean and segment-traceable, whether similarities and differences are explicit, whether each category handles boundary/negative cases, "
            "whether categories are coherent and non-redundant, and whether the synthesis follows from incidents, categories, and memos."
        ),
        "schema": EVALUATION_SCHEMA,
    }
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=EVALUATION_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return {
        "overall_score": data.get("overall_score", 0),
        "criteria": data.get("criteria") or [],
        "audit_notes": data.get("audit_notes") or [],
    }


def run_integrated_reflective_evaluation(
    *,
    sdk,
    per_coder_evaluations: List[Dict[str, Any]],
    incident_patterns: Dict[str, Any],
    categories: Dict[str, Any],
    memo_digest: Dict[str, Any],
    summary: Dict[str, Any],
    attempts: int = 1,
    timeout_s: float = 150.0,
) -> Dict[str, Any]:
    system = (
        "You are a reflective evaluation agent auditing cross-coder constant comparative integration. "
        "Assess cross-coder alignment, unresolved divergence, traceability, and coherence of the integrated summary. "
        "Do not rewrite the analysis. Score each criterion from 1 to 5. Return ONLY JSON following the supplied schema."
    )
    payload = {
        "per_coder_evaluations": per_coder_evaluations,
        "integrated_counts": {
            "incident_patterns": len(incident_patterns.get("incident_patterns") or []),
            "categories": len(categories.get("categories") or []),
            "memo_digest": len(memo_digest.get("memo_digest") or []),
            "has_summary": bool((summary.get("comparative_summary") or "").strip()),
        },
        "incident_patterns_sample": (incident_patterns.get("incident_patterns") or [])[:80],
        "categories": categories.get("categories") or [],
        "memo_digest": memo_digest.get("memo_digest") or [],
        "summary": summary,
        "instructions": (
            "Evaluate process fidelity, cross-coder convergence/divergence handling, analytic consistency, traceability, quote cleanliness, boundary/negative-case handling, category granularity, and integrated synthesis coherence."
        ),
        "schema": EVALUATION_SCHEMA,
    }
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=EVALUATION_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return {
        "overall_score": data.get("overall_score", 0),
        "criteria": data.get("criteria") or [],
        "audit_notes": data.get("audit_notes") or [],
    }
