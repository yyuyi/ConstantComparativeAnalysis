from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional

SYNTH_INCIDENT_SCHEMA = "{\"incident_patterns\": [{\"label\": str, \"representative_segments\": [str], \"comparative_note\": str}]}"
SYNTH_CATEGORY_SCHEMA = "{\"categories\": [{\"name\": str, \"synthesis_note\": str, \"combined_properties\": [str], \"supporting_segments\": [str], \"boundary_or_negative_cases\": [{\"segment_id\": str, \"case_summary\": str, \"category_implication\": str}], \"no_boundary_case_reason\": str, \"coder_convergence\": str, \"coder_divergence\": str, \"divergence_boundary_impact\": str}], \"granularity_note\": str}"
SYNTH_MEMO_SCHEMA = "{\"memo_digest\": [{\"focus\": str, \"cross_coder_insight\": str, \"unresolved_tensions\": [str]}]}"
SYNTH_SUMMARY_SCHEMA = "{\"comparative_summary\": str}"


_LABEL_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _normalize_label_token(token: str) -> str:
    if token.endswith("er") and len(token) > 5:
        token = token[:-2]
    elif token.endswith("ing") and len(token) > 6:
        token = token[:-3]
    elif token.endswith("s") and len(token) > 4:
        token = token[:-1]
    return token


def _label_tokens(label: str) -> set[str]:
    return {
        _normalize_label_token(tok)
        for tok in _LABEL_TOKEN_RE.findall(label.lower())
        if tok not in _STOPWORDS and len(tok) > 1
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _clip_text(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0].rstrip()


def _segment_ref(row: Dict[str, Any]) -> str:
    return f"{row.get('transcript','')}#{row.get('segment_number','')}"


def _cluster_label_rows(rows: List[Dict[str, Any]], *, threshold: float = 0.4) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    for row in rows:
        label = str(row.get("label") or "").strip()
        tokens = _label_tokens(label)
        if not label or not tokens:
            continue
        best_idx = -1
        best_score = 0.0
        for idx, cluster in enumerate(clusters):
            score = _jaccard(tokens, cluster["_tokens"])
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx >= 0 and best_score >= threshold:
            cluster = clusters[best_idx]
            cluster["_tokens"] |= tokens
            cluster["_labels"].append(label)
            cluster["_coders"].add(row.get("coder"))
            seg = _segment_ref(row)
            if seg not in cluster["representative_segments"]:
                cluster["representative_segments"].append(seg)
            memo = str(row.get("memo") or "").strip()
            if memo and len(cluster["memo_samples"]) < 4:
                cluster["memo_samples"].append(memo[:360])
            cluster["occurrence_count"] += 1
            continue
        clusters.append(
            {
                "_tokens": set(tokens),
                "_labels": [label],
                "_coders": {row.get("coder")},
                "representative_segments": [_segment_ref(row)],
                "memo_samples": [str(row.get("memo") or "").strip()[:360]] if row.get("memo") else [],
                "occurrence_count": 1,
            }
        )

    compact: List[Dict[str, Any]] = []
    for cluster in clusters:
        counts = Counter(cluster["_labels"])
        common_labels = [label for label, _count in counts.most_common(12)]
        compact.append(
            {
                "name_hint": common_labels[0] if common_labels else "",
                "member_labels": common_labels,
                "representative_segments": cluster["representative_segments"][:12],
                "coder_count": len({c for c in cluster["_coders"] if c}),
                "occurrence_count": cluster["occurrence_count"],
                "memo_samples": cluster["memo_samples"][:4],
            }
        )
    compact.sort(key=lambda item: (item.get("coder_count", 0), item.get("occurrence_count", 0)), reverse=True)
    return compact


def _dedupe_patterns(patterns: List[Dict[str, Any]], *, threshold: float = 0.62) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    for pattern in patterns:
        label = str(pattern.get("label") or "").strip()
        tokens = _label_tokens(label)
        if not label or not tokens:
            continue
        merge_idx = -1
        for idx, existing in enumerate(deduped):
            if _jaccard(tokens, _label_tokens(str(existing.get("label") or ""))) >= threshold:
                merge_idx = idx
                break
        if merge_idx < 0:
            deduped.append(pattern)
            continue
        existing = deduped[merge_idx]
        segs = list(existing.get("representative_segments") or [])
        for seg in pattern.get("representative_segments") or []:
            if seg not in segs:
                segs.append(seg)
        existing["representative_segments"] = segs[:12]
        existing["comparative_note"] = " ".join(
            part.strip()
            for part in [
                str(existing.get("comparative_note") or ""),
                str(pattern.get("comparative_note") or ""),
            ]
            if part.strip()
        )
    return deduped


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
    for coder_idx, coder_list in enumerate(per_coder_incidents, start=1):
        for note in coder_list:
            labels = note.get("labels") or []
            for label in labels:
                flat.append(
                    {
                        "coder": f"coder{coder_idx}",
                        "label": str(label),
                        "transcript": note.get("transcript"),
                        "segment_number": note.get("segment_number"),
                        "memo": note.get("analytic_memo", ""),
                    }
                )
    clustered = _cluster_label_rows(flat)
    payload = {
        "analysis_mode": analysis_mode,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "label_clusters": clustered,
        "instructions": (
            "Integrate these deterministic label clusters into higher-level incident patterns. "
            "Merge clusters that are conceptually similar, keep representative segments, identify coder convergence/divergence, "
            "and avoid returning one pattern per label. Prefer <= 60 integrated patterns unless the data clearly require more."
        ),
        "schema": SYNTH_INCIDENT_SCHEMA,
    }
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=SYNTH_INCIDENT_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    patterns = data.get("incident_patterns") or []
    if not patterns:
        patterns = [
            {
                "label": cluster.get("name_hint", ""),
                "representative_segments": cluster.get("representative_segments", []),
                "comparative_note": (
                    f"Deterministic fallback cluster with {cluster.get('occurrence_count', 0)} occurrences "
                    f"across {cluster.get('coder_count', 0)} coder(s). Member labels: "
                    + ", ".join(cluster.get("member_labels", [])[:8])
                ),
            }
            for cluster in clustered
        ]
    patterns = _dedupe_patterns(patterns)
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
        "Prefer a mid-grained category set: usually 5-7 integrated categories when the coder outputs contain differentiated mechanisms. "
        "Do not over-merge distinct patient-experience mechanisms, social determinants, education/health literacy, stigma/support, clinic workflow, and governance/data systems into a few macro buckets unless the evidence truly supports that merge. "
        "For every integrated category, include boundary_or_negative_cases or no_boundary_case_reason, and explain how coder divergence affects the category boundary. "
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
                    "boundary_or_negative_cases": cat.get("boundary_or_negative_cases", []),
                    "no_boundary_case_reason": cat.get("no_boundary_case_reason", ""),
                    "supporting_quote_evidence": cat.get("supporting_quote_evidence", []),
                }
            )
    payload = {
        "analysis_mode": analysis_mode,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "categories": cats_in,
        "input_category_count": len(cats_in),
        "minimum_integrated_categories": min(5, len(cats_in)) if len(cats_in) >= 5 else 0,
        "preferred_integrated_category_range": "5-7 when the input category set contains enough differentiated mechanisms",
        "instructions": (
            "Create an integrated category matrix. Aim for analytic sufficiency rather than compression; prefer 5-7 categories when the inputs contain distinct mechanisms. "
            "If minimum_integrated_categories is greater than 0, return at least that many integrated categories unless the data truly cannot support it; if fewer are returned, explain why in granularity_note. "
            "For each category, preserve concrete supporting segments, name convergence, name divergence, and state whether divergence narrows, splits, or complicates the boundary. "
            "Include at least one boundary_or_negative_case per category when available; otherwise give no_boundary_case_reason."
        ),
        "schema": SYNTH_CATEGORY_SCHEMA,
    }
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=SYNTH_CATEGORY_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    cats = data.get("categories") or []
    note = data.get("granularity_note", "")
    target_min = min(5, len(cats_in)) if len(cats_in) >= 5 else 0
    if target_min and len(cats) < target_min and not note:
        note = (
            f"Granularity warning: {len(cats_in)} coder-level categories were compressed into {len(cats)} integrated categories; "
            f"human review should check whether distinct mechanisms were over-merged."
        )
    return {"categories": cats, "granularity_note": note}


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
    incident_patterns: Optional[Dict[str, Any]] = None,
    integrated_categories: Optional[Dict[str, Any]] = None,
    attempts: int = 2,
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    system = (
        "Integrate the coders' comparative summaries into a single unified paragraph. "
        "Highlight the key relationships or contrasts among the major categories without introducing bullet lists. "
        "Use integrated categories and incident patterns when provided so the summary does not collapse distinct mechanisms into vague macro-themes. "
        "Return ONLY JSON per schema."
    )
    payload = {
        "analysis_mode": analysis_mode,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "syntheses": per_coder_syntheses,
        "incident_patterns": incident_patterns or {},
        "integrated_categories": integrated_categories or {},
        "instructions": (
            "Write one coherent comparative paragraph. Preserve specific mechanisms and variation rather than over-aggregating. "
            "When divergence or boundary cases change interpretation, include that nuance briefly."
        ),
        "schema": SYNTH_SUMMARY_SCHEMA,
    }
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=SYNTH_SUMMARY_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    return {"comparative_summary": data.get("comparative_summary", "")}
