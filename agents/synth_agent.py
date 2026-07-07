from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional

SYNTH_INCIDENT_SCHEMA = "{\"incident_patterns\": [{\"label\": str, \"representative_segments\": [str], \"comparative_note\": str, \"coder_convergence\": str, \"coder_divergence\": str, \"non_local_reconciliation\": str}]}"
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
            cluster["_transcripts"].add(str(row.get("transcript") or ""))
            seg = _segment_ref(row)
            if seg not in cluster["representative_segments"]:
                cluster["representative_segments"].append(seg)
            memo = str(row.get("memo") or "").strip()
            if memo and len(cluster["memo_samples"]) < 4:
                cluster["memo_samples"].append(memo[:360])
            comparison = str(row.get("comparison") or "").strip()
            if comparison and len(cluster["comparison_samples"]) < 5:
                cluster["comparison_samples"].append(comparison[:420])
            cluster["occurrence_count"] += 1
            continue
        clusters.append(
            {
                "_tokens": set(tokens),
                "_labels": [label],
                "_coders": {row.get("coder")},
                "_transcripts": {str(row.get("transcript") or "")},
                "representative_segments": [_segment_ref(row)],
                "memo_samples": [str(row.get("memo") or "").strip()[:360]] if row.get("memo") else [],
                "comparison_samples": [str(row.get("comparison") or "").strip()[:420]] if row.get("comparison") else [],
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
                "transcript_count": len({t for t in cluster["_transcripts"] if t}),
                "occurrence_count": cluster["occurrence_count"],
                "memo_samples": cluster["memo_samples"][:4],
                "comparison_samples": cluster["comparison_samples"][:5],
            }
        )
    compact.sort(
        key=lambda item: (
            item.get("coder_count", 0),
            item.get("transcript_count", 0),
            item.get("occurrence_count", 0),
        ),
        reverse=True,
    )
    return compact


def _cluster_tokens(cluster: Dict[str, Any]) -> set[str]:
    labels = [str(cluster.get("name_hint") or "")]
    labels.extend(str(label) for label in (cluster.get("member_labels") or [])[:8])
    return set().union(*(_label_tokens(label) for label in labels if label))


def _merge_cluster_into_pattern(pattern: Dict[str, Any], cluster: Dict[str, Any]) -> None:
    pattern["_tokens"] |= _cluster_tokens(cluster)
    pattern["_member_labels"].extend(str(label) for label in (cluster.get("member_labels") or []) if label)
    pattern["_coder_count"] = max(int(pattern.get("_coder_count") or 0), int(cluster.get("coder_count") or 0))
    pattern["_occurrence_count"] = int(pattern.get("_occurrence_count") or 0) + int(cluster.get("occurrence_count") or 0)
    pattern["_transcript_count"] = max(int(pattern.get("_transcript_count") or 0), int(cluster.get("transcript_count") or 0))
    for seg in cluster.get("representative_segments") or []:
        if seg not in pattern["representative_segments"]:
            pattern["representative_segments"].append(seg)
    for sample in cluster.get("comparison_samples") or []:
        if sample and sample not in pattern["_comparison_samples"]:
            pattern["_comparison_samples"].append(sample)
    for sample in cluster.get("memo_samples") or []:
        if sample and sample not in pattern["_memo_samples"]:
            pattern["_memo_samples"].append(sample)


def _best_pattern_idx(patterns: List[Dict[str, Any]], tokens: set[str]) -> tuple[int, float]:
    best_idx = -1
    best_score = 0.0
    for idx, pattern in enumerate(patterns):
        score = _jaccard(tokens, pattern.get("_tokens") or set())
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx, best_score


def _condense_clusters_for_llm(clusters: List[Dict[str, Any]], *, limit: int = 180) -> List[Dict[str, Any]]:
    if len(clusters) <= limit:
        return clusters
    two_coder = [c for c in clusters if int(c.get("coder_count") or 0) > 1]
    two_coder_ids = {id(c) for c in two_coder}
    multi_transcript = [
        c for c in clusters if int(c.get("transcript_count") or 0) > 1 and id(c) not in two_coder_ids
    ]
    selected_ids = two_coder_ids | {id(c) for c in multi_transcript}
    remainder = [c for c in clusters if id(c) not in selected_ids]
    selected = (two_coder + multi_transcript + remainder)[:limit]
    return selected


def _build_fallback_incident_patterns(
    clusters: List[Dict[str, Any]],
    *,
    max_patterns: int = 40,
    merge_threshold: float = 0.18,
) -> List[Dict[str, Any]]:
    patterns: List[Dict[str, Any]] = []
    for cluster in clusters:
        tokens = _cluster_tokens(cluster)
        if not tokens:
            continue
        best_idx, best_score = _best_pattern_idx(patterns, tokens)
        if best_idx >= 0 and (best_score >= merge_threshold or len(patterns) >= max_patterns):
            _merge_cluster_into_pattern(patterns[best_idx], cluster)
            continue
        patterns.append(
            {
                "label": str(cluster.get("name_hint") or "").strip(),
                "representative_segments": [],
                "_tokens": set(tokens),
                "_member_labels": [],
                "_coder_count": 0,
                "_occurrence_count": 0,
                "_transcript_count": 0,
                "_comparison_samples": [],
                "_memo_samples": [],
            }
        )
        _merge_cluster_into_pattern(patterns[-1], cluster)

    finalized: List[Dict[str, Any]] = []
    for pattern in patterns:
        member_counts = Counter(pattern["_member_labels"])
        common_labels = [label for label, _count in member_counts.most_common(10)]
        comparison_samples = pattern["_comparison_samples"][:3]
        memo_samples = pattern["_memo_samples"][:2]
        coder_count = int(pattern.get("_coder_count") or 0)
        occurrence_count = int(pattern.get("_occurrence_count") or 0)
        transcript_count = int(pattern.get("_transcript_count") or 0)
        convergence = (
            f"Cross-coder convergence: {occurrence_count} coded occurrences across {coder_count} coder(s) "
            f"and at least {transcript_count} transcript(s). Shared labels include: {', '.join(common_labels[:6])}."
            if coder_count > 1
            else f"Single-coder or weakly matched signal: {occurrence_count} coded occurrence(s), retained only as part of a broader integrated pattern."
        )
        divergence = (
            "Divergence/boundary evidence: "
            + " ".join(sample for sample in comparison_samples if sample)[:700]
            if comparison_samples
            else "Divergence/boundary evidence was not explicitly available for this grouped pattern."
        )
        reconciliation = (
            "Non-local reconciliation: pattern groups labels across multiple transcript locations and preserves representative segment IDs for audit. "
            + ("Memo evidence: " + " ".join(sample for sample in memo_samples if sample)[:500] if memo_samples else "")
        )
        comparative_note = " ".join(part for part in [convergence, divergence, reconciliation] if part).strip()
        finalized.append(
            {
                "label": pattern["label"],
                "representative_segments": pattern["representative_segments"][:12],
                "comparative_note": comparative_note,
                "coder_convergence": convergence,
                "coder_divergence": divergence,
                "non_local_reconciliation": reconciliation,
                "occurrence_count": occurrence_count,
                "coder_count": coder_count,
                "member_labels": common_labels,
            }
        )
    finalized.sort(key=lambda item: (item.get("coder_count", 0), item.get("occurrence_count", 0)), reverse=True)
    return finalized[:max_patterns]


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
            comparison_bits: List[str] = []
            for comparison in note.get("comparison_notes") or []:
                if not isinstance(comparison, dict):
                    continue
                comparison_bits.extend(
                    str(comparison.get(key) or "").strip()
                    for key in ("focus", "similarities", "differences")
                    if str(comparison.get(key) or "").strip()
                )
            comparison_text = " | ".join(comparison_bits[:6])
            base_row = {
                "coder": f"coder{coder_idx}",
                "transcript": note.get("transcript"),
                "segment_number": note.get("segment_number"),
                "memo": note.get("analytic_memo", ""),
                "comparison": comparison_text,
            }
            for cluster_label in note.get("label_clusters") or []:
                flat.append(
                    {
                        **base_row,
                        "label": str(cluster_label),
                    }
                )
            for label in labels:
                flat.append(
                    {
                        **base_row,
                        "label": str(label),
                    }
                )
    clustered = _cluster_label_rows(flat)
    clustered_for_llm = _condense_clusters_for_llm(clustered, limit=180)
    payload = {
        "analysis_mode": analysis_mode,
        "theoretical_framework": theoretical_framework if analysis_mode != "classic" else "",
        "label_clusters": clustered_for_llm,
        "cluster_inventory": {
            "total_deterministic_clusters": len(clustered),
            "clusters_sent_to_model": len(clustered_for_llm),
            "selection_rule": "prioritize cross-coder, cross-transcript, and high-occurrence clusters to keep integration auditable on long datasets",
        },
        "instructions": (
            "Integrate these deterministic label clusters into higher-level incident patterns. "
            "Merge clusters that are conceptually similar, keep representative segments, identify coder convergence/divergence, "
            "and avoid returning one pattern per label. Prefer roughly 20-40 integrated patterns unless the data clearly require more. "
            "Each pattern should explain incident-specific convergence, divergence or boundary shifts, and non-local reconciliation across transcripts."
        ),
        "schema": SYNTH_INCIDENT_SCHEMA,
    }
    user = json.dumps(payload, ensure_ascii=False)
    data = sdk.run_json(system, user, schema_hint=SYNTH_INCIDENT_SCHEMA, attempts=attempts, timeout_s=timeout_s)
    patterns = data.get("incident_patterns") or []
    used_fallback = False
    if not patterns:
        used_fallback = True
        patterns = _build_fallback_incident_patterns(clustered, max_patterns=40)
    if not used_fallback:
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
        "Do not over-merge distinct mechanisms, stakeholder positions, contexts, conditions, actions, consequences, or boundary cases into a few macro buckets unless the evidence truly supports that merge. "
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
    if not cats_in:
        return {
            "categories": [],
            "granularity_note": (
                "No coder-level comparative categories were available, so integrated categories were not generated. "
                "This indicates an upstream category-generation failure rather than evidence for an empty category structure."
            ),
        }
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
