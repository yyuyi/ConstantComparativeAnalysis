from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime
import time
import re

# Ensure the current directory is on sys.path so absolute imports work on platforms like Render
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Make imports work whether this file is imported as part of a package (relative) or top-level
try:
    from . import config  # package import
    from .rag.vector_store import VectorIndex
    from .agents.sdk import AgentSDK
    from .agents.coder_agent import (
        run_incident_coding,
        run_category_comparison,
        run_comparative_memos,
        run_cca_synthesis,
        summarize_transcript,
    )
    from .agents.synth_agent import (
        synthesize_incident_patterns,
        synthesize_category_matrix,
        synthesize_memo_digest,
        synthesize_cca_summary,
    )
    # from .agents.stats_agent import build_summary
    from .agents.tools import write_json_txt, build_segment_maps
except Exception:
    try:
        import config  # top-level import
    except Exception:
        import configure as config  # optional alias if renamed
    from rag.vector_store import VectorIndex
    from agents.sdk import AgentSDK
    from agents.coder_agent import (
        run_incident_coding,
        run_category_comparison,
        run_comparative_memos,
        run_cca_synthesis,
        summarize_transcript,
    )
    from agents.synth_agent import (
        synthesize_incident_patterns,
        synthesize_category_matrix,
        synthesize_memo_digest,
        synthesize_cca_summary,
    )
    # from agents.stats_agent import build_summary
    from agents.tools import write_json_txt, build_segment_maps

# If you reference _cfg later, create an alias now so both paths share it
_cfg = config


def _log(run_dir: Path, msg: str) -> None:
    ts = datetime.utcnow().strftime("%H:%M:%S")
    with open(run_dir / "progress.log", "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def run_job(run_dir: Path) -> None:
    try:
        params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    except Exception:
        _log(run_dir, "Failed to read params.json")
        return

    coders = int(params.get("coders", 1))
    coders = max(1, min(2, coders))
    analysis_mode = params.get("analysis_mode", "classic")
    cac_enabled = bool(params.get("cac_enabled", False))
    max_categories = int(params.get("max_categories", config.MAX_CATEGORIES_DEFAULT))
    segment_len = int(params.get("segment_length", config.SEGMENT_LENGTH_DEFAULT))
    model = params.get("model", config.DEFAULT_MODEL)

    run_id = run_dir.name[4:] if run_dir.name.startswith("run_") else run_dir.name
    api_key_path = run_dir.parent / "_secrets" / f"{run_id}.secret"
    api_key: str | None = None
    if api_key_path.exists():
        try:
            api_key = api_key_path.read_text(encoding="utf-8").strip()
        except Exception:
            api_key = None
        finally:
            try:
                api_key_path.unlink()
            except Exception:
                pass

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY") or None
    if not api_key:
        _log(run_dir, "Error: Missing OpenAI API key. Aborting run.")
        (run_dir / "result.json").write_text(json.dumps({"files": [], "error": "Missing OpenAI API key"}, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    sdk = AgentSDK(model=model, api_key=api_key)
    _log(run_dir, f"Agent SDK initialized | model={model} | key_present={'yes' if api_key else 'no'}")


    # Connectivity sanity check
    # Lightweight connectivity sanity check; tolerate transient backend errors
    ping_ok = False
    ping_error: str | None = None
    for attempt in range(2):
        ping = sdk.run_json(
            system="You are a JSON responder.",
            user=json.dumps({"instruction": "Return ONLY {\"ok\": true} as a JSON object."}),
            schema_hint="{\"ok\": bool}",
            attempts=1,
        )
        if isinstance(ping, dict) and ping.get("ok") is True:
            ping_ok = True
            break
        diag = sdk.diagnostics()
        ping_error = diag.get("last_error")
        # Retry once on transient server errors (HTTP 500)
        if ping_error and "Error code: 500" in ping_error and attempt == 0:
            time.sleep(1.0)
            continue
        break
    if not ping_ok:
        if ping_error:
            _log(run_dir, f"Agent connectivity check warning (non-blocking): {ping_error}")
        else:
            _log(run_dir, "Agent connectivity check warning (non-blocking): empty/invalid JSON response.")

    inputs_dir = run_dir / "inputs"
    transcripts_dir = inputs_dir / "transcripts"
    uploads = params.get("uploads", [])

    study_background = (inputs_dir / "study_background.txt").read_text(encoding="utf-8")
    theoretical_framework = (inputs_dir / "theoretical_framework.txt").read_text(encoding="utf-8")

    # No refinement here; if the user used the refine preview, the posted text is already refined.

    # Load transcripts and segment
    all_segments: List[Dict[str, Any]] = []
    # per_tx_info no longer used
    transcript_raw: Dict[str, str] = {}
    _log(run_dir, "Loading, chunking, and indexing transcripts...")
    vindex = VectorIndex(api_key=api_key)
    def _read_text(path: Path) -> str:
        suf = path.suffix.lower()
        try:
            if suf == ".txt":
                return path.read_text(encoding="utf-8", errors="replace")
            if suf == ".pdf":
                try:
                    from pypdf import PdfReader  # type: ignore
                    reader = PdfReader(str(path))
                    return "\n".join(page.extract_text() or "" for page in reader.pages)
                except Exception:
                    return ""
            if suf in (".docx", ".docs"):
                try:
                    import docx  # type: ignore
                    doc = docx.Document(str(path))
                    return "\n".join(p.text for p in doc.paragraphs)
                except Exception:
                    return ""
            if suf == ".doc":
                # Legacy .doc not supported without external tools
                return ""
            # Fallback
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    for name in uploads:
        raw_text = _read_text(transcripts_dir / name)
        base = Path(name).stem
        segs = vindex.add_transcript(name=name, raw_text=raw_text, segment_len_tokens=segment_len)
        write_json_txt(run_dir, f"segments_{base}.txt", {"transcript": name, "segments": {str(s["segment_number"]): s["text"] for s in segs}})
        all_segments.extend(segs)
        transcript_raw[name] = raw_text

    # Build segment maps (raw text only; coder agents decide interviewee vs interviewer)
    by_key_map, _ = build_segment_maps(all_segments)

    # Build comprehensive per-transcript summaries (multi-paragraph) for use in axial and selective coding (obligatory)
    transcript_summaries: Dict[str, str] = {}
    try:
        _log(run_dir, "Generating comprehensive transcript summaries...")
        for name, text in transcript_raw.items():
            if (text or "").strip():
                summary = summarize_transcript(sdk=sdk, transcript_name=name, text=text, attempts=1, timeout_s=240.0)
                transcript_summaries[name] = summary
            else:
                transcript_summaries[name] = ""
    except Exception as e:
        _log(run_dir, f"Warning: transcript summaries failed: {e}")
        transcript_summaries = {name: "" for name in transcript_raw.keys()}

    def _trim_to_sentences(text: str, max_sentences: int = 3) -> str:
        if not isinstance(text, str):
            return ""
        t = text.strip()
        if not t:
            return t
        # Find sentence enders (., !, ?) optionally followed by quotes/brackets and whitespace or EOS
        enders = list(re.finditer(r"[.!?](?=[\)\]\"'”’]*\s|$)", t))
        if not enders:
            return t
        if len(enders) <= max_sentences:
            return t
        cut = enders[max_sentences - 1].end()
        return t[:cut].strip()

    def _normalize_incident_notes(notes: Any) -> List[Dict[str, Any]]:
        if isinstance(notes, dict):
            if {"transcript", "segment_number", "labels"}.issubset(notes.keys()):
                return [notes]
            values = list(notes.values())
            return [v for v in values if isinstance(v, dict)]
        if isinstance(notes, list):
            return [v for v in notes if isinstance(v, dict)]
        return []

    def _has_labels(notes: List[Dict[str, Any]]) -> bool:
        for note in notes:
            labels = note.get("labels")
            if isinstance(labels, list) and any(str(lbl).strip() for lbl in labels):
                return True
        return False

    def _iter_chunks(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
        if size <= 0:
            return [items]
        return [items[i:i + size] for i in range(0, len(items), size)]

    def _segment_key(segment: Dict[str, Any]) -> Tuple[str, int]:
        tx = str(segment.get("transcript"))
        try:
            num = int(segment.get("segment_number") or 0)
        except Exception:
            num = 0
        return (tx, num)

    # Per-coder analysis agents (constant comparative analysis pipeline)
    per_coder_incidents: List[List[Dict[str, Any]]] = []
    per_coder_categories: List[List[Dict[str, Any]]] = []
    per_coder_memos: List[List[Dict[str, Any]]] = []
    per_coder_syntheses: List[Dict[str, Any]] = []

    for i in range(1, coders + 1):
        coder_id = f"coder{i}"
        _log(run_dir, f"[{coder_id}] Incident coding & comparisons...")
        incident_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        prior_incident_context: List[Dict[str, Any]] = []
        chunk_size = 1
        chunks = _iter_chunks(all_segments, chunk_size) if all_segments else []
        diag_msg: str | None = None
        for idx, chunk in enumerate(chunks, start=1):
            chunk_summaries = {
                str(seg.get("transcript")): transcript_summaries.get(str(seg.get("transcript")), "")
                for seg in chunk
                if transcript_summaries.get(str(seg.get("transcript")), "").strip()
            }
            seg_info = chunk[0] if chunk else {}
            seg_label = f"{seg_info.get('transcript','')}#{seg_info.get('segment_number','')}" if seg_info else "?"
            _log(run_dir, f"[{coder_id}] Segment {idx}/{len(chunks)} | {seg_label}")
            notes: List[Dict[str, Any]] = []
            for attempt in range(2):
                notes = run_incident_coding(
                    sdk=sdk,
                    segments=chunk,
                    study_background=study_background,
                    analysis_mode=analysis_mode,
                    theoretical_framework=theoretical_framework,
                    transcript_summaries=chunk_summaries,
                    prior_incidents=prior_incident_context[-20:],
                    attempts=2,
                    timeout_s=150.0,
                )
                notes = _normalize_incident_notes(notes)
                diag_info = sdk.diagnostics() or {}
                diag_msg = diag_info.get("last_error")
                diag_raw = diag_info.get("last_raw")
                if notes and _has_labels(notes):
                    break
                if diag_msg or diag_raw:
                    msg = diag_msg or "no explicit error"
                    if diag_raw:
                        msg += f" | raw={diag_raw}"
                    _log(run_dir, f"[{coder_id}] Incident note warning: {msg}")
                if attempt < 1:
                    _log(run_dir, f"[{coder_id}] Retrying incident comparison for segment {idx}...")
            if not notes:
                continue
            expected_tx = seg_info.get("transcript") if isinstance(seg_info, dict) else None
            expected_num = seg_info.get("segment_number") if isinstance(seg_info, dict) else None
            for note in notes:
                if expected_tx is not None:
                    note["transcript"] = expected_tx
                if expected_num is not None:
                    note["segment_number"] = expected_num
                labels = []
                for lbl in note.get("labels", []) or []:
                    lbl_text = str(lbl).strip()
                    if lbl_text:
                        labels.append(lbl_text)
                note["labels"] = labels
                key = _segment_key(note)
                if key not in incident_map or not incident_map[key].get("labels"):
                    incident_map[key] = note
                for lbl in labels:
                    prior_incident_context.append(
                        {
                            "label": lbl,
                            "transcript": note.get("transcript"),
                            "segment_number": note.get("segment_number"),
                            "analytic_memo": note.get("analytic_memo", ""),
                        }
                    )
            if len(prior_incident_context) > 60:
                prior_incident_context = prior_incident_context[-60:]

        ordered_keys = [_segment_key(seg) for seg in all_segments]
        incident_records = [incident_map[key] for key in ordered_keys if key in incident_map]
        write_json_txt(run_dir, f"incident_coding_{coder_id}.txt", {"coder": coder_id, "incident_notes": incident_records})

        _log(run_dir, f"[{coder_id}] Building comparative categories...")
        cats: List[Dict[str, Any]] = []
        for attempt in range(2):
            t0 = time.time()
            cats = run_category_comparison(
                sdk=sdk,
                incident_notes=incident_records,
                max_categories=max_categories,
                study_background=study_background,
                analysis_mode=analysis_mode,
                theoretical_framework=theoretical_framework,
                transcript_summaries=transcript_summaries,
                attempts=1,
                timeout_s=240.0,
            )
            _log(run_dir, f"[{coder_id}] Category response in {time.time()-t0:.2f}s; categories={len(cats) if cats else 0}.")
            if cats:
                break
            _log(run_dir, f"[{coder_id}] Retrying category comparison (attempt {attempt + 2})...")

        QUOTES_BATCH_SCHEMA = "{\"quotes_by_category\": [{\"index\": int, \"quotes\": [str]}]}"
        payload_items = []
        for idx, cat in enumerate(cats):
            name = cat.get("name", "") or ""
            props = "\n".join(str(p) for p in (cat.get("defining_properties") or []) if str(p).strip())
            insights = "\n".join(str(p) for p in (cat.get("comparative_insights") or []) if str(p).strip())
            query_parts = [name, props, insights]
            query = "\n\n".join(part for part in query_parts if part)
            results = vindex.query(text=query, k=getattr(_cfg, "RAG_K_DEFAULT", 2))
            contexts = [t for (t, _meta) in results if t and t.strip()]
            payload_items.append(
                {
                    "index": idx,
                    "category": {"name": name, "definition": props, "insights": insights},
                    "contexts": contexts,
                }
            )
        if payload_items:
            system = (
                "Extract 1–3 verbatim quotes per category that best evidence the comparative insight. "
                "Quotes must be substrings of the provided contexts, prioritise participant speech, and limit each quote to ≤ 3 sentences. JSON only."
            )
            user = json.dumps({"items": payload_items, "schema": QUOTES_BATCH_SCHEMA}, ensure_ascii=False)
            t0 = time.time()
            data = sdk.run_json(system, user, schema_hint=QUOTES_BATCH_SCHEMA, attempts=1, timeout_s=240.0)
            _log(run_dir, f"[{coder_id}] Quote batch response in {time.time()-t0:.2f}s.")
            by_cat = {
                int(row.get("index", -1)): [str(q).strip() for q in (row.get("quotes") or []) if str(q).strip()]
                for row in (data.get("quotes_by_category") or [])
            }
            for idx, cat in enumerate(cats):
                qlist = (by_cat.get(idx) or [])[:3]
                cat["supporting_quotes"] = [
                    q for q in ([_trim_to_sentences(q, 3) for q in qlist] if qlist else []) if q and q.strip()
                ]
        write_json_txt(run_dir, f"category_comparisons_{coder_id}.txt", {"coder": coder_id, "comparative_categories": cats})

        _log(run_dir, f"[{coder_id}] Writing comparative memos...")
        memos: List[Dict[str, Any]] = []
        for attempt in range(2):
            memos = run_comparative_memos(
                sdk=sdk,
                incident_notes=incident_records,
                comparative_categories=cats,
                study_background=study_background,
                analysis_mode=analysis_mode,
                theoretical_framework=theoretical_framework,
                transcript_summaries=transcript_summaries,
                attempts=1,
                timeout_s=200.0,
            )
            if memos:
                break
            _log(run_dir, f"[{coder_id}] Retrying memo generation (attempt {attempt + 2})...")
        write_json_txt(run_dir, f"cca_memos_{coder_id}.txt", {"coder": coder_id, "comparative_memos": memos})

        _log(run_dir, f"[{coder_id}] Synthesizing comparative findings...")
        synthesis = run_cca_synthesis(
            sdk=sdk,
            incident_notes=incident_records,
            comparative_categories=cats,
            comparative_memos=memos,
            cac_enabled=cac_enabled,
            study_background=study_background,
            analysis_mode=analysis_mode,
            theoretical_framework=theoretical_framework,
            transcript_summaries=transcript_summaries,
            attempts=1,
            timeout_s=220.0,
        )
        write_json_txt(run_dir, f"cca_synthesis_{coder_id}.txt", {"coder": coder_id, **synthesis})

        per_coder_incidents.append(incident_records)
        per_coder_categories.append(cats)
        per_coder_memos.append(memos)
        per_coder_syntheses.append(synthesis)

    # Integration (report removed) with robust error handling
    try:
        integrated_counts: Dict[str, int] | None = None
        incident_patterns: Dict[str, Any] = {}
        merged_categories: Dict[str, Any] = {}
        memo_digest: Dict[str, Any] = {}
        merged_summary: Dict[str, Any] = {}

        if coders > 1:
            _log(run_dir, "Integrating incident patterns across coders...")
            incident_patterns = synthesize_incident_patterns(
                sdk=sdk,
                per_coder_incidents=per_coder_incidents,
                analysis_mode=analysis_mode,
                theoretical_framework=theoretical_framework,
                attempts=1,
                timeout_s=120.0,
            )
            write_json_txt(run_dir, "integrated_incident_patterns.txt", incident_patterns)

            _log(run_dir, "Integrating comparative categories...")
            for attempt in range(2):
                merged_categories = synthesize_category_matrix(
                    sdk=sdk,
                    per_coder_categories=per_coder_categories,
                    analysis_mode=analysis_mode,
                    theoretical_framework=theoretical_framework,
                    attempts=1,
                    timeout_s=150.0,
                )
                if merged_categories.get("categories"):
                    break
                _log(run_dir, f"Retry category integration (attempt {attempt + 2})...")
            write_json_txt(run_dir, "integrated_categories.txt", merged_categories)

            _log(run_dir, "Integrating comparative memos...")
            memo_digest = synthesize_memo_digest(
                sdk=sdk,
                per_coder_memos=per_coder_memos,
                analysis_mode=analysis_mode,
                theoretical_framework=theoretical_framework,
                attempts=1,
                timeout_s=120.0,
            )
            write_json_txt(run_dir, "integrated_memo_digest.txt", memo_digest)

            _log(run_dir, "Synthesizing comparative summary...")
            merged_summary = synthesize_cca_summary(
                sdk=sdk,
                per_coder_syntheses=per_coder_syntheses,
                analysis_mode=analysis_mode,
                theoretical_framework=theoretical_framework,
                attempts=1,
                timeout_s=150.0,
            )
            write_json_txt(run_dir, "integrated_cca_summary.txt", merged_summary)

            integrated_counts = {
                "incident_patterns": len((incident_patterns.get("incident_patterns") or [])),
                "categories": len((merged_categories.get("categories") or [])),
                "memo_digest": len((memo_digest.get("memo_digest") or [])),
                "summary": 1 if (merged_summary.get("comparative_summary") or "").strip() else 0,
            }

        per_coder_counts: Dict[str, Dict[str, int]] = {}
        for i, (inc, cats, memos, synth) in enumerate(
            zip(per_coder_incidents, per_coder_categories, per_coder_memos, per_coder_syntheses), start=1
        ):
            label_total = sum(len(note.get("labels") or []) for note in inc)
            if not label_total:
                label_total = len(inc)
            per_coder_counts[f"coder{i}"] = {
                "incidents": label_total,
                "categories": len(cats),
                "memos": len(memos),
                "summary": 1 if (synth.get("comparative_summary") or "").strip() else 0,
            }

        mode_val = analysis_mode
        cac_val = str(bool(cac_enabled)).lower()
        settings = (
            f"Analysis settings: {coders} coders; analysis_mode = \"{mode_val}\"; "
            f"cac_enabled = {cac_val}; max_categories = {max_categories}; "
            f"segment_length = {segment_len}; model = \"{model}\"; method = CCA."
        )

        if coders == 2:
            c1 = per_coder_counts.get("coder1", {"incidents": 0, "categories": 0, "memos": 0, "summary": 0})
            c2 = per_coder_counts.get("coder2", {"incidents": 0, "categories": 0, "memos": 0, "summary": 0})
            per_coder_line = (
                "Per-coder counts: "
                f"coder1(incidents={c1['incidents']}, categories={c1['categories']}, memos={c1['memos']}, summary={c1['summary']}); "
                f"coder2(incidents={c2['incidents']}, categories={c2['categories']}, memos={c2['memos']}, summary={c2['summary']})."
            )
        else:
            pc = per_coder_counts.get("coder1", {"incidents": 0, "categories": 0, "memos": 0, "summary": 0})
            per_coder_line = (
                "Per-coder counts: "
                f"coder1(incidents={pc['incidents']}, categories={pc['categories']}, memos={pc['memos']}, summary={pc['summary']})."
            )

        summary_parts = [settings, per_coder_line]
        if integrated_counts is not None:
            integrated_line = (
                "Integrated counts: "
                f"incident_patterns = {integrated_counts['incident_patterns']}; "
                f"categories = {integrated_counts['categories']}; "
                f"memo_digest = {integrated_counts['memo_digest']}; "
                f"summary = {integrated_counts['summary']}."
            )
            summary_parts.append(integrated_line)
        else:
            _log(run_dir, "Single coder run; skipping integrated synthesis.")
            summary_parts.append("Integrated synthesis skipped (single coder run).")

        summary_text = " ".join(summary_parts)
        write_json_txt(run_dir, "analysis_summary.txt", {"summary": summary_text})

        files = [f.name for f in sorted(run_dir.glob("*.txt"))]
        (run_dir / "result.json").write_text(json.dumps({"files": files}, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(run_dir, "Analysis complete. Outputs ready for download.")
    except Exception as e:
        _log(run_dir, f"Error during integration/report: {e}")
        files = [f.name for f in sorted(run_dir.glob("*.txt"))]
        (run_dir / "result.json").write_text(json.dumps({"files": files, "error": str(e)}, ensure_ascii=False, indent=2), encoding="utf-8")
