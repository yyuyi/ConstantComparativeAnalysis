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
    from .agents.coder_agent import run_open_coding, run_axial_coding, run_selective_coding
    from .agents.synth_agent import synthesize_categories, synthesize_core_story, synthesize_open_codes
    # from .agents.stats_agent import build_summary
    from .agents.tools import write_json_txt, build_segment_maps
except Exception:
    try:
        import config  # top-level import
    except Exception:
        import configure as config  # optional alias if renamed
    from rag.vector_store import VectorIndex
    from agents.sdk import AgentSDK
    from agents.coder_agent import run_open_coding, run_axial_coding, run_selective_coding
    from agents.synth_agent import synthesize_categories, synthesize_core_story, synthesize_open_codes
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
        try:
            from .agents.coder_agent import summarize_transcript  # package import
        except Exception:
            from agents.coder_agent import summarize_transcript  # top-level import
        _log(run_dir, "Generating comprehensive transcript summaries...")
        for name, text in transcript_raw.items():
            if (text or "").strip():
                summary = summarize_transcript(sdk=sdk, transcript_name=name, text=text, attempts=1, timeout_s=120.0)
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

    def _normalize_open_blocks(blocks: Any) -> List[Dict[str, Any]]:
        if isinstance(blocks, dict):
            # Single object or keyed map
            if {"transcript", "segment_number", "codes"}.issubset(blocks.keys()):
                return [blocks]
            values = list(blocks.values())
            return [b for b in values if isinstance(b, dict)]
        if isinstance(blocks, list):
            return [b for b in blocks if isinstance(b, dict)]
        return []

    def _has_codes(blocks: List[Dict[str, Any]]) -> bool:
        return any((isinstance(block.get("codes"), list) and block["codes"]) for block in blocks)

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

    # Per-coder analysis agents
    per_coder_open: List[List[Dict[str, Any]]] = []
    per_coder_categories: List[List[Dict[str, Any]]] = []
    per_coder_selective: List[Dict[str, Any]] = []

    for i in range(1, coders + 1):
        coder_id = f"coder{i}"
        _log(run_dir, f"[{coder_id}] Open coding (sequential segments)...")
        oc_block_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
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
            chunk_blocks: List[Dict[str, Any]] = []
            for attempt in range(2):
                chunk_blocks = run_open_coding(
                    sdk=sdk,
                    segments=chunk,
                    study_background=study_background,
                    analysis_mode=analysis_mode,
                    theoretical_framework=theoretical_framework,
                    transcript_summaries=chunk_summaries,
                    attempts=2,
                    timeout_s=75.0,
                )
                chunk_blocks = _normalize_open_blocks(chunk_blocks)
                diag_info = sdk.diagnostics() or {}
                diag_msg = diag_info.get("last_error")
                diag_raw = diag_info.get("last_raw")
                if chunk_blocks and _has_codes(chunk_blocks):
                    break
                if diag_msg or diag_raw:
                    msg = diag_msg or "no explicit error"
                    if diag_raw:
                        msg += f" | raw={diag_raw}"
                    _log(run_dir, f"[{coder_id}] Chunk {idx} warning: {msg}")
                if attempt < 1:
                    _log(run_dir, f"[{coder_id}] Retrying chunk {idx}...")
            if not chunk_blocks:
                diag_info = sdk.diagnostics() or {}
                diag_msg = diag_info.get("last_error")
                diag_raw = diag_info.get("last_raw")
                if diag_msg or diag_raw:
                    msg = diag_msg or "no explicit error"
                    if diag_raw:
                        msg += f" | raw={diag_raw}"
                    _log(run_dir, f"[{coder_id}] Chunk {idx} produced no usable codes: {msg}")
            expected_tx = seg_info.get("transcript") if isinstance(seg_info, dict) else None
            expected_num = seg_info.get("segment_number") if isinstance(seg_info, dict) else None
            for block in chunk_blocks or []:
                if expected_tx is not None:
                    block["transcript"] = expected_tx
                if expected_num is not None:
                    block["segment_number"] = expected_num
                key = _segment_key(block)
                if key not in oc_block_map or not oc_block_map[key].get("codes"):
                    oc_block_map[key] = block

        ordered_keys = [_segment_key(seg) for seg in all_segments]
        oc_blocks = [oc_block_map[key] for key in ordered_keys if key in oc_block_map]
        if all_segments and not (oc_blocks and _has_codes(oc_blocks)) and diag_msg:
            _log(run_dir, f"[{coder_id}] Open coding warning: {diag_msg}")
        # normalize to list of {code, transcript, segment_number}
        oc: List[Dict[str, Any]] = []
        for block in oc_blocks:
            codes_field = (block.get("codes") or [])
            for citem in codes_field[:3]:
                code_text = str(citem).strip() if not isinstance(citem, dict) else str(citem.get("code", "")).strip()
                seg_num = block.get("segment_number")
                tx = block.get("transcript")
                if not code_text:
                    continue
                oc.append({
                    "code": code_text,
                    "segment_number": seg_num,
                    "transcript": tx,
                })
        write_json_txt(run_dir, f"open_coding_{coder_id}.txt", {"coder": coder_id, "open_codes": oc})

        _log(run_dir, f"[{coder_id}] Axial coding...")
        cats: List[Dict[str, Any]] = []
        for attempt in range(2):
            t0 = time.time()
            cats = run_axial_coding(
                sdk=sdk,
                open_codes=oc,
                max_categories=max_categories,
                study_background=study_background,
                analysis_mode=analysis_mode,
                theoretical_framework=theoretical_framework,
                transcript_summaries=transcript_summaries,
                attempts=1,
                timeout_s=120.0,
            )
            _log(run_dir, f"[{coder_id}] Axial response in {time.time()-t0:.2f}s; categories={len(cats) if cats else 0}.")
            if cats and all(c.get("name") for c in cats):
                break
            _log(run_dir, f"[{coder_id}] Retrying axial coding (attempt {attempt + 2})...")
        # RAG + LLM (batched): attach quotes by querying the vector index once per category, then batch extract
        QUOTES_BATCH_SCHEMA = "{\"quotes_by_category\": [{\"index\": int, \"quotes\": [str]}]}"
        payload_items = []
        for idx, cat in enumerate(cats):
            query = (cat.get("name", "") or "") + "\n\n" + (cat.get("description", "") or "")
            results = vindex.query(text=query, k=getattr(_cfg, 'RAG_K_DEFAULT', 2))
            contexts = [t for (t, m) in results if t and t.strip()]
            payload_items.append({
                "index": idx,
                "category": {"name": cat.get("name", ""), "description": cat.get("description", "")},
                "contexts": contexts,
            })
        if payload_items:
            system = (
                "You extract short, faithful quotes for grounded-theory categories. "
                "Given category names/descriptions and several transcript contexts per category, return 1–3 quotes (each 1–3 sentences) per category that are VERBATIM substrings of the provided contexts. "
                "Prefer interviewee speech; exclude interviewer prompts. JSON only."
            )
            user = json.dumps({"items": payload_items, "schema": QUOTES_BATCH_SCHEMA}, ensure_ascii=False)
            t0 = time.time()
            data = sdk.run_json(system, user, schema_hint=QUOTES_BATCH_SCHEMA, attempts=1, timeout_s=120.0)
            _log(run_dir, f"[{coder_id}] Quote batch response in {time.time()-t0:.2f}s.")
            by_cat = {int(row.get("index", -1)): [str(q).strip() for q in (row.get("quotes") or []) if str(q).strip()] for row in (data.get("quotes_by_category") or [])}
            for idx, cat in enumerate(cats):
                qlist = (by_cat.get(idx) or [])[:3]
                cat["supporting_quotes"] = [q for q in ([_trim_to_sentences(q, 3) for q in qlist] if qlist else []) if q and q.strip()]
        write_json_txt(run_dir, f"axial_coding_{coder_id}.txt", {"coder": coder_id, "categories": cats})

        _log(run_dir, f"[{coder_id}] Selective coding...")
        sel: Dict[str, Any] = {}
        for attempt in range(2):
            sel = run_selective_coding(
                sdk=sdk,
                categories=cats,
                cac_enabled=cac_enabled,
                study_background=study_background,
                analysis_mode=analysis_mode,
                theoretical_framework=theoretical_framework,
                transcript_summaries=transcript_summaries,
                attempts=1,
                timeout_s=75.0,
            )
            if sel.get("core_story"):
                break
            _log(run_dir, f"[{coder_id}] Retrying selective coding (attempt {attempt + 2})...")
        write_json_txt(run_dir, f"selective_coding_{coder_id}.txt", {"coder": coder_id, **sel})

        per_coder_open.append(oc)
        per_coder_categories.append(cats)
        per_coder_selective.append(sel)

    # Integration (report removed) with robust error handling
    try:
        integrated_counts: Dict[str, int] | None = None
        merged_categories: List[Dict[str, Any]] = []
        merged_core: Dict[str, Any] = {}

        if coders > 1:
            _log(run_dir, "Integrating open codes...")
            open_synth = synthesize_open_codes(
                sdk=sdk,
                per_coder_codes=per_coder_open,
                analysis_mode=analysis_mode,
                theoretical_framework=theoretical_framework,
                attempts=1,
                timeout_s=60.0,
            )
            write_json_txt(run_dir, "integrated_open_codes.txt", open_synth)

            _log(run_dir, "Synthesizing categories across coders...")
            for attempt in range(2):
                merged_categories = synthesize_categories(
                    sdk=sdk,
                    per_coder_categories=per_coder_categories,
                    analysis_mode=analysis_mode,
                    theoretical_framework=theoretical_framework,
                    attempts=1,
                    timeout_s=60.0,
                )
                if merged_categories:
                    break
                _log(run_dir, f"Retry category synthesis (attempt {attempt + 2})...")
            write_json_txt(run_dir, "integrated_categories.txt", {"categories": merged_categories, "count": len(merged_categories)})

            _log(run_dir, "Synthesizing integrated core story...")
            for attempt in range(2):
                merged_core = synthesize_core_story(
                    sdk=sdk,
                    per_coder_stories=per_coder_selective,
                    cac_enabled=cac_enabled,
                    analysis_mode=analysis_mode,
                    theoretical_framework=theoretical_framework,
                    attempts=1,
                    timeout_s=75.0,
                )
                if merged_core.get("core_story"):
                    break
                _log(run_dir, f"Retry core story synthesis (attempt {attempt + 2})...")
            write_json_txt(run_dir, "integrated_core_story.txt", {"core_story": merged_core.get("core_story", "")})

            integrated_counts = {
                "open_codes": len((open_synth.get("open_codes") or [])),
                "categories": len(merged_categories),
                "core_story": 1 if (merged_core.get("core_story") or "").strip() else 0,
            }

        per_coder_counts: Dict[str, Dict[str, int]] = {}
        for i, (oc, cats, sel) in enumerate(zip(per_coder_open, per_coder_categories, per_coder_selective), start=1):
            per_coder_counts[f"coder{i}"] = {
                "open_codes": len(oc),
                "categories": len(cats),
                "core_story": 1 if (sel.get("core_story") or "").strip() else 0,
            }

        mode_val = analysis_mode
        cac_val = str(bool(cac_enabled)).lower()
        settings = (
            f"Analysis settings: {coders} coders; analysis_mode = \"{mode_val}\"; "
            f"cac_enabled = {cac_val}; max_categories = {max_categories}; "
            f"segment_length = {segment_len}; model = \"{model}\"."
        )

        if coders == 2:
            c1 = per_coder_counts.get("coder1", {"open_codes": 0, "categories": 0, "core_story": 0})
            c2 = per_coder_counts.get("coder2", {"open_codes": 0, "categories": 0, "core_story": 0})
            per_coder_line = (
                "Per-coder counts: "
                f"coder1(open_codes={c1['open_codes']}, categories={c1['categories']}, core_story={c1['core_story']}); "
                f"coder2(open_codes={c2['open_codes']}, categories={c2['categories']}, core_story={c2['core_story']})."
            )
        else:
            pc = per_coder_counts.get("coder1", {"open_codes": 0, "categories": 0, "core_story": 0})
            per_coder_line = (
                "Per-coder counts: "
                f"coder1(open_codes={pc['open_codes']}, categories={pc['categories']}, core_story={pc['core_story']})."
            )

        summary_parts = [settings, per_coder_line]
        if integrated_counts is not None:
            integrated_line = (
                "Integrated counts (post-integration): "
                f"open_codes = {integrated_counts['open_codes']}; "
                f"categories = {integrated_counts['categories']}; "
                f"core_story = {integrated_counts['core_story']}."
            )
            summary_parts.append(integrated_line)
        else:
            _log(run_dir, "Single coder run; skipping integrated synthesis.")
            summary_parts.append("Integrated analysis skipped (single coder run).")

        summary_text = " ".join(summary_parts)
        write_json_txt(run_dir, "analysis_summary.txt", {"summary": summary_text})

        files = [f.name for f in sorted(run_dir.glob("*.txt"))]
        (run_dir / "result.json").write_text(json.dumps({"files": files}, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(run_dir, "Analysis complete. Outputs ready for download.")
    except Exception as e:
        _log(run_dir, f"Error during integration/report: {e}")
        files = [f.name for f in sorted(run_dir.glob("*.txt"))]
        (run_dir / "result.json").write_text(json.dumps({"files": files, "error": str(e)}, ensure_ascii=False, indent=2), encoding="utf-8")
