from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

# Make imports work whether this file is imported as part of a package (relative)
# or as a top-level module (absolute)
try:
    from . import config  # package import
    from .rag.vector_store import VectorIndex
    from .agents.sdk import AgentSDK
    from .agents.coder_agent import run_open_coding, run_axial_coding, run_selective_coding
    from .agents.synth_agent import synthesize_categories, synthesize_core_story, synthesize_open_codes
    from .agents.stats_agent import build_summary
    from .agents.tools import write_json_txt, build_segment_maps
except ImportError:
    import config  # top-level import
    from rag.vector_store import VectorIndex
    from agents.sdk import AgentSDK
    from agents.coder_agent import run_open_coding, run_axial_coding, run_selective_coding
    from agents.synth_agent import synthesize_categories, synthesize_core_story, synthesize_open_codes
    from agents.stats_agent import build_summary
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
    analysis_mode = params.get("analysis_mode", "classic")
    cac_enabled = bool(params.get("cac_enabled", False))
    max_categories = int(params.get("max_categories", config.MAX_CATEGORIES_DEFAULT))
    segment_len = int(params.get("segment_length", config.SEGMENT_LENGTH_DEFAULT))
    model = params.get("model", config.DEFAULT_MODEL)
    api_key = params.get("api_key") or None

    # Fallback to key file if UI provided none
    if not api_key:
        try:
            kf = Path("instructions/openai_api_key.txt")
            if kf.exists():
                api_key = kf.read_text(encoding="utf-8").strip()
        except Exception:
            api_key = None
    sdk = AgentSDK(model=model, api_key=api_key)
    _log(run_dir, f"Agent SDK initialized | model={model} | key_present={'yes' if api_key else 'no'}")

    # Connectivity sanity check
    ping = sdk.run_json(
        system="You are a JSON responder.",
        user=json.dumps({"instruction": "Return ONLY {\"ok\": true} as a JSON object."}),
        schema_hint="{\"ok\": bool}",
        attempts=1,
    )
    if not (isinstance(ping, dict) and ping.get("ok") is True):
        diag = sdk.diagnostics()
        if diag.get("last_error"):
            _log(run_dir, f"Warning: Agent call failed: {diag['last_error']}")
        else:
            _log(run_dir, "Warning: Agent call returned empty/invalid JSON.")

    inputs_dir = run_dir / "inputs"
    transcripts_dir = inputs_dir / "transcripts"
    uploads = params.get("uploads", [])

    study_background = (inputs_dir / "study_background.txt").read_text(encoding="utf-8")
    theoretical_framework = (inputs_dir / "theoretical_framework.txt").read_text(encoding="utf-8")

    # Load transcripts and segment
    all_segments: List[Dict[str, Any]] = []
    per_tx_info: List[Dict[str, Any]] = []
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
        per_tx_info.append({"display_name": name, "name": name, "segments": segs})
        all_segments.extend(segs)

    # Build segment maps (raw text only; coder agents decide interviewee vs interviewer)
    by_key_map, _ = build_segment_maps(all_segments)

    # Per-coder analysis agents
    per_coder_open: List[List[Dict[str, Any]]] = []
    per_coder_categories: List[List[Dict[str, Any]]] = []
    per_coder_selective: List[Dict[str, Any]] = []

    for i in range(1, coders + 1):
        coder_id = f"coder{i}"
        _log(run_dir, f"[{coder_id}] Open coding...")
        # Provide original segments; coder agent decides interviewee content
        oc_blocks: List[Dict[str, Any]] = []
        for attempt in range(2):
            oc_blocks = run_open_coding(
                sdk=sdk,
                segments=all_segments,
                study_background=study_background,
                analysis_mode=analysis_mode,
                theoretical_framework=theoretical_framework,
                attempts=1,
                timeout_s=60.0,
            )
            if oc_blocks and any((b.get("codes") for b in oc_blocks)):
                break
            _log(run_dir, f"[{coder_id}] Retrying open coding (attempt {attempt + 2})...")
        # normalize to list of {code, transcript, segment_number}
        oc: List[Dict[str, Any]] = []
        for block in oc_blocks:
            for code in (block.get("codes") or [])[:3]:
                oc.append({
                    "code": str(code).strip(),
                    "segment_number": block.get("segment_number"),
                    "coder": coder_id,
                    "transcript": block.get("transcript"),
                })
        write_json_txt(run_dir, f"open_coding_{coder_id}.txt", {"coder": coder_id, "open_codes": oc})

        _log(run_dir, f"[{coder_id}] Axial coding...")
        cats: List[Dict[str, Any]] = []
        for attempt in range(2):
            cats = run_axial_coding(
                sdk=sdk,
                open_codes=oc,
                max_categories=max_categories,
                study_background=study_background,
                analysis_mode=analysis_mode,
                theoretical_framework=theoretical_framework,
                attempts=1,
                timeout_s=75.0,
            )
            if cats and all(c.get("name") for c in cats):
                break
            _log(run_dir, f"[{coder_id}] Retrying axial coding (attempt {attempt + 2})...")
        # RAG + LLM: attach quotes by querying the vector index, then ask model to extract 1–3 verbatim quotes
        def _extract_quotes_for_category(name: str, description: str, contexts: List[str]) -> List[str]:
            QUOTES_SCHEMA = "{\"quotes\": [str]}"
            system = (
                "You extract short, faithful quotes for grounded-theory categories. "
                "Given a category name and description, and several transcript contexts, return 1–3 quotes (each 1–3 sentences) that are VERBATIM substrings of the provided contexts. "
                "Prefer interviewee speech; exclude interviewer prompts. JSON only."
            )
            user = json.dumps({
                "category": {"name": name, "description": description},
                "contexts": contexts,
                "schema": QUOTES_SCHEMA,
            }, ensure_ascii=False)
            data = sdk.run_json(system, user, schema_hint=QUOTES_SCHEMA, attempts=1, timeout_s=60.0)
            quotes = [str(q).strip() for q in (data.get("quotes") or []) if str(q).strip()]
            return quotes[:3]

        for cat in cats:
            query = (cat.get("name", "") or "") + "\n\n" + (cat.get("description", "") or "")
            from . import config as _cfg
            results = vindex.query(text=query, k=getattr(_cfg, 'RAG_K_DEFAULT', 3))
            contexts = [t for (t, m) in results if t and t.strip()]
            cat["supporting_quotes"] = _extract_quotes_for_category(cat.get("name", ""), cat.get("description", ""), contexts)
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
        _log(run_dir, "Integrating open codes...")
        # Agent-based synthesis of open codes with sources
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
        merged_categories: List[Dict[str, Any]] = []
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
        # Verify integrated quotes against entire corpus
        # No separate quote verification; quotes sourced via RAG above

        _log(run_dir, "Synthesizing integrated core story...")
        merged_core: Dict[str, Any] = {}
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

        # Build analysis summary
        per_coder_counts = {}
        for i, (oc, cats, sel) in enumerate(zip(per_coder_open, per_coder_categories, per_coder_selective), start=1):
            per_coder_counts[f"coder{i}"] = {
                "open_codes": len(oc),
                "categories": len(cats),
                "core_story": 1 if (sel.get("core_story") or "").strip() else 0,
            }
        integrated_counts = {
            "open_codes": len((open_synth.get("open_codes") or [])),
            "categories": len(merged_categories),
            "core_story": 1 if (merged_core.get("core_story") or "").strip() else 0,
        }
        params_out = {
            "coders": coders,
            "analysis_mode": analysis_mode,
            "cac_enabled": cac_enabled,
            "max_categories": max_categories,
            "segment_length": segment_len,
            "model": model,
        }
        # Deterministic summary (nothing more than required fields)
        mode_val = analysis_mode
        cac_val = str(bool(cac_enabled)).lower()
        settings = (
            f"Analysis settings: {coders} coders; analysis_mode = \"{mode_val}\"; "
            f"cac_enabled = {cac_val}; max_categories = {max_categories}; "
            f"segment_length = {segment_len}; model = \"{model}\"."
        )
        # Per-coder counts in a compact pattern
        if coders == 2:
            per_coder_line = (
                f"Per-coder counts (coder1, coder2): open_codes = {per_coder_counts['coder1']['open_codes']} each; "
                f"categories = {per_coder_counts['coder1']['categories']} each; core_story = {per_coder_counts['coder1']['core_story']} each."
            )
        else:
            per_coder_parts = []
            for i in range(1, coders + 1):
                pc = per_coder_counts.get(f"coder{i}", {})
                per_coder_parts.append(
                    f"coder{i}: open_codes={pc.get('open_codes',0)}, categories={pc.get('categories',0)}, core_story={pc.get('core_story',0)}"
                )
            per_coder_line = "Per-coder counts: " + "; ".join(per_coder_parts) + "."
        integrated_line = (
            "Integrated counts (post-integration): "
            f"open_codes = {integrated_counts['open_codes']}; categories = {integrated_counts['categories']}; core_story = {integrated_counts['core_story']}."
        )
        summary_text = f"{settings} {per_coder_line} {integrated_line}"
        write_json_txt(run_dir, "analysis_summary.txt", {"summary": summary_text})

        files = [f.name for f in sorted(run_dir.glob("*.txt"))]
        (run_dir / "result.json").write_text(json.dumps({"files": files}, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(run_dir, "Analysis complete. Outputs ready for download.")
    except Exception as e:
        _log(run_dir, f"Error during integration/report: {e}")
        files = [f.name for f in sorted(run_dir.glob("*.txt"))]
        (run_dir / "result.json").write_text(json.dumps({"files": files, "error": str(e)}, ensure_ascii=False, indent=2), encoding="utf-8")
