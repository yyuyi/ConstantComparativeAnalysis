from __future__ import annotations

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List
import threading
import re

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import sys

# Make imports work both as package and as top-level module (Render/Gunicorn)
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from . import config  # package import
except Exception:
    try:
        import config     # top-level
    except Exception:
        import configure as config  # optional alias

try:
    from .worker import run_job
except Exception:
    from worker import run_job

try:
    from .task_queue import get_queue  # type: ignore
except Exception:
    try:
        from task_queue import get_queue  # type: ignore
    except Exception:
        def get_queue():
            return None


app = Flask(__name__, template_folder="templates", static_folder="static")


def _append_progress_line(run_dir: Path, message: str) -> None:
    ts = datetime.utcnow().strftime("%H:%M:%S")
    with open(run_dir / "progress.log", "a", encoding="utf-8") as log:
        log.write(f"[{ts}] {message}\n")


def _safe_int(v: str | None, default: int) -> int:
    try:
        return int(v) if v else default
    except Exception:
        return default


_WORD_RE = re.compile(r"\S+")


def _enforce_word_limit(text: str, limit: int) -> str:
    if limit <= 0:
        return text
    matches = list(_WORD_RE.finditer(text))
    if len(matches) <= limit:
        return text
    cutoff = matches[limit - 1].end()
    return text[:cutoff].rstrip()


@app.route("/")
def index() -> str:
    return render_template(
        "index.html",
        default_model=config.DEFAULT_MODEL,
        model_options=config.MODEL_OPTIONS,
        default_segment_len=config.SEGMENT_LENGTH_DEFAULT,
        segment_length_options=getattr(config, "SEGMENT_LENGTH_OPTIONS", [config.SEGMENT_LENGTH_DEFAULT]),
        default_max_categories=config.MAX_CATEGORIES_DEFAULT,
    )


@app.route("/start", methods=["POST"])
def start() -> str:
    # Read inputs
    study_background = request.form.get("study_background", "").strip()
    background_limit = getattr(config, "STUDY_BACKGROUND_WORD_LIMIT", 1000)
    study_background = _enforce_word_limit(study_background, background_limit)
    coders = _safe_int(request.form.get("coders", "1"), 1)
    coders = max(1, min(2, coders))
    analysis_mode = request.form.get("analysis_mode", "classic").strip()
    theoretical_framework = request.form.get("theoretical_framework", "").strip()
    tf_limit = getattr(config, "THEORETICAL_FRAMEWORK_WORD_LIMIT", 1000)
    theoretical_framework = _enforce_word_limit(theoretical_framework, tf_limit)
    cac_enabled = request.form.get("cac_enabled") == "on"
    # Max categories: allow auto (no limit) if checkbox set or value <= 0
    auto_categories = request.form.get("auto_categories") == "on"
    if auto_categories:
        max_categories = 0
    else:
        max_categories = _safe_int(request.form.get("max_categories"), config.MAX_CATEGORIES_DEFAULT)
    segment_len = _safe_int(request.form.get("segment_length"), config.SEGMENT_LENGTH_DEFAULT)
    allowed_segments = set(getattr(config, "SEGMENT_LENGTH_OPTIONS", [config.SEGMENT_LENGTH_DEFAULT]))
    if segment_len not in allowed_segments:
        segment_len = config.SEGMENT_LENGTH_DEFAULT
    model = request.form.get("model", config.DEFAULT_MODEL).strip()
    api_key = request.form.get("openai_api_key", "").strip()

    # Create run directory
    base = Path(__file__).parent / config.OUTPUT_DIR
    base.mkdir(parents=True, exist_ok=True)
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = base / f"run_{run_id}"
    inputs_dir = run_dir / "inputs"
    transcripts_dir = inputs_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    # Persist text inputs for the worker
    (inputs_dir / "study_background.txt").write_text(study_background, encoding="utf-8")
    (inputs_dir / "theoretical_framework.txt").write_text(theoretical_framework, encoding="utf-8")

    uploads: List[str] = []
    allowed_exts = {".txt", ".pdf", ".docx"}
    for file in request.files.getlist("transcripts"):
        if file and file.filename:
            p = Path(file.filename)
            safe = p.stem
            ext = p.suffix.lower()
            if ext not in allowed_exts:
                # Skip unsupported formats (.doc etc.)
                continue
            safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in safe).strip("-") or "file"
            fname = f"{safe}{ext}"
            (transcripts_dir / fname).write_bytes(file.read())
            uploads.append(fname)

    params = {
        "coders": coders,
        "analysis_mode": analysis_mode,
        "cac_enabled": cac_enabled,
        "max_categories": max_categories,
        "auto_categories": auto_categories,
        "segment_length": segment_len,
        "model": model,
        "uploads": uploads,
    }
    # Persist the API key separately and do not include it in params.json
    (run_dir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    # Store the API key in a protected secrets directory outside the downloadable outputs.
    secrets_dir = base / "_secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    secret_path = secrets_dir / f"{run_id}.secret"
    secret_path.write_text(api_key, encoding="utf-8")
    secret_path.chmod(0o600)

    queue = None
    enqueue_error: str | None = None
    try:
        queue = get_queue()
    except Exception as exc:
        enqueue_error = str(exc)
        queue = None

    if queue:
        try:
            import importlib

            job_path = "constant_comparative_analysis_agent.worker_entry.run_queued_job"
            try:
                importlib.import_module("constant_comparative_analysis_agent.worker_entry")
            except ModuleNotFoundError:
                job_path = "worker_entry.run_queued_job"

            queue.enqueue(job_path, str(run_dir), job_id=run_id, at_front=False)
            _append_progress_line(run_dir, "Job enqueued for background worker.")
        except Exception as exc:
            enqueue_error = str(exc)
            queue = None

    if not queue:
        fallback_msg = "Queue unavailable; running job in local background thread."
        if enqueue_error:
            fallback_msg = f"Queue unavailable ({enqueue_error}); running job in local background thread."
        _append_progress_line(run_dir, fallback_msg)
        threading.Thread(target=run_job, args=(run_dir,), daemon=True).start()
    else:
        _append_progress_line(run_dir, "Awaiting worker pickup...")

    return redirect(url_for("status", run_id=run_id))


@app.route("/refine", methods=["POST"])
def refine():
    """Refine background/framework; return refined text so the user can review/edit before running."""
    try:
        payload = request.get_json(force=True) or {}
        study_background = (payload.get("study_background") or "").strip()
        theoretical_framework = (payload.get("theoretical_framework") or "").strip()
        analysis_mode = (payload.get("analysis_mode") or "classic").strip()
        api_key = (payload.get("openai_api_key") or "").strip()
        if not api_key:
            return jsonify({"ok": False, "error": "OPENAI_API_KEY is required for refinement."}), 400
        bg_limit = getattr(config, "STUDY_BACKGROUND_WORD_LIMIT", 1000)
        tf_limit = getattr(config, "THEORETICAL_FRAMEWORK_WORD_LIMIT", 1000)
        study_background = _enforce_word_limit(study_background, bg_limit)
        theoretical_framework = _enforce_word_limit(theoretical_framework, tf_limit)

        # Initialize SDK + refine function
        try:
            from .agents.sdk import AgentSDK
            from .agents.coder_agent import refine_context as _refine
        except Exception:
            from agents.sdk import AgentSDK
            from agents.coder_agent import refine_context as _refine

        sdk = AgentSDK(model=config.DEFAULT_MODEL, api_key=api_key)
        out = _refine(
            sdk=sdk,
            study_background=study_background,
            theoretical_framework=theoretical_framework,
            analysis_mode=analysis_mode,
            attempts=1,
            timeout_s=45.0,
        )
        if isinstance(out, dict):
            if "study_background" in out:
                out["study_background"] = _enforce_word_limit(out.get("study_background", ""), bg_limit)
            if "theoretical_framework" in out:
                out["theoretical_framework"] = _enforce_word_limit(out.get("theoretical_framework", ""), tf_limit)
        return jsonify({"ok": True, **(out or {})})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/status/<run_id>")
def status(run_id: str):
    return render_template(
        "results.html",
        run_id=run_id,
        output_dir=str(Path(config.OUTPUT_DIR) / f"run_{run_id}"),
    )


@app.route("/progress/<run_id>")
def progress(run_id: str):
    run_dir = Path(__file__).parent / config.OUTPUT_DIR / f"run_{run_id}"
    log_path = run_dir / "progress.log"
    result_path = run_dir / "result.json"
    lines: List[str] = []
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            lines = []
    files: List[str] = []
    complete = False
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            files = data.get("files", [])
            complete = True
        except Exception:
            pass
    return {"lines": lines, "complete": complete, "files": files}


@app.route("/download/<run_id>/<path:filename>")
def download(run_id: str, filename: str):
    base = Path(__file__).parent / config.OUTPUT_DIR / f"run_{run_id}"
    return send_from_directory(base, filename, as_attachment=True)


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT") or os.getenv("FLASK_RUN_PORT") or os.getenv("LIGHTNING_PORT") or 5000)
    print(f"[Agent UI] Listening on http://{host}:{port}")
    app.run(host=host, port=port, debug=True)
