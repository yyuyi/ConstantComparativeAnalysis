from __future__ import annotations

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List
import threading

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


app = Flask(__name__, template_folder="templates", static_folder="static")


def _safe_int(v: str | None, default: int) -> int:
    try:
        return int(v) if v else default
    except Exception:
        return default


@app.route("/")
def index() -> str:
    return render_template(
        "index.html",
        default_model=config.DEFAULT_MODEL,
        model_options=config.MODEL_OPTIONS,
        default_segment_len=config.SEGMENT_LENGTH_DEFAULT,
        default_max_categories=config.MAX_CATEGORIES_DEFAULT,
    )


@app.route("/start", methods=["POST"])
def start() -> str:
    # Read inputs
    study_background = request.form.get("study_background", "").strip()
    coders = max(1, _safe_int(request.form.get("coders", "1"), 1))
    analysis_mode = request.form.get("analysis_mode", "classic").strip()
    theoretical_framework = request.form.get("theoretical_framework", "").strip()
    cac_enabled = request.form.get("cac_enabled") == "on"
    # Max categories: allow auto (no limit) if checkbox set or value <= 0
    auto_categories = request.form.get("auto_categories") == "on"
    if auto_categories:
        max_categories = 0
    else:
        max_categories = _safe_int(request.form.get("max_categories"), config.MAX_CATEGORIES_DEFAULT)
    segment_len = _safe_int(request.form.get("segment_length"), config.SEGMENT_LENGTH_DEFAULT)
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
        "api_key": api_key,
        "uploads": uploads,
    }
    (run_dir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    threading.Thread(target=run_job, args=(run_dir,), daemon=True).start()
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
        return jsonify({"ok": True, **out})
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
