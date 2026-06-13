from __future__ import annotations

import os
import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from . import config
    from .evaluation_runner import FILE_REQUIREMENTS, generate_evaluation
except Exception:
    import config  # type: ignore
    from evaluation_runner import FILE_REQUIREMENTS, generate_evaluation  # type: ignore


app = Flask(__name__, template_folder="templates", static_folder="static")

EVALUATION_OUTPUT_DIR = os.getenv("CCA_EVALUATION_OUTPUT_DIR", "evaluation_generated")
ALLOWED_UPLOAD_EXTS = {".zip", ".txt", ".json"}
ALLOWED_EXTRACTED_EXTS = {".txt", ".json"}


def _safe_name(filename: str, fallback: str = "file") -> str:
    p = Path(filename)
    stem = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in p.stem).strip("-")
    stem = stem or fallback
    return f"{stem}{p.suffix.lower()}"


def _safe_extract_zip(zip_path: Path, dest: Path) -> int:
    dest = dest.resolve()
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            source_name = member.filename
            suffix = Path(source_name).suffix.lower()
            if suffix not in ALLOWED_EXTRACTED_EXTS:
                continue
            target = (dest / source_name).resolve()
            if not str(target).startswith(str(dest) + os.sep):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            count += 1
    return count


@app.route("/")
def index() -> str:
    return render_template(
        "evaluation_index.html",
        default_model=config.DEFAULT_MODEL,
        model_options=config.MODEL_OPTIONS,
        file_requirements=FILE_REQUIREMENTS,
    )


@app.route("/evaluate", methods=["POST"])
def evaluate() -> str:
    api_key = request.form.get("openai_api_key", "").strip()
    model = request.form.get("model", config.DEFAULT_MODEL).strip() or config.DEFAULT_MODEL
    if not api_key:
        return render_template(
            "evaluation_index.html",
            default_model=config.DEFAULT_MODEL,
            model_options=config.MODEL_OPTIONS,
            file_requirements=FILE_REQUIREMENTS,
            error="OpenAI API key is required for the machine evaluation.",
        ), 400

    files = [f for f in request.files.getlist("cca_outputs") if f and f.filename]
    if not files:
        return render_template(
            "evaluation_index.html",
            default_model=config.DEFAULT_MODEL,
            model_options=config.MODEL_OPTIONS,
            file_requirements=FILE_REQUIREMENTS,
            error="Upload a completed CCA output zip or output .txt/.json files.",
        ), 400

    eval_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    base = BASE_DIR / EVALUATION_OUTPUT_DIR / f"eval_{eval_id}"
    input_dir = base / "input"
    output_dir = base / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for upload in files:
        ext = Path(upload.filename).suffix.lower()
        if ext not in ALLOWED_UPLOAD_EXTS:
            continue
        safe = _safe_name(upload.filename, fallback="cca_output")
        target = input_dir / safe
        upload.save(target)
        if ext == ".zip":
            saved_count += _safe_extract_zip(target, input_dir)
        else:
            saved_count += 1

    if saved_count == 0:
        return render_template(
            "evaluation_index.html",
            default_model=config.DEFAULT_MODEL,
            model_options=config.MODEL_OPTIONS,
            file_requirements=FILE_REQUIREMENTS,
            error="No supported CCA output files were found. Upload .zip, .txt, or .json files.",
        ), 400

    generate_evaluation(
        input_dir=input_dir,
        output_dir=output_dir,
        api_key=api_key,
        model=model,
        use_llm=True,
    )
    return redirect(url_for("result", eval_id=eval_id))


@app.route("/result/<eval_id>")
def result(eval_id: str) -> str:
    base = BASE_DIR / EVALUATION_OUTPUT_DIR / f"eval_{eval_id}"
    return render_template(
        "evaluation_complete.html",
        eval_id=eval_id,
        output_dir=str(base / "output"),
        report_url=url_for("report", eval_id=eval_id),
        report_download_url=url_for("download", eval_id=eval_id, filename="evaluation_report.html"),
        json_url=url_for("download", eval_id=eval_id, filename="machine_evaluation.json"),
    )


@app.route("/report/<eval_id>")
def report(eval_id: str):
    base = BASE_DIR / EVALUATION_OUTPUT_DIR / f"eval_{eval_id}" / "output"
    return send_from_directory(base, "evaluation_report.html", as_attachment=False)


@app.route("/download/<eval_id>/<path:filename>")
def download(eval_id: str, filename: str):
    base = BASE_DIR / EVALUATION_OUTPUT_DIR / f"eval_{eval_id}" / "output"
    return send_from_directory(base, filename, as_attachment=True)


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT") or os.getenv("FLASK_RUN_PORT") or 5004)
    print(f"[CCA Evaluation UI] Listening on http://{host}:{port}")
    app.run(host=host, port=port, debug=True)
