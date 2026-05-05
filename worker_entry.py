from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

try:
    from .worker import run_job
except Exception:  # pragma: no cover - allow top-level import when package layout differs
    from worker import run_job  # type: ignore


def _log(run_dir: Path, msg: str) -> None:
    ts = datetime.utcnow().strftime("%H:%M:%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "progress.log", "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def run_queued_job(run_dir: str) -> None:
    """Adapter callable for RQ workers."""
    path = Path(run_dir)
    _log(path, "Worker picked up job.")
    try:
        run_job(path)
    except Exception as exc:
        _log(path, f"Worker failed: {exc}")
        try:
            (path / "result.json").write_text(
                json.dumps({"files": [], "error": str(exc)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        raise
