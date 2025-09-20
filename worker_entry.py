from __future__ import annotations

from pathlib import Path

try:
    from .worker import run_job
except Exception:  # pragma: no cover - allow top-level import when package layout differs
    from worker import run_job  # type: ignore


def run_queued_job(run_dir: str) -> None:
    """Adapter callable for RQ workers."""
    path = Path(run_dir)
    run_job(path)
