from __future__ import annotations

from pathlib import Path

from .worker import run_job


def run_queued_job(run_dir: str) -> None:
    """Adapter callable for RQ workers."""
    path = Path(run_dir)
    run_job(path)
