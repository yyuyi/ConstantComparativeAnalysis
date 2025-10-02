from __future__ import annotations

import os
from typing import Optional

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - redis is optional during local dev
    redis = None  # type: ignore

try:
    from rq import Queue  # type: ignore
except Exception:  # pragma: no cover - rq optional when queue disabled
    Queue = None  # type: ignore

try:
    from . import config  # type: ignore
except Exception:  # pragma: no cover - support top-level imports
    import config  # type: ignore


def _redis_url() -> Optional[str]:
    url = getattr(config, "REDIS_URL", None)
    if url:
        return url
    return os.getenv("REDIS_URL")


def get_queue() -> Optional["Queue"]:
    """Return an RQ queue if Redis/RQ are available; otherwise ``None``."""
    if redis is None or Queue is None:
        return None
    url = _redis_url()
    if not url:
        return None
    connection = redis.from_url(url)
    return Queue(
        name=getattr(config, "RQ_QUEUE_NAME", "constant_comparative_analysis"),
        connection=connection,
        default_timeout=getattr(config, "RQ_DEFAULT_TIMEOUT", 900),
    )
