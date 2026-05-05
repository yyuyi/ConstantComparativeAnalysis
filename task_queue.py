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
    from rq import Worker  # type: ignore
except Exception:  # pragma: no cover - rq optional when queue disabled
    Worker = None  # type: ignore

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
    connection.ping()
    return Queue(
        name=getattr(config, "RQ_QUEUE_NAME", "constant_comparative_analysis"),
        connection=connection,
        default_timeout=getattr(config, "RQ_DEFAULT_TIMEOUT", 900),
    )


def has_active_worker(queue: "Queue") -> bool:
    """Return whether Redis has a live RQ worker listening to this queue."""
    if Worker is None:
        return True
    try:
        workers = Worker.all(connection=queue.connection)
    except Exception:
        return False
    queue_name = queue.name
    for worker in workers:
        try:
            names = worker.queue_names()
        except Exception:
            names = [getattr(q, "name", str(q)) for q in getattr(worker, "queues", [])]
        if queue_name in names:
            return True
    return False
