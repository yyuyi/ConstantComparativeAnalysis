import os

DEFAULT_MODEL = "gpt-5-nano"
MODEL_OPTIONS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
]
SEGMENT_LENGTH_OPTIONS = [500, 1000, 2000, 3000, 4000, 5000]
SEGMENT_LENGTH_DEFAULT = 1000
# If <= 0, treat as AUTO (no max; agent decides best number)
MAX_CATEGORIES_DEFAULT = 0
# Text limits to keep context concise for the agents
STUDY_BACKGROUND_WORD_LIMIT = 1000
THEORETICAL_FRAMEWORK_WORD_LIMIT = 1000
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
RQ_QUEUE_NAME = os.getenv("RQ_QUEUE_NAME", "constant_comparative_analysis")
# Allow multi-hour jobs by default; override via env if you need shorter windows.
RQ_DEFAULT_TIMEOUT = int(os.getenv("RQ_DEFAULT_TIMEOUT", str(60 * 60 * 5)))
# Writable output directory. On Render or other PaaS, prefer a writable path (e.g., /tmp/generated).
# Overridable via env var GT_OUTPUT_DIR.
OUTPUT_DIR = os.getenv("GT_OUTPUT_DIR", "generated")
# RAG retrieval top-k for category-to-quote matching
RAG_K_DEFAULT = 2

# Quality toggles
# UI-based refinement happens before run; default disabled.
REFINE_CONTEXT_ENABLED_DEFAULT = False

# Concurrency controls (bounded via asyncio.Semaphore in worker)
TIER_CONCURRENCY_PRESET = {
    1: {"summary": 2, "open": 12},
    2: {"summary": 20, "open": 120},
    3: {"summary": 40, "open": 250},
    4: {"summary": 100, "open": 600},
    5: {"summary": 1800, "open": 9000},
}


def _sanitize_tier(value: int) -> int:
    try:
        val = int(value)
    except Exception:
        return 1
    return val if val in TIER_CONCURRENCY_PRESET else 1


def _sanitize_concurrency(value: str | None, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        parsed = int(value)
        return parsed if parsed > 0 else fallback
    except Exception:
        return fallback


DEFAULT_API_TIER = _sanitize_tier(int(os.getenv("GT_API_TIER", "1")))


def concurrency_for_tier(tier: int | None) -> dict:
    """Return summary/open concurrency values for the requested tier with env overrides."""
    resolved_tier = _sanitize_tier(tier if tier is not None else DEFAULT_API_TIER)
    base = TIER_CONCURRENCY_PRESET[resolved_tier]
    return {
        "tier": resolved_tier,
        "summary": _sanitize_concurrency(os.getenv("GT_SUMMARY_CONCURRENCY"), base["summary"]),
        "open": _sanitize_concurrency(os.getenv("GT_OPEN_CODING_CONCURRENCY"), base["open"]),
    }


_DEFAULT_CONCURRENCY = concurrency_for_tier(DEFAULT_API_TIER)
SUMMARY_CONCURRENCY = _DEFAULT_CONCURRENCY["summary"]
OPEN_CODING_CONCURRENCY = _DEFAULT_CONCURRENCY["open"]

# Expose tier options so the UI can render a selector.
API_TIER_OPTIONS = tuple(sorted(TIER_CONCURRENCY_PRESET.keys()))