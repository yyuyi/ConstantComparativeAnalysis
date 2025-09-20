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
SEGMENT_LENGTH_OPTIONS = [1000, 2000, 3000, 4000, 5000]
SEGMENT_LENGTH_DEFAULT = 3000
# If <= 0, treat as AUTO (no max; agent decides best number)
MAX_CATEGORIES_DEFAULT = 0
# Text limits to keep context concise for the agents
STUDY_BACKGROUND_WORD_LIMIT = 1000
THEORETICAL_FRAMEWORK_WORD_LIMIT = 1000
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
RQ_QUEUE_NAME = os.getenv("RQ_QUEUE_NAME", "grounded_theory")
RQ_DEFAULT_TIMEOUT = int(os.getenv("RQ_DEFAULT_TIMEOUT", "900"))
# Writable output directory. On Render or other PaaS, prefer a writable path (e.g., /tmp/generated).
# Overridable via env var GT_OUTPUT_DIR.
OUTPUT_DIR = os.getenv("GT_OUTPUT_DIR", "generated")
# RAG retrieval top-k for category-to-quote matching
RAG_K_DEFAULT = 2

# Quality toggles
# UI-based refinement happens before run; default disabled.
REFINE_CONTEXT_ENABLED_DEFAULT = False
