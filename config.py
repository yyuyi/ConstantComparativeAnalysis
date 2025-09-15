import os

DEFAULT_MODEL = "gpt-5-mini"
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
SEGMENT_LENGTH_DEFAULT = 500
# If <= 0, treat as AUTO (no max; agent decides best number)
MAX_CATEGORIES_DEFAULT = 0
# Writable output directory. On Render or other PaaS, prefer a writable path (e.g., /tmp/generated).
# Overridable via env var GT_OUTPUT_DIR.
OUTPUT_DIR = os.getenv("GT_OUTPUT_DIR", "generated")
# RAG retrieval top-k for category-to-quote matching
RAG_K_DEFAULT = 2

# Quality toggles
# UI-based refinement happens before run; default disabled.
REFINE_CONTEXT_ENABLED_DEFAULT = False
