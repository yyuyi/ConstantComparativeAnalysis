#!/usr/bin/env bash

set -euo pipefail

# Make the repo root importable
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

QUEUE_NAME="${RQ_QUEUE_NAME:-constant_comparative_analysis}"
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
PORT="${PORT:-5000}"
WEB_CONCURRENCY="${WEB_CONCURRENCY:-2}"
GUNICORN_THREADS="${GUNICORN_THREADS:-4}"
# Keep Gunicorn alive for long-running analysis jobs (default 5 hours).
GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-18000}"

# Ensure the output directory exists before jobs run.
python - <<'PY'
import importlib
from pathlib import Path

module = None
for name in ("config", "constant_comparative_analysis_agent.config"):
    try:
        module = importlib.import_module(name)
        break
    except ModuleNotFoundError:
        continue

if module is None:
    raise SystemExit("Unable to import config module; check PYTHONPATH/start command.")

output_dir = getattr(module, "OUTPUT_DIR", "generated")
Path(output_dir).mkdir(parents=True, exist_ok=True)
PY

APP_MODULE=$(python - <<'PY'
import importlib.util

for name in ("constant_comparative_analysis_agent.app", "app"):
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        continue
    if spec is not None:
        print(f"{name}:app")
        break
else:
    raise SystemExit("Unable to locate Flask app module.")
PY
)

echo "Starting RQ worker -> queue=${QUEUE_NAME} redis=${REDIS_URL}" >&2
rq worker --url "${REDIS_URL}" "${QUEUE_NAME}" &
WORKER_PID=$!

cleanup() {
  if kill -0 "${WORKER_PID}" >/dev/null 2>&1; then
    kill "${WORKER_PID}" >/dev/null 2>&1 || true
    wait "${WORKER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting Gunicorn (${APP_MODULE}) -> port=${PORT} workers=${WEB_CONCURRENCY} threads=${GUNICORN_THREADS} timeout=${GUNICORN_TIMEOUT}" >&2
gunicorn "${APP_MODULE}" \
  --bind "0.0.0.0:${PORT}" \
  --workers "${WEB_CONCURRENCY}" \
  --threads "${GUNICORN_THREADS}" \
  --timeout "${GUNICORN_TIMEOUT}" \
  --graceful-timeout "${GUNICORN_TIMEOUT}" \
  --preload

WEB_STATUS=$?

cleanup

exit "${WEB_STATUS}"
