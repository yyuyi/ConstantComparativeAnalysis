#!/usr/bin/env bash

set -euo pipefail

# Ensure our package is discoverable by rq when launched from the repo root.
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

QUEUE_NAME="${RQ_QUEUE_NAME:-grounded_theory}"
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
PORT="${PORT:-5000}"
WEB_CONCURRENCY="${WEB_CONCURRENCY:-2}"
GUNICORN_THREADS="${GUNICORN_THREADS:-4}"
GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-900}"

# Make sure the configured output directory exists before jobs start.
python - <<'PY'
from grounded_theory_agent import config
from pathlib import Path

Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
PY

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

echo "Starting Gunicorn -> port=${PORT} workers=${WEB_CONCURRENCY} threads=${GUNICORN_THREADS} timeout=${GUNICORN_TIMEOUT}" >&2
gunicorn grounded_theory_agent.app:app \
  --bind "0.0.0.0:${PORT}" \
  --workers "${WEB_CONCURRENCY}" \
  --threads "${GUNICORN_THREADS}" \
  --timeout "${GUNICORN_TIMEOUT}" \
  --graceful-timeout "${GUNICORN_TIMEOUT}" \
  --preload

WEB_STATUS=$?

# Explicit cleanup in case Gunicorn exits normally before trap fires.
cleanup

exit "${WEB_STATUS}"
