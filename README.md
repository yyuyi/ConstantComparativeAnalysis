# Constant Comparative Analysis Agent APP

An agent-based constant comparative analysis (CCA) tool with per-coder agents and an integrating agent. Uses ChromaDB RAG to attach faithful, verbatim quotes and provides a clean UI for progress and downloads.

## Features
- Multi-coder pipeline: incident comparisons → comparative categories → memoing → CCA synthesis
- Incident coding: per-segment incident labels + comparison notes against earlier incidents
- Comparative categories: data-driven category building with explicit contrasts, quotes attached via RAG
- Comparative memos: generates focused analytic memos with insights, tensions, and next steps
- Synthesis: produces a single-paragraph comparative narrative grounded in data comparisons
- Optional refine preview: polish background/framework before running (preview in UI)
- File types: `.txt`, `.pdf` (PyPDF), `.docx` (python-docx); `.doc` is not supported
- Academic‑style UI with progress log and downloadable outputs

## Quickstart
- Install: `pip install -r requirements.txt`
- Run: `PORT=5001 python app.py`
- UI: open `http://localhost:5001`
- Provide:
  - OpenAI API key (required)
  - Study background, transcripts (.txt/.pdf/.docx), coders, analysis mode, optional CAC
  - Optionally click “Refine Text” to preview polished background/framework before running

## How It Works
- Segmentation: LangChain `RecursiveCharacterTextSplitter` with user-selectable chunk sizes (500, 1000, 2000, 3000, 4000, or 5000 tokens; default 1000).
- Vector DB: in-memory ChromaDB with OpenAI `text-embedding-3-small`. Retrieval `k` is fixed (`config.RAG_K_DEFAULT`).
- Incident coding: per segment incident labels, comparison notes, and analytic memoing with prior incidents provided for context.
- Comparative categories: clusters incidents into data-driven categories highlighting contrasts; RAG attaches 1–3 verbatim quotes (trimmed to ≤ 3 sentences) per category.
- Comparative memos: 3–6 memos outlining insights, tensions, and theoretical sampling leads.
- CCA synthesis: single-paragraph comparative summary describing how the major categories relate (optionally noting data-supported condition–action–consequence links when CAC is enabled).
- Integration: merges incident patterns, category matrix, memo digest, and a unified CCA summary.
- Summary: deterministic settings + counts in `analysis_summary.txt`.

## Outputs
- Per transcript: `segments_<transcript>.txt`
- Per coder: `incident_coding_<coder>.txt`, `category_comparisons_<coder>.txt`, `cca_memos_<coder>.txt`, `cca_synthesis_<coder>.txt`
- Integrated (two coders): `integrated_incident_patterns.txt`, `integrated_categories.txt`, `integrated_memo_digest.txt`, `integrated_cca_summary.txt`
- Summary: `analysis_summary.txt`

## Configuration
- Defaults: see `grounded_theory_agent/config.py` (model, segment length, RAG k, output dir)
- API key: UI field (required) or env `OPENAI_API_KEY` for non-UI calls. Form-submitted keys are stored only in a transient secrets directory and removed once the worker starts (never written to downloadable outputs).
- Port: default 5000 (override with `PORT`)

## Notes
- Summaries are always generated and used (no toggle).
- Refinement is preview‑only; the worker does not auto‑refine.
- Quotes are always verbatim substrings of the source, retrieved via RAG during the comparative category stage and trimmed to ≤ 3 sentences.

## Deployment Notes
- Configure `REDIS_URL` (and optionally `RQ_QUEUE_NAME`) so the web app can enqueue jobs onto Redis-backed RQ queues.
- For single-service deployments, use the provided `render_start.sh` in the repo root. Set the Render start command to `./render_start.sh` so one container launches both Gunicorn and the RQ worker, and keep `REDIS_URL`/`RQ_QUEUE_NAME` aligned across instances.
- If you prefer dedicated services, run a background worker with `rq worker --url $REDIS_URL grounded_theory` (or your chosen queue name) so jobs execute outside the HTTP request path.
- Ensure both web and worker processes point `GT_OUTPUT_DIR` to the same writable location (e.g., a shared persistent disk) so the UI can stream progress and downloads.
