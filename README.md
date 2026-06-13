# Constant Comparative Analysis Agent APP

An agent-based constant comparative analysis (CCA) tool with per-coder agents and an integrating agent. Uses ChromaDB RAG to attach faithful, verbatim quotes and provides a clean UI for progress and downloads.

## Features
- Multi-coder pipeline: incident comparisons → comparative categories → memoing → CCA synthesis
- Incident coding: per-segment incident labels + comparison notes against earlier incidents
- Hybrid incident memory: recent incidents + retrieved earlier cases + stable anchors/boundary cases
- Global incident reconciliation: second-pass full-dataset comparison before category building
- Comparative categories: data-driven category building with explicit contrasts, quotes attached via RAG
- Clean quote evidence: removes speaker labels, line numbers, demographic headers, and question fragments while preserving segment IDs and source context
- Robust transcript/document extraction: extracts PDFs page by page and records failed pages instead of dropping the whole file when one page fails
- Boundary/negative-case handling: each category records boundary cases or an explicit reason none were found
- Comparative memos: generates focused analytic memos with insights, tensions, and next steps
- Synthesis: produces a single-paragraph comparative narrative grounded in data comparisons
- Standalone evaluation app: upload completed CCA outputs separately to generate a machine-filled HTML audit report
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
- After completion, click `Download Full Run ZIP` on the results page. This zip can be uploaded directly into the standalone evaluation app.

## Standalone Evaluation
- Run the separate evaluation app: `PORT=5004 python evaluation_app.py`
- UI: open `http://localhost:5004`
- Upload a completed CCA run as a `.zip` whenever possible. The zip should include `segments_*.txt` so the evaluation report can show transcript context for segment IDs.
- If selecting files manually, include the generated analysis, incident, category, memo, synthesis, integrated, and `segments_*.txt` files.
- The evaluation upload page and generated HTML report include a file-to-criteria table explaining which files are used for each evaluation criterion.
- Zip uploads are parsed automatically; nested `.txt/.json` files are extracted, unsupported files are ignored, and legacy built-in evaluation outputs are ignored.
- Output: `evaluation_report.html` and `machine_evaluation.json` under `evaluation_generated/eval_<id>/output/`.
- The machine section evaluates six CCA-specific process criteria without ground truth. Five interpretive criteria are LLM-scored using two criterion-specific calls: one process-quality packet for constant comparison, boundary/negative cases, and category differentiation; one synthesis/integration packet for memo-to-synthesis coherence and cross-coder divergence integration. Traceability and quote evidence is code-scored with segment/source-context checks, loose 95% quote-to-source matching, and quote artifact checks. Output completeness is checked separately as a deterministic file inventory.
- The human section is intentionally blank. A qualitative researcher fills it after inspecting the original transcripts and CCA outputs.

## How It Works
- Segmentation: LangChain `RecursiveCharacterTextSplitter` with user-selectable chunk sizes (100, 500, 1000, 2000, 3000, 4000, or 5000 tokens; default 500).
- Vector DB: in-memory ChromaDB with OpenAI `text-embedding-3-large`. Retrieval `k` is fixed (`config.RAG_K_DEFAULT`).
- Incident coding: per segment incident labels, comparison notes, and analytic memoing with hybrid prior context:
  recent incidents preserve local continuity, retrieved incidents surface older/full-history analogues or contrasts, and anchor incidents keep early/boundary cases visible.
- Global reconciliation: after first-pass incident coding, a second-pass agent clusters semantically similar labels and records cross-transcript, early-late, and negative/boundary comparisons before categories are built.
- Comparative categories: clusters incidents into data-driven categories highlighting contrasts and boundary/negative cases; each category must include a boundary case or a reason none was found.
- Quote verification: extracted quotes are checked against retrieved source context; a deterministic cleaner removes transcript artifacts and writes `supporting_quote_evidence` with `segment_id`, `quote`, `source_context`, and cleanup flags. If verification fails, the pipeline falls back to source segments listed as category support.
- Comparative memos: 3–6 memos outlining insights, tensions, and theoretical sampling leads.
- CCA synthesis: single-paragraph comparative summary describing how the major categories relate (optionally noting data-supported condition–action–consequence links when CAC is enabled).
- Integration: merges incident patterns, category matrix, memo digest, and a unified CCA summary. Category integration now prefers mid-grained 5-7 category structures when the coder outputs contain enough distinct mechanisms.
- Summary: deterministic settings + counts in `analysis_summary.txt`, including separate incident-note and incident-label counts.

## Outputs
- Per transcript: `segments_<transcript>.txt`
- Per coder: `incident_coding_<coder>.txt`, `incident_reconciliation_<coder>.txt`, `category_comparisons_<coder>.txt`, `cca_memos_<coder>.txt`, `cca_synthesis_<coder>.txt`
- Integrated (two coders): `integrated_incident_patterns.txt`, `integrated_categories.txt`, `integrated_memo_digest.txt`, `integrated_cca_summary.txt`
- Summary: `analysis_summary.txt`

## Configuration
- Defaults: see `config.py` (model, segment length, RAG k, output dir)
- CCA memory settings: `CCA_RECENT_INCIDENT_CONTEXT_LIMIT`, `CCA_RETRIEVED_INCIDENT_CONTEXT_LIMIT`, `CCA_ANCHOR_INCIDENT_CONTEXT_LIMIT`, `CCA_GLOBAL_RECONCILIATION_ENABLED`
- API key: UI field (required) or env `OPENAI_API_KEY` for non-UI calls. Form-submitted keys are stored only in a transient secrets directory and removed once the worker starts (never written to downloadable outputs).
- Port: default 5000 (override with `PORT`)

## Notes
- Summaries are always generated and used (no toggle).
- Refinement is preview‑only; the worker does not auto‑refine.
- `supporting_quotes` contains cleaned quote strings for backward compatibility. Use `supporting_quote_evidence` for audit because it includes segment IDs, source context, and cleanup flags.
- PDF extraction is text-based. The app keeps partial text when only some pages fail; scanned/image-only transcript PDFs still require OCR text.

## Deployment Notes
- Configure `REDIS_URL` (and optionally `RQ_QUEUE_NAME`) so the web app can enqueue jobs onto Redis-backed RQ queues.
- For single-service deployments, use the provided `render_start.sh` in the repo root. Set the Render start command to `./render_start.sh` so one container launches both Gunicorn and the RQ worker, and keep `REDIS_URL`/`RQ_QUEUE_NAME` aligned across instances.
- If you prefer dedicated services, run a background worker with `rq worker --url $REDIS_URL constant_comparative_analysis` (or your chosen queue name) so jobs execute outside the HTTP request path.
- Ensure both web and worker processes point `GT_OUTPUT_DIR` to the same writable location (e.g., a shared persistent disk) so the UI can stream progress and downloads.
- Deploy the standalone evaluator as a separate service if needed, with start command `gunicorn evaluation_app:app --bind 0.0.0.0:$PORT`.
