# Grounded-Theory Agent App

An agent‑based grounded‑theory analysis tool with per‑coder agents and an integrating agent. Uses ChromaDB RAG to attach faithful, verbatim quotes and provides a clean UI for progress and downloads.

## Features
- Multi‑coder pipeline: open → axial → selective coding per coder
- Open coding: 1–3 concise codes per segment; supporting quotes are attached later during axial coding via RAG
- Axial coding: RAG‑sourced quotes (1–3 sentences) added in a single batched call
- Summaries: comprehensive per‑transcript summaries inform axial and selective coding
- Categories: Auto mode (default) or cap to ≤ N
- Optional refine preview: polish background/framework before running (preview in UI)
- File types: `.txt`, `.pdf` (PyPDF), `.docx` (python-docx); `.doc` is not supported
- Academic‑style UI with progress log and downloadable outputs

## Quickstart
- Install: `pip install -r grounded_theory_agent/requirements.txt`
- Run: `PORT=5000 python -m grounded_theory_agent.app`
- UI: open `http://localhost:5000`
- Provide:
  - OpenAI API key (required)
  - Study background, transcripts (.txt/.pdf/.docx), coders, analysis mode, optional CAC
  - Optionally click “Refine Text” to preview polished background/framework before running

## How It Works
- Segmentation: LangChain `RecursiveCharacterTextSplitter` with user-selectable chunk sizes (1000, 2000, 3000, 4000, or 5000 tokens; default 3000).
- Vector DB: in‑memory ChromaDB with OpenAI `text-embedding-3-small`. Retrieval `k` is fixed (`config.RAG_K_DEFAULT`).
- Open coding: returns 1–3 concise codes per segment (no quotes). Quotes are retrieved later for categories during axial coding.
- Axial coding: clusters codes into categories with names/descriptions and members `{transcript, segment_number}`; RAG attaches 1–3 quotes per category (trimmed to ≤ 3 sentences). No quotes are passed to the axial prompt itself.
- Selective coding: multi‑paragraph core story (CAC‑aware when enabled), using only category quotes and summaries as background.
- Integration: merges open codes (simple list), categories, and a unified core story.
- Summary: deterministic settings + counts in `analysis_summary.txt`.

## Outputs
- Per transcript: `segments_<transcript>.txt`
- Per coder: `open_coding_<coder>.txt` (codes only), `axial_coding_<coder>.txt`, `selective_coding_<coder>.txt`
- Integrated: `integrated_open_codes.txt` `{ "open_codes": [str] }`, `integrated_categories.txt`, `integrated_core_story.txt`
- Summary: `analysis_summary.txt`

## Configuration
- Defaults: see `grounded_theory_agent/config.py` (model, segment length, RAG k, output dir)
- API key: UI field (required) or env `OPENAI_API_KEY` for non-UI calls. Form-submitted keys are stored only in a transient secrets directory and removed once the worker starts (never written to downloadable outputs).
- Port: default 5000 (override with `PORT`)

## Notes
- Summaries are always generated and used (no toggle).
- Refinement is preview‑only; the worker does not auto‑refine.
- Quotes are always verbatim substrings of the source, retrieved via RAG during axial coding and trimmed to ≤ 3 sentences.
