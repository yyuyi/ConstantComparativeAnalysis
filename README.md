# Grounded-Theory Agent App

A lightweight, agent‑based grounded‑theory analysis tool with per‑coder agents and an integrating agent. It uses retrieval‑augmented generation (RAG) to select faithful quotes from the original transcripts.

## Features
- Multi‑coder pipeline: open → axial → selective coding per coder
- Integration: open codes (with canonical + aliases + sources), categories, core story (CAC‑aware)
- RAG quotes: ChromaDB + OpenAI `text-embedding-3-large` to retrieve verbatim quotes
- Academic‑style UI with progress log and downloadable outputs

## Quickstart
- Install deps: `pip install -r grounded_theory_agent/requirements.txt`
- Run the app: `PORT=5000 python -m grounded_theory_agent.app` (or any open port)
- Open the UI and provide:
  - OpenAI API key (used for this run only)
  - Study background, transcripts (.txt, .pdf, .doc, .docx), number of coders, analysis mode (Classic/Interpretive/Constructionist), optional CAC

## Getting Started
Option A — Makefile
```
cd grounded_theory_agent
make install
make run PORT=5000
```

Option B — Manual commands
```
pip install -r grounded_theory_agent/requirements.txt
PORT=5000 python -m grounded_theory_agent.app
```

Optional: Screenshots or GIF
- Record a short GIF of the flow (inputs → status → downloads), place it at `grounded_theory_agent/static/overview.gif`, then add:
  `![Overview](static/overview.gif)`

## How It Works
- Segmentation: LangChain `RecursiveCharacterTextSplitter` creates sentence‑respecting segments (default 500 tokens per segment; user can change in UI).
- Vector DB: ChromaDB stores segment embeddings with transcript filename (with extension) + segment_number metadata. Retrieval `k` is fixed in code (see `config.RAG_K_DEFAULT`).
- Open coding: agents emit 1–3 codes per segment, in transcript order, referencing transcript + segment number.
- Axial coding: agents cluster open codes and write category names/descriptions + members as `{transcript, segment_number}`. The orchestrator uses RAG and then asks the LLM to extract 1–3 short, verbatim quotes (1–3 sentences) from retrieved contexts.
- Synthesis: integrates open codes (simple list), categories, and a single core story.
- Summary: deterministic analysis summary with settings and counts (nothing more).

## Outputs
- Per transcript: `segments_<transcript>.txt`
- Per coder: `open_coding_<coder>.txt`, `axial_coding_<coder>.txt`, `selective_coding_<coder>.txt`
- Integrated: `integrated_open_codes.txt` (simple `{ "open_codes": [str] }`), `integrated_categories.txt`, `integrated_core_story.txt`
- Summary: `analysis_summary.txt`

## Configuration
- Default model and segment length: see `grounded_theory_agent/config.py`
- OpenAI key: UI field (preferred), env var `OPENAI_API_KEY`, or `instructions/openai_api_key.txt`
- File types: `.txt`, `.pdf`, `.doc`, `.docx` (best effort for binary formats)

## Notes
- Models that don’t support response_format or temperature defaults are handled via fallback parsing.
- Quotes are selected via retrieval; there is no separate verification step.
