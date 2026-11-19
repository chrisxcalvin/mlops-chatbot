# CHRIST Student Helpdesk – RAG + LLMOps

AI chatbot answering student questions strictly from the **CHRIST Student Handbook 2026-27** (`data/handbook_*.pdf`), with source citations.

## Architecture
```
PDF ─► ingest (PyMuPDF, clean, section/page tagging, chunk 900/150) ─► Chroma (MiniLM embeddings)
Question ─► guardrails ─► hybrid retrieval (dense + BM25, RRF) ─► grounding threshold
         ─► LLM (Groq | Gemini | OpenAI | offline extractive fallback) ─► answer + [n] citations
         ─► telemetry (JSONL + MLflow) ─► 👍/👎 feedback
```
## Run
```
pip install -r requirements.txt
copy .env.example .env      # optional: add GROQ_API_KEY / GOOGLE_API_KEY for generative answers
python -m src.ingest        # build index (also logged to MLflow)
python -m uvicorn src.server:app --port 8000   # open http://localhost:8000
```
Without an API key it runs in **extractive mode** (quotes the best handbook sentences with citations).

## LLMOps
| Concern | Implementation |
|---|---|
| Experiment tracking | MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`): ingest params/stats, eval runs, per-request latency/score/tokens |
| Prompt versioning | `PROMPT_VERSION` in `src/config.py`, tagged on every run/log |
| Evaluation / regression gate | `python -m eval.run_eval` – hit@k, MRR, citation rate, out-of-scope refusal; CI fails if hit@k < 0.8 |
| Guardrails | prompt-injection filter, PII redaction/flag, smalltalk routing, low-similarity refusal (no hallucination) |
| Monitoring | `GET /api/metrics` (p50/p95 latency, answer rate, feedback), `GET /api/health`, `logs/interactions.jsonl` |
| Feedback loop | thumbs → `logs/feedback.jsonl` → add failing questions to `eval/golden.jsonl` |
| Deployment | `Dockerfile` (index baked in), GitHub Actions CI, config via env vars |

Baseline (MiniLM, extractive): hit@5 0.95, MRR 0.85, 100% out-of-scope refusal.
Tune `CHUNK_SIZE`, `EMBED_MODEL` (e.g. `BAAI/bge-small-en-v1.5`), `TOP_K`, `MIN_SCORE` via env, re-ingest, re-run eval, compare in MLflow.

## Setup notes
- Put the handbook PDF at `data/handbook_20260724011902.pdf` (not included in this repo), then run `python -m src.ingest`.
- Local models (~2.3 GB) are not included: run `./download_models.sh` (resumable), then `python -m eval.compare_models` to benchmark them.
- Copy `.env.example` to `.env` for optional API keys. `.env` is git-ignored: never commit it.
- Optional avatar images go in `web/photo.png` (you) and `web/photo2.png` (bot); the UI hides them if missing.
- Note: the CI workflow needs the PDF to build the index, so it fails until the document is available to the runner.
