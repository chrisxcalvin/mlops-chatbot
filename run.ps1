# One-shot: index (if needed) and start the app
if (-not (Test-Path data\chroma)) { python -m src.ingest }
python -m uvicorn src.server:app --port 8000
