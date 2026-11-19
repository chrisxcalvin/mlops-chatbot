"""Offline evaluation + MLflow logging.  Run: python -m eval.run_eval
Metrics: hit@k / MRR (retrieval), answer-grounding (cited), OOS refusal rate, latency.
Use it as a regression gate whenever chunking / embeddings / prompt / model change."""
import json, sys, time
from pathlib import Path
import mlflow
from src import config as C, rag

OOS = ["Who won the FIFA world cup in 2022?", "Give me a recipe for biryani", "What is the price of bitcoin today?"]


def main(min_hit=0.8):
    rows = [json.loads(l) for l in (Path(__file__).parent / "golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    r = rag.retriever()
    hits, rr, lat, cited = 0, 0.0, [], 0
    for row in rows:
        t = time.time()
        chunks, _ = r.search(row["q"])
        lat.append((time.time() - t) * 1000)
        rank = next((i for i, c in enumerate(chunks, 1)
                     if all(m.lower() in c["text"].lower() for m in row["must"])), None)
        hits += rank is not None
        rr += 1 / rank if rank else 0
        out = rag.answer(row["q"])
        cited += "[" in out["answer"] and bool(out["sources"])
        print(("PASS" if rank else "FAIL"), rank, row["q"])
    refused = sum(not rag.answer(q)["sources"] for q in OOS)
    m = {"hit_at_k": hits / len(rows), "mrr": rr / len(rows), "cited_rate": cited / len(rows),
         "oos_refusal_rate": refused / len(OOS), "avg_retrieval_ms": sum(lat) / len(lat)}
    mlflow.set_tracking_uri(C.MLFLOW_URI); mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="eval"):
        mlflow.log_params({"embed_model": C.EMBED_MODEL, "chunk_size": C.CHUNK_SIZE, "top_k": C.TOP_K,
                           "prompt_version": C.PROMPT_VERSION, "min_score": C.MIN_SCORE})
        mlflow.log_metrics(m)
    print(json.dumps(m, indent=2))
    if m["hit_at_k"] < min_hit:
        sys.exit("Eval gate FAILED")


if __name__ == "__main__":
    main()
