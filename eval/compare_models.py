"""Compare the selectable models (3 local quantised + Groq cloud reference) on the same handbook Q&A set.

Run:  python -m eval.compare_models                 # every model that is available
      python -m eval.compare_models qwen2.5-0.5b    # only some (results merge into the same file)

Metrics per model (see eval/metrics.py for faithfulness, relevancy, context recall/precision; plus token usage, tokens/sec)
  fact_accuracy   avg share of expected facts present in the answer (answerable questions)
  exact_rate      share of answers containing ALL expected facts
  citation_rate   share of answers that cite a source like [1]
  abstain_rate    share of unanswerable questions where the model correctly says it is not in the handbook
  avg_latency_s   mean seconds per answer (CPU for local models)
Results -> eval/results/model_comparison.json (read by the app's "Model Comparison" tab) + MLflow.
"""
import json, re, sys, time
from pathlib import Path
import mlflow
from src import config as C, llm, rag
from eval import metrics as M

OUT = Path(__file__).parent / "results" / "model_comparison.json"
ABSTAIN = re.compile(r"couldn.t find|could not find|cannot find|can.t find|not (mentioned|specified|available|provided|stated|covered|found|included)|"
                     r"no (specific )?information|does not (mention|specify|contain|provide|include)|don.t have|do not have|"
                     r"isn.t (mentioned|specified|provided)|not (explicitly|clearly|directly)|not in the", re.I)


def norm(t):
    return re.sub(r"[^a-z0-9%$₹]", "", t.lower())


def run_model(mid, rows):
    ret = rag.retriever()
    embed = ret.model
    details, errors = [], 0
    for i, row in enumerate(rows, 1):
        chunks, _ = ret.search(row["q"])
        ctx = chunks if mid == "groq" else chunks[:C.LOCAL_TOP_K]
        usage = {}
        t = time.time()
        try:
            if mid == "groq":
                ans, used, usage = llm.generate(row["q"], chunks, None, "groq")
                if used.startswith("extractive"):
                    raise RuntimeError(used)
            else:
                ans, used, usage = llm.generate_local(mid, row["q"], chunks)
        except Exception as e:
            errors += 1
            ans = ""
            print("   error:", e)
        dt = time.time() - t
        d = {"q": row["q"], "answer": ans, "latency_s": round(dt, 2),
             "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
             "words": len(ans.split())}
        if ans:
            d["relevancy"] = round(M.relevancy(row["q"], ans, embed), 3)
            f, bad = M.faithfulness(ans, ctx, embed)
            d["faithfulness"] = None if f is None else round(f, 3)
            d["bad_number"] = bool(bad)
        if not row.get("unanswerable"):
            d["ctx_recall"], d["ctx_precision"] = [round(v, 3) for v in M.context_scores(row["facts"], ctx)]
        if row.get("unanswerable"):
            d["kind"] = "unanswerable"; d["ok"] = bool(ABSTAIN.search(ans))
        else:
            n = norm(ans)
            hit = [any(norm(a) in n for a in grp) for grp in row["facts"]]
            d["kind"] = "answerable"; d["fact_score"] = sum(hit) / len(hit); d["ok"] = all(hit)
            d["cited"] = bool(re.search(r"\[\d+\]", ans))
        details.append(d)
        print(f"  [{i}/{len(rows)}] {'OK ' if d['ok'] else 'MISS'} {dt:4.1f}s  {row['q'][:70]}")
        if mid == "groq":
            time.sleep(1.2)
    A = [d for d in details if d["kind"] == "answerable"]
    U = [d for d in details if d["kind"] == "unanswerable"]
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    F = [d["faithfulness"] for d in A if d.get("faithfulness") is not None]
    gen_s = sum(d["latency_s"] for d in details if d["completion_tokens"])
    m = {"fact_accuracy": avg([d["fact_score"] for d in A]), "exact_rate": avg([d["ok"] for d in A]),
         "citation_rate": avg([d["cited"] for d in A]), "abstain_rate": avg([d["ok"] for d in U]),
         "hallucination_rate": 1 - avg([d["ok"] for d in U]),
         "faithfulness": avg(F), "bad_number_rate": avg([d.get("bad_number", False) for d in A]),
         "relevancy": avg([d["relevancy"] for d in details if "relevancy" in d]),
         "context_recall": avg([d["ctx_recall"] for d in A]), "context_precision": avg([d["ctx_precision"] for d in A]),
         "avg_latency_s": avg([d["latency_s"] for d in details]),
         "avg_prompt_tokens": avg([d["prompt_tokens"] for d in details]),
         "avg_completion_tokens": avg([d["completion_tokens"] for d in details]),
         "avg_total_tokens": avg([d["prompt_tokens"] + d["completion_tokens"] for d in details]),
         "tokens_per_sec": (sum(d["completion_tokens"] for d in details) / gen_s) if gen_s else 0.0,
         "avg_answer_words": avg([d["words"] for d in details]), "errors": errors}
    return m, details


def main():
    rows = [json.loads(l) for l in (Path(__file__).parent / "qa.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    avail = {m["id"]: m for m in llm.available_models()}
    wanted = sys.argv[1:] or list(avail)
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"models": []}
    mlflow.set_tracking_uri(C.MLFLOW_URI); mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    for mid in wanted:
        if mid not in avail:
            print(f"skip {mid}: not available (not downloaded / no key)"); continue
        print(f"\n=== {avail[mid]['label']} ===")
        m, details = run_model(mid, rows)
        entry = {"id": mid, "label": avail[mid]["label"], "params": avail[mid]["params"], "local": avail[mid]["local"],
                 "size_mb": avail[mid].get("size_mb"), "metrics": {k: round(v, 3) for k, v in m.items()},
                 "n_answerable": sum(d["kind"] == "answerable" for d in details),
                 "n_unanswerable": sum(d["kind"] == "unanswerable" for d in details),
                 "prompt_version": C.PROMPT_VERSION, "ran_at": time.strftime("%Y-%m-%d %H:%M"), "details": details}
        res["models"] = [e for e in res["models"] if e["id"] != mid] + [entry]
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        with mlflow.start_run(run_name=f"compare-{mid}"):
            mlflow.log_params({"model": mid, "prompt_version": C.PROMPT_VERSION, "local_top_k": C.LOCAL_TOP_K,
                               "ctx": C.LOCAL_CTX, "quant": "Q4_K_M" if avail[mid]["local"] else "n/a"})
            mlflow.log_metrics(m)
        print(json.dumps(entry["metrics"]))


if __name__ == "__main__":
    main()
