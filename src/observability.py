"""LLMOps telemetry: JSONL interaction log, MLflow metrics, feedback store, rolling stats."""
import json, time, threading, uuid
from collections import deque
from . import config as C

_lock = threading.Lock()
_recent = deque(maxlen=500)
C.LOG_DIR.mkdir(exist_ok=True)
_mlflow_ready = False


def _mlflow():
    global _mlflow_ready
    import mlflow
    if not _mlflow_ready:
        mlflow.set_tracking_uri(C.MLFLOW_URI)
        mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
        _mlflow_ready = True
    return mlflow


def log_interaction(rec: dict) -> str:
    rec["id"] = uuid.uuid4().hex[:10]
    rec["ts"] = time.time()
    rec["prompt_version"] = C.PROMPT_VERSION
    with _lock:
        _recent.append(rec)
        with open(C.LOG_DIR / "interactions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    threading.Thread(target=_to_mlflow, args=(rec,), daemon=True).start()
    return rec["id"]


def _to_mlflow(rec):
    try:
        ml = _mlflow()
        with _lock, ml.start_run(run_name="chat", tags={"kind": "chat", "prompt_version": rec["prompt_version"]}):
            ml.log_params({"model": rec["model"], "answered": rec["answered"]})
            ml.log_metrics({"latency_ms": rec["latency_ms"], "retrieval_ms": rec["retrieval_ms"],
                            "top_score": rec["top_score"], "tokens": rec.get("tokens", 0)})
    except Exception:
        pass


def log_feedback(interaction_id: str, vote: str):
    with _lock, open(C.LOG_DIR / "feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": interaction_id, "vote": vote, "ts": time.time()}) + "\n")


def stats():
    with _lock:
        r = list(_recent)
    if not r:
        return {"requests": 0}
    lat = sorted(x["latency_ms"] for x in r)
    fb = {"up": 0, "down": 0}
    p = C.LOG_DIR / "feedback.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            v = json.loads(line)["vote"]
            fb[v] = fb.get(v, 0) + 1
    return {"requests": len(r), "answer_rate": round(sum(x["answered"] for x in r) / len(r), 3),
            "p50_ms": lat[len(lat) // 2], "p95_ms": lat[int(len(lat) * .95) - 1],
            "avg_top_score": round(sum(x["top_score"] for x in r) / len(r), 3),
            "pii_flagged": sum(x.get("pii", False) for x in r), "feedback": fb,
            "prompt_version": C.PROMPT_VERSION}
