"""RAG orchestration: guardrails -> retrieval -> grounding check -> generation -> telemetry."""
import time
from . import config as C, guardrails, llm, observability as obs
from .retriever import Retriever

_ret = None
NOT_FOUND = ("I couldn't find that in the Student Handbook 2026-27. Could you rephrase it or add a few more details? "
             "I can help with exams, library, fees, hostels, the academic calendar, programmes and university "
             "rules. For anything else, please contact the Registrar / Dean's office / Student Welfare.")


def retriever():
    global _ret
    if _ret is None:
        _ret = Retriever()
    return _ret


def _search_query(q: str, history) -> str:
    """Short follow-ups ('and for PG?', 'what about the fine?') borrow context from the previous user turn."""
    prev = [h["content"] for h in (history or []) if h.get("role") == "user"]
    if prev and len(q.split()) <= 7:
        return f"{prev[-1]} {q}"
    return q


def answer(question: str, history=None, model=None) -> dict:
    t0 = time.time()
    ok, reply, q, pii = guardrails.check_input(question)
    if not ok:
        iid = obs.log_interaction({"q": question[:200], "answered": False, "model": "guardrail", "latency_ms": 1,
                                   "retrieval_ms": 0, "top_score": 0.0, "pii": False})
        return {"id": iid, "answer": reply, "sources": [], "model": "guardrail", "latency_ms": 1}
    t1 = time.time()
    chunks, best = retriever().search(_search_query(q, history))
    ret_ms = (time.time() - t1) * 1000
    answered, used, usage = False, "none", {}
    if best < C.MIN_SCORE or not chunks:
        text = NOT_FOUND
    else:
        text, used, usage = llm.generate(q, chunks, history, model)
        answered = bool(text)
        text = text or NOT_FOUND
    sources = [{"n": i, "section": c["section"], "page": c["page"], "score": c["score"],
                "snippet": c["text"][:240] + ("…" if len(c["text"]) > 240 else "")}
               for i, c in enumerate(chunks, 1)] if answered else []
    lat = (time.time() - t0) * 1000
    iid = obs.log_interaction({"q": q[:300], "answered": answered, "model": used, "latency_ms": round(lat),
                               "retrieval_ms": round(ret_ms), "top_score": round(best, 3), "pii": pii,
                               "tokens": usage.get("tokens", 0), "pages": [s["page"] for s in sources]})
    return {"id": iid, "answer": text, "sources": sources, "model": used, "latency_ms": round(lat)}
