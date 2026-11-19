"""LLM provider abstraction with an offline extractive fallback (no API key needed)."""
import os, re
from . import config as C


def _context(chunks):
    return "\n\n".join(f"[{i}] (Section: {c['section']}, p.{c['page']})\n{c['text']}" for i, c in enumerate(chunks, 1))


def _messages(q, chunks, history=None):
    convo = ""
    if history:
        lines = [f"{'Student' if h.get('role') == 'user' else 'Assistant'}: {str(h.get('content', ''))[:400]}"
                 for h in history[-4:]]
        convo = "Recent conversation (for resolving follow-up questions only):\n" + "\n".join(lines) + "\n\n"
    return [{"role": "system", "content": C.SYSTEM_PROMPT},
            {"role": "user", "content": f"{convo}Handbook excerpts:\n{_context(chunks)}\n\nStudent question: {q}"}]


def provider() -> str:
    p = C.LLM_PROVIDER
    if p != "auto":
        return p
    if os.getenv("GROQ_API_KEY"): return "groq"
    if os.getenv("GOOGLE_API_KEY"): return "gemini"
    if os.getenv("OPENAI_API_KEY"): return "openai"
    return "extractive"


def extractive(q, chunks):
    """Pick the most query-relevant sentences from retrieved chunks, with citations."""
    qt = set(re.findall(r"[a-z0-9]+", q.lower())) - {"the","a","an","is","of","to","what","how","do","i","for","in","and","are","can","my","me","on","at","does"}
    scored = []
    for n, c in enumerate(chunks, 1):
        for s in re.split(r"(?<=[.!?])\s+|\n", c["text"]):
            s = s.strip()
            if len(s) < 25: continue
            w = set(re.findall(r"[a-z0-9]+", s.lower()))
            sc = len(qt & w) / (len(qt) + 1) + (0.15 if n == 1 else 0)
            scored.append((sc, n, s))
    scored.sort(reverse=True)
    picked, seen = [], set()
    for sc, n, s in scored:
        if s in seen or sc <= 0: continue
        seen.add(s); picked.append((n, s))
        if len(picked) == 4: break
    if not picked:
        return None
    return "Here's what the Student Handbook says:\n\n" + "\n".join(f"• {s} [{n}]" for n, s in picked)


def _normalize(t):
    """Tidy model output: unify citation brackets and odd unicode spaces/hyphens."""
    if not t: return t
    for bad, good in ((0x3010, "["), (0x3011, "]"), (0x2011, "-"), (0x2009, " "), (0x202f, " "), (0xa0, " ")):
        t = t.replace(chr(bad), good)
    while "  " in t:
        t = t.replace("  ", " ")
    return t.replace(" [", " [").strip()


_local = {}
# The project is local-first: the cloud model is hidden unless you explicitly set ENABLE_CLOUD=1 in .env
CLOUD_ENABLED = os.getenv("ENABLE_CLOUD") == "1"


def available_models():
    """Models the UI can offer. Local ones appear only once their GGUF file is downloaded."""
    out = []
    if CLOUD_ENABLED and os.getenv("GROQ_API_KEY"):
        out.append({"id": "groq", "label": f"Groq {C.GROQ_MODEL} (cloud)", "params": "120B", "local": False})
    for mid, m in C.LOCAL_MODELS.items():
        f = C.MODELS_DIR / m["file"]
        if f.exists() and f.stat().st_size > 100_000_000 and not (C.MODELS_DIR / (m["file"] + ".part")).exists():
            out.append({"id": mid, "label": m["label"], "params": m["params"], "local": True,
                        "size_mb": round(f.stat().st_size / 1e6)})
    return out


def generate_local(model_id, q, chunks, history=None):
    """Run a quantised local model. Raises on failure (callers decide on fallback)."""
    from llama_cpp import Llama
    if model_id not in _local:
        _local.clear()  # keep only one model in RAM
        _local[model_id] = Llama(model_path=str(C.MODELS_DIR / C.LOCAL_MODELS[model_id]["file"]),
                                 n_ctx=C.LOCAL_CTX, n_threads=os.cpu_count() or 4, verbose=False)
    llm_ = _local[model_id]
    msgs = _messages(q, chunks[:C.LOCAL_TOP_K], history)
    r = llm_.create_chat_completion(messages=msgs, temperature=0.1, max_tokens=C.LOCAL_MAX_TOKENS)
    u = r["usage"]
    return (_normalize(r["choices"][0]["message"]["content"]), f"local:{model_id}",
            {"tokens": u["total_tokens"], "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"]})


def generate(q, chunks, history=None, model=None):
    """Returns (answer, provider_name, usage dict). `model` = UI choice (local id / 'groq'); None = auto."""
    if model is None and not CLOUD_ENABLED:
        av = available_models()
        model = av[0]["id"] if av else None
    if model in C.LOCAL_MODELS:
        try:
            return generate_local(model, q, chunks, history)
        except Exception as e:
            return extractive(q, chunks), f"extractive(fallback:{type(e).__name__})", {"tokens": 0}
    p = "groq" if (model == "groq" and CLOUD_ENABLED) else ("extractive" if not CLOUD_ENABLED else provider())
    msgs = _messages(q, chunks, history)
    try:
        if p == "groq":
            from groq import Groq
            r = Groq().chat.completions.create(model=C.GROQ_MODEL, messages=msgs, temperature=0.1, max_tokens=600)
            return _normalize(r.choices[0].message.content), f"groq:{C.GROQ_MODEL}", {"tokens": r.usage.total_tokens}
        if p == "openai":
            from openai import OpenAI
            r = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None).chat.completions.create(
                model=C.OPENAI_MODEL, messages=msgs, temperature=0.1, max_tokens=600)
            return _normalize(r.choices[0].message.content), f"openai:{C.OPENAI_MODEL}", {"tokens": r.usage.total_tokens}
        if p == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            m = genai.GenerativeModel(C.GEMINI_MODEL, system_instruction=C.SYSTEM_PROMPT)
            r = m.generate_content(msgs[1]["content"], generation_config={"temperature": 0.1, "max_output_tokens": 600})
            return _normalize(r.text), f"gemini:{C.GEMINI_MODEL}", {"tokens": getattr(r.usage_metadata, "total_token_count", 0)}
    except Exception as e:  # degrade gracefully to extractive mode
        ans = extractive(q, chunks)
        return ans, f"extractive(fallback:{type(e).__name__})", {"tokens": 0}
    return extractive(q, chunks), "extractive", {"tokens": 0}
