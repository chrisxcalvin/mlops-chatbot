"""FastAPI app. Run: uvicorn src.server:app --port 8000"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from . import config as C, rag, observability as obs, llm

app = FastAPI(title="CHRIST Student Helpdesk")


@app.middleware("http")
async def no_cache(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/api"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=2000)


class ChatReq(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=12)
    model: str | None = None


class FeedbackReq(BaseModel):
    id: str
    vote: str = Field(pattern="^(up|down)$")


@app.on_event("startup")
def warm():
    rag.retriever()


@app.post("/api/chat")
def chat(r: ChatReq):
    return rag.answer(r.message, [t.model_dump() for t in r.history], r.model)


@app.post("/api/feedback")
def feedback(r: FeedbackReq):
    obs.log_feedback(r.id, r.vote)
    return {"ok": True}


@app.get("/api/health")
def health():
    return {"status": "ok", "llm": llm.provider(), "chunks": len(rag.retriever().chunks),
            "embed_model": C.EMBED_MODEL, "prompt_version": C.PROMPT_VERSION}


@app.get("/api/models")
def models():
    return {"models": llm.available_models(), "default": (llm.available_models() or [{"id": None}])[0]["id"]}


@app.get("/api/compare")
def compare():
    import json
    p = C.ROOT / "eval" / "results" / "model_comparison.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"models": [], "missing": True}


@app.get("/api/metrics")
def metrics():
    return obs.stats()


@app.get("/")
def index():
    return FileResponse(C.ROOT / "web" / "index.html")


app.mount("/static", StaticFiles(directory=C.ROOT / "web"), name="static")
