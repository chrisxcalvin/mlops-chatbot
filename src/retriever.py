"""Hybrid retriever: dense (Chroma/MiniLM) + BM25, fused with Reciprocal Rank Fusion."""
import json, re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from . import config as C

_tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())


class Retriever:
    def __init__(self):
        self.chunks = json.loads(C.CHUNKS_PATH.read_text(encoding="utf-8"))
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([_tok(c["text"]) for c in self.chunks])
        self.model = SentenceTransformer(C.EMBED_MODEL)
        self.col = chromadb.PersistentClient(path=str(C.CHROMA_DIR)).get_collection("handbook")

    def search(self, query: str, k: int = C.TOP_K):
        q = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        res = self.col.query(query_embeddings=[q], n_results=C.CANDIDATES)
        dense_ids = res["ids"][0]
        dense_sim = {i: 1 - d for i, d in zip(dense_ids, res["distances"][0])}
        scores = self.bm25.get_scores(_tok(query))
        bm_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:C.CANDIDATES]
        bm_ids = [self.chunks[i]["id"] for i in bm_idx if scores[i] > 0]
        fused = {}
        for ranks in (dense_ids, bm_ids):
            for r, cid in enumerate(ranks):
                fused[cid] = fused.get(cid, 0) + 1 / (60 + r)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        out = []
        for cid in top:
            c = dict(self.by_id[cid])
            c["score"] = round(dense_sim.get(cid, 0.0), 4)
            out.append(c)
        best = max(dense_sim.values()) if dense_sim else 0.0
        return out, best
