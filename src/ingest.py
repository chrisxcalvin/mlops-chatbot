"""Ingestion pipeline: PDF -> cleaned, section-tagged chunks -> Chroma vector store.
Run:  python -m src.ingest
Each run is tracked in MLflow (params, chunk stats, artifact)."""
import json, re, time, hashlib
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from . import config as C


def section_for(printed_page: int) -> str:
    title = "Front Matter"
    for start, name in C.SECTIONS:
        if printed_page >= start:
            title = name
    return title


def clean(text: str) -> str:
    text = re.sub(r"Handbook 2026-27|CHRIST \(Deemed to be University\)\n\d+\n?", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split(text: str, size: int, overlap: int):
    """Paragraph/line-aware sliding window."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks, cur = [], ""
    for line in lines:
        if len(cur) + len(line) + 1 > size and cur:
            chunks.append(cur)
            cur = cur[-overlap:].split(" ", 1)[-1] + " " + line
        else:
            cur = (cur + " " + line).strip() if cur else line
    if len(cur) > 80:
        chunks.append(cur)
    return chunks


def build_chunks():
    doc = fitz.open(C.PDF_PATH)
    chunks = []
    for i, page in enumerate(doc):
        printed = i + 1 - C.PAGE_OFFSET
        text = clean(page.get_text())
        if len(text) < 80:
            continue
        sec = section_for(printed)
        for j, piece in enumerate(split(text, C.CHUNK_SIZE, C.CHUNK_OVERLAP)):
            cid = hashlib.md5(f"{i}-{j}-{piece[:40]}".encode()).hexdigest()[:12]
            chunks.append({"id": cid, "text": piece, "section": sec,
                           "pdf_page": i + 1, "page": printed if printed > 0 else i + 1})
    return chunks


def main():
    import mlflow
    t0 = time.time()
    chunks = build_chunks()
    C.CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    model = SentenceTransformer(C.EMBED_MODEL)
    emb = model.encode([c["text"] for c in chunks], batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True)
    client = chromadb.PersistentClient(path=str(C.CHROMA_DIR))
    try:
        client.delete_collection("handbook")
    except Exception:
        pass
    col = client.create_collection("handbook", metadata={"hnsw:space": "cosine"})
    for s in range(0, len(chunks), 256):
        b = chunks[s:s + 256]
        col.add(ids=[c["id"] for c in b], embeddings=emb[s:s + 256].tolist(),
                documents=[c["text"] for c in b],
                metadatas=[{"section": c["section"], "page": c["page"], "pdf_page": c["pdf_page"]} for c in b])
    lens = [len(c["text"]) for c in chunks]
    mlflow.set_tracking_uri(C.MLFLOW_URI)
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="ingest"):
        mlflow.log_params({"embed_model": C.EMBED_MODEL, "chunk_size": C.CHUNK_SIZE,
                           "chunk_overlap": C.CHUNK_OVERLAP, "pdf": C.PDF_PATH.name})
        mlflow.log_metrics({"n_chunks": len(chunks), "avg_chunk_chars": sum(lens) / len(lens),
                            "ingest_seconds": time.time() - t0})
        mlflow.log_artifact(str(C.CHUNKS_PATH))
    print(f"Indexed {len(chunks)} chunks in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
