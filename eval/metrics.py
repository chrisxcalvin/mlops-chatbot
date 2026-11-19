"""RAG answer-quality metrics (no LLM judge needed, all computed locally).

faithfulness      share of answer statements that are supported by the retrieved context
                  (embedding similarity of each statement to the best-matching context passage)
bad_numbers       answer contains a number that does not appear anywhere in the context (hallucination signal)
relevancy         cosine similarity between the question and the answer
context_recall    share of the expected facts that are present in the retrieved context (retrieval ceiling)
context_precision share of the retrieved passages that contain at least one expected fact
"""
import re
import numpy as np

SUPPORT_THRESHOLD = 0.55
BOILER = re.compile(r"further assistance|please contact|contact the|feel free|let me know|if you need|for more details", re.I)


def norm(t):
    return re.sub(r"[^a-z0-9%$₹]", "", t.lower())


def statements(answer):
    """Split an answer into factual statements (drops bullets, markdown, citations and polite boilerplate)."""
    out = []
    for piece in re.split(r"(?<=[.!?])\s+|\n", answer):
        piece = re.sub(r"\[\d+\]|[*_#>`]", "", piece).strip(" -•\t")
        if len(piece.split()) >= 4 and not BOILER.search(piece):
            out.append(piece)
    return out


def context_passages(chunks):
    """Sentences + 24-word sliding windows of the context (handbook tables are messy, so windows help)."""
    out = []
    for c in chunks:
        text = c["text"]
        out += [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n", text) if len(x.split()) >= 4]
        w = text.split()
        out += [" ".join(w[i:i + 24]) for i in range(0, max(1, len(w) - 11), 12)]
    return out


def numbers(text):
    return {re.sub(r"[^0-9.%]", "", n).strip(".") for n in re.findall(r"\d[\d,]*\.?\d*%?", text)} - {""}


def faithfulness(answer, chunks, embed):
    """Returns (support_ratio or None if the answer makes no claims, has_unsupported_number)."""
    ctx = " ".join(c["text"] for c in chunks)
    ctx_n = norm(ctx)
    bad_num = any(n.replace(",", "") not in ctx_n and n not in ctx_n for n in numbers(answer))
    st = statements(answer)
    if not st:
        return None, bad_num
    P = context_passages(chunks)
    if not P:
        return 0.0, bad_num
    E = embed.encode(st + P, normalize_embeddings=True, batch_size=64)
    S, Pe = E[:len(st)], E[len(st):]
    best = (S @ Pe.T).max(axis=1)
    return float((best >= SUPPORT_THRESHOLD).mean()), bad_num


def relevancy(question, answer, embed):
    a = " ".join(statements(answer)) or answer
    q, v = embed.encode([question, a], normalize_embeddings=True)
    return float(max(0.0, np.dot(q, v)))


def context_scores(facts, chunks):
    """(context_recall, context_precision) for an answerable question."""
    per_chunk = [norm(c["text"]) for c in chunks]
    joined = "".join(per_chunk)
    rec = sum(any(norm(a) in joined for a in grp) for grp in facts) / len(facts)
    allalts = [norm(a) for grp in facts for a in grp]
    prec = sum(any(a in t for a in allalts) for t in per_chunk) / max(1, len(per_chunk))
    return rec, prec
