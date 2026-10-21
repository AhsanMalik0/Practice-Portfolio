import hashlib, os, re, sys
import numpy as np
from google import genai
import logging

from google.genai import types
from pymongo.operations import SearchIndexModel
from app.config import DOCS_DIR, EMBED_MODEL, EMBED_DIM, VECTOR_INDEX, USE_ATLAS, MIN_SCORE, GEMINI_API_KEY
from app.database import get_mongo_client

log = logging.getLogger("bot")
client = genai.Client(api_key=GEMINI_API_KEY)

POLICY = """Refund eligibility: Items can be returned within 14 days of delivery if they are unused and in original packaging.
Refund timeline: Approved refunds are paid to the original payment method within 5-7 business days after we receive the item.
Non-refundable items: Opened earbuds, underwear, socks, gift cards and clearance items cannot be refunded.
Damaged or wrong item: If an item arrives damaged or incorrect, contact us within 48 hours with a photo. We replace it or refund in full, including shipping.
How to request a refund: Email your order number to support@demostore.example or use the Returns page. We reply within 1 business day with a return label.
Return shipping: Return shipping is free for damaged or wrong items. For change-of-mind returns the customer pays return shipping.
Cash on delivery refunds: COD orders are refunded by bank transfer. Share your account details after the refund is approved."""

def read_docs():
    docs = []
    for f in sorted(os.listdir(DOCS_DIR)) if os.path.isdir(DOCS_DIR) else []:
        path, ext = os.path.join(DOCS_DIR, f), f.lower().rsplit(".", 1)[-1]
        try:
            if ext in ("txt", "md"):
                docs.append((f, open(path, encoding="utf-8").read()))
            elif ext == "pdf":
                from pypdf import PdfReader
                docs.append((f, "\n".join(pg.extract_text() or "" for pg in PdfReader(path).pages)))
        except Exception as e:
            log.warning("Skipped %s: %s", f, e)
    return docs or [("built-in-sample-policy.txt", POLICY)]

def chunk_text(text, max_chars=500):
    chunks = []
    for para in [p.strip() for p in text.split("\n") if p.strip()]:
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        buf, prev = "", ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if buf and len(buf) + len(sent) > max_chars:
                chunks.append(buf.strip())
                buf = prev
            buf, prev = buf + " " + sent, sent
        if buf.strip():
            chunks.append(buf.strip())
    return chunks

def embed(texts, task):
    out = []
    for i in range(0, len(texts), 100):
        r = client.models.embed_content(
            model=EMBED_MODEL, contents=texts[i:i + 100],
            config=types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM)
        )
        out += [e.values for e in r.embeddings]
    v = np.array(out, dtype=np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).tolist()

class MongoRAG:
    def __init__(self):
        self.col = get_mongo_client()["policy_chunks"]
        self.sync()

    def sync(self):
        want = {}
        for src, text in read_docs():
            for c in chunk_text(text):
                want[hashlib.md5(f"{EMBED_MODEL}|{EMBED_DIM}|{src}|{c}".encode()).hexdigest()] = (c, src)
        have = {d["_id"] for d in self.col.find({}, {"_id": 1})}
        stale = list(have - want.keys())
        if stale:
            self.col.delete_many({"_id": {"$in": stale}})
        new = [i for i in want if i not in have]
        if new:
            vecs = embed([want[i][0] for i in new], "RETRIEVAL_DOCUMENT")
            self.col.insert_many([{"_id": i, "text": want[i][0], "source": want[i][1], "embedding": v} for i, v in zip(new, vecs)])
        if USE_ATLAS:
            self._ensure_atlas_index()
        else:
            self._load_matrix()
        log.info("MongoDB RAG ready: %d chunks (%d new, %d removed), mode=%s", len(want), len(new), len(stale), "atlas" if USE_ATLAS else "in-memory")

    def _ensure_atlas_index(self):
        if VECTOR_INDEX not in [i["name"] for i in self.col.list_search_indexes()]:
            self.col.create_search_index(SearchIndexModel(
                name=VECTOR_INDEX, type="vectorSearch",
                definition={"fields": [{"type": "vector", "path": "embedding", "numDimensions": EMBED_DIM, "similarity": "cosine"}]}
            ))

    def _load_matrix(self):
        rows = list(self.col.find({}, {"text": 1, "source": 1, "embedding": 1}))
        self.rows = rows
        self.matrix = np.array([r["embedding"] for r in rows], dtype=np.float32) if rows else np.zeros((0, EMBED_DIM), np.float32)

    def search(self, question, k=3):
        try:
            qv = embed([question], "RETRIEVAL_QUERY")[0]
            if USE_ATLAS:
                pipeline = [{"$vectorSearch": {"index": VECTOR_INDEX, "path": "embedding", "queryVector": qv, "numCandidates": 50, "limit": k}},
                            {"$project": {"_id": 0, "text": 1, "source": 1, "score": {"$meta": "vectorSearchScore"}}}]
                hits = [{"text": r["text"], "source": r["source"], "score": round(2 * r["score"] - 1, 2)} for r in self.col.aggregate(pipeline)]
            else:
                sims = self.matrix @ np.array(qv, dtype=np.float32)
                hits = [{"text": self.rows[i]["text"], "source": self.rows[i]["source"], "score": round(float(sims[i]), 2)} for i in np.argsort(-sims)[:k]]
            return [h for h in hits if h["score"] >= MIN_SCORE]
        except Exception as e:
            log.warning("Vector search failed (%s); using keyword fallback", e)
            q = set(re.findall(r"[a-z]{4,}", question.lower()))
            scored = sorted(((len(q & set(re.findall(r"[a-z]{4,}", r["text"].lower()))), r["text"], r["source"]) for r in self.col.find({}, {"text": 1, "source": 1})), reverse=True)
            return [{"text": t, "source": src, "score": 0.0} for n, t, src in scored[:2] if n > 0]

_rag = None
def rag():
    global _rag
    if _rag is None:
        _rag = MongoRAG()
    return _rag