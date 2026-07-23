"""
Lightweight RAG layer (TF-IDF + cosine similarity) — no model downloads,
runs instantly, fully deterministic. Swap for a real vector DB + embedding
model in production (see README).
"""
import os
import glob
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


class RagIndex:
    def __init__(self, docs_dir: str = None):
        self.docs_dir = docs_dir or config.DOCS_DIR
        self.chunks: List[Chunk] = []
        self.vectorizer = None
        self.matrix = None
        self._build()

    def _build(self):
        filepaths = sorted(glob.glob(os.path.join(self.docs_dir, "*.txt")))
        if not filepaths:
            raise RuntimeError(f"No .txt documents found in {self.docs_dir}")
        chunk_id = 0
        for path in filepaths:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            source = os.path.basename(path)
            for piece in _chunk_text(text):
                self.chunks.append(Chunk(text=piece, source=source, chunk_id=chunk_id))
                chunk_id += 1
        corpus = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = None) -> List[dict]:
        top_k = top_k or config.TOP_K_CHUNKS
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        ranked_idx = sims.argsort()[::-1][:top_k]
        results = []
        for idx in ranked_idx:
            if sims[idx] <= 0:
                continue
            c = self.chunks[idx]
            results.append({"text": c.text, "source": c.source, "score": float(sims[idx])})
        return results


_index = None


def get_index() -> RagIndex:
    global _index
    if _index is None:
        _index = RagIndex()
    return _index
