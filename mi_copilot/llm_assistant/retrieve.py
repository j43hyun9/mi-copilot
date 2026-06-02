"""Semantic search over the ChromaDB vector store."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
COLLECTION_NAME = "mi_copilot_kb"


@dataclass
class RetrievalHit:
    text: str
    source: str
    file: str
    chunk_idx: int
    distance: float

    def short(self, max_chars: int = 200) -> str:
        body = self.text[:max_chars].replace("\n", " ")
        if len(self.text) > max_chars:
            body += "..."
        return f"[{self.source} · {self.file} #{self.chunk_idx}  d={self.distance:.3f}]\n  {body}"


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(COLLECTION_NAME)


def retrieve(query: str, k: int = 6, source_filter: str | None = None) -> list[RetrievalHit]:
    """Find the top-k most relevant chunks for `query`.

    Args:
        query: natural-language question (KO or EN)
        k: number of results
        source_filter: optional — restrict to one source label
                       (e.g., "transformerlens", "papers", "mi_copilot_repo")

    Returns:
        list of RetrievalHit ordered by ascending distance (closest = best)
    """
    embedder = _get_embedder()
    collection = _get_collection()

    is_e5 = "e5" in EMBEDDING_MODEL.lower()
    q_input = f"query: {query}" if is_e5 else query
    q_emb = embedder.encode([q_input], normalize_embeddings=True, convert_to_numpy=True)[0]

    where = {"source": source_filter} if source_filter else None
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=k,
        where=where,
    )

    hits: list[RetrievalHit] = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append(RetrievalHit(
            text=doc,
            source=meta.get("source", "?"),
            file=meta.get("file", "?"),
            chunk_idx=meta.get("chunk_idx", -1),
            distance=float(dist),
        ))
    return hits


def cli_test():
    """Quick interactive test mode: type queries and see top hits."""
    print(f"[retrieve test mode]  Collection: {COLLECTION_NAME}  Embedder: {EMBEDDING_MODEL}")
    print(f"  Type a query and press Enter. Empty line to quit.\n")
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        hits = retrieve(q, k=4)
        for h in hits:
            print(h.short(180))
            print()


if __name__ == "__main__":
    cli_test()
