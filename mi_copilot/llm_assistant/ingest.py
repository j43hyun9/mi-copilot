"""Vectorize TransformerLens docs + MI papers + own docs into ChromaDB."""
from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ============ Config ============
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
COLLECTION_NAME = "mi_copilot_kb"

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_TL = PROJECT_ROOT / "external" / "TransformerLens"
PAPERS_DIR = Path(r"C:\Users\j43hy\Downloads\2026_capstone\papers")


def _gather(label: str, root: Path, exts: tuple[str, ...]) -> list[tuple[str, Path]]:
    if not root.exists():
        return []
    return [(label, p) for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]


def discover_sources() -> list[tuple[str, Path]]:
    """Return [(source_label, path), ...] for all documents to ingest."""
    items: list[tuple[str, Path]] = []
    # mi_copilot's own docs
    items.append(("mi_copilot_repo", PROJECT_ROOT / "README.md"))
    items += _gather("mi_copilot_repo", PROJECT_ROOT / "docs", (".md",))
    # TransformerLens docs (only .md/.rst — skip code, demos handled separately)
    items += _gather("transformerlens", EXTERNAL_TL, (".md", ".rst"))
    # MI papers
    items += _gather("papers", PAPERS_DIR, (".pdf",))
    return [(label, p) for label, p in items if p.exists()]


# ============ Document loaders ============
def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    return load_text(path)


# ============ Chunking ============
def chunk_text(text: str, target_size: int = 800, overlap: int = 100) -> list[str]:
    """Paragraph-aware chunker with sentence fallback for oversized paragraphs."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = current[-overlap:] if (overlap and current) else ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > target_size:
            # Split oversized paragraph by sentence
            for sent in re.split(r"(?<=[.!?])\s+", para):
                if len(current) + len(sent) + 1 > target_size and current:
                    flush()
                current += sent + " "
            continue
        if len(current) + len(para) + 2 > target_size and current:
            flush()
        current += para + "\n\n"

    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 50]


# ============ Main ingest pipeline ============
def main():
    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"[ingest] Embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print(f"[ingest] ChromaDB path: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[ingest] (deleted existing collection)")
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    print(f"\n[ingest] Discovering sources...")
    sources = discover_sources()
    print(f"[ingest] Found {len(sources)} source files")

    all_chunks: list[str] = []
    all_meta: list[dict] = []

    for source_label, path in sources:
        try:
            text = load_document(path)
        except Exception as e:
            print(f"  [warn] {path.name}: load failed — {e}")
            continue
        if not text.strip():
            print(f"  [skip] {path.name}: empty")
            continue
        chunks = chunk_text(text)
        try:
            rel = path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = path.name
        print(f"  {source_label:18s}  {str(rel)[:60]:60s}  → {len(chunks):4d} chunks")
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            all_meta.append({
                "source": source_label,
                "file": str(rel),
                "chunk_idx": i,
            })

    print(f"\n[ingest] Total chunks: {len(all_chunks)}")
    print(f"[ingest] Embedding (this is the slow part)...")
    # e5 models expect "passage: " prefix for documents, "query: " for queries
    is_e5 = "e5" in EMBEDDING_MODEL.lower()
    inputs = [f"passage: {c}" if is_e5 else c for c in all_chunks]
    embeddings = embedder.encode(
        inputs, batch_size=64, show_progress_bar=True, normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print(f"[ingest] Upserting to ChromaDB...")
    ids = [f"chunk_{i:05d}" for i in range(len(all_chunks))]
    # ChromaDB upsert in batches (some limits on size)
    batch = 500
    for s in range(0, len(ids), batch):
        e = min(s + batch, len(ids))
        collection.upsert(
            ids=ids[s:e],
            embeddings=[v.tolist() for v in embeddings[s:e]],
            documents=all_chunks[s:e],
            metadatas=all_meta[s:e],
        )

    print(f"\n[ingest] Done. Collection '{COLLECTION_NAME}' size: {collection.count()}")
    by_source: dict[str, int] = {}
    for m in all_meta:
        by_source[m["source"]] = by_source.get(m["source"], 0) + 1
    print(f"[ingest] Chunks by source: {by_source}")


if __name__ == "__main__":
    main()
