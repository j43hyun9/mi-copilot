"""CLI entry point for mi-copilot.

Commands:
    mi-copilot ingest         (re)build the local ChromaDB vector store
    mi-copilot ask "..."      ask a question, OpenAI-synthesized answer
    mi-copilot search "..."   inspect top retrieved chunks (no LLM call)
"""
from __future__ import annotations

import sys

# Force UTF-8 on Windows console (cp949 default mangles em-dashes, Korean, etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

import click


@click.group()
def main():
    """mi-copilot — RAG-based LLM assistant for Mechanistic Interpretability."""


@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option("-k", "--top-k", type=int, default=6,
              help="Number of context chunks to retrieve (default 6).")
@click.option("-s", "--source", type=click.Choice(["transformerlens", "papers", "mi_copilot_repo"]),
              default=None, help="Restrict retrieval to one source.")
@click.option("--no-stream", is_flag=True, help="Disable token streaming.")
def ask(query, top_k, source, no_stream):
    """Ask a question. Example: mi-copilot ask "How do I reproduce Nanda 2023?" """
    from .chat import answer
    q = " ".join(query)
    answer(q, k=top_k, source_filter=source, stream=not no_stream)


@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option("-k", "--top-k", type=int, default=6)
@click.option("-s", "--source", type=click.Choice(["transformerlens", "papers", "mi_copilot_repo"]),
              default=None)
def search(query, top_k, source):
    """Retrieve raw chunks without LLM synthesis (debug / inspect)."""
    from .retrieve import retrieve
    q = " ".join(query)
    hits = retrieve(q, k=top_k, source_filter=source)
    for i, h in enumerate(hits, 1):
        click.echo(f"\n#{i}  {h.source} · {h.file}  (d={h.distance:.3f})")
        click.echo("  " + h.text[:300].replace("\n", "\n  "))
        if len(h.text) > 300:
            click.echo("  ...")


@main.command()
def ingest():
    """Build / refresh the local vector store."""
    from .ingest import main as ingest_main
    ingest_main()


@main.command()
@click.option("--port", type=int, default=8501, help="Port for Streamlit server (default 8501).")
def ui(port):
    """Launch the Streamlit chat UI in your browser."""
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path
    app_path = _Path(__file__).parent / "app.py"
    cmd = [_sys.executable, "-m", "streamlit", "run", str(app_path),
           "--server.port", str(port), "--server.headless", "false"]
    click.echo(f"Launching mi-copilot UI on http://localhost:{port}")
    subprocess.run(cmd)


@main.command()
@click.option("-n", "--samples", type=int, default=2,
              help="Number of sample chunks to show per source (default 2).")
@click.option("-s", "--source",
              type=click.Choice(["transformerlens", "papers", "mi_copilot_repo"]),
              default=None, help="Show only one source's stats + samples.")
def inspect(samples, source):
    """Browse the ChromaDB vector store contents (stats + sample chunks)."""
    from collections import Counter
    import chromadb
    from .ingest import CHROMA_DB_PATH, COLLECTION_NAME

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    coll = client.get_collection(COLLECTION_NAME)
    total = coll.count()
    data = coll.get(include=["metadatas"])

    src_counter = Counter(m["source"] for m in data["metadatas"])
    file_counter = Counter(m["file"] for m in data["metadatas"])

    click.echo(click.style(f"\n[ChromaDB: {COLLECTION_NAME}]  total {total} chunks\n",
                            fg="cyan", bold=True))

    # Source-level distribution
    click.echo("source 별 분포:")
    for src, n in sorted(src_counter.items(), key=lambda x: -x[1]):
        if source and src != source:
            continue
        bar = "█" * int(n / total * 50)
        click.echo(f"  {src:20s}  {n:4d}  {bar}")

    # File-level distribution (top 10 of the chosen source, or top 15 overall)
    click.echo("\n파일 별 분포 (top 15):")
    items = file_counter.most_common()
    if source:
        items = [(f, n) for f, n in items
                 if any(m["file"] == f and m["source"] == source for m in data["metadatas"])][:15]
    else:
        items = items[:15]
    for f, n in items:
        short = f if len(f) <= 60 else "..." + f[-57:]
        click.echo(f"  {n:4d}  {short}")

    # Sample chunks
    click.echo(click.style(f"\n샘플 chunks ({samples}/source):\n", fg="cyan", bold=True))
    seen_per_source: Counter = Counter()
    for id_, meta in zip(data["ids"], data["metadatas"]):
        src = meta["source"]
        if source and src != source:
            continue
        if seen_per_source[src] >= samples:
            continue
        seen_per_source[src] += 1
        chunk_obj = coll.get(ids=[id_], include=["documents", "metadatas"])
        doc = chunk_obj["documents"][0]
        click.echo(click.style(
            f"─── [{src}]  {meta['file']}  chunk #{meta['chunk_idx']}  ({id_}) ───",
            fg="yellow",
        ))
        preview = doc[:500].rstrip()
        click.echo(preview)
        if len(doc) > 500:
            click.echo(click.style(f"  ...(전체 {len(doc)} chars 중 처음 500자)", dim=True))
        click.echo()

    # Embedding info
    if data["ids"]:
        sample = coll.get(ids=[data["ids"][0]], include=["embeddings"])
        emb = sample["embeddings"][0]
        norm = sum(v * v for v in emb) ** 0.5
        click.echo(click.style("임베딩 정보:", fg="cyan", bold=True))
        click.echo(f"  차원:      {len(emb)}")
        click.echo(f"  L2 norm:   {norm:.4f}  (1.0 에 가까우면 normalized)")
        click.echo(f"  첫 5 값:   {[round(float(v), 4) for v in emb[:5]]}")


if __name__ == "__main__":
    main()
