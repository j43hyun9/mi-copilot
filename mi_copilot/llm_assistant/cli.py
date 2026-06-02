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


if __name__ == "__main__":
    main()
