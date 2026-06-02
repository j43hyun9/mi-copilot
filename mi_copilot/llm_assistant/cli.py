"""CLI entry point: `mi-copilot ask "..."` and related commands."""
from __future__ import annotations

import click


@click.group()
def main():
    """mi-copilot — RAG-based LLM assistant for Mechanistic Interpretability."""


@main.command()
@click.argument("query", nargs=-1, required=True)
def ask(query: tuple[str, ...]):
    """Ask a question. Example: mi-copilot ask "How do I reproduce Nanda 2023?" """
    q = " ".join(query)
    # TODO(D-1 AM): retrieve + chat
    click.echo(f"[stub] Query: {q}")
    click.echo("[stub] Implementation pending — D-1 AM.")


@main.command()
def ingest():
    """Build / refresh the local vector store from docs/ and papers/."""
    # TODO(D-2 PM): call ingest module
    click.echo("[stub] ingest — implementation pending.")


if __name__ == "__main__":
    main()
