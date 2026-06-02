"""RAG-based LLM assistant — the primary deliverable of mi-copilot.

Pipeline:
    ingest    — vectorize docs/, TransformerLens docs, papers/ into ChromaDB
    retrieve  — semantic search for relevant context given a query
    chat      — generate answers via Claude API using retrieved context
    cli       — command-line entry point: `mi-copilot ask "..."`
"""
