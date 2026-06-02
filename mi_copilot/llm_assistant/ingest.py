"""Vectorize documentation, TransformerLens docs, and papers into ChromaDB."""
from __future__ import annotations

import os
from pathlib import Path

# TODO(D-2 PM): implement
# - walk docs/ for markdown
# - download TransformerLens docs (or use installed package's docs)
# - parse Nanda/Charton PDFs from papers/ via pypdf
# - chunk into ~500-token segments
# - embed via sentence-transformers (multilingual-e5-large)
# - upsert into ChromaDB collection
