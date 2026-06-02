"""Generate answers via Claude API using retrieved context."""
from __future__ import annotations

import os
from typing import Iterator

from dotenv import load_dotenv

from .retrieve import RetrievalHit, retrieve

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")

SYSTEM_PROMPT = """\
You are mi-copilot, a domain-specific assistant for Mechanistic Interpretability (MI) research.

Your job is to help users — from absolute beginners to seasoned researchers — reproduce
MI papers and use the existing MI tooling (TransformerLens, paper-specific repos like
Charton's GCD code, etc.).

You will be given retrieved context from:
  - TransformerLens documentation
  - Foundational MI papers (Nanda 2023 on grokking modular addition, Charton 2024 on
    GCD circuits, Friedman 2023 on Transformer Programs, Weiss 2021 on RASP)
  - The mi-copilot library's own README

Rules:
  1. ALWAYS ground your answer in the retrieved context. If the context does not contain
     the answer, say so explicitly rather than guessing.
  2. When recommending code, prefer pointing the user to existing tools (TransformerLens
     functions, paper-specific repos) over writing custom code from scratch.
  3. Be concrete: give actual function calls, file paths, paper section references.
  4. Cite which source each claim came from at the end (e.g., "Source: Nanda 2023 §3.2").
  5. If the user's language is Korean, answer in Korean. Otherwise English.
  6. Keep answers focused. Long lists of options are less helpful than one clear path.
"""


def build_context_block(hits: list[RetrievalHit]) -> str:
    """Format retrieved hits into a labeled context block for the prompt."""
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[Source #{i}: {h.source} · {h.file}]\n{h.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_message(query: str, hits: list[RetrievalHit]) -> str:
    context = build_context_block(hits)
    return f"""Retrieved context:

{context}

---

User question:
{query}

Answer using ONLY the retrieved context. Cite sources by name (paper / file)."""


def answer(query: str, k: int = 6, source_filter: str | None = None,
           stream: bool = True) -> str:
    """Retrieve context and synthesize an answer via Claude.

    Args:
        query: natural-language question
        k: number of context chunks to retrieve
        source_filter: optional source restriction (e.g., "transformerlens")
        stream: if True, stream tokens to stdout as they arrive; also returns full text

    Returns:
        full answer text
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env in the project root."
        )

    from anthropic import Anthropic
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    hits = retrieve(query, k=k, source_filter=source_filter)
    user_msg = build_user_message(query, hits)

    if stream:
        full = []
        with client.messages.stream(
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        ) as s:
            for text in s.text_stream:
                print(text, end="", flush=True)
                full.append(text)
        print()
        return "".join(full)
    else:
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text
