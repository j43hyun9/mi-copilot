"""Generate answers via OpenAI API using retrieved context."""
from __future__ import annotations

import os

from dotenv import load_dotenv

from .retrieve import RetrievalHit, retrieve

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

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


def answer_stream(query: str, k: int = 6, source_filter: str | None = None):
    """Retrieve context and stream answer tokens (generator-style).

    Returns:
        (hits, token_generator) — hits is list[RetrievalHit] for the UI to
        show as sources; token_generator yields str chunks for st.write_stream.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to .env.")

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    hits = retrieve(query, k=k, source_filter=source_filter)
    user_msg = build_user_message(query, hits)

    def token_gen():
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=2048,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    return hits, token_gen()


def answer(query: str, k: int = 6, source_filter: str | None = None,
           stream: bool = True) -> str:
    """Retrieve context and synthesize an answer via OpenAI.

    Args:
        query: natural-language question
        k: number of context chunks to retrieve
        source_filter: optional source restriction (e.g., "transformerlens")
        stream: if True, stream tokens to stdout as they arrive; also returns full text

    Returns:
        full answer text
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to .env in the project root."
        )

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    hits = retrieve(query, k=k, source_filter=source_filter)
    user_msg = build_user_message(query, hits)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    if stream:
        full: list[str] = []
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=2048,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                print(delta, end="", flush=True)
                full.append(delta)
        print()
        return "".join(full)
    else:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""
