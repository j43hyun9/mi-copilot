# mi-copilot

> **RAG-based LLM assistant for Mechanistic Interpretability** — guides users through MI paper reproduction by retrieving from existing tools (TransformerLens, paper repos) and synthesizing actionable answers via Claude.

Capstone project (2026 Spring, 컴퓨터공학과). The first docs-RAG learning assistant for the mechanistic interpretability research field.

## What this is

`mi-copilot` is a domain-specific LLM assistant. **It does not reimplement MI techniques.** Instead, it retrieves from:

- **TransformerLens** documentation (the de facto standard MI library)
- Foundational MI papers (Nanda 2023, Charton 2024, and more)
- Companion docs in this repository

and then guides users — from absolute beginners to MI researchers — on how to apply the existing tools to specific tasks via natural-language queries answered by Claude.

## Why

The MI tooling ecosystem (TransformerLens, SAELens, CircuitsVis) is powerful but unforgiving for newcomers:

- Steep learning curve (TransformerLens alone: 50K+ LOC, hundreds of functions)
- Documentation is static (markdown + Jupyter)
- Community support is human-only (GitHub Issues, Slack)
- No conversational entry point

`mi-copilot` fills this gap by giving newcomers a natural-language interface to the entire MI knowledge base.

## Validation

Practical utility is validated by **paper reproduction via LLM guidance**:

1. A persona (beginner or intermediate) asks `mi-copilot` how to reproduce a specific paper.
2. Following only the assistant's responses, the persona writes and runs the code (using TransformerLens, paper-specific repos, etc.).
3. **Successful reproduction = the LLM is practically useful.**

Two scenarios:
- **Beginner persona** — Nanda 2023 modular addition (Fourier circuit analysis)
- **Intermediate persona** — Charton 2024 GCD (staircase grokking + sieve algorithm)

The reproduction scripts captured in `mi_copilot/reproductions/` are the *output* of these persona sessions and serve as evidence.

## Architecture

```
mi_copilot/
├── llm_assistant/       ← THE PRIMARY DELIVERABLE
│   ├── ingest.py        ─ vectorize TransformerLens docs + papers + own docs → ChromaDB
│   ├── retrieve.py      ─ semantic search over the vector store
│   ├── chat.py          ─ Claude API + retrieved context → answer
│   └── cli.py           ─ `mi-copilot ask "..."` command
├── common/              ← minimal training utilities (TransformerLens targets
│   ├── models.py        ─ pre-trained LLMs, not small algorithmic-task models;
│   └── data.py          ─ so a small custom transformer helper lives here)
└── reproductions/       ← validation: LLM-guided paper reproductions
    ├── nanda2023_modular.py
    └── charton2024_gcd.py
```

## Quickstart

```bash
pip install -e .
cp .env.example .env  # then add your ANTHROPIC_API_KEY
mi-copilot ingest      # build local vector store (one-time)
mi-copilot ask "Nanda 2023 의 modular addition 회로 분석을 어떻게 재현하나요?"
```

## Status

Active development. Capstone presentation: **2026-06-05**.

## License

MIT
