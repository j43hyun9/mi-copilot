"""mi-copilot: RAG-based LLM assistant for Mechanistic Interpretability.

This package is primarily an LLM assistant. It does *not* reimplement MI
analysis techniques — instead, it guides users to apply existing tools
(TransformerLens, paper-specific repos like Charton's GCD code) via natural-
language queries answered with retrieval-augmented context.

Subpackages:
    llm_assistant — the RAG assistant (ingest / retrieve / chat / cli)
    common        — minimal utilities for reproduction scripts
                    (small custom transformer + data generators —
                    these are kept because TransformerLens targets
                    pre-trained large models, not small algorithmic-task
                    models trained from scratch as in Nanda 2023.)
    reproductions — validation scripts: did following the LLM's guidance
                    actually let us reproduce paper results?
"""

__version__ = "0.1.0"
