# mi-copilot

> **RAG-based LLM assistant for Mechanistic Interpretability** — guides users through MI paper reproduction via natural language.

Capstone project (2026 Spring, 컴퓨터공학과). The first docs-RAG learning assistant for the mechanistic interpretability research field.

## What

`mi-copilot` is a domain-specific LLM assistant that helps users — from absolute beginners to MI researchers — navigate the mechanistic interpretability literature and reproduce key results. It retrieves over:

- A bundled library of MI techniques (`mi_copilot/nanda2023/`, `mi_copilot/charton2024/`)
- TransformerLens documentation
- Foundational papers (Nanda 2023, Charton 2024, and more)

and synthesizes answers via Claude API.

## Why

Existing MI tooling (TransformerLens, SAELens, CircuitsVis) is powerful but unforgiving:
- Steep learning curve (50K+ LOC, hundreds of functions)
- Documentation is static (markdown + Jupyter)
- Community support is human-only (GitHub Issues, Slack)
- No conversational entry point for newcomers

`mi-copilot` fills this gap.

## Validation

The assistant's practical utility is validated by **paper reproduction via LLM guidance**:

1. A persona (beginner or intermediate) asks `mi-copilot` how to reproduce a specific paper.
2. Following only the assistant's responses, the persona writes and runs the code.
3. **Successful reproduction = validated practical utility.**

Two reproduction scenarios:
- **Beginner persona** — reproduces Nanda 2023 modular addition grokking analysis.
- **Intermediate persona** — reproduces Charton 2024 GCD staircase grokking + sieve algorithm.

## Quickstart

```bash
pip install mi-copilot
mi-copilot ask "Nanda 2023 의 modular addition 회로 분석을 어떻게 재현하나요?"
```

Or programmatically:

```python
from mi_copilot.nanda2023 import fourier, attribution
from mi_copilot.common import models

model = models.OneLayerTransformer(p=113)
# ... train ...
fourier.fft_weights(model.W_E).plot()                # W_E Fourier decomposition
attribution.logit_fourier_decomp(model).plot()       # final logit signature
```

## Status

Active development. Capstone presentation: 2026-06-05.

## License

MIT
