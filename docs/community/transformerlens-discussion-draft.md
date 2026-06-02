# TransformerLens Community Discussion — Draft

> Draft of the Discussion post to open at
> https://github.com/TransformerLensOrg/TransformerLens/discussions
> Category: **Ideas** or **Show and Tell**

---

## Title

> **Proposal / Show & Tell: mi-copilot — a RAG-based learning assistant for MI newcomers, grounded in TransformerLens docs + foundational papers**

---

## Body

Hi all,

I'm a CS undergraduate working on a capstone project, and the deliverable is something I think might be useful for the broader MI community — so I wanted to share it here for feedback before announcing more widely.

### What I built

`mi-copilot` ([github.com/j43hyun9/mi-copilot](https://github.com/j43hyun9/mi-copilot)) is a domain-specific RAG (retrieval-augmented generation) assistant for mechanistic interpretability. The retrieval corpus is:

- **TransformerLens documentation** (README + `docs/source/content/**.md` + adapter / migration guides)
- **Foundational MI papers**: Nanda 2023 (grokking modular addition), Charton 2024 (GCD circuits), Friedman 2023 (Transformer Programs), Weiss 2021 (RASP)
- Its own thin docs

Queries are embedded with a multilingual model (so Korean/English mixed queries work — which matters for non-English MI learners), retrieved from a local ChromaDB, and answered by Claude with explicit "ground in the retrieved sources, cite which paper / file" instructions.

The motivation: TransformerLens has become the de facto standard for MI research, but for newcomers the on-ramp is steep — 50k+ LOC, dense docs, hundreds of functions. The existing support channels (GitHub Issues, Slack) are great but human-bandwidth-bound. A natural-language entry point that's grounded in the same docs+papers feels like a useful complement, not a replacement.

### Validation methodology

What makes this more than a generic doc-chatbot: practical utility is validated by **paper reproduction**.

- A persona (beginner / intermediate MI student) asks `mi-copilot` how to reproduce a specific paper figure.
- Following **only** the assistant's responses, the persona writes code.
- The resulting code is captured as a reproduction script.
- **Successful reproduction = the assistant is practically useful** (and grounded enough to not hallucinate the API).

Initial two scenarios:
- **Beginner** — reproduce Nanda 2023's W_E Fourier decomposition + frequency ablation.
- **Intermediate** — reproduce Charton 2024's staircase grokking + sieve-divisor accuracy split.

### What I'd love feedback on

1. **Scope of the retrieval corpus** — Are there MI docs / papers / blog posts you'd consider essential to include? (Currently: TL docs + 4 foundational papers. I have room for ~50 more documents before context-quality degrades.)

2. **Hallucination guardrails** — The assistant is instructed to ground in retrieved context only and to cite sources. What other guardrails have you seen work for domain-RAG (especially for code-recommendation answers)?

3. **Is there appetite for this as a TL companion project?** — I'd love to add a link to it from the TL README "Related projects" section once the assistant has been validated by a few external users. If a maintainer would consider that, I'm happy to open a tiny one-line PR. If it doesn't fit the project's scope, I'm equally happy to maintain it as a standalone — I don't want to scope-creep TL itself.

4. **TransformerLens-specific test queries** — If anyone has a set of "questions newcomers actually ask" that I could use as eval queries, that would be hugely valuable.

### What this is *not*

- Not a chatbot wrapped around the TL API — TL itself isn't part of the inference path; only its docs are retrieved.
- Not a replacement for reading the papers or the docs. It's a starting point that points users *at* the right paper section / docs page.
- Not yet polished for production use. This is a capstone project at v0.1, public for feedback.

Thanks for considering — I learned an enormous amount of MI by reading TL code while building this, and would be glad to give back in whatever form is useful.

— [@j43hyun9](https://github.com/j43hyun9)

---

## Posting checklist

- [ ] Wait until D-1 (after end-to-end Claude API test confirms assistant produces sensible output)
- [ ] Verify the repo URL works publicly (rename done)
- [ ] Push a clean tagged release `v0.1.0-capstone` so the linked repo state is stable
- [ ] Add screenshot or short asciinema of the CLI working
- [ ] Post in category **Ideas** (not Bug; not Q&A)
- [ ] Cross-link from the LessWrong / EleutherAI Discord posts if posted there too
