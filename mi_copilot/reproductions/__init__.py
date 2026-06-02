"""Validation scripts — did the LLM's guidance reproduce the papers?

Each script is the *output* of a persona-driven LLM session:
    1. A persona (beginner / intermediate) asks mi-copilot how to reproduce paper X.
    2. The persona follows ONLY the LLM's responses to write code.
    3. The resulting code is captured here and run to confirm reproduction.

Successful reproduction = the LLM is practically useful.

Scripts (planned):
    nanda2023_modular.py  — beginner persona reproduces Nanda's modular addition figures
                             (Fourier decomposition, PCA unit circle, frequency ablation)
                             using TransformerLens-style analysis on a custom small model.
    charton2024_gcd.py    — intermediate persona reproduces Charton's GCD staircase grokking
                             using the original Charton repo + custom analysis hooks.
"""
