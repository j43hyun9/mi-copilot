"""Techniques from Nanda et al. 2023 — 'Progress Measures for Grokking via Mechanistic Interpretability'.

Modules:
    fourier        — DFT-based weight analysis (W_E, W_U Fourier decomposition)
    ablation       — frequency-based causal ablation
    attribution    — logit decomposition (cos(2πk(a+b−c)/p) signature)
    progress       — restricted/excluded loss (progress measures)
    head_analysis  — per-head Fourier alignment, MLP neuron 2D activation
    viz            — standardized plots for all above
"""
