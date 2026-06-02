"""Logit decomposition / attribution (Nanda 2023, §5 — trig identity signature)."""
from __future__ import annotations

import numpy as np


def logit_fourier_signature(P: int, k: int, a: int, b: int) -> np.ndarray:
    """Expected logit vector if model uses trig-identity algorithm at freq k.

    The Nanda 2023 algorithm produces:
        logit(c | a, b) ∝ cos(2π · k · (a + b − c) / P)

    This peaks at c = (a + b) mod P, giving cos(0) = 1.

    Args:
        P: modulus
        k: frequency
        a, b: input numbers

    Returns:
        shape (P,) — expected logit shape for output positions c ∈ {0, ..., P-1}
    """
    c = np.arange(P)
    return np.cos(2 * np.pi * k * (a + b - c) / P)


def decompose_logits_fourier(logits: np.ndarray, P: int) -> np.ndarray:
    """FFT-decompose model logits over output index.

    Args:
        logits: shape (..., P) — model output logits
        P: modulus (= logits.shape[-1])

    Returns:
        power spectrum along output dim. If the trig-identity signature is
        present, power concentrates at the model's dominant frequencies.
    """
    fft = np.fft.fft(logits, axis=-1)
    return (np.abs(fft) ** 2).sum(axis=tuple(range(logits.ndim - 1)))


def match_trig_signature(actual_logits: np.ndarray, P: int, a: int, b: int,
                          freqs: list[int]) -> float:
    """Correlation between actual model logits and predicted trig-identity logits.

    Args:
        actual_logits: shape (P,) — model output for input (a, b)
        P: modulus
        a, b: inputs
        freqs: list of k values the model is hypothesized to use

    Returns:
        Pearson correlation in [-1, 1]. Close to 1.0 = trig identity confirmed.
    """
    predicted = sum(logit_fourier_signature(P, k, a, b) for k in freqs)
    actual = actual_logits - actual_logits.mean()
    predicted = predicted - predicted.mean()
    denom = np.linalg.norm(actual) * np.linalg.norm(predicted)
    if denom == 0:
        return 0.0
    return float(np.dot(actual, predicted) / denom)
