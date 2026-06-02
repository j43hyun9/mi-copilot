"""Fourier-domain analysis of transformer weights (Nanda 2023, §2-3)."""
from __future__ import annotations

import numpy as np


def fft_power(W: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute Fourier power spectrum of a weight matrix along ``axis``.

    Args:
        W: weight matrix of shape (P, d) or (d, P)
        axis: axis along which the input index varies (default 0)

    Returns:
        power: power spectrum summed across the non-axis dim, length P
               (use ``power[:P//2 + 1]`` for the unique half)

    Example:
        >>> W_E = model.W_E[:113].detach().cpu().numpy()    # (113, d_model)
        >>> power = fft_power(W_E, axis=0)
        >>> half = power[:57]
        >>> top_k = np.argsort(half)[::-1][:5]    # top-5 dominant frequencies
    """
    fft = np.fft.fft(W, axis=axis)
    other = tuple(i for i in range(W.ndim) if i != axis)
    return (np.abs(fft) ** 2).sum(axis=other)


def top_frequencies(W: np.ndarray, k: int = 5, axis: int = 0,
                    exclude_dc: bool = True) -> list[int]:
    """Return indices of the top-k dominant Fourier frequencies in W.

    Args:
        W: weight matrix
        k: number of top frequencies to return
        axis: axis for FFT
        exclude_dc: if True (default), exclude k=0 (DC component)

    Returns:
        list of k integer frequency indices, sorted by descending power.
    """
    power = fft_power(W, axis=axis)
    P = power.shape[0]
    half = power[: P // 2 + 1].copy()
    if exclude_dc:
        half[0] = 0
    return np.argsort(half)[::-1][:k].tolist()


def pca_embedding(W: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """PCA on rows of W. Used to visualize embedding structure.

    Args:
        W: shape (P, d_model)
        n_components: number of principal components

    Returns:
        (PC_scores, explained_variance_ratio) where PC_scores has shape (P, n_components)

    Example:
        >>> W_E = model.W_E[:113].detach().cpu().numpy()
        >>> PC, evr = pca_embedding(W_E, n_components=2)
        >>> import matplotlib.pyplot as plt
        >>> plt.scatter(PC[:, 0], PC[:, 1], c=range(113), cmap="hsv")
        >>> # → 단위원 모양이 나타남 (Fourier 임베딩의 직접 증거)
    """
    X = W - W.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    PC = U[:, :n_components] * S[:n_components]
    total_var = (S ** 2).sum()
    evr = (S[:n_components] ** 2) / total_var
    return PC, evr


def expected_unit_circle(P: int, k: int) -> np.ndarray:
    """Expected (cos, sin) coordinates if W_E is Fourier-aligned at frequency k.

    Args:
        P: modulus (number of distinct inputs)
        k: frequency

    Returns:
        shape (P, 2) array with columns [cos(2πk·n/P), sin(2πk·n/P)] for n in 0..P-1.

    Used to verify the Fourier embedding hypothesis by comparing against
    PCA(W_E)[:, :2] — they should match up to rotation.
    """
    n = np.arange(P)
    theta = 2 * np.pi * k * n / P
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)
