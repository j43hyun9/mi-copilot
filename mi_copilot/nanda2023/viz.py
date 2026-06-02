"""Standardized plots for Nanda 2023 analyses."""
from __future__ import annotations

import numpy as np


def plot_fourier_power(powers: np.ndarray, P: int, ax=None, color: str = "#2ca02c",
                       title: str = "Fourier power"):
    """Bar plot of Fourier power spectrum (unique half).

    Args:
        powers: shape (P,) — output of fourier.fft_power
        P: modulus
        ax: matplotlib axes (creates new if None)
        color, title: styling
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    half = P // 2 + 1
    ax.bar(np.arange(half), powers[:half], color=color)
    ax.set_xlabel("frequency k")
    ax.set_ylabel("power")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_unit_circle(PC: np.ndarray, P: int, expected: np.ndarray | None = None, ax=None):
    """Scatter PCA scores on 2D plane, optionally with expected (cos, sin) overlay.

    If PC[:, :2] forms a circle, that confirms Fourier embedding.

    Args:
        PC: shape (P, ≥2) — output of fourier.pca_embedding
        P: modulus
        expected: optional shape (P, 2) — output of fourier.expected_unit_circle
        ax: matplotlib axes
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(PC[:, 0], PC[:, 1], c=np.arange(P), cmap="hsv",
               s=50, edgecolors="black", linewidths=0.4, label="actual PCA")
    if expected is not None:
        ax.scatter(expected[:, 0], expected[:, 1], c=np.arange(P), cmap="hsv",
                   s=20, marker="x", label="expected (cos, sin)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("PC1 / cos")
    ax.set_ylabel("PC2 / sin")
    ax.legend()
    return ax


def plot_mlp_neuron_2d(acts_grid: np.ndarray, neuron_ids: list[int], axes=None):
    """Plot multiple MLP neurons as P×P heatmaps.

    Args:
        acts_grid: shape (P, P, d_mlp) — output of head_analysis.mlp_neuron_2d_activation
        neuron_ids: which neurons to plot
        axes: optional list of matplotlib axes (one per neuron)
    """
    import matplotlib.pyplot as plt
    n = len(neuron_ids)
    if axes is None:
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        _, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).flatten()
    for i, neuron in enumerate(neuron_ids):
        ax = axes[i]
        vmax = np.abs(acts_grid[:, :, neuron]).max()
        im = ax.imshow(acts_grid[:, :, neuron], cmap="RdBu_r", aspect="auto",
                       vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(f"Neuron {neuron}")
        ax.set_xlabel("b")
        ax.set_ylabel("a")
        plt.colorbar(im, ax=ax)
    return axes
