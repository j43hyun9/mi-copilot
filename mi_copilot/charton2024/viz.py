"""Visualizations for GCD circuit analysis (Charton 2024)."""
from __future__ import annotations

import numpy as np


def plot_epoch_gcd_heatmap(entries: list[dict], gcd_max: int = 30,
                            base: int | None = 1000, ax=None):
    """Heatmap of test accuracy per (epoch, GCD value). Reveals staircase pattern.

    Each row (GCD k) shows red→green phase transition at the epoch where the
    model "learns" that GCD.

    Args:
        entries: parsed Charton log entries
        gcd_max: max GCD to display
        base: if given, overlay cyan lines at base-divisors (sieve-learnable GCDs)
        ax: matplotlib axes
    """
    import matplotlib.pyplot as plt
    from .divisor_analysis import base_divisors

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 6))

    epochs = [e["epoch"] for e in entries]
    heatmap = np.full((len(entries), gcd_max + 1), np.nan)
    for i, e in enumerate(entries):
        for g in range(gcd_max + 1):
            v = e.get(f"test_arithmetic_acc_{g}")
            if v is not None:
                heatmap[i, g] = v

    im = ax.imshow(heatmap.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100,
                   origin="lower", extent=[epochs[0], epochs[-1], 0, gcd_max])
    ax.set_xlabel("epoch")
    ax.set_ylabel("GCD k")
    if base is not None:
        for g in base_divisors(base, max_gcd=gcd_max):
            ax.axhline(g, color="cyan", alpha=0.3, linewidth=0.5)
    return ax, im


def plot_per_gcd_bar(per_gcd_acc: dict[int, float], base: int = 1000,
                     gcd_max: int = 30, ax=None):
    """Bar chart of per-GCD accuracy at a single epoch. Color-code by base-divisor status.

    Green bar = base-divisor (sieve-learnable);  Red bar = non-divisor.
    """
    import matplotlib.pyplot as plt
    from .divisor_analysis import base_divisors

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4.5))

    divs = set(base_divisors(base, max_gcd=gcd_max))
    gcds = sorted(g for g in per_gcd_acc if g <= gcd_max)
    accs = [per_gcd_acc[g] for g in gcds]
    colors = ["#2ca02c" if g in divs else "#d62728" for g in gcds]
    ax.bar(gcds, accs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("GCD k")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(-2, 105)
    ax.grid(alpha=0.3, axis="y")
    return ax
