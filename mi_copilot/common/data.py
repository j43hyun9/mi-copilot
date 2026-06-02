"""Standard data generators for grokking tasks."""
from __future__ import annotations

import torch


def modular_addition_data(p: int = 113, train_frac: float = 0.3,
                          seed: int = 42, device: str = "cpu"
                          ) -> tuple[tuple[torch.Tensor, torch.Tensor],
                                     tuple[torch.Tensor, torch.Tensor]]:
    """All (a, b, c=a+b mod p) triples, split into train/test.

    Token sequence: [a, b, "="] where "=" has id p. Target is c at last position.

    Args:
        p: prime modulus (113 in Nanda 2023)
        train_frac: fraction for train set
        seed: rng seed
        device: torch device

    Returns:
        ((train_x, train_y), (test_x, test_y)) where
        x has shape (N, 3), y has shape (N,).
    """
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)
    x = torch.stack([a, b, eq], dim=1)
    y = (a + b) % p

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=g)
    cut = int(p * p * train_frac)
    train_idx, test_idx = perm[:cut], perm[cut:]

    return ((x[train_idx].to(device), y[train_idx].to(device)),
            (x[test_idx].to(device), y[test_idx].to(device)))
