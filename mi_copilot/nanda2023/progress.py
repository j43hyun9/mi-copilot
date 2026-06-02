"""Progress measures (Nanda 2023, §6) — internal-circuit-growth metrics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .ablation import filter_to_frequencies


@torch.no_grad()
def restricted_loss(model, keep_ks: list[int], data) -> float:
    """Loss when W_E is restricted to only top-k frequencies.

    Monotonically *decreases* during training. Measures "current circuit strength":
    higher loss = circuit not yet formed; lower loss = circuit complete.

    Args:
        model: 1-layer transformer with .W_E
        keep_ks: frequencies to keep
        data: (inputs, labels) tensors

    Returns:
        cross-entropy loss (scalar)
    """
    device = next(model.parameters()).device
    P = getattr(model, "p", model.W_U.shape[-1])
    orig = model.W_E.detach().clone()

    W_E_np = model.W_E[:P].detach().cpu().numpy()
    filtered = filter_to_frequencies(W_E_np, keep_ks, axis=0)
    new_W_E = orig.clone()
    new_W_E[:P] = torch.tensor(filtered, dtype=orig.dtype, device=device)
    model.W_E.data = new_W_E

    try:
        inputs, labels = data
        logits = model(inputs.to(device))
        loss = F.cross_entropy(logits, labels.to(device)).item()
    finally:
        model.W_E.data = orig

    return loss


@torch.no_grad()
def excluded_loss(model, exclude_ks: list[int], data) -> float:
    """Loss when top-k frequencies are *removed* from W_E.

    Monotonically *increases* during training. Measures "memorization residue":
    high value = model now depends on the structured circuit; low value = model
    still relies on memorization.
    """
    device = next(model.parameters()).device
    P = getattr(model, "p", model.W_U.shape[-1])
    orig = model.W_E.detach().clone()

    W_E_np = model.W_E[:P].detach().cpu().numpy()
    all_ks = list(range(P // 2 + 1))
    keep_ks = [k for k in all_ks if k not in exclude_ks]
    filtered = filter_to_frequencies(W_E_np, keep_ks, axis=0)
    new_W_E = orig.clone()
    new_W_E[:P] = torch.tensor(filtered, dtype=orig.dtype, device=device)
    model.W_E.data = new_W_E

    try:
        inputs, labels = data
        logits = model(inputs.to(device))
        loss = F.cross_entropy(logits, labels.to(device)).item()
    finally:
        model.W_E.data = orig

    return loss
