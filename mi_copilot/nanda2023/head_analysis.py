"""Per-head circuit analysis (Nanda 2023, §7)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def per_head_fourier(model) -> tuple[np.ndarray, list[int]]:
    """Compute Fourier power spectrum of each attention head's effective OV circuit.

    For head h, the effective input-to-output map is
        W_E[:P] @ W_V[h] @ W_O[h]
    Applying FFT along the input axis (P) reveals which frequency k each head
    specializes in.

    Args:
        model: 1-layer transformer with .W_E, .W_V, .W_O, .n_heads, .p

    Returns:
        powers: shape (n_heads, P) — full Fourier power per head per frequency
        dominant_ks: list of length n_heads — top non-DC frequency per head
    """
    P = getattr(model, "p", model.W_U.shape[-1])
    n_heads = model.n_heads
    W_E_p = model.W_E[:P].detach().cpu().numpy()    # (P, d_model)

    powers = np.zeros((n_heads, P))
    dominant_ks: list[int] = []
    half = P // 2 + 1
    for h in range(n_heads):
        eff = W_E_p @ model.W_V[h].detach().cpu().numpy() @ model.W_O[h].detach().cpu().numpy()
        fft = np.fft.fft(eff, axis=0)
        powers[h] = (np.abs(fft) ** 2).sum(axis=1)
        half_pow = powers[h, :half].copy()
        half_pow[0] = 0  # exclude DC
        dominant_ks.append(int(np.argmax(half_pow)))

    return powers, dominant_ks


@torch.no_grad()
def mlp_neuron_2d_activation(model, batch_size: int = 128) -> np.ndarray:
    """MLP neuron activation as function of (a, b) for all input pairs.

    The Nanda 2023 algorithm predicts MLP neurons compute trig products like
    cos(2πk·a/p)·cos(2πk·b/p) — observable as smooth product patterns in this
    P × P × d_mlp tensor.

    Args:
        model: 1-layer transformer
        batch_size: chunks of `a` values per forward pass

    Returns:
        acts_grid: shape (P, P, d_mlp) — neuron activations at "=" position
                   for input (a, b) ∈ {0..P-1}².
    """
    device = next(model.parameters()).device
    P = getattr(model, "p", model.W_U.shape[-1])

    all_acts = []
    for a_start in range(0, P, batch_size):
        a_end = min(a_start + batch_size, P)
        chunk = a_end - a_start
        a_grid = torch.arange(a_start, a_end, device=device)[:, None].expand(chunk, P)
        b_grid = torch.arange(P, device=device)[None, :].expand(chunk, P)
        eq_tok = torch.full((chunk, P), P, device=device)
        tokens = torch.stack([a_grid, b_grid, eq_tok], dim=-1).reshape(-1, 3)

        x = model.W_E[tokens] + model.W_pos[None, :, :]
        B, T, _ = x.shape
        q = torch.einsum("btd,hde->bthe", x, model.W_Q)
        k = torch.einsum("btd,hde->bthe", x, model.W_K)
        v = torch.einsum("btd,hde->bthe", x, model.W_V)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(model.d_head)
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))
        pattern = scores.softmax(dim=-1)
        attn_out = torch.einsum("bhqk,bkhd->bqhd", pattern, v)
        attn_out = torch.einsum("bthd,hde->bte", attn_out, model.W_O)
        x = x + attn_out
        h_act = F.relu(x @ model.W_in + model.b_in)
        all_acts.append(h_act[:, -1, :].cpu())     # "=" position

    return torch.cat(all_acts, dim=0).reshape(P, P, -1).numpy()


def top_neurons_by_variance(acts_grid: np.ndarray, k: int = 6) -> list[int]:
    """Pick the k MLP neurons with highest variance across (a, b) inputs.

    These are the "interesting" neurons most likely to show clean trig patterns.
    """
    variances = acts_grid.var(axis=(0, 1))
    return np.argsort(variances)[::-1][:k].tolist()
