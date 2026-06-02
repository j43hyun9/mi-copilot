"""Standard transformer model class used across mi_copilot."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OneLayerTransformer(nn.Module):
    """Single-layer decoder-only transformer matching Nanda 2023 setup.

    Designed for small algorithmic tasks (modular arithmetic, GCD, etc.).
    All weight names match conventions used in the mi_copilot.nanda2023 module.

    Architecture:
        x = W_E[tokens] + W_pos
        attn(x):  Q, K, V via W_Q, W_K, W_V;  out via W_O;  causal mask
        mlp(x):   F.relu(x @ W_in + b_in) @ W_out + b_out
        logits:   x[:, -1] @ W_U      (last-position only)

    Args:
        p: modulus / output vocab size (numbers 0 .. p-1)
        d_model: residual stream dimension
        d_head: per-head attention dimension
        n_heads: number of attention heads
        d_mlp: MLP hidden dimension
        n_ctx: context length (e.g., 3 for [a, b, "="])
    """

    def __init__(self, p: int, d_model: int = 128, d_head: int = 32,
                 n_heads: int = 4, d_mlp: int = 512, n_ctx: int = 3):
        super().__init__()
        self.p, self.d_model, self.n_heads, self.d_head = p, d_model, n_heads, d_head

        # +1 for "=" token at index p
        self.W_E = nn.Parameter(torch.randn(p + 1, d_model) / math.sqrt(d_model))
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d_model) / math.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_model, d_head) / math.sqrt(d_model))
        self.W_K = nn.Parameter(torch.randn(n_heads, d_model, d_head) / math.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(n_heads, d_model, d_head) / math.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(n_heads, d_head, d_model) / math.sqrt(d_model))
        self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) / math.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) / math.sqrt(d_mlp))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.W_U = nn.Parameter(torch.randn(d_model, p) / math.sqrt(d_model))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run forward pass, return logits at the final position.

        Args:
            tokens: shape (B, T) of long integers

        Returns:
            logits: shape (B, p)
        """
        x = self.W_E[tokens] + self.W_pos[None, :, :]
        T = x.shape[1]

        # Self-attention
        q = torch.einsum("btd,hde->bthe", x, self.W_Q)
        k = torch.einsum("btd,hde->bthe", x, self.W_K)
        v = torch.einsum("btd,hde->bthe", x, self.W_V)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()
        scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))
        pattern = scores.softmax(dim=-1)
        attn_out = torch.einsum("bhqk,bkhd->bqhd", pattern, v)
        attn_out = torch.einsum("bthd,hde->bte", attn_out, self.W_O)
        x = x + attn_out

        # MLP
        h = F.relu(x @ self.W_in + self.b_in)
        x = x + h @ self.W_out + self.b_out

        return x[:, -1, :] @ self.W_U
