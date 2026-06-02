"""Frequency-based causal ablation (Nanda 2023, §4)."""
from __future__ import annotations

import numpy as np
import torch


def filter_to_frequencies(W: np.ndarray, keep_ks: list[int], axis: int = 0) -> np.ndarray:
    """Keep only specified Fourier frequencies in W, zero out the rest.

    Used to verify causal sufficiency: if top-5 frequencies are kept and the
    model still achieves high accuracy, those frequencies are causally enough.

    Args:
        W: weight matrix (e.g., W_E of shape (P, d_model))
        keep_ks: list of frequency indices to keep (positive frequencies; their
                 negative counterparts P-k are kept automatically for real output)
        axis: axis of FFT

    Returns:
        filtered W of same shape and dtype as input

    Example:
        >>> W_E_top5 = filter_to_frequencies(W_E_np, keep_ks=[14, 35, 42, 52, 100])
        >>> # restore as model.W_E and check accuracy
    """
    P = W.shape[axis]
    fft = np.fft.fft(W, axis=axis)

    mask_shape = [1] * W.ndim
    mask_shape[axis] = P
    mask = np.zeros(P)
    for k in keep_ks:
        mask[k] = 1
        mask[(-k) % P] = 1  # negative freq for real output
    mask = mask.reshape(mask_shape)

    fft_filtered = fft * mask
    return np.real(np.fft.ifft(fft_filtered, axis=axis)).astype(W.dtype)


@torch.no_grad()
def frequency_ablation_accuracy(model, keep_ks: list[int], data, device=None) -> float:
    """Evaluate model accuracy with W_E restricted to only specified frequencies.

    Temporarily replaces model.W_E (number embeddings) with frequency-filtered
    version, runs evaluation, then restores. Use to find the minimum set of
    frequencies needed for high accuracy.

    Args:
        model: trained 1-layer transformer with .W_E attribute
        keep_ks: frequencies to keep
        data: (inputs, labels) tuple of tensors. inputs shape (N, T), labels shape (N,)
        device: torch device (defaults to model's device)

    Returns:
        accuracy in [0, 1]
    """
    if device is None:
        device = next(model.parameters()).device
    P = getattr(model, "p", model.W_U.shape[-1])

    # Save original W_E
    orig = model.W_E.detach().clone()

    # Filter number embeddings only (first P rows; last row = '=' token)
    W_E_np = model.W_E[:P].detach().cpu().numpy()
    filtered = filter_to_frequencies(W_E_np, keep_ks, axis=0)
    new_W_E = orig.clone()
    new_W_E[:P] = torch.tensor(filtered, dtype=orig.dtype, device=device)
    model.W_E.data = new_W_E

    try:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
    finally:
        # Always restore
        model.W_E.data = orig

    return acc
