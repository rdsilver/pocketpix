"""Quantisers with straight-through estimators (STE).

Every `*_values` function returns the quantised tensor with no gradient path; the public wrappers add
the STE: forward = quantised value, backward = identity to the latent full-precision tensor.

Weights have shape (..., n_out, n_in). Leading dimensions index independent matrices (experts,
product-key heads); the last dimension is the INPUT dimension, the one the GBA table kernel walks
4 weights at a time.
"""
from __future__ import annotations

import torch

WQ_MODES = ("fp", "tern", "tern_median", "sherry")
KEY_MODES = ("none", "sign", "tern34")


def ste(x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """Value of xq, gradient of x."""
    return x + (xq - x).detach()


def hard_sign(x: torch.Tensor) -> torch.Tensor:
    """sign() that never returns 0 (zero counts as positive), so a 'non-zero' trit really is +-1."""
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


# --------------------------------------------------------------------------- weights

def tern_values(w: torch.Tensor, median: bool = False, eps: float = 1e-5) -> torch.Tensor:
    """BitNet b1.58: Wq = clip(round(W / s), -1, 1) * s, with s = mean|W| (or median|W|) per matrix."""
    a = w.abs().flatten(-2)
    s = (a.median(dim=-1).values if median else a.mean(dim=-1))[..., None, None]
    return torch.clamp(torch.round(w / (s + eps)), -1, 1) * s


def sherry_pattern(w: torch.Tensor) -> torch.Tensor:
    """Trits in {-1, 0, +1} with exactly one zero (the smallest |w|, first on ties) per block of 4 inputs."""
    n_in = w.shape[-1]
    assert n_in % 4 == 0, f"sherry needs an input dimension that is a multiple of 4, got {n_in}"
    blocks = w.reshape(*w.shape[:-1], n_in // 4, 4)
    keep = torch.ones_like(blocks)
    keep.scatter_(-1, blocks.abs().argmin(dim=-1, keepdim=True), 0.0)
    return (hard_sign(blocks) * keep).reshape(w.shape)


def sherry_values(w: torch.Tensor) -> torch.Tensor:
    """Sherry 3:4: pattern * alpha, alpha = mean |w| over the non-zero positions of each row."""
    t = sherry_pattern(w)
    alpha = (w.abs() * t.abs()).sum(dim=-1, keepdim=True) / (0.75 * w.shape[-1])
    return t * alpha


def weight_values(w: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "tern":
        return tern_values(w, median=False)
    if mode == "tern_median":
        return tern_values(w, median=True)
    if mode == "sherry":
        return sherry_values(w)
    raise ValueError(f"no weight quantiser for mode {mode!r}")


def quantize_weight(w: torch.Tensor, mode: str) -> torch.Tensor:
    """Quantised weight with a straight-through gradient to the latent weight."""
    return ste(w, weight_values(w, mode))


# --------------------------------------------------------------------------- activations

def act_values(x: torch.Tensor, bits: int, eps: float = 1e-5) -> torch.Tensor:
    """Per-token absmax quantisation to `bits` (symmetric: integer levels -Q..Q, Q = 2^(bits-1) - 1)."""
    q = float(2 ** (bits - 1) - 1)
    scale = q / x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    return torch.clamp(torch.round(x * scale), -q, q) / scale


def quantize_act(x: torch.Tensor, bits: int) -> torch.Tensor:
    return ste(x, act_values(x, bits))


def quantize_rows(table_rows: torch.Tensor, bits: int) -> torch.Tensor:
    """8-bit-style storage of lookup rows: one absmax scale per row (last dim), straight-through."""
    return ste(table_rows, act_values(table_rows, bits))


# --------------------------------------------------------------------------- attention keys

def key_values(k: torch.Tensor, mode: str) -> torch.Tensor:
    """sign: every key dim becomes +-1.  tern34: in each block of 4 dims the smallest |k| becomes 0."""
    if mode == "sign":
        return hard_sign(k)
    if mode == "tern34":
        return sherry_pattern(k)
    raise ValueError(f"no key quantiser for mode {mode!r}")


def quantize_key(k: torch.Tensor, mode: str) -> torch.Tensor:
    return ste(k, key_values(k, mode))
