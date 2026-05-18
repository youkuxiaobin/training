"""Rotary positional encoding helpers."""

from __future__ import annotations

import torch


def build_rope_cache(
    context_length: int,
    head_dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (dim_indices / head_dim))
    positions = torch.arange(context_length, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.to(device=x.device, dtype=x.dtype).view(1, 1, x.size(-2), x.size(-1) // 2)
    sin = sin.to(device=x.device, dtype=x.dtype).view(1, 1, x.size(-2), x.size(-1) // 2)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        dim=-1,
    )
    return rotated.flatten(-2)
