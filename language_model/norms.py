"""Normalization layers used by the language model."""

from __future__ import annotations

import torch
from torch import nn

from language_model.config import GPTConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms


def build_norm(cfg: GPTConfig) -> nn.Module:
    if cfg.norm_type == "rmsnorm":
        return RMSNorm(cfg.n_embd, cfg.norm_eps)
    if cfg.norm_type == "layernorm":
        return nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps)
    raise ValueError(f"unsupported norm_type: {cfg.norm_type}")
