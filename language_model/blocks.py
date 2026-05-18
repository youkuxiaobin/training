"""Transformer block composition."""

from __future__ import annotations

import torch
from torch import nn

from language_model.attention import CausalSelfAttention
from language_model.config import GPTConfig
from language_model.feed_forward import FeedForward
from language_model.norms import build_norm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.norm1 = build_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = build_norm(cfg)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
