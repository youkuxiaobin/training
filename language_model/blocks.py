"""Transformer block composition."""

from __future__ import annotations

import torch
from torch import nn

from language_model.attention import CausalSelfAttention, KVCache
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

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        attn_output = self.attn(self.norm1(x), past_kv=past_kv, use_cache=use_cache)
        present_kv = None
        if use_cache:
            attn_output, present_kv = attn_output
        x = x + attn_output
        x = x + self.ff(self.norm2(x))
        if use_cache:
            assert present_kv is not None
            return x, present_kv
        return x
