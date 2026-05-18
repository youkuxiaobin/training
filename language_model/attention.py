"""Causal self-attention with RoPE."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from language_model.config import GPTConfig
from language_model.rope import apply_rope, build_rope_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.qkv_bias)
        self.attn_dropout = nn.Dropout(cfg.dropout_rate)
        self.resid_dropout = nn.Dropout(cfg.dropout_rate)

        rope_cos, rope_sin = build_rope_cache(
            cfg.context_length,
            self.head_dim,
            cfg.rope_theta,
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, channels = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)

        query = self._split_heads(query, batch_size, token_count)
        key = self._split_heads(key, batch_size, token_count)
        value = self._split_heads(value, batch_size, token_count)
        query = apply_rope(query, self.rope_cos[:token_count], self.rope_sin[:token_count])
        key = apply_rope(key, self.rope_cos[:token_count], self.rope_sin[:token_count])

        scores = query @ key.transpose(-1, -2)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:token_count, :token_count], -torch.inf)
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        context = weights @ value
        context = context.transpose(1, 2).contiguous().view(batch_size, token_count, channels)
        return self.resid_dropout(self.out_proj(context))

    def _split_heads(
        self,
        x: torch.Tensor,
        batch_size: int,
        token_count: int,
    ) -> torch.Tensor:
        return x.view(batch_size, token_count, self.n_head, self.head_dim).transpose(1, 2)
