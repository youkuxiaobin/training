"""Feed-forward network used inside each Transformer block."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from language_model.config import GPTConfig


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ffn_type = cfg.ffn_type
        hidden_size = 4 * cfg.n_embd
        if self.ffn_type == "swiglu":
            self.gate_proj = nn.Linear(cfg.n_embd, hidden_size)
            self.up_proj = nn.Linear(cfg.n_embd, hidden_size)
            self.down_proj = nn.Linear(hidden_size, cfg.n_embd)
            self.dropout = nn.Dropout(cfg.dropout_rate)
        elif self.ffn_type == "gelu":
            self.net = nn.Sequential(
                nn.Linear(cfg.n_embd, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, cfg.n_embd),
                nn.Dropout(cfg.dropout_rate),
            )
        else:
            raise ValueError(f"unsupported ffn_type: {self.ffn_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_type == "swiglu":
            x = F.silu(self.gate_proj(x)) * self.up_proj(x)
            return self.dropout(self.down_proj(x))
        return self.net(x)
