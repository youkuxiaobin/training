"""Feed-forward network used inside each Transformer block."""

from __future__ import annotations

import torch
from torch import nn

from language_model.config import GPTConfig


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
