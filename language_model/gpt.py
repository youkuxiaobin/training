"""GPT-style causal language model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from language_model.attention import KVCache
from language_model.blocks import TransformerBlock
from language_model.config import GPTConfig
from language_model.norms import build_norm


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout_rate)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = build_norm(cfg)
        self.out_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[
        torch.Tensor,
        torch.Tensor | None,
        list[KVCache],
    ]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")

        _, token_count = input_ids.shape
        if token_count > self.cfg.context_length:
            raise ValueError("input sequence is longer than context_length")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must match the number of blocks")

        x = self.token_emb(input_ids)
        x = self.dropout(x)
        present_key_values = []
        for index, block in enumerate(self.blocks):
            past_kv = None if past_key_values is None else past_key_values[index]
            if use_cache:
                x, present_kv = block(x, past_kv=past_kv, use_cache=True)
                present_key_values.append(present_kv)
            else:
                x = block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        if use_cache:
            return logits, loss, present_key_values
        return logits, loss

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
