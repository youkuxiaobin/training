"""Autoregressive text generation helpers."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from language_model.gpt import GPTLanguageModel


@torch.no_grad()
def generate(
    model: GPTLanguageModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, tokens]")
    if input_ids.size(1) == 0:
        raise ValueError("input_ids must contain at least one token")

    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            input_window = input_ids[:, -model.cfg.context_length :]
            logits, _ = model(input_window)
            logits = logits[:, -1, :]

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    keep = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, keep).values[:, [-1]]
                    logits = logits.masked_fill(logits < threshold, -torch.inf)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_id), dim=1)
            if eos_id is not None and torch.all(next_id == eos_id):
                break
    finally:
        model.train(was_training)
    return input_ids
