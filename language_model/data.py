"""Data helpers for next-token language-model training."""

from __future__ import annotations

import torch


def make_token_tensor(token_ids: list[int]) -> torch.Tensor:
    if len(token_ids) < 2:
        raise ValueError("at least two token ids are required")
    return torch.tensor(token_ids, dtype=torch.long)


def get_batch(
    token_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be a 1D tensor")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if token_ids.numel() <= context_length:
        raise ValueError("token_ids must be longer than context_length")

    max_start = token_ids.numel() - context_length
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    x = torch.stack([token_ids[start : start + context_length] for start in starts])
    y = torch.stack([token_ids[start + 1 : start + context_length + 1] for start in starts])
    return x.to(device), y.to(device)
