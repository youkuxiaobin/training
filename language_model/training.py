"""Training helpers for small language-model pretraining."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch

from language_model.data import get_batch
from language_model.gpt import GPTLanguageModel


def cosine_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    if step <= 0:
        raise ValueError("step must be positive")
    if max_lr <= 0:
        raise ValueError("max_lr must be positive")
    if min_lr < 0:
        raise ValueError("min_lr must be non-negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if min_lr > max_lr:
        raise ValueError("min_lr must be less than or equal to max_lr")

    if warmup_steps > 0 and step <= warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr

    decay_steps = max(total_steps - warmup_steps, 1)
    progress = (step - warmup_steps) / decay_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    eval_iters: int,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    if eval_iters <= 0:
        raise ValueError("eval_iters must be positive")

    was_training = model.training
    model.eval()
    losses: dict[str, float] = {}
    try:
        for split, tokens in {"train": train_tokens, "val": val_tokens}.items():
            split_losses = []
            for _ in range(eval_iters):
                x, y = get_batch(
                    tokens,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device,
                    generator=generator,
                )
                _, loss = model(x, y)
                assert loss is not None
                split_losses.append(loss.item())
            losses[split] = sum(split_losses) / len(split_losses)
    finally:
        model.train(was_training)
    return losses


def save_checkpoint(
    path: str | Path,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    special_tokens: Sequence[str],
    train_loss: float | None = None,
    val_loss: float | None = None,
) -> None:
    checkpoint = {
        "config": model.cfg.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "special_tokens": list(special_tokens),
    }
    torch.save(checkpoint, path)
