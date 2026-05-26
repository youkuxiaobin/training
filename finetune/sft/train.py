"""Run supervised fine-tuning from an existing checkpoint."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune.sft.data import (
    IGNORE_INDEX,
    SFTSample,
    build_sft_samples,
    load_sft_examples,
    sample_sft_batch,
    split_sft_examples,
)
from finetune.sft.methods import (
    DEFAULT_LORA_TARGETS,
    FinetuneMethodConfig,
    apply_finetune_method,
    export_inference_model,
    trainable_parameter_count,
)
from language_model.config import GPTConfig
from language_model.gpt import GPTLanguageModel
from language_model.tokenization import (
    DEFAULT_SPECIAL_TOKENS,
    eos_token_id,
    load_language_tokenizer,
)
from language_model.training import cosine_lr, save_checkpoint, set_optimizer_lr
from scripts.train_small_model import resolve_device


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tune a local GPT model.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "runs" / "tiny_model")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--valid-input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "sft_model")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--train-on-prompt", action="store_true")
    parser.add_argument(
        "--method",
        choices=["full", "freeze", "lora", "qlora"],
        default="full",
        help="Fine-tuning method.",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default=DEFAULT_LORA_TARGETS)
    parser.add_argument(
        "--freeze-last-layers",
        type=int,
        default=0,
        help="For --method freeze, also train the last N transformer blocks.",
    )
    parser.add_argument(
        "--freeze-train-embeddings",
        action="store_true",
        help="For --method freeze, also train token embeddings.",
    )
    parser.add_argument(
        "--adapter-train-head",
        action="store_true",
        help="For LoRA/QLoRA, also train the output head.",
    )
    parser.add_argument(
        "--adapter-train-norms",
        action="store_true",
        help="For LoRA/QLoRA, also train norm layers.",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    run_sft(args)


def run_sft(args: argparse.Namespace) -> None:
    _validate_args(args)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "loading base model | model_dir=%s | checkpoint=%s | device=%s",
        args.model_dir,
        args.checkpoint,
        device,
    )
    checkpoint_path = args.model_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    special_tokens = list(checkpoint.get("special_tokens", DEFAULT_SPECIAL_TOKENS))
    tokenizer = load_language_tokenizer(args.model_dir, special_tokens)
    eos_id = eos_token_id(tokenizer, special_tokens[0])

    cfg = GPTConfig.from_dict(checkpoint["config"])
    model = GPTLanguageModel(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.set_gradient_checkpointing(args.gradient_checkpointing)
    method_config = build_method_config(args)
    apply_finetune_method(model, method_config)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise ValueError("selected fine-tuning method left no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    train_examples = load_sft_examples(args.train_input, max_samples=args.max_samples)
    if args.valid_input is None:
        train_examples, valid_examples = split_sft_examples(
            train_examples,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
    else:
        valid_examples = load_sft_examples(args.valid_input, max_samples=args.max_samples)

    train_samples = build_sft_samples(
        tokenizer,
        train_examples,
        context_length=cfg.context_length,
        eos_token=special_tokens[0],
        train_on_prompt=args.train_on_prompt,
    )
    valid_samples = build_sft_samples(
        tokenizer,
        valid_examples,
        context_length=cfg.context_length,
        eos_token=special_tokens[0],
        train_on_prompt=args.train_on_prompt,
    )
    _copy_tokenizer_files(args.model_dir, args.output_dir)

    logger.info(
        "prepared SFT data | train_examples=%d | valid_examples=%d "
        "| train_samples=%d | valid_samples=%d",
        len(train_examples),
        len(valid_examples),
        len(train_samples),
        len(valid_samples),
    )
    logger.info(
        "starting SFT loop | method=%s | trainable_parameters=%d/%d | steps=%d "
        "| batch_size=%d | lr=%.2e | min_lr=%.2e",
        method_config.method,
        trainable_parameter_count(model),
        model.num_parameters(),
        args.steps,
        args.batch_size,
        args.lr,
        args.min_lr,
    )

    generator = torch.Generator().manual_seed(args.seed)
    best_val_loss = float("inf")
    last_train_loss = None
    last_val_loss = None
    model.train()
    for step in range(1, args.steps + 1):
        lr = cosine_lr(
            step,
            max_lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            total_steps=args.steps,
        )
        set_optimizer_lr(optimizer, lr)
        input_ids, labels = sample_sft_batch(
            train_samples,
            batch_size=args.batch_size,
            pad_id=eos_id,
            device=device,
            generator=generator,
        )
        loss = sft_loss(model, input_ids, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        should_eval = step == 1 or step == args.steps or step % args.eval_interval == 0
        if should_eval:
            last_train_loss = estimate_sft_loss(
                model,
                train_samples,
                batch_size=args.batch_size,
                eval_iters=args.eval_iters,
                pad_id=eos_id,
                device=device,
                generator=generator,
            )
            last_val_loss = estimate_sft_loss(
                model,
                valid_samples,
                batch_size=args.batch_size,
                eval_iters=args.eval_iters,
                pad_id=eos_id,
                device=device,
                generator=generator,
            )
            is_best = last_val_loss < best_val_loss
            if is_best:
                best_val_loss = last_val_loss
            logger.info(
                "eval | step=%d/%d | train_loss=%.4f | val_loss=%.4f | lr=%.2e "
                "| best_val_loss=%.4f",
                step,
                args.steps,
                last_train_loss,
                last_val_loss,
                lr,
                best_val_loss,
            )
            _save_sft_checkpoint(
                args.output_dir / "latest.pt",
                model,
                optimizer,
                step,
                best_val_loss,
                special_tokens,
                last_train_loss,
                last_val_loss,
            )
            if is_best:
                _save_sft_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    optimizer,
                    step,
                    best_val_loss,
                    special_tokens,
                    last_train_loss,
                    last_val_loss,
                )

    _save_sft_checkpoint(
        args.output_dir / "model.pt",
        model,
        optimizer,
        args.steps,
        best_val_loss,
        special_tokens,
        last_train_loss,
        last_val_loss,
    )
    logger.info("finished SFT | final_checkpoint=%s", args.output_dir / "model.pt")


def sft_loss(
    model: GPTLanguageModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits, _ = model(input_ids)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


@torch.no_grad()
def estimate_sft_loss(
    model: GPTLanguageModel,
    samples: list[SFTSample],
    batch_size: int,
    eval_iters: int,
    pad_id: int,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> float:
    if eval_iters <= 0:
        raise ValueError("eval_iters must be positive")
    was_training = model.training
    model.eval()
    losses = []
    try:
        for _ in range(eval_iters):
            input_ids, labels = sample_sft_batch(
                samples,
                batch_size=batch_size,
                pad_id=pad_id,
                device=device,
                generator=generator,
            )
            losses.append(sft_loss(model, input_ids, labels).item())
    finally:
        model.train(was_training)
    return sum(losses) / len(losses)


def _save_sft_checkpoint(
    path: Path,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    special_tokens: list[str],
    train_loss: float | None,
    val_loss: float | None,
) -> None:
    checkpoint_model = export_inference_model(model)
    checkpoint_optimizer = (
        optimizer
        if checkpoint_model is model
        else torch.optim.AdamW(checkpoint_model.parameters(), lr=0.0)
    )
    save_checkpoint(
        path,
        model=checkpoint_model,
        optimizer=checkpoint_optimizer,
        step=step,
        best_val_loss=best_val_loss,
        special_tokens=special_tokens,
        train_loss=train_loss,
        val_loss=val_loss,
    )
    logger.info("saved checkpoint | path=%s", path)


def build_method_config(args: argparse.Namespace) -> FinetuneMethodConfig:
    return FinetuneMethodConfig(
        method=getattr(args, "method", "full"),
        lora_rank=getattr(args, "lora_rank", 8),
        lora_alpha=getattr(args, "lora_alpha", 16.0),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        lora_targets=getattr(args, "lora_targets", DEFAULT_LORA_TARGETS),
        freeze_last_layers=getattr(args, "freeze_last_layers", 0),
        freeze_train_embeddings=getattr(args, "freeze_train_embeddings", False),
        adapter_train_head=getattr(args, "adapter_train_head", False),
        adapter_train_norms=getattr(args, "adapter_train_norms", False),
    )


def _copy_tokenizer_files(model_dir: Path, output_dir: Path) -> None:
    for filename in ("vocab.json", "merges.json"):
        source = model_dir / filename
        target = output_dir / filename
        if source.resolve() == target.resolve():
            continue
        shutil.copy2(source, target)


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if args.eval_iters <= 0:
        raise ValueError("eval_iters must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if getattr(args, "lora_rank", 8) <= 0:
        raise ValueError("lora_rank must be positive")
    if getattr(args, "lora_alpha", 16.0) <= 0:
        raise ValueError("lora_alpha must be positive")
    lora_dropout = getattr(args, "lora_dropout", 0.05)
    if not 0 <= lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    if getattr(args, "freeze_last_layers", 0) < 0:
        raise ValueError("freeze_last_layers must be non-negative")


if __name__ == "__main__":
    main()
