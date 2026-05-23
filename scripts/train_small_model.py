from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from language_model.data import get_batch, make_token_tensor, split_train_val
from language_model.gpt import GPTLanguageModel
from language_model.config import GPTConfig
from language_model.training import (
    cosine_lr,
    estimate_loss,
    save_checkpoint,
    set_optimizer_lr,
)
from language_model.tokenization import (
    DEFAULT_SPECIAL_TOKENS,
    encode_text,
    read_training_text,
    train_language_tokenizer,
)


class CorpusInputs(NamedTuple):
    train_input: Path
    valid_input: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small GPT-style model.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "examples" / "tiny_corpus.txt",
        help="Backward-compatible training input path. Use --train-input for explicit train/valid files.",
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        default=None,
        help="Training corpus file or directory. Overrides --input when provided.",
    )
    parser.add_argument(
        "--valid-input",
        type=Path,
        default=None,
        help="Validation corpus file or directory. If omitted, validation is split from training data.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "tiny_model")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=96)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--n-head", type=int, default=3)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--norm-type", choices=["rmsnorm", "layernorm"], default="rmsnorm")
    parser.add_argument("--ffn-type", choices=["swiglu", "gelu"], default="swiglu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_corpus_inputs(args: argparse.Namespace) -> CorpusInputs:
    train_input = args.train_input if args.train_input is not None else args.input
    valid_input = args.valid_input
    if valid_input is not None and train_input.resolve() == valid_input.resolve():
        raise ValueError("train_input and valid_input must be different paths")
    return CorpusInputs(train_input=train_input, valid_input=valid_input)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_train_val_tokens(
    tokenizer,
    train_input: Path,
    valid_input: Path | None,
    val_fraction: float,
    context_length: int,
    eos_token: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_text = read_training_text(train_input, eos_token)
    train_tokens = make_token_tensor(encode_text(tokenizer, train_text))
    if train_tokens.numel() <= context_length:
        raise ValueError("train input text is too short for the requested context length")

    if valid_input is None:
        return split_train_val(
            train_tokens,
            val_fraction=val_fraction,
            context_length=context_length,
        )

    valid_text = read_training_text(valid_input, eos_token)
    valid_tokens = make_token_tensor(encode_text(tokenizer, valid_text))
    if valid_tokens.numel() <= context_length:
        raise ValueError("valid input text is too short for the requested context length")
    return train_tokens, valid_tokens


def main() -> None:
    args = parse_args()
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if args.eval_iters <= 0:
        raise ValueError("eval_iters must be positive")
    torch.manual_seed(args.seed)

    corpus_inputs = resolve_corpus_inputs(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = train_language_tokenizer(
        corpus_inputs.train_input,
        args.vocab_size,
        DEFAULT_SPECIAL_TOKENS,
    )
    tokenizer.save(args.output_dir / "vocab.json", args.output_dir / "merges.json")

    train_tokens, val_tokens = build_train_val_tokens(
        tokenizer,
        train_input=corpus_inputs.train_input,
        valid_input=corpus_inputs.valid_input,
        val_fraction=args.val_fraction,
        context_length=args.context_length,
        eos_token=DEFAULT_SPECIAL_TOKENS[0],
    )

    cfg = GPTConfig(
        vocab_size=len(tokenizer.vocab),
        context_length=args.context_length,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout_rate=args.dropout_rate,
        norm_type=args.norm_type,
        ffn_type=args.ffn_type,
    )
    device = resolve_device(args.device)
    model = GPTLanguageModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
        x, y = get_batch(
            train_tokens,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
            generator=generator,
        )
        _, loss = model(x, y)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        should_eval = step == 1 or step == args.steps or step % args.eval_interval == 0
        if should_eval:
            losses = estimate_loss(
                model,
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
                eval_iters=args.eval_iters,
                generator=generator,
            )
            last_train_loss = losses["train"]
            last_val_loss = losses["val"]
            is_best = last_val_loss < best_val_loss
            if is_best:
                best_val_loss = last_val_loss
            print(
                f"step {step:4d} | train {last_train_loss:.4f} "
                f"| val {last_val_loss:.4f} | lr {lr:.2e}"
            )
            save_checkpoint(
                args.output_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                best_val_loss=best_val_loss,
                special_tokens=DEFAULT_SPECIAL_TOKENS,
                train_loss=last_train_loss,
                val_loss=last_val_loss,
            )
            if is_best:
                save_checkpoint(
                    args.output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_val_loss=best_val_loss,
                    special_tokens=DEFAULT_SPECIAL_TOKENS,
                    train_loss=last_train_loss,
                    val_loss=last_val_loss,
                )

    save_checkpoint(
        args.output_dir / "model.pt",
        model=model,
        optimizer=optimizer,
        step=args.steps,
        best_val_loss=best_val_loss,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        train_loss=last_train_loss,
        val_loss=last_val_loss,
    )
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
