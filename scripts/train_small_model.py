from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from language_model.data import get_batch, make_token_tensor
from language_model.gpt import GPTLanguageModel
from language_model.config import GPTConfig
from language_model.tokenization import (
    DEFAULT_SPECIAL_TOKENS,
    append_eos_if_needed,
    encode_text,
    train_language_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small GPT-style model.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "examples" / "tiny_corpus.txt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "tiny_model")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=96)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--n-head", type=int, default=3)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--norm-type", choices=["rmsnorm", "layernorm"], default="rmsnorm")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    text = args.input.read_text(encoding="utf-8")
    training_text = append_eos_if_needed(text, DEFAULT_SPECIAL_TOKENS[0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = train_language_tokenizer(
        args.input,
        args.vocab_size,
        DEFAULT_SPECIAL_TOKENS,
    )
    tokenizer.save(args.output_dir / "vocab.json", args.output_dir / "merges.json")

    token_ids = make_token_tensor(encode_text(tokenizer, training_text))
    if token_ids.numel() <= args.context_length:
        raise ValueError("input text is too short for the requested context length")

    cfg = GPTConfig(
        vocab_size=len(tokenizer.vocab),
        context_length=args.context_length,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout_rate=args.dropout_rate,
        norm_type=args.norm_type,
    )
    device = resolve_device(args.device)
    model = GPTLanguageModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(
            token_ids,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
            generator=generator,
        )
        _, loss = model(x, y)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step == args.steps or step % max(args.steps // 10, 1) == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    checkpoint = {
        "config": cfg.to_dict(),
        "model_state": model.state_dict(),
        "special_tokens": DEFAULT_SPECIAL_TOKENS,
    }
    torch.save(checkpoint, args.output_dir / "model.pt")
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
