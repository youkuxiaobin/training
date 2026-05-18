from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from language_model.config import GPTConfig
from language_model.generation import generate
from language_model.gpt import GPTLanguageModel
from language_model.tokenization import encode_text, eos_token_id, load_language_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained small model.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "runs" / "tiny_model")
    parser.add_argument("--prompt", default="Language models")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
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
    device = resolve_device(args.device)

    checkpoint = torch.load(args.model_dir / "model.pt", map_location=device)
    special_tokens = checkpoint["special_tokens"]
    tokenizer = load_language_tokenizer(args.model_dir, special_tokens)

    cfg = GPTConfig.from_dict(checkpoint["config"])
    model = GPTLanguageModel(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])

    eos_id = eos_token_id(tokenizer, special_tokens[0]) if special_tokens else None
    prompt_ids = encode_text(tokenizer, args.prompt)
    if not prompt_ids:
        prompt_ids = [eos_id if eos_id is not None else 0]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=eos_id,
    )
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
