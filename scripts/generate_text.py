from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import TextGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained small model.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "runs" / "tiny_model")
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--prompt", default="Language models")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = TextGenerator.from_checkpoint(
        args.model_dir,
        checkpoint_name=args.checkpoint,
        device=args.device,
    )
    text = generator.generate_text(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        use_cache=not args.no_cache,
    )
    print(text)


if __name__ == "__main__":
    main()
