"""Load trained checkpoints and generate text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from bpe_tokenizer import Tokenizer
from language_model.config import GPTConfig
from language_model.generation import generate
from language_model.gpt import GPTLanguageModel
from language_model.tokenization import encode_text, eos_token_id, load_language_tokenizer


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TextGenerator:
    model: GPTLanguageModel
    tokenizer: Tokenizer
    special_tokens: list[str]
    device: torch.device

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str | Path,
        checkpoint_name: str = "model.pt",
        device: str | torch.device = "auto",
    ) -> "TextGenerator":
        resolved_device = resolve_device(device) if isinstance(device, str) else device
        resolved_model_dir = Path(model_dir)
        checkpoint_path = resolved_model_dir / checkpoint_name
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
        special_tokens = list(checkpoint.get("special_tokens", []))
        tokenizer = load_language_tokenizer(resolved_model_dir, special_tokens)

        cfg = GPTConfig.from_dict(checkpoint["config"])
        model = GPTLanguageModel(cfg).to(resolved_device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        return cls(
            model=model,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            device=resolved_device,
        )

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int | None = 40,
        use_cache: bool = True,
    ) -> str:
        eos_id = self._eos_id()
        prompt_ids = encode_text(self.tokenizer, prompt)
        if not prompt_ids:
            prompt_ids = [eos_id if eos_id is not None else 0]

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        output_ids = generate(
            self.model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id,
            use_cache=use_cache,
        )
        return self.tokenizer.decode(output_ids[0].tolist())

    def _eos_id(self) -> int | None:
        if not self.special_tokens:
            return None
        return eos_token_id(self.tokenizer, self.special_tokens[0])
