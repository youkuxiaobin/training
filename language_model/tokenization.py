"""Tokenizer setup and token encoding for language-model training."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from bpe_tokenizer import Tokenizer, iter_text_files, train_bpe


DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]


def train_language_tokenizer(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: Sequence[str] | None = None,
) -> Tokenizer:
    resolved_special_tokens = list(special_tokens or DEFAULT_SPECIAL_TOKENS)
    vocab, merges = train_bpe(input_path, vocab_size, resolved_special_tokens)
    return Tokenizer(vocab, merges, resolved_special_tokens)


def load_language_tokenizer(
    model_dir: str | Path,
    special_tokens: Sequence[str] | None = None,
) -> Tokenizer:
    model_path = Path(model_dir)
    return Tokenizer.from_files(
        model_path / "vocab.json",
        model_path / "merges.json",
        special_tokens,
    )


def append_eos_if_needed(text: str, eos_token: str = DEFAULT_SPECIAL_TOKENS[0]) -> str:
    return text if text.endswith(eos_token) else text + eos_token


def read_training_text(
    input_path: str | Path,
    eos_token: str = DEFAULT_SPECIAL_TOKENS[0],
) -> str:
    texts = [
        append_eos_if_needed(file_path.read_text(encoding="utf-8"), eos_token)
        for file_path in iter_text_files(input_path)
    ]
    return "\n".join(texts)


def encode_text(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text)


def eos_token_id(tokenizer: Tokenizer, eos_token: str = DEFAULT_SPECIAL_TOKENS[0]) -> int:
    return tokenizer.encode(eos_token)[0]
