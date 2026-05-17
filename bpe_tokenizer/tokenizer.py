"""A small byte-level BPE tokenizer."""

from __future__ import annotations

import base64
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import regex


GPT2_PRETOKENIZER_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

Token = bytes
Pair = tuple[Token, Token]
Word = tuple[Token, ...]


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: Sequence[str] | None = None,
) -> tuple[dict[int, bytes], list[Pair]]:
    """Train a byte-level BPE vocabulary from a UTF-8 text file.

    Returns:
        A pair of (vocab, merges). The vocab maps token ids to raw bytes, and
        merges contains ordered token-byte pairs to merge during encoding.
    """

    special_tokens = _unique(special_tokens or [])
    min_vocab_size = 256 + len(special_tokens)
    if vocab_size < min_vocab_size:
        raise ValueError(
            f"vocab_size must be at least {min_vocab_size} for byte tokens and specials"
        )

    text = Path(input_path).read_text(encoding="utf-8")
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for index, special_token in enumerate(special_tokens, start=256):
        vocab[index] = special_token.encode("utf-8")

    next_token_id = len(vocab)
    word_counts = _pretoken_counts(text, special_tokens)
    merges: list[Pair] = []

    while next_token_id < vocab_size:
        pair_counts = _pair_counts(word_counts)
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merged_token = best_pair[0] + best_pair[1]

        vocab[next_token_id] = merged_token
        merges.append(best_pair)
        word_counts = _merge_word_counts(word_counts, best_pair, merged_token)
        next_token_id += 1

    return vocab, merges


class Tokenizer:
    """Byte-level BPE tokenizer with special-token support."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: Sequence[Pair],
        special_tokens: Sequence[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = _unique(special_tokens or [])
        self._token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        missing_specials = [
            token
            for token in self.special_tokens
            if token.encode("utf-8") not in self._token_to_id
        ]
        if missing_specials:
            raise ValueError(f"special tokens missing from vocab: {missing_specials}")

        self._special_to_id = {
            token: self._token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
        }
        self._special_pattern = _compile_special_pattern(self.special_tokens)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: Sequence[str] | None = None,
    ) -> "Tokenizer":
        vocab_payload = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
        merges_payload = json.loads(Path(merges_filepath).read_text(encoding="utf-8"))

        vocab = {
            int(item["id"]): _decode_token(item["token"])
            for item in vocab_payload["vocab"]
        }
        merges = [
            (_decode_token(left), _decode_token(right))
            for left, right in merges_payload["merges"]
        ]

        stored_special_tokens = vocab_payload.get("special_tokens", [])
        resolved_special_tokens = (
            stored_special_tokens if special_tokens is None else special_tokens
        )
        return cls(vocab, merges, resolved_special_tokens)

    def save(self, vocab_filepath: str | Path, merges_filepath: str | Path) -> None:
        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        merges_path.parent.mkdir(parents=True, exist_ok=True)

        vocab_payload = {
            "version": 1,
            "special_tokens": self.special_tokens,
            "vocab": [
                {"id": token_id, "token": _encode_token(token)}
                for token_id, token in sorted(self.vocab.items())
            ],
        }
        merges_payload = {
            "version": 1,
            "merges": [
                [_encode_token(left), _encode_token(right)] for left, right in self.merges
            ],
        }

        vocab_path.write_text(
            json.dumps(vocab_payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        merges_path.write_text(
            json.dumps(merges_payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for part in self._split_special_tokens(text):
            if part in self._special_to_id:
                token_ids.append(self._special_to_id[part])
            else:
                token_ids.extend(self._encode_plain_text(part))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        tokens: list[bytes] = []
        for token_id in token_ids:
            if token_id not in self.vocab:
                raise ValueError(f"unknown token id: {token_id}")
            tokens.append(self.vocab[token_id])
        return b"".join(tokens).decode("utf-8", errors="replace")

    def _split_special_tokens(self, text: str) -> list[str]:
        if self._special_pattern is None:
            return [text]
        return [part for part in self._special_pattern.split(text) if part]

    def _encode_plain_text(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for pretoken in regex.findall(GPT2_PRETOKENIZER_PATTERN, text):
            token_ids.extend(self._encode_pretoken(pretoken.encode("utf-8")))
        return token_ids

    @lru_cache(maxsize=100_000)
    def _encode_pretoken(self, pretoken_bytes: bytes) -> tuple[int, ...]:
        pieces: Word = tuple(bytes([byte]) for byte in pretoken_bytes)

        while len(pieces) >= 2:
            best_pair: Pair | None = None
            best_rank: int | None = None
            for pair in zip(pieces, pieces[1:]):
                rank = self._merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            pieces = _merge_word(pieces, best_pair, best_pair[0] + best_pair[1])

        return tuple(self._token_to_id[piece] for piece in pieces)


def _pretoken_counts(text: str, special_tokens: Sequence[str]) -> Counter[Word]:
    special_pattern = _compile_special_pattern(special_tokens)
    chunks = special_pattern.split(text) if special_pattern else [text]

    counts: Counter[Word] = Counter()
    for chunk in chunks:
        if not chunk or chunk in special_tokens:
            continue
        for pretoken in regex.findall(GPT2_PRETOKENIZER_PATTERN, chunk):
            word = tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))
            if word:
                counts[word] += 1
    return counts


def _pair_counts(word_counts: Counter[Word]) -> Counter[Pair]:
    counts: Counter[Pair] = Counter()
    for word, frequency in word_counts.items():
        for pair in zip(word, word[1:]):
            counts[pair] += frequency
    return counts


def _merge_word_counts(
    word_counts: Counter[Word],
    pair: Pair,
    merged_token: Token,
) -> Counter[Word]:
    merged_counts: Counter[Word] = Counter()
    for word, frequency in word_counts.items():
        merged_counts[_merge_word(word, pair, merged_token)] += frequency
    return merged_counts


def _merge_word(word: Word, pair: Pair, merged_token: Token) -> Word:
    merged: list[Token] = []
    index = 0
    while index < len(word):
        if index + 1 < len(word) and word[index] == pair[0] and word[index + 1] == pair[1]:
            merged.append(merged_token)
            index += 2
        else:
            merged.append(word[index])
            index += 1
    return tuple(merged)


def _compile_special_pattern(special_tokens: Sequence[str]) -> regex.Pattern[str] | None:
    if not special_tokens:
        return None
    escaped = [regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    return regex.compile("(" + "|".join(escaped) + ")")


def _unique(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _encode_token(token: bytes) -> str:
    return base64.b64encode(token).decode("ascii")


def _decode_token(token: str) -> bytes:
    return base64.b64decode(token.encode("ascii"))
