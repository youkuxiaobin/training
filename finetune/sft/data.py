"""Data preparation for supervised fine-tuning."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from bpe_tokenizer import Tokenizer, iter_text_files

IGNORE_INDEX = -100
SUPPORTED_ROLES = {"system", "user", "assistant"}
ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


@dataclass(frozen=True)
class SFTExample:
    prompt: str
    response: str


@dataclass(frozen=True)
class SFTSample:
    input_ids: list[int]
    labels: list[int]


def load_sft_examples(input_path: str | Path, max_samples: int | None = None) -> list[SFTExample]:
    """Load SFT examples from a JSON/JSONL file or a directory of those files."""

    examples: list[SFTExample] = []
    for file_path in _iter_data_files(input_path):
        for record in _read_records(file_path):
            examples.extend(examples_from_record(record))
            if max_samples is not None and len(examples) >= max_samples:
                return examples[:max_samples]
    if not examples:
        raise ValueError(f"no SFT examples found in {input_path}")
    return examples


def examples_from_record(record: dict[str, Any]) -> list[SFTExample]:
    """Convert one common SFT record shape into prompt/response examples."""

    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list):
        return examples_from_messages(messages)

    response = _first_string(record, "response", "output", "answer", "completion")
    instruction = _first_string(record, "instruction", "question")
    if instruction is not None and response is not None:
        context = _first_string(record, "input", "context")
        user_content = instruction.strip()
        if context and context.strip():
            user_content = f"{user_content}\n\n{context.strip()}"
        return [SFTExample(prompt=f"User: {user_content}\nAssistant:", response=response.strip())]

    prompt = _first_string(record, "prompt")
    if prompt is not None and response is not None:
        return [SFTExample(prompt=normalize_prompt(prompt), response=response.strip())]

    return []


def examples_from_messages(messages: Iterable[dict[str, Any]]) -> list[SFTExample]:
    """Build one training example for each assistant turn in a conversation."""

    history: list[dict[str, str]] = []
    examples: list[SFTExample] = []
    for message in messages:
        role = message.get("role") or message.get("from")
        content = message.get("content") or message.get("value")
        role = _normalize_role(role)
        if role not in SUPPORTED_ROLES or not isinstance(content, str) or not content.strip():
            continue

        normalized = {"role": role, "content": content.strip()}
        if role == "assistant":
            prompt = format_messages(history + [{"role": "assistant", "content": ""}])
            response = normalized["content"]
            if prompt and response:
                examples.append(SFTExample(prompt=prompt, response=response))
        history.append(normalized)
    return examples


def normalize_prompt(prompt: str) -> str:
    text = prompt.strip()
    if not text:
        raise ValueError("prompt must not be empty")
    has_role_label = any(label in text for label in ("System:", "User:", "Assistant:"))
    if has_role_label:
        return text if text.endswith("Assistant:") else f"{text}\nAssistant:"
    return f"User: {text}\nAssistant:"


def format_messages(messages: Iterable[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message["role"]
        if role not in SUPPORTED_ROLES:
            continue
        label = ROLE_LABELS[role]
        content = message["content"].strip()
        parts.append(f"{label}:" if role == "assistant" and not content else f"{label}: {content}")
    return "\n".join(parts)


def build_sft_sample(
    tokenizer: Tokenizer,
    example: SFTExample,
    context_length: int,
    eos_token: str,
    train_on_prompt: bool = False,
) -> SFTSample:
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    prompt_ids = tokenizer.encode(example.prompt.rstrip())
    response_text = f" {example.response.strip()}{eos_token}"
    response_ids = tokenizer.encode(response_text)
    token_ids = prompt_ids + response_ids
    target_mask = ([train_on_prompt] * len(prompt_ids)) + ([True] * len(response_ids))

    max_tokens = context_length + 1
    if len(token_ids) > max_tokens:
        token_ids = token_ids[-max_tokens:]
        target_mask = target_mask[-max_tokens:]

    if len(token_ids) < 2:
        raise ValueError("SFT sample must contain at least two tokens")

    input_ids = token_ids[:-1]
    shifted_labels = token_ids[1:]
    shifted_mask = target_mask[1:]
    labels = [
        label if should_train else IGNORE_INDEX
        for label, should_train in zip(shifted_labels, shifted_mask)
    ]
    if all(label == IGNORE_INDEX for label in labels):
        raise ValueError("SFT sample has no assistant tokens to train on")
    return SFTSample(input_ids=input_ids, labels=labels)


def build_sft_samples(
    tokenizer: Tokenizer,
    examples: Iterable[SFTExample],
    context_length: int,
    eos_token: str,
    train_on_prompt: bool = False,
) -> list[SFTSample]:
    samples = []
    for example in examples:
        try:
            samples.append(
                build_sft_sample(
                    tokenizer,
                    example,
                    context_length=context_length,
                    eos_token=eos_token,
                    train_on_prompt=train_on_prompt,
                )
            )
        except ValueError:
            continue
    if not samples:
        raise ValueError("no usable SFT samples after tokenization")
    return samples


def collate_sft_batch(
    samples: list[SFTSample],
    pad_id: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not samples:
        raise ValueError("samples must not be empty")
    max_len = max(len(sample.input_ids) for sample in samples)
    input_rows = []
    label_rows = []
    for sample in samples:
        pad_count = max_len - len(sample.input_ids)
        input_rows.append(sample.input_ids + [pad_id] * pad_count)
        label_rows.append(sample.labels + [IGNORE_INDEX] * pad_count)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(label_rows, dtype=torch.long, device=device),
    )


def sample_sft_batch(
    samples: list[SFTSample],
    batch_size: int,
    pad_id: int,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = torch.randint(0, len(samples), (batch_size,), generator=generator)
    batch = [samples[index.item()] for index in indices]
    return collate_sft_batch(batch, pad_id=pad_id, device=device)


def split_sft_examples(
    examples: list[SFTExample],
    val_fraction: float,
    seed: int,
) -> tuple[list[SFTExample], list[SFTExample]]:
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    if not examples:
        raise ValueError("examples must not be empty")
    if val_fraction == 0 or len(examples) < 2:
        return examples, examples

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_fraction))
    if len(shuffled) - val_count < 1:
        return examples, examples
    return shuffled[val_count:], shuffled[:val_count]


def _iter_data_files(input_path: str | Path) -> list[Path]:
    files = [
        file_path
        for file_path in iter_text_files(input_path)
        if file_path.suffix.lower() in {".json", ".jsonl"}
    ]
    if not files:
        raise ValueError(f"no .json or .jsonl files found in {input_path}")
    return files


def _read_records(file_path: Path) -> Iterable[dict[str, Any]]:
    if file_path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{file_path}:{line_number} must contain a JSON object")
            yield record
        return

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("examples", [payload]))
    if not isinstance(payload, list):
        raise ValueError(f"{file_path} must contain a JSON object or list")
    for index, record in enumerate(payload, 1):
        if not isinstance(record, dict):
            raise ValueError(f"{file_path}:{index} must contain a JSON object")
        yield record


def _first_string(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _normalize_role(role: Any) -> str:
    if not isinstance(role, str):
        return ""
    normalized = role.lower().strip()
    return {
        "human": "user",
        "prompter": "user",
        "gpt": "assistant",
        "bot": "assistant",
        "assistant": "assistant",
        "system": "system",
        "user": "user",
    }.get(normalized, normalized)
