"""OpenAI-compatible chat service backed by the local language model."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference import TextGenerator
from language_model.tokenization import encode_text


DEFAULT_LOCAL_MODEL = "local-tiny-gpt"


class OpenAIChatError(RuntimeError):
    """Raised when a local OpenAI-compatible chat request is invalid."""


@dataclass
class OpenAIChatClient:
    generator: TextGenerator
    model_name: str = DEFAULT_LOCAL_MODEL

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str | Path,
        checkpoint_name: str = "best.pt",
        device: str = "auto",
        model_name: str = DEFAULT_LOCAL_MODEL,
    ) -> "OpenAIChatClient":
        generator = TextGenerator.from_checkpoint(
            model_dir,
            checkpoint_name=checkpoint_name,
            device=device,
        )
        return cls(generator=generator, model_name=model_name)

    def create_reply(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int | None = 40,
        use_cache: bool = True,
    ) -> str:
        prompt = messages_to_prompt(messages)
        return self.generator.generate_text(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            use_cache=use_cache,
            include_prompt=False,
        ).strip()

    def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            raise OpenAIChatError("messages must be a list")
        if payload.get("stream", False):
            raise OpenAIChatError("streaming responses are not supported yet")

        max_tokens = _positive_int(
            payload.get("max_tokens", payload.get("max_completion_tokens", 128)),
            "max_tokens",
        )
        temperature = float(payload.get("temperature", 0.8))
        top_k = payload.get("top_k", 40)
        if top_k is not None:
            top_k = _positive_int(top_k, "top_k")

        prompt = messages_to_prompt(messages)
        content = self.generator.generate_text(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            use_cache=True,
            include_prompt=False,
        ).strip()
        model_name = payload.get("model") or self.model_name
        usage = self._usage(prompt, content)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    def list_models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    def _usage(self, prompt: str, completion: str) -> dict[str, int]:
        prompt_tokens = len(encode_text(self.generator.tokenizer, prompt))
        completion_tokens = len(encode_text(self.generator.tokenizer, completion))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }


def messages_to_prompt(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role not in {"system", "user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        label = {"system": "System", "user": "User", "assistant": "Assistant"}[role]
        parts.append(f"{label}: {content.strip()}")
    if not parts:
        raise OpenAIChatError("messages must contain at least one non-empty message")
    parts.append("Assistant:")
    return "\n".join(parts)


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise OpenAIChatError(f"{name} must be positive")
    return parsed
