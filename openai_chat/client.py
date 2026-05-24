"""Small OpenAI Responses API client used by the web chat server."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "gpt-5.2"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
Transport = Callable[[Request, int], Any]


class OpenAIChatError(RuntimeError):
    """Raised when the OpenAI chat request cannot be completed."""


@dataclass
class OpenAIChatClient:
    api_key: str | None = None
    model: str | None = None
    base_url: str = DEFAULT_BASE_URL
    timeout: int = 60
    transport: Transport | None = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        self.model = self.model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.transport = self.transport or _urlopen_transport

    def create_reply(
        self,
        messages: list[dict[str, str]],
        instructions: str = "",
        max_output_tokens: int = 512,
    ) -> str:
        if not self.api_key:
            raise OpenAIChatError("OPENAI_API_KEY is not set")
        if max_output_tokens <= 0:
            raise OpenAIChatError("max_output_tokens must be positive")
        normalized_messages = _normalize_messages(messages)
        if not normalized_messages:
            raise OpenAIChatError("messages must contain at least one user message")

        payload: dict[str, Any] = {
            "model": self.model,
            "input": normalized_messages,
            "max_output_tokens": max_output_tokens,
        }
        if instructions.strip():
            payload["instructions"] = instructions.strip()

        response_payload = self._post_json("/responses", payload)
        return _extract_output_text(response_payload)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            assert self.transport is not None
            with self.transport(request, self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise OpenAIChatError(f"OpenAI request failed: {exc.code} {details}") from exc
        except URLError as exc:
            raise OpenAIChatError(f"OpenAI request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise OpenAIChatError("OpenAI returned invalid JSON") from exc


def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        normalized.append({"role": role, "content": content.strip()})
    return normalized


def _urlopen_transport(request: Request, timeout: int) -> Any:
    return urlopen(request, timeout=timeout)


def _extract_output_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    text_parts = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                text_parts.append(text)

    if text_parts:
        return "\n".join(text_parts)
    raise OpenAIChatError("OpenAI response did not contain text output")
