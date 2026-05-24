"""OpenAI-compatible web chat interface for local checkpoints."""

from inference.openai_chat.client import OpenAIChatClient, OpenAIChatError

__all__ = ["OpenAIChatClient", "OpenAIChatError"]
