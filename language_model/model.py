"""Compatibility imports for the language model package."""

from language_model.config import GPTConfig
from language_model.generation import generate
from language_model.gpt import GPTLanguageModel

__all__ = ["GPTConfig", "GPTLanguageModel", "generate"]
