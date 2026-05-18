"""Configuration for the small GPT-style language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    context_length: int = 128
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 4
    dropout_rate: float = 0.1
    qkv_bias: bool = False
    rope_theta: float = 10_000.0
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    ffn_type: str = "swiglu"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("per-head dimension must be even for RoPE")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.norm_type not in {"rmsnorm", "layernorm"}:
            raise ValueError("norm_type must be 'rmsnorm' or 'layernorm'")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if self.ffn_type not in {"swiglu", "gelu"}:
            raise ValueError("ffn_type must be 'swiglu' or 'gelu'")

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, int | float | bool | str]) -> "GPTConfig":
        return cls(**payload)
