"""Parameter-efficient SFT methods."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from language_model.gpt import GPTLanguageModel

DEFAULT_LORA_TARGETS = "qkv,out_proj,gate_proj,up_proj,down_proj,net.0,net.2"


@dataclass(frozen=True)
class FinetuneMethodConfig:
    method: str = "full"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_targets: str = DEFAULT_LORA_TARGETS
    freeze_last_layers: int = 0
    freeze_train_embeddings: bool = False
    adapter_train_head: bool = False
    adapter_train_norms: bool = False


class LoRALinear(nn.Module):
    """Linear layer with a trainable low-rank adapter."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(
            torch.empty(
                rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "LoRALinear":
        return cls(linear, rank=rank, alpha=alpha, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        adapter_output = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        return base_output + adapter_output * self.scaling

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + (self.lora_b @ self.lora_a) * self.scaling

    def merged_bias(self) -> torch.Tensor | None:
        return self.base.bias


class QuantizedLoRALinear(nn.Module):
    """Frozen 4-bit row-wise quantized linear layer with a LoRA adapter."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        num_bits: int = 4,
    ) -> None:
        super().__init__()
        if num_bits != 4:
            raise ValueError("only 4-bit quantization is supported")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        qweight, scale = quantize_weight_4bit(linear.weight.detach())
        self.register_buffer("qweight", qweight, persistent=True)
        self.register_buffer("scale", scale, persistent=True)
        if linear.bias is None:
            self.register_buffer("base_bias", None, persistent=True)
        else:
            self.register_buffer("base_bias", linear.bias.detach().clone(), persistent=True)

        self.lora_a = nn.Parameter(
            torch.empty(
                rank,
                self.in_features,
                device=linear.weight.device,
                dtype=linear.weight.dtype,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                self.out_features,
                rank,
                device=linear.weight.device,
                dtype=linear.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "QuantizedLoRALinear":
        return cls(linear, rank=rank, alpha=alpha, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_weight = self.dequantized_weight().to(dtype=x.dtype)
        base_bias = None if self.base_bias is None else self.base_bias.to(dtype=x.dtype)
        base_output = F.linear(x, base_weight, base_bias)
        adapter_output = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        return base_output + adapter_output * self.scaling

    def dequantized_weight(self) -> torch.Tensor:
        return self.qweight.to(dtype=self.scale.dtype) * self.scale

    def merged_weight(self) -> torch.Tensor:
        return self.dequantized_weight() + (self.lora_b @ self.lora_a) * self.scaling

    def merged_bias(self) -> torch.Tensor | None:
        return self.base_bias


def quantize_weight_4bit(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 7
    max_abs = weight.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = max_abs / qmax
    qweight = torch.round(weight.detach() / scale).clamp(-qmax, qmax).to(torch.int8)
    return qweight, scale


def apply_finetune_method(
    model: GPTLanguageModel,
    config: FinetuneMethodConfig,
) -> GPTLanguageModel:
    method = config.method.lower()
    if method == "full":
        set_requires_grad(model, True)
    elif method == "freeze":
        apply_freeze(
            model,
            last_layers=config.freeze_last_layers,
            train_embeddings=config.freeze_train_embeddings,
        )
    elif method == "lora":
        apply_lora(model, config, quantized=False)
    elif method == "qlora":
        apply_lora(model, config, quantized=True)
    else:
        raise ValueError(f"unsupported finetune method: {config.method}")
    return model


def apply_freeze(
    model: GPTLanguageModel,
    last_layers: int = 0,
    train_embeddings: bool = False,
) -> None:
    if last_layers < 0:
        raise ValueError("last_layers must be non-negative")
    set_requires_grad(model, False)
    if train_embeddings:
        set_requires_grad(model.token_emb, True)
    set_requires_grad(model.final_norm, True)
    set_requires_grad(model.out_head, True)
    if last_layers:
        for block in model.blocks[-last_layers:]:
            set_requires_grad(block, True)


def apply_lora(
    model: GPTLanguageModel,
    config: FinetuneMethodConfig,
    quantized: bool,
) -> None:
    targets = parse_target_modules(config.lora_targets)
    set_requires_grad(model, False)
    wrapper_cls = QuantizedLoRALinear if quantized else LoRALinear
    replaced = replace_linear_modules(
        model,
        targets=targets,
        wrapper_cls=wrapper_cls,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )
    if replaced == 0:
        raise ValueError(f"no Linear modules matched LoRA targets: {sorted(targets)}")
    if config.adapter_train_head:
        set_requires_grad(model.out_head, True)
    if config.adapter_train_norms:
        for name, module in model.named_modules():
            if "norm" in name:
                set_requires_grad(module, True)


def replace_linear_modules(
    module: nn.Module,
    targets: set[str],
    wrapper_cls: type[LoRALinear] | type[QuantizedLoRALinear],
    rank: int,
    alpha: float,
    dropout: float,
    prefix: str = "",
) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        qualified_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and should_replace_linear(qualified_name, targets):
            setattr(
                module,
                child_name,
                wrapper_cls.from_linear(child, rank=rank, alpha=alpha, dropout=dropout),
            )
            replaced += 1
        else:
            replaced += replace_linear_modules(
                child,
                targets=targets,
                wrapper_cls=wrapper_cls,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                prefix=qualified_name,
            )
    return replaced


def should_replace_linear(module_name: str, targets: set[str]) -> bool:
    return any(module_name == target or module_name.endswith(f".{target}") for target in targets)


def parse_target_modules(targets: str) -> set[str]:
    parsed = {target.strip() for target in targets.split(",") if target.strip()}
    if not parsed:
        raise ValueError("at least one LoRA target module is required")
    return parsed


def export_inference_model(model: GPTLanguageModel) -> GPTLanguageModel:
    clean = GPTLanguageModel(model.cfg).to(next(model.parameters()).device)
    clean_state = clean.state_dict()
    current_state = model.state_dict()
    for key in list(clean_state):
        if key in current_state:
            clean_state[key] = current_state[key].detach().clone()

    for name, module in clean.named_modules():
        source = model.get_submodule(name) if name else model
        if isinstance(source, (LoRALinear, QuantizedLoRALinear)):
            clean_state[f"{name}.weight"] = source.merged_weight().detach().clone()
            bias = source.merged_bias()
            if bias is not None:
                clean_state[f"{name}.bias"] = bias.detach().clone()

    clean.load_state_dict(clean_state)
    clean.train(model.training)
    return clean


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled
