# Language model modules

`language_model` is split by training and model-building steps:

- `tokenization.py`: tokenizer training, text preparation, token encode helpers.
- `data.py`: token tensor creation and training batch sampling.
- `training.py`: validation loss, learning-rate schedule, checkpoint saving.
- `config.py`: model configuration and validation.
- `norms.py`: RMSNorm by default, with LayerNorm still available.
- `rope.py`: RoPE cache creation and application.
- `attention.py`: causal self-attention with RoPE.
- `feed_forward.py`: SwiGLU by default, with GELU still available.
- `blocks.py`: Transformer block composition.
- `gpt.py`: full GPT-style language model.
- `generation.py`: autoregressive token generation with KV cache.
- `model.py`: compatibility imports for older code.
