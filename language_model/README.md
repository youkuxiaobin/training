# Language model modules

`language_model` is split by training and model-building steps:

- `tokenization.py`: tokenizer training, text preparation, token encode helpers.
- `data.py`: token tensor creation and training batch sampling.
- `config.py`: model configuration and validation.
- `norms.py`: RMSNorm by default, with LayerNorm still available.
- `rope.py`: RoPE cache creation and application.
- `attention.py`: causal self-attention with RoPE.
- `feed_forward.py`: feed-forward network inside each block.
- `blocks.py`: Transformer block composition.
- `gpt.py`: full GPT-style language model.
- `generation.py`: autoregressive token generation.
- `model.py`: compatibility imports for older code.
