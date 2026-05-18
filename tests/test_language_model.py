import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from language_model.data import get_batch, make_token_tensor
from language_model.generation import generate
from language_model.gpt import GPTLanguageModel
from language_model.config import GPTConfig
from language_model.norms import RMSNorm
from language_model.rope import apply_rope, build_rope_cache
from language_model.tokenization import (
    append_eos_if_needed,
    encode_text,
    train_language_tokenizer,
)


class LanguageModelTest(unittest.TestCase):
    def test_forward_returns_logits_and_loss(self) -> None:
        cfg = GPTConfig(
            vocab_size=32,
            context_length=8,
            n_embd=16,
            n_layer=2,
            n_head=4,
            dropout_rate=0.0,
        )
        model = GPTLanguageModel(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
        y = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))

        logits, loss = model(x, y)

        self.assertEqual(logits.shape, (2, cfg.context_length, cfg.vocab_size))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(hasattr(model, "pos_emb"))
        self.assertTrue(hasattr(model.blocks[0].attn, "rope_cos"))
        self.assertTrue(hasattr(model.blocks[0].attn, "rope_sin"))
        self.assertIsInstance(model.blocks[0].norm1, RMSNorm)
        self.assertIsInstance(model.final_norm, RMSNorm)

    def test_generate_extends_prompt(self) -> None:
        torch.manual_seed(123)
        cfg = GPTConfig(
            vocab_size=20,
            context_length=6,
            n_embd=12,
            n_layer=1,
            n_head=3,
            dropout_rate=0.0,
        )
        model = GPTLanguageModel(cfg)
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

        output = generate(model, prompt, max_new_tokens=4, temperature=0)

        self.assertEqual(output.shape, (1, 7))
        self.assertTrue(torch.equal(output[:, :3], prompt))

    def test_get_batch_shapes(self) -> None:
        token_ids = make_token_tensor(list(range(20)))
        x, y = get_batch(
            token_ids,
            batch_size=3,
            context_length=5,
            device="cpu",
            generator=torch.Generator().manual_seed(123),
        )

        self.assertEqual(x.shape, (3, 5))
        self.assertEqual(y.shape, (3, 5))
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))

    def test_config_rejects_invalid_head_count(self) -> None:
        with self.assertRaises(ValueError):
            GPTConfig(vocab_size=10, n_embd=10, n_head=3)

    def test_config_rejects_odd_rope_head_dim(self) -> None:
        with self.assertRaises(ValueError):
            GPTConfig(vocab_size=10, n_embd=10, n_head=2)

    def test_config_rejects_unknown_norm(self) -> None:
        with self.assertRaises(ValueError):
            GPTConfig(vocab_size=10, norm_type="batchnorm")

    def test_layernorm_remains_available(self) -> None:
        cfg = GPTConfig(
            vocab_size=16,
            context_length=4,
            n_embd=8,
            n_layer=1,
            n_head=2,
            norm_type="layernorm",
        )
        model = GPTLanguageModel(cfg)

        self.assertIsInstance(model.blocks[0].norm1, nn.LayerNorm)
        self.assertIsInstance(model.final_norm, nn.LayerNorm)

    def test_rmsnorm_preserves_shape_and_is_finite(self) -> None:
        norm = RMSNorm(6)
        x = torch.randn(2, 3, 6)

        output = norm(x)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_rope_preserves_shape(self) -> None:
        cos, sin = build_rope_cache(context_length=5, head_dim=4, theta=10_000.0)
        x = torch.randn(2, 3, 5, 4)

        rotated = apply_rope(x, cos, sin)

        self.assertEqual(rotated.shape, x.shape)

    def test_tokenization_helpers_train_and_encode(self) -> None:
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "corpus.txt"
            input_path.write_text("hello hello world", encoding="utf-8")
            tokenizer = train_language_tokenizer(input_path, 270)

        text = append_eos_if_needed("hello")
        token_ids = encode_text(tokenizer, text)

        self.assertGreater(len(token_ids), 0)
        self.assertEqual(tokenizer.decode(token_ids), "hello<|endoftext|>")


if __name__ == "__main__":
    unittest.main()
