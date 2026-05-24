import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from inference import TextGenerator
from language_model.config import GPTConfig
from language_model.gpt import GPTLanguageModel
from language_model.tokenization import DEFAULT_SPECIAL_TOKENS, train_language_tokenizer
from language_model.training import save_checkpoint


class InferenceTest(unittest.TestCase):
    def test_text_generator_loads_checkpoint_and_generates(self) -> None:
        with TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            corpus_path = model_dir / "corpus.txt"
            corpus_path.write_text("hello world. hello tiny model.", encoding="utf-8")

            tokenizer = train_language_tokenizer(corpus_path, 270, log_every=0)
            tokenizer.save(model_dir / "vocab.json", model_dir / "merges.json")

            cfg = GPTConfig(
                vocab_size=len(tokenizer.vocab),
                context_length=8,
                n_embd=16,
                n_layer=1,
                n_head=4,
                dropout_rate=0.0,
            )
            model = GPTLanguageModel(cfg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            save_checkpoint(
                model_dir / "model.pt",
                model=model,
                optimizer=optimizer,
                step=1,
                best_val_loss=1.0,
                special_tokens=DEFAULT_SPECIAL_TOKENS,
            )

            generator = TextGenerator.from_checkpoint(model_dir, device="cpu")
            text = generator.generate_text(
                "hello",
                max_new_tokens=2,
                temperature=0,
                use_cache=True,
            )

        self.assertIsInstance(text, str)
        self.assertTrue(text.startswith("hello"))


if __name__ == "__main__":
    unittest.main()
