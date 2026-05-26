import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from finetune.sft.data import (
    IGNORE_INDEX,
    SFTExample,
    SFTSample,
    build_sft_sample,
    collate_sft_batch,
    load_sft_examples,
)
from finetune.sft.train import run_sft
from language_model.config import GPTConfig
from language_model.gpt import GPTLanguageModel
from language_model.tokenization import DEFAULT_SPECIAL_TOKENS, train_language_tokenizer
from language_model.training import save_checkpoint


class SFTDataTest(unittest.TestCase):
    def test_load_sft_examples_supports_messages_and_instruction_records(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "train.jsonl"
            records = [
                {
                    "messages": [
                        {"role": "system", "content": "be concise"},
                        {"role": "user", "content": "who are you?"},
                        {"role": "assistant", "content": "I am a local assistant."},
                    ]
                },
                {
                    "instruction": "Translate this",
                    "input": "hello",
                    "response": "你好",
                },
            ]
            data_path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )

            examples = load_sft_examples(data_path)

        self.assertEqual(len(examples), 2)
        self.assertEqual(
            examples[0].prompt,
            "System: be concise\nUser: who are you?\nAssistant:",
        )
        self.assertEqual(examples[0].response, "I am a local assistant.")
        self.assertEqual(
            examples[1].prompt,
            "User: Translate this\n\nhello\nAssistant:",
        )
        self.assertEqual(examples[1].response, "你好")

    def test_build_sft_sample_masks_prompt_tokens(self) -> None:
        with TemporaryDirectory() as tmpdir:
            corpus_path = Path(tmpdir) / "corpus.txt"
            corpus_path.write_text(
                "User: hello\nAssistant: hi<|endoftext|>",
                encoding="utf-8",
            )
            tokenizer = train_language_tokenizer(corpus_path, 280, log_every=0)

            example = SFTExample(prompt="User: hello\nAssistant:", response="hi")
            sample = build_sft_sample(
                tokenizer,
                example,
                context_length=32,
                eos_token=DEFAULT_SPECIAL_TOKENS[0],
            )

        prompt_len = len(tokenizer.encode(example.prompt))
        self.assertTrue(all(label == IGNORE_INDEX for label in sample.labels[: prompt_len - 1]))
        self.assertNotEqual(sample.labels[prompt_len - 1], IGNORE_INDEX)
        trained_target_ids = [label for label in sample.labels if label != IGNORE_INDEX]
        trained_target_text = tokenizer.decode(trained_target_ids)
        self.assertEqual(trained_target_text, " hi<|endoftext|>")

    def test_collate_sft_batch_pads_inputs_and_ignores_padding_labels(self) -> None:
        x, y = collate_sft_batch(
            [
                SFTSample(input_ids=[1, 2], labels=[IGNORE_INDEX, 3]),
                SFTSample(input_ids=[4], labels=[5]),
            ],
            pad_id=0,
            device="cpu",
        )

        self.assertEqual(x.tolist(), [[1, 2], [4, 0]])
        self.assertEqual(y.tolist(), [[IGNORE_INDEX, 3], [5, IGNORE_INDEX]])


class SFTTrainingTest(unittest.TestCase):
    def test_run_sft_smoke_trains_and_writes_checkpoint(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "base"
            output_dir = root / "sft"
            model_dir.mkdir()
            corpus_path = root / "tokenizer_corpus.txt"
            corpus_path.write_text(
                "User: who are you?\nAssistant: I am a local assistant.<|endoftext|>\n"
                "User: say hello\nAssistant: hello<|endoftext|>",
                encoding="utf-8",
            )
            tokenizer = train_language_tokenizer(corpus_path, 300, log_every=0)
            tokenizer.save(model_dir / "vocab.json", model_dir / "merges.json")

            cfg = GPTConfig(
                vocab_size=len(tokenizer.vocab),
                context_length=32,
                n_embd=16,
                n_layer=1,
                n_head=4,
                dropout_rate=0.0,
            )
            model = GPTLanguageModel(cfg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            save_checkpoint(
                model_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                step=1,
                best_val_loss=1.0,
                special_tokens=DEFAULT_SPECIAL_TOKENS,
            )

            train_path = root / "train.jsonl"
            train_records = [
                {
                    "messages": [
                        {"role": "user", "content": "who are you?"},
                        {"role": "assistant", "content": "I am a local assistant."},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "say hello"},
                        {"role": "assistant", "content": "hello"},
                    ]
                },
            ]
            train_path.write_text(
                "\n".join(json.dumps(record) for record in train_records),
                encoding="utf-8",
            )

            run_sft(
                Namespace(
                    model_dir=model_dir,
                    checkpoint="best.pt",
                    train_input=train_path,
                    valid_input=None,
                    output_dir=output_dir,
                    max_samples=None,
                    batch_size=2,
                    steps=2,
                    lr=1e-4,
                    min_lr=1e-5,
                    warmup_steps=1,
                    grad_clip=1.0,
                    val_fraction=0.5,
                    eval_interval=1,
                    eval_iters=1,
                    train_on_prompt=False,
                    gradient_checkpointing=False,
                    seed=123,
                    device="cpu",
                )
            )

            self.assertTrue((output_dir / "best.pt").is_file())
            self.assertTrue((output_dir / "model.pt").is_file())
            self.assertTrue((output_dir / "vocab.json").is_file())
            self.assertTrue((output_dir / "merges.json").is_file())


if __name__ == "__main__":
    unittest.main()
