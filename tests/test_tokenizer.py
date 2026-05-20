from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bpe_tokenizer import Tokenizer, iter_text_files, read_text_corpus, train_bpe


class TokenizerTest(unittest.TestCase):
    def train_tokenizer(self) -> Tokenizer:
        corpus = (
            "hello hello world\n"
            "low lower lowest\n"
            "中文 mixed text 🙂\n"
            "<|endoftext|>"
        )
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "corpus.txt"
            input_path.write_text(corpus, encoding="utf-8")
            vocab, merges = train_bpe(input_path, 280, ["<|endoftext|>"])
        return Tokenizer(vocab, merges, ["<|endoftext|>"])

    def test_round_trip_multilingual_text(self) -> None:
        tokenizer = self.train_tokenizer()
        text = "hello 中文 🙂\nworld"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_special_token_is_single_token(self) -> None:
        tokenizer = self.train_tokenizer()
        ids = tokenizer.encode("a<|endoftext|>b")
        special_id = tokenizer.encode("<|endoftext|>")
        self.assertEqual(len(special_id), 1)
        self.assertIn(special_id[0], ids)

    def test_save_and_load_preserves_behavior(self) -> None:
        tokenizer = self.train_tokenizer()
        text = "low lower<|endoftext|>中文"

        with TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.json"
            merges_path = Path(tmpdir) / "merges.json"
            tokenizer.save(vocab_path, merges_path)
            loaded = Tokenizer.from_files(vocab_path, merges_path)

        self.assertEqual(loaded.encode(text), tokenizer.encode(text))
        self.assertEqual(loaded.decode(loaded.encode(text)), text)

    def test_training_is_deterministic(self) -> None:
        corpus = "aba aba abd\n"
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "corpus.txt"
            input_path.write_text(corpus, encoding="utf-8")
            first_vocab, first_merges = train_bpe(input_path, 270, [])
            second_vocab, second_merges = train_bpe(input_path, 270, [])

        self.assertEqual(first_vocab, second_vocab)
        self.assertEqual(first_merges, second_merges)

    def test_training_from_directory_reads_files_in_order(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "b.txt").write_text("banana banana", encoding="utf-8")
            (root / "a.txt").write_text("apple apple", encoding="utf-8")
            (root / ".hidden.txt").write_text("hidden", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "c.txt").write_text("citrus", encoding="utf-8")

            files = iter_text_files(root)
            corpus = read_text_corpus(root)
            vocab, merges = train_bpe(root, 270, [])

        self.assertEqual([path.name for path in files], ["a.txt", "b.txt", "c.txt"])
        self.assertEqual(corpus, "apple apple\nbanana banana\ncitrus")
        self.assertGreater(len(vocab), 256)
        self.assertGreater(len(merges), 0)

    def test_empty_text_round_trip(self) -> None:
        tokenizer = self.train_tokenizer()
        self.assertEqual(tokenizer.encode(""), [])
        self.assertEqual(tokenizer.decode([]), "")

    def test_vocab_size_must_include_byte_and_special_tokens(self) -> None:
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "corpus.txt"
            input_path.write_text("hello", encoding="utf-8")

            with self.assertRaises(ValueError):
                train_bpe(input_path, 256, ["<|endoftext|>"])

    def test_invalid_token_id_raises(self) -> None:
        tokenizer = self.train_tokenizer()
        with self.assertRaises(ValueError):
            tokenizer.decode([999_999])


if __name__ == "__main__":
    unittest.main()
