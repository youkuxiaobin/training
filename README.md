# BPE Tokenizer

Small byte-level BPE tokenizer for pretraining experiments.

## What it does

- Trains a byte-level BPE vocabulary from UTF-8 text.
- Encodes text into token ids.
- Decodes token ids back to text.
- Keeps configured special tokens intact.
- Saves and reloads vocab and merge files.

## Example

```python
from bpe_tokenizer import Tokenizer, train_bpe

special_tokens = ["<|endoftext|>"]
vocab, merges = train_bpe("corpus.txt", vocab_size=50257, special_tokens=special_tokens)

tokenizer = Tokenizer(vocab, merges, special_tokens)
ids = tokenizer.encode("hello<|endoftext|>world")
text = tokenizer.decode(ids)

tokenizer.save("vocab.json", "merges.json")
loaded = Tokenizer.from_files("vocab.json", "merges.json")
```

## Test

```bash
python3 -m unittest discover -s tests
```
