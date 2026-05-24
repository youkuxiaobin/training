# Inference

This directory contains the checkpoint loading and text generation code.

Expected files in a trained model directory:

```text
model.pt or best.pt
vocab.json
merges.json
```

Command-line usage:

```bash
python3 scripts/generate_text.py \
  --model-dir runs/tinystories_model \
  --checkpoint best.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 100
```

Python usage:

```python
from inference import TextGenerator

generator = TextGenerator.from_checkpoint(
    "runs/tinystories_model",
    checkpoint_name="best.pt",
    device="auto",
)

print(generator.generate_text("Once upon a time", max_new_tokens=100))
```
