# SFT Fine-Tuning

This folder fine-tunes an existing checkpoint on instruction/chat data.

Supported input shapes include JSONL records like:

```json
{"messages":[{"role":"user","content":"who are you?"},{"role":"assistant","content":"I am a small local assistant."}]}
```

and Dolly-style records:

```json
{"instruction":"who are you?","input":"","response":"I am a small local assistant."}
```

Run:

```bash
python3 -m finetune.sft.train \
  --model-dir runs/tiny_model \
  --checkpoint best.pt \
  --train-input data/sft_train.jsonl \
  --output-dir runs/sft_model \
  --steps 1000
```

The output directory contains `best.pt`, `latest.pt`, `model.pt`, `vocab.json`,
and `merges.json`, so it can be used by the existing inference server.
