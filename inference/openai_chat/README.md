# Local OpenAI-Compatible Chat

This directory serves a locally trained model through an OpenAI-compatible chat API.

Expected model directory:

```text
runs/tiny_model/
  best.pt
  latest.pt
  model.pt
  vocab.json
  merges.json
```

Start the local API and web page:

```bash
python3 -m inference.openai_chat.server \
  --model-dir runs/tiny_model \
  --checkpoint best.pt \
  --host 127.0.0.1 \
  --port 8000
```

Open the page:

```text
http://127.0.0.1:8000
```

OpenAI-compatible endpoint:

```text
POST http://127.0.0.1:8000/v1/chat/completions
```

Example request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-tiny-gpt",
    "messages": [{"role": "user", "content": "Once upon a time"}],
    "max_tokens": 100,
    "temperature": 0.8
  }'
```
