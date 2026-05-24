# OpenAI Chat

This directory contains a small web chat page backed by the OpenAI Responses API.

Set your API key before starting the server:

```bash
export OPENAI_API_KEY="your_api_key"
```

Run the page locally:

```bash
python3 -m openai_chat.server --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

Optional model override:

```bash
export OPENAI_MODEL="gpt-5.2"
```
