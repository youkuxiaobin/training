"""Serve a local-model OpenAI-compatible chat API and web page."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from inference.openai_chat.client import (
    DEFAULT_LOCAL_MODEL,
    OpenAIChatClient,
    OpenAIChatError,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class OpenAIChatHandler(BaseHTTPRequestHandler):
    server_version = "LocalOpenAIChat/0.1"
    chat_client: OpenAIChatClient | None = None

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if self.path == "/v1/models":
            self._send_json(self._client().list_models())
            return
        if self.path.startswith("/static/"):
            requested = unquote(self.path.removeprefix("/static/"))
            file_path = (STATIC_DIR / requested).resolve()
            if STATIC_DIR.resolve() not in file_path.parents:
                self._send_json({"error": "invalid static path"}, HTTPStatus.BAD_REQUEST)
                return
            self._serve_file(file_path, _content_type(file_path))
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            self._handle_chat_completion()
            return
        if self.path == "/api/chat":
            self._handle_page_chat()
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _handle_chat_completion(self) -> None:
        try:
            payload = self._read_json()
            response = self._client().create_chat_completion(payload)
            self._send_json(response)
        except (ValueError, TypeError, json.JSONDecodeError, OpenAIChatError) as exc:
            self._send_json({"error": {"message": str(exc)}}, HTTPStatus.BAD_REQUEST)

    def _handle_page_chat(self) -> None:
        try:
            payload = self._read_json()
            response = self._client().create_chat_completion(
                {
                    "model": payload.get("model"),
                    "messages": payload.get("messages", []),
                    "max_tokens": payload.get("max_output_tokens", 128),
                    "temperature": payload.get("temperature", 0.8),
                    "top_k": payload.get("top_k", 40),
                }
            )
            content = response["choices"][0]["message"]["content"]
            self._send_json({"reply": content, "model": response["model"]})
        except (ValueError, TypeError, json.JSONDecodeError, OpenAIChatError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _client(self) -> OpenAIChatClient:
        if self.chat_client is None:
            raise OpenAIChatError("local model is not loaded")
        return self.chat_client

    def _read_json(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        return json.loads(raw_body.decode("utf-8"))

    def _serve_file(self, file_path: Path, content_type: str) -> None:
        if not file_path.is_file():
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(
        self,
        payload: dict[str, object],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def _content_type(file_path: Path) -> str:
    if file_path.suffix == ".css":
        return "text/css; charset=utf-8"
    if file_path.suffix == ".js":
        return "text/javascript; charset=utf-8"
    if file_path.suffix == ".html":
        return "text/html; charset=utf-8"
    return "application/octet-stream"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local OpenAI-compatible chat API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "runs" / "tiny_model")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--model-name", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OpenAIChatHandler.chat_client = OpenAIChatClient.from_checkpoint(
        args.model_dir,
        checkpoint_name=args.checkpoint,
        device=args.device,
        model_name=args.model_name,
    )
    server = ThreadingHTTPServer((args.host, args.port), OpenAIChatHandler)
    print(f"loaded local model: {args.model_dir / args.checkpoint}")
    print(f"OpenAI-compatible API: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"chat page: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
