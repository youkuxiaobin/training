"""Serve the OpenAI chat page and JSON API."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from openai_chat.client import OpenAIChatClient, OpenAIChatError


STATIC_DIR = Path(__file__).resolve().parent / "static"


class OpenAIChatHandler(BaseHTTPRequestHandler):
    server_version = "OpenAIChat/0.1"

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
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
        if self.path != "/api/chat":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json()
            messages = payload.get("messages", [])
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            instructions = payload.get("instructions", "")
            if not isinstance(instructions, str):
                raise ValueError("instructions must be a string")
            model = payload.get("model")
            if model is not None and not isinstance(model, str):
                raise ValueError("model must be a string")
            client = OpenAIChatClient(model=model.strip() if model else None)
            reply = client.create_reply(
                messages=messages,
                instructions=instructions,
                max_output_tokens=int(payload.get("max_output_tokens", 512)),
            )
            self._send_json({"reply": reply, "model": client.model})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"error": f"bad request: {exc}"}, HTTPStatus.BAD_REQUEST)
        except OpenAIChatError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

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

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
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
    return "application/octet-stream"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OpenAI chat web server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), OpenAIChatHandler)
    print(f"OpenAI chat page: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
