#!/usr/bin/env python3
"""Dummy OpenAI-compatible chat completions server for testing markdown rendering.

Streams content from test_markdown.md with random chunk sizes to mimic a real LLM.

Usage:
    python dummy_api.py          # starts on port 8099
    python dummy_api.py --port 8080

Then point kivi at it:
    OPENAI_BASE_URL=http://localhost:8099/v1 python kivi.py
"""

import argparse
import json
import time
import random
import http.server
from pathlib import Path

_MD_FILE = Path(__file__).parent / "test_markdown.md"
RESPONSE_TEXT = _MD_FILE.read_text()


def _make_chunk(content: str, finish: bool = False) -> bytes:
    delta = {"role": "assistant", "content": content} if not finish else {}
    chunk = {
        "id": "chatcmpl-dummy",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "dummy-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": "stop" if finish else None}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _stream_response(wfile, text: str):
    """Stream text in random-sized chunks (1–40 chars) to mimic a real LLM."""
    i = 0
    while i < len(text):
        size = random.randint(1, 40)
        piece = text[i:i + size]
        wfile.write(_make_chunk(piece))
        wfile.flush()
        i += size
        time.sleep(random.uniform(0.001, 0.015))
    wfile.write(_make_chunk("", finish=True))
    wfile.write(b"data: [DONE]\n\n")
    wfile.flush()


class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[dummy_api] {fmt % args}")

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            req = json.loads(body) if body else {}
        except json.JSONDecodeError:
            req = {}

        if self.path in ("/v1/chat/completions", "/chat/completions"):
            if req.get("stream", False):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                _stream_response(self.wfile, RESPONSE_TEXT)
            else:
                resp = {
                    "id": "chatcmpl-dummy",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "dummy-model",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": RESPONSE_TEXT},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 500, "total_tokens": 510},
                }
                self._send_json(200, resp)
        else:
            self._send_json(404, {"error": f"unknown path: {self.path}"})

    def _send_json(self, status: int, data: dict):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    print(f"Loaded {len(RESPONSE_TEXT)} chars from {_MD_FILE.name}")
    server = http.server.HTTPServer((args.host, args.port), Handler)
    print(f"dummy_api running at http://{args.host}:{args.port}/v1")
    print("  POST /v1/chat/completions  →  streams test_markdown.md with random chunk sizes")
    print("  GET  /health               →  {status: ok}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
