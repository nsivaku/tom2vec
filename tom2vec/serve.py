from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tom2vec.model import ToM2Vec

DEFAULTS = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}


def split_reasoning(text: str) -> tuple[str, str]:
    """Split Qwen3 output into (reasoning, answer) at the last </think>."""
    if "</think>" not in text:
        return "", text.strip()
    reasoning, answer = text.rsplit("</think>", 1)
    return reasoning.replace("<think>", "").strip(), answer.strip()


def make_handler(model: ToM2Vec, model_name: str, max_tokens: int, keep_reasoning: bool):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/").endswith("/models"):
                self._send(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):  # noqa: N802
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._send(404, {"error": {"message": "only /v1/chat/completions is served"}})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                messages = [
                    {"role": m["role"], "content": _text(m["content"])} for m in req["messages"]
                ]
            except (ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": {"message": f"bad request: {e}"}})
                return

            kw = {k: req[k] if req.get(k) is not None else v for k, v in DEFAULTS.items()}
            limit = req.get("max_completion_tokens") or req.get("max_tokens") or max_tokens
            with lock:
                text = model.generate(
                    messages, max_new_tokens=int(limit), seed=req.get("seed"), **kw
                )
            reasoning, answer = split_reasoning(text)
            message = {"role": "assistant", "content": text if keep_reasoning else answer}
            if reasoning and not keep_reasoning:
                message["reasoning_content"] = reasoning
            self._send(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            })

        def log_message(self, fmt, *args):
            print(f"[tom2vec.serve] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def _text(content) -> str:
    """Accept both string content and OpenAI content-part lists."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def main() -> None:
    p = argparse.ArgumentParser(description="Serve ToM2Vec behind an OpenAI-compatible API.")
    p.add_argument("--ckpt", required=True, help="trained checkpoint (or a base model id)")
    p.add_argument("--model-name", default="tom2vec", help="name reported by /v1/models")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--k-latents", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=4096, help="default reply budget")
    p.add_argument("--no-thinking", action="store_true", help="disable Qwen3 thinking mode")
    p.add_argument("--keep-reasoning", action="store_true",
                   help="return the <think> block inside `content` (vLLM without a reasoning parser)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    model = ToM2Vec(
        model_id=args.ckpt, k_latents=args.k_latents, dtype=args.dtype, device=args.device,
        enable_thinking=not args.no_thinking,
    )
    model.eval()
    handler = make_handler(model, args.model_name, args.max_tokens, args.keep_reasoning)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[tom2vec.serve] {args.ckpt} at http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
