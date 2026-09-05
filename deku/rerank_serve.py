"""Optional MiniCPM-Reranker HTTP sidecar (torch lives here, not in agent core).

Install: ``uv sync --extra rerank``
Run: ``uv run deku-rerank`` or ``mise run serve-rerank``

Env:
  DEKU_RERANK_HOST (default 127.0.0.1)
  DEKU_RERANK_PORT (default 8091)
  DEKU_RERANK_MODEL (default openbmb/MiniCPM-Reranker-Light)
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


DEFAULT_HOST = os.environ.get("DEKU_RERANK_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("DEKU_RERANK_PORT", "8091"))
DEFAULT_MODEL = os.environ.get(
    "DEKU_RERANK_MODEL", "openbmb/MiniCPM-Reranker-Light"
)


def _load_model():
    try:
        import torch
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise SystemExit(
            "rerank extra missing; install with: uv sync --extra rerank\n"
            f"({e})"
        ) from e

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"deku-rerank: loading {DEFAULT_MODEL} on {device}", flush=True)
    model = CrossEncoder(
        DEFAULT_MODEL,
        max_length=512,
        trust_remote_code=True,
        device=device,
        automodel_args={"torch_dtype": dtype},
    )
    model.tokenizer.padding_side = "right"
    print("deku-rerank: ready", flush=True)
    return model


def _score(model, query: str, documents: list[str]) -> list[float]:
    if not documents:
        return []
    pairs = [[f"Query: {query}", doc] for doc in documents]
    scores = model.predict(pairs, convert_to_numpy=True)
    return [float(s) for s in scores]


def make_handler(model):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # quieter
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, payload: dict) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in ("", "/health"):
                self._send(200, {"status": "ok", "model": DEFAULT_MODEL})
                return
            self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/rerank":
                self._send(404, {"error": "not_found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) if length else b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "bad_json"})
                return
            query = body.get("query") or ""
            docs = body.get("documents")
            if not isinstance(docs, list) or not all(
                isinstance(d, str) for d in docs
            ):
                self._send(400, {"error": "documents_must_be_string_list"})
                return
            try:
                scores = _score(model, str(query), docs)
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
                return
            self._send(200, {"scores": scores})

    return Handler


def main(argv: list[str] | None = None) -> int:
    _ = argv
    model = _load_model()
    host = DEFAULT_HOST
    port = DEFAULT_PORT
    server = ThreadingHTTPServer((host, port), make_handler(model))
    print(f"deku-rerank: listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\ndeku-rerank: stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
