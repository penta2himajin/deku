"""OpenAI-compatible chat completion client (stdlib only).

Talks HTTP only — never imports llama.cpp, MLX, or GGUF parsers.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

BASE_URL = os.environ.get("DEKU_URL", "http://127.0.0.1:8080").rstrip("/")
MODEL = os.environ.get("DEKU_MODEL", "MiniCPM5-1B")
API_KEY = os.environ.get("DEKU_API_KEY", "")


class LLMError(RuntimeError):
    """Raised when the completion endpoint is unreachable or returns an error."""


def complete(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    think: bool = False,
    timeout: float = 120.0,
) -> str:
    """POST /v1/chat/completions and return the assistant text.

    MiniCPM5 defaults to thinking unless ``enable_thinking`` is false; deku
    asks for short grounded answers, so ``think=False`` is the product default.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise LLMError(f"HTTP {e.code} from {BASE_URL}: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(
            f"server not reachable at {BASE_URL} ({e.reason}); "
            "start it with ./bin/deku-serve"
        ) from e
    try:
        choice = payload["choices"][0]
        message = choice.get("message") or {}
        text = message.get("content") or message.get("reasoning_content") or choice.get("text") or ""
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"unexpected response shape: {payload!r}") from e
    return text
