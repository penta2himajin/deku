"""OpenAI-compatible completion client (stdlib only).

Talks HTTP only — never imports llama.cpp, MLX, or GGUF parsers.

Supports both chat completions and raw completions with a reply prefill
(used by grounded extract prompts for MiniCPM5-1B).
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

# Recommended sampling per MiniCPM5-1B model card (think / no-think).
_SAMPLING = {False: 0.7, True: 0.9}


class LLMError(RuntimeError):
    """Raised when the completion endpoint is unreachable or returns an error."""


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def _post(path: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers=_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise LLMError(f"HTTP {e.code} from {BASE_URL}: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(
            f"server not reachable at {BASE_URL} ({e.reason}); "
            "start it with mise run serve"
        ) from e


def complete(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 256,
    temperature: float | None = None,
    temp: float | None = None,
    think: bool = False,
    prefill: str | None = None,
    stop: list[str] | None = None,
    seed: int | None = None,
    timeout: float = 120.0,
) -> str:
    """Complete via chat or raw completions.

    When ``prefill`` is set, uses ``/v1/completions`` with a ChatML prompt so
    MiniCPM stays on the forced reply prefix (extract path). Otherwise uses
    ``/v1/chat/completions`` with ``enable_thinking`` from ``think``.
    """
    use_temp = (
        temp if temp is not None else temperature if temperature is not None else _SAMPLING[think]
    )
    body: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": use_temp,
        "top_p": 0.95,
        "stream": False,
        "repetition_penalty": 1.05,
    }
    if stop:
        body["stop"] = stop
    if seed is not None:
        body["seed"] = seed

    if prefill is not None:
        sys_block = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
        body["prompt"] = (
            f"{sys_block}<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{prefill}"
        )
        payload = _post("/v1/completions", body, timeout=timeout)
        try:
            choice = payload["choices"][0]
            text = choice.get("text") or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {payload!r}") from e
        return prefill + text

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body["messages"] = messages
    body["chat_template_kwargs"] = {"enable_thinking": think}
    payload = _post("/v1/chat/completions", body, timeout=timeout)
    try:
        choice = payload["choices"][0]
        message = choice.get("message") or {}
        text = (
            message.get("content")
            or message.get("reasoning_content")
            or choice.get("text")
            or ""
        )
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"unexpected response shape: {payload!r}") from e
    return text
