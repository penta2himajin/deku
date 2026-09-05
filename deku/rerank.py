"""HTTP client for an optional MiniCPM rerank sidecar.

Agent packages stay HTTP-only (no torch / transformers). When
``DEKU_RERANK_URL`` is unset, callers fall back to lexical rank.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def _base_url(url: str | None = None) -> str:
    if url is not None:
        return url.rstrip("/")
    return (os.environ.get("DEKU_RERANK_URL") or "").rstrip("/")


def _timeout() -> float:
    return float(os.environ.get("DEKU_RERANK_TIMEOUT", "30"))


def rerank_enabled(url: str | None = None) -> bool:
    return bool(_base_url(url))


def hit_document(hit: dict) -> str:
    title = (hit.get("title") or "").strip()
    snip = (hit.get("snippet") or "").strip()
    if title and snip:
        return f"{title}\n{snip}"
    return title or snip


def fetch_scores(
    query: str,
    documents: list[str],
    *,
    url: str | None = None,
    timeout: float | None = None,
) -> list[float]:
    """POST ``/v1/rerank`` → ``{"scores": [...]}`` (same length as documents)."""
    base = _base_url(url)
    if not base:
        raise RuntimeError("DEKU_RERANK_URL is not set")
    if not documents:
        return []
    body = json.dumps({"query": query or "", "documents": documents}).encode()
    req = urllib.request.Request(
        f"{base}/v1/rerank",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(
        req, timeout=timeout if timeout is not None else _timeout()
    ) as resp:
        payload: dict[str, Any] = json.loads(resp.read())
    scores = payload.get("scores")
    if not isinstance(scores, list) or len(scores) != len(documents):
        raise RuntimeError(f"bad rerank response: {payload!r}")
    return [float(s) for s in scores]


def try_rerank_hits(
    question: str,
    hits: list[dict],
    *,
    k: int = 4,
    url: str | None = None,
) -> list[tuple[float, dict]] | None:
    """Rerank ``hits`` when the sidecar is configured; else ``None``.

    On transport / shape errors returns ``None`` so callers keep lexical rank.
    """
    if not rerank_enabled(url):
        return None
    if not hits:
        return []
    docs = [hit_document(h) for h in hits]
    try:
        scores = fetch_scores(question, docs, url=url)
    except (
        OSError,
        urllib.error.HTTPError,
        urllib.error.URLError,
        RuntimeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):
        return None
    scored = sorted(
        zip(scores, hits, strict=True),
        key=lambda x: (-x[0], hits.index(x[1]) if x[1] in hits else 0),
    )
    return [(s, h) for s, h in scored[:k]]
