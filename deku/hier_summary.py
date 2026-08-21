"""Hierarchical (map-reduce) summarization tuned for MiniCPM5-1B.

Research backdrop (kept deliberately thin in code):
- Map/reduce over chunks is the standard long-doc pattern (Google Cloud
  Workflows; industry writeups on hierarchical merging).
- ACL 2025 Findings (Ou et al., Context-Aware Hierarchical Merging) warn that
  recursive abstractive merge amplifies hallucination; they anchor merges with
  extractive source context. We do the same: leaf nodes are extractive
  sentences; MiniCPM only compresses those notes, and the final reply must
  stay grounded in the union of leaf text.

MiniCPM constraints: tiny chunks, short prompts, max 2 reduce levels, hard
cap on chunk count. No chain-of-thought.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from deku import extract
from deku import web_search as ws

CHUNK_CHARS = 900
CHUNK_OVERLAP = 100
MAX_CHUNKS = 10
MAX_REDUCE_LEVELS = 2
LEAF_SENTENCES = 3
COMPLETE_FN = Callable[[str], str]


@dataclass
class Result:
    answer: str | None = None
    status: str = ""
    document: str = ""
    detail: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    *,
    size: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
    max_chunks: int = MAX_CHUNKS,
) -> list[str]:
    """Split on paragraph boundaries when possible; else sliding windows."""
    body = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if not body:
        return []
    if len(body) <= size:
        return [body]
    parts: list[str] = []
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= size:
            buf = f"{buf}\n\n{p}"
        else:
            parts.append(buf)
            buf = p
            if len(parts) >= max_chunks:
                return parts
    if buf:
        parts.append(buf)
    if len(parts) >= 2:
        return parts[:max_chunks]
    # Monolith paragraph — sliding window with overlap.
    parts = []
    i = 0
    while i < len(body) and len(parts) < max_chunks:
        parts.append(body[i : i + size])
        if i + size >= len(body):
            break
        i += max(size - overlap, 1)
    return parts


def extractive_leaf(chunk: str, *, k: int = LEAF_SENTENCES) -> str:
    """Top-k long sentences from a chunk (no model). Anchors later merges."""
    sents = []
    for block in re.split(r"\n+", chunk or ""):
        for sent in re.split(r"(?<=[.!?])\s+", block):
            s = sent.strip()
            if len(s.split()) < 6:
                continue
            if s.lower().startswith("source:"):
                continue
            sents.append(s)
    if not sents:
        return (chunk or "").strip()[:400]
    # Prefer informative mid-length sentences.
    scored = sorted(
        sents,
        key=lambda s: (-min(len(s.split()), 40), -len(s)),
    )
    picked = scored[:k]
    # Restore approximate document order.
    picked.sort(key=lambda s: (chunk or "").find(s))
    return " ".join(picked)


MAP_PROMPT = """List up to 3 key facts from the notes. One short line each.
Use only words that appear in the notes. No preamble.

Notes:
{notes}
"""

REDUCE_PROMPT = """Combine the notes into 2-4 short English sentences.
Use only facts present in the notes. No preamble.

Notes:
{notes}
"""


def _complete_default(prompt: str, *, seed: int = 0) -> str:
    from deku import llm
    return (llm.complete(prompt, think=False, temp=0.3, seed=seed, max_tokens=80) or "").strip()


def map_chunk(
    chunk: str,
    *,
    live: bool = False,
    complete_fn: COMPLETE_FN | None = None,
    seed: int = 0,
) -> str:
    leaf = extractive_leaf(chunk)
    if not live:
        return leaf
    fn = complete_fn or (lambda p: _complete_default(p, seed=seed))
    raw = fn(MAP_PROMPT.format(notes=leaf[:1200]))
    body = (raw or "").strip()
    if not body:
        return leaf
    # Keep only lines grounded in the leaf (or whole chunk).
    kept = []
    for line in body.splitlines():
        line = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
        if len(line.split()) < 4:
            continue
        if ws.reply_grounded(line, leaf) or ws.reply_grounded(line, chunk):
            kept.append(line)
    return " ".join(kept) if kept else leaf


def reduce_notes(
    notes: list[str],
    *,
    live: bool = False,
    complete_fn: COMPLETE_FN | None = None,
    seed: int = 0,
) -> str:
    blob = "\n".join(f"- {n}" for n in notes if n.strip())
    if not blob.strip():
        return ""
    if not live:
        # Offline reduce: take first sentence of each note, cap length.
        bits = []
        for n in notes:
            sent = re.split(r"(?<=[.!?])\s+", n.strip())[0].strip()
            if sent:
                bits.append(sent)
        return " ".join(bits[:6])
    fn = complete_fn or (lambda p: _complete_default(p, seed=seed))
    raw = fn(REDUCE_PROMPT.format(notes=blob[:3500]))
    body = (raw or "").strip().splitlines()[0].strip() if (raw or "").strip() else ""
    if body and ws.reply_grounded(body, blob):
        return body
    return reduce_notes(notes, live=False)


def summarize(
    text: str,
    *,
    question: str = "",
    live: bool = False,
    complete_fn: COMPLETE_FN | None = None,
    seed: int = 0,
) -> Result:
    """Map-reduce hierarchical summary. `live` enables MiniCPM map/reduce."""
    out = Result(detail={"mode": "hier_summary"})
    chunks = chunk_text(text)
    out.detail["n_chunks"] = len(chunks)
    if not chunks:
        out.status = "cannot_answer"
        out.answer = ws.CANNOT_ANSWER
        return out
    leaves = [
        map_chunk(c, live=live, complete_fn=complete_fn, seed=seed)
        for c in chunks
    ]
    out.detail["leaves"] = leaves
    level = leaves
    for depth in range(MAX_REDUCE_LEVELS):
        if len(level) == 1:
            break
        # Pairwise / batched reduce to keep prompts small.
        nxt = []
        batch_size = 4
        for i in range(0, len(level), batch_size):
            batch = level[i : i + batch_size]
            nxt.append(
                reduce_notes(
                    batch, live=live, complete_fn=complete_fn, seed=seed + depth
                )
            )
        level = [x for x in nxt if x.strip()]
        out.detail[f"reduce_{depth}"] = level
    answer = (level[0] if level else "").strip()
    # Final grounding against full source (soft: claim tokens ⊆ source).
    if not answer:
        out.status = "cannot_answer"
        out.answer = ws.CANNOT_ANSWER
        return out
    if live and not ws.reply_grounded(answer, text):
        # Fall back to extractive concatenation of leaves.
        answer = reduce_notes(leaves, live=False)
        out.detail["ungrounded_reduce"] = True
    if question and float(extract.term_score(question, answer)) < 0.5:
        # Soft: still OK for pure "summarize this" questions.
        out.detail["low_question_overlap"] = True
    out.document = "\n\n".join(leaves)
    out.answer, out.status = answer, "ok"
    return out


def wants_summary(question: str) -> bool:
    return bool(
        re.search(r"(?i)\b(summarize|summarise|summary of|tl;dr|tldr)\b", question or "")
    )
