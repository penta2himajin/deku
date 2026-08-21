"""diff_search: readonly git diff (working / staged / unstaged) → grounded reply.

Path filters come from the question when present. Ranking is lexical over
hunks; MiniCPM is optional for compose.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from deku import extract
from deku import render
from deku import web_search as ws

ALLOWED = frozenset({"diff", "rev-parse"})
MAX_DIFF_CHARS = 20_000
PATH_RE = re.compile(
    r"(?i)\b([\w./-]+\.(?:py|md|txt|toml|yml|yaml|json|sh|rs))\b"
)


@dataclass
class Result:
    query: str = ""
    hits: list[dict] = field(default_factory=list)
    document: str = ""
    answer: str | None = None
    status: str = ""
    detail: dict = field(default_factory=dict)


def git_run(root: Path, args: Sequence[str], *, timeout: float = 20.0) -> str:
    if not args or args[0] not in ALLOWED:
        raise ValueError(f"git subcommand not allowed: {args[:1]}")
    r = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if r.returncode not in (0, 1):  # 1 = differences found for diff
        raise RuntimeError((r.stderr or r.stdout or "git failed").strip())
    return r.stdout


def diff_scope(question: str) -> str:
    q = question or ""
    if re.search(r"(?i)\bstaged\b", q):
        return "staged"
    if re.search(r"(?i)\bunstaged\b", q):
        return "unstaged"
    return "working"


def extract_path(question: str) -> str | None:
    m = PATH_RE.search(question or "")
    return m.group(1) if m else None


def diff_args(scope: str, path: str | None = None) -> list[str]:
    if scope == "staged":
        args = ["diff", "--cached"]
    elif scope == "unstaged":
        args = ["diff"]
    else:
        args = ["diff", "HEAD"]
    if path:
        args.extend(["--", path])
    return args


def split_hunks(diff_text: str) -> list[dict]:
    """Split a unified diff into per-file hits."""
    hits = []
    current_path = ""
    buf: list[str] = []

    def flush():
        nonlocal buf, current_path
        if not buf:
            return
        body = "\n".join(buf).strip()
        if body:
            hits.append({
                "title": current_path or "diff",
                "snippet": body[:2000],
                "path": current_path or "",
                "url": f"diff:{current_path or 'working'}",
            })
        buf = []

    for line in (diff_text or "").splitlines():
        if line.startswith("diff --git "):
            flush()
            m = re.search(r"b/(.+)$", line)
            current_path = m.group(1) if m else ""
            buf = [line]
        else:
            buf.append(line)
    flush()
    return hits


def rank_hits(question: str, hits: list[dict], k: int = 4) -> list[dict]:
    want = extract_path(question)
    scored = []
    for h in hits:
        text = f"{h.get('path', '')}\n{h.get('snippet', '')}"
        score = float(extract.term_score(question, text))
        if want and (h.get("path") == want or str(h.get("path", "")).endswith(want)):
            score += 8.0
        # Prefer added/removed lines that mention question tokens.
        score += 0.5 * len(re.findall(r"(?m)^[+-]", h.get("snippet") or ""))
        scored.append((score, h))
    scored.sort(key=lambda x: (-x[0], x[1].get("path") or ""))
    return [h for _, h in scored[:k]]


def hits_to_document(hits: list[dict]) -> str:
    parts = []
    for h in hits:
        parts.append(f"{h.get('path') or h.get('title')}\n{h.get('snippet')}\nSource: {h.get('url')}")
    return "\n\n".join(parts)


def _change_lines(snip: str) -> list[str]:
    return [
        ln for ln in (snip or "").splitlines()
        if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))
    ]


def _noise_line(line: str) -> bool:
    body = line[1:].strip() if line[:1] in "+-" else line.strip()
    if not body or len(body) < 3:
        return True
    if re.fullmatch(r"[{}()\[\];,]+", body):
        return True
    if re.match(r"(?i)^(import |from |#|pass\b)", body):
        return True
    return False


def lexical_answer(question: str, hits: list[dict]) -> str | None:
    if not hits:
        return None
    top = hits[0]
    snip = top.get("snippet") or ""
    path = top.get("path") or "file"
    changes = [ln for ln in _change_lines(snip) if not _noise_line(ln)]
    if not changes:
        changes = _change_lines(snip)
    if not changes:
        return None

    # Identifier / keyword questions → best overlapping +/- line.
    idents = re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", question or "")
    idents += re.findall(r"\b[a-z_][a-z0-9_]{3,}\b", question or "")
    stop = {"what", "changed", "about", "unstaged", "staged", "diff", "with", "from"}
    idents = [i for i in idents if i.casefold() not in stop]
    if idents:
        best, best_sc = None, -1.0
        for line in changes:
            body = line[1:]
            sc = float(extract.term_score(question, body))
            for ident in idents:
                if ident in body:
                    sc += 5.0
                elif ident.casefold() in body.casefold():
                    sc += 2.0
            if line.startswith("+"):
                sc += 1.0
            if sc > best_sc:
                best, best_sc = line, sc
        if best is not None and best_sc >= 2.0:
            return render.diff_line(path, best)

    # Open "what's in the diff" → compact English + evidence lines.
    if re.search(r"(?i)\b(diff|changed|unstaged|staged|what is in)\b", question or ""):
        return render.diff_hunk(path, changes[:8])

    best, best_sc = None, -1.0
    for line in changes:
        body = line[1:].strip()
        sc = float(extract.term_score(question, body))
        if line.startswith("+"):
            sc += 1.0
        if sc > best_sc:
            best, best_sc = line, sc
    if best is not None and best_sc >= 1.0:
        return render.diff_line(path, best)
    return render.diff_hunk(path, changes[:6])


def run(
    question: str,
    *,
    root: str | Path,
    seed: int = 0,
    live_answer: bool = True,
) -> Result:
    root = Path(root)
    scope = diff_scope(question)
    path = extract_path(question)
    out = Result(
        query=path or scope,
        detail={"root": str(root), "scope": scope, "path": path},
    )
    try:
        raw = git_run(root, diff_args(scope, path))
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        out.status = "git_error"
        out.answer = ws.CANNOT_ANSWER
        out.detail["error"] = str(e)
        return out
    if len(raw) > MAX_DIFF_CHARS:
        raw = raw[:MAX_DIFF_CHARS] + "\n…"
        out.detail["truncated"] = True
    if not raw.strip():
        out.status = "no_diff"
        out.answer = ws.CANNOT_ANSWER
        out.detail["abstain_reason"] = "no_diff"
        return out
    hits = split_hunks(raw)
    if not hits:
        hits = [{
            "title": path or scope,
            "snippet": raw[:2000],
            "path": path or "",
            "url": f"diff:{scope}",
        }]
    ranked = rank_hits(question, hits, k=4)
    out.hits = ranked
    doc = hits_to_document(ranked)
    out.document = doc

    if not live_answer:
        ans = lexical_answer(question, ranked)
        if not ans:
            out.answer, out.status = ws.CANNOT_ANSWER, "cannot_answer"
            return out
        out.detail["reply_source"] = "lexical"
        out.answer, out.status = ans, "ok"
        return out

    ans = lexical_answer(question, ranked)
    # Diff questions are usually answered by the changed lines themselves.
    if ans and (
        path
        or re.search(r"(?i)\b(diff|changed|unstaged|staged)\b", question or "")
    ):
        out.detail["reply_source"] = "lexical"
        out.answer, out.status = ans, "ok"
        return out

    core, status = ws.minicpm_extract(question, doc, seed=seed)
    out.detail["core"] = core
    out.detail["extract_status"] = status
    score = float(extract.term_score(question, doc))
    if not core or ws.should_abstain(
        question=question, doc=doc, score=max(score, ws.MIN_HIT_SCORE), core=core
    ):
        if ans:
            out.detail["reply_source"] = "lexical_fallback"
            out.answer, out.status = ans, "ok"
            return out
        out.answer, out.status = ws.CANNOT_ANSWER, "cannot_answer"
        out.detail["abstain_reason"] = "no_grounded_core"
        return out
    summary = ws.minicpm_summarize(question, doc, seed=seed)
    out.detail["summary"] = summary
    reply = ws.compose_reply(core, summary, doc, question=question) or ans
    if not reply:
        out.answer, out.status = ws.CANNOT_ANSWER, "cannot_answer"
        return out
    out.detail["reply_source"] = "compose"
    out.answer, out.status = reply, "ok"
    return out
