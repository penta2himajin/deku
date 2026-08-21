"""git_search: readonly git log/show → rank → grounded reply.

Only allowlisted subcommands (log, show, rev-parse). MiniCPM optional;
lexical answers cover commit subject / author questions.
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

ALLOWED = frozenset({"log", "show", "rev-parse"})
LOG_FORMAT = "%H%x09%an%x09%ad%x09%s"
MAX_SHOW_CHARS = 8_000


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
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "git failed").strip())
    return r.stdout


def index_log(root: Path, *, limit: int = 20) -> list[dict]:
    out = git_run(
        root,
        ["log", f"-n{limit}", f"--format={LOG_FORMAT}", "--date=short"],
    )
    return _parse_log(out)


def index_log_path(root: Path, path: str, *, limit: int = 10) -> list[dict]:
    """Commits that touched `path` (newest first)."""
    out = git_run(
        root,
        [
            "log", f"-n{limit}", f"--format={LOG_FORMAT}", "--date=short",
            "--", path,
        ],
    )
    hits = _parse_log(out)
    for h in hits:
        h["path_filter"] = path
    return hits


def _parse_log(out: str) -> list[dict]:
    hits = []
    for i, line in enumerate((out or "").splitlines()):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        sha, author, date, subject = parts[0], parts[1], parts[2], parts[3]
        short = sha[:12]
        snippet = f"{short}  {date}  {author}  {subject}"
        hits.append({
            "title": f"commit {short}",
            "snippet": snippet,
            "path": short,
            "url": f"git:{short}",
            "sha": sha,
            "author": author,
            "date": date,
            "subject": subject,
            "idx": i,
        })
    return hits


PATH_RE = re.compile(
    r"\b((?:[\w.-]+/)*[\w.-]+\.(?:py|md|json|txt|toml|sh|ya?ml|csv))\b"
)


def extract_repo_path(question: str) -> str | None:
    m = PATH_RE.search(question or "")
    return m.group(1) if m else None


def wants_path_history(question: str) -> bool:
    """Last commit / message / author scoped to a file path."""
    if not extract_repo_path(question):
        return False
    return bool(re.search(
        r"(?i)\b("
        r"last commit|commit message|who (?:authored|last (?:changed|touched))|"
        r"commit (?:that )?(?:changed|touched|modified)|"
        r"when was .+ (?:changed|committed|touched)"
        r")\b",
        question or "",
    ))


def show_commit(root: Path, rev: str = "HEAD") -> str:
    text = git_run(root, ["show", "--stat", "--format=fuller", rev])
    if len(text) > MAX_SHOW_CHARS:
        text = text[:MAX_SHOW_CHARS].rsplit("\n", 1)[0] + "\n…"
    return text


def wants_last_commit(question: str) -> bool:
    return bool(re.search(
        r"(?i)\b(last commit|most recent commit|latest commit|head commit)\b",
        question or "",
    ))


def wants_author(question: str) -> bool:
    return bool(re.search(r"(?i)\b(who (?:authored|wrote|committed)|author)\b", question or ""))


def wants_message(question: str) -> bool:
    return bool(re.search(r"(?i)\b(commit message|subject)\b", question or ""))


def wants_files(question: str) -> bool:
    return bool(re.search(
        r"(?i)\b(what files?|which files?|files changed|what changed in the last commit)\b",
        question or "",
    ))


def parse_stat_files(show_text: str) -> list[str]:
    """Filenames from `git show --stat` summary lines."""
    files = []
    for line in (show_text or "").splitlines():
        m = re.match(r"^\s*([^|]+?)\s*\|\s*\d+", line)
        if not m:
            continue
        name = m.group(1).strip()
        if name and "file changed" not in name and "files changed" not in name:
            files.append(name)
    return files


def rank_hits(question: str, hits: list[dict], k: int = 4) -> list[dict]:
    scored = []
    for h in hits:
        text = f"{h.get('subject', '')} {h.get('author', '')} {h.get('snippet', '')}"
        score = float(extract.term_score(question, text))
        if wants_last_commit(question) and h.get("idx") == 0:
            score += 10.0
        scored.append((score, h))
    scored.sort(key=lambda x: (-x[0], x[1].get("idx", 0)))
    return [h for _, h in scored[:k]]


def hits_to_document(hits: list[dict], *, show_text: str = "") -> str:
    parts = []
    if show_text:
        parts.append(f"git show\n{show_text.strip()}\nSource: git:show")
    for h in hits:
        parts.append(
            f"{h.get('title')}\n{h.get('snippet')}\nSource: {h.get('url')}"
        )
    return "\n\n".join(parts)


def lexical_answer(question: str, hits: list[dict], show_text: str = "") -> str | None:
    if not hits:
        return None
    top = hits[0]
    path = extract_repo_path(question) or top.get("path_filter")
    if wants_author(question):
        return render.git_author(
            str(top.get("author") or ""),
            str(top.get("sha") or ""),
            str(top.get("subject") or ""),
        )
    if wants_files(question) and not path:
        files = parse_stat_files(show_text)
        subj = str(top.get("subject") or "").strip()
        if files:
            return render.git_files(subj, files)
        return render.git_message(subj) if subj else None
    if wants_message(question) or wants_last_commit(question) or wants_path_history(question):
        subj = str(top.get("subject") or "").strip()
        if not subj:
            return None
        if path and wants_path_history(question):
            return render.git_path_message(path, subj)
        return render.git_message(subj)
    pool = show_text + "\n" + "\n".join(h.get("snippet", "") for h in hits)
    best, best_sc = None, 0.0
    for line in pool.splitlines():
        s = line.strip()
        if len(s.split()) < 3:
            continue
        if s.startswith("diff --git") or s.startswith("index "):
            continue
        sc = float(extract.term_score(question, s))
        if sc > best_sc:
            best, best_sc = s, sc
    return best if best_sc >= 1.0 else (top.get("subject") or None)


def finalize_reply(
    *,
    question: str,
    doc: str,
    hits: list[dict],
    show_text: str,
    core: str | None,
    summary: str | None,
) -> str | None:
    lex = lexical_answer(question, hits, show_text=show_text)
    if wants_author(question) or wants_message(question) or wants_last_commit(question) or wants_files(question) or wants_path_history(question):
        return lex
    if (
        core
        and summary
        and ws.core_in_reply(core, summary)
        and ws.reply_grounded(summary, doc)
        and len(summary.split()) >= ws.MIN_SUMMARY_WORDS
    ):
        return summary.strip()
    if core and ws.reply_grounded(core, doc):
        sent = ws.sentence_with_core(core, doc)
        if sent:
            return sent
        return core.strip()
    return lex


def run(
    question: str,
    *,
    root: str | Path,
    limit: int = 20,
    seed: int = 0,
    live_answer: bool = True,
) -> Result:
    root = Path(root)
    out = Result(detail={"root": str(root)})
    path = extract_repo_path(question)
    try:
        if path and wants_path_history(question):
            hits = index_log_path(root, path, limit=limit)
            out.detail["path"] = path
        else:
            hits = index_log(root, limit=limit)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        out.status = "git_error"
        out.answer = ws.CANNOT_ANSWER
        out.detail["error"] = str(e)
        return out
    if not hits:
        out.status = "no_hits"
        out.answer = ws.CANNOT_ANSWER
        if path:
            out.detail["abstain_reason"] = "no_commits_for_path"
        return out
    ranked = rank_hits(question, hits, k=4)
    out.hits = ranked
    rev = ranked[0].get("sha") or "HEAD"
    if wants_last_commit(question) or wants_path_history(question):
        rev = hits[0].get("sha") or "HEAD"
        ranked = [hits[0]] + [h for h in ranked if h is not hits[0]]
        out.hits = ranked
    try:
        show_text = show_commit(root, rev)
    except RuntimeError as e:
        show_text = ""
        out.detail["show_error"] = str(e)
    doc = hits_to_document(ranked, show_text=show_text)
    out.document = doc

    if not live_answer:
        ans = lexical_answer(question, ranked, show_text=show_text)
        if not ans:
            out.answer, out.status = ws.CANNOT_ANSWER, "cannot_answer"
            return out
        out.detail["reply_source"] = "lexical"
        out.answer, out.status = ans, "ok"
        return out

    # Prefer deterministic fields for the common ask shapes.
    if (
        wants_author(question)
        or wants_message(question)
        or wants_last_commit(question)
        or wants_files(question)
        or wants_path_history(question)
    ):
        ans = lexical_answer(question, ranked, show_text=show_text)
        if ans:
            out.detail["reply_source"] = "lexical"
            out.answer, out.status = ans, "ok"
            return out

    core, status = ws.minicpm_extract(question, doc, seed=seed)
    out.detail["core"] = core
    out.detail["extract_status"] = status
    summary = ws.minicpm_summarize(question, doc, seed=seed)
    out.detail["summary"] = summary
    reply = finalize_reply(
        question=question,
        doc=doc,
        hits=ranked,
        show_text=show_text,
        core=core,
        summary=summary,
    )
    if not reply:
        out.answer, out.status = ws.CANNOT_ANSWER, "cannot_answer"
        out.detail["abstain_reason"] = "no_grounded_core"
        return out
    if summary and reply == summary.strip():
        out.detail["reply_source"] = "compose"
    else:
        out.detail["reply_source"] = "lexical_or_core"
    out.answer, out.status = reply, "ok"
    return out
