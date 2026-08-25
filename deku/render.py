"""Turn grounded evidence into short natural-language replies.

Every renderer only rearranges tokens already present in the evidence; it does
not invent facts. Callers must still run grounding checks where MiniCPM is involved.
"""
from __future__ import annotations

import re


def assignment(
    ident: str,
    value: str,
    *,
    path: str | None = None,
    where: bool = False,
) -> str:
    """Short grounded assignment reply; optionally name the defining file."""
    v = (value or "").strip()
    p = (path or "").strip()
    if p and where:
        return f"{ident} is defined in {p} as {v}."
    if p:
        return f"{ident} is set to {v} in {p}."
    return f"{ident} is set to {v}."


def definition(sig: str, gloss: str | None = None) -> str:
    """`def foo(...):` plus optional first docstring sentence."""
    name = ""
    m = re.match(r"def\s+(\w+)\s*\(", sig or "")
    if m:
        name = m.group(1)
    if gloss and name:
        g = gloss.strip().rstrip(".")
        # Drop leading "True if" duplication noise lightly.
        return f"{name}: {g}."
    if name:
        return f"{name} is defined as: {(sig or '').strip()}"
    return (sig or "").strip()


def git_message(subject: str) -> str:
    s = (subject or "").strip()
    return f'The last commit message is "{s}".'


def git_path_message(path: str, subject: str) -> str:
    sub = (subject or "").strip()
    p = (path or "").strip()
    if not sub:
        return ""
    if p:
        return f'The last commit that changed {p} is "{sub}".'
    return git_message(sub)


def git_author(author: str, sha: str, subject: str) -> str:
    short = (sha or "")[:12]
    sub = (subject or "").strip()
    return f'{author} authored commit {short} ("{sub}").'


def git_files(subject: str, files: list[str]) -> str:
    sub = (subject or "").strip()
    joined = ", ".join(files)
    if sub:
        return f'The last commit ("{sub}") changed: {joined}.'
    return f"Changed files: {joined}."


def diff_line(path: str, line: str) -> str:
    """One +/- line → a short English sentence when it looks like an assignment."""
    raw = (line or "").rstrip("\n")
    body = raw[1:].strip() if raw[:1] in "+-" else raw.strip()
    sign = raw[:1] if raw[:1] in "+-" else ""
    m = re.match(r"^([A-Za-z_][\w]*)\s*=\s*(.+)$", body)
    if m and sign == "+":
        return f"In {path}, {m.group(1)} changed to {m.group(2).rstrip()}."
    if m and sign == "-":
        return f"In {path}, {m.group(1)} was previously {m.group(2).rstrip()}."
    if sign == "+":
        return f"In {path}, added: {body}"
    if sign == "-":
        return f"In {path}, removed: {body}"
    return f"In {path}: {body}"


def diff_hunk(path: str, lines: list[str]) -> str:
    """Compact English intro + the changed lines (evidence stays visible)."""
    rendered = [diff_line(path, ln) for ln in lines[:4] if ln.strip()]
    # If every line rendered as assignment-style, join; else show block.
    if rendered and all("changed to" in r or "was previously" in r for r in rendered):
        return " ".join(rendered)
    body = "\n".join(ln for ln in lines[:8])
    return f"In {path}, the diff includes:\n{body}"
