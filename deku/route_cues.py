"""Shared lexical cues for routing / plan building (no import cycles)."""
from __future__ import annotations

import re
from pathlib import Path

# SNAKE_CAPS always count as code constants (MAX_TOKENS, DEKU_URL, …).
# Bare ALLCAPS (PREFILL) come from repo discovery of config-like assignments —
# not a hard-coded allowlist, and not arbitrary acronyms (LVMH, NASA).
_DIR_SNAKE = re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b")
_DIR_BARE_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")
_ASSIGN_LINE = re.compile(r"(?m)^([A-Z][A-Z0-9]{2,})\s*=\s*(.+)$")

_SKIP_DIR_PARTS = frozenset({
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    "tests",
    "evals",
    "dist",
    "build",
})

DIR_WORDS = re.compile(
    r"(?i)\b(project|purpose|overview|harness|readme|about|prefill|max_tokens|"
    r"is_looping|how does the client|language models? does|deku)\b"
)

_ident_cache: dict[str, frozenset[str]] = {}


def clear_ident_cache() -> None:
    """Drop discovered-ident cache (tests / root changes)."""
    _ident_cache.clear()


def _is_config_literal_rhs(rhs: str) -> bool:
    """True for string/number/bool/None or a simple list — not catalogs."""
    s = (rhs or "").strip()
    if not s:
        return False
    # Multi-line string opener or normal quotes.
    if s.startswith(('"""', "'''", '"', "'")):
        return True
    if re.match(r"^-?\d", s):
        return True
    if re.match(r"^(True|False|None)\b", s):
        return True
    # STOPS = ["\n", "<|im_end|>", ...]
    if s.startswith("["):
        return True
    return False


def discover_bare_config_idents(root: str = ".") -> frozenset[str]:
    """Bare ALLCAPS names assigned to config-like literals under ``root``."""
    key = str(Path(root).resolve())
    cached = _ident_cache.get(key)
    if cached is not None:
        return cached
    found: set[str] = set()
    base = Path(root)
    if not base.is_dir():
        _ident_cache[key] = frozenset()
        return _ident_cache[key]
    for path in base.rglob("*.py"):
        if any(part in _SKIP_DIR_PARTS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in _ASSIGN_LINE.finditer(text):
            name, rhs = m.group(1), m.group(2)
            if "_" in name:
                continue
            if _is_config_literal_rhs(rhs):
                found.add(name)
    out = frozenset(found)
    _ident_cache[key] = out
    return out


def has_dir_ident(text: str, *, root: str = ".") -> bool:
    """True when the text names a code-like constant, not a public acronym."""
    if _DIR_SNAKE.search(text or ""):
        return True
    known = discover_bare_config_idents(root)
    for m in _DIR_BARE_RE.finditer(text or ""):
        name = m.group(1)
        if "_" in name:
            continue
        if name in known:
            return True
    return False


def dir_idents(text: str, *, root: str = ".") -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for m in _DIR_SNAKE.finditer(text or ""):
        tok = m.group(0)
        if tok not in seen:
            seen.add(tok)
            found.append(tok)
    known = discover_bare_config_idents(root)
    for m in _DIR_BARE_RE.finditer(text or ""):
        name = m.group(1)
        if "_" in name or name not in known or name in seen:
            continue
        seen.add(name)
        found.append(name)
    return found
