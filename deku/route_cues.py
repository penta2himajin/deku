"""Shared lexical cues for routing / plan building (no import cycles)."""
from __future__ import annotations

import re
from pathlib import Path

# SNAKE_CAPS always count as code constants when written that way in the ask.
# Bare ALLCAPS (PREFILL) and lowercase mentions come from repo discovery of
# config-like assignments — not a hard-coded product allowlist.
_DIR_SNAKE = re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b")
_DIR_BARE_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")
_ASSIGN_LINE = re.compile(r"(?m)^([A-Z][A-Z0-9_]{2,})\s*=\s*(.+)$")
_PYPROJECT_NAME = re.compile(
    r"(?m)^name\s*=\s*[\"']([A-Za-z0-9][A-Za-z0-9._-]*)[\"']"
)

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

# Product-agnostic overview / local-docs cues (no package name, no PREFILL).
DIR_WORDS = re.compile(
    r"(?i)\b("
    r"project|purpose|overview|readme|about|"
    r"this (?:project|repo|repository)|"
    r"the readme|"
    r"how does (?:the )?(?:client|code|module|function|package)"
    r")\b"
)

_ident_cache: dict[str, frozenset[str]] = {}
_snake_cache: dict[str, frozenset[str]] = {}
_name_cache: dict[str, frozenset[str]] = {}


def clear_ident_cache() -> None:
    """Drop discovered-ident / project-name caches (tests / root changes)."""
    _ident_cache.clear()
    _snake_cache.clear()
    _name_cache.clear()


def _is_config_literal_rhs(rhs: str) -> bool:
    """True for string/number/bool/None or a simple list — not catalogs."""
    s = (rhs or "").strip()
    if not s:
        return False
    if s.startswith(('"""', "'''", '"', "'")):
        return True
    if re.match(r"^-?\d", s):
        return True
    if re.match(r"^(True|False|None)\b", s):
        return True
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


def discover_snake_config_idents(root: str = ".") -> frozenset[str]:
    """SNAKE_CAPS config-like assignments (e.g. MAX_TOKENS)."""
    key = str(Path(root).resolve())
    cached = _snake_cache.get(key)
    if cached is not None:
        return cached
    found: set[str] = set()
    base = Path(root)
    if not base.is_dir():
        _snake_cache[key] = frozenset()
        return _snake_cache[key]
    for path in base.rglob("*.py"):
        if any(part in _SKIP_DIR_PARTS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in _ASSIGN_LINE.finditer(text):
            name, rhs = m.group(1), m.group(2)
            if "_" not in name:
                continue
            if _is_config_literal_rhs(rhs):
                found.add(name)
    out = frozenset(found)
    _snake_cache[key] = out
    return out


def discover_project_names(root: str = ".") -> frozenset[str]:
    """Local package / directory names for soft dir routing."""
    key = str(Path(root).resolve())
    cached = _name_cache.get(key)
    if cached is not None:
        return cached
    names: set[str] = set()
    base = Path(root)
    if base.is_dir():
        stem = base.resolve().name
        if stem and stem not in {".", ".."} and not stem.startswith("."):
            names.add(stem)
        pyproject = base / "pyproject.toml"
        if pyproject.is_file():
            try:
                text = pyproject.read_text(encoding="utf-8")
            except OSError:
                text = ""
            m = _PYPROJECT_NAME.search(text)
            if m:
                names.add(m.group(1))
                # setuptools normalizes deku_foo → often keep raw; also bare.
                names.add(m.group(1).replace("-", "_"))
                names.add(m.group(1).replace("_", "-"))
    out = frozenset(n for n in names if len(n) >= 3)
    _name_cache[key] = out
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


def mentions_repo_constant(text: str, *, root: str = ".") -> bool:
    """ALLCAPS/snake in ask, or lowercase mention of a discovered config name."""
    if has_dir_ident(text, root=root):
        return True
    low = (text or "").casefold()
    for name in discover_bare_config_idents(root) | discover_snake_config_idents(root):
        if re.search(rf"\b{re.escape(name.casefold())}\b", low):
            return True
    return False


def soft_dir_match(text: str, *, root: str = ".") -> bool:
    """Soft dir_search cue: overview words, repo constants, or project name."""
    if DIR_WORDS.search(text or ""):
        return True
    if mentions_repo_constant(text, root=root):
        return True
    low = (text or "").casefold()
    for name in discover_project_names(root):
        if re.search(rf"\b{re.escape(name.casefold())}\b", low):
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
