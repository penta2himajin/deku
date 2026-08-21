"""Shared lexical cues for routing / plan building (no import cycles)."""
from __future__ import annotations

import re

# Code constants: SNAKE_CAPS, or a small allowlist of bare ALLCAPS used in-repo.
# Do NOT treat arbitrary acronyms (LVMH, NASA, UNESCO) as dir_search ids.
_DIR_BARE = frozenset({
    "PREFILL", "TEMP", "STOPS", "PROMPT", "UA",
})
_DIR_SNAKE = re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b")
_DIR_BARE_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")

DIR_WORDS = re.compile(
    r"(?i)\b(project|purpose|overview|harness|readme|about|prefill|max_tokens|"
    r"is_looping|how does the client|language models? does|deku)\b"
)


def has_dir_ident(text: str) -> bool:
    """True when the text names a code-like constant, not a public acronym."""
    if _DIR_SNAKE.search(text or ""):
        return True
    for m in _DIR_BARE_RE.finditer(text or ""):
        if m.group(1) in _DIR_BARE:
            return True
    return False


def dir_idents(text: str) -> list[str]:
    found = []
    for m in _DIR_SNAKE.finditer(text or ""):
        found.append(m.group(0))
    for m in _DIR_BARE_RE.finditer(text or ""):
        if m.group(1) in _DIR_BARE:
            found.append(m.group(1))
    return found
