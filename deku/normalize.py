"""English front-door for routing: canonicalize only.

Japanese (and other JA-script) input is not accepted — refuse at route time.
No JA→EN template bridge.
"""
from __future__ import annotations

import re

_JA = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


def looks_japanese(question: str) -> bool:
    """True when the text contains hiragana, katakana, or CJK ideographs."""
    return bool(_JA.search(question or ""))


def prepare_question(question: str) -> tuple[str, dict]:
    """English canonicalize — single front-door for routing.

    Does not translate Japanese. Callers must refuse JA via ``refuse``.
    """
    from deku import canonicalize as can

    q = (question or "").strip()
    q2, d2 = can.canonicalize_question(q)
    detail = dict(d2)
    if d2.get("canonicalized") and "original" not in detail:
        detail["original"] = q
    return q2, detail
