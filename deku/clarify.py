"""Ask one English clarifying question when a lookup is missing a key slot.

Not a planner. Used when the user clearly wants a supported tool but omitted
a required concrete handle (file path, URL, …).
"""

from __future__ import annotations

import re

from deku import refuse as refuse_mod
from deku import url_read as ur

_SUMMARIZE = re.compile(
    r"(?i)\b(summarize|summarise|summary of|tl;dr|tldr)\b"
)
_DOC_DEIXIS = re.compile(
    r"(?i)\b(this (?:document|page|article|text|file)|the (?:document|page|article))\b"
)

QUESTIONS = {
    "path": (
        "Which file path should I look at? "
        "For example: path/to/file.py"
    ),
    "url": (
        "Which URL should I read? "
        "Please include a full http(s) link."
    ),
}


def detect(question: str) -> str | None:
    """Return a clarify kind, or None when no clarifying question is needed."""
    q = question or ""
    if refuse_mod.MATH.search(q) or refuse_mod.CODE.search(q):
        return None
    if refuse_mod.CHITCHAT.search(q) or refuse_mod.DEEP.search(q):
        return None
    if refuse_mod.is_underspecified_path(q):
        return "path"
    if _SUMMARIZE.search(q) and not ur.extract_url(q) and _DOC_DEIXIS.search(q):
        return "url"
    if _SUMMARIZE.search(q) and not ur.extract_url(q) and re.search(
        r"(?i)\b(this|that|it)\b", q
    ):
        return "url"
    return None


def question_for(question: str) -> str:
    kind = detect(question) or "path"
    return QUESTIONS.get(kind, QUESTIONS["path"])
