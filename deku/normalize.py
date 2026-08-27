"""Normalize non-English lookup questions into English before routing.

Product claim stays English-first: Japanese is accepted only via deterministic
template translation, then the existing English route / refuse / tools path.
No Japanese UX promise — untranslated JA still falls through to refuse.
"""
from __future__ import annotations

import re

_JA = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


def looks_japanese(question: str) -> bool:
    return bool(_JA.search(question or ""))


# No closed JA→EN place gloss: keep the surface token and let retrieval handle it.
def _place_en(raw: str) -> str:
    return (raw or "").strip()


def normalize_question(question: str) -> tuple[str, dict]:
    """Return (possibly English) question and normalize detail.

    When the input is not Japanese, returns it unchanged with empty detail.
    When Japanese matches a template, returns English + ``normalized_from=ja``.
    When Japanese does not match, returns the original (route will refuse).
    """
    q = (question or "").strip()
    if not q or not looks_japanese(q):
        return q, {}
    detail = {"normalized_from": "ja", "original": q}

    m = re.search(r"(.+?)の首都", q)
    if m:
        place = _place_en(m.group(1))
        return f"What is the capital of {place}?", detail

    m = re.search(r"(?i)(.+?)の\s*CEOは誰", q)
    if m:
        org = m.group(1).strip()
        return f"Who is the CEO of {org}?", detail

    m = re.search(r"(.+?)の人口", q)
    if m:
        place = _place_en(m.group(1))
        return f"What is the population of {place}?", detail

    m = re.search(r"(.+?)の創業者は誰", q)
    if m:
        org = m.group(1).strip()
        return f"Who founded {org}?", detail

    m = re.search(r"(.+?)はいつ設立", q)
    if m:
        org = m.group(1).strip()
        return f"When was {org} founded?", detail

    m = re.search(r"(.+?)の首相は誰", q)
    if m:
        place = _place_en(m.group(1))
        return f"Who is the prime minister of {place}?", detail

    detail["untranslated"] = True
    return q, detail


def prepare_question(question: str) -> tuple[str, dict]:
    """JA normalize then English canonicalize — single front-door for routing."""
    from deku import canonicalize as can

    q, d1 = normalize_question(question)
    q, d2 = can.canonicalize_question(q)
    detail = {**d1, **d2}
    if d1.get("original") and "original" not in d2:
        detail["original"] = d1["original"]
    elif d2.get("canonicalized") and "original" not in detail:
        detail["original"] = (question or "").strip()
    return q, detail
