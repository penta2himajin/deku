"""Normalize non-English lookup questions into English before routing.

Product claim stays English-first: Japanese is accepted only via deterministic
template translation, then the existing English route / refuse / tools path.
No Japanese UX promise — untranslated JA still falls through to refuse.
"""
from __future__ import annotations

import re

_JA = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")

# Common place names that appear in JA lookup templates.
_PLACES = {
    "日本": "Japan",
    "東京": "Tokyo",
    "京都": "Kyoto",
    "大阪": "Osaka",
    "アメリカ": "the United States",
    "米国": "the United States",
    "フランス": "France",
    "ドイツ": "Germany",
    "中国": "China",
    "韓国": "South Korea",
    "イギリス": "the United Kingdom",
    "英国": "the United Kingdom",
    "オーストラリア": "Australia",
    "カナダ": "Canada",
    "ペルー": "Peru",
    "ケニア": "Kenya",
}


def looks_japanese(question: str) -> bool:
    return bool(_JA.search(question or ""))


def _place_en(raw: str) -> str:
    s = (raw or "").strip()
    return _PLACES.get(s, s)


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
