"""Shared question-shape cues for retrieval inspection.

Used by rank / enrich / pack / fit. These detect *what the question asks for*
so inspection can prefer evidence — they are not answer templates or glosses.
"""
from __future__ import annotations

import re

# Role words that signal an office / leadership ask (inspection only).
OFFICE_ROLE = re.compile(
    r"(?i)\b(ceo|chief executive(?: officer)?|president|prime minister|"
    r"pope|emperor)\b"
)
# Slightly narrower set used where pope/emperor are out of scope.
OFFICE_ROLE_CORP = re.compile(
    r"(?i)\b(ceo|chief executive(?: officer)?|president|prime minister)\b"
)
# Words that poison MediaWiki opensearch when left in the query.
SEARCH_ROLE_NOISE = re.compile(
    r"(?i)\b(ceo|cfo|cto|founder|president|prime minister|chairman|"
    r"company|parent|owner)\b"
)

_BIRTHDAY = re.compile(
    r"(?i)\b(birthday|birth date|date of birth)\b"
)
_POPULATION = re.compile(r"(?i)\bpopulation\b")
_WHO_FOUNDED = re.compile(r"(?i)\bwho founded\b")
_CAPITAL = re.compile(r"(?i)\bcapital of\b")
_HOW_OLD = re.compile(r"(?i)\bhow old\b")


def asks_who_office(question: str) -> bool:
    q = question or ""
    return bool(re.search(r"(?i)\bwho is\b", q) and OFFICE_ROLE.search(q))


def asks_who_office_corp(question: str) -> bool:
    q = question or ""
    return bool(re.search(r"(?i)\bwho is\b", q) and OFFICE_ROLE_CORP.search(q))


def asks_office_role(question: str) -> bool:
    return bool(OFFICE_ROLE_CORP.search(question or ""))


def has_office_role(text: str) -> bool:
    return bool(OFFICE_ROLE.search(text or ""))


def has_office_role_corp(text: str) -> bool:
    return bool(OFFICE_ROLE_CORP.search(text or ""))


def asks_birthday(question: str) -> bool:
    return bool(_BIRTHDAY.search(question or "") or _HOW_OLD.search(question or ""))


def asks_birthday_strict(question: str) -> bool:
    """Birthday / birth-date wording only (not how-old)."""
    return bool(_BIRTHDAY.search(question or ""))


def asks_population(question: str) -> bool:
    return bool(_POPULATION.search(question or ""))


def asks_who_founded(question: str) -> bool:
    return bool(_WHO_FOUNDED.search(question or ""))


def asks_capital(question: str) -> bool:
    return bool(_CAPITAL.search(question or ""))


def asks_born_where(question: str) -> bool:
    q = question or ""
    return bool(
        re.search(r"(?i)\bborn\b", q) and re.search(r"(?i)\bwhere\b", q)
    )


def asks_born_when(question: str) -> bool:
    q = question or ""
    return bool(
        re.search(r"(?i)\bwhen\b", q) and re.search(r"(?i)\bborn\b", q)
    )


def asks_when_year(question: str) -> bool:
    """When-ask that is not a birth date ask."""
    q = question or ""
    return bool(
        re.search(r"(?i)\bwhen\b", q) and not re.search(r"(?i)\bborn\b", q)
    )


def asks_founded_when(question: str) -> bool:
    q = question or ""
    return bool(
        re.search(r"(?i)\bfounded\b", q) and re.search(r"(?i)\bwhen\b", q)
    )


def asks_released_when(question: str) -> bool:
    q = question or ""
    return bool(
        re.search(r"(?i)\breleased\b", q) and re.search(r"(?i)\bwhen\b", q)
    )


def asks_chemical_symbol(question: str) -> bool:
    return bool(re.search(r"(?i)\bchemical symbol\b", question or ""))


def asks_what_is_acronym(question: str) -> re.Match[str] | None:
    return re.search(r"(?i)^\s*what is\s+([A-Z]{2,8})\??\s*$", question or "")
