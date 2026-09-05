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
_WHO_WROTE = re.compile(r"(?i)^\s*who wrote (.+?)\??\s*$")
_CAPITAL = re.compile(r"(?i)\bcapital of\b")
_HOW_OLD = re.compile(r"(?i)\bhow old\b")
_HQ = re.compile(r"(?i)\b(headquarters?|headquartered|based)\b")
_MAKER = re.compile(r"(?i)what company makes (?:the )?(.+?)\??\s*$")
_MONTH_NAME = re.compile(
    r"(?i)^(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b"
)
_ONOMATIC = re.compile(r"(?i)\((given name|surname|name)\)")
# Officeholder birthday asks (emperor/president/…), including "current".
_OFFICEHOLDER_BIRTHDAY_Q = re.compile(
    r"(?i)\b(current|emperor|empress|president|prime minister|pope)\b"
)
_WHO_PERSON = re.compile(
    r"(?i)\bwho\b.+\b(ceo|chief executive|prime minister|president|pope|emperor|"
    r"founded|wrote)\b|\bwho\s+is\s+the\s+(ceo|prime minister|president)\b"
)


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


def asks_who_wrote(question: str) -> re.Match[str] | None:
    return _WHO_WROTE.search(question or "")


def asks_who_person(question: str) -> bool:
    """Who-asks that expect a person core (office / founded / wrote)."""
    return bool(_WHO_PERSON.search(question or ""))


def asks_capital(question: str) -> bool:
    return bool(_CAPITAL.search(question or ""))


def asks_hq(question: str) -> bool:
    return bool(_HQ.search(question or ""))


def asks_maker(question: str) -> re.Match[str] | None:
    return _MAKER.search(question or "")


def asks_officeholder_birthday(question: str) -> bool:
    """Birthday ask about an officeholder (not a bare personal birthday)."""
    q = question or ""
    return bool(asks_birthday_strict(q) and _OFFICEHOLDER_BIRTHDAY_Q.search(q))


def looks_month_name(text: str) -> bool:
    return bool(_MONTH_NAME.search(text or ""))


def looks_onomastic_title(title: str) -> bool:
    return bool(_ONOMATIC.search(title or ""))


def looks_holiday_observance(text: str, title: str = "") -> bool:
    """True for public-holiday / 'The X's birthday' noise pages."""
    if re.search(r"(?i)\bpublic holiday\b", text or ""):
        return True
    if re.match(r"(?i)^the .+'s birthday$", (title or "").strip()):
        return True
    return False


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


# ---- tenure inspection (office asks; not answer templates) ----------------

_OFFICE_INNER = (
    r"(?:ceo|chief executive(?: officer)?|president|prime minister|"
    r"pope|emperor)"
)


def looks_past_tenure(text: str) -> bool:
    """True when text describes a former / dated past office-holder."""
    t = text or ""
    if re.search(
        rf"(?i)\b(former|previously served|first {_OFFICE_INNER}|"
        rf"was the (?:first )?{_OFFICE_INNER})\b",
        t,
    ):
        return True
    if re.search(
        r"(?i)\bfrom\s+(?:[A-Za-z]+\s+)?\d{4}\s+to\s+(?:[A-Za-z]+\s+)?\d{4}\b",
        t,
    ):
        return True
    if re.search(r"(?i)\buntil\s+(?:[A-Za-z]+\s+)?\d{4}\b", t):
        return True
    return False


def looks_present_tenure(text: str) -> bool:
    """True when text signals incumbent / present tenure."""
    t = text or ""
    if looks_past_tenure(t) and not re.search(r"(?i)\bsince\s+20\d{2}\b", t):
        return False
    return bool(
        re.search(
            r"(?i)\b(current|incumbent|has served as|serving as|has been the|"
            r"since\s+20\d{2})\b",
            t,
        )
    )
