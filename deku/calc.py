"""Structured calculation tool (not free-form math).

Allowed ops are explicit and narrow (e.g. ``years_since``). Free-form
algebra stays ``refuse:math``. Used by weak multi-hop plans such as
named age: web_search (birth date) → calc (years_since).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

_MONTHS = {
    name.lower(): i
    for i, name in enumerate(
        (
            "January February March April May June July August "
            "September October November December"
        ).split(),
        start=1,
    )
}


@dataclass
class Result:
    answer: str | None = None
    status: str = ""
    document: str = ""
    hits: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)


def parse_date(text: str) -> date | None:
    """Parse a short prose or ISO calendar date."""
    s = (text or "").strip().rstrip(".")
    if not s:
        return None
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.search(r"(?i)^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$", s)
    if m:
        day, mon, year = int(m.group(1)), _MONTHS.get(m.group(2).lower()), int(m.group(3))
    else:
        m = re.search(r"(?i)^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$", s)
        if not m:
            return None
        mon, day, year = _MONTHS.get(m.group(1).lower()), int(m.group(2)), int(m.group(3))
    if not mon or not (1 <= day <= 31):
        return None
    try:
        return date(year, mon, day)
    except ValueError:
        return None


def years_since(birth: str, *, today: date | None = None) -> int | None:
    """Whole years from a calendar date to ``today`` (default: date.today())."""
    born = parse_date(birth)
    if not born:
        return None
    now = today or date.today()
    years = now.year - born.year
    if (now.month, now.day) < (born.month, born.day):
        years -= 1
    return years if years >= 0 else None


def run(query: str, *, today: date | None = None, seed: int = 0, root: str = ".") -> Result:
    """Execute one structured calc query.

    Recognized forms:
      ``years_since: <date>``
      ``years_since <date>``
    """
    del seed, root  # API parity with other tools
    out = Result(detail={"tool": "calc"})
    q = (query or "").strip()
    m = re.match(r"(?i)^years_since\s*:?\s+(.+)$", q)
    if not m:
        out.status = "cannot_answer"
        out.answer = "I cannot answer from the available sources."
        out.detail["abstain_reason"] = "unknown_calc_op"
        return out
    raw_date = m.group(1).strip()
    years = years_since(raw_date, today=today)
    out.detail["op"] = "years_since"
    out.detail["input"] = raw_date
    if years is None:
        out.status = "cannot_answer"
        out.answer = "I cannot answer from the available sources."
        out.detail["abstain_reason"] = "unparseable_date"
        return out
    core = str(years)
    out.detail["core"] = core
    out.answer = core
    out.status = "ok"
    return out
