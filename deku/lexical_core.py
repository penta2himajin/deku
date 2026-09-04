"""Document-grounded lexical extractors (surface patterns, not slot taxonomy).

Policy (all tools, not only web):
  Allowed — named extractors (``extract_date``, ``extract_person``, …) that
  pull grounded spans from a document via surface patterns; question cues
  choose which extractors to try. Special cases may be formalized as
  generic tools (e.g. ``calc.years_since``) when derivation is needed.

  Forbidden — closed gloss tables; product control via POS / noun-class /
  slot labels; shape-specialized reply shortcuts embedded in a tool
  (templates keyed to president/boiling/age, office-title digs, etc.).
"""
from __future__ import annotations

import re

from deku import extract

_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)
_DATE = (
    rf"(?:"
    rf"\d{{1,2}}\s+{_MONTH}\s+\d{{4}}"
    rf"|{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}"
    rf"|\d{{1,2}}/\d{{1,2}}/\d{{4}}"
    rf")"
)


def _has_person_name(text: str) -> bool:
    return bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text or ""))


def extract_date(question: str, doc: str) -> str | None:
    """Pull a grounded calendar date / year from ``doc``."""
    patterns = (
        rf"(?i)\(born\s+({_DATE})\)",
        rf"(?i)\bborn\s+on\s+({_DATE})\b",
        rf"(?i)\bborn\s+({_DATE})\b",
        rf"(?i)\bbirthday[:\s]+({_DATE})\b",
        rf"(?i)\bdate of birth[:\s]+({_DATE})\b",
        rf"(?i)\bbirth date[:\s]+({_DATE})\b",
    )
    for pat in patterns:
        m = re.search(pat, doc)
        if not m:
            continue
        core = m.group(1).strip().rstrip(".,;")
        if extract.verify(core.split()[0], doc) or extract.verify(core, doc):
            return core
    if re.search(r"(?i)\b(birthday|born|birth)\b", question or ""):
        for line in doc.splitlines():
            if not re.search(r"(?i)\b(born|birthday|birth)\b", line):
                continue
            m = re.search(_DATE, line, flags=re.I)
            if m:
                core = m.group(0).strip()
                if extract.verify(core.split()[0], doc):
                    return core
    if re.search(
        r"(?i)\b(founded|released|unveiled|published|launched|when|landed|landing)\b",
        question or "",
    ):
        for pat in (
            r"(?i)\bfounded\s+(?:in|on)\b.{0,24}\b((?:19|20)\d{2})\b",
            r"(?i)\bfounded\s+on\s+\w+\s+\d{1,2},?\s+(\d{4})",
            r"(?i)\bestablished\s+in\s+(\d{4})",
            r"(?i)\breleased\s+(?:in|on)\b.{0,24}\b((?:19|20)\d{2})\b",
            r"(?i)\bunveiled\b.{0,60}\b((?:19|20)\d{2})\b",
            r"(?i)\blaunched\b.{0,40}\b((?:19|20)\d{2})\b",
            r"(?i)\bpublished\s+(?:in|on)\b.{0,24}\b((?:19|20)\d{2})\b",
            r"(?i)\b(?:landed|landing|touchdown|happened)\b.{0,60}\b((?:19|20)\d{2})\b",
        ):
            m = re.search(pat, doc)
            if m and extract.verify(m.group(1), doc):
                return m.group(1)
    return None


def extract_place(question: str, doc: str) -> str | None:
    """Pull a grounded place name from ``doc``."""
    for pat in (
        r"(?i)\bborn\s+(?:in|at)\s+"
        r"([A-Z][A-Za-z-]+(?:,[ \t]*[A-Z][A-Za-z-]+)?)",
        r"(?i)\bborn\b(?:[^.]{0,80}?)\bin\s+(?:the\s+city\s+of\s+)?"
        r"([A-Z][A-Za-z-]+(?:,[ \t]*[A-Z][A-Za-z-]+)?)",
        r"(?i)\bbirthplace[:\s]+([A-Z][A-Za-z-]+(?:,[ \t]*[A-Z][A-Za-z-]+)?)",
        r"(?i)\braised\s+in\s+"
        r"([A-Z][A-Za-z-]+(?:,[ \t]*[A-Z][A-Za-z-]+)?)",
        r"(?i)\bheadquartered\s+in\s+"
        r"([A-Z][A-Za-z-]+(?:,[ \t]*[A-Z][A-Za-z-]+)?)",
        r"(?i)\bheadquarters\s+(?:are\s+)?in\s+"
        r"([A-Z][A-Za-z-]+(?:,[ \t]*[A-Z][A-Za-z-]+)?)",
        r"(?i)\bbased\s+in\s+"
        r"([A-Z][A-Za-z-]+(?:,[ \t]*[A-Z][A-Za-z-]+)?)",
        r"\b([A-Z][A-Za-z-]+(?:[ \t]+[A-Z][A-Za-z-]+)?)\s+is the capital\b",
        r"(?i)\bcapital (?:city )?of\s+[^.]+?\bis\s+"
        r"([A-Z][A-Za-z-]+(?:[ \t]+[A-Z][A-Za-z-]+)?)\b",
        r"(?i)\blocated\s+in\s+"
        r"([A-Z][A-Za-z-]+(?:,[ \t]*[A-Z][A-Za-z-]+)?)",
    ):
        m = re.search(pat, doc)
        if not m:
            continue
        core = m.group(1).strip().rstrip(".")
        if re.match(
            r"(?i)(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\b",
            core,
        ):
            continue
        if extract.verify(core.split(",")[0].strip(), doc):
            return core
    return None


def extract_person(question: str, doc: str) -> str | None:
    """Pull a grounded person name from ``doc``."""
    title_line = doc.splitlines()[0].strip() if doc else ""
    if (
        re.search(r"(?i)\bwho\b", question or "")
        and re.search(
            r"(?i)\b(emperor|president|prime minister|pope|ceo|chief executive)\b",
            question or "",
        )
        and title_line
        and not re.search(
            r"(?i)\b(official|car|residence|palace|office|building|museum)\b",
            title_line,
        )
        and (
            _has_person_name(title_line)
            or re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", title_line)
        )
        and re.search(
            r"(?i)\b(is Emperor|Emperor of|acceded|throne|president|prime minister|"
            r"chief executive|pope)\b",
            doc,
        )
        and extract.norm(title_line) in extract.norm(doc)
    ):
        return title_line
    # Wiki-style title that embeds a person ("Presidency of X").
    m = re.match(r"(?i)^(?:presidency|premiership) of (.+)$", title_line or "")
    if m and re.search(r"(?i)\bwho\b", question or ""):
        name = m.group(1).strip()
        if _has_person_name(name) and extract.norm(name) in extract.norm(doc):
            return name
    for pat in (
        r"(?i)\bfounded\s+by\s+"
        r"(?-i:([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+(?:[ \t]+and[ \t]+"
        r"[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+)?))",
        r"(?i)\bfounded\s+in\s+\d{4}\s+by\s+"
        r"(?-i:([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+(?:[ \t]+and[ \t]+"
        r"[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+)?))",
        r"(?i)\bco-?founded?\s+(?:by\s+)?"
        r"(?-i:([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+))",
        r"(?i)\b(?:written|authored) by\s+"
        r"(?-i:([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+))\b",
        r"(?i)\b(?:play|tragedy|comedy|novel|dystopian)\s+by\s+"
        r"(?-i:([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+))\b",
        r"(?i)\b(?:the\s+)?(?:CEO|chief executive(?: officer)?|president|"
        r"prime minister)\s+of\s+[^.]{0,40}?\bis\s+"
        r"(?-i:([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+))\b",
        r"\b([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+)\s+is\s+(?:the\s+)?"
        r"(?:CEO|chief executive(?: officer)?|president|prime minister|"
        r"Emperor(?:\s+of\s+[A-Z][A-Za-z ]+)?)\b",
        r"(?-i:([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+))'s\s+"
        r"(?:presidency|premiership)\b",
    ):
        m = re.search(pat, doc)
        if not m:
            continue
        name = m.group(1).strip()
        if re.search(r"(?i)^(who|the|a|an)\b", name):
            continue
        if extract.verify(name.split()[0], doc):
            return name
    if (
        re.search(r"(?i)\bwho founded\b", question or "")
        and re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", title_line or "")
        and re.search(r"(?i)\b(co-?founder|founded)\b", doc)
        and extract.verify(title_line.split()[0], doc)
    ):
        return title_line
    return None


def extract_number(question: str, doc: str) -> str | None:
    """Pull a grounded numeric figure from ``doc``."""
    for pat in (
        r"(?i)\bpopulation\s+was\s+(?:roughly\s+|about\s+|approximately\s+)?"
        r"([\d.,]+\s*(?:million|billion|thousand)?)",
        r"(?i)\bpopulation\s+of\s+the\s+(?:city\s+proper\s+)?"
        r"was\s+(?:over\s+|about\s+)?([\d.,]+\s*(?:million|billion|thousand)?)",
        r"(?i)\bpopulation\s+of\s+(?:about\s+)?([\d.,]+\s*(?:million|billion|thousand)?)",
        r"(?i)\bpopulation[:\s]+(?:of\s+)?(?:about\s+)?([\d.,]+\s*(?:million|billion|thousand)?)",
        r"(?i)\b(?:has|with)\s+a\s+population\s+of\s+(?:about\s+)?"
        r"([\d.,]+\s*(?:million|billion|thousand)?)",
        r"(?i)\b(?:over|about|approximately|roughly)\s+([\d.,]+\s*million)\b",
        r"(?i)\bboils at\s+([\d.,]+)\b",
    ):
        m = re.search(pat, doc)
        if m:
            core = m.group(1).strip().rstrip(".")
            token = core.split()[0]
            if not re.search(rf"(?<![\d.]){re.escape(token)}(?![\d.])", doc):
                continue
            if extract.verify(token, doc):
                return core
    return None


def extract_org(question: str, doc: str) -> str | None:
    """Pull a grounded organization name from ``doc``."""
    for pat in (
        r"(?i)\b(?:developed|manufactured|made|created|produced|owned)"
        r"(?:\s+and\s+(?:developed|manufactured|made|created|produced|owned))?"
        r"\s+by\s+([A-Z][A-Za-z0-9]+)",
        r"(?i)\bsubsidiary of(?:\s+\w+)?\s+([A-Z][A-Za-z0-9]+)",
    ):
        m = re.search(pat, doc)
        if m and extract.verify(m.group(1), doc):
            return m.group(1)
    return None


def extractors_for_question(question: str) -> list:
    """Order extractors to try for ``question`` (cues only; no slot labels)."""
    q = question or ""
    fns: list = []
    if re.search(r"(?i)\bwhere\b.*\bborn\b|\bbirthplace\b", q):
        fns.append(extract_place)
    if re.search(r"(?i)\bbirthday|birth date|born on\b", q):
        fns.append(extract_date)
    if re.search(r"(?i)\bwho\b", q):
        fns.append(extract_person)
    if re.search(r"(?i)\b(population|boiling point|how many|how much|atomic number)\b", q):
        fns.append(extract_number)
    if re.search(r"(?i)\b(what company|makes the|manufacturer)\b", q):
        fns.append(extract_org)
    if re.search(r"(?i)\bwhen\b", q):
        fns.append(extract_date)
    if re.search(r"(?i)\bcapital of|headquarter|based in|located\b", q):
        fns.append(extract_place)
    if re.search(r"(?i)\bwho founded\b", q):
        fns.append(extract_person)
    seen: set = set()
    ordered: list = []
    for fn in fns:
        if fn not in seen:
            seen.add(fn)
            ordered.append(fn)
    for fn in (extract_person, extract_date, extract_place, extract_number, extract_org):
        if fn not in seen:
            ordered.append(fn)
    return ordered


def lexical_core_from_doc(question: str, document: str) -> str | None:
    """Best-effort short core from notes using question-guided extractors."""
    doc = document or ""
    if not doc:
        return None
    q = question or ""
    for fn in extractors_for_question(q):
        core = fn(q, doc)
        if core:
            return core
    return None
