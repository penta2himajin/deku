"""Typed slot classification and document-grounded extractors.

Division of labour:
  - Rules assign a closed slot label (date|place|person|number|org|role|none).
  - Extractors pull a short core from the document for that slot only.
  - Optional Needle (Needle2 / cactus-needle) may suggest a slot label when
    rules return none — never free-form answers, never tool plans.
  - MiniCPM stays out of slot decisions.
"""
from __future__ import annotations

import re
from typing import Callable

from deku import extract

SLOTS = ("date", "place", "person", "number", "org", "role", "none")

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

DATE_CUES = re.compile(
    r"(?i)\b("
    r"birthday|birth date|date of birth|born on|when (?:was|were|did|is)|"
    r"how old|age of|released|founded|ended"
    r")\b"
)
PLACE_CUES = re.compile(
    r"(?i)\b("
    r"where|located|headquarter|birthplace|based in|capital of"
    r")\b"
)
PERSON_CUES = re.compile(
    r"(?i)\b("
    r"who (?:is|was|are|were)|ceo|president|prime minister|founded by|wrote|author"
    r")\b"
)
NUMBER_CUES = re.compile(
    r"(?i)\b("
    r"population|how many|how much|boiling point|atomic number"
    r")\b"
)
ORG_CUES = re.compile(
    r"(?i)\b("
    r"what company|which company|subsidiary|manufacturer|makes the"
    r")\b"
)
ROLE_CUES = re.compile(
    r"(?i)\b("
    r"what (?:is|are) (?:the )?(?:title|role|position)|job title"
    r")\b"
)


def rule_slot(question: str) -> str:
    """Deterministic closed-label slot for a fact question."""
    q = question or ""
    if not q.strip():
        return "none"
    # More specific first.
    if re.search(r"(?i)\bwhere\b.*\bborn\b|\bborn\b.*\bwhere\b|\bbirthplace\b", q):
        return "place"
    if re.search(r"(?i)\bbirthday|birth date|date of birth|born on\b", q):
        return "date"
    if NUMBER_CUES.search(q):
        return "number"
    if ORG_CUES.search(q):
        return "org"
    if ROLE_CUES.search(q):
        return "role"
    if PERSON_CUES.search(q):
        return "person"
    if PLACE_CUES.search(q):
        return "place"
    if DATE_CUES.search(q):
        return "date"
    return "none"


def needle_slot(question: str) -> str | None:
    """Optional Needle closed-label slot. Returns None if unavailable / unsure."""
    try:
        from needle import Needle, tool, Field
    except ImportError:
        return None

    @tool
    def choose_slot(
        slot: str = Field(
            ...,
            description="One of: date | place | person | number | org | role | none",
        ),
    ):
        """Pick the answer slot type for a short factual question.
        date = birthday, birth date, when/year events.
        place = where / location / headquarters / birthplace.
        person = who / CEO / founder / author.
        number = population / counts / measurements.
        org = which company / manufacturer.
        role = job title / office name.
        none = not a typed factual slot.
        """
        return slot

    try:
        n = Needle(
            tools=[choose_slot],
            system=(
                "Always call choose_slot with exactly one slot label from the "
                "closed set. Never invent free-form answers. "
                "If unsure, choose none."
            ),
        )
        r = n.complete(f"Question: {question}", 32)
    except Exception:
        return None
    calls = r.get("function_calls") or []
    if not calls:
        return None
    raw = str((calls[0].get("arguments") or {}).get("slot") or "").strip().lower()
    if raw in SLOTS:
        return raw
    # Tolerate "slot=date" style.
    m = re.search(
        r"\b(date|place|person|number|org|role|none)\b", raw, flags=re.I
    )
    return m.group(1).lower() if m else None


def classify_slot(
    question: str,
    *,
    use_needle: bool = False,
    needle_fn: Callable[[str], str | None] | None = None,
) -> tuple[str, str]:
    """Return (slot, source). Rules win; Needle only fills rule `none`."""
    slot = rule_slot(question)
    if slot != "none":
        return slot, "rule"
    if use_needle:
        fn = needle_fn or needle_slot
        got = fn(question)
        if got and got in SLOTS:
            return got, "needle"
    return "none", "rule"


def extract_typed(slot: str, question: str, document: str) -> str | None:
    """Pull a short core of the given slot type from document text."""
    doc = document or ""
    if not doc or slot in ("", "none"):
        return None
    if slot == "date":
        return _extract_date(question, doc)
    if slot == "place":
        return _extract_place(question, doc)
    if slot == "person":
        return _extract_person(question, doc)
    if slot == "number":
        return _extract_number(question, doc)
    if slot == "org":
        return _extract_org(question, doc)
    if slot == "role":
        return _extract_role(question, doc)
    return None


def _extract_date(question: str, doc: str) -> str | None:
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
    # Last resort: a clear ISO / full date near birthday cues in the same line.
    if re.search(r"(?i)\b(birthday|born|birth)\b", question or ""):
        for line in doc.splitlines():
            if not re.search(r"(?i)\b(born|birthday|birth)\b", line):
                continue
            m = re.search(_DATE, line, flags=re.I)
            if m:
                core = m.group(0).strip()
                if extract.verify(core.split()[0], doc):
                    return core
    # Event years: founded / released / unveiled (generic, not entity-specific).
    if re.search(r"(?i)\b(founded|released|unveiled|published|launched)\b", question or ""):
        for pat in (
            r"(?i)\bfounded\s+(?:in|on)\b.{0,24}\b((?:19|20)\d{2})\b",
            r"(?i)\breleased\s+(?:in|on)\b.{0,24}\b((?:19|20)\d{2})\b",
            r"(?i)\bunveiled\b.{0,60}\b((?:19|20)\d{2})\b",
            r"(?i)\blaunched\b.{0,40}\b((?:19|20)\d{2})\b",
            r"(?i)\bpublished\s+(?:in|on)\b.{0,24}\b((?:19|20)\d{2})\b",
        ):
            m = re.search(pat, doc)
            if m and extract.verify(m.group(1), doc):
                return m.group(1)
    return None


def _extract_place(question: str, doc: str) -> str | None:
    for pat in (
        r"(?i)\bborn\s+(?:in|at)\s+"
        r"([A-Z][A-Za-z.-]+(?:,\s*[A-Z][A-Za-z.-]+)?)",
        r"(?i)\bheadquartered\s+in\s+"
        r"([A-Z][A-Za-z.-]+(?:,\s*[A-Z][A-Za-z.-]+)?)",
        r"(?i)\bcapital (?:city )?of\s+[^.]+?\bis\s+"
        r"([A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+)?)\b",
        r"(?i)\blocated\s+in\s+"
        r"([A-Z][A-Za-z.-]+(?:,\s*[A-Z][A-Za-z.-]+)?)",
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


def _extract_person(question: str, doc: str) -> str | None:
    for pat in (
        r"(?i)\b(?:the\s+)?(?:CEO|chief executive(?: officer)?|president|"
        r"prime minister)\s+of\s+[^.]{0,40}?\bis\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+is\s+(?:the\s+)?"
        r"(?:CEO|chief executive(?: officer)?|president|prime minister)\b",
        r"(?i)\bfounded by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s+and\s+"
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)?)",
        r"(?i)\bwritten by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
    ):
        m = re.search(pat, doc)
        if not m:
            continue
        name = m.group(1).strip()
        if re.search(r"(?i)^(who|the|a|an)\b", name):
            continue
        if extract.verify(name.split()[0], doc):
            return name
    return None


def _extract_number(question: str, doc: str) -> str | None:
    for pat in (
        r"(?i)\bpopulation\s+of\s+the\s+(?:city\s+proper\s+)?"
        r"was\s+(?:over\s+|about\s+)?([\d.,]+\s*(?:million|billion|thousand)?)",
        r"(?i)\bpopulation\s+of\s+(?:about\s+)?([\d.,]+\s*(?:million|billion)?)",
        r"(?i)\b(?:over|about|approximately)\s+([\d.,]+\s*million)\b",
        r"(?i)\bboils at\s+([\d.,]+)\b",
    ):
        m = re.search(pat, doc)
        if m:
            core = m.group(1).strip().rstrip(".")
            if extract.verify(core.split()[0], doc):
                return core
    return None


def _extract_org(question: str, doc: str) -> str | None:
    for pat in (
        r"(?i)\b(?:developed|manufactured|made|created|produced|owned)\s+by\s+"
        r"([A-Z][A-Za-z0-9]+)",
        r"(?i)\bsubsidiary of(?: Japanese conglomerate)?\s+([A-Z][A-Za-z0-9]+)",
    ):
        m = re.search(pat, doc)
        if m and extract.verify(m.group(1), doc):
            return m.group(1)
    return None


def _extract_role(question: str, doc: str) -> str | None:
    m = re.search(
        r"(?i)\bis\s+the\s+((?:CEO|chief executive(?: officer)?|president|"
        r"prime minister)(?:\s+of\s+[^.]+)?)",
        doc,
    )
    if m:
        core = m.group(1).strip().rstrip(".")
        if extract.verify(core.split()[0], doc):
            return core
    return None


def typed_reply(
    slot: str,
    question: str,
    core: str,
    document: str,
) -> str | None:
    """Thin NL wrapper when a registered template does not match."""
    c = (core or "").strip()
    if not c or not document:
        return None
    if extract.norm(c) not in extract.norm(document):
        return None
    q = (question or "").strip()

    if slot == "date":
        m = re.search(
            r"(?i)^\s*what is the birthday of (?:the )?(.+?)\??\s*$", q
        )
        if m:
            who = m.group(1).strip().rstrip("?.")
            return f"The birthday of {who} is {c}."
        m = re.search(r"(?i)^\s*what is (.+?)(?:'s)? birthday\??\s*$", q)
        if m:
            who = m.group(1).strip().rstrip("'s").strip()
            return f"The birthday of {who} is {c}."
        m = re.search(r"(?i)^\s*when (?:was|were) (.+?) born\??\s*$", q)
        if m:
            who = m.group(1).strip().rstrip("?.")
            return f"{who} was born on {c}."
        return None

    if slot == "place":
        m = re.search(r"(?i)^\s*where (?:was|is) (.+?) born\??\s*$", q)
        if m:
            return f"{m.group(1).strip().rstrip('?.')} was born in {c}."
        return None

    if slot == "number":
        m = re.search(r"(?i)^\s*what is the population of (.+?)\??\s*$", q)
        if m:
            return f"The population of {m.group(1).strip().rstrip('?.')} is {c}."
        return None

    return None


def entity_hint(question: str) -> str | None:
    """Best-effort entity string for search expansion (not the answer)."""
    q = question or ""
    m = re.search(
        r"(?i)(?:birthday|birth date|date of birth)\s+of\s+(?:the\s+)?(.+?)\??\s*$",
        q,
    )
    if m:
        return m.group(1).strip().rstrip("?.")
    m = re.search(r"(?i)where (?:was|is) (.+?) born", q)
    if m:
        return m.group(1).strip().rstrip("?.")
    m = re.search(r"(?i)who is the (?:ceo|president|prime minister) of (.+?)\??\s*$", q)
    if m:
        return m.group(1).strip().rstrip("?.")
    return None
