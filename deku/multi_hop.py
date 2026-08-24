"""Weak multi-step lookup: code plans hops; MiniCPM only answers each hop.

Not model chain-of-thought. The orchestrator:
  1. Splits joined factual questions (rules).
  2. Runs each hop with web_search (or an injected runner).
  3. If a later hop uses a pronoun, rewrites it with the prior hop's core
     ("where was he born?" + Tim Cook → "Where was Tim Cook born?").
  4. Integrates: independent hops → numbered list; dependent → one short
     paragraph from the hop answers (no free-form reasoning).

If any hop abstains, the whole episode abstains with an explicit message.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from deku import web_search as ws

RunFn = Callable[..., ws.Result]

PRONOUN = re.compile(
    r"\b(he|she|they|him|her|his|their|them|this person|that person)\b",
    re.I,
)
# Follow-ups that usually need the prior entity even without a pronoun.
ANAPHORA_CUES = re.compile(
    r"(?i)\b("
    r"born|birthplace|age|nationality|founded|died|spouse|wife|husband|"
    r"headquarters?|headquartered|based|population|released|release date|"
    r"published"
    r")\b"
)
IT_REF = re.compile(r"(?i)\b(it|its|this|that)\b")


def has_concrete_topic(sub: str) -> bool:
    """True when the clause already names its own subject entity."""
    s = sub or ""
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", s):
        return True
    # "the iPhone", "the population of Tokyo", "of France"
    if re.search(r"(?i)\b(?:of|for)\s+(?:the\s+)?[A-Z][A-Za-z0-9.-]+\b", s):
        return True
    if re.search(
        r"(?i)\b(?:the\s+)?(iPhone|iPad|iPod|PlayStation|Xbox|Android)\b", s
    ):
        return True
    if re.search(
        r"^(?:when|where|what|who)\b.+\b(?:the\s+)?[A-Z][A-Za-z0-9.-]+\b",
        s,
        flags=re.I,
    ) and re.search(r"\b[A-Z][A-Za-z0-9.-]+\b", s) and not IT_REF.search(s) and not PRONOUN.search(s):
        # Require a true capitalised token (re.I alone would match "born").
        caps = re.findall(r"\b[A-Z][A-Za-z0-9.-]+\b", s)
        wh = re.findall(r"(?i)\b(?:when|where|what|who|the)\b", s)
        if any(t.casefold() not in {w.casefold() for w in wh} for t in caps):
            return True
    return False


def needs_prior(sub: str) -> bool:
    """True when this hop likely refers to the previous answer's entity."""
    s = sub or ""
    if PRONOUN.search(s):
        return True
    if IT_REF.search(s) and ANAPHORA_CUES.search(s):
        return True
    if has_concrete_topic(s):
        return False
    # "Where was born?" is broken English; thin cue without a name.
    if ANAPHORA_CUES.search(s) and not re.search(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", s
    ):
        return True
    return False


def bind_core(prior_query: str, prior_core: str, followup: str) -> str:
    """Choose the entity to inject into a dependent follow-up.

    Pronoun *he/she* usually want the prior answer (a person). *it/its* + founded /
    released / published / headquarters usually want the org/place / work named in
    the prior question — never a bare numeric core. *its population* after a
    capital question prefers the capital city (prior answer) when available.
    """
    fu = followup or ""
    core = (prior_core or "").strip()
    if IT_REF.search(fu) and re.search(r"(?i)\bpopulation\b", fu):
        # Capital-of X → its population: bind the capital city answer.
        if re.search(r"(?i)\bcapital of\b", prior_query or "") and core:
            if not re.fullmatch(
                r"[\d.,]+(?:\s*(?:million|billion|thousand))?", core, flags=re.I
            ):
                return core
        for pat in (
            r"(?i)population of (.+?)\??\s*$",
            r"(?i)capital of (.+?)\??\s*$",
        ):
            m = re.search(pat, prior_query or "")
            if m and not re.search(r"(?i)\bcapital of\b", prior_query or ""):
                return m.group(1).strip().rstrip("?.")
    if IT_REF.search(fu) and re.search(
        r"(?i)\b(founded|released|published|headquarters?|headquartered|"
        r"based)\b",
        fu,
    ):
        for pat in (
            r"(?i)who founded (.+?)\??\s*$",
            r"(?i)who wrote (.+?)\??\s*$",
            r"(?i)(?:ceo|president|prime minister) of (.+?)\??\s*$",
            r"(?i)when (?:was|were) (?:the )?(.+?) founded",
            r"(?i)when (?:was|were) (?:the )?(.+?) (?:released|published)",
            r"(?i)where (?:is|are) (.+?) (?:headquartered|based)",
            r"(?i)headquarters (?:city |of )?(.+?)\??\s*$",
        ):
            m = re.search(pat, prior_query or "")
            if m:
                return m.group(1).strip().rstrip("?.")
        # Never bind a year / count into who-founded follow-ups.
        if re.fullmatch(
            r"[\d.,]+(?:\s*(?:million|billion|thousand))?", core, flags=re.I
        ):
            return core  # still wrong; caller should not need_prior — defensive
    if PRONOUN.search(fu):
        # Prefer person-shaped prior core; strip template wrappers if present.
        m = re.search(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", core
        )
        if m and not re.search(r"(?i)\b(million|billion|inc|corp|current)\b", m.group(1)):
            return m.group(1)
    return core


@dataclass
class Result:
    answer: str | None = None
    status: str = ""
    document: str = ""
    hits: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)


SPLIT = re.compile(
    r"\s+(?:and\s+|then\s+|also,?\s+)?(?=(?:who|what|when|where|which)\b)"
    r"|\s+and\s+(?=(?:should|do you|is that|are they|would you|could you)\b)"
    r"|\s*;\s*(?=(?:who|what|when|where|which)\b)"
    r"|,\s+then\s+(?=(?:who|what|when|where|which)\b)"
    r"|\?\s*also,?\s+(?=(?:who|what|when|where|which|is|are|was|were|should|do)\b)",
    re.I,
)


def _normalize_sub(text: str) -> str:
    s = (text or "").strip().strip(" ,;")
    if not s:
        return ""
    if not s.endswith("?"):
        s += "?"
    # MiniCPM / templates are brittle on leading lowercase wh-words.
    return re.sub(
        r"^(who|what|when|where|which)\b",
        lambda m: m.group(1).capitalize(),
        s,
        count=1,
        flags=re.I,
    )


def decompose(question: str) -> list[str]:
    """Split a joined factual question into sub-questions (max 3)."""
    q = (question or "").strip().rstrip("?")
    if not q:
        return []
    # "Who is X? What is Y?" / "Who is X? Also, …"
    parts = re.split(r"\?\s+(?:also,?\s+)?", q, flags=re.I)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return [_normalize_sub(p) for p in parts[:3]]
    # "Who is X and what is Y?" / "… then what …"
    bits = SPLIT.split(q)
    bits = [b.strip(" ,;") for b in bits if b.strip(" ,;")]
    if len(bits) < 2:
        return [_normalize_sub(question)]
    out = []
    for b in bits[:3]:
        if not re.match(
            r"(?i)^(who|what|when|where|which|is|are|should|do|would|could)\b",
            b,
        ):
            # First clause may already include the wh-word.
            if not out:
                out.append(_normalize_sub(b))
            continue
        out.append(_normalize_sub(b))
    return out if len(out) >= 2 else [_normalize_sub(question)]


def looks_multi_hop(question: str) -> bool:
    subs = decompose(question)
    return len(subs) >= 2


def rewrite_followup(sub: str, prior_core: str) -> str:
    """Bind pronouns / thin follow-ups to the prior hop's core name."""
    core = (prior_core or "").strip()
    s = (sub or "").strip()
    if not s:
        return s
    if not core or not needs_prior(s):
        return _normalize_sub(s)
    if PRONOUN.search(s):
        # Prefer full-name substitution for subject pronouns.
        out = re.sub(r"\b(he|she|they)\b", core, s, count=1, flags=re.I)
        out = re.sub(r"\b(him|her|them)\b", core, out, count=1, flags=re.I)
        out = re.sub(r"\b(his|their)\b", f"{core}'s", out, count=1, flags=re.I)
        out = re.sub(
            r"\b(this person|that person)\b", core, out, count=1, flags=re.I
        )
        return _normalize_sub(out)
    if IT_REF.search(s) and ANAPHORA_CUES.search(s):
        # Prefer shapes that expand_search / templates already handle.
        if re.search(r"(?i)\bpopulation\b", s):
            return _normalize_sub(f"What is the population of {core}?")
        if re.search(r"(?i)\bheadquarters?\b|\bheadquartered\b|\bbased\b", s):
            return _normalize_sub(f"Where is {core} headquartered?")
        if re.search(r"(?i)\bpublished\b", s):
            return _normalize_sub(f"When was {core} published?")
        if re.search(r"(?i)\bwho\b", s) and re.search(r"(?i)\bfounded\b", s):
            return _normalize_sub(f"Who founded {core}?")
        if re.search(r"(?i)\bwhen\b", s) and re.search(r"(?i)\bfounded\b", s):
            return _normalize_sub(f"When was {core} founded?")
        if re.search(r"(?i)\breleased\b", s):
            return _normalize_sub(f"When was {core} released?")
        out = re.sub(r"\b(it|this|that)\b", core, s, count=1, flags=re.I)
        out = re.sub(r"\b(its)\b", f"{core}'s", out, count=1, flags=re.I)
        return _normalize_sub(out)
    # No pronoun but anaphoric cue — insert the core after the wh-word.
    out = re.sub(
        r"(?i)^(where|when|who|what|which)\b\s+",
        lambda m: f"{m.group(1).capitalize()} {core} ",
        s,
        count=1,
    )
    return _normalize_sub(out)


def _reject_office_glue(name: str | None) -> str | None:
    """Drop cores that are office titles, not people (e.g. Current Pope)."""
    n = (name or "").strip()
    if not n:
        return None
    if re.search(
        r"(?i)^(current|the)\s+(pope|president|prime minister|emperor|ceo)\b",
        n,
    ):
        return None
    if re.fullmatch(r"(?i)current pope|pope|president|prime minister", n):
        return None
    return n


def core_from_result(got) -> str | None:
    detail = getattr(got, "detail", None) or {}
    core = _reject_office_glue((detail.get("core") or "").strip())
    # Drop sentence bleed ("Tokyo. Throughout") before binding.
    if core and re.search(r"\.\s+[A-Za-z]", core):
        core = core.split(".", 1)[0].strip()
    if core and len(core.split()) <= 6:
        return core
    ans = (getattr(got, "answer", None) or "").strip()
    # "The capital of Japan is Tokyo." → Tokyo (stop at sentence end)
    m = re.search(
        r"(?i)\b(?:capital of|population of)\s+[^.]{0,40}?\bis\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*\.",
        ans,
    )
    if m:
        return _reject_office_glue(m.group(1))
    # "The CEO of Apple is Tim Cook." → Tim Cook
    m = re.search(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\.?\s*$", ans
    )
    if m:
        return _reject_office_glue(m.group(1))
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", ans)
    return _reject_office_glue(m.group(1) if m else None)


def _core_from_result(got) -> str | None:
    """Backward-compatible alias."""
    return core_from_result(got)

def integrate(
    hops: list[tuple[str, str]],
    *,
    dependent: bool,
) -> str:
    """Join hop answers. Dependent → one paragraph; else numbered list."""
    if not hops:
        return ""
    if not dependent or len(hops) == 1:
        return "\n".join(f"{i + 1}. {a}" for i, (_, a) in enumerate(hops))
    parts = []
    for _, ans in hops:
        s = (ans or "").strip()
        if not s:
            continue
        if not s.endswith((".", "!", "?")):
            s += "."
        parts.append(s)
    return " ".join(parts)


def run(
    question: str,
    *,
    seed: int = 0,
    web_run: RunFn | None = None,
    root: str = ".",
) -> Result:
    """Delegate to the A+B orchestrator (catalog select + execute)."""
    from deku import orchestrate as orch

    runners = None
    if web_run is not None:
        def _web(q, seed=0, root=".", **kw):
            return web_run(q, seed=seed)
        runners = {"web_search": _web}
    got = orch.run(question, seed=seed, root=root, runners=runners)
    return Result(
        answer=got.answer,
        status=got.status,
        document=got.document,
        hits=got.hits,
        detail=got.detail,
    )
