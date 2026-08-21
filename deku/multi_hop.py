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
    r"headquarters?|headquartered|based|population|released|release date"
    r")\b"
)
IT_REF = re.compile(r"(?i)\b(it|its|this|that)\b")


def bind_core(prior_query: str, prior_core: str, followup: str) -> str:
    """Choose the entity to inject into a dependent follow-up.

    Pronoun *he/she* usually want the prior answer (a person). *it* + founded /
    released / headquarters usually want the org named in the prior question.
    """
    fu = followup or ""
    core = (prior_core or "").strip()
    if IT_REF.search(fu) and re.search(
        r"(?i)\b(founded|released|headquarters?|headquartered|population|based)\b",
        fu,
    ):
        for pat in (
            r"(?i)who founded (.+?)\??\s*$",
            r"(?i)(?:ceo|president|prime minister) of (.+?)\??\s*$",
            r"(?i)capital of (.+?)\??\s*$",
            r"(?i)population of (.+?)\??\s*$",
            r"(?i)when (?:was|were) (?:the )?(.+?) released",
            r"(?i)where (?:is|are) (.+?) (?:headquartered|based)",
        ):
            m = re.search(pat, prior_query or "")
            if m:
                return m.group(1).strip().rstrip("?.")
    return core


@dataclass
class Result:
    answer: str | None = None
    status: str = ""
    document: str = ""
    hits: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)


SPLIT = re.compile(
    r"(?i)\s+(?:and\s+)?(?=(?:who|what|when|where|which)\b)"
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
    # "Who is X? What is Y?" style
    parts = re.split(r"\?\s+", q)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return [_normalize_sub(p) for p in parts[:3]]
    # "Who is X and what is Y?"
    bits = SPLIT.split(q)
    bits = [b.strip(" ,;") for b in bits if b.strip(" ,;")]
    if len(bits) < 2:
        return [_normalize_sub(question)]
    out = []
    for b in bits[:3]:
        if not re.match(r"(?i)^(who|what|when|where|which)\b", b):
            # First clause may already include the wh-word.
            if not out:
                out.append(_normalize_sub(b))
            continue
        out.append(_normalize_sub(b))
    return out if len(out) >= 2 else [_normalize_sub(question)]


def looks_multi_hop(question: str) -> bool:
    subs = decompose(question)
    return len(subs) >= 2


def needs_prior(sub: str) -> bool:
    """True when this hop likely refers to the previous answer's entity."""
    s = sub or ""
    if PRONOUN.search(s):
        return True
    if IT_REF.search(s) and ANAPHORA_CUES.search(s):
        return True
    # "Where was born?" is broken English; "where … born" without a name.
    if ANAPHORA_CUES.search(s) and not re.search(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", s
    ):
        return True
    return False


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


def core_from_result(got) -> str | None:
    detail = getattr(got, "detail", None) or {}
    core = (detail.get("core") or "").strip()
    if core and len(core.split()) <= 6:
        return core
    ans = (getattr(got, "answer", None) or "").strip()
    # "The CEO of Apple is Tim Cook." → Tim Cook
    m = re.search(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\.?\s*$", ans
    )
    if m:
        return m.group(1)
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", ans)
    return m.group(1) if m else None


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
