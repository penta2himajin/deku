"""Collapse English paraphrases into canonical lookup forms.

Flexibility at the front door: many surface forms map to one typed question
the existing route / query / template machinery already handles. No model.
"""
from __future__ import annotations

import re


def _clean_entity(raw: str) -> str | None:
    entity = (raw or "").strip().rstrip(".,")
    entity = re.sub(r"(?i)^(the|a|an)\s+", "", entity).strip()
    if not entity or "?" in entity:
        return None
    if re.search(r"(?i)\b(also|and|then)\b", entity):
        return None
    if len(entity) > 60:
        return None
    return entity


def canonicalize_question(question: str) -> tuple[str, dict]:
    """Return (canonical question, detail). Unchanged when no rule matches."""
    q = (question or "").strip()
    if not q:
        return q, {}
    # Leave joined / multi-clause questions alone (decomposer owns those).
    if re.search(r"\?", q.rstrip("?")):
        return q, {}
    if re.search(r"(?i)(?:\balso\b|,?\s+then\s+|;)", q):
        return q, {}

    m = re.match(
        r"(?i)^who\s+(?:currently\s+)?(?:runs|leads|heads)\s+(.+?)\s+"
        r"as\s+(?:the\s+)?(?:chief\s+executive(?:\s+officer)?|ceo)\s*\??\s*$",
        q,
    )
    if m:
        entity = _clean_entity(m.group(1))
        if entity:
            return (
                f"Who is the CEO of {entity}?",
                {"canonicalized": True, "canon_rule": "runs_as_ceo"},
            )

    m = re.match(
        r"(?i)^who\s+is\s+(?:the\s+)?(?:current\s+)?"
        r"(?:chief\s+executive(?:\s+officer)?|ceo)\s+(?:of\s+)?(.+?)\s*\??\s*$",
        q,
    )
    if m:
        entity = _clean_entity(m.group(1))
        if entity and not re.search(r"(?i)^ceo\b", entity):
            return (
                f"Who is the CEO of {entity}?",
                {"canonicalized": True, "canon_rule": "ceo_of"},
            )

    m = re.match(
        r"(?i)^(.+?)'s\s+(?:ceo|chief\s+executive(?:\s+officer)?)"
        r"\s*[-–—,:]?\s*who\s+is\s+it\s*\??\s*$",
        q,
    )
    if m:
        entity = _clean_entity(m.group(1))
        if entity:
            return (
                f"Who is the CEO of {entity}?",
                {"canonicalized": True, "canon_rule": "possessive_ceo"},
            )

    m = re.match(
        r"(?i)^what\s+is\s+([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+)?)'s\s+"
        r"population\s*\??\s*$",
        q,
    )
    if m:
        place = m.group(1).strip()
        return (
            f"What is the population of {place}?",
            {"canonicalized": True, "canon_rule": "possessive_population"},
        )

    m = re.match(
        r"(?i)^what\s+is\s+([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+)?)'s\s+"
        r"capital\s*\??\s*$",
        q,
    )
    if m:
        place = m.group(1).strip()
        return (
            f"What is the capital of {place}?",
            {"canonicalized": True, "canon_rule": "possessive_capital"},
        )

    return q, {}
