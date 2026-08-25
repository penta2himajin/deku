"""Explicit refuse messages and out-of-scope classification.

Router `refuse` must tell the caller *why*, not return an empty skip.
Tool-level abstain (`cannot_answer`) stays separate: those tried retrieval
and failed grounding.

`audience=\"human\"` → full English prose; `audience=\"agent\"` → short reason codes
for parent agents / `--json` callers.
"""
from __future__ import annotations

import os
import re

# Reasons surface in Routed.detail["reason"] and pick a fixed English line.
REASONS = (
    "math",
    "code",
    "chitchat",
    "deep_reasoning",
    "underspecified",
    "age",
    "out_of_scope",
    "no_plan",
)

AUDIENCES = ("human", "agent")

MESSAGES = {
    "math": (
        "I cannot help with math or calculation questions."
    ),
    "code": (
        "I cannot write or implement code. "
        "Ask about existing project files, git history, or diffs instead."
    ),
    "chitchat": (
        "I only answer factual lookup questions "
        "(public web facts, this repository, git history, diffs, or a given URL)."
    ),
    "deep_reasoning": (
        "I cannot do open-ended reasoning, proofs, or long analysis. "
        "Ask a short factual question, summarize a specific URL or document, "
        "or ask two linked factual questions for a shallow multi-hop lookup."
    ),
    "underspecified": (
        "I need an explicit file path (for example path/to/file.py). "
        "I cannot resolve vague references like \"this part\"."
    ),
    "age": (
        "I cannot answer vague age questions (for example \"how old is he?\"). "
        "Ask with a full name: How old is Tim Cook?"
    ),
    "out_of_scope": (
        "I cannot handle that request with the available tools. "
        "Ask clear factual questions (web fact, repo file, git history, or diff), "
        "or join up to three such questions with 'and who/what/when…'. "
        "Opinion and open-ended judgment are out of scope."
    ),
    "no_plan": (
        "I could not build a multi-step plan for that. "
        "Ask one short factual question, or join two clear questions "
        "with 'and what/who/when…'."
    ),
}

# Compact codes for parent agents (stable; do not paraphrase).
AGENT_MESSAGES = {
    "math": "refused:math",
    "code": "refused:code",
    "chitchat": "refused:chitchat",
    "deep_reasoning": "refused:deep_reasoning",
    "underspecified": "refused:underspecified",
    "age": "refused:age",
    "out_of_scope": "refused:out_of_scope",
    "no_plan": "refused:no_plan",
}

# Hard cues — win over Needle / soft web routing.
MATH = re.compile(
    r"(?i)^\s*(what is\s+)?(\d|compute|calculate|2\s*\+|solve\s+\d|"
    r"\d+\s*[\+\-\*/]\s*\d)"
)
CODE = re.compile(
    r"(?i)("
    r"^\s*(write a |implement |fn |def |class |code (?:me |a )|"
    r"sort the |implement a |create a function)"
    r"|"
    r"\bfix (?:the |a |this )?bug\b"
    r"|"
    r"\bfix .{0,60}\.(?:py|js|ts|go|rs)\b"
    r"|"
    r"\b(please |can you )?fix\b.{0,40}\b(bug|error|issue)\b"
    r")"
)
CHITCHAT = re.compile(
    r"(?i)^\s*(hello|hi there|hey\b|how are you|good (?:morning|evening)|"
    r"thanks|thank you|lol|what's up)\b"
)
DEEP = re.compile(
    r"(?i)("
    r"\b(step[- ]by[- ]step|in detail|at length|essay|prove that|"
    r"elaborate|thorough(?:ly)? analysis|deep dive)\b|"
    r"\b(why (?:is|are|does|do|did)|explain why)\b.{"
    r"40,}|"  # long why/explain — short "why" facts still OK via other tools
    r"\bcompare (?:and contrast )?.{10,}\b(and|with|versus|vs\.?)\b|"
    r"\b(discuss|argue|critique)\b.{20,}"
    r")"
)
AGE = re.compile(r"(?i)\bhow old\b")
# Named person age is a searchable fact; vague / anaphoric age stays refused.
AGE_NAMED = re.compile(
    r"(?i)^\s*how old (?:is|are) "
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\??\s*$"
)

# Path-scoped git history without a concrete path (deixis or bare "that changed").
_PATH_HISTORY = re.compile(
    r"(?i)\b("
    r"(?:last )?commit that (?:changed|touched|modified)|"
    r"commit (?:log|message)? (?:of )?(?:the )?(?:last )?commit that|"
    r"(?:changed|touched|modified) this (?:part|file|code|section|bit)|"
    r"this part|"
    r"who last (?:changed|touched|modified)"
    r")\b"
)
_REPO_PATH = re.compile(
    r"\b((?:[\w.-]+/)*[\w.-]+\.(?:py|md|json|txt|toml|sh|ya?ml|csv))\b"
)


def is_underspecified_path(question: str) -> bool:
    """True when a path-scoped git ask lacks an explicit file path."""
    q = question or ""
    if _REPO_PATH.search(q):
        return False
    return bool(_PATH_HISTORY.search(q))


def classify(question: str) -> str:
    """Return a refuse reason for an out-of-scope question."""
    q = question or ""
    if MATH.search(q):
        return "math"
    if CODE.search(q):
        return "code"
    if CHITCHAT.search(q):
        return "chitchat"
    if AGE.search(q):
        if not AGE_NAMED.match(q):
            return "age"
    if DEEP.search(q):
        return "deep_reasoning"
    if is_underspecified_path(q):
        return "underspecified"
    return "out_of_scope"


def message(reason: str, *, audience: str | None = None) -> str:
    """Human prose or agent reason-code, depending on audience.

    Audience resolution order: explicit arg → ``DEKU_AUDIENCE`` → ``human``.
    """
    aud = (audience or os.environ.get("DEKU_AUDIENCE") or "human").strip().lower()
    if aud not in AUDIENCES:
        aud = "human"
    key = reason if reason in MESSAGES else "out_of_scope"
    if aud == "agent":
        return AGENT_MESSAGES.get(key, AGENT_MESSAGES["out_of_scope"])
    return MESSAGES[key]


def is_hard_refuse(question: str) -> bool:
    """True when hard_route should pick refuse (math/code/chitchat/deep/vague age).

    Underspecified path asks go to clarify instead of refuse.
    Named \"How old is Tim Cook?\" is allowed through to web_search.
    """
    q = question or ""
    return bool(
        MATH.search(q)
        or CODE.search(q)
        or CHITCHAT.search(q)
        or (AGE.search(q) and not AGE_NAMED.match(q))
        or DEEP.search(q)
    )
