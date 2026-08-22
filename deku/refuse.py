"""Explicit refuse messages and out-of-scope classification.

Router `refuse` must tell the user *why*, not return an empty skip.
Tool-level abstain (`cannot_answer`) stays separate: those tried retrieval
and failed grounding.
"""
from __future__ import annotations

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
)

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
        "I need an explicit file path (for example deku/route.py). "
        "I cannot resolve vague references like \"this part\"."
    ),
    "age": (
        "I cannot answer vague age questions (for example \"how old is he?\"). "
        "Ask with a full name: How old is Tim Cook?"
    ),
    "out_of_scope": (
        "I cannot handle that request with the available tools. "
        "Ask one kind of question at a time "
        "(web fact, repo file, git history, or diff), "
        "or join two questions of the same kind."
    ),
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


def message(reason: str) -> str:
    return MESSAGES.get(reason, MESSAGES["out_of_scope"])


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
