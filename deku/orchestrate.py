"""Weak multi-step plans via propose + validate (code is the planner).

Not model planning. Clauses are classified in code; a plan is accepted only
when it passes explicit invariants (tool allow-list, length, bind rules).
Legacy catalog builders remain for tests / telemetry labels when shapes match.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from deku import multi_hop as mh
from deku import refuse as refuse_mod
from deku import route_cues as cues
from deku import web_search as ws

TOOL_OK = frozenset({
    "web_search", "dir_search", "git_search", "diff_search", "url_read",
})
MAX_STEPS = 3

DIR_WORDS = cues.DIR_WORDS
GIT_CUES = re.compile(
    r"(?i)\b(last commit|git (?:log|show|blame|history)|who (?:changed|committed)|"
    r"who authored|authored (?:the )?(?:last )?commit|author of|"
    r"when was .+ committed|commit message|git history|"
    r"commit (?:that )?(?:changed|touched|modified))\b"
)
DIFF_CUES = re.compile(
    r"(?i)\b(unstaged|staged|working tree|git diff|what changed in .+\.py|"
    r"patch for|unstaged diff)\b"
)
WEB_CUES = ws.SEARCH_CUES
JOIN_CUES = re.compile(
    r"(?i)\b("
    r"and (?:who|what|when|where|which)\b|"
    r"\?\s+(?:and\s+)?(?:who|what|when|where|which)\b|"
    r"and (?:what|show|list)\b"
    r")"
)


@dataclass
class Step:
    tool: str
    query: str
    bind_prior: bool = False


@dataclass
class Plan:
    plan_id: str
    steps: list[Step]
    dependent: bool = False


@dataclass
class Result:
    answer: str | None = None
    status: str = ""
    document: str = ""
    hits: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)


def classify_clause(clause: str) -> str | None:
    """Map one clause to a single retrieval tool, or None if unclear."""
    c = clause or ""
    if refuse_mod.is_hard_refuse(c):
        return None
    # Opinion / evaluative tails are not retrieval clauses.
    if re.search(
        r"(?i)\b(good|bad|better|worse|should|ought|"
        r"opinion|think|feel|worth it|a good thing)\b",
        c,
    ):
        return None
    # More specific first.
    if DIFF_CUES.search(c):
        return "diff_search"
    if GIT_CUES.search(c):
        return "git_search"
    if cues.has_dir_ident(c) or DIR_WORDS.search(c):
        return "dir_search"
    if WEB_CUES.search(c):
        return "web_search"
    return None


def _clauses(question: str) -> list[str]:
    return [c for c in mh.decompose(question) if c.strip()]


def _looks_joined(question: str) -> bool:
    q = question or ""
    if JOIN_CUES.search(q):
        return True
    # Two code constants joined by and.
    if len(cues.dir_idents(q)) >= 2 and re.search(r"(?i)\band\b", q):
        return True
    if GIT_CUES.search(q) and DIFF_CUES.search(q) and re.search(r"(?i)\band\b", q):
        return True
    # Prefer multi_hop.decompose over a parallel join regex (one decomposer).
    if mh.looks_multi_hop(q):
        return True
    return False


def mixed_tools_without_plan(question: str) -> bool:
    """Joined question that cannot form a validated plan (e.g. opinion tail)."""
    if not _looks_joined(question):
        return False
    if select_and_build(question):
        return False
    clauses = _clauses(question)
    if len(clauses) < 2:
        return False
    tools = {classify_clause(c) for c in clauses}
    # Unclassifiable clause (opinion / hard refuse) → refuse path.
    if None in tools:
        return True
    tools.discard(None)
    if len(tools) >= 2:
        return True
    if len(tools) == 1:
        return True
    return False


def build_git_and_diff(question: str) -> Plan | None:
    if not (GIT_CUES.search(question or "") and DIFF_CUES.search(question or "")):
        return None
    if not re.search(r"(?i)\band\b", question or ""):
        return None
    clauses = _clauses(question)
    if len(clauses) < 2:
        return None
    tools = [classify_clause(c) for c in clauses]
    if tools.count("git_search") != 1 or tools.count("diff_search") != 1:
        return None
    if any(t not in ("git_search", "diff_search") for t in tools):
        return None
    steps = [
        Step(tool=t, query=c, bind_prior=False)
        for c, t in zip(clauses, tools) if t
    ]
    if len(steps) != 2:
        return None
    return Plan(plan_id="git_and_diff", steps=steps, dependent=False)


def build_dir_pair(question: str) -> Plan | None:
    if not _looks_joined(question):
        return None
    clauses = _clauses(question)
    if len(clauses) < 2:
        return None
    tools = [classify_clause(c) for c in clauses]
    if not all(t == "dir_search" for t in tools):
        return None
    steps = [Step(tool="dir_search", query=c) for c in clauses[:3]]
    return Plan(plan_id="dir_pair", steps=steps, dependent=False)


def build_git_pair(question: str) -> Plan | None:
    """Two independent git history questions joined by and."""
    if not _looks_joined(question):
        return None
    clauses = _clauses(question)
    if len(clauses) < 2:
        return None
    tools = [classify_clause(c) for c in clauses]
    if not all(t == "git_search" for t in tools):
        return None
    steps = [Step(tool="git_search", query=c) for c in clauses[:3]]
    return Plan(plan_id="git_pair", steps=steps, dependent=False)


def _clause_topics(clause: str) -> set[str]:
    """Rough proper-noun / ALLCAPS topics for relatedness checks."""
    s = clause or ""
    out = set()
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", s):
        out.add(extract_norm_topic(m.group(1)))
    # Single capitalized tokens (Apple, France, Tesla) excluding wh-words.
    skip = {
        "who", "what", "when", "where", "which", "the", "a", "an", "how",
        "why", "is", "are", "was", "were", "of", "and", "or",
    }
    for m in re.finditer(r"\b([A-Z][a-z]{1,})\b", s):
        tok = m.group(1)
        if tok.casefold() in skip:
            continue
        out.add(tok.casefold())
    for m in re.finditer(r"\b([A-Z]{2,}[A-Z0-9_]*)\b", s):
        if m.group(1).casefold() not in skip:
            out.add(m.group(1).casefold())
    for m in re.finditer(
        r"(?i)\b(?:of|for)\s+(?:the\s+)?([A-Za-z][A-Za-z0-9.-]+)\b", s
    ):
        out.add(m.group(1).casefold())
    return {t for t in out if t}


def extract_norm_topic(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def clauses_related(clauses: list[str]) -> bool:
    """True when joined clauses share a topic or use anaphora."""
    if len(clauses) < 2:
        return True
    if any(mh.needs_prior(c) for c in clauses[1:]):
        return True
    topics = [_clause_topics(c) for c in clauses]
    base = topics[0]
    if not base:
        return False
    for other in topics[1:]:
        if other & base:
            return True
        # Shared stem: Apple / Apple Inc
        for a in base:
            for b in other:
                if a in b or b in a:
                    return True
    return False


def build_git_and_web(question: str) -> Plan | None:
    """One git-history clause and one web-fact clause joined by and."""
    if not _looks_joined(question):
        return None
    clauses = _clauses(question)
    if len(clauses) < 2:
        return None
    tools = [classify_clause(c) for c in clauses]
    if tools.count("git_search") != 1 or tools.count("web_search") != 1:
        return None
    if any(t not in ("git_search", "web_search") for t in tools):
        return None
    steps = [
        Step(tool=t, query=c, bind_prior=False)
        for c, t in zip(clauses, tools) if t
    ]
    if len(steps) != 2:
        return None
    return Plan(plan_id="git_and_web", steps=steps, dependent=False)


def build_web_pair(question: str) -> Plan | None:
    if not _looks_joined(question):
        return None
    clauses = _clauses(question)
    if len(clauses) < 2:
        return None
    tools = [classify_clause(c) for c in clauses]
    if not all(t == "web_search" for t in tools):
        return None
    dependent = any(mh.needs_prior(c) for c in clauses[1:])
    # Independent web lookups may concern different entities; numbered answers
    # are fine. Relatedness only affects bind_prior / dependent labelling.
    steps = []
    for i, c in enumerate(clauses[:3]):
        steps.append(
            Step(tool="web_search", query=c, bind_prior=(i > 0 and mh.needs_prior(c)))
        )
    return Plan(
        plan_id="web_dependent" if dependent else "web_independent",
        steps=steps,
        dependent=dependent,
    )


# Legacy catalog (optional fallback / label source). Prefer propose+validate.
BUILDERS: tuple[Callable[[str], Plan | None], ...] = (
    build_git_and_diff,
    build_git_and_web,
    build_git_pair,
    build_dir_pair,
    build_web_pair,
)


def propose_steps(question: str) -> list[Step]:
    """Liberal clause → step proposal. May include mixes builders never listed."""
    if refuse_mod.is_hard_refuse(question or ""):
        return []
    if not _looks_joined(question):
        return []
    clauses = _clauses(question)[:MAX_STEPS]
    if len(clauses) < 2:
        return []
    steps: list[Step] = []
    for i, c in enumerate(clauses):
        tool = classify_clause(c)
        if tool is None:
            return []
        steps.append(
            Step(
                tool=tool,
                query=c,
                bind_prior=(i > 0 and mh.needs_prior(c)),
            )
        )
    return steps


def _legacy_plan_id(steps: list[Step], *, dependent: bool) -> str:
    """Stable telemetry labels for common shapes; else tool join."""
    tools = [s.tool for s in steps]
    if tools and all(t == "web_search" for t in tools):
        return "web_dependent" if dependent else "web_independent"
    if tools and all(t == "dir_search" for t in tools):
        return "dir_pair"
    if tools and all(t == "git_search" for t in tools):
        return "git_pair"
    if set(tools) == {"git_search", "diff_search"} and len(tools) == 2:
        return "git_and_diff"
    if set(tools) == {"git_search", "web_search"} and len(tools) == 2:
        return "git_and_web"
    short = {
        "web_search": "web",
        "dir_search": "dir",
        "git_search": "git",
        "diff_search": "diff",
        "url_read": "url",
    }
    return "+".join(short.get(t, t) for t in tools)


def validate_plan(steps: list[Step] | None) -> Plan | None:
    """Accept only plans that satisfy harness invariants."""
    if not steps or len(steps) < 2 or len(steps) > MAX_STEPS:
        return None
    for i, s in enumerate(steps):
        if s.tool not in TOOL_OK:
            return None
        if not (s.query or "").strip():
            return None
        if refuse_mod.is_hard_refuse(s.query):
            return None
        if s.bind_prior:
            if i == 0:
                return None
            if not mh.needs_prior(s.query):
                return None
    dependent = any(s.bind_prior for s in steps)
    return Plan(
        plan_id=_legacy_plan_id(steps, dependent=dependent),
        steps=list(steps),
        dependent=dependent,
    )


def select_and_build(question: str) -> Plan | None:
    """Propose steps from clauses, then validate — code remains planner of record."""
    if refuse_mod.is_hard_refuse(question or ""):
        return None
    plan = validate_plan(propose_steps(question))
    if plan:
        return plan
    # Fallback: legacy builders (should be redundant once propose covers shapes).
    for build in BUILDERS:
        built = build(question)
        if built and len(built.steps) >= 2:
            if any(s.tool not in TOOL_OK or not s.query.strip() for s in built.steps):
                continue
            return built
    return None


def _run_one(
    tool: str,
    query: str,
    *,
    root: str,
    seed: int,
    runners: dict | None,
):
    if runners and tool in runners:
        return runners[tool](query, seed=seed, root=root)
    if tool == "web_search":
        return ws.run(query, router="rule", seed=seed)
    if tool == "dir_search":
        from deku import dir_search as ds
        return ds.run(query, root=root, seed=seed)
    if tool == "git_search":
        from deku import git_search as gits
        return gits.run(query, root=root, seed=seed, live_answer=True)
    if tool == "diff_search":
        from deku import diff_search as diffs
        return diffs.run(query, root=root, seed=seed, live_answer=True)
    if tool == "url_read":
        from deku import url_read as ur
        return ur.run(query, seed=seed, live_answer=True)
    raise ValueError(f"unsupported tool {tool}")


def run(
    question: str,
    *,
    seed: int = 0,
    root: str = ".",
    runners: dict | None = None,
    plan: Plan | None = None,
) -> Result:
    """Select (unless `plan` given) and execute a weak multi-step plan."""
    out = Result(detail={"mode": "orchestrate"})
    built = plan or select_and_build(question)
    if not built:
        out.status = "refused"
        out.answer = (
            "I could not build a multi-step plan for that. "
            "Ask one short factual question, or join two clear questions "
            "with 'and what/who/when…'."
        )
        out.detail["reason"] = "no_plan"
        return out
    out.detail["plan_id"] = built.plan_id
    out.detail["steps"] = [
        {"tool": s.tool, "query": s.query, "bind_prior": s.bind_prior}
        for s in built.steps
    ]
    hops: list[tuple[str, str]] = []
    docs: list[str] = []
    rewritten: list[str] = []
    failed: list[dict] = []
    prior_core: str | None = None
    prior_query: str | None = None
    dependent = built.dependent
    for i, step in enumerate(built.steps):
        q = step.query
        if step.bind_prior and prior_core:
            bind = mh.bind_core(prior_query or "", prior_core, step.query)
            q = mh.rewrite_followup(q, bind)
            dependent = True
        elif step.bind_prior and not prior_core:
            failed.append({"query": q, "tool": step.tool, "reason": "no_prior_core"})
            continue
        rewritten.append(q)
        got = _run_one(
            step.tool, q, root=root, seed=seed + i, runners=runners,
        )
        out.hits.extend(getattr(got, "hits", None) or [])
        docs.append(getattr(got, "document", None) or "")
        if getattr(got, "status", None) != "ok" or not (getattr(got, "answer", None) or "").strip():
            failed.append({
                "query": q,
                "tool": step.tool,
                "status": getattr(got, "status", None),
            })
            # Dependent chain cannot continue without this hop's core.
            if built.dependent or step.bind_prior:
                if not hops:
                    out.status = "cannot_answer"
                    out.answer = (
                        f"I cannot answer from the available sources "
                        f"(failed on: {q} via {step.tool})"
                    )
                    out.detail["failed_sub"] = q
                    out.detail["failed_tool"] = step.tool
                    out.detail["sub_status"] = getattr(got, "status", None)
                    out.detail["rewritten"] = rewritten
                    out.detail["dependent"] = dependent
                    out.detail["failed_steps"] = failed
                    return out
                break
            continue
        prior_core = mh.core_from_result(got) or prior_core
        prior_query = q
        hops.append((q, got.answer.strip()))
    out.detail["rewritten"] = rewritten
    out.detail["dependent"] = dependent
    out.detail["failed_steps"] = failed
    out.document = "\n\n".join(d for d in docs if d)
    if not hops:
        out.status = "cannot_answer"
        fail_q = (failed[0]["query"] if failed else question)
        fail_tool = (failed[0].get("tool") if failed else "unknown")
        out.answer = (
            f"I cannot answer from the available sources "
            f"(failed on: {fail_q} via {fail_tool})"
        )
        if failed:
            out.detail["failed_sub"] = failed[0]["query"]
            out.detail["failed_tool"] = failed[0].get("tool")
        return out
    body = mh.integrate(hops, dependent=dependent and not failed)
    if failed:
        miss = "; ".join(f["query"] for f in failed)
        out.answer = f"{body}\n\n(could not answer: {miss})"
        out.status = "partial"
    else:
        out.answer = body
        out.status = "ok"
    return out
