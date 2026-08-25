"""Weak multi-step plans via propose + validate (code is the planner).

Not model planning. Clauses are classified in code; a plan is accepted only
when it passes explicit invariants (tool allow-list, length, bind rules).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from deku import multi_hop as mh
from deku import refuse as refuse_mod
from deku import route_cues as cues
from deku import url_read as ur
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
    r"and (?:who|what|when|where|which|should|do you)\b|"
    r"\?\s+(?:and\s+)?(?:who|what|when|where|which|should)\b|"
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
        r"opinion|think|feel|worth it|a good thing|"
        r"buy the stock|invest in)\b",
        c,
    ):
        return None
    # Explicit URL → url_read (before web "what/who" cues).
    if ur.URL_RE.search(c):
        return "url_read"
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


def propose_steps(question: str) -> list[Step]:
    """Liberal clause → step proposal for any TOOL_OK mix."""
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


def plan_id_for(steps: list[Step], *, dependent: bool) -> str:
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
        plan_id=plan_id_for(steps, dependent=dependent),
        steps=list(steps),
        dependent=dependent,
    )


def select_and_build(question: str) -> Plan | None:
    """Propose steps from clauses, then validate — sole planner path."""
    if refuse_mod.is_hard_refuse(question or ""):
        return None
    return validate_plan(propose_steps(question))


def _next_hint(status: str, *, failed: list[dict], reason: str | None = None) -> dict:
    """Machine-oriented hint for a parent agent after a non-ok outcome."""
    if status == "partial" and failed:
        return {
            "action": "retry_failed_clauses",
            "clauses": [f.get("query") for f in failed if f.get("query")],
        }
    if status == "cannot_answer":
        return {
            "action": "abstain_or_narrow",
            "failed_tool": (failed[0].get("tool") if failed else None),
            "failed_query": (failed[0].get("query") if failed else None),
        }
    if status == "refused":
        return {
            "action": "ask_in_scope_fact",
            "reason": reason or "no_plan",
        }
    return {"action": "none"}


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
        out.detail["cores"] = []
        out.detail["locations"] = []
        out.detail["next_hint"] = _next_hint("refused", failed=[], reason="no_plan")
        return out
    out.detail["plan_id"] = built.plan_id
    out.detail["steps"] = [
        {"tool": s.tool, "query": s.query, "bind_prior": s.bind_prior}
        for s in built.steps
    ]
    hops: list[tuple[str, str]] = []
    cores: list[str] = []
    locations: list[dict] = []
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
                    out.detail["cores"] = cores
                    out.detail["locations"] = locations
                    out.detail["next_hint"] = _next_hint(
                        "cannot_answer", failed=failed
                    )
                    return out
                break
            continue
        hop_detail = getattr(got, "detail", None) or {}
        hop_core = mh.core_from_result(got)
        if hop_core:
            cores.append(hop_core)
        for loc in hop_detail.get("locations") or []:
            if isinstance(loc, dict) and loc.get("path"):
                locations.append(dict(loc))
        if hop_detail.get("path") and not hop_detail.get("locations"):
            locations.append({
                "path": hop_detail["path"],
                "value": hop_detail.get("core"),
                "query": q,
            })
        prior_core = hop_core or prior_core
        prior_query = q
        hops.append((q, got.answer.strip()))
    out.detail["rewritten"] = rewritten
    out.detail["dependent"] = dependent
    out.detail["failed_steps"] = failed
    out.detail["cores"] = cores
    out.detail["locations"] = locations
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
        out.detail["next_hint"] = _next_hint("cannot_answer", failed=failed)
        return out
    body = mh.integrate(hops, dependent=dependent and not failed)
    if failed:
        miss = "; ".join(f["query"] for f in failed)
        out.answer = f"{body}\n\n(could not answer: {miss})"
        out.status = "partial"
        out.detail["next_hint"] = _next_hint("partial", failed=failed)
    else:
        out.answer = body
        out.status = "ok"
        out.detail["next_hint"] = {"action": "none"}
    return out
