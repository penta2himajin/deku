"""Unified tool router for deku retrieval tasks.

Hard cues (URL present, math, git/diff keywords) win over the model.
Needle (optional) picks among a small closed tool set; rules are the fallback.
URL / path / rev extraction stays in code — Needle only emits a tool label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from deku import normalize as nz
from deku import refuse as refuse_mod
from deku import route_cues as cues
from deku import url_read as ur
from deku import web_search as ws

TOOLS = (
    "web_search",
    "dir_search",
    "git_search",
    "diff_search",
    "url_read",
    "multi_hop",
    "clarify",
    "refuse",
)
TOOL_RE = re.compile(
    r"(?i)\b(web_search|dir_search|git_search|diff_search|url_read|"
    r"multi_hop|clarify|refuse)\b"
)

NONSEARCH = ws.NONSEARCH
WEB_CUES = ws.SEARCH_CUES
DIR_WORDS = cues.DIR_WORDS
# Back-compat pattern; hard_route uses cues.has_dir_ident (no LVMH/NASA).
DIR_IDENT = re.compile(r"\b(?:PREFILL|MAX_TOKENS|[A-Z][A-Z0-9]*_[A-Z0-9_]+)\b")
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
SUMMARIZE_CUES = re.compile(
    r"(?i)\b(summarize|summarise|summary of|tl;dr|tldr)\b"
)
MULTI_HOP_CUES = re.compile(
    r"(?i)\b("
    r"and (?:who|what|when|where|which)\b|"
    r"\?\s+(?:and\s+)?(?:who|what|when|where|which)\b"
    r")"
)


@dataclass
class Decision:
    tool: str
    url: str | None = None
    query: str = ""
    detail: dict = field(default_factory=dict)


@dataclass
class Routed:
    tool: str
    status: str = ""
    answer: str | None = None
    query: str = ""
    url: str | None = None
    detail: dict = field(default_factory=dict)
    document: str = ""
    hits: list = field(default_factory=list)


def parse_tool(raw: str) -> str:
    m = TOOL_RE.search(raw or "")
    return m.group(1).lower() if m else "refuse"


def extract_url(text: str) -> str | None:
    return ur.extract_url(text)


def hard_route(question: str) -> Decision | None:
    """High-confidence cues that must not be overridden by Needle."""
    from deku import normalize as nz

    q = nz.prepare_question(question)[0]
    url = extract_url(q)
    if url:
        return Decision(tool="url_read", url=url, detail={"cue": "url"})
    from deku import clarify as cl
    clarify_kind = cl.detect(q)
    if clarify_kind:
        return Decision(
            tool="clarify",
            detail={"cue": "clarify", "reason": clarify_kind},
        )
    if refuse_mod.is_hard_refuse(q) or NONSEARCH.search(q):
        reason = refuse_mod.classify(q)
        return Decision(
            tool="refuse",
            detail={"cue": "nonsearch", "reason": reason},
        )
    # Compound plans beat a single hard cue (e.g. git+diff, two ALLCAPS).
    from deku import orchestrate as orch
    plan = orch.select_and_build(q)
    if plan and len(plan.steps) >= 2:
        return Decision(
            tool="multi_hop",
            query=ws.rule_query(q),
            detail={"cue": "plan", "plan_id": plan.plan_id},
        )
    if orch.mixed_tools_without_plan(q):
        return Decision(
            tool="refuse",
            detail={
                "cue": "mixed_tools",
                "reason": "out_of_scope",
            },
        )
    if DIFF_CUES.search(q):
        return Decision(tool="diff_search", detail={"cue": "diff"})
    if GIT_CUES.search(q):
        return Decision(tool="git_search", detail={"cue": "git"})
    if cues.has_dir_ident(q):
        return Decision(
            tool="dir_search",
            query=ws.rule_query(q),
            detail={"cue": "dir_ident"},
        )
    return None


def rule_route(question: str) -> Decision:
    """Deterministic tool choice. Hard cues first, then soft lexical cues."""
    q, nd = nz.prepare_question(question)
    hard = hard_route(q)
    if hard:
        hard.detail = {**nd, **hard.detail, "router": "rule"}
        return hard
    from deku import orchestrate as orch
    plan = orch.select_and_build(q)
    if plan and len(plan.steps) >= 2:
        return Decision(
            tool="multi_hop",
            query=ws.rule_query(q),
            detail={**nd, "router": "rule", "cue": "plan", "plan_id": plan.plan_id},
        )
    if orch.mixed_tools_without_plan(q):
        return Decision(
            tool="refuse",
            detail={
                **nd,
                "router": "rule",
                "cue": "mixed_tools",
                "reason": "out_of_scope",
            },
        )
    if DIR_WORDS.search(q):
        return Decision(
            tool="dir_search",
            query=ws.rule_query(q),
            detail={**nd, "router": "rule", "cue": "dir"},
        )
    if WEB_CUES.search(q):
        return Decision(
            tool="web_search",
            query=ws.rule_query(q),
            detail={**nd, "router": "rule", "cue": "web"},
        )
    reason = refuse_mod.classify(q)
    return Decision(
        tool="refuse",
        detail={**nd, "router": "rule", "cue": "default", "reason": reason},
    )


def needle_route(question: str) -> Decision:
    """Needle tool-call → one of TOOLS. Hard cues win; else Needle; else rule."""
    hard = hard_route(question)
    if hard:
        hard.detail = {**hard.detail, "router": "needle", "hard": True}
        return hard
    try:
        from needle import Needle, tool, Field
    except ImportError:
        d = rule_route(question)
        d.detail["router"] = "needle_fallback_rule"
        return d

    @tool
    def choose(tool_name: str = Field(
        ...,
        description=(
            "web_search | dir_search | git_search | diff_search | url_read | "
            "multi_hop | clarify | refuse"
        ),
    )):
        """Pick the retrieval tool.
        web_search = public web facts (people, companies, places).
        dir_search = this local repository's files / README / overview.
        git_search = git history, last commit, blame (only if not already hard-cued).
        diff_search = staged/unstaged/working-tree diffs.
        url_read = only when an http(s) URL is present (usually hard-cued).
        multi_hop = two short factual questions joined by and/?; else web_search.
        clarify = missing file path or URL for an otherwise supported ask.
        refuse = math, code authoring, chitchat, long analysis.
        """
        return tool_name

    n = Needle(
        tools=[choose],
        system=(
            "Always call choose with exactly one tool_name. "
            "Local project/README/overview → dir_search. "
            "Public who/what/where facts → web_search. "
            "Two linked factual questions → multi_hop. "
            "Vague 'this part' without a path, or summarize-this without a URL → clarify. "
            "Greeting, math, code, or long essay/analysis → refuse. "
            "Do not invent URLs or commit SHAs."
        ),
    )
    try:
        r = n.complete(f"Question: {question}", 64)
        calls = r.get("function_calls") or []
        if calls:
            name = (calls[0].get("arguments") or {}).get("tool_name")
            if name in TOOLS:
                d = Decision(tool=name, detail={"router": "needle", "hard": False})
                if name == "url_read":
                    d.url = extract_url(question)
                if name in ("web_search", "dir_search"):
                    d.query = ws.rule_query(question)
                return d
    finally:
        try:
            n.reset()
        except Exception:
            pass
    d = rule_route(question)
    d.detail["router"] = "needle_fallback_rule"
    return d


def route(question: str, *, router: str = "rule") -> Decision:
    if router == "needle":
        return needle_route(question)
    return rule_route(question)


def dispatch(
    question: str,
    *,
    router: str = "rule",
    seed: int = 0,
    root: str = ".",
    live_answer: bool = True,
    url_fetch=None,
    use_needle_slots: bool = False,
    audience: str | None = None,
) -> Routed:
    """Route then run the chosen tool (stubs for git/diff until implemented)."""
    import os

    from deku import normalize as nz

    aud = (audience or os.environ.get("DEKU_AUDIENCE") or "human").strip().lower()
    if aud not in refuse_mod.AUDIENCES:
        aud = "human"
    q, nd = nz.prepare_question(question)
    dec = route(q, router=router)
    out = Routed(
        tool=dec.tool,
        query=dec.query,
        url=dec.url,
        detail={**nd, **dict(dec.detail), "audience": aud},
    )
    if dec.tool == "refuse":
        reason = dec.detail.get("reason") or refuse_mod.classify(q)
        out.status = "refused"
        out.answer = refuse_mod.message(reason, audience=aud)
        out.detail["reason"] = reason
        out.detail["cores"] = []
        out.detail["next_hint"] = {
            "action": "ask_in_scope_fact",
            "reason": reason,
        }
        return out
    if dec.tool == "clarify":
        from deku import clarify as cl
        reason = dec.detail.get("reason") or cl.detect(q) or "path"
        out.status = "clarify"
        out.answer = cl.question_for(q)
        out.detail["reason"] = reason
        out.detail["next_hint"] = {
            "action": "provide_path",
            "reason": reason,
        }
        return out
    if dec.tool == "url_read":
        got = ur.run(
            q,
            url=dec.url,
            seed=seed,
            fetch=url_fetch,
            live_answer=live_answer,
        )
        out.status = got.status
        out.answer = got.answer
        out.url = got.url or dec.url
        out.document = got.document
        out.detail.update(got.detail)
        if SUMMARIZE_CUES.search(q or ""):
            out.detail["mode"] = got.detail.get("mode", "summarize")
        return out
    if dec.tool == "multi_hop":
        from deku import orchestrate as orch
        if not live_answer:
            out.status = "skipped_offline"
            out.detail["note"] = "multi_hop needs live_answer"
            return out
        got = orch.run(q, seed=seed, root=root)
        out.status = got.status
        out.answer = got.answer
        out.document = got.document
        out.hits = got.hits
        out.detail.update(got.detail)
        if got.status == "refused" and got.detail.get("reason"):
            out.answer = refuse_mod.message(
                str(got.detail["reason"]), audience=aud
            )
        if "cores" not in out.detail:
            out.detail["cores"] = []
        if "next_hint" not in out.detail:
            out.detail["next_hint"] = {"action": "none"}
        return out
    if dec.tool == "web_search":
        if not live_answer:
            out.status = "skipped_offline"
            out.detail["note"] = "web_search needs live_answer"
            return out
        got = ws.run(
            q,
            router="rule",
            seed=seed,
            use_needle_slots=use_needle_slots,
        )
        out.status = got.status
        out.answer = got.answer
        out.query = got.query
        out.document = got.document
        out.hits = got.hits
        out.detail.update(got.detail)
        return out
    if dec.tool == "dir_search":
        from deku import dir_search as ds
        got = ds.run(q, root=root, seed=seed)
        out.status = got.status
        out.answer = got.answer
        out.query = got.query
        out.document = got.document
        out.hits = got.hits
        out.detail.update(got.detail)
        return out
    if dec.tool == "git_search":
        from deku import git_search as gits
        got = gits.run(
            q, root=root, seed=seed, live_answer=live_answer,
        )
        out.status = got.status
        out.answer = got.answer
        out.query = got.query
        out.document = got.document
        out.hits = got.hits
        out.detail.update(got.detail)
        return out
    if dec.tool == "diff_search":
        from deku import diff_search as diffs
        got = diffs.run(
            q, root=root, seed=seed, live_answer=live_answer,
        )
        out.status = got.status
        out.answer = got.answer
        out.query = got.query
        out.document = got.document
        out.hits = got.hits
        out.detail.update(got.detail)
        return out
    out.status = "refused"
    out.answer = refuse_mod.message("out_of_scope", audience=aud)
    out.detail["reason"] = "out_of_scope"
    out.detail["cores"] = []
    out.detail["next_hint"] = {
        "action": "ask_in_scope_fact",
        "reason": "out_of_scope",
    }
    return out


def envelope(got: Routed) -> dict:
    """Stable machine contract for parent agents / ``deku ask --json``."""
    d = dict(got.detail or {})
    cores = d.get("cores")
    if cores is None:
        core = d.get("core")
        cores = [core] if core else []
    return {
        "status": got.status,
        "tool": got.tool,
        "answer": got.answer,
        "reason": d.get("reason"),
        "plan_id": d.get("plan_id"),
        "cores": cores,
        "failed_steps": d.get("failed_steps"),
        "next_hint": d.get("next_hint") or {"action": "none"},
        "query": got.query,
        "url": got.url,
        "detail": d,
    }
