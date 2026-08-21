#!/usr/bin/env python3
"""Fixed capability smoke against the default GGUF serve (not MLX scores).

Real-world questions covering the product surface:
  refuse, clarify, web templates, multi-hop (independent / dependent / git_pair),
  url_read summarize, git_search, dir_search (overview + assignment).

Network + live MiniCPM required for non-refuse/clarify rows. Unit tests stay
offline; this is the measured product path.

Usage:
  mise run serve          # other terminal
  mise run capability-smoke
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results" / "capability_smoke.json"

from deku import llm
from deku import route as rt


@dataclass(frozen=True)
class Case:
    name: str
    question: str
    expect_tool: str | None = None
    expect_status: str | None = None
    expect_reason: str | None = None
    answer_must_match: str | None = None  # regex, case-insensitive
    answer_must_not_match: str | None = None
    detail_must_match: str | None = None  # regex over json-ish detail dump


CASES: list[Case] = [
    # ---- refuse -----------------------------------------------------------
    Case(
        "refuse:math",
        "What is 2+2?",
        expect_tool="refuse",
        expect_status="refused",
        expect_reason="math",
    ),
    Case(
        "refuse:code",
        "Write a sort function",
        expect_tool="refuse",
        expect_status="refused",
        expect_reason="code",
    ),
    Case(
        "refuse:chitchat",
        "hello there",
        expect_tool="refuse",
        expect_status="refused",
        expect_reason="chitchat",
    ),
    Case(
        "refuse:deep",
        "Compare capitalism and socialism in a long essay",
        expect_tool="refuse",
        expect_status="refused",
        expect_reason="deep_reasoning",
    ),
    Case(
        "refuse:fix_bug",
        "Fix the bug in route.py",
        expect_tool="refuse",
        expect_status="refused",
        expect_reason="code",
    ),
    # ---- clarify ----------------------------------------------------------
    Case(
        "clarify:path",
        "Show me the commit log of the last commit that changed this part.",
        expect_tool="clarify",
        expect_status="clarify",
        expect_reason="path",
        answer_must_match=r"path|deku/",
    ),
    Case(
        "clarify:url",
        "Summarize this document for me.",
        expect_tool="clarify",
        expect_status="clarify",
        expect_reason="url",
        answer_must_match=r"url|http",
    ),
    # ---- web templates / fact_core ----------------------------------------
    Case(
        "web:apple_ceo",
        "Who is the CEO of Apple?",
        expect_tool="web_search",
        expect_status="ok",
        answer_must_match=r"Tim Cook",
    ),
    Case(
        "web:france_capital",
        "What is the capital of France?",
        expect_tool="web_search",
        expect_status="ok",
        answer_must_match=r"Paris",
    ),
    Case(
        "web:apple_hq",
        "Where is Apple headquartered?",
        expect_tool="web_search",
        expect_status="ok",
        answer_must_match=r"Cupertino",
    ),
    Case(
        "web:cook_born",
        "Where was Tim Cook born?",
        expect_tool="web_search",
        expect_status="ok",
        answer_must_match=r"Mobile|Alabama",
        answer_must_not_match=r"born in Tim\b",
    ),
    Case(
        "web:tokyo_population",
        "What is the population of Tokyo?",
        expect_tool="web_search",
        expect_status="ok",
        answer_must_match=r"million|\d{1,3}(?:,\d{3})+",
        answer_must_not_match=r"population of Tokyo is The\b",
    ),
    Case(
        "web:iphone_released",
        "When was the iPhone released?",
        expect_tool="web_search",
        expect_status="ok",
        answer_must_match=r"2007",
    ),
    Case(
        "web:microsoft_founded_year",
        "When was Microsoft founded?",
        expect_tool="web_search",
        expect_status="ok",
        answer_must_match=r"1975",
    ),
    Case(
        "web:emperor_birthday",
        "What is the birthday of the current Emperor of Japan?",
        expect_tool="web_search",
        expect_status="ok",
        answer_must_match=r"23 February 1960|February 23,?\s*1960|1960",
        answer_must_not_match=r"public holiday|cannot answer",
    ),
    # ---- multi-hop catalog ------------------------------------------------
    Case(
        "hop:independent",
        "Who is the CEO of Apple and what is the capital of France?",
        expect_tool="multi_hop",
        expect_status="ok",
        answer_must_match=r"(?s)Tim Cook.*Paris|Paris.*Tim Cook",
    ),
    Case(
        "hop:dependent_founded",
        "Who founded Microsoft and when was it founded?",
        expect_tool="multi_hop",
        expect_status="ok",
        answer_must_match=r"(?s)(Gates|Allen).*(1975)|1975.*(Gates|Allen)",
    ),
    Case(
        "hop:dependent_born",
        "Who is the CEO of Apple and where was he born?",
        expect_tool="multi_hop",
        expect_status="ok",
        answer_must_match=r"(?s)Tim Cook.*(Mobile|Alabama)|(Mobile|Alabama).*Tim Cook",
    ),
    Case(
        "hop:git_pair",
        "What is the last commit message and who authored the last commit?",
        expect_tool="multi_hop",
        expect_status="ok",
        answer_must_match=r"(?s).+",
        detail_must_match=r"git_pair",
    ),
    # ---- url / git / dir --------------------------------------------------
    Case(
        "url:apollo_summary",
        "Summarize https://en.wikipedia.org/wiki/Apollo_11",
        expect_tool="url_read",
        expect_status="ok",
        answer_must_match=r"Apollo|Moon|NASA",
    ),
    Case(
        "git:last_commit",
        "What is the last commit message?",
        expect_tool="git_search",
        expect_status="ok",
        answer_must_match=r".+",
    ),
    Case(
        "dir:project_overview",
        "What is this project about?",
        expect_tool="dir_search",
        expect_status="ok",
        answer_must_match=r"deku|MiniCPM|harness|task",
        detail_must_match=r"prose_lead",
    ),
    Case(
        "dir:assignment",
        "What is the PREFILL string?",
        expect_tool="dir_search",
        expect_status="ok",
        answer_must_match=r"PREFILL|prefill|=",
    ),
]


def _check_server() -> None:
    try:
        llm.complete("ping", max_tokens=1, temperature=0.0, timeout=30.0)
    except llm.LLMError as e:
        raise SystemExit(
            f"capability-smoke: server not ready at {llm.BASE_URL}: {e}\n"
            "Start with: mise run serve"
        ) from e


def run_case(case: Case, *, retries: int = 2) -> dict:
    # Soft pacing so Wikipedia does not 429 mid-suite.
    if case.expect_tool in {"web_search", "multi_hop", "url_read"}:
        time.sleep(1.0)
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            got = rt.dispatch(
                case.question,
                router="rule",
                seed=0,
                root=str(ROOT),
                live_answer=True,
            )
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            return {
                "name": case.name,
                "pass": False,
                "tool": None,
                "status": "error",
                "reason": None,
                "answer": f"{type(e).__name__}: {e}",
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                "fail_reasons": [f"exception: {type(e).__name__}: {e}"],
                "detail_keys": [],
                "reply_source": None,
            }
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        answer = got.answer or ""
        detail = got.detail or {}
        detail_blob = json.dumps(detail, ensure_ascii=False, default=str)
        reasons: list[str] = []
        ok = True
        if case.expect_tool and got.tool != case.expect_tool:
            ok = False
            reasons.append(f"tool={got.tool!r} want {case.expect_tool!r}")
        if case.expect_status and got.status != case.expect_status:
            ok = False
            reasons.append(f"status={got.status!r} want {case.expect_status!r}")
        reason = detail.get("reason")
        if case.expect_reason and reason != case.expect_reason:
            ok = False
            reasons.append(f"reason={reason!r} want {case.expect_reason!r}")
        if case.answer_must_match and not re.search(
            case.answer_must_match, answer, re.I
        ):
            ok = False
            reasons.append(f"answer missing /{case.answer_must_match}/")
        if case.answer_must_not_match and re.search(
            case.answer_must_not_match, answer, re.I
        ):
            ok = False
            reasons.append(
                f"answer matched forbidden /{case.answer_must_not_match}/"
            )
        if case.detail_must_match and not re.search(
            case.detail_must_match, detail_blob, re.I
        ):
            ok = False
            reasons.append(f"detail missing /{case.detail_must_match}/")
        # Retry transient empty retrieval once.
        if (
            not ok
            and attempt < retries
            and got.status in {"cannot_answer", "fetch_error", "empty_page"}
            and case.expect_status == "ok"
        ):
            time.sleep(2.0 * (attempt + 1))
            continue
        return {
            "name": case.name,
            "pass": ok,
            "tool": got.tool,
            "status": got.status,
            "reason": reason,
            "answer": answer[:500],
            "elapsed_ms": elapsed_ms,
            "fail_reasons": reasons,
            "detail_keys": sorted(detail.keys()),
            "reply_source": detail.get("reply_source") or detail.get("mode"),
        }
    return {
        "name": case.name,
        "pass": False,
        "tool": None,
        "status": "error",
        "reason": None,
        "answer": str(last_err),
        "elapsed_ms": 0,
        "fail_reasons": [f"retries exhausted: {last_err}"],
        "detail_keys": [],
        "reply_source": None,
    }


def main() -> int:
    _check_server()
    rows = [run_case(c) for c in CASES]
    passed = sum(1 for r in rows if r["pass"])
    payload = {
        "backend": "gguf",
        "base_url": llm.BASE_URL,
        "model": llm.MODEL,
        "when": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "total": len(rows),
        "rows": rows,
        "cases": [asdict(c) for c in CASES],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"{passed}/{len(rows)} passed → {OUT}")
    for r in rows:
        mark = "ok" if r["pass"] else "FAIL"
        print(f"  [{mark}] {r['name']}: tool={r['tool']} status={r['status']}")
        if not r["pass"]:
            print(f"         {r['fail_reasons']}")
            print(f"         answer={r['answer']!r}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
