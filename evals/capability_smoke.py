#!/usr/bin/env python3
"""Fixed capability smoke against the default GGUF serve (not MLX scores).

Covers refuse / URL summarize / multi-hop / web control. Network + live MiniCPM
required for non-refuse rows. Unit tests stay offline; this is the measured
product path.

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


CASES: list[Case] = [
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
    Case(
        "control:apple_ceo",
        "Who is the CEO of Apple?",
        expect_tool="web_search",
        expect_status="ok",
        answer_must_match=r"Tim Cook",
    ),
    Case(
        "control:france_capital",
        "What is the capital of France?",
        expect_tool="web_search",
        expect_status="ok",
        answer_must_match=r"Paris",
    ),
    Case(
        "sum:url_apollo",
        "Summarize https://en.wikipedia.org/wiki/Apollo_11",
        expect_tool="url_read",
        expect_status="ok",
        answer_must_match=r"Apollo|Moon|NASA",
    ),
    Case(
        "hop:cook_paris",
        "Who is the CEO of Apple and what is the capital of France?",
        expect_tool="multi_hop",
        expect_status="ok",
        answer_must_match=r"(?s)Tim Cook.*Paris|Paris.*Tim Cook",
    ),
    Case(
        "git:last_commit",
        "What is the last commit message?",
        expect_tool="git_search",
        expect_status="ok",
        answer_must_match=r".+",
    ),
    Case(
        "dir:project",
        "What is this project about?",
        expect_tool="dir_search",
        expect_status="ok",
        answer_must_match=r"deku|MiniCPM|harness|task",
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


def run_case(case: Case) -> dict:
    t0 = time.perf_counter()
    got = rt.dispatch(
        case.question,
        router="rule",
        seed=0,
        root=str(ROOT),
        live_answer=True,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    answer = got.answer or ""
    reasons: list[str] = []
    ok = True
    if case.expect_tool and got.tool != case.expect_tool:
        ok = False
        reasons.append(f"tool={got.tool!r} want {case.expect_tool!r}")
    if case.expect_status and got.status != case.expect_status:
        ok = False
        reasons.append(f"status={got.status!r} want {case.expect_status!r}")
    reason = (got.detail or {}).get("reason")
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
        reasons.append(f"answer matched forbidden /{case.answer_must_not_match}/")
    return {
        "name": case.name,
        "pass": ok,
        "tool": got.tool,
        "status": got.status,
        "reason": reason,
        "answer": answer[:500],
        "elapsed_ms": elapsed_ms,
        "fail_reasons": reasons,
        "detail_keys": sorted((got.detail or {}).keys()),
    }


def main() -> int:
    _check_server()
    rows = [run_case(c) for c in CASES]
    # Deduplicate hop cases into one logical scoreboard row? Keep both checks.
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
