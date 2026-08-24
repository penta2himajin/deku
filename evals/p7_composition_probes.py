"""Live spot-check battery for validated composition (P7+).

Run against a local deku-serve / OpenAI-compatible endpoint:
  uv run python evals/p7_composition_probes.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from deku import route as rt

PROBES: list[tuple[str, str, str]] = [
    # id, question, expect_tool (approx; status judged loosely)
    ("C01", "Who is the CEO of Apple and what is the PREFILL string?", "multi_hop"),
    ("C02", "What does the README say about deku and who is the CEO of Apple?", "multi_hop"),
    ("C03", "Who authored the last commit and what is the PREFILL string?", "multi_hop"),
    ("C04", "What is the last commit message and what is in the unstaged diff?", "multi_hop"),
    ("C05", "Who is the CEO of Apple and where was he born?", "multi_hop"),
    ("C06", "Who founded Microsoft and when was it founded?", "multi_hop"),
    ("C07", "What is the capital of France and what is the capital of Japan?", "multi_hop"),
    ("C08", "Who is the CEO of Apple and what is the capital of France?", "multi_hop"),
    ("C09", "What is MAX_TOKENS and what is the PREFILL string?", "multi_hop"),
    ("C10", "What is in the unstaged diff and who is the CEO of Apple?", "multi_hop"),
    ("C11", "Who authored the last commit and who is the CEO of Apple?", "multi_hop"),
    ("C12", "What does the README say about deku and what is MAX_TOKENS?", "multi_hop"),
    ("C13", "What is this project about and who authored the last commit?", "multi_hop"),
    ("C14", "Who is the CEO of Toyota and what is the PREFILL string?", "multi_hop"),
    ("C15", "Who founded SpaceX and where was he born?", "multi_hop"),
    ("C16", "What is the capital of Japan and what is its population?", "multi_hop"),
    ("C17", "Who is the CEO of Apple? Also, is that a good thing for the company?", "refuse"),
    ("C18", "What is 2+2 and who is the CEO of Apple?", "refuse"),
    ("C19", "Write a sorting function and what is PREFILL?", "refuse"),
    ("C20", "Who is the CEO of Apple and should I buy the stock?", "refuse"),
    ("C21", "Who is the CEO of Apple?", "web_search"),
    ("C22", "What is the PREFILL string?", "dir_search"),
    ("C23", "What does the README say about deku?", "dir_search"),
    ("C24", "What is this project about?", "dir_search"),
    ("C25", "Who authored the last commit?", "git_search"),
    ("C26", "What is in the unstaged diff?", "diff_search"),
    ("C27", "What language models does deku run?", "dir_search"),
    ("C28", "Where is Apple headquartered?", "web_search"),
    ("C29", "Who is the prime minister of Japan?", "web_search"),
    ("C30", "Who is the CEO of Google?", "web_search"),
    ("C31", "What is the capital of Peru?", "web_search"),
    ("C32", "Who founded Stripe?", "web_search"),
    ("C33", "When was Tesla founded?", "web_search"),
    ("C34", "Who wrote Hamlet?", "web_search"),
    ("C35", "What is the last commit message and who authored it?", "multi_hop"),
    ("C36", "Who currently runs Apple as chief executive?", "web_search"),
    ("C37", "What is Tokyo's population?", "web_search"),
    ("C38", "Summarize https://example.com", "url_read"),
    ("C39", "What is the capital of France and who is the CEO of Apple?", "multi_hop"),
    ("C40", "What does AGENTS.md say about the harness and what is PREFILL?", "multi_hop"),
    ("C41", "Who is the CEO of Apple and what is MAX_TOKENS?", "multi_hop"),
    ("C42", "What is PREFILL and who authored the last commit?", "multi_hop"),
    ("C43", "What changed in the last commit and what is the capital of France?", "multi_hop"),
    ("C44", "Who is the CEO of Sony and where is Sony headquartered?", "multi_hop"),
    ("C45", "What is this project about? Also, do you like it?", "refuse"),
]


def main() -> int:
    rows = []
    for pid, q, expect_tool in PROBES:
        t0 = time.time()
        try:
            r = rt.dispatch(q, live_answer=True, root=".")
            row = {
                "id": pid,
                "q": q,
                "expect_tool": expect_tool,
                "tool": r.tool,
                "status": r.status,
                "plan_id": (r.detail or {}).get("plan_id"),
                "answer": (r.answer or "")[:280].replace("\n", " / "),
                "secs": round(time.time() - t0, 2),
                "abstain": (r.detail or {}).get("abstain_reason"),
                "failed_steps": (r.detail or {}).get("failed_steps"),
            }
        except Exception as e:  # noqa: BLE001 — probe battery must continue
            row = {
                "id": pid,
                "q": q,
                "expect_tool": expect_tool,
                "tool": "EXCEPTION",
                "status": "exception",
                "plan_id": None,
                "answer": repr(e),
                "secs": round(time.time() - t0, 2),
            }
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
        time.sleep(0.4)

    out = Path("evals/results/p7_composition_probes.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    okish = sum(1 for r in rows if r["status"] in ("ok", "partial", "refused"))
    print(f"# wrote {out}  n={len(rows)} okish={okish}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
