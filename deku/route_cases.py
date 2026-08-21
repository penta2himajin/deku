"""Gold labels for the unified tool router.

`want` is the tool a correct router must pick. Hard-cued cases should never
reach Needle; soft cases may.
"""
from __future__ import annotations

# (question, want_tool, kind)  kind: hard | soft
ROUTE_CASES: list[tuple[str, str, str]] = [
    # hard — URL / math / clear git / diff / ALLCAPS / deep refuse
    ("Who is Tim Cook according to https://en.wikipedia.org/wiki/Tim_Cook?", "url_read", "hard"),
    ("Summarize https://example.com/ports", "url_read", "hard"),
    ("What is 2+2?", "refuse", "hard"),
    ("Write a sort function", "refuse", "hard"),
    ("Compare capitalism and socialism in a long essay", "refuse", "hard"),
    ("Explain why quantum entanglement works step by step", "refuse", "hard"),
    ("What is the last commit message?", "git_search", "hard"),
    ("Who authored the last commit?", "git_search", "hard"),
    ("What changed in the last commit?", "git_search", "hard"),
    (
        "Show me the commit log of the last commit that changed this part.",
        "clarify",
        "hard",
    ),
    ("Fix the bug in route.py", "refuse", "hard"),
    (
        "What is the commit message of the last commit that changed harness/route.py?",
        "git_search",
        "hard",
    ),
    ("What is in the unstaged diff for extract.py?", "diff_search", "hard"),
    ("Show the staged diff", "diff_search", "hard"),
    (
        "What is the last commit message and what is in the unstaged diff?",
        "multi_hop",
        "hard",
    ),
    ("What is the PREFILL string?", "dir_search", "hard"),
    ("What is MAX_TOKENS?", "dir_search", "hard"),
    (
        "What is the PREFILL string and what is MAX_TOKENS?",
        "multi_hop",
        "hard",
    ),
    # soft — overview / web facts / multi-hop / chitchat
    ("What is this project about?", "dir_search", "soft"),
    ("How does the client guard against repetition?", "dir_search", "soft"),
    ("Who is the CEO of Apple?", "web_search", "soft"),
    ("What is the capital of France?", "web_search", "soft"),
    ("What company is Alphabet?", "web_search", "soft"),
    (
        "Who is the CEO of Apple and what is the capital of France?",
        "multi_hop",
        "soft",
    ),
    (
        "Who is the CEO of Apple and where was he born?",
        "multi_hop",
        "soft",
    ),
    (
        "Who is the CEO of Apple and what is the PREFILL string?",
        "refuse",
        "soft",
    ),
    ("Who is the current CEO of LVMH?", "web_search", "soft"),
    ("hello there", "refuse", "soft"),
]
