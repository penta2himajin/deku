---
name: deku-ask
description: >-
  Call the local deku factual harness (web/repo/git/diff/url, weak multi-hop)
  via CLI with agent JSON envelope. Use for short grounded facts, PREFILL/MAX_TOKENS,
  commit/diff lookups, CEO/capital-style web facts, or joined factual questions—
  not for coding, editing, or open-ended reasoning.
---

# deku-ask (project)

Parent agent → deku as a **read-only fact worker**. Planning stays in the parent;
deku routes, retrieves, refuses, or returns short grounded answers.

## When to use

- Repo constants / README / overview (`dir_search`)
- Last commit / unstaged diff (`git_search` / `diff_search`)
- Short public web facts or URL summarize
- Two–three factual clauses joined with `and who/what/when…`

## When NOT to use

- Writing or fixing code, commits, PRs
- Math, essays, opinions, “should I…”
- Vague deixis without a path (“this part”)

## How to call

From the deku repo root (mise + synced `.venv`):

```bash
mise exec -- uv run deku ask --json --audience agent --root . "QUESTION"
```

Offline lexical refuse only:

```bash
mise exec -- uv run deku ask --json --audience agent --no-live "What is 2+2?"
```

Needs `mise run serve` (or equivalent) for live MiniCPM answers on retrieval paths.

## Envelope (trust these fields)

Parse JSON stdout:

| Field | Use |
| --- | --- |
| `status` | `ok` / `partial` / `refused` / `cannot_answer` / `clarify` |
| `tool` | tool or `multi_hop` / `refuse` |
| `answer` | short text, or `refused:<reason>` when audience=agent |
| `reason` | refuse reason code |
| `plan_id` | multi-hop plan label |
| `cores` | bindable cores from hops |
| `failed_steps` | failed clauses |
| `next_hint` | e.g. `retry_failed_clauses`, `ask_in_scope_fact` |

Treat `refused` / `cannot_answer` as normal outcomes—do not retry as if the CLI crashed.

## Question style

One job per call; English factual phrasing; ≤3 joined clauses.
