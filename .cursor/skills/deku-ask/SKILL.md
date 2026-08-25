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

`--json` is **slim** by default (no nested `detail`). For harness debug only:

```bash
mise exec -- uv run deku ask --json-full --audience agent --root . "QUESTION"
# or: DEKU_JSON_FULL=1 … --json …
```

Offline lexical refuse only:

```bash
mise exec -- uv run deku ask --json --audience agent --no-live "What is 2+2?"
```

Needs `mise run serve` (or equivalent) for live MiniCPM answers on retrieval paths.

## Envelope (trust these fields)

Parse JSON stdout. Prefer slim (`envelope: "slim"`). Do **not** paste raw JSON
wholesale into the next turn — lift only the carry set below.

| Field | Use |
| --- | --- |
| `envelope` | `slim` (default) or `full` |
| `status` | `ok` / `partial` / `refused` / `cannot_answer` / `clarify` |
| `tool` | tool or `multi_hop` / `refuse` |
| `answer` | short text (includes path when known), or `refused:<reason>` |
| `reason` | refuse reason code |
| `plan_id` | multi-hop plan label |
| `cores` | bindable cores from hops |
| `locations` | `[{path, ident?, value?, kind?}, …]` — assignment / definition / prose |
| `failed_steps` | failed clauses |
| `next_hint` | machine next step — see actions below |
| `detail` | **only with `--json-full`** — retrieval internals; do not carry forward |

Treat `refused` / `cannot_answer` as normal outcomes—do not retry as if the CLI crashed.

### Carry forward (keep context small)

Into the next parent turn, keep at most:

- `status`
- `answer` (one short line; skip if `locations` / `cores` already hold the fact)
- `locations` and/or `cores`
- `next_hint.action` (plus `clauses` / `reason` / `abstain_reason` when present)

On `ok`, prefer `locations` or `cores` over re-quoting a long `answer`.
Never retain `detail`, hit snippets, or full stdout dumps.

### `next_hint.action` (branch on these)

| action | Meaning |
| --- | --- |
| `none` | done / ok |
| `ask_in_scope_fact` | refused — rephrase as a short in-scope fact |
| `provide_path` | clarify — add a file path (or path-scoped git ask) |
| `provide_url` | missing URL |
| `retry_failed_clauses` | partial multi-hop — retry `clauses` only |
| `name_symbol_or_path` | repo ask lacked a clear symbol/path |
| `narrow_or_rephrase` | weak / off-topic hit — tighter wording |
| `check_workdir` | no diff / empty working tree |
| `retry_or_other_url` | fetch failed |
| `abstain_or_narrow` | cannot answer — do not invent; narrow or stop |
| `enable_live_or_serve` | offline skip — start serve / drop `--no-live` |

## Question style

One job per call; English factual phrasing; ≤3 joined clauses.

Good locate asks: `Where is PREFILL set?`, `Where is find_assignment defined?`,
`What is this project about?`, `Where is this project described?`
