# Architecture

Engineering notes for deku. Product intro: [README.md](../README.md).

## Roles

### Harness (this repo’s core)

Deterministic (or lightly lexical) machinery:

- **Route** — pick a tool or `refuse` (bare ALLCAPS / project name via
  discovery under `--root`; soft dir cues stay product-agnostic overview words)
- **Refuse** — fixed English reasons (math, code, chitchat, deep_reasoning, out_of_scope)
- **Tools** — `web_search`, `dir_search`, `git_search`, `diff_search`, `url_read`,
  `calc` (structured ops only, e.g. `years_since`; free-form math stays refuse)
- **Weak multi-step** — propose clause→tool steps then validate (allow-list, ≤3 steps, bind rules; tools include `url_read` / `calc`); optional bind of prior hop core; integrate; expose `cores` / `next_hint` for parent agents (`deku.hints.next_hint_for` maps status + abstain_reason → action codes)
- **Audience** — refuse prose for humans, `refused:<reason>` codes for agents (`--audience` / `DEKU_AUDIENCE`)
- **Lexical extractors** — named surface extractors (`extract_date`, `extract_person`, …) pull grounded spans from a document; question cues choose which to try. Special cases may be formalized as generic tools (e.g. named age → `web_search` then `calc`). Forbidden across **all** tools: closed gloss tables; product control via POS / noun-class / slot labels; shape-specialized reply shortcuts embedded in a tool
- **Optional Needle** — tool routing only; never free-form answers or plans. Product smokes measure the rule path without Needle
- **Optional rerank sidecar** — MiniCPM-Reranker over HTTP when `DEKU_RERANK_URL` is set (`deku.rerank` client only; torch stays in `deku-rerank`). Lexical rank is the fallback. See [ADR 0002](decisions/0002-rerank-sidecar.md)
- **Hierarchical summary** — map/reduce with extractive leaf anchors (MiniCPM only compresses short notes)

The model is never the planner of record.

### LLM client

Single module responsibility: call an OpenAI-compatible chat completions API and return text. Configuration via environment, for example:

| Variable | Role | Default (planned) |
| --- | --- | --- |
| `DEKU_URL` | API base (no trailing `/v1` or with — pick one convention and stick to it) | `http://127.0.0.1:8080` |
| `DEKU_MODEL` | Model id string the server expects | `MiniCPM5-1B` / GGUF path as required by llama-server |
| `DEKU_API_KEY` | Optional bearer | empty / `unused` |
| `DEKU_RERANK_URL` | MiniCPM-Reranker sidecar base (no path) | unset → lexical rank |
| `DEKU_RERANK_TIMEOUT` | Rerank HTTP timeout seconds | `30` |

The client **must not** import llama.cpp, MLX, or GGUF parsers.

### Default serve (convenience)

`bin/deku-serve` (planned) wraps:

1. Ensure `MiniCPM5-1B-Q4_K_M.gguf` is present (download from `openbmb/MiniCPM5-1B-GGUF` if missing).
2. Exec `llama-server -m … --host 127.0.0.1 --port 8080 --jinja` (flags aligned with [OpenBMB llama.cpp docs](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/llama_cpp.md)).

Binary acquisition: document installing `llama-server` on PATH; do not vendor large prebuilt blobs in git.

## Boundary rules

1. Agent code depends only on the HTTP completion contract.
2. Product tuning (prompts, refuse cues, chunk sizes, English-only) may assume MiniCPM5-1B.
3. Alternate backends (oMLX, other OpenAI-compatible servers) are env overrides, not second first-class code paths in v1.
4. No silent fallback from refuse → web_search for deep/essay prompts.
5. Eval numbers claimed in docs must be measured on the **default GGUF** path unless labeled otherwise.

## In-scope agent modules

**In scope:** `route`, `refuse`, `render`, `web_search`, `dir_search`, `git_search`, `diff_search`, `url_read`, `calc`, `multi_hop`, `hier_summary`, `orchestrate`, `lexical_core`, `rerank` (HTTP client), small related unit tests.

**Parent-agent contract:** `deku ask --json --audience agent` returns a **slim**
envelope (`envelope: "slim"`): `status`, `tool`, `answer`, `reason`, `plan_id`,
`cores`, `locations`, `failed_steps`, `next_hint` (see `route.envelope`). Nested
`detail` is omitted by default so parents do not paste retrieval internals into
context. Use `--json-full` or `DEKU_JSON_FULL=1` for the previous full payload.
Human asks get the same facts as short English with file paths when known. Locate patterns:

- assignment: `PREFILL is set to "ANSWER: " in deku/extract.py.`
- definition: `find_assignment in deku/dir_search.py: …`
- prose / overview: lead sentence plus `(see README.md)` or `README.md says: …`

`locations` entries may include `kind` (`definition` / `prose`) plus
`path` / `ident` / `value` as available.

**Out of scope for deku:** `swebench*`, large coding-agent A/B farms, research training loops, ranking experiments, a separate MCP extract server (keep extract helpers in-process only).

## Weak multi-step (definition)

“Weak” / pseudo multi-step inference means:

- Code selects or builds a short plan (`[{tool, query, bind_prior?}, …]`).
- Steps run in order; a later step may rewrite pronouns using the prior hop’s core.
- Integration is templated (list or short paragraph), not free-form debate.

This is an **orchestrator**, not model chain-of-thought.
