# Roadmap

Phased plan for standing up deku. Adjust dates as needed; keep phase order.

## Phase 0 — Product face (half day)

- [x] Replace templates boilerplate with deku README / AGENTS (this repo).
- [x] First commit on `main`.
- [x] State MiniCPM5-1B-only product intent + GGUF default serve + HTTP indifference.
- [ ] Optional: open `session-handoff` issue for the workstream.

**Done when:** a stranger can read README and know what deku is / is not.

## Phase 1 — GGUF serve wiring (1–2 days)

- [x] `bin/deku-serve`: download official `MiniCPM5-1B-Q4_K_M.gguf` if needed; launch `llama-server --jinja`.
- [x] `deku/llm.py` (or equivalent): OpenAI-compatible `complete()`.
- [x] Smoke: `curl` chat completions against localhost (needs `llama-server` on PATH).
- [x] Document PATH requirement for `llama-server` / `huggingface-cli`.

**Done when:** clone + install llama.cpp tools + `./bin/deku-serve` yields a working API (no agent yet).

Toolchain polish: `mise.toml` + `uv.lock` + `mise run {sync,test,serve,doctor}`.

## Phase 2 — Agent core port (several days)

- [x] Package layout under `deku/`.
- [x] Port whitelist from prior experiments (see [architecture.md](./architecture.md)); TDD.
- [x] CLI `deku ask "…"`.
- [ ] Re-run capability smokes **on GGUF** (refuse / summarize / multi-hop / controls). Do not copy MLX scores.

**Done when:** unit tests green; demo smokes pass against default serve.

## Phase 3 — Demo pack (1–2 days)

- [ ] Scripted demo script (CEO, URL summary, dependent hop, math refuse).
- [ ] Tiny `evals/` fixed set + result JSON.
- [ ] Short architecture diagram in README (harness plans → MiniCPM reads).
- [ ] Honest limitations (EN/ZH, no JA, no CoT planner).

**Done when:** a 2-minute walkthrough is reproducible from README alone.

## Phase 4 — Publish

- [x] Host the repository (private or public) with a reproducible README path.

**Done when:** external party can run the default path from the README alone.

## Explicit non-phases (defer)

- Dual-maintained MLX + GGUF serve in-tree
- SWE-bench / Ling coding loops inside deku
- Model-generated JSON plans without code validation
