# deku

## Overview

MiniCPM5-1B–oriented local **task harness** plus a thin OpenAI-compatible
**client**. Code routes tools, refuses out-of-scope work, and runs weak
multi-step plans; the model only returns short grounded completions.

Product docs: @README.md, @docs/architecture.md, @docs/roadmap.md.
Decision: @docs/decisions/0001-gguf-default-serve.md.

## Project Structure

```
deku/           # Python package (llm client, serve helpers; route/tools next)
bin/            # deku-serve (GGUF + llama-server)
tests/          # unittest
docs/           # architecture, roadmap, ADRs, handoff/i18n policy
evals/          # small fixed demo smokes only (not a research warehouse)
mise.toml       # python + uv pins and tasks
```

Generated / downloaded artifacts (GGUF weights, llama.cpp builds) stay **out of
git**; document download paths in README / `bin/deku-serve`.

## Development Setup

Preferred (mise + uv):

```bash
# https://mise.jdx.dev/getting-started.html
curl https://mise.run | sh   # or brew install mise
mise trust && mise install   # python 3.12 + uv
mise run sync                # uv sync → .venv + editable deku
mise run doctor              # PATH / GGUF readiness
# llama-server on PATH (not vendored):
brew install llama.cpp       # macOS; or build ggml-org/llama.cpp
mise run serve               # download GGUF once, exec llama-server --jinja
```

Fallback without mise:

```bash
# Install uv: https://docs.astral.sh/uv/
uv sync
uv run python -m unittest discover -s tests
uv run deku-serve
```

```bash
cp git-hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push
# or: git config core.hooksPath git-hooks
```

## Build & Test

```bash
mise run test
# or: uv run python -m unittest discover -s tests
```

## Development Principles

- **Harness plans; MiniCPM reads.** No model-authored tool plans without code validation.
- **Product default = official GGUF + `llama-server --jinja`.** Measure smokes on that path.
- **HTTP indifference:** agent code must not import llama.cpp / MLX / GGUF parsers.
- Port from prior experiments by **whitelist only** (see architecture.md). Do not drag SWE-bench / Ling farms in.
- English-first; do not promise Japanese UX.

## Architectural Boundaries

1. `deku` agent packages depend only on an OpenAI-compatible completion client.
2. `bin/deku-serve` may shell out to `llama-server` / downloaders; the agent must not.
3. Engineering docs and ADRs stay English-only (`docs/i18n-policy.md`).
4. ADRs under `docs/decisions/` for serve/backend and port-boundary decisions.

## Prohibitions

1. Do not vendor MiniCPM weights or `llama-server` binaries in git.
2. Do not make MLX the documented product default without a new ADR reversing 0001.
3. Do not add silent “deep reasoning” or code-authoring paths that bypass `refuse`.
4. Do not grow `evals/` into a research warehouse; keep a small fixed demo set.

## Git Conventions

- Conventional Commits as in common rules; prefer `feat:`, `docs:`, `test:` early on.
- Branch prefix: `claude/`, `codex/`, or `human/` + topic.

## Session Handoff

Long-running workstreams use GitHub issues for cross-session continuity. See `docs/handoff-protocol.md`.

- Label: `session-handoff`
- One issue per workstream (not per session)
- On session start, read the relevant handoff issue and confirm the **Next action** with the user before executing.

## Internationalisation

Follow `docs/i18n-policy.md`:

- Suffix files (`README.ja.md` next to `README.md`); no language directories.
- Only `README.md` and user-facing intro docs are in scope. Engineering docs and ADRs stay English-only.
- Each translated file carries a `> Source: <name>.md @ <sha>` header.

---

<!-- Common rules below this line apply to every project. -->

## Common Development Rules

### TDD (Red → Green → Refactor)

All implementation work proceeds in this cycle:

1. **Red**: write a failing test that captures the intended behaviour.
2. **Green**: write the minimum code that makes the test pass.
3. **Refactor**: tidy up while keeping tests green.

When a test fails, fix the production code — do not delete, skip, or weaken the test.

### Measure, Don't Conjecture

Base decisions on observed data, not assumptions. Before optimising, claiming a bottleneck, or asserting that something is slow or broken, measure it — profile, benchmark, log, or reproduce. When you report a cause, cite the measurement that supports it.

### Git Conventions

- **Conventional Commits**: `feat:` `fix:` `docs:` `refactor:` `test:` `ci:` `chore:`. Project-specific prefixes (e.g. `data:`, `experiments:`) live in the project's `AGENTS.md`.
- **Branch naming**: use a short prefix for the agent or author followed by a topic, e.g. `claude/<topic>`, `codex/<topic>`, or `human/<topic>`.
- **Trailer**: when an AI agent authors the commit, append a trailer crediting the agent. Do not embed model name or session info in the trailer; put those in the commit body if needed.
- **Pre-push hook**: install via `cp git-hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push` (or `git config core.hooksPath git-hooks`). The hook runs format / lint / clippy before every push. Tests are intentionally omitted — TDD keeps them green at commit time.

### Pull Requests

- **Always ready for review.** Open PRs in the "ready" state, never as drafts. Draft PRs do not fire review-requested events and slow the loop.
- **Auto-subscribe after creating a PR.** Immediately after the PR is created, subscribe to its activity without asking the user. Rationale: the user explicitly opted into the "agent opens and watches its own PRs" workflow at the template level, so the per-PR confirmation is noise. Unsubscribe only when the user says to stop, when the PR merges, or when it is closed unmerged.
- **One PR per workstream**, matching the handoff issue. Reference the issue with `Closes #N` per `.github/PULL_REQUEST_TEMPLATE.md`.

### Stream Idle Timeout Mitigation

Cloud agent sessions occasionally fail with `Stream idle timeout - partial response received` on long output. To reduce risk:

1. **Stage long writes.** For long documents or source files, write the skeleton (headings, function signatures, trait stubs) first, then fill each section in follow-up edits. Avoid single blocks larger than ~200 lines.
2. **Watch out after large reads.** Reading a big file (e.g. `Cargo.lock`, large generated modules) and then immediately producing long output is a common trigger. Split into separate turns or excerpt only the relevant portion.
3. **Recover carefully.** A timeout can still leave the file write completed. Run `git status` before retrying so the same content is not written twice.

### Common Prohibitions

1. Do not delete, skip, or comment out existing tests.
2. Do not modify CI configuration without explicit instruction.
3. Do not weaken production code merely to make tests pass.
4. Do not commit credentials, API keys, signed URLs, or anything in `.env*`.
