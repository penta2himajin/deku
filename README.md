# deku

[日本語](./README.ja.md)

A **MiniCPM5-1B–oriented** local task harness: small factual jobs (web facts, URL
summaries, repo lookup, git/diff, weak multi-step plans) with **explicit
refusals** when the request is out of scope. The model only does short grounded
completions; planning and tool choice live in code.

> **Status:** greenfield. Layout and serve wiring are specified here; agent code
> is being ported from experiments in 
> (runtime / eval warehouse — not a dependency at runtime).

## Design in one page

| Layer | Responsibility |
| --- | --- |
| **Agent / harness** | Route, refuse, tools, weak multi-hop / orchestrate, integrate answers |
| **LLM client** | OpenAI-compatible HTTP (`/v1/chat/completions`) only |
| **Default serve** | Official **GGUF** + `llama-server` (see below) |

The harness **does not care** how the HTTP API is implemented (llama.cpp, oMLX,
deku-serve, …). Semantics are tuned for **MiniCPM5-1B** (English-first,
no free-form long reasoning, no code authoring).

### Default backend (GGUF)

- Weights: [`openbmb/MiniCPM5-1B-GGUF`](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF)
- Recommended quant: **`MiniCPM5-1B-Q4_K_M.gguf`** (~657 MB)
- Server: vanilla [`llama.cpp`](https://github.com/ggerganov/llama.cpp) `llama-server`
  with `--jinja` (matches [OpenBMB’s llama.cpp cookbook](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/llama_cpp.md))

Any other OpenAI-compatible MiniCPM5-1B endpoint can be selected via env vars;
MLX is an optional alternate path for Apple Silicon developers, not the product
default.

## Intended capabilities (target)

- Public short facts (`web_search`)
- Summarize a given URL or local README (`url_read` / `dir_search` + hierarchical summary)
- Repo constants / overview (`dir_search`)
- Last commit / blame-style questions (`git_search`); staged/unstaged diffs (`diff_search`)
- Weak multi-step: rule-built plans, optional pronoun bind across hops, numbered or paragraph integrate
- **Refuse with a reason**: math, code authoring, chitchat, open-ended essays / proofs

## Non-goals

- General coding agent / SWE-bench runner
- Model-authored multi-step “chain of thought” planning
- Japanese as a supported input language (MiniCPM5-1B is EN/ZH; JA loops)
- Bundling MLX conversion stacks or large eval matrices from prior experiments

## Planned layout

```
deku/           # Python package: llm client, route, tools, orchestrate
bin/deku-serve  # thin wrapper: ensure GGUF → llama-server
bin/deku        # CLI: ask / dispatch
tests/          # unittest
docs/           # architecture, roadmap, decisions
evals/          # small fixed demo smokes only (not a research warehouse)
```

## Docs

| Doc | Purpose |
| --- | --- |
| [docs/architecture.md](./docs/architecture.md) | Boundaries, HTTP contract, GGUF default |
| [docs/roadmap.md](./docs/roadmap.md) | Phased build plan (0→4) |
| [docs/decisions/0001-gguf-default-serve.md](./docs/decisions/0001-gguf-default-serve.md) | Why GGUF + llama-server |


## License

MIT. See `LICENSE`. MiniCPM5 model weights are under their own licenses on Hugging Face.
