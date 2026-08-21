# ADR 0001: GGUF + llama-server as default serve

## Status

Accepted (portable default for clone-and-run).

## Context

deku is a MiniCPM5-1B–oriented task harness. Development experiments used MLX
(4-bit) via local MLX servers on Apple Silicon. For a repo others can clone, the
default backend should be portable and aligned with upstream deployment docs.

The upstream project publishes [`openbmb/MiniCPM5-1B-GGUF`](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF)
and documents `llama-server` with `--jinja` in their llama.cpp cookbook.

## Decision

1. **Default serve** = official GGUF (`MiniCPM5-1B-Q4_K_M.gguf` recommended) +
   vanilla `llama-server`.
2. **Harness** talks only OpenAI-compatible HTTP; it does not embed GGUF/MLX.
3. **Alternate backends** (MLX, oMLX) remain env overrides for developers; not
   dual first-class in-tree servers in v1.
4. **Claims and smokes** for the product default are measured on the GGUF path.

## Consequences

- Broader OS/hardware reach; closer to the upstream public deployment story.
- Quality/latency may differ from MLX 4-bit; must re-eval after port.
- Maintainers must document installing `llama-server` (PATH), not commit binaries.
- Temptation to “also ship MLX” is deferred to keep the narrative single.
