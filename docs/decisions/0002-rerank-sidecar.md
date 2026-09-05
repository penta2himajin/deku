# ADR 0002: MiniCPM-Reranker as optional HTTP sidecar

## Status

Accepted (product ranking path when configured).

## Context

Lexical `generic_hit_score` in `web_search` accumulated many question-shape
coefficients. OpenBMB ships MiniCPM-Reranker(-Light) for RAG re-ranking, but
loading torch / transformers inside agent packages would break the
HTTP-indifference boundary (ADR 0001).

Measured probe (`scratch/rerank_ab_probe.py`): Reranker-Light matched lexical
top-1 on held unit fixtures (6/6) on Apple Silicon via CrossEncoder.

## Decision

1. **Product ranking premise** = MiniCPM-Reranker sidecar over HTTP when
   `DEKU_RERANK_URL` is set.
2. **Harness** talks only HTTP (`deku.rerank`); no torch in agent imports.
3. **Sidecar** (`deku-rerank` / `deku.rerank_serve`) may load
   `openbmb/MiniCPM-Reranker-Light` (or override via `DEKU_RERANK_MODEL`).
4. **Lexical rank remains** the offline / CI / fallback path when the sidecar
   is unset or unreachable.
5. Optional deps live under `uv sync --extra rerank` — not the default sync.

## Consequences

- Clone-and-run without the sidecar stays green (lexical).
- Smokes that claim product ranking quality should set `DEKU_RERANK_URL`.
- Future work can thin lexical coefficients once live smokes are measured on
  the rerank path.
- Commercial use of MiniCPM-Reranker weights must follow MiniCPM Model License.
