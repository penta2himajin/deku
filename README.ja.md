# deku

> Source: README.md @ uncommitted
>
> [English](./README.md)

**MiniCPM5-1B 向け**のローカル・タスク・ハーネス。短い事実タスク（Web 事実、URL 要約、リポ照会、git/diff、弱い多段プラン）をコード側で回し、範囲外は**理由付きで拒否**する。モデルは短い grounded 完了だけを担当する。

> **現状:** Phase 2 エージェント芯（`deku ask` / route / refuse / tools）まで。能力スモークは既定 GGUF 上で測る。

## 設計（1 ページ）

| 層 | 責務 |
| --- | --- |
| **エージェント / ハーネス** | ルート、拒否、ツール、弱い多段、答の統合 |
| **LLM クライアント** | OpenAI 互換 HTTP のみ |
| **既定 serve** | 公式 **GGUF** + `llama-server` |

HTTP の向こうの実装（llama.cpp / oMLX / 他）は**関知しない**。セマンティクスは MiniCPM5-1B 向け（英語前提、自由な長推理・コード生成はしない）。

### 既定バックエンド（GGUF）

- 重み: [`openbmb/MiniCPM5-1B-GGUF`](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF)
- 推奨量子化: **`MiniCPM5-1B-Q4_K_M.gguf`**
- サーバ: `llama-server … --jinja`（OpenBMB の llama.cpp 手順に合わせる）

詳細は [docs/architecture.md](./docs/architecture.md)、進め方は [docs/roadmap.md](./docs/roadmap.md)。

## クイックスタート

[mise](https://mise.jdx.dev/)（Python + [uv](https://docs.astral.sh/uv/)）と PATH 上の `llama-server` が必要です。

```bash
curl https://mise.run | sh    # または brew install mise
mise trust && mise install
mise run sync
brew install llama.cpp        # macOS
mise run serve
```

テスト: `mise run test`。診断: `mise run doctor`。

## 非目標

- 汎用コーディングエージェント / SWE-bench
- モデルにプランを書かせる CoT
- 日本語入力の公式サポート（`non_english` で拒否。英語で尋ねること）
- 大きな研究用 eval 行列や MLX 変換スタックの持ち込み

## ライセンス

Apache-2.0（`LICENSE`）。モデル重みは Hugging Face 上の各ライセンスに従う。
