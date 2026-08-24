"""CLI entry: ``deku ask "…"`` → route + tool dispatch."""

from __future__ import annotations

import argparse
import json
import sys

from deku import route as rt


def ask(
    question: str,
    *,
    root: str = ".",
    router: str = "rule",
    seed: int = 0,
    live: bool = True,
    as_json: bool = False,
    use_needle_slots: bool = False,
    audience: str | None = None,
) -> int:
    got = rt.dispatch(
        question,
        router=router,
        seed=seed,
        root=root,
        live_answer=live,
        use_needle_slots=use_needle_slots,
        audience=audience,
    )
    if as_json:
        print(
            json.dumps(
                rt.envelope(got),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        text = (got.answer or "").strip()
        if not text:
            text = f"[{got.status}] via {got.tool}"
        print(text)
    return 0 if got.status in ("ok", "refused", "clarify", "partial") else 1


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="deku",
        description="MiniCPM5-1B–oriented local task harness",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ask_p = sub.add_parser("ask", help="Route a question and run the chosen tool")
    ask_p.add_argument("question", help="Natural-language question")
    ask_p.add_argument(
        "--root",
        default=".",
        help="Repo root for dir/git/diff tools (default: .)",
    )
    ask_p.add_argument(
        "--router",
        default="rule",
        choices=("rule", "needle"),
        help="Tool router (needle needs the optional Needle package)",
    )
    ask_p.add_argument(
        "--needle-slots",
        action="store_true",
        help=(
            "Optional: use Needle only to classify answer slot type "
            "(date|place|person|…) when rules return none; never for answers"
        ),
    )
    ask_p.add_argument("--seed", type=int, default=0)
    ask_p.add_argument(
        "--no-live",
        action="store_true",
        help="Skip MiniCPM calls; lexical / template paths only",
    )
    ask_p.add_argument(
        "--json",
        action="store_true",
        help="Print stable envelope JSON (status/tool/cores/next_hint/…)",
    )
    ask_p.add_argument(
        "--audience",
        default=None,
        choices=("human", "agent"),
        help=(
            "Refuse wording: human prose or agent reason codes "
            "(default: DEKU_AUDIENCE or human)"
        ),
    )

    args = p.parse_args(argv)
    if args.cmd == "ask":
        raise SystemExit(
            ask(
                args.question,
                root=args.root,
                router=args.router,
                seed=args.seed,
                live=not args.no_live,
                as_json=args.json,
                use_needle_slots=args.needle_slots,
                audience=args.audience,
            )
        )
    p.error(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
