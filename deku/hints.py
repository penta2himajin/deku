"""Machine-oriented ``next_hint`` for parent agents after a deku outcome.

Hints are small action codes — not prose plans. Parents should branch on
``action`` and ignore unknown keys forward-compatibly.
"""
from __future__ import annotations

_ABSTAIN_ACTIONS: dict[str, str] = {
    "identifiers_missing_from_top_hit": "name_symbol_or_path",
    "weak_or_off_topic_hit": "narrow_or_rephrase",
    "no_prose_lead": "narrow_or_rephrase",
    "no_grounded_core": "abstain_or_narrow",
    "empty_reply": "abstain_or_narrow",
    "core_without_sentence": "abstain_or_narrow",
    "no_diff": "check_workdir",
    "no_commits_for_path": "provide_path",
    "no_url": "provide_url",
    "fetch_error": "retry_or_other_url",
    "not_found": "retry_or_other_url",
    "empty_page": "retry_or_other_url",
    "no_lexical_sentence": "narrow_or_rephrase",
}


def next_hint_for(
    *,
    status: str,
    tool: str | None = None,
    reason: str | None = None,
    abstain_reason: str | None = None,
    failed: list[dict] | None = None,
) -> dict:
    """Return a stable next_hint dict for envelope / dispatch detail."""
    st = (status or "").strip().lower()
    failed = failed or []

    if st in ("ok", ""):
        return {"action": "none"}

    if st == "partial" and failed:
        return {
            "action": "retry_failed_clauses",
            "clauses": [f.get("query") for f in failed if f.get("query")],
        }

    if st == "clarify":
        return {
            "action": "provide_path",
            "reason": reason or "path",
        }

    if st == "refused":
        return {
            "action": "ask_in_scope_fact",
            "reason": reason or "out_of_scope",
        }

    if st == "skipped_offline":
        return {
            "action": "enable_live_or_serve",
            "failed_tool": tool,
        }

    if st == "cannot_answer":
        action = _ABSTAIN_ACTIONS.get(
            abstain_reason or "", "abstain_or_narrow"
        )
        out: dict = {
            "action": action,
            "failed_tool": tool
            or (failed[0].get("tool") if failed else None),
        }
        if abstain_reason:
            out["abstain_reason"] = abstain_reason
        if failed and failed[0].get("query"):
            out["failed_query"] = failed[0].get("query")
        return out

    return {"action": "none"}
