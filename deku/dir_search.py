"""dir_search: directory corpus → rank chunks → MiniCPM grounded reply.

No Needle / LFM. Ranking stays lexical (measured: MiniCPM does not earn
the file-picker stage). Answer path: assignment / def first, else the same
extract → summarize → grounding / abstain stack as web_search.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from deku import extract
from deku import web_search as ws
from deku import render

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".yml", ".yaml",
    ".json", ".sh", ".rs", ".mdc",
}
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build",
}
SKIP_PATH_PREFIXES = (
    "evals/",  # probes embed foreign Q&A; not part of the product corpus
    "tests/",  # fixtures quote identifiers and poison ranking
)
PROSE_PATHS = ("README.md", "AGENTS.md", "CLAUDE.md", "README.ja.md")
MAX_FILE_BYTES = 200_000
MAX_CHUNK_CHARS = 900

DIR_SEARCH_CUES = re.compile(
    r"(?i)\b(project|purpose|overview|harness|readme|about|how does|"
    r"configure|default|timeout|loop|repeat)\b"
)


@dataclass
class Result:
    intent: str
    query: str = ""
    hits: list[dict] = field(default_factory=list)
    document: str = ""
    answer: str | None = None
    status: str = ""
    detail: dict = field(default_factory=dict)


def rule_intent(question: str) -> str:
    if ws.NONSEARCH.search(question or ""):
        return "refuse"
    if ws.SEARCH_CUES.search(question or "") or DIR_SEARCH_CUES.search(question or ""):
        return "search"
    return "refuse"


def rule_query(question: str) -> str:
    return ws.rule_query(question)


def question_identifiers(question: str) -> list[str]:
    """ALLCAPS / snake_case tokens the hit must preferably contain."""
    q = question or ""
    found = re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", q)
    found += re.findall(r"\b[a-z_][a-z0-9_]{3,}\b", q)
    stop = {
        "what", "when", "where", "which", "does", "this", "that", "with",
        "from", "into", "about", "string", "value", "file", "code", "detect",
        "project", "purpose", "overview", "default", "harness",
    }
    out = []
    seen = set()
    for t in found:
        low = t.casefold()
        if low in stop or low in seen:
            continue
        seen.add(low)
        out.append(t)
    return out


# Product doc basenames that look ALLCAPS but are not code constants.
_DOC_ALLCAPS = frozenset({"README", "AGENTS", "CLAUDE", "LICENSE", "CHANGELOG"})


def corpus_mode(question: str) -> str:
    """code = ALLCAPS(len≥4) / snake_case idents; prose = overview without them."""
    idents = question_identifiers(question)
    codeish = [
        i for i in idents
        if (
            ((i.isupper() and len(i) >= 4) or "_" in i or i.endswith((".py", ".md")))
            and i.casefold() not in {d.casefold() for d in _DOC_ALLCAPS}
            and not i.lower().endswith((".md",))
        )
    ]
    # README / overview questions stay prose even if "README" was mentioned.
    if re.search(
        r"(?i)\b(readme|overview|purpose|this project|about (?:this|deku|the project))\b",
        question or "",
    ):
        # Still code mode when a real assignment-style constant remains.
        real_code = [
            i for i in codeish
            if "_" in i or (i.isupper() and i not in _DOC_ALLCAPS and len(i) >= 4)
        ]
        if not real_code:
            return "prose"
    return "code" if codeish else "prose"


def prose_lead_sentence(document: str, question: str | None = None) -> str | None:
    """Best substantial sentence from a README/docs pack (question-aware)."""
    candidates = []
    for line in (document or "").splitlines():
        s = line.strip()
        if not s or s.lower().startswith("source:") or s.startswith("#"):
            continue
        if s.startswith("```") or s.startswith("|"):
            continue
        plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
        plain = re.sub(r"[*`]", "", plain)
        path_map = bool(re.match(r"^\S+\s+#\s+\S", plain))
        min_words = 4 if path_map else 6
        if len(plain.split()) < min_words:
            continue
        sent = re.split(r"(?<=[.!?])\s+", plain, maxsplit=1)[0].strip()
        if len(sent.split()) < min_words:
            continue
        score = float(extract.term_score(question or "", sent)) if question else 0.0
        if re.search(r"(?i)\b(harness|local-LLM|MiniCPM)\b", sent):
            score += 2.0
        if question and re.search(r"(?i)\b(about|overview|purpose|project)\b", question):
            if re.search(r"(?i)\b(harness|MiniCPM|local-LLM)\b", sent):
                score += 6.0
            if path_map or re.search(r"(?i)\bbin/", sent):
                score -= 4.0
        if question and re.search(r"(?i)\b(server|wrapper|mlx)\b", question):
            if re.search(r"(?i)\bbin/|deku-serve|llama-server|mlx_lm\.server", sent):
                score += 8.0
            if path_map and re.search(r"(?i)\bbin/", sent):
                score += 6.0
            if re.search(r"(?i)\bwhere\b", question) and not re.search(
                r"(?i)\bbin/|deku-serve|llama-server", sent
            ):
                score -= 4.0
        if question and re.search(r"(?i)\b(models?|minicpm|llms?)\b", question):
            if re.search(r"(?i)\b(MiniCPM|local-LLM|mlx_lm|4-bit)\b", sent):
                score += 8.0
            if path_map or re.search(r"(?i)\bbin/", sent):
                score -= 5.0
        candidates.append((score, len(sent), sent))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    return candidates[0][2]


def looks_where_ident(question: str) -> bool:
    """True when the ask is primarily about where a constant/symbol lives."""
    return bool(
        re.search(
            r"(?i)\b("
            r"where (?:is|are)|where (?:can I find|do I find)|"
            r"in which file|which file|defined in|set in"
            r")\b",
            question or "",
        )
    )


def path_near_line(document: str, assign_line: str) -> str | None:
    """Best file path header / Source: line above an assignment in packed notes."""
    lines = (document or "").splitlines()
    needle = (assign_line or "").strip()
    ident = needle.split("=", 1)[0].strip() if "=" in needle else needle
    hit_i = None
    for i, ln in enumerate(lines):
        if ident and re.search(rf"(?m)^\s*{re.escape(ident)}\s*=", ln):
            hit_i = i
            break
        if needle and ln.strip() == needle:
            hit_i = i
            break
    if hit_i is None:
        return None
    for j in range(hit_i, -1, -1):
        m = re.match(r"(?i)^Source:\s*file:(\S+)", lines[j].strip())
        if m:
            return m.group(1).split("#")[0].strip()
        m = re.match(
            r"^((?:[\w.-]+/)*[\w.-]+\.(?:py|md|json|txt|toml|sh|ya?ml|csv))\s*$",
            lines[j].strip(),
        )
        if m:
            return m.group(1)
    return None


def find_assignment(
    question: str, document: str
) -> tuple[str | None, str | None, str | None]:
    """If notes contain `IDENT = value`, return (line, value, path)."""
    for ident in question_identifiers(question):
        m = re.search(
            rf"(?m)^\s*{re.escape(ident)}\s*=\s*(.+)$", document or ""
        )
        if not m:
            continue
        raw = m.group(1).split("#")[0].strip().rstrip(",")
        if not raw:
            continue
        line = f"{ident} = {raw}"
        path = path_near_line(document, line)
        return line, raw, path
    return None, None, None


def find_definition(question: str, document: str) -> str | None:
    """First `def ident(...):` line for a snake_case identifier in the question."""
    for ident in question_identifiers(question):
        if not re.match(r"^[a-z_]", ident):
            continue
        m = re.search(
            rf"(?m)^(def\s+{re.escape(ident)}\b.*)$", document or ""
        )
        if m:
            return m.group(1).strip()
    return find_relevant_definition(question, document)


def find_relevant_definition(question: str, document: str) -> str | None:
    """def line in the notes whose name/docstring overlaps the question."""
    lines = (document or "").splitlines()
    for i, ln in enumerate(lines):
        m = re.match(r"^(def\s+([a-z_][a-z0-9_]*)\b.*)$", ln.strip())
        if not m:
            continue
        line, name = m.group(1), m.group(2)
        window = "\n".join(lines[i: i + 12])
        soft = name.replace("_", " ")
        if soft in (question or "").casefold() or name in (question or ""):
            return line
        if extract.term_score(question, window) >= 2:
            return line
    return None


def definition_reply(document: str, def_line: str) -> str:
    """Natural-language gloss from signature + docstring / comment."""
    lines = (document or "").splitlines()
    try:
        i = next(n for n, ln in enumerate(lines) if ln.strip() == def_line.strip())
    except StopIteration:
        return render.definition(def_line)
    gloss = None
    for ln in lines[i + 1: i + 10]:
        s = ln.strip()
        if s.startswith('"""') or s.startswith("'''"):
            quote = s[:3]
            body = s[3:]
            if body.endswith(quote):
                body = body[:-3].strip()
            else:
                chunks = [body] if body else []
                for ln2 in lines[i + 2: i + 14]:
                    t = ln2.strip()
                    if t.endswith(quote):
                        chunks.append(t[:-3].strip())
                        break
                    chunks.append(t)
                body = " ".join(c for c in chunks if c)
            if body:
                gloss = re.split(r"(?<=[.!?])\s+", body, maxsplit=1)[0].strip()
            break
        if s.startswith("#"):
            gloss = s.lstrip("#").strip()
            break
        if s.startswith("def ") or s.startswith("class "):
            break
    return render.definition(def_line, gloss)


def core_ok_for_dir(question: str, core: str | None, document: str = "") -> bool:
    """Dir answers are usually names/strings; reject tiny bare integers."""
    if not ws.core_fits_question(question, core):
        return False
    c = (core or "").strip()
    if re.fullmatch(r"\d{1,2}", c):
        for ident in question_identifiers(question):
            if re.search(rf"{re.escape(ident)}\s*=\s*{re.escape(c)}\b", document or ""):
                return True
        if not re.search(r"(?i)\b(port|seed|timeout|count|how many)\b", question or ""):
            return False
    return True


def should_abstain(
    *, question: str, doc: str, score: float, core: str | None
) -> bool:
    if score < ws.MIN_HIT_SCORE:
        return True
    if not core_ok_for_dir(question, core, doc):
        return True
    return False


def chunk_text(text: str, *, path: str, max_chars: int = MAX_CHUNK_CHARS) -> list[dict]:
    """Split on blank lines; markdown also splits before ATX headings."""
    raw = text or ""
    if path.endswith(".md"):
        raw = re.sub(r"(?m)^(#{1,6}\s)", r"\n\n\1", raw)
    parts = re.split(r"\n\s*\n", raw)
    chunks = []
    for i, part in enumerate(parts):
        block = part.strip()
        if not block:
            continue
        while len(block) > max_chars:
            cut = block[:max_chars].rsplit("\n", 1)[0] or block[:max_chars]
            chunks.append(_hit(path, cut, i))
            block = block[len(cut):].lstrip()
            i += 1
        if block:
            chunks.append(_hit(path, block, i))
    return chunks


def _hit(path: str, snippet: str, idx: int) -> dict:
    title = f"{path}#{idx}"
    return {
        "title": title,
        "snippet": snippet,
        "path": path,
        "url": f"file:{path}#{idx}",
    }


def _skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.startswith(".")


def index_dir(root: Path, *, suffixes: set[str] | None = None) -> list[dict]:
    """Walk `root` and return text chunks (relative paths)."""
    root = root.resolve()
    suffixes = suffixes or TEXT_SUFFIXES
    hits: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(_skip_dir(p) for p in rel_parts[:-1]):
            continue
        if path.suffix.lower() not in suffixes:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(path.relative_to(root))
        if any(rel.startswith(p) for p in SKIP_PATH_PREFIXES):
            continue
        hits.extend(chunk_text(text, path=rel))
    return hits


def rank_chunks(question: str, hits: list[dict], k: int = 4) -> list[dict]:
    return [h for _, h in rank_chunks_scored(question, hits, k=k)]


def rank_chunks_scored(
    question: str, hits: list[dict], k: int = 4
) -> list[tuple[float, dict]]:
    idents = question_identifiers(question)
    mode = corpus_mode(question)
    scored = []
    for h in hits:
        text = f"{h.get('path', '')} {h.get('title', '')} {h.get('snippet', '')}"
        score = float(extract.term_score(question, text))
        path = h.get("path") or ""
        stem = Path(path).stem.casefold()
        for term in extract.question_terms(question):
            if term.casefold() == stem or term.casefold() in stem:
                score += 1.5
        for ident in idents:
            if re.search(rf"(?m)^\s*{re.escape(ident)}\s*=", text):
                score += 12.0
            elif re.search(rf"\bdef\s+{re.escape(ident)}\b", text):
                score += 10.0
            elif ident in text:
                # Mentions in gold-case strings / fixtures are not definitions.
                if re.search(
                    rf"""['"].*{re.escape(ident)}.*['"]""", text
                ) and not re.search(rf"{re.escape(ident)}\s*=", text):
                    score -= 6.0
                elif path.startswith(("evals/", "tests/")) or path.endswith(
                    "route_cases.py"
                ):
                    score -= 8.0
                else:
                    score += 5.0
            elif ident.casefold() in text.casefold():
                score += 2.0
        if mode == "code":
            if path.endswith(".py") and idents:
                score += 1.0
            if path.startswith("docs/") and idents:
                score -= 2.0
            if idents and path.startswith(("evals/", "tests/")):
                score -= 4.0
            if idents and path.endswith("route_cases.py"):
                score -= 10.0
        else:
            if path in PROSE_PATHS or path.startswith("docs/"):
                score += 4.0
            if path == "README.md":
                score += 2.0
        # Topic boosts for common harness questions.
        if re.search(r"(?i)\b(loop|repetition|repeat)\b", question or ""):
            if re.search(r"(?i)\b(is_looping|looping|repetition)\b", text):
                score += 8.0
        if re.search(r"(?i)\b(server|wrapper|mlx)\b", question or ""):
            if (
                "bin/" in path
                or re.search(r"(?i)deku-serv|llama-server", path + " " + text)
                or re.search(r"(?i)mlx_lm\.server", text)
            ):
                score += 8.0
        if re.search(r"(?i)\b(models?|minicpm|llms?)\b", question or ""):
            if path == "README.md":
                score += 6.0
            if re.search(r"(?i)\b(MiniCPM|mlx_lm|model)\b", text):
                score += 3.0
            if "ling-integration" in path:
                score -= 4.0
        scored.append((score, h))
    scored.sort(key=lambda x: (-x[0], hits.index(x[1]) if x[1] in hits else 0))
    return scored[:k]


def pack_hits(
    scored: list[tuple[float, dict]], *, n: int = 2, mode: str = "code"
) -> list[dict]:
    """Top hit plus same-file neighbors for a denser MiniCPM note pack."""
    if not scored:
        return []
    top = scored[0][1]
    same = [h for _, h in scored if h.get("path") == top.get("path")]
    if not same:
        same = [top]
    if mode == "prose":
        substantial = [h for h in same if len((h.get("snippet") or "").split()) >= 6]
        if substantial:
            return substantial[: max(n, 2)]
        # Top was a bare heading — pull more same-path chunks from the full list.
        return same[: max(n, 3)]
    return same[:n]


def hits_to_document(hits: list[dict], *, snippet_chars: int = 700) -> str:
    parts = []
    for h in hits:
        path = (h.get("path") or h.get("title") or "").strip()
        snip = (h.get("snippet") or "").strip()
        if len(snip) > snippet_chars:
            snip = snip[:snippet_chars].rsplit("\n", 1)[0] + "…"
        parts.append(f"{path}\n{snip}\nSource: file:{path}")
    return "\n\n".join(parts)


def prose_file_document(root: Path, path: str, *, max_chars: int = 12_000) -> str | None:
    """Load a product README/AGENTS file for question-aware lead selection."""
    if path not in PROSE_PATHS:
        return None
    p = Path(root) / path
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0] + "\n…"
    return f"{path}\n{text}\nSource: file:{path}"


def run(
    question: str,
    *,
    root: str | Path,
    k: int = 6,
    seed: int = 0,
    hits: list[dict] | None = None,
) -> Result:
    """Search `root` and answer with grounded compose."""
    from deku import hier_summary as hs

    intent = rule_intent(question)
    mode = corpus_mode(question)
    out = Result(
        intent=intent,
        detail={"router": "rule", "root": str(root), "mode": mode},
    )
    if intent != "search":
        out.status = "skipped"
        return out
    if hs.wants_summary(question or ""):
        parts = []
        for name in ("README.md", "AGENTS.md"):
            p = Path(root) / name
            if p.is_file():
                try:
                    parts.append(p.read_text(encoding="utf-8", errors="replace")[:20_000])
                except OSError:
                    pass
        if parts:
            got = hs.summarize(
                "\n\n".join(parts), question=question or "", live=True, seed=seed
            )
            out.answer = got.answer
            out.status = got.status
            out.document = got.document
            out.detail.update(got.detail)
            out.detail["mode"] = "hier_summary"
            return out
    query = rule_query(question)
    out.query = query
    corpus = hits if hits is not None else index_dir(Path(root))
    out.detail["raw_hits"] = len(corpus)
    if mode == "prose":
        prose_only = [
            h for h in corpus
            if (h.get("path") or "") in PROSE_PATHS
            or (h.get("path") or "").startswith("docs/")
            or (h.get("path") or "").endswith(".md")
        ]
        if prose_only:
            # Product README/AGENTS beat research notes for overview/model Qs.
            if re.search(r"(?i)\b(about|overview|purpose|project|readme|models?|minicpm|llms?)\b", question or ""):
                product = [h for h in prose_only if (h.get("path") or "") in PROSE_PATHS]
                corpus = product or prose_only
            else:
                corpus = prose_only
            out.detail["prose_filtered"] = True
    scored = rank_chunks_scored(question, corpus, k=k)
    out.hits = [h for _, h in scored]
    if not scored:
        out.status = "no_hits"
        out.answer = ws.CANNOT_ANSWER
        return out
    top_score, top = scored[0]
    out.detail["top_score"] = top_score
    idents = question_identifiers(question)
    top_text = f"{top.get('path', '')} {top.get('snippet', '')}"
    if idents and not any(
        i in top_text or i.casefold() in top_text.casefold() for i in idents
    ):
        out.answer, out.status = ws.CANNOT_ANSWER, "cannot_answer"
        out.detail["abstain_reason"] = "identifiers_missing_from_top_hit"
        return out
    if top_score < ws.MIN_HIT_SCORE:
        out.answer, out.status = ws.CANNOT_ANSWER, "cannot_answer"
        out.detail["abstain_reason"] = "weak_or_off_topic_hit"
        return out

    packed = pack_hits(scored, n=2, mode=mode)
    # Prose packs same-file neighbors so heading-only tops still get body text.
    doc_top = hits_to_document(packed if mode == "prose" else [top])
    full_prose = None
    if mode == "prose":
        full_prose = prose_file_document(Path(root), top.get("path") or "")
        if full_prose:
            doc_top = full_prose
            out.detail["prose_full_file"] = top.get("path")
    out.document = doc_top
    if mode == "prose":
        lead = prose_lead_sentence(doc_top, question)
        if lead:
            out.detail["prose_lead_candidate"] = lead

    assign_line, assign_val, assign_path = find_assignment(question, doc_top)
    if assign_line:
        ident = assign_line.split("=", 1)[0].strip()
        path = assign_path or (top.get("path") or "").split("#")[0].strip() or None
        out.detail["core"] = assign_val
        out.detail["reply_source"] = "assignment"
        if path:
            out.detail["path"] = path
            out.detail["locations"] = [
                {"path": path, "ident": ident, "value": assign_val},
            ]
        out.answer, out.status = (
            render.assignment(
                ident,
                assign_val,
                path=path,
                where=looks_where_ident(question),
            ),
            "ok",
        )
        return out
    defn = find_definition(question, doc_top)
    if defn:
        out.detail["core"] = defn
        out.detail["reply_source"] = "definition"
        out.answer, out.status = definition_reply(doc_top, defn), "ok"
        return out

    # Prose / overview: prefer a grounded lead sentence over weak MiniCPM
    # summary for about/purpose/project questions.
    doc = full_prose or hits_to_document(packed)
    out.document = doc
    if mode == "prose":
        lead = prose_lead_sentence(doc, question) or out.detail.get("prose_lead_candidate")
        overview = bool(
            re.search(
                r"(?i)\b(about|purpose|overview|readme|this project|"
                r"what (?:is|does) (?:this|deku|the readme))\b",
                question or "",
            )
        )
        if overview and lead:
            out.detail["core"] = lead
            out.detail["reply_source"] = "prose_lead"
            out.detail["summary_skipped"] = "overview_prefers_lead"
            out.answer, out.status = lead, "ok"
            return out
        summary = ws.minicpm_summarize(question, doc, seed=seed)
        out.detail["summary"] = summary
        use_summary = bool(
            summary
            and len(summary.split()) >= ws.MIN_SUMMARY_WORDS
            and ws.reply_grounded(summary, doc)
            and extract.term_score(question, summary) >= 1
        )
        if use_summary and lead:
            # Prefer the README lead when it answers at least as well.
            if float(extract.term_score(question, lead)) >= float(
                extract.term_score(question, summary)
            ):
                use_summary = False
                out.detail["summary_rejected"] = "lead_covers_better"
            elif re.search(r"(?i)\b(models?|minicpm|llms?)\b", question or ""):
                if re.search(r"(?i)MiniCPM", lead) and not re.search(
                    r"(?i)MiniCPM", summary
                ):
                    use_summary = False
                    out.detail["summary_rejected"] = "summary_misses_minicpm"
                elif not re.search(r"(?i)\b(MiniCPM|LLM|model)\b", summary):
                    use_summary = False
                    out.detail["summary_rejected"] = "summary_misses_model"
        if use_summary:
            out.detail["core"] = summary
            out.detail["reply_source"] = "summary"
            out.answer, out.status = summary.strip(), "ok"
            return out
        if summary and not out.detail.get("summary_rejected"):
            out.detail["summary_rejected"] = "ungrounded_or_off_topic"
        if lead:
            out.detail["core"] = lead
            out.detail["reply_source"] = "prose_lead"
            out.answer, out.status = lead, "ok"
            return out
        out.answer, out.status = ws.CANNOT_ANSWER, "cannot_answer"
        out.detail["abstain_reason"] = "no_prose_lead"
        return out

    core, status = ws.minicpm_extract(question, doc, seed=seed)
    out.detail["core"] = core
    out.detail["extract_status"] = status
    if should_abstain(question=question, doc=doc, score=top_score, core=core):
        out.answer, out.status = ws.CANNOT_ANSWER, "cannot_answer"
        out.detail["abstain_reason"] = "no_grounded_core"
        return out
    summary = ws.minicpm_summarize(question, doc, seed=seed)
    out.detail["summary"] = summary
    if (
        summary and core and ws.core_in_reply(core, summary)
        and not ws.reply_grounded(summary, doc)
    ):
        out.detail["summary_rejected"] = "ungrounded_claims"
    reply = ws.compose_reply(core, summary, doc, question=question)
    if not reply:
        out.answer, out.status = ws.CANNOT_ANSWER, "cannot_answer"
        out.detail["abstain_reason"] = "empty_reply"
        return out
    if reply.strip() == (core or "").strip() and not ws.sentence_with_core(core, doc):
        out.answer, out.status = ws.CANNOT_ANSWER, "cannot_answer"
        out.detail["abstain_reason"] = "core_without_sentence"
        return out
    if summary and reply == summary.strip():
        out.detail["reply_source"] = "summary"
    elif reply != core:
        out.detail["reply_source"] = "source_sentence"
    else:
        out.detail["reply_source"] = "core"
    out.answer, out.status = reply, "ok"
    return out
