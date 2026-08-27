"""url_read: fetch a user-specified URL → notes → grounded reply.

No search ranking. The URL is extracted from the question (or passed in).
HTML is stripped to text; MiniCPM extract / compose / abstain reuse web_search.
"""
from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from deku import extract
from deku import hier_summary as hs
from deku import web_search as ws

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
MAX_BYTES = 400_000
MAX_DOC_CHARS = 12_000
MAX_SUMMARY_CHARS = 40_000
USER_AGENT = "deku-url-read/0.1 (+local)"


@dataclass
class Result:
    url: str = ""
    document: str = ""
    answer: str | None = None
    status: str = ""
    detail: dict = field(default_factory=dict)


def extract_url(text: str) -> str | None:
    m = URL_RE.search(text or "")
    if not m:
        return None
    url = m.group(0).rstrip(".,;:!?")
    # Drop a lone trailing ")" if unmatched (markdown links).
    if url.endswith(")") and url.count("(") < url.count(")"):
        url = url[:-1]
    return url


def html_to_text(raw: bytes | str) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    text = raw
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    # Prefer real prose paragraphs when the page has them (Wikipedia, etc.).
    paras = re.findall(r"(?is)<p\b[^>]*>(.*?)</p>", text)
    cleaned_paras = []
    for p in paras:
        plain = re.sub(r"<[^>]+>", " ", p)
        plain = html.unescape(plain)
        plain = re.sub(r"\s+", " ", plain).strip()
        if len(plain.split()) >= 8:
            cleaned_paras.append(plain)
    if len(cleaned_paras) >= 2:
        return "\n\n".join(cleaned_paras)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n", text)
    text = re.sub(r"(?i)</(div|h[1-6]|li|tr)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_url(url: str, *, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain,*/*"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        data = data[:MAX_BYTES]
    return data


def _wiki_title_from_url(url: str) -> str | None:
    m = re.search(
        r"(?i)https?://(?:[a-z]+\.)?wikipedia\.org/wiki/([^?#]+)", url or ""
    )
    if not m:
        return None
    title = urllib.parse.unquote(m.group(1).replace("_", " ")).strip()
    return title or None


def fetch_wikipedia_fallback(url: str) -> bytes | None:
    """When /wiki/Title 404s, try the REST summary extract as HTML-ish text."""
    title = _wiki_title_from_url(url)
    if not title:
        return None
    api = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + urllib.parse.quote(title.replace(" ", "_"))
    )
    try:
        raw = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    api, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
                ),
                timeout=20,
            ).read()
        )
    except Exception:
        return None
    extract_text = (raw.get("extract") or "").strip()
    if len(extract_text.split()) < 8:
        return None
    page_title = (raw.get("title") or title).strip()
    html_doc = (
        f"<html><head><title>{html.escape(page_title)}</title></head>"
        f"<body><p>{html.escape(extract_text)}</p></body></html>"
    ).encode("utf-8")
    return html_doc


def to_document(url: str, text: str, *, limit: int = MAX_DOC_CHARS) -> str:
    body = (text or "").strip()
    if len(body) > limit:
        body = body[:limit].rsplit("\n", 1)[0] + "\n…"
    title = url.rstrip("/").rsplit("/", 1)[-1] or url
    return f"{title}\n{body}\nSource: {url}"


def lexical_answer(question: str, document: str) -> str | None:
    """Best source sentence by term overlap (no model)."""
    best, best_sc = None, 0.0
    who = bool(re.search(r"(?i)\bwho\b", question or ""))
    for block in re.split(r"\n+", document or ""):
        for sent in re.split(r"(?<=[.!?])\s+", block):
            s = sent.strip()
            if len(s.split()) < 5:
                continue
            if s.lower().startswith("source:"):
                continue
            if re.search(r"(?i)\b(disambiguation|may refer to|for other people)\b", s):
                continue
            sc = float(extract.term_score(question, s))
            if who and re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", s):
                sc += 3.0
            if re.search(r"(?i)\b(CEO|chief executive|founded|capital)\b", s):
                sc += 1.5
            if sc > best_sc:
                best, best_sc = s, sc
    if best is None or best_sc < 1.5:
        return None
    return best


_ERA_MARKERS = re.compile(
    r"(?i)\b("
    r"sengoku|heian|edo|meiji|kamakura|nara|muromachi|"
    r"until\s+\d{4}|in the \d+(?:st|nd|rd|th) century|"
    r"\d+(?:st|nd|rd|th) century|medieval|ancient|"
    r"was the (?:imperial )?capital|from \d{4} to \d{4}"
    r")\b"
)
_PRESENT_TENSE = re.compile(r"(?i)\b(is|are|has|have|today)\b")


def prefer_coherent_summary_span(document: str, *, max_sents: int = 2) -> str | None:
    """Pick a short present-tense lead; avoid concatenating distant eras."""
    lines = []
    for line in (document or "").split("\n"):
        s = line.strip()
        if not s or s.lower().startswith("source:"):
            continue
        lines.append(s)
    if not lines:
        return None
    # Drop a bare title line when the next line restates the topic.
    body = " ".join(lines[1:] if len(lines) > 1 and len(lines[0].split()) <= 4 else lines)
    sents = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", body)
        if len(s.split()) >= 4
    ]
    if not sents:
        return None
    modern = [
        s for s in sents
        if _PRESENT_TENSE.search(s) and not _ERA_MARKERS.search(s)
    ]
    if modern:
        return " ".join(modern[:max_sents])
    # Fall back to the first sentence alone (do not stitch eras).
    return sents[0]


def document_mixes_eras(document: str) -> bool:
    """True when the page has both modern and historical era sentences."""
    body = " ".join(
        ln.strip()
        for ln in (document or "").split("\n")
        if ln.strip() and not ln.strip().lower().startswith("source:")
    )
    sents = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", body)
        if len(s.split()) >= 4
    ]
    modern = any(_PRESENT_TENSE.search(s) and not _ERA_MARKERS.search(s) for s in sents)
    historical = any(_ERA_MARKERS.search(s) for s in sents)
    return modern and historical


def finalize_reply(
    *,
    question: str,
    doc: str,
    core: str | None,
    summary: str | None,
) -> str | None:
    """Keep grounded compose; else lexical lead; else None."""
    lead = lexical_answer(question, doc)
    if (
        core
        and summary
        and ws.core_in_reply(core, summary)
        and len(summary.split()) >= ws.MIN_SUMMARY_WORDS
        and ws.reply_grounded(summary, doc)
    ):
        # Prefer lead when it covers the question at least as well.
        if lead and float(extract.term_score(question, lead)) >= float(
            extract.term_score(question, summary)
        ):
            return lead
        return summary.strip()
    if core and ws.reply_grounded(core, doc) and ws.core_fits_question(question, core):
        sent = ws.sentence_with_core(core, doc)
        if sent and (not lead or extract.term_score(question, sent) >= extract.term_score(question, lead)):
            return sent
        if lead:
            return lead
        return core.strip()
    # Prefer a classless grounded reply when MiniCPM core is usable.
    composed = ws.compose_reply(core or "", "", doc, question=question or "")
    if composed and ws.reply_grounded(composed, doc):
        return composed
    if not ws.classless_web_enabled():
        templ = ws.template_reply(question, core or "", doc)
        if templ:
            return templ
    return lead


def run(
    question: str,
    *,
    url: str | None = None,
    seed: int = 0,
    fetch: Callable[..., bytes] | None = None,
    live_answer: bool = True,
) -> Result:
    """Fetch `url` (or extract from question) and answer from the page text."""
    out = Result(detail={})
    target = url or extract_url(question)
    if not target:
        out.status = "no_url"
        out.answer = ws.CANNOT_ANSWER
        out.detail["abstain_reason"] = "no_url"
        return out
    out.url = target
    fetcher = fetch or fetch_url
    try:
        raw = fetcher(target)
    except urllib.error.HTTPError as e:
        if e.code == 404 and fetch is None:
            fb = fetch_wikipedia_fallback(target)
            if fb:
                raw = fb
                out.detail["fetch_fallback"] = "wikipedia_rest_summary"
            else:
                out.status = "not_found"
                out.answer = "That page was not found."
                out.detail["abstain_reason"] = "not_found"
                out.detail["error"] = str(e)
                return out
        else:
            out.status = "fetch_error"
            out.answer = ws.CANNOT_ANSWER
            out.detail["abstain_reason"] = "fetch_error"
            out.detail["error"] = str(e)
            return out
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        out.status = "fetch_error"
        out.answer = ws.CANNOT_ANSWER
        out.detail["abstain_reason"] = "fetch_error"
        out.detail["error"] = str(e)
        return out
    text = html_to_text(raw)
    if len(text.split()) < 8:
        out.status = "empty_page"
        out.answer = ws.CANNOT_ANSWER
        out.detail["abstain_reason"] = "empty_page"
        return out

    if hs.wants_summary(question or ""):
        # Prefer a temporally coherent lead when the page mixes eras.
        doc_preview = to_document(target, text)
        if document_mixes_eras(doc_preview):
            coherent = prefer_coherent_summary_span(doc_preview)
            if coherent:
                out.document = doc_preview
                out.detail["core"] = coherent
                out.detail["reply_source"] = "coherent_span"
                out.detail["mode"] = "summarize_coherent"
                out.answer, out.status = coherent, "ok"
                return out
        lead = lexical_answer(question, doc_preview)
        if lead and float(extract.term_score(question, lead)) >= 2.0:
            out.document = doc_preview
            out.detail["core"] = lead
            out.detail["reply_source"] = "lexical"
            out.detail["mode"] = "summarize_lead"
            out.answer, out.status = lead, "ok"
            return out
        # Hierarchical map-reduce over a longer window (extractive-anchored).
        body = text if len(text) <= MAX_SUMMARY_CHARS else text[:MAX_SUMMARY_CHARS]
        out.document = to_document(target, body, limit=MAX_SUMMARY_CHARS)
        got = hs.summarize(
            body, question=question or "", live=live_answer, seed=seed
        )
        out.answer = got.answer
        out.status = got.status
        out.detail.update(got.detail)
        out.detail["mode"] = "hier_summary"
        return out

    doc = to_document(target, text)
    out.document = doc
    score = float(extract.term_score(question, doc))
    out.detail["doc_score"] = score

    if not live_answer:
        lead = lexical_answer(question, doc)
        if not lead:
            out.answer, out.status = ws.CANNOT_ANSWER, "cannot_answer"
            out.detail["abstain_reason"] = "no_lexical_sentence"
            return out
        out.detail["reply_source"] = "lexical"
        out.answer, out.status = lead, "ok"
        return out

    core, status = ws.minicpm_extract(question, doc, seed=seed)
    out.detail["core"] = core
    out.detail["extract_status"] = status
    summary = ws.minicpm_summarize(question, doc, seed=seed)
    out.detail["summary"] = summary
    reply = finalize_reply(question=question, doc=doc, core=core, summary=summary)
    if not reply:
        out.answer, out.status = ws.CANNOT_ANSWER, "cannot_answer"
        out.detail["abstain_reason"] = "no_grounded_core"
        return out
    if summary and reply == summary.strip():
        out.detail["reply_source"] = "summary"
    elif core and reply == (core or "").strip():
        out.detail["reply_source"] = "core"
    elif reply == lexical_answer(question, doc):
        out.detail["reply_source"] = "lexical"
        if summary:
            out.detail["summary_rejected"] = "ungrounded_or_weaker"
    else:
        out.detail["reply_source"] = "source_sentence"
    out.answer, out.status = reply, "ok"
    return out
