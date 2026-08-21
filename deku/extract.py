#!/usr/bin/env python3
"""Grounded extraction with server-side verification (the /v1/extract contract).

Division of labour this exists to serve: locating text is solved and cheap
(ripgrep; Cursor's Instant Grep does it in 13ms on chromium-scale repos), and
deciding is the big model's job. The gap is READING: grep answers "where does
X appear", not "what is the timeout value here" — the answer is not known in
advance, so it cannot be a pattern. A 1B at 351-562 tok/s aggregate can read
16 chunks a second, which is the actual speed advantage (breadth, not
single-stream latency).

Trust comes from structure, not from the model. The model emits ONE line (the
answer); the server checks `answer ⊆ source` normalized, and the supporting
quote is then LOCATED deterministically — the model never writes it.

Asking for the quote was the first design and it failed on measurement: the
1B echoed the format description verbatim as the value
("QUOTE: one phrase copied word-for-word from the document..."), the known
paste-the-template pathology, so recall was 0/4 while the answers themselves
were correct. Deriving the quote in code removes the failure mode entirely
and makes the span verbatim by construction — the same "deterministic harness
beats model output" lesson as the rest of this repo.

The contract is extractive: an answer that is not a span of the document is
DROPPED, never returned with a warning. That turns model unreliability into
misses instead of confident lies — the principle Cursor states for their
index ("false positives are always acceptable, because the final matching is
performed deterministically on the text itself").

Honest limit: `answer ⊆ source` proves the answer came FROM the document, not
that it ANSWERS the question — a real span can be irrelevant. That residual
is measured, not assumed away: evals/extract_eval.py scores the false-positive
rate on distractor chunks (real text, same topic, no answer).

REJECTED, because it exploits exactly that residual: re-running `unverified`
chunks at finer granularity to recover them. More granularity means more
candidate substrings, so more IRRELEVANT fragments survive the substring
check — recall rises and precision falls. Asked for a tokens-per-second rate
it recovers `max_tokens: 600` and `temperature: 0.7`, both grounded, both
wrong; on-target recovery was 0. Evidence in evals/escalate_ab.py. The knob
that works is question SPECIFICITY, and it is free: same corpus and same
filter, "What is the value of MAX_CHUNKS?" is stable at 4/4 while "What is
the chunk cap?" returns the wrong constant or abstains. This tool is a
reader, not a searcher — granularity is a lever for a specific question,
never a substitute for one.
"""

import re
import unicodedata

PREFILL = "ANSWER: "
STOPS = ["\n", "<|im_end|>", "<|im_start|>"]
MAX_TOKENS = 40
TEMP = 0.2
MIN_ANSWER = 1

PROMPT = """Answer the question using only the document below.

Question: {question}

Document:
{doc}

Copy the answer word-for-word from the document. Write NONE if the document \
does not answer the question. Answer only, one line."""


def chatml(user: str, prefill: str) -> str:
    """Raw-completion prompt with a forced reply prefix — the only reliable
    on-format anchor for this 1B (capability profile)."""
    return (f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{prefill}")


def build_prompt(question: str, doc: str) -> str:
    return chatml(PROMPT.format(question=question, doc=doc), PREFILL)


# NFKC leaves typographic punctuation alone, so a document written with an
# en dash loses to a model that types a hyphen: measured, `556–562 tok/s` in
# the source vs `556-562 tok/s` from the model, and the CORRECT answer was
# dropped as ungrounded. Folding these is pure false-negative repair.
PUNCT = str.maketrans({
    "–": "-", "—": "-", "‐": "-", "‑": "-", "‒": "-", "―": "-", "−": "-",
    "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "‚": "'",
    " ": " ", "　": " ", "​": "",
})


def norm(s: str) -> str:
    """NFKC + punctuation folding + casefold + whitespace collapse. The model
    reflows whitespace, normalizes fullwidth digits (１２０ -> 120) and retypes
    typographic dashes as hyphens; none of that should count as a verification
    failure."""
    return " ".join(
        unicodedata.normalize("NFKC", s).translate(PUNCT).casefold().split())


def parse(raw: str) -> str:
    """The answer from a model reply. The prefill already emitted 'ANSWER: ',
    so the completion's first non-empty line IS the answer; a reply that
    re-states the prefix is tolerated."""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("ANSWER:"):
            line = line[len("ANSWER:"):].strip()
        return line
    return ""


WINDOW = 3  # consecutive lines joined; prose wraps, but a whole-chunk quote
            # would stop being evidence


def spans(source: str) -> list[str]:
    """Candidate quote spans: lines, sentence-ish pieces of each, and joins of
    up to WINDOW consecutive lines. Split rather than index-mapped because NFKC
    changes lengths (１２０ -> 120) and a wrong offset would quote the wrong
    text.

    The multi-line windows exist because of a real miss: a wrapped markdown
    bullet put "TTFT 36.8x less" and "(4769ms -> 130ms)" on different lines, so
    a correct extraction was unlocatable and got dropped as a hallucination.
    """
    lines = [l.strip() for l in source.splitlines() if l.strip()]
    out = []
    for line in lines:
        out.append(line)
        piece = ""
        for ch in line:
            piece += ch
            if ch in "。．.!?！？":
                if piece.strip():
                    out.append(piece.strip())
                piece = ""
        if piece.strip() and piece.strip() != line:
            out.append(piece.strip())
    for w in range(2, WINDOW + 1):
        for i in range(len(lines) - w + 1):
            out.append(" ".join(lines[i:i + w]))
    return out


def locate_quote(answer: str, source: str) -> str | None:
    """Shortest source span containing the answer, or None if the answer is not
    in the document. Deterministic — the model never writes the quote, so it
    cannot fabricate or paraphrase it.

    Containment goes through has_term, so a numeric answer needs the SAME digit
    boundary that governs term matching. Plain substring located '8' inside
    'http://127.0.0.1:8765' — spans() splits that line on its dots, so the
    7-char fragment '1:8765"' outranked the real 'HIT_CONTEXT = 8' on length,
    and the quote then carried none of the question's terms and was dropped as
    irrelevant. The correct answer was thrown away by its own citation.

    verify() is defined as `locate_quote(...) is not None`, so this closes a
    hole in the core contract as well: '8' used to verify against a document
    whose only 8 lives inside 8765, which is a different value.
    """
    a = norm(answer)
    if not a:
        return None
    hits = [s for s in spans(source) if has_term(a, norm(s))]
    return min(hits, key=len) if hits else None


CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿ｦ-ﾝ]")
STOPWORDS = {"what", "which", "who", "whom", "whose", "where", "when", "why",
             "how", "many", "much", "is", "are", "was", "were", "do", "does",
             "did", "the", "a", "an", "of", "in", "on", "at", "to", "for",
             "from", "by", "with", "and", "or", "this", "that", "these",
             "those", "it", "its", "there", "here", "value", "number", "name",
             # generic words that match almost any code/config line and so
             # carry no relevance signal: 'default' alone let
             # `default=0.7` answer a question about the default PORT
             "default", "set", "use", "used", "get", "run", "make", "type",
             "file", "line", "code", "data", "text", "size", "list"}


SENT_END = re.compile(r"[?!;,]+")


def question_terms(question: str) -> set[str]:
    """Content words of the question. Numbers are kept at ANY length: the
    len>2 rule silently dropped '16' from 'throughput with 16 concurrent
    requests', which is precisely the token that says WHICH throughput —
    the 8-concurrent line then outscored the 16-concurrent one 3 terms to 2.

    Sentence punctuation is separated, and a TRAILING period is stripped per
    token — trailing only, so a decimal keeps its dot and 10:15 keeps its
    colon. Previously only '?' was removed, which is invisible for questions
    and wrong for statements: 'the temperature is 0.2.' yielded the term
    '0.2.', the digit-bounded matcher cannot find that in 'TEMP = 0.2', and
    the answer-bearing chunk scored 0 and ranked 6th of 31. Measured on
    claims, where every input ends in a period by construction.
    """
    words = (w.rstrip(".") for w in SENT_END.sub(" ", norm(question)).split())
    return {w for w in words
            if w not in STOPWORDS and (len(w) > 2 or any(c.isdigit() for c in w))}


def has_term(term: str, text: str) -> bool:
    """Substring for words (so 'read' finds 'read_timeout'), but digit-bounded
    for numbers (so '5' does not match '562' and '16' does not match '162')."""
    if any(c.isdigit() for c in term):
        return re.search(rf"(?<!\d){re.escape(term)}(?!\d)", text) is not None
    return term in text


def abstains(quote: str) -> bool:
    """True when lexical scoring cannot say anything about this span: it is
    CJK and carries no Latin token a Latin question could ever match. Both
    relevant() and keep_best() defer to it, so a Japanese span is never
    filtered — nor outranked — for lacking English words it cannot contain."""
    return bool(CJK.search(quote)) and not re.search(r"[a-z]{3}", norm(quote))


def relevant(question: str, quote: str) -> bool:
    """Does the quote lexically relate to the question? Cheap grep-shaped
    relevance check for the residual the grounding check cannot cover: on
    near-miss chunks (a config with other numeric fields but not the asked-for
    one) the 1B answers with SOME real span rather than NONE — measured 18/22
    false positives without this, 0/22 with it.

    ABSTAINS (returns True) when the question shares no vocabulary with the
    chunk at all, because that is also what cross-lingual retrieval looks
    like: an English question against a Japanese document ('port' vs
    'ポート') has zero overlap by construction, and filtering there would
    drop correct answers. So this raises precision when question and document
    share a language and is a no-op when they do not — callers relying on it
    for cross-lingual precision must prefilter upstream instead.
    """
    terms = question_terms(question)
    if not terms:
        return True
    q = norm(quote)
    # Abstain ONLY when the span is CJK *and* carries no Latin token the
    # question could ever match — that is the cross-script case, where zero
    # overlap is structural rather than evidence of irrelevance. Both halves
    # are load-bearing: keying on "no Latin" alone let degenerate symbolic
    # quotes ('0)', '05,') abstain into the results (measured on a real repo
    # scan), and keying on "has CJK" alone stopped filtering mixed lines like
    # 'host は 10.0.0.5 です', which a port question should still reject.
    if abstains(quote):
        return True
    return any(has_term(t, q) for t in terms)


def term_score(question: str, quote: str) -> int:
    """How many distinct question terms the quote carries."""
    q = norm(quote)
    return sum(1 for t in question_terms(question) if has_term(t, q))


def diagnose(question: str, sources: list[str]) -> dict:
    """Why did this question find nothing — wrong words, or absent content?

    NO_ANSWER is derived from term overlap with the chunk (this 1B never emits
    NONE, so the status has to be inferred), which makes "my wording missed the
    corpus" and "the answer is not here" look identical to the caller. Naming
    the terms separates them: {chunk, cap} scoring on 2 of 3 chunks says the
    wording is picking the wrong constant, {max_chunks} scoring on 1 says the
    corpus is right and the model failed on it.

    Cross-script chunks are counted APART, never as misses. term_score is 0 on
    a Japanese span by construction and relevant() deliberately abstains there,
    so folding them into the denominator would tell the caller to rephrase a
    question that is working — the ja1 -> 9100 case in evals/extract_wide.py.
    """
    return {"terms": sorted(question_terms(question)),
            "on_topic": sum(1 for s in sources if term_score(question, s) > 0),
            "chunks": len(sources),
            "cross_script": sum(1 for s in sources if abstains(s))}


def keep_best(results: list[dict], question: str) -> list[dict]:
    """Of the results competing for one question, keep only the best-matching.

    Needed once chunks are small: at one line per chunk the model cannot
    confuse read_timeout with write_timeout — it only sees one — but every
    line happily answers, and relevant() passes them all on the shared word
    'timeout'. Measured: 'What is the read timeout?' came back with 30, 60, 5
    AND 120. Term overlap keeps 'read_timeout = 30' (2 terms), drops the rest.

    A number in the question is a SELECTOR, not a description — 'throughput
    with 16 concurrent requests' wants the 16 line, and plain term counting
    ties it against the 8 line (both carry 3 terms, each missing one of
    'throughput'/'16'). So candidates carrying every numeric term win outright
    when any candidate does; term count only breaks the remaining ties.
    A tie keeps everything: equal evidence is no reason to pick, and
    cross-script quotes all score 0 by construction.
    """
    if not results:
        return results

    # Rank on the CHUNK, not the quote. The quote is wherever the answer sits,
    # which for record-shaped text is not where the evidence sits: asked which
    # commit mentions tool-call, the answer is a hash and its line reads
    # 'commit 9423764' — every record scores the same and all of them survive.
    # The chunk holds the subject line that actually discriminates.
    def text_of(r: dict) -> str:
        return r.get("_src") or r["quote"]

    # cross-script spans score 0 by construction, so ranking them against
    # Latin spans silently deletes them — measured: a correct Japanese hit
    # (ポートは9100) vanished the moment an English hit shared the corpus.
    # They are kept unconditionally instead.
    free = [r for r in results if abstains(text_of(r))]
    ranked = [r for r in results if not abstains(text_of(r))]
    if not ranked:
        return results
    nums = {t for t in question_terms(question) if any(c.isdigit() for c in t)}
    if nums:
        exact = [r for r in ranked
                 if all(has_term(t, norm(text_of(r))) for t in nums)]
        if exact:
            ranked = exact
    scored = [(term_score(question, text_of(r)), r) for r in ranked]
    best = max(s for s, _ in scored)
    return free + [r for s, r in scored if s == best]


def verify(answer: str, source: str) -> bool:
    """answer ⊆ source, normalized. NONE never passes: a chunk with no answer
    must yield no result, not a verified 'NONE'."""
    if not answer or answer.strip().upper().rstrip(".") == "NONE":
        return False
    if len(norm(answer)) < MIN_ANSWER:
        return False
    return locate_quote(answer, source) is not None


def locate_line(quote: str, source: str) -> int | None:
    """1-based line where `quote` starts, or None. Lets a caller jump straight
    to file:line instead of re-reading the chunk to find the span."""
    q = norm(quote)
    if not q:
        return None
    lines = source.splitlines()
    # widen the window only after every single line has been tried, so a span
    # that fits on one line reports THAT line rather than the first line of
    # some wider window that happens to contain it
    for w in range(1, WINDOW + 1):
        for i in range(len(lines) - w + 1):
            window = " ".join(l.strip() for l in lines[i:i + w] if l.strip())
            if window and q in norm(window):
                return i + 1
    return None


OK, NO_ANSWER, UNVERIFIED, FILTERED = "ok", "no_answer", "unverified", "filtered"


def classify(chunk_id, raw: str, source: str,
             question: str | None = None) -> tuple[dict | None, str]:
    """The whole server-side contract for one chunk: (result|None, status).

    The status matters to the caller as much as the result: an empty response
    is ambiguous otherwise. NO_ANSWER means the model said the chunk does not
    answer the question (so reading it yourself is probably wasted), while
    UNVERIFIED means it produced something ungrounded (so the chunk may well
    hold the answer and be worth reading). FILTERED means grounded but
    lexically unrelated to the question.
    """
    answer = parse(raw)
    if not answer or answer.strip().upper().rstrip(".") == "NONE":
        return None, NO_ANSWER
    if not verify(answer, source):
        # Ungrounded. Split the two cases the caller actually acts on: if the
        # chunk does not mention the question's terms at all, it is simply not
        # about this (reading it yourself is wasted) — otherwise the chunk is
        # on topic and the model failed on it, so it may still hold the answer.
        # Needed because this 1B never emits NONE: it invents something, so
        # without this NO_ANSWER would always be 0 and the split would be dead.
        if question is not None and not relevant(question, source):
            return None, NO_ANSWER
        return None, UNVERIFIED
    quote = locate_quote(answer, source)
    if question is not None and not relevant(question, quote):
        return None, FILTERED
    return ({"chunk_id": chunk_id, "answer": answer, "quote": quote,
             "line": locate_line(quote, source)}, OK)


def request_body(question: str, doc: str, seed: int | None = None) -> dict:
    """Worker request for one chunk (raw completion + prefill + stops)."""
    body = {
        "prompt": build_prompt(question, doc),
        "temperature": TEMP,
        "top_p": 0.95,
        "max_tokens": MAX_TOKENS,
        "stop": STOPS,
        "repetition_penalty": 1.05,
    }
    if seed is not None:
        body["seed"] = seed
    return body
