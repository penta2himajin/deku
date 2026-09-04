"""web_search task: classify → query → search → rank → MiniCPM extract.

Division of labour (measured):
  intent/query   LFM2.5-350M (or rule fallback) — short labels / one line
  search/rank    deterministic — duckduckgo/wikipedia + extract.term_score
  answer         MiniCPM grounded extract — answer ⊆ source

MiniCPM does NOT summarize hits (summarize probe 0/15) and does NOT pick
among candidates (quote_pick worse than lexical). Ranking stays in code.
"""
from __future__ import annotations

import html
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from deku import extract
from deku import lexical_core as lex

INTENTS = ("search", "extract", "refuse")
INTENT_RE = re.compile(r"(?i)\b(search|extract|refuse)\b")
QUERY_RE = re.compile(r"(?i)^(?:QUERY:\s*)?(.+)$")

# Very small closed set — enough to keep math/chitchat off the search path
# without a model. The LFM path can override.
SEARCH_CUES = re.compile(
    r"(?i)\b(who|what|when|where|which|company|founded|ceo|parent|"
    r"capital of|meaning of|define|how old|birthday|birth date)\b"
)
NONSEARCH = re.compile(
    r"(?i)^\s*(what is\s+)?(\d|compute|calculate|2\s*\+|sort the|write a |"
    r"implement |fn |def )"
)

# Closed country→city map removed; ranking uses lexical capital cues instead.


@dataclass
class Hit:
    title: str
    snippet: str
    url: str

    def as_dict(self) -> dict:
        return {"title": self.title, "snippet": self.snippet, "url": self.url}


@dataclass
class Result:
    intent: str
    query: str = ""
    hits: list[dict] = field(default_factory=list)
    document: str = ""
    answer: str | None = None
    status: str = ""
    detail: dict = field(default_factory=dict)


def parse_intent(raw: str) -> str:
    m = INTENT_RE.search(raw or "")
    return m.group(1).lower() if m else "refuse"


def parse_query(raw: str) -> str:
    line = (raw.splitlines()[0] if (raw or "").strip() else "").strip()
    m = QUERY_RE.match(line)
    return (m.group(1) if m else line).strip().strip('"')


def rule_intent(question: str) -> str:
    if NONSEARCH.search(question or ""):
        return "refuse"
    if SEARCH_CUES.search(question or ""):
        return "search"
    return "refuse"


MIN_HIT_SCORE = 4
MIN_SUMMARY_WORDS = 4
CANNOT_ANSWER = "I cannot answer from the available sources."


def expand_search_queries(question: str, query: str) -> list[str]:
    """Wikipedia queries: wiki-friendly normalization, rule query, and raw question.

    No shape-specific or entity-specific query injection — retrieval breadth
    comes from the question text and search()/enrich dig, not closed templates.
    """
    out: list[str] = []
    alt = wiki_friendly_query(query)
    if alt:
        out.append(alt)
    if query:
        out.append(query)
    q = (question or "").strip()
    if q:
        out.append(q)
    seen: set[str] = set()
    uniq: list[str] = []
    for item in out:
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            uniq.append(item)
    return uniq


def _question_content_terms(question: str) -> list[str]:
    stop = frozenset(
        """
        what who when where which how why is are was were the a an of
        wrote written authored founded makes manufacturer population
        headquarters released published born birthday
        """.split()
    )
    return [
        w
        for w in re.findall(r"[a-z]+", (question or "").casefold())
        if w not in stop and len(w) > 2
    ]


def looks_title_near_miss(title: str, topic: str) -> bool:
    """Title looks like a near-miss for topic (e.g. Perugia vs Peru)."""
    t = (title or "").strip().casefold()
    top = (topic or "").strip().casefold()
    if not t or not top or t == top:
        return False
    if hit_title_matches_topic(title, topic):
        return False
    return t.startswith(top) and len(t) > len(top)


def generic_hit_score(question: str, hit: dict) -> float:
    """Lexical rerank on a small hit set: overlap, topic match, generic noise."""
    text = f"{hit.get('title', '')} {hit.get('snippet', '')}"
    title = (hit.get("title") or "").strip()
    score = float(extract.term_score(question, text))
    topic = question_topic(question or "")

    if topic:
        if hit_title_matches_topic(title, topic):
            score += 6.0
        elif topic.casefold() in text.casefold():
            score += 3.0
        elif looks_title_near_miss(title, topic):
            score -= 8.0
        if re.fullmatch(re.escape(topic), title.strip(), flags=re.I):
            extra = [
                t
                for t in _question_content_terms(question or "")
                if t != topic.casefold()
            ]
            if extra:
                if not all(extract.has_term(t, text) for t in extra):
                    score -= 5.0
        if re.search(rf"(?i)\bof\s+{re.escape(topic)}\b", title):
            score += 4.0

    named_exact = re.match(
        r"(?i)^\s*(?:how old (?:is|are)|what is) "
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
        r"(?:'s birthday)?\??\s*$",
        question or "",
    )
    if named_exact:
        who = named_exact.group(1).strip()
        if re.fullmatch(re.escape(who), title, flags=re.I):
            score += 12.0
        elif who.casefold() in title.casefold():
            score -= 4.0

    for pat in (
        r"(?i)where (?:was|is) (.+?) born",
        r"(?i)when (?:was|were) (.+?) born",
        r"(?i)(?:birthday|birth date|date of birth)\s+of\s+(?:the\s+)?(.+?)\??\s*$",
        r"(?i)what is ([A-Z][^?'\"]+?)(?:'s)? birthday",
    ):
        m = re.search(pat, question or "")
        if m:
            who = m.group(1).strip().rstrip("?.").rstrip("'s").strip()
            # Office phrases ("current emperor of …") are not person titles.
            if re.search(
                r"(?i)\b(current|emperor|president|prime minister|pope)\b", who
            ):
                break
            if re.fullmatch(re.escape(who), title, flags=re.I):
                score += 10.0
            elif who.casefold() not in title.casefold():
                score -= 8.0
            break

    if re.search(r"(?i)\bwho is\b", question or "") and re.search(
        r"(?i)\b(ceo|chief executive|president|prime minister|pope|emperor)\b",
        question or "",
    ):
        if looks_role_object_title(title, question=question or ""):
            score -= 15.0
        if looks_historical_office(text):
            score -= 10.0
        if looks_current_office(text):
            score += 6.0

    if re.search(r"(?i)\(disambiguation\)", title):
        score -= 20.0
    if re.search(r"(?i)\((film|movie|song|album|TV series|restaurant)\)", title):
        if not re.search(
            r"(?i)\b(film|movie|song|album|restaurant)\b", question or ""
        ):
            score -= 4.0
    # Birthday of an officeholder: down-rank observance/holiday pages.
    if re.search(r"(?i)\b(birthday|birth date|date of birth)\b", question or ""):
        if re.search(
            r"(?i)\b(emperor|president|prime minister|pope)\b",
            question or "",
        ):
            if re.search(r"(?i)\bpublic holiday\b", text):
                score -= 10.0
            elif re.match(r"(?i)^the .+'s birthday$", title.strip()):
                score -= 10.0

    snip = (hit.get("snippet") or "").strip()
    if snip.startswith("Because of this") or (snip and snip[0].islower()):
        score -= 6.0

    acr_m = re.search(r"(?i)^\s*what is\s+(\S+?)\??\s*$", question or "")
    if acr_m:
        ent = acr_m.group(1).strip()
        if title.strip().casefold() == ent.casefold():
            score += 8.0
        if re.search(
            rf"(?i)\b{re.escape(ent)}\b.{{0,80}}\b(is an?|is the|stands for)\b",
            text,
        ):
            score += 4.0

    for m in re.finditer(
        r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)+)\b", question or ""
    ):
        phrase = m.group(1).strip()
        if re.fullmatch(re.escape(phrase), title.strip(), flags=re.I):
            score += 8.0

    if topic and hit_title_matches_topic(title, topic):
        if not re.fullmatch(re.escape(topic), title.strip(), flags=re.I):
            if not title.strip().casefold().startswith(topic.casefold()):
                score -= 4.0

    if re.search(r"(?i)\bwho is\b", question or "") and re.search(
        r"(?i)\b(ceo|chief executive|president|prime minister)\b", question or ""
    ):
        if topic and re.fullmatch(re.escape(topic), title.strip(), flags=re.I):
            if not re.search(
                r"(?i)\b(ceo|chief executive|president|prime minister)\b", text
            ):
                score -= 8.0
        elif topic and topic.casefold() in text.casefold() and re.search(
            r"(?i)\b(ceo|chief executive|president|prime minister)\b", text
        ):
            if has_person_name(text) or re.search(
                r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", text
            ):
                score += 6.0

    if re.search(r"(?i)\b(what company makes|manufacturer)\b", question or "") and topic:
        if re.fullmatch(re.escape(topic), title.strip(), flags=re.I):
            if not re.search(
                r"(?i)\b(developed|manufactured|made|created)\s+by\b", text
            ):
                score -= 4.0
        elif topic.casefold() in text.casefold() and re.search(
            r"(?i)\b(developed|manufactured|made|created)\s+by\b", text
        ):
            score += 6.0

    # When-founded / when-released: prefer the entity page (exact title) over
    # compound near-misses ("X Foundation…", "X Foundry") and boost year cues.
    want_when_founded = bool(
        re.search(r"(?i)\bfounded\b", question or "")
        and re.search(r"(?i)\bwhen\b", question or "")
    )
    want_when_released = bool(
        re.search(r"(?i)\breleased\b", question or "")
        and re.search(r"(?i)\bwhen\b", question or "")
    )
    if topic and (want_when_founded or want_when_released):
        bare = re.sub(r"\s*\([^)]*\)\s*", " ", title).strip()
        if re.fullmatch(re.escape(topic), bare, flags=re.I) or re.fullmatch(
            rf"{re.escape(topic)}(?:\s*,?\s*Inc\.?)?", bare, flags=re.I
        ):
            score += 10.0
        elif hit_title_matches_topic(title, topic) and not re.fullmatch(
            re.escape(topic), bare, flags=re.I
        ):
            # History of X is useful for release; other compounds are noise.
            if want_when_released and re.match(
                rf"(?i)^history of (?:the )?{re.escape(topic)}\b", title
            ):
                score += 8.0
            else:
                score -= 8.0
        if want_when_founded and re.search(
            r"(?i)\b(?:founded|established)\s+(?:in|on)\b.{0,20}\b(?:19|20)\d{2}\b",
            text,
        ):
            score += 5.0
        if want_when_released and re.search(
            r"(?i)\b(?:released|unveiled|launched)\b.{0,40}\b(?:19|20)\d{2}\b",
            text,
        ):
            score += 5.0

    # Onomastic / dictionary pages are poor person biographies.
    if re.search(r"(?i)\((given name|surname|name)\)", title):
        score -= 12.0

    # Birthday of an officeholder: boost bios with birth dates / person titles.
    if re.search(r"(?i)\b(birthday|birth date|date of birth)\b", question or ""):
        if re.search(
            r"(?i)\b(current|emperor|president|prime minister|pope)\b",
            question or "",
        ):
            if re.search(r"(?i)\(born\s+\d|\bborn\s+(?:on\s+)?\d", text):
                score += 8.0
            if has_person_name(title) and not re.search(
                r"(?i)\((given name|surname|name)\)", title
            ):
                score += 4.0
            # Prefer the asked office over a differently titled relative.
            role_m = re.search(
                r"(?i)\b(emperor|empress|president|prime minister|pope)\b",
                question or "",
            )
            if role_m:
                role = role_m.group(1).casefold()
                if re.search(
                    rf"(?i)\bis\s+(?:the\s+)?{re.escape(role)}\b"
                    rf"|\b{re.escape(role)}\s+of\b",
                    text,
                ):
                    score += 12.0
                # Spouse / emerita pages: title is a different office word.
                if role == "emperor" and re.search(r"(?i)\bempress\b", title):
                    score -= 18.0
                elif role == "empress" and re.search(
                    r"(?i)\bemperor\b", title
                ) and not re.search(r"(?i)\bempress\b", title):
                    score -= 18.0

    # who + office: prefer person bios that state the role over org/product pages.
    if re.search(r"(?i)\bwho is\b", question or "") and re.search(
        r"(?i)\b(ceo|chief executive|president|prime minister|pope|emperor)\b",
        question or "",
    ):
        role_in_text = bool(
            re.search(
                r"(?i)\b(ceo|chief executive(?: officer)?|president|"
                r"prime minister|pope|emperor)\b",
                text,
            )
        )
        if has_person_name(title) and role_in_text:
            # Person bio (John Ternus), not org/product titled with the topic.
            if topic and hit_title_matches_topic(title, topic):
                score -= 4.0
            else:
                score += 12.0
        elif (
            topic
            and hit_title_matches_topic(title, topic)
            and not has_person_name(title)
        ):
            score -= 12.0

    # who founded: prefer the org page (or founder bio) over product compounds.
    if re.search(r"(?i)\bwho founded\b", question or "") and topic:
        bare = re.sub(r"\s*\([^)]*\)\s*", " ", title).strip()
        exact_org = bool(
            re.fullmatch(re.escape(topic), bare, flags=re.I)
            or re.fullmatch(
                rf"{re.escape(topic)}(?:\s*,?\s*Inc\.?)?", bare, flags=re.I
            )
        )
        if exact_org:
            score += 12.0 if re.search(
                r"(?i)\bfounded\s+by\b|\bco-?founders?\b", text
            ) else 6.0
        elif hit_title_matches_topic(title, topic) and not exact_org:
            score -= 8.0
        if (
            has_person_name(title)
            and topic.casefold() in text.casefold()
            and re.search(r"(?i)\b(co-?founder|founded)\b", text)
        ):
            score += 8.0

    return score


def looks_like_fragment(snippet: str) -> bool:
    """True for mid-sentence / anaphoric wiki search snippets."""
    s = (snippet or "").strip()
    if not s:
        return True
    if re.match(r"(?i)^(because of this|due to this|as a result|therefore)\b", s):
        return True
    if s[0].islower():
        return True
    return False


def has_person_name(text: str) -> bool:
    """Heuristic: Cap Cap bigram that is not an obvious title/corp phrase.

    Uses structural role words only — no closed country/demonym table.
    """
    t = text or ""
    if re.search(
        r"(?i)\b(inc\.?|corp\.?|corps|corporation|ltd\.?|limited|llc|gmbh|"
        r"entertainment|motors?|company|group|holdings|interactive|"
        r"industries|technologies|systems)\b",
        t,
    ):
        return False
    roleish = re.compile(
        r"(?i)^(republic|states|kingdom|empire|minister|president|congress|"
        r"parliament|company|element|point|union|war|east|west|north|south|"
        r"list|tenure|history|presidency|chief|executive|officer|"
        r"the|a|an|of|and|or|for|in|on|at|to)$"
    )
    for m in re.finditer(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", t):
        a, b = m.group(1), m.group(2)
        if roleish.match(a) or roleish.match(b):
            continue
        return True
    return False


_ROLE_OBJECT_TITLE = re.compile(
    r"(?i)\b("
    r"official\s+car|official\s+residence|state\s+car|"
    r"list of|office of the"
    r")\b"
)


def looks_role_object_title(title: str, question: str = "") -> bool:
    """True for office artefacts (cars, residences, lists) mistaken for people.

    When the question itself asks about the artefact (car / residence / …),
    do not demote — the page is the intended answer.
    """
    t = (title or "").strip()
    if not t:
        return False
    if re.search(r"(?i)\b(car|residence|building|palace|vehicle)\b", question or ""):
        return False
    if _ROLE_OBJECT_TITLE.search(t):
        return True
    if re.search(r"(?i)^list of\b", t):
        return True
    if re.search(r"(?i)\b(car|residence|building|palace)\s*\(", t):
        return True
    if re.search(
        r"(?i)\b(prime minister|president|ceo|minister).{0,40}\b"
        r"(car|residence|building|palace)\b",
        t,
    ):
        return True
    return False


def is_predecessor_core(core: str | None, document: str, question: str = "") -> bool:
    """True when `core` only appears as a succeeded / replaced predecessor."""
    name = (core or "").strip()
    if not name or len(name.split()) > 6:
        return False
    if not re.search(
        r"(?i)\b(prime minister|president|ceo|pope|emperor|who is)\b",
        question or "",
    ):
        return False
    doc = document or ""
    if not re.search(re.escape(name), doc, flags=re.I):
        return False
    if not re.search(
        rf"(?i)\b(succeeding|succeeded|replacing|replaced|preceded by)\s+"
        rf"{re.escape(name)}\b",
        doc,
    ):
        return False
    first = (doc.strip().splitlines() or [""])[0]
    if re.search(
        rf"(?i)^(premiership|presidency)\s+of\s+{re.escape(name)}\b",
        first,
    ):
        return False
    # Subject attestation must attach to this name (not a neighbour sentence).
    if re.search(
        rf"(?i)\b{re.escape(name)}(?:'s)?\s+"
        rf"(?:has served as|tenure as|is the (?:current )?(?:prime minister|president))\b",
        doc,
    ):
        return False
    return True


def _strip_article(s: str) -> str:
    return re.sub(r"(?i)^(the|a|an)\s+", "", (s or "").strip()).strip()


def core_echoes_topic(question: str, core: str | None) -> bool:
    """True when the core merely repeats the asked-about org/place entity."""
    c = (core or "").strip()
    if not c:
        return False
    topic = question_topic(question or "")
    if not topic:
        m = re.search(
            r"(?i)\b(?:ceo|chief executive(?: officer)?|president|"
            r"prime minister|capital|population)\s+of\s+(.+?)\??\s*$",
            question or "",
        )
        if m:
            topic = m.group(1).strip()
    if not topic:
        return False
    cn, tn = extract.norm(_strip_article(c)), extract.norm(_strip_article(topic))
    if not cn or not tn:
        return False
    if cn == tn:
        return True
    if tn in cn and not has_person_name(c):
        return True
    if cn in tn and not has_person_name(c):
        return True
    return False


def wiki_page_summary(title: str) -> str:
    """Wikipedia lead extract for a title (MediaWiki extracts, REST fallback)."""
    if not (title or "").strip():
        return ""
    q = urllib.parse.urlencode({
        "action": "query",
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "titles": title,
        "format": "json",
    })
    try:
        raw = json.loads(_get(f"https://en.wikipedia.org/w/api.php?{q}"))
        pages = (raw.get("query") or {}).get("pages") or {}
        for page in pages.values():
            extract_text = (page.get("extract") or "").strip()
            if extract_text:
                return extract_text
    except Exception:
        pass
    try:
        page = json.loads(_get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote(title.replace(" ", "_"))
        ))
    except Exception:
        return ""
    return (page.get("extract") or page.get("description") or "").strip()


def wiki_page_extract(title: str, *, chars: int | None = 2500) -> str:
    """Plain-text extract beyond the lead (for birthplace / early-life facts).

    Pass ``chars=None`` to omit MediaWiki ``exchars`` (lead-only caps miss
    early-life birthplace sentences).
    """
    if not (title or "").strip():
        return ""
    params: dict[str, str] = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "titles": title,
        "format": "json",
    }
    if chars is not None:
        params["exchars"] = str(chars)
    q = urllib.parse.urlencode(params)
    try:
        raw = json.loads(_get(f"https://en.wikipedia.org/w/api.php?{q}"))
        pages = (raw.get("query") or {}).get("pages") or {}
        for page in pages.values():
            extract_text = (page.get("extract") or "").strip()
            if extract_text:
                return extract_text
    except Exception:
        pass
    return wiki_page_summary(title)


def enrich_hits_for_answer(
    question: str,
    hits: list[dict],
    *,
    summary_fn=None,
) -> list[dict]:
    """Replace thin office/fragment snippets with Wikipedia page summaries."""
    fetch = summary_fn or wiki_page_summary
    office = bool(re.search(
        r"(?i)\b(ceo|president|prime minister|chief executive)\b",
        question or "",
    ))
    want_when_year = bool(
        re.search(r"(?i)\bwhen\b", question or "")
        and not re.search(r"(?i)\bborn\b", question or "")
    )
    want_founded_when = bool(
        re.search(r"(?i)\bfounded\b", question or "")
        and re.search(r"(?i)\bwhen\b", question or "")
    )
    want_released_when = bool(
        re.search(r"(?i)\breleased\b", question or "")
        and re.search(r"(?i)\bwhen\b", question or "")
    )
    want_who_founded = bool(re.search(r"(?i)\bwho founded\b", question or ""))
    want_born_where = bool(
        re.search(r"(?i)\bborn\b", question or "")
        and re.search(r"(?i)\bwhere\b", question or "")
    )
    want_population = bool(re.search(r"(?i)\bpopulation\b", question or ""))
    want_birthday = bool(
        re.search(r"(?i)\b(birthday|birth date|date of birth|how old)\b", question or "")
    )
    want_born_when = bool(
        re.search(r"(?i)\bwhen\b", question or "")
        and re.search(r"(?i)\bborn\b", question or "")
    )
    out = []
    for h in hits:
        item = dict(h)
        snip = item.get("snippet") or ""
        title = item.get("title") or ""
        url = item.get("url") or ""
        need = looks_like_fragment(snip) or (office and not has_person_name(snip))
        # Keep strong CEO snippets when a person + role + asked org appear.
        topic = question_topic(question or "") or ""
        if (
            office
            and has_person_name(snip)
            and re.search(r"(?i)\b(ceo|chief executive officer)\b.{0,40}\b", snip)
            and topic
            and topic.casefold() in snip.casefold()
        ):
            need = False
        if want_when_year and not re.search(r"\b(?:1[89]\d{2}|20\d{2})\b", snip):
            need = True
        if want_founded_when and not re.search(r"(?i)\bfounded in\s+\d{4}\b", snip):
            need = True
        if want_released_when and not re.search(
            r"(?i)\b(?:released|unveiled|launched)\b.{0,40}\b(?:19|20)\d{2}\b",
            snip,
        ):
            need = True
        if want_who_founded and not re.search(
            r"(?i)\bfounded by\b|\bco-?founders?\b", snip
        ):
            need = True
        if want_born_where and not re.search(
            r"(?i)\bborn\b(?:[^.]{0,80}?)\bin\s+(?:the\s+city\s+of\s+)?[A-Z]",
            snip,
        ):
            need = True
        if want_population and not re.search(
            r"(?i)\bpopulation\b.*\d|\d[\d.,]*\s*(?:million|billion)", snip
        ):
            need = True
        if want_birthday and not re.search(
            r"(?i)\(born\s+\d|\bborn\s+(?:on\s+)?\d|\bborn\s+[A-Z][a-z]+\s+\d",
            snip,
        ):
            need = True
        if want_born_when and not re.search(
            r"(?i)\(born\s+\d|\bborn\s+(?:on\s+)?\d", snip
        ):
            need = True
        if need and title and "wikipedia.org" in url and "(disambiguation)" not in title.lower():
            if want_born_where:
                # Lead ``exchars`` truncates before early-life birthplace.
                extract_text = wiki_page_extract(title, chars=None) or fetch(title) or ""
            elif want_birthday or want_born_when:
                extract_text = wiki_page_extract(title) or fetch(title) or ""
            else:
                extract_text = fetch(title) or ""
            if extract_text:
                better = False
                q_snip = float(extract.term_score(question, snip))
                q_ext = float(extract.term_score(question, extract_text))
                if office and has_person_name(extract_text) and not has_person_name(snip):
                    better = True
                elif want_when_year and re.search(
                    r"\b(?:1[89]\d{2}|20\d{2})\b", extract_text
                ) and not re.search(r"\b(?:1[89]\d{2}|20\d{2})\b", snip):
                    better = True
                elif want_founded_when and re.search(
                    r"(?i)\bfounded in\s+\d{4}\b", extract_text
                ) and not re.search(r"(?i)\bfounded in\s+\d{4}\b", snip):
                    better = True
                elif want_released_when and re.search(
                    r"(?i)\b(?:released|unveiled|launched)\b.{0,40}\b(?:19|20)\d{2}\b",
                    extract_text,
                ) and not re.search(
                    r"(?i)\b(?:released|unveiled|launched)\b.{0,40}\b(?:19|20)\d{2}\b",
                    snip,
                ):
                    better = True
                elif want_who_founded and re.search(
                    r"(?i)\bfounded by\b", extract_text
                ) and not re.search(r"(?i)\bfounded by\b", snip):
                    better = True
                elif want_born_where and re.search(
                    r"(?i)\bborn\b(?:[^.]{0,80}?)\bin\s+(?:the\s+city\s+of\s+)?[A-Z]",
                    extract_text,
                ) and not re.search(
                    r"(?i)\bborn\b(?:[^.]{0,80}?)\bin\s+(?:the\s+city\s+of\s+)?[A-Z]",
                    snip,
                ):
                    better = True
                elif want_birthday and re.search(
                    r"(?i)\(born\s+\d|\bborn\s+(?:on\s+)?\d", extract_text
                ) and not re.search(
                    r"(?i)\(born\s+\d|\bborn\s+(?:on\s+)?\d", snip
                ):
                    better = True
                elif want_born_when and re.search(
                    r"(?i)\(born\s+\d|\bborn\s+(?:on\s+)?\d", extract_text
                ) and not re.search(
                    r"(?i)\(born\s+\d|\bborn\s+(?:on\s+)?\d", snip
                ):
                    better = True
                elif want_population and re.search(
                    r"(?i)\b\d[\d.,]*\s*(?:million|billion)", extract_text
                ) and not re.search(
                    r"(?i)\b\d[\d.,]*\s*(?:million|billion)", snip
                ):
                    better = True
                elif looks_like_fragment(snip) and not looks_like_fragment(extract_text):
                    if q_ext >= q_snip and (
                        has_person_name(extract_text)
                        or re.search(r"\d", extract_text)
                        or re.search(r"(?i)\bboils at\b", extract_text)
                    ):
                        better = True
                elif q_ext > q_snip + 0.5 and (
                    has_person_name(extract_text) or not looks_like_fragment(extract_text)
                ):
                    better = True
                if better:
                    # Prefer the birthplace sentence over a long page dump.
                    packed = extract_text
                    if want_born_where:
                        for sent in re.split(r"(?<=[.!?])\s+", extract_text):
                            if re.search(
                                r"(?i)\bborn\b(?:[^.]{0,80}?)\bin\s+"
                                r"(?:the\s+city\s+of\s+)?[A-Z]",
                                sent,
                            ):
                                packed = sent.strip()
                                break
                    item["snippet"] = packed[:900]
                    item["enriched"] = "wiki_summary"
        out.append(item)
    return out


def looks_historical_office(text: str) -> bool:
    """True when snippet describes a former / first / dated past office-holder."""
    t = text or ""
    if re.search(
        r"(?i)\b(former|previously served|first ceo|"
        r"was the (?:first )?(?:ceo|chief executive(?: officer)?|"
        r"president|prime minister))\b",
        t,
    ):
        return True
    if re.search(
        r"(?i)\bfrom\s+(?:[A-Za-z]+\s+)?\d{4}\s+to\s+(?:[A-Za-z]+\s+)?\d{4}\b",
        t,
    ):
        return True
    if re.search(r"(?i)\buntil\s+(?:[A-Za-z]+\s+)?\d{4}\b", t):
        return True
    return False


def looks_current_office(text: str) -> bool:
    """True when snippet signals incumbent / present tenure."""
    t = text or ""
    if looks_historical_office(t) and not re.search(r"(?i)\bsince\s+20\d{2}\b", t):
        return False
    return bool(
        re.search(
            r"(?i)\b(current|incumbent|has served as|serving as|"
            r"since\s+20\d{2})\b",
            t,
        )
    )


def rank_hits(question: str, hits: list[dict], k: int = 4) -> list[dict]:
    return [h for _, h in rank_hits_scored(question, hits, k=k)]


def rank_hits_scored(
    question: str, hits: list[dict], k: int = 4
) -> list[tuple[float, dict]]:
    scored = [(generic_hit_score(question, h), h) for h in hits]
    scored.sort(key=lambda x: (-x[0], hits.index(x[1]) if x[1] in hits else 0))
    return scored[:k]


def population_figure_grounded(core: str, document: str) -> bool:
    """True when a population figure appears as a whole token (not '4' from '123.4')."""
    c = (core or "").strip()
    doc = document or ""
    if not c or not doc:
        return False
    # Prefer exact phrase match for "123.4 million".
    if re.search(re.escape(c), doc, flags=re.I):
        token = c.split()[0]
        if re.search(r"[.]", token):
            return True
        if re.search(r"(?i)million|billion|thousand", c):
            return bool(
                re.search(
                    rf"(?<![\d.]){re.escape(token)}(?!\.\d)\s*"
                    r"(?:million|billion|thousand)\b",
                    doc,
                    flags=re.I,
                )
            )
        return bool(re.search(rf"(?<![\d.]){re.escape(token)}(?![\d.])", doc))
    return False


def fact_core_from_doc(question: str, document: str) -> str | None:
    """Lexical short core from notes (question-guided patterns + acronym/chem)."""
    doc = document or ""
    if not doc:
        return None
    typed = lex.lexical_core_from_doc(question or "", doc)
    if typed and core_fits_question(question, typed):
        return typed
    if re.search(r"(?i)\bchemical symbol\b", question or ""):
        for pat in (
            r"(?i)\b(?:chemical )?symbol (?:is |of |:|=)?\s*([A-Z][a-z]?)\b",
            r"(?i)\bsymbol\s+([A-Z][a-z]?)\b",
        ):
            m = re.search(pat, doc)
            if m and extract.verify(m.group(1), doc):
                return m.group(1)
    acr_m = re.search(r"(?i)^\s*what is\s+([A-Z]{2,8})\??\s*$", question or "")
    if acr_m:
        acr = acr_m.group(1)
        mm = re.search(
            rf"(?i)((?:[A-Z][A-Za-z]+(?:\s+(?:and\s+)?[A-Z][A-Za-z]+){{1,8}}))"
            rf"\s*\(\s*{re.escape(acr)}\s*\)",
            doc,
        )
        if mm:
            core = re.sub(r"\s+", " ", mm.group(1)).strip()
            if core.split()[0].casefold() == acr.casefold():
                core = " ".join(core.split()[1:]).strip()
            if len(core.split()) >= 3 and extract.verify(core.split()[0], doc):
                return core
        mm = re.search(
            rf"(?i)\b{re.escape(acr)}\b\s+is\s+(?:an?\s+|the\s+)?([^.]+)",
            doc,
        )
        if mm:
            core = mm.group(1).strip().rstrip(",;:")
            if re.search(r"(?i)\bU\.S\.\s*$", core):
                rest = doc[mm.end():]
                m2 = re.match(r"\s*([^.]+)", rest)
                if m2:
                    core = (core + " " + m2.group(1)).strip()
            if len(core.split()) >= 3 and extract.verify(core.split()[0], doc):
                return core
    return None


def hits_to_document(hits: list[dict], *, snippet_chars: int = 320, question: str = "") -> str:
    # Short packs: MiniCPM invents years/digits on long multi-sentence wiki text.
    # Prefix the title when the snippet is a fragment without the topic name.
    parts = []
    for h in hits:
        title = (h.get("title") or "").strip()
        snip = (h.get("snippet") or "").strip()
        limit = 560 if h.get("enriched") else snippet_chars
        if len(snip) > limit:
            # Prefer keeping a sentence that answers the question shape.
            kept = None
            if re.search(r"(?i)\bwhen\b", question or ""):
                for sent in re.split(r"(?<=[.!?])\s+", snip):
                    if re.search(r"\b(1[89]\d{2}|20\d{2})\b", sent):
                        kept = sent.strip()
                        break
            if not kept and re.search(
                r"(?i)\b(birthday|birth date|date of birth)\b", question or ""
            ):
                for sent in re.split(r"(?<=[.!?])\s+", snip):
                    if re.search(r"(?i)\(born\s+\d|\bborn\s+(?:on\s+)?\d", sent):
                        kept = sent.strip()
                        break
            if not kept and re.search(r"(?i)\bpopulation\b", question or ""):
                for sent in re.split(r"(?<=[.!?])\s+", snip):
                    if re.search(
                        r"(?i)\bpopulation\b.*\d|\d[\d.,]*\s*(?:million|billion)",
                        sent,
                    ):
                        kept = sent.strip()
                        break
            if not kept and re.search(r"(?i)\bcapital of\b", question or ""):
                for sent in re.split(r"(?<=[.!?])\s+", snip):
                    if re.search(r"(?i)\bis the capital\b", sent):
                        kept = sent.strip()
                        break
            if not kept and re.search(r"(?i)\bborn\b", question or ""):
                for sent in re.split(r"(?<=[.!?])\s+", snip):
                    if re.search(
                        r"(?i)\bborn\b(?:[^.]{0,80}?)\bin\s+"
                        r"(?:the\s+city\s+of\s+)?[A-Z]",
                        sent,
                    ):
                        kept = sent.strip()
                        break
            if not kept:
                for sent in re.split(r"(?<=[.!?])\s+", snip):
                    if has_person_name(sent) and len(sent.split()) >= 5:
                        kept = sent.strip()
                        break
            if kept and len(kept) <= limit:
                snip = kept
            else:
                snip = (kept or snip)[:limit].rsplit(" ", 1)[0] + "…"
        if title and snip and title.casefold() not in snip[:120].casefold():
            snip = f"{title} — {snip}"
        parts.append(
            f"{title}\n"
            f"{snip}\n"
            f"Source: {h.get('url', '').strip()}"
        )
    return "\n\n".join(parts)


# ---- search backends (stdlib) ----------------------------------------------

UA = "deku-web-search/0.1 (+https://github.com/local)"


def _get(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def search_wikipedia(query: str, limit: int = 5) -> list[dict]:
    """MediaWiki opensearch + extract. Stable smoke backend."""
    q = urllib.parse.quote(query)
    try:
        raw = json.loads(_get(
            f"https://en.wikipedia.org/w/api.php?action=opensearch&search={q}"
            f"&limit={limit}&namespace=0&format=json"
        ))
    except Exception:
        return []
    # [query, titles, descriptions, urls]
    titles, descs, urls = raw[1], raw[2], raw[3]
    hits = []
    for t, d, u in zip(titles, descs, urls):
        # enrich empty descriptions with a short extract
        snippet = d
        if not snippet:
            try:
                page = json.loads(_get(
                    "https://en.wikipedia.org/api/rest_v1/page/summary/"
                    + urllib.parse.quote(t.replace(" ", "_"))
                ))
                snippet = page.get("extract") or page.get("description") or ""
            except Exception:
                snippet = ""
        hits.append({"title": t, "snippet": snippet[:500], "url": u})
    return hits


def search_duckduckgo(query: str, limit: int = 5) -> list[dict]:
    """DuckDuckGo HTML. Best-effort; Wikipedia is the reliable fallback."""
    q = urllib.parse.quote_plus(query)
    try:
        body = _get(f"https://html.duckduckgo.com/html/?q={q}").decode("utf-8", "replace")
    except Exception:
        return []
    hits = []
    # result blocks: <a class="result__a" href=...>title</a> ... result__snippet
    for m in re.finditer(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
        r'.*?class="result__snippet[^"]*"[^>]*>(.*?)</(?:a|td|div)',
        body, re.S,
    ):
        url, title, snip = m.group(1), m.group(2), m.group(3)
        title = re.sub(r"<[^>]+>", "", html.unescape(title)).strip()
        snip = re.sub(r"<[^>]+>", "", html.unescape(snip)).strip()
        # DDG wraps redirects; keep as-is
        if title and snip:
            hits.append({"title": title, "snippet": snip[:500], "url": url})
        if len(hits) >= limit:
            break
    return hits


ROLE_NOISE = re.compile(
    r"(?i)\b(ceo|cfo|cto|founder|president|prime minister|chairman|company|parent|owner)\b"
)


def wiki_friendly_query(query: str) -> str:
    """Drop role words that poison MediaWiki opensearch ('CEO Apple' → fruit)."""
    q = ROLE_NOISE.sub(" ", query or "")
    q = re.sub(r"\s+", " ", q).strip()
    return q if q and q.lower() != (query or "").lower() else ""


def search_wikipedia_text(query: str, limit: int = 5) -> list[dict]:
    """Full-text MediaWiki search (better for 'Apple CEO' than opensearch)."""
    q = urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": str(limit), "format": "json",
    })
    try:
        raw = json.loads(_get(f"https://en.wikipedia.org/w/api.php?{q}"))
    except Exception:
        return []
    hits = []
    for row in (raw.get("query") or {}).get("search") or []:
        title = row.get("title") or ""
        snip = re.sub(r"<[^>]+>", "", html.unescape(row.get("snippet") or ""))
        if not snip:
            try:
                page = json.loads(_get(
                    "https://en.wikipedia.org/api/rest_v1/page/summary/"
                    + urllib.parse.quote(title.replace(" ", "_"))
                ))
                snip = page.get("extract") or ""
            except Exception:
                snip = ""
        url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
            title.replace(" ", "_")
        )
        hits.append({"title": title, "snippet": snip[:500], "url": url})
    return hits


def search(query: str, limit: int = 5, *, question: str = "") -> list[dict]:
    queries = expand_search_queries(question, query) if question else (
        [wiki_friendly_query(query), query] if wiki_friendly_query(query) else [query]
    )
    queries = [q for q in queries if q]
    hits: list[dict] = []
    seen: set[str] = set()

    def _add(batch: list[dict]) -> None:
        for h in batch:
            if h["url"] not in seen:
                seen.add(h["url"])
                hits.append(h)

    for q in queries:
        _add(search_wikipedia(q, limit=limit))
        _add(search_wikipedia_text(q, limit=limit))
    if len(hits) < 2:
        _add(search_duckduckgo(query, limit=limit))
    return hits[: max(limit * 3, 15)]


# ---- LFM prompts ------------------------------------------------------------

INTENT_PROMPT = """Classify the user question into exactly one label.
Labels: search (needs the web), extract (answer is already in a provided document), refuse (math, code, chitchat, or unsafe).
Reply with one word only: search OR extract OR refuse.

Question: {question}
"""

QUERY_PROMPT = """Write one web search query that would find the answer.
No quotes, no explanation, one line only.

Question: {question}
"""


def needle_intent(question: str) -> str:
    """Needle tool call → search|extract|refuse."""
    from needle import Needle, tool, Field

    @tool
    def classify(intent: str = Field(
        ..., description="search | extract | refuse"
    )):
        """Classify the user question for a retrieval agent.
        search = needs the public web (companies, people, places, facts).
        extract = answer already in a provided document.
        refuse = math, code, chitchat, or unsafe.
        """
        return intent

    n = Needle(
        tools=[classify],
        system="Always call classify. Company and factual who/what questions are search.",
    )
    try:
        r = n.complete(f"Question: {question}", 64)
        calls = r.get("function_calls") or []
        if calls:
            intent = (calls[0].get("arguments") or {}).get("intent")
            if intent in INTENTS:
                return intent
    finally:
        try:
            n.reset()
        except Exception:
            pass
    return rule_intent(question)


def needle_query(question: str) -> str:
    from needle import Needle, tool, Field

    @tool
    def web_search(query: str = Field(..., description="short keyword search query")):
        """Search the public web for companies, people, places, facts."""
        return query

    n = Needle(
        tools=[web_search],
        system="For factual questions call web_search with short keywords.",
    )
    try:
        r = n.complete(question, 64)
        calls = r.get("function_calls") or []
        if calls:
            q = (calls[0].get("arguments") or {}).get("query")
            if q and str(q).strip():
                return str(q).strip()
    finally:
        try:
            n.reset()
        except Exception:
            pass
    return rule_query(question)


def rule_query(question: str) -> str:
    """Strip question syntax into rough keywords when the router abstains."""
    m = re.match(
        r"(?i)^\s*how old (?:is|are) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\??\s*$",
        question or "",
    )
    if m:
        return m.group(1).strip()
    q = re.sub(r"(?i)^(what|who|when|where|which|how|why)\b", "", question or "")
    q = re.sub(r"(?i)\b(is|are|was|were|the|a|an|of|company|mean|means)\b", " ", q)
    q = re.sub(r"[?!.]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q or (question or "").strip()


def recover_core(raw: str, document: str, question: str) -> str | None:
    """Pull a grounded name/number out of a messy extract completion."""
    body = (raw or "").strip()
    body = re.sub(r"(?i)^answer:\s*", "", body).strip()
    body = re.sub(r"^\s*\d+[.)]\s*", "", body).strip()
    # Prefer multi-word proper names, then single Caps, then years.
    candidates = []
    if re.search(r"(?i)\bchemical symbol\b", question or ""):
        candidates += re.findall(r"\b([A-Z][a-z]?)\b", body)
    candidates += re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", body)
    candidates += re.findall(r"\b([A-Z][a-z]{2,})\b", body)
    candidates += re.findall(r"\b(\d{3,4})\b", body)
    # Chemical symbols like Au
    candidates += re.findall(r"\b([A-Z][a-z]?)\b", body)
    seen = set()
    for cand in candidates:
        key = cand.casefold()
        if key in seen:
            continue
        seen.add(key)
        if not extract.verify(cand, document):
            continue
        if core_fits_question(question, cand):
            return cand
    # Whole cleaned body if it is a short grounded span.
    if body and len(body.split()) <= 6 and extract.verify(body, document):
        if core_fits_question(question, body):
            return body
    return fact_core_from_doc(question, document)


def minicpm_extract(question: str, document: str, seed: int = 0) -> tuple[str | None, str]:
    """Grounded core span via MiniCPM. Returns (answer|None, status).

    Question-term filtering is skipped for the core: a short name like
    "Tim Cook" is often the right span but fails `relevant(question, quote)`.
    Retries once with seed+1; recovers names from list-style answers like '1. Paris'.

    Uses chat completions (no reply prefill).     Prefill + /v1/completions is an alternate raw-completion path;
    llama-server GGUF with --jinja emits degenerate
    digit loops on that shape — measured on MiniCPM5-1B-Q4_K_M.
    """
    from deku import llm

    last_status = "no_answer"
    for s in (seed, seed + 1):
        raw = llm.complete(
            extract.PROMPT.format(question=question, doc=document),
            think=False,
            temp=extract.TEMP,
            seed=s,
            max_tokens=extract.MAX_TOKENS,
        )
        body = re.sub(r"(?i)^\s*answer:\s*", "", (raw or "").strip())
        cleaned = re.sub(r"^\s*\d+[.)]\s*", "", body)
        hit, status = extract.classify("web", cleaned, document, question=None)
        last_status = status
        if hit and (hit.get("answer") or "").strip():
            ans = (hit.get("answer") or "").strip()
            if core_fits_question(question, ans):
                return ans, status
            recovered = recover_core(body, document, question)
            if recovered:
                return recovered, "ok"
            # Wrong-type extract — keep looking / fall through.
            last_status = "bad_core_type"
            continue
        recovered = recover_core(body, document, question)
        if recovered:
            return recovered, "ok"
    fact = fact_core_from_doc(question, document)
    if fact:
        return fact, "doc_core"
    return None, last_status


SUMMARIZE_PROMPT = """Answer from the notes. Be direct.
- The title line names the topic (often a person or company).
- Incomplete sentences still count: use names that appear in the notes.
- Reply in 1 full English sentence (at least 6 words) that answers the question.
- Include the person, place, or organization name from the notes in that sentence.
- Do not refuse if a name in the notes answers the question.

Question: {question}

Notes:
{doc}
"""


def minicpm_summarize(question: str, document: str, seed: int = 0) -> str:
    """Natural-language one-liner from notes (no-think). May hallucinate — verify."""
    from deku import llm

    raw = llm.complete(
        SUMMARIZE_PROMPT.format(question=question, doc=document),
        think=False, temp=0.3, seed=seed, max_tokens=80,
    )
    return (raw or "").strip().splitlines()[0].strip() if (raw or "").strip() else ""


def core_in_reply(core: str, reply: str) -> bool:
    """True when the verified extract span still appears in the reply text."""
    c, r = extract.norm(core or ""), extract.norm(reply or "")
    return bool(c) and c in r


def is_degenerate_core(core: str | None, question: str = "") -> bool:
    """True when a core is question-echo, glue, or too thin to be an answer."""
    c = (core or "").strip()
    if not c:
        return True
    # Sentence bleed into the next clause: "Tokyo. Throughout"
    if re.search(r"\.\s+[A-Za-z]", c):
        return True
    # Allow short chemical symbols; reject other 1-char crumbs.
    if len(c) < 2:
        return True
    if len(c) == 1 and not re.fullmatch(r"[A-Za-z]", c):
        return True
    if re.fullmatch(
        r"(?i)located in|based in|known as|referred to|the capital|"
        r"the population|the headquarters|a company|an organization",
        c,
    ):
        return True
    words = c.split()
    stopish = frozenset("""
        a an the and or but if then so as at by for from in into of on onto to with
        is are was were be been being has have had do does did will would can could
        that this these those it its they them he she his her who whom which what
        when where why how not no yes also just only about than more most such
        located based known called named capital population headquarters company
        organization
        """.split())
    if words and all(w.casefold() in stopish for w in words):
        return True
    qn = extract.norm(question or "")
    cn = extract.norm(c)
    if qn and cn and cn in qn and len(words) <= 3:
        if re.search(
            r"(?i)^(the\s+)?(capital|population|headquarters|ceo|"
            r"president|prime minister|founder|author|company)\b",
            c,
        ):
            return True
    if core_echoes_topic(question, c):
        return True
    return False


def question_topic(question: str) -> str | None:
    """Primary entity the question asks about (for hit identity checks)."""
    q = (question or "").strip()
    for pat in (
        r"(?i)capital of (.+?)\??\s*$",
        r"(?i)population of (.+?)\??\s*$",
        r"(?i)who founded (.+?)\??\s*$",
        r"(?i)who wrote (.+?)\??\s*$",
        r"(?i)what company makes (?:the )?(.+?)\??\s*$",
        r"(?i)(?:ceo|chief executive(?: officer)?|president|prime minister) of (.+?)\??\s*$",
        r"(?i)when (?:was|were) (?:the )?(.+?) founded",
        r"(?i)when (?:was|were) (?:the )?(.+?) released",
        r"(?i)where (?:is|are) (.+?) (?:headquartered|based)",
        r"(?i)headquarters of (.+?)\??\s*$",
        r"(?i)(?:birthday|birth date|date of birth)\s+of\s+(?:the\s+)?(.+?)\??\s*$",
        r"(?i)what is (.+?)(?:'s)? birthday",
    ):
        m = re.search(pat, q)
        if m:
            return m.group(1).strip().strip("\"'").rstrip("?.")
    return None


def hit_title_matches_topic(title: str, topic: str | None) -> bool:
    if not topic:
        return True
    t = (title or "").strip()
    top = topic.strip()
    if not t or not top:
        return True
    tn, to = extract.norm(t), extract.norm(top)
    if to == tn:
        return True
    # Prefix / parenthetical / comma forms: "Stripe, Inc.", "Hamlet (play)".
    if tn.startswith(to + " ") or tn.startswith(to + ",") or tn.startswith(to + " ("):
        return True
    # Whole-token containment only (avoid Peru ⊂ Perugia).
    if re.search(rf"(?<![a-z0-9]){re.escape(to)}(?![a-z0-9])", tn):
        return True
    return False


def person_attested(core: str, document: str) -> bool:
    """True when the full person name or a distinctive surname is in the doc."""
    c = (core or "").strip()
    doc = document or ""
    if not c or not doc:
        return False
    if extract.norm(c) in extract.norm(doc):
        return True
    parts = [p for p in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", c) if len(p) > 1]
    if len(parts) >= 2 and extract.norm(parts[-1]) in extract.norm(doc):
        return True
    return False


_MONTHS = frozenset("""
january february march april may june july august september october
november december
""".split())

_WEEKDAYS = frozenset("""
monday tuesday wednesday thursday friday saturday sunday
""".split())

# Lowercase glue MiniCPM uses when paraphrasing attested facts (no entity names).
_PARAPHRASE_OK = frozenset("""
makes made making locate located named names holds held create created
owns owned founded based known called became become company status
largest deepest oceanic divisions engineer after parent holding
marketed developed smartphones line run runs system located tower
ended end ends concluded conclude ocean oceans
""".split())

_GROUND_STOP = frozenset("""
a an the and or but if then so as at by for from in into of on onto to with
is are was were be been being has have had do does did will would can could
that this these those it its they them he she his her who whom which what
when where why how not no yes also just only about than more most such
""".split())


def claim_tokens(text: str) -> list[str]:
    """Numbers, capitalized names, and calendar words a summary must not invent."""
    out = []
    for i, m in enumerate(re.finditer(
        r"[A-Za-z][A-Za-z'-]*|\d+(?:st|nd|rd|th)?", text or ""
    )):
        tok = m.group(0)
        low = tok.casefold()
        if any(c.isdigit() for c in tok):
            out.append(tok)
            continue
        if low in _MONTHS or low in _WEEKDAYS:
            out.append(tok)
            continue
        if not tok[0].isupper() or len(tok) < 2:
            continue
        if i == 0 and low in ("the", "a", "an"):
            continue
        out.append(tok)
    return out


def content_tokens(text: str) -> list[str]:
    """Lowercase content words that are not paraphrase-glue."""
    out = []
    for m in re.finditer(r"[A-Za-z][A-Za-z'-]*", text or ""):
        tok = m.group(0)
        low = tok.casefold()
        if tok[0].isupper():
            continue  # handled as claim_tokens
        if low in _GROUND_STOP or low in _PARAPHRASE_OK or low in _MONTHS:
            continue
        if len(low) < 5:
            continue
        out.append(low)
    return out


def _reply_question_terms(question: str) -> frozenset[str]:
    """Content terms from the question — paraphrase need not re-hit the notes."""
    return frozenset(
        w
        for w in re.findall(r"[a-z]{4,}", (question or "").casefold())
        if w not in _GROUND_STOP and w not in _MONTHS
    )


_CAUSAL_GLUE = (
    "because", "due to", "which means", "therefore", "so that",
)


def reply_grounded(reply: str, document: str, *, question: str = "") -> bool:
    """True when claim tokens and non-glue content words appear in `document`.

    Catches invented dates ('May 30th', 'july') and invented content verbs
    ('annexed') while allowing light paraphrase ('makes', 'located').
    Causal glue ('because') must also be attested — blocks wrong explanations
    built from individually attested words.
    Lowercase words from the question itself are not required in the notes.
    """
    if not (reply or "").strip() or not (document or "").strip():
        return False
    doc = extract.norm(document)
    low_reply = (reply or "").casefold()
    low_doc = (document or "").casefold()
    q_terms = _reply_question_terms(question)
    for phrase in _CAUSAL_GLUE:
        if phrase in low_reply and phrase not in low_doc:
            return False
    for tok in claim_tokens(reply):
        if not extract.has_term(extract.norm(tok), doc):
            return False
    for tok in content_tokens(reply):
        if tok in q_terms:
            continue
        if not extract.has_term(tok, doc):
            return False
    return True


def sentence_with_core(
    core: str, document: str, question: str = ""
) -> str | None:
    """Best document sentence containing the core (skips anaphora / bare titles)."""
    if not core or not document:
        return None
    c = extract.norm(core)
    bad_start = re.compile(
        r"(?i)^(because of this|due to this|as a result|therefore|however|"
        r"this |that |it |they |these |those )\b"
    )
    candidates: list[tuple[float, int, str]] = []
    for line in document.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("source:"):
            continue
        nline = extract.norm(line)
        if c not in nline:
            continue
        if nline == c or len(line.split()) <= 3:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line) if re.search(r"[.!?]", line) else [line]
        for part in parts:
            sent = part.strip()
            if c not in extract.norm(sent) or len(sent.split()) <= 3:
                continue
            if bad_start.match(sent):
                continue
            if len(sent.split()) > 45:
                continue
            sc = float(extract.term_score(question, sent)) if question else 0.0
            # Prefer sentences that look like direct answers.
            if re.search(r"(?i)\b(is|was|are|were|wrote|founded|located|boils)\b", sent):
                sc += 1.0
            candidates.append((sc, -len(sent.split()), sent))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    return candidates[0][2]


def best_sentence_for_question(question: str, document: str) -> str | None:
    """Highest question-term-score sentence in `document` (no relation class)."""
    if not (document or "").strip():
        return None
    bad_start = re.compile(
        r"(?i)^(because of this|due to this|as a result|therefore|however|"
        r"this |that |it |they |these |those )\b"
    )
    candidates: list[tuple[float, int, str]] = []
    for line in document.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("source:"):
            continue
        parts = (
            re.split(r"(?<=[.!?])\s+", line)
            if re.search(r"[.!?]", line)
            else [line]
        )
        for part in parts:
            sent = part.strip()
            if len(sent.split()) < 5 or len(sent.split()) > 45:
                continue
            if bad_start.match(sent):
                continue
            sc = float(extract.term_score(question, sent)) if question else 0.0
            if sc <= 0:
                continue
            if re.search(
                r"(?i)\b(is|was|are|were|wrote|founded|located|boils|"
                r"appointed|served|became)\b",
                sent,
            ):
                sc += 1.0
            candidates.append((sc, -len(sent.split()), sent))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def compose_reply(
    core: str | None,
    summary: str | None,
    document: str,
    question: str = "",
) -> str | None:
    """Score grounded candidates; prefer attested summary / sentence / core."""
    numeric_core = bool(core and re.fullmatch(r"[\d.,]+", core.strip()))
    core_ok = bool(core) and not is_degenerate_core(core, question)
    if core_ok and core_echoes_topic(question, core):
        core_ok = False
    if core_ok and is_predecessor_core(core, document, question):
        core_ok = False
    candidates: list[tuple[float, str, str]] = []

    if (
        not numeric_core
        and summary
        and len(summary.split()) >= MIN_SUMMARY_WORDS
        and reply_grounded(summary, document, question=question)
    ):
        sc = 8.0
        if core_ok and core_in_reply(core or "", summary):
            sc = 12.0
        elif not core_ok:
            sc = 11.0
        else:
            sc = 7.0
        if (
            not core_ok
            and core
            and has_person_name(summary)
            and extract.norm(core) not in extract.norm(summary)
        ):
            sc = 12.5
        candidates.append((sc, summary.strip(), "summary"))

    if core_ok and core:
        sent = sentence_with_core(core, document, question=question)
        if sent:
            candidates.append((8.5, sent, "sentence"))
        elif (
            re.search(r"(?i)\bwho\b", question or "")
            and person_attested(core, document)
            and (
                has_person_name(core)
                or re.fullmatch(
                    r"[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+)+",
                    core.strip(),
                )
            )
        ):
            name = core.strip()
            if not name.endswith("."):
                name += "."
            candidates.append((9.2, name, "core_name"))
        if (
            person_attested(core, document)
            or extract.norm(core) in extract.norm(document)
        ):
            candidates.append((3.0, core.strip(), "core"))

    if (not core_ok) and document:
        best = best_sentence_for_question(question, document)
        if best and reply_grounded(best, document, question=question):
            candidates.append((9.0, best, "best_sentence"))
        elif best:
            candidates.append((8.0, best, "best_sentence_loose"))

    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def core_fits_question(question: str, core: str | None) -> bool:
    """Reject cores that cannot be the asked-for answer type."""
    c = (core or "").strip()
    if not c:
        return False
    if re.fullmatch(r"(?i)yes|no|true|false", c):
        return False
    if core_echoes_topic(question, c):
        return False
    if re.search(r"(?i)\bwho\b", question or "") and re.fullmatch(r"[\d\s./-]+", c):
        return False
    # "Who is the CEO/PM/…" needs a person-shaped core, not an org echo.
    if re.search(
        r"(?i)\bwho\b.+\b(ceo|chief executive|prime minister|president|pope|emperor|"
        r"founded|wrote)\b|\bwho\s+is\s+the\s+(ceo|prime minister|president)\b",
        question or "",
    ):
        if has_person_name(c) or re.search(r"(?i)^pope\s+\S+", c):
            pass
        elif re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", c):
            pass
        elif re.fullmatch(r"[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+)*", c):
            # Mononym / surname / latin-extended (Pichai, Naruhito, Kōji Satō).
            pass
        else:
            return False
    if re.search(r"(?i)\bwho founded\b", question or ""):
        # Founders are people (or "A and B"), not dates / years.
        if re.search(r"\d{4}", c) and not re.search(
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", c
        ):
            return False
        if re.match(r"(?i)^(january|february|march|april|may|june|july|"
                    r"august|september|october|november|december)\b", c):
            return False
    if re.search(r"(?i)\bwho\b", question or "") and re.search(
        r"(?i)\d+(st|nd|rd|th)\s+president\b", c
    ):
        return False
    if re.search(r"(?i)\bwhen\b.*\bfounded\b|\bfounded\b.*\bwhen\b", question or ""):
        if not re.search(r"\d{4}", c):
            return False
        # Reject person names mistaken for founding years.
        if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", c):
            return False
    elif re.search(r"(?i)\bwhen\b", question or "") and not re.search(r"\d", c):
        return False
    if re.search(r"(?i)\b(birthday|birth date|date of birth)\b", question or "") and not re.search(
        r"\d", c
    ):
        return False
    if re.search(r"(?i)\bpopulation\b", question or "") and not re.search(r"\d", c):
        return False
    if re.search(r"(?i)\b(headquarters?|headquartered|based)\b", question or ""):
        # Org named in the question is never the place answer.
        org_m = re.search(
            r"(?i)(?:where (?:is|are)|headquarters of)\s+(.+?)"
            r"(?:\s+(?:headquartered|based))?\??\s*$",
            question or "",
        )
        if not org_m:
            org_m = re.search(
                r"(?i)where (?:is|are) (.+?) (?:headquartered|based)",
                question or "",
            )
        if org_m:
            org = org_m.group(1).strip().rstrip("?.")
            org = re.sub(r"(?i)^(the)\s+", "", org).strip()
            if org and extract.norm(c) == extract.norm(org):
                return False
            if org and extract.norm(org) in extract.norm(c) and len(c.split()) <= 2:
                return False
    if re.search(r"(?i)\bborn\b", question or "") and re.search(
        r"(?i)\bwhere\b", question or ""
    ):
        # Birthplace must look like a place, not a person fragment / article.
        if re.fullmatch(r"(?i)the|a|an|he|she|they|him|her", c):
            return False
        if re.search(
            r"(?i)^(january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\b",
            c,
        ):
            return False
        if re.fullmatch(r"\d{4}", c):
            return False
        who = re.search(
            r"(?i)where (?:was|is) (.+?) born", question or ""
        )
        if who and extract.norm(c) in extract.norm(who.group(1)):
            return False
        if len(c) < 3:
            return False
    if re.search(r"(?i)\bchemical symbol\b", question or ""):
        # Atomic numbers are not chemical symbols.
        if re.fullmatch(r"\d+", c):
            return False
        # Prefer 1–2 letter element symbols (Fe, Au, O).
        if not re.fullmatch(r"[A-Z][a-z]?", c) and not re.search(
            r"(?i)\b[A-Z][a-z]?\b", c
        ):
            return False
    m = re.search(r"(?i)^\s*who wrote (.+?)\??\s*$", question or "")
    if m:
        work = m.group(1).strip().strip('"\'')
        if work and extract.norm(c) == extract.norm(work):
            return False
    m = re.search(r"(?i)what company makes (?:the )?(.+?)\??\s*$", question or "")
    if m:
        product = m.group(1).strip().rstrip("?.")
        # Reject product-page titles that embed the product ("X PlayStation").
        if (
            product
            and extract.norm(product) in extract.norm(c)
            and extract.norm(c) != extract.norm(product)
        ):
            return False
    return True


def should_abstain(
    *, question: str, doc: str, score: float, core: str | None
) -> bool:
    """Refuse when the top hit is weak or off-topic, or no grounded core."""
    if score < MIN_HIT_SCORE:
        return True
    doc_score = float(extract.term_score(question, doc))

    def _usable(c: str | None) -> bool:
        return bool(
            c
            and not is_degenerate_core(c, question)
            and core_fits_question(question, c)
            and extract.norm(c) in extract.norm(doc)
        )

    if doc_score < MIN_HIT_SCORE:
        # Short grounded cores (person/symbol/company) often sit in docs that
        # share few question terms ("Hamlet" page vs "who wrote").
        if _usable(core):
            return False
        fact = fact_core_from_doc(question, doc)
        if _usable(fact):
            return False
        return True
    # Degenerate / missing core: let compose_reply try summary or fact_core.
    if not core or is_degenerate_core(core, question):
        return False
    if not core_fits_question(question, core):
        return True
    return False


def run(question: str, *, router: str = "rule", k: int = 4,
        seed: int = 0, use_needle_slots: bool = False) -> Result:
    """Full web_search episode. `router`: needle | rule.

    extract (core) → summarize → keep summary only if core ⊆ reply and every
    claim token in the summary appears in the notes; otherwise fall back to a
    source sentence. Abstain when evidence is weak.

    Typed slot / office-title / age-template shortcuts removed — lexical
    extract + MiniCPM + compose_reply. Named age is ``web → calc`` via
    orchestrate (plan_id age_years).
    """
    if router == "needle":
        intent = needle_intent(question)
    else:
        intent = rule_intent(question)
    out = Result(intent=intent, detail={"router": router})
    if intent != "search":
        out.status = "skipped"
        return out
    query = needle_query(question) if router == "needle" else rule_query(question)
    out.query = query
    hits = search(query, question=question)
    out.detail["raw_hits"] = len(hits)
    scored = rank_hits_scored(question, hits, k=max(k, 4))
    # Enrich thin/office snippets, then re-rank so complete answers can rise.
    enriched = enrich_hits_for_answer(question, [h for _, h in scored])
    if any(h.get("enriched") for h in enriched):
        out.detail["enriched"] = True
    scored = rank_hits_scored(question, enriched, k=k)
    out.hits = [h for _, h in scored]
    if not scored:
        out.status = "no_hits"
        out.answer = CANNOT_ANSWER
        return out
    top_score, top = scored[0]
    out.detail["top_score"] = top_score
    doc = hits_to_document([top], question=question)
    out.document = doc
    doc_score = float(extract.term_score(question, doc))
    out.detail["doc_score"] = doc_score
    if top_score < MIN_HIT_SCORE:
        out.answer, out.status = CANNOT_ANSWER, "cannot_answer"
        out.detail["abstain_reason"] = "weak_or_off_topic_hit"
        return out
    fact = fact_core_from_doc(question, doc)

    def _pop_ok(val: str | None) -> bool:
        if not val:
            return False
        if not re.search(r"(?i)\bpopulation\b", question or ""):
            return True
        return population_figure_grounded(val, doc)

    if (
        fact
        and core_fits_question(question, fact)
        and _pop_ok(fact)
    ):
        core, status = fact, "doc_core"
    elif fact and re.search(r"(?i)^\s*what is\s+[A-Z]{2,8}\??\s*$", question or ""):
        core, status = fact, "doc_core"
    else:
        core, status = minicpm_extract(question, doc, seed=seed)
        if core and not _pop_ok(core):
            core = None
        if (not core or not core_fits_question(question, core)) and fact and _pop_ok(
            fact
        ):
            core, status = fact, "doc_core"
    if core and (
        is_degenerate_core(core, question)
        or core_echoes_topic(question, core)
        or is_predecessor_core(core, doc, question)
    ):
        out.detail["core_rejected"] = core
        rejected = core
        core, status = None, "core_rejected"
        if (
            fact
            and extract.norm(fact) != extract.norm(rejected)
            and not is_degenerate_core(fact, question)
            and not core_echoes_topic(question, fact)
            and not is_predecessor_core(fact, doc, question)
            and core_fits_question(question, fact)
            and _pop_ok(fact)
        ):
            core, status = fact, "doc_core"
    out.detail["core"] = core
    out.detail["extract_status"] = status
    if should_abstain(question=question, doc=doc, score=top_score, core=core):
        out.answer, out.status = CANNOT_ANSWER, "cannot_answer"
        out.detail["abstain_reason"] = "no_grounded_core"
        return out
    summary = minicpm_summarize(question, doc, seed=seed)
    out.detail["summary"] = summary
    if summary and core and core_in_reply(core, summary) and not reply_grounded(summary, doc, question=question):
        out.detail["summary_rejected"] = "ungrounded_claims"
    reply = compose_reply(core, summary, doc, question=question)
    if not reply:
        out.answer, out.status = CANNOT_ANSWER, "cannot_answer"
        out.detail["abstain_reason"] = "empty_reply"
        return out
    if reply.strip() == (core or "").strip() and not sentence_with_core(core, doc, question=question):
        out.answer, out.status = CANNOT_ANSWER, "cannot_answer"
        out.detail["abstain_reason"] = "core_without_sentence"
        out.detail["reply_source"] = "core"
        return out
    if not out.detail.get("reply_source"):
        if summary and reply == summary.strip():
            out.detail["reply_source"] = "summary"
        elif reply != core:
            out.detail["reply_source"] = "source_sentence"
        else:
            out.detail["reply_source"] = "core"
    out.answer, out.status = reply, "ok"
    return out
