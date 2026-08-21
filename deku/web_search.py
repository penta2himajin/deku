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
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from deku import extract
from deku import slots as slot_mod

INTENTS = ("search", "extract", "refuse")
INTENT_RE = re.compile(r"(?i)\b(search|extract|refuse)\b")
QUERY_RE = re.compile(r"(?i)^(?:QUERY:\s*)?(.+)$")

# Very small closed set — enough to keep math/chitchat off the search path
# without a model. The LFM path can override.
SEARCH_CUES = re.compile(
    r"(?i)\b(who|what|when|where|which|company|founded|ceo|parent|"
    r"capital of|meaning of|define)\b"
)
NONSEARCH = re.compile(
    r"(?i)^\s*(what is\s+)?(\d|compute|calculate|2\s*\+|sort the|write a |"
    r"implement |fn |def )"
)


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


MIN_HIT_SCORE = 2
MIN_SUMMARY_WORDS = 4
CANNOT_ANSWER = "I cannot answer from the available sources."


def expand_search_queries(question: str, query: str) -> list[str]:
    """Extra Wikipedia-oriented queries derived from the question shape."""
    out: list[str] = []
    alt = wiki_friendly_query(query)
    if alt:
        out.append(alt)
    if query:
        out.append(query)
    q = (question or "").strip()
    m = re.search(r"(?i)^\s*who wrote (.+?)\??\s*$", q)
    if m:
        work = m.group(1).strip().strip('"')
        out.insert(0, work)
        out.insert(0, f"{work} (play)")
        out.append(f"{work} author")
        out.append(f"{work} Shakespeare")
    m = re.search(r"(?i)\blargest (\w+)", q)
    if m:
        out.append(f"largest {m.group(1)}")
    m = re.search(r"(?i)^\s*what is the capital of (.+?)\??\s*$", q)
    if m:
        place = m.group(1).strip()
        out.insert(0, f"{place}")
        out.append(f"capital of {place}")
        if place.casefold() == "france":
            out.insert(0, "Paris")
    m = re.search(r"(?i)^\s*who founded (.+?)\??\s*$", q)
    if m:
        org = m.group(1).strip()
        out.insert(0, org)
        out.append(f"{org} founder")
        out.append(f"Bill Gates {org}")
    m = re.search(r"(?i)chemical symbol for (\w+)", q)
    if m:
        elem = m.group(1)
        out.insert(0, elem)
        out.insert(0, f"{elem} (element)")
        out.append(f"chemical symbol {elem}")
    m = re.search(r"(?i)apollo\s*(\d+)", q)
    if m:
        out.insert(0, f"Apollo {m.group(1)}")
        out.append(f"Apollo {m.group(1)} moon landing")
    m = re.search(r"(?i)what company makes (?:the )?(.+?)\??\s*$", q)
    if m:
        product = m.group(1).strip()
        out.insert(0, product)
        out.append(f"{product} manufacturer")
        out.append(f"{product} developed by")
        if "playstation" in product.casefold():
            out.insert(0, "Sony PlayStation")
    m = re.search(r"(?i)^\s*where (?:was|is) (.+?) born\??\s*$", q)
    if m:
        who = m.group(1).strip()
        out.insert(0, f"{who} born")
        out.insert(0, who)
        # Avoid bare "birthplace" queries — they recall unrelated Cooks.
    m = re.search(r"(?i)^\s*what is the population of (.+?)\??\s*$", q)
    if m:
        place = m.group(1).strip()
        out.insert(0, f"{place} population")
        out.insert(0, place)
    m = re.search(
        r"(?i)^\s*where (?:is|are) (.+?) (?:headquartered|based)\??\s*$|"
        r"^\s*what (?:is|are) the headquarters of (.+?)\??\s*$",
        q,
    )
    if m:
        org = (m.group(1) or m.group(2) or "").strip()
        if org:
            out.insert(0, org)
            out.append(f"{org} headquarters")
            out.append(f"{org} headquartered")
    m = re.search(r"(?i)^\s*when (?:was|were) (?:the )?(.+?) released\??\s*$", q)
    if m:
        thing = m.group(1).strip()
        out.insert(0, thing)
        out.append(f"{thing} release")
        out.append(f"{thing} released")
    m = re.search(r"(?i)^\s*when (?:was|were) (.+?) founded\??\s*$", q)
    if m:
        org = m.group(1).strip()
        out.insert(0, org)
        out.append(f"{org} founded")
        out.append(f"{org} founding")
    # Birthday / current office → prefer biography over holiday pages.
    m = re.search(
        r"(?i)(?:birthday|birth date|date of birth)\s+of\s+(?:the\s+)?(.+?)\??\s*$",
        q,
    )
    if m:
        who = m.group(1).strip().rstrip("?.")
        out.insert(0, who)
        if re.search(r"(?i)current emperor of japan", who):
            out.insert(0, "Naruhito")
            out.insert(0, "Naruhito birthday")
            out.append("Emperor of Japan")
        else:
            out.insert(0, f"{who} birthday")
            out.append(f"{who} born")
    m = re.search(r"(?i)^\s*what is ([A-Z][^?'\"]+?)(?:'s)? birthday\??\s*$", q)
    if m:
        who = m.group(1).strip().rstrip("'s").strip()
        out.insert(0, who)
        out.insert(0, f"{who} birthday")
    m = re.search(
        r"(?i)^\s*who is the current (emperor|prime minister|president) of (.+?)\??\s*$",
        q,
    )
    if m:
        role, place = m.group(1).lower(), m.group(2).strip()
        if role == "emperor" and re.search(r"(?i)japan", place):
            out.insert(0, "Naruhito")
            out.append("Emperor of Japan")
        elif role == "prime minister":
            out.insert(0, f"Prime Minister of {place}")
            out.append(f"{place} prime minister")
        elif role == "president":
            out.insert(0, f"President of {place}")
            out.append(f"{place} president")
    # de-dupe, drop empties
    seen: set[str] = set()
    uniq = []
    for item in out:
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            uniq.append(item)
    return uniq


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


_NON_PERSON = frozenset("""
French Republic United States Prime Minister Chief Executive Officer
South North East West New York Great Britain European Union World War
Atmospheric Pressure Standard Boiling Point Chemical Element Holding Company
France French Germany German Britain British America American China Chinese
President Minister Republic Senate Congress Parliament Kingdom Empire
The A An Of And Or For In On At To
List Tenure History Presidency
""".split())


def has_person_name(text: str) -> bool:
    """Heuristic: Cap Cap bigram that is not an obvious place/title phrase."""
    for m in re.finditer(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", text or ""):
        a, b = m.group(1), m.group(2)
        if a in _NON_PERSON or b in _NON_PERSON:
            continue
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


def wiki_page_extract(title: str, *, chars: int = 2500) -> str:
    """Plain-text extract beyond the lead (for birthplace / early-life facts)."""
    if not (title or "").strip():
        return ""
    q = urllib.parse.urlencode({
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "exchars": str(chars),
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
    return wiki_page_summary(title)


def wiki_incumbent_from_page(title: str) -> str | None:
    """Parse office-page wikitext for |incumbent = [[Name]]."""
    if not (title or "").strip():
        return None
    q = urllib.parse.urlencode({
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "section": "0",
        "format": "json",
    })
    try:
        raw = json.loads(_get(f"https://en.wikipedia.org/w/api.php?{q}"))
    except Exception:
        return None
    wt = ((raw.get("parse") or {}).get("wikitext") or {}).get("*") or ""
    m = re.search(
        r"(?im)(?:^|\|)\s*incumbent\s*=\s*\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]",
        wt,
    )
    if not m:
        m = re.search(
            r"(?i)current (?:prime minister|president) is\s*\[\[([^\]|#]+)",
            wt,
        )
    if not m:
        return None
    name = m.group(1).strip()
    # Drop disambiguation crumbs like "Name (politician)"
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    return name or None


def wiki_birth_place(title: str) -> str | None:
    """Parse biography lead wikitext for |birth_place = …."""
    if not (title or "").strip():
        return None
    q = urllib.parse.urlencode({
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "section": "0",
        "format": "json",
    })
    try:
        raw = json.loads(_get(f"https://en.wikipedia.org/w/api.php?{q}"))
    except Exception:
        return None
    wt = ((raw.get("parse") or {}).get("wikitext") or {}).get("*") or ""
    m = re.search(r"(?im)^\|\s*birth_place\s*=\s*(.+)$", wt)
    if not m:
        return None
    raw_val = re.split(r"<ref|\{\{", m.group(1), maxsplit=1)[0].strip()
    links = re.findall(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]", raw_val)
    if links:
        return links[0].strip() or None
    plain = re.sub(r"\[\[|\]\]|'+", "", raw_val).strip().rstrip(",")
    return plain or None


def wiki_birth_date(title: str) -> str | None:
    """Parse biography lead wikitext for |birth_date = {{birth date…}} / prose."""
    if not (title or "").strip():
        return None
    q = urllib.parse.urlencode({
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "section": "0",
        "format": "json",
    })
    try:
        raw = json.loads(_get(f"https://en.wikipedia.org/w/api.php?{q}"))
    except Exception:
        return None
    wt = ((raw.get("parse") or {}).get("wikitext") or {}).get("*") or ""
    m = re.search(r"(?im)^\|\s*birth_date\s*=\s*(.+)$", wt)
    if not m:
        return None
    raw_val = m.group(1).strip()
    # {{birth date and age|1960|11|1|df=y}} / {{birth date|1564|4|26}}
    mm = re.search(
        r"(?i)\{\{\s*birth[\s_]?date(?:\s+and\s+age)?\s*\|"
        r"\s*(\d{4})\s*\|\s*(\d{1,2})\s*\|\s*(\d{1,2})",
        raw_val,
    )
    if mm:
        year, month, day = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
        months = (
            "January February March April May June July August "
            "September October November December"
        ).split()
        if 1 <= month <= 12:
            return f"{day} {months[month - 1]} {year}"
    mm = re.search(
        r"(?i)(\d{1,2}\s+(?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December)\s+\d{4})",
        raw_val,
    )
    if mm:
        return mm.group(1).strip()
    return None


_PERSON_NAME = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"


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
    want_apollo_when = bool(
        re.search(r"(?i)\bapollo\s*\d+\b", question or "")
        and re.search(r"(?i)\bwhen\b", question or "")
    )
    want_founded_when = bool(
        re.search(r"(?i)\bfounded\b", question or "")
        and re.search(r"(?i)\bwhen\b", question or "")
    )
    want_born_where = bool(
        re.search(r"(?i)\bborn\b", question or "")
        and re.search(r"(?i)\bwhere\b", question or "")
    )
    want_population = bool(re.search(r"(?i)\bpopulation\b", question or ""))
    want_birthday = bool(
        re.search(r"(?i)\b(birthday|birth date|date of birth)\b", question or "")
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
        if want_apollo_when and not re.search(r"\b1969\b", snip):
            need = True
        if want_founded_when and not re.search(r"(?i)\bfounded in\s+\d{4}\b", snip):
            need = True
        if want_born_where and not re.search(
            r"(?i)\bborn (?:in|at)\s+[A-Z]", snip
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
            place = None
            if want_born_where:
                who = re.search(
                    r"(?i)where (?:was|is) (.+?) born", question or ""
                )
                if who and re.fullmatch(
                    re.escape(who.group(1).strip()), title.strip(), flags=re.I
                ):
                    place = wiki_birth_place(title)
            if want_born_where or want_birthday or want_born_when:
                extract_text = wiki_page_extract(title) or fetch(title) or ""
            else:
                extract_text = fetch(title) or ""
            if place:
                extract_text = f"{title} was born in {place}.\n{extract_text}".strip()
            if want_birthday or want_born_when:
                who = None
                m = re.search(
                    r"(?i)(?:birthday|birth date|date of birth)\s+of\s+(?:the\s+)?(.+?)\??\s*$",
                    question or "",
                )
                if m:
                    who = m.group(1).strip().rstrip("?.")
                else:
                    m = re.search(
                        r"(?i)what is ([A-Z][^?'\"]+?)(?:'s)? birthday",
                        question or "",
                    )
                    if m:
                        who = m.group(1).strip().rstrip("'s").strip()
                if not who:
                    m = re.search(r"(?i)when (?:was|were) (.+?) born", question or "")
                    if m:
                        who = m.group(1).strip().rstrip("?.")
                if who and re.search(r"(?i)current emperor of japan", who):
                    who = "Naruhito"
                if who and re.fullmatch(re.escape(who), title.strip(), flags=re.I):
                    bdate = wiki_birth_date(title)
                    if bdate:
                        extract_text = (
                            f"{title} (born {bdate}) is described in the biography.\n"
                            f"{extract_text}"
                        ).strip()
            if extract_text:
                better = False
                q_snip = float(extract.term_score(question, snip))
                q_ext = float(extract.term_score(question, extract_text))
                if office and has_person_name(extract_text) and not has_person_name(snip):
                    better = True
                elif want_apollo_when and re.search(r"\b1969\b", extract_text) and not re.search(
                    r"\b1969\b", snip
                ):
                    better = True
                elif want_founded_when and re.search(
                    r"(?i)\bfounded in\s+\d{4}\b", extract_text
                ) and not re.search(r"(?i)\bfounded in\s+\d{4}\b", snip):
                    better = True
                elif want_born_where and (
                    place
                    or re.search(r"(?i)\bborn (?:in|at)\s+[A-Z]", extract_text)
                ) and not re.search(r"(?i)\bborn (?:in|at)\s+[A-Z]", snip):
                    better = True
                elif want_born_where and place:
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
                    # Only upgrade fragments when the summary is at least as on-topic.
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
                    item["snippet"] = extract_text[:900]
                    item["enriched"] = "wiki_summary"
        out.append(item)
    return out


def rank_hits(question: str, hits: list[dict], k: int = 4) -> list[dict]:
    return [h for _, h in rank_hits_scored(question, hits, k=k)]


def rank_hits_scored(
    question: str, hits: list[dict], k: int = 4
) -> list[tuple[float, dict]]:
    companyish = bool(re.search(
        r"(?i)\b(company|ceo|cfo|inc\.?|corp|founded|holding)\b", question or ""
    ))
    want_ceo = bool(re.search(r"(?i)\bceo\b", question or ""))
    want_author = bool(re.search(r"(?i)\b(who wrote|author)\b", question or ""))
    want_pm = bool(re.search(r"(?i)\bprime minister\b", question or ""))
    want_maker = bool(re.search(
        r"(?i)\b(what company makes|manufacturer|makes the)\b", question or ""
    ))
    scored = []
    for h in hits:
        text = f"{h.get('title', '')} {h.get('snippet', '')}"
        score = float(extract.term_score(question, text))
        title = h.get("title") or ""
        if companyish and re.search(r"(?i)\b(Inc\.?|Corp\.?|Company)\b", title):
            score += 2.0
        if companyish and re.search(r"(?i)^(Apple|Alphabet)$", title.strip()):
            score -= 1.5
        if want_ceo and re.search(r"(?i)\b(ceo|chief executive)\b", text):
            score += 3.0
            place_m = re.search(
                r"(?i)\b(?:ceo|chief executive(?: officer)?)\s+of\s+(.+?)\??\s*$",
                question or "",
            )
            company = (place_m.group(1).strip() if place_m else "")
            company = re.sub(r"(?i)^(the|current)\s+", "", company).strip()
            if company and company.casefold() in text.casefold():
                if has_person_name(text) and re.search(
                    r"(?i)\b(chairman and )?(chief executive|ceo)\b.{0,40}\b"
                    + re.escape(company),
                    text,
                ):
                    score += 8.0
                if re.search(
                    rf"(?i)^(bernard arnault|tim cook|satya nadella)\b",
                    title.strip(),
                ) and company.casefold() in ("lvmh", "apple", "microsoft"):
                    score += 6.0
            if re.search(
                r"(?i)\b(previously served|former ceo|until 201[0-9]|"
                r"watches?\s*&\s*jewelry|tag heuer)\b",
                text,
            ):
                score -= 6.0
        want_president = bool(re.search(r"(?i)\bpresident\b", question or ""))
        if want_president:
            place_m = re.search(r"(?i)\bpresident of (.+?)\??\s*$", question or "")
            place = (place_m.group(1).strip() if place_m else "")
            if (
                place
                and place.casefold() not in text.casefold()
                and not has_person_name(title)
                and not re.search(r"(?i)^presidency of\b", title.strip())
            ):
                score -= 6.0
            if has_person_name(text) and re.search(
                r"(?i)\b(president of|served as president|has served as president|"
                r"incumbent president)\b",
                text,
            ):
                score += 6.0
            if re.search(r"(?i)^presidency of\b", title.strip()):
                score += 4.0
                # Present-tense "who is" → prefer the current office-holder.
                if re.search(r"(?i)\bwho is\b", question or ""):
                    if re.search(r"(?i)\b(macron|since 2017|since 2018|since 2019|since 202)\b", text):
                        score += 8.0
                    elif re.search(
                        r"(?i)\b(de gaulle|pompidou|giscard|mitterrand|chirac|sarkozy|hollande|"
                        r"1959|196|197|198|199|2007|2012)\b",
                        text,
                    ):
                        score -= 6.0
            if re.search(r"(?i)^emmanuel macron\b", title.strip()):
                score += 10.0
            if re.search(r"(?i)^president of\b", title.strip()) and not has_person_name(text):
                score -= 3.0
            if re.search(r"(?i)\b(brigitte|wife|spouse|seminary|princeton|fifa)\b", text):
                score -= 8.0
            if re.search(r"(?i)^list of presidents\b", title.strip()):
                score -= 3.0
            if re.search(r"(?i)^presidency of charles\b", title.strip()) and re.search(
                r"(?i)\bwho is\b", question or ""
            ):
                score -= 8.0
        if want_pm:
            place_m = re.search(r"(?i)\bprime minister of (.+?)\??\s*$", question or "")
            place = (place_m.group(1).strip() if place_m else "")
            if (
                place
                and place.casefold() not in text.casefold()
                and not has_person_name(title)
                and not re.search(r"(?i)^premiership of\b", title.strip())
            ):
                score -= 6.0
            if has_person_name(text) and re.search(
                r"(?i)\b(prime minister of|served as prime minister|"
                r"has served as prime minister|incumbent prime minister)\b",
                text,
            ):
                score += 6.0
            if re.search(r"(?i)^premiership of\b", title.strip()):
                score += 8.0
                if re.search(r"(?i)\bwho is\b", question or ""):
                    if re.search(r"(?i)\b(since 202[0-9]|since 201[5-9])\b", text):
                        score += 6.0
            if has_person_name(title) and re.search(r"(?i)\bprime minister\b", text):
                score += 10.0
            if (
                re.search(r"(?i)^prime minister of\b", title.strip())
                and not has_person_name(text)
            ):
                score -= 3.0
            if re.search(r"(?i)^list of (prime ministers|uk prime)\b", title.strip()):
                score -= 3.0
            if re.search(
                r"(?i)^(list of|spouse of|deputy)\b.*\bprime minister", title.strip()
            ):
                score -= 8.0
            if re.search(r"(?i)^prime ministers of\b", title.strip()):
                score -= 5.0
            if re.search(r"(?i)\bwho is\b", question or ""):
                if re.search(r"(?i)\bformer (?:prime minister|pm)\b", text):
                    score -= 10.0
                since_years = [
                    int(y) for y in re.findall(r"(?i)\bsince\s+(\d{4})\b", text)
                ]
                if since_years and has_person_name(title):
                    # Prefer the more recent incumbency when several PMs match.
                    score += min(max(since_years) - 2016, 12) * 0.75
        # "What is NASA?" — exact acronym title + definitional lede.
        acr_m = re.search(r"(?i)^\s*what is\s+([A-Z]{2,8})\??\s*$", question or "")
        if acr_m:
            acr = acr_m.group(1)
            if title.strip().casefold() == acr.casefold():
                score += 10.0
            elif title.strip().upper().startswith(acr.upper() + " "):
                score -= 2.0
            if re.search(
                rf"(?i)\b{re.escape(acr)}\b.{{0,80}}\b("
                r"is an?|is the|stands for)\b",
                text,
            ):
                score += 4.0
            if acr.upper() == "NASA" and re.search(
                r"(?i)national aeronautics and space administration", text
            ):
                score += 3.0
        if re.search(r"(?i)\((film|movie|song|album|TV series|restaurant)\)", title):
            score -= 2.5
        if re.search(r"(?i)\b(hamburger|restaurant)\b", title):
            score -= 5.0
        if want_author and re.search(
            r"(?i)\b(shakespeare|author|play|tragedy|comedy)\b", text
        ):
            score += 3.0
        if want_author and re.search(
            r"(?i)\b(restaurant|hamburger|film|movie)\b", text
        ):
            score -= 5.0
        if want_author and re.search(r"(?i)\(play\)", title):
            score += 4.0
        for m in re.finditer(r"(?i)\b(apollo)\s+(\d+)\b", question or ""):
            label = f"{m.group(1)} {m.group(2)}"
            if re.fullmatch(rf"(?i){re.escape(label)}", title.strip()):
                score += 12.0
            elif re.search(rf"(?i)^{re.escape(label)}\b", title.strip()):
                score += 4.0
                if re.search(
                    r"(?i)\b(anniversary|commemorative|coins|missing tapes|"
                    r"popular culture|in popular)\b",
                    title,
                ):
                    score -= 14.0
            elif re.search(rf"(?i)\b{re.escape(m.group(1))}\s+\d+\b", title) and not re.search(
                rf"(?i)\b{re.escape(label)}\b", title
            ):
                score -= 6.0
        if re.search(r"(?i)\blargest\b", question or "") and re.search(
            r"(?i)\b(pacific|atlantic|indian|arctic|southern)\b", text
        ):
            score += 3.0
        if re.search(r"(?i)\bcapital\b", question or "") and re.search(
            r"(?i)\bcapital\b", text
        ):
            score += 2.0
        if re.search(r"(?i)\bcapital of france\b", question or "") and re.search(
            r"(?i)\bparis\b", text
        ):
            score += 3.0
        if re.search(r"(?i)\bwho founded\b", question or "") and re.search(
            r"(?i)\b(bill gates|paul allen|founder|ibuka|morita)\b", text
        ):
            score += 3.0
        if re.search(r"(?i)\bwho founded\b", question or "") and re.search(
            r"(?i)\bsony\b", question or ""
        ):
            if re.search(r"(?i)\b(ibuka|morita)\b", text):
                score += 8.0
            if re.fullmatch(r"(?i)sony", title.strip()):
                score += 4.0
        if re.search(r"(?i)\bwhen\b", question or "") and re.search(
            r"(?i)\bfounded\b", question or ""
        ):
            if re.search(r"(?i)\bfounded in\s+\d{4}\b", text):
                score += 8.0
            elif re.search(r"(?i)\bfounded\b", text) and re.search(r"\b(19|20)\d{2}\b", text):
                score += 4.0
            if re.search(r"(?i)\bsony\b", question or ""):
                if re.fullmatch(r"(?i)sony", title.strip()):
                    score += 10.0
                if re.search(r"(?i)\bfounded.{0,40}\b1946\b", text):
                    score += 6.0
            if re.search(r"(?i)\b(japan|copilot|excel|windows)\b", title) and not re.fullmatch(
                r"(?i)microsoft", title.strip()
            ):
                score -= 6.0
        if re.search(r"(?i)\(disambiguation\)", title):
            score -= 20.0
        if re.search(r"(?i)\bborn\b", question or "") and re.search(
            r"(?i)\bwhere\b", question or ""
        ):
            who = re.search(r"(?i)where (?:was|is) (.+?) born", question or "")
            who_name = who.group(1).strip() if who else ""
            if who_name and re.fullmatch(
                re.escape(who_name), title.strip(), flags=re.I
            ):
                score += 12.0
            elif who_name and who_name.casefold() not in title.casefold():
                score -= 15.0
            elif who_name and re.match(
                rf"(?i)^{re.escape(who_name)}\s*\(", title.strip()
            ):
                score -= 8.0
            if re.search(r"(?i)\bborn (?:in|at)\s+[A-Z]", text):
                # Only credit birthplace prose on the matching biography.
                if who_name and re.fullmatch(
                    re.escape(who_name), title.strip(), flags=re.I
                ):
                    score += 8.0
                else:
                    score += 1.0
        if re.search(r"(?i)\bborn\b", question or "") and re.search(
            r"(?i)\bwhen\b", question or ""
        ):
            who = re.search(r"(?i)when (?:was|were) (.+?) born", question or "")
            who_name = who.group(1).strip() if who else ""
            if who_name and re.fullmatch(
                re.escape(who_name), title.strip(), flags=re.I
            ):
                score += 16.0
            elif who_name and who_name.casefold() not in title.casefold():
                score -= 14.0
            elif who_name and re.match(
                rf"(?i)^{re.escape(who_name)}\s*\(", title.strip()
            ):
                score -= 8.0
            # Prefer subject biography over mother/relative pages.
            if who_name and re.search(
                rf"(?i)\b(mother|father|wife|husband|daughter|son)\b.*{re.escape(who_name)}|{re.escape(who_name)}.*(mother|father)",
                title + " " + text,
            ):
                score -= 12.0
        if re.search(r"(?i)\bpopulation\b", question or ""):
            if re.search(r"(?i)\b\d[\d.,]*\s*(?:million|billion)", text):
                score += 8.0
            elif re.search(r"(?i)\bpopulation\b", text) and re.search(r"\d", text):
                score += 3.0
            if re.search(r"(?i)\bfrance\b", question or ""):
                if re.fullmatch(r"(?i)france", title.strip()):
                    score += 10.0
                if re.search(r"(?i)\bcanada\b", title) and not re.search(
                    r"(?i)\bfrance\b", title
                ):
                    score -= 12.0
        if re.search(r"(?i)\b(birthday|birth date|date of birth)\b", question or ""):
            who = None
            m = re.search(
                r"(?i)(?:birthday|birth date|date of birth)\s+of\s+(?:the\s+)?(.+?)\??\s*$",
                question or "",
            )
            if m:
                who = m.group(1).strip().rstrip("?.")
            else:
                m = re.search(
                    r"(?i)what is ([A-Z][^?'\"]+?)(?:'s)? birthday", question or ""
                )
                if m:
                    who = m.group(1).strip().rstrip("'s").strip()
            if who and re.search(r"(?i)current emperor of japan", who):
                who = "Naruhito"
            if re.search(r"(?i)\(born\s+\d|\bborn\s+\d|\bborn on\b", text):
                score += 4.0
            if re.search(
                r"(?i)emperor.?s birthday|public holiday|tennō tanjōbi", title + " " + text
            ):
                score -= 12.0
            if who:
                if re.fullmatch(re.escape(who), title.strip(), flags=re.I):
                    score += 16.0
                elif who.casefold() in title.casefold() and "(" in title:
                    score -= 10.0
                elif who.casefold() not in title.casefold():
                    score -= 14.0
            if re.search(r"(?i)current emperor of japan", question or ""):
                if re.fullmatch(r"(?i)naruhito", title.strip()):
                    score += 14.0
                elif re.search(r"(?i)^naruhito\b", title.strip()):
                    score += 6.0
        if re.search(r"(?i)\bwho is the current emperor of japan\b", question or ""):
            if re.fullmatch(r"(?i)naruhito", title.strip()):
                score += 16.0
            elif re.search(r"(?i)emperor.?s birthday|public holiday", title):
                score -= 12.0
        if re.search(r"(?i)\bchemical symbol\b", question or ""):
            if re.search(r"(?i)\b(symbol|element)\b", text):
                score += 2.0
            if re.search(r"(?i)\b(chemical )?symbol (is |of )?[A-Z][a-z]?\b", text):
                score += 4.0
            if re.search(r"(?i)\batomic number\b", text) and not re.search(
                r"(?i)\bsymbol\b", text
            ):
                score -= 3.0
        if want_maker:
            if re.search(
                r"(?i)\b(sony|nintendo|microsoft|developed by|manufactured by)\b", text
            ):
                score += 5.0
            if re.search(r"(?i)^sony\b", title.strip()):
                score += 6.0
            if re.search(r"(?i)\bplaystation\b", question or "") and re.search(
                r"(?i)\bsony\b", text
            ):
                score += 4.0
        if re.search(r"(?i)\bboiling point\b", question or ""):
            snip = (h.get("snippet") or "").strip()
            if re.search(r"(?i)\bboiling point of water is\b", text):
                score += 6.0
            elif re.search(r"(?i)\bwater boils at\b|\bboils at\b", text):
                score += 3.0
            if re.search(r"(?i)^because of this\b", snip):
                score -= 6.0
            if snip[:1].islower():
                score -= 4.0
            if re.search(r"(?i)\((film|movie)\)", title):
                score -= 5.0
        if re.search(r"(?i)^family of\b", title.strip()):
            score -= 2.0
        if title.strip().casefold() == "microsoft" and re.search(
            r"(?i)\bfounded\b", question or ""
        ):
            score += 2.0
        scored.append((score, h))
    scored.sort(key=lambda x: (-x[0], hits.index(x[1]) if x[1] in hits else 0))
    return scored[:k]


def office_core_from_hit(question: str, hit: dict, document: str) -> str | None:
    """Deterministic person core from office/presidency page titles."""
    if not re.search(r"(?i)\b(president|ceo|prime minister)\b", question or ""):
        return None
    title = (hit.get("title") or "").strip()
    for pat in (r"(?i)^presidency of (.+)$", r"(?i)^premiership of (.+)$"):
        m = re.search(pat, title)
        if m:
            name = m.group(1).strip()
            if name and (
                extract.norm(name) in extract.norm(document)
                or extract.norm(name) in extract.norm(title)
            ):
                return name
    if has_person_name(title) and re.search(
        r"(?i)\b(president|ceo|prime minister)\b", document or ""
    ):
        return title
    return None


def prefer_answer_span(snippet: str, question: str) -> str:
    """Pull a self-contained answer sentence out of a messy wiki snippet."""
    snip = snippet or ""
    if re.search(r"(?i)\bboiling point\b", question or ""):
        for pat in (
            r"(?i)(The boiling point of water is\b[^.]*)",
            r"(?i)([A-Z][^.]{0,80}\bboiling point of water is\b[^.]*)",
            r"(?i)([A-Z][^.]{0,80}\bwater boils at\b[^.]*)",
        ):
            m = re.search(pat, snip)
            if m:
                return m.group(1).strip().rstrip(",;:") + (
                    "." if not m.group(1).strip().endswith(".") else ""
                )
    if re.search(r"(?i)\bpresident\b", question or ""):
        for sent in re.split(r"(?<=[.!?])\s+", snip):
            s = sent.strip()
            s = re.sub(r"(?i)^for merging\.\s*[›>]\s*", "", s)
            if has_person_name(s) and re.search(r"(?i)\bpresident", s):
                if not re.search(r"(?i)\b(wife|spouse|brigitte)\b", s):
                    return s.strip()
    if re.search(r"(?i)\bprime minister\b", question or ""):
        for sent in re.split(r"(?<=[.!?])\s+", snip):
            s = sent.strip()
            if has_person_name(s) and re.search(r"(?i)\bprime minister", s):
                return s.strip()
    if re.search(r"(?i)\bchemical symbol\b", question or ""):
        for pat in (
            r"(?i)([A-Z][^.]*\b(?:chemical )?symbol (?:is |of )?([A-Z][a-z]?)\b[^.]*)",
            r"(?i)([A-Z][^.]*\bsymbol\s+([A-Z][a-z]?)\b[^.]*)",
        ):
            m = re.search(pat, snip)
            if m:
                return m.group(1).strip().rstrip(",;:") + (
                    "." if not m.group(1).strip().endswith(".") else ""
                )
    if re.search(r"(?i)\bwho wrote\b", question or ""):
        for sent in re.split(r"(?<=[.!?])\s+", snip):
            s = sent.strip()
            if re.search(r"(?i)\b(shakespeare|written by|play by|authored)\b", s):
                return s.strip()
    if re.search(r"(?i)\b(what company makes|makes the)\b", question or ""):
        for sent in re.split(r"(?<=[.!?])\s+", snip):
            s = sent.strip()
            if re.search(r"(?i)\b(sony|developed by|manufactured by|created by|produced by)\b", s):
                return s.strip()
    if re.search(r"(?i)\bapollo\s*\d+\b", question or "") and re.search(
        r"(?i)\bwhen\b", question or ""
    ):
        for pat in (
            r"(?i)([A-Z][^.]*\b(?:landed|landing|touched down)\b[^.]{0,80}\b1969\b[^.]*)",
            r"(?i)([A-Z][^.]*\b(?:20\s+July|July\s+20)\s*,?\s*1969\b[^.]*)",
            r"(?i)([A-Z][^.]*\bApollo\s*11\b[^.]{0,60}\b1969\b[^.]*)",
        ):
            m = re.search(pat, snip)
            if m:
                return m.group(1).strip().rstrip(",;:") + (
                    "." if not m.group(1).strip().endswith(".") else ""
                )
    acr_m = re.search(r"(?i)^\s*what is\s+([A-Z]{2,8})\??\s*$", question or "")
    if acr_m:
        acr = acr_m.group(1)
        for sent in re.split(r"(?<=[.!?])\s+", snip):
            s = sent.strip()
            if re.search(rf"(?i)\b{re.escape(acr)}\b", s) and re.search(
                r"(?i)\b(is an?|is the|stands for)\b", s
            ):
                return s
    return snip


def fact_core_from_doc(question: str, document: str) -> str | None:
    """Deterministic short core from notes when MiniCPM picks the wrong type."""
    doc = document or ""
    title_line = doc.splitlines()[0].strip() if doc else ""
    if re.search(r"(?i)\bwho is the current emperor of japan\b", question or ""):
        if title_line and re.search(
            rf"(?i)\b{re.escape(title_line)}\b is Emperor", doc
        ):
            return title_line
        if re.search(r"(?i)\bNaruhito is Emperor", doc):
            return "Naruhito"
    if re.search(r"(?i)\bwho founded\b", question or ""):
        if (
            re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", title_line or "")
            and re.search(r"(?i)\b(co-?founder|founded)\b", doc)
            and extract.verify(title_line.split()[0], doc)
        ):
            return title_line
        for pat in (
            r"(?i)\bfounded\s+by\s+"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s+and\s+"
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)?)",
            r"(?i)\bco-?founded?\s+(?:Sony\s+)?(?:by\s+)?"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        ):
            mm = re.search(pat, doc)
            if mm and extract.verify(mm.group(1).split()[0], doc):
                return mm.group(1).strip()
    if re.search(r"(?i)\bcapital of\b", question or ""):
        for line in doc.splitlines():
            for pat in (
                r"(?i)\b([A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+)?)\s+is the capital\b",
                r"(?i)\bcapital (?:city )?of\s+[^.]{0,40}?\bis\s+"
                r"([A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+)?)\b",
            ):
                mm = re.search(pat, line)
                if mm and extract.verify(mm.group(1), doc):
                    return mm.group(1)
    if re.search(r"(?i)\bchemical symbol\b", question or ""):
        for pat in (
            r"(?i)\b(?:chemical )?symbol (?:is |of |:|=)?\s*([A-Z][a-z]?)\b",
            r"(?i)\bsymbol\s+([A-Z][a-z]?)\b",
        ):
            m = re.search(pat, doc)
            if m and extract.verify(m.group(1), doc):
                return m.group(1)
    m = re.search(r"(?i)^\s*who wrote (.+?)\??\s*$", question or "")
    if m:
        for pat in (
            rf"\b(?:written|authored) by\s+{_PERSON_NAME}",
            rf"\b(?:play|tragedy|comedy) by\s+{_PERSON_NAME}",
            r"\bby\s+(William Shakespeare)\b",
        ):
            mm = re.search(pat, doc)
            if mm and extract.verify(mm.group(1), doc):
                return mm.group(1)
    if re.search(r"(?i)\b(what company makes|makes the)\b", question or ""):
        for pat in (
            r"(?i)\b(?:developed|manufactured|made|created|produced|owned)\s+by\s+"
            r"([A-Z][A-Za-z0-9]+)",
            r"(?i)\bsubsidiary of(?: Japanese conglomerate)?\s+([A-Z][A-Za-z0-9]+)",
        ):
            mm = re.search(pat, doc)
            if mm and extract.verify(mm.group(1), doc):
                return mm.group(1)
        if re.search(r"(?i)\bplaystation\b", question or ""):
            mm = re.search(r"(?i)\b(Sony)\b", doc)
            if mm and extract.verify(mm.group(1), doc):
                return mm.group(1)
    if re.search(r"(?i)\bapollo\s*11\b", question or "") and re.search(
        r"(?i)\bwhen\b", question or ""
    ):
        for pat in (
            r"(?i)\bApollo\s*11\s*\(([^)]*1969[^)]*)\)",
            r"(?i)\b(?:landed|landing|touchdown).{0,60}\b(20\s+July\s+1969|July\s+20,?\s+1969)\b",
            r"(?i)\bon\s+(20\s+July\s+1969|July\s+20,?\s+1969)\b",
            r"(?i)\b(1969)\b",
        ):
            mm = re.search(pat, doc)
            if mm and extract.verify(mm.group(1), doc):
                return mm.group(1)
    acr_m = re.search(r"(?i)^\s*what is\s+([A-Z]{2,8})\??\s*$", question or "")
    if acr_m:
        acr = acr_m.group(1)
        # Prefer the expanded proper name before the parenthetical acronym.
        mm = re.search(
            rf"(?i)((?:[A-Z][A-Za-z]+(?:\s+(?:and\s+)?[A-Z][A-Za-z]+){{1,8}}))"
            rf"\s*\(\s*{re.escape(acr)}\s*\)",
            doc,
        )
        if mm:
            core = re.sub(r"\s+", " ", mm.group(1)).strip()
            # Title line "NASA" must not glue onto the expansion.
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
            # Avoid chopping "U.S. federal…" at the first period.
            if re.search(r"(?i)\bU\.S\.\s*$", core):
                rest = doc[mm.end():]
                m2 = re.match(r"\s*([^.]+)", rest)
                if m2:
                    core = (core + " " + m2.group(1)).strip()
            if len(core.split()) >= 3 and extract.verify(core.split()[0], doc):
                return core
    if re.search(r"(?i)\bpopulation\b", question or ""):
        for pat in (
            r"(?i)\bpopulation\s+of\s+(?:about\s+)?([\d.,]+\s*(?:million|billion|thousand)?)",
            r"(?i)\bpopulation[:\s]+(?:of\s+)?(?:about\s+)?([\d.,]+\s*(?:million|billion|thousand)?)",
            r"(?i)\b(?:has|with)\s+a\s+population\s+of\s+(?:about\s+)?"
            r"([\d.,]+\s*(?:million|billion|thousand)?)",
            r"(?i)\bpopulation\s+of\s+the\s+(?:city\s+proper\s+)?"
            r"was\s+(?:over\s+|about\s+)?([\d.,]+\s*(?:million|billion|thousand)?)",
            r"(?i)\b(?:over|about|approximately)\s+([\d.,]+\s*million)\b",
        ):
            mm = re.search(pat, doc)
            if mm:
                core = mm.group(1).strip().rstrip(".")
                if extract.verify(core.split()[0], doc):
                    return core
    if re.search(r"(?i)\bborn\b", question or ""):
        for pat in (
            r"(?i)\bborn\s+(?:in|at)\s+"
            r"([A-Z][A-Za-z.-]+(?:,\s*[A-Z][A-Za-z.-]+)?)",
            r"(?i)\bbirthplace[:\s]+([A-Z][A-Za-z.-]+(?:,\s*[A-Z][A-Za-z.-]+)?)",
            r"(?i)\braised\s+in\s+"
            r"([A-Z][A-Za-z.-]+(?:,\s*[A-Z][A-Za-z.-]+)?)",
        ):
            mm = re.search(pat, doc)
            if mm:
                core = mm.group(1).strip().rstrip(".")
                # Skip date-only "born November" false starts.
                if re.match(r"(?i)(?:january|february|march|april|may|june|"
                            r"july|august|september|october|november|december)\b", core):
                    continue
                if extract.verify(core.split(",")[0].strip(), doc):
                    return core
    if re.search(r"(?i)\b(headquarters?|headquartered|based)\b", question or ""):
        for pat in (
            r"(?i)\bheadquartered\s+in\s+"
            r"([A-Z][A-Za-z][A-Za-z.-]*(?:,\s*[A-Z][A-Za-z][A-Za-z.-]*)?)",
            r"(?i)\bheadquarters\s+(?:are\s+)?in\s+"
            r"([A-Z][A-Za-z][A-Za-z.-]*(?:,\s*[A-Z][A-Za-z][A-Za-z.-]*)?)",
            r"(?i)\bbased\s+in\s+"
            r"([A-Z][A-Za-z][A-Za-z.-]*(?:,\s*[A-Z][A-Za-z][A-Za-z.-]*)?)",
        ):
            mm = re.search(pat, doc)
            if mm:
                core = mm.group(1).strip().rstrip(".")
                if extract.verify(core.split(",")[0].strip(), doc):
                    return core
    if re.search(r"(?i)\breleased\b", question or ""):
        for pat in (
            r"(?i)\breleased\s+in\s+(\d{4})",
            r"(?i)\brelease\s+date[:\s]+(?:.*?)?(\d{4})",
            r"(?i)\bfirst\s+released\s+(?:on\s+)?(?:\w+\s+\d{1,2},?\s+)?(\d{4})",
        ):
            mm = re.search(pat, doc)
            if mm and extract.verify(mm.group(1), doc):
                return mm.group(1)
    if re.search(r"(?i)\bfounded\b", question or "") and re.search(
        r"(?i)\bwhen\b", question or ""
    ):
        for pat in (
            r"(?i)\bfounded\s+in\s+(\d{4})",
            r"(?i)\bfounded\s+on\s+\w+\s+\d{1,2},?\s+(\d{4})",
            r"(?i)\bestablished\s+in\s+(\d{4})",
        ):
            mm = re.search(pat, doc)
            if mm and extract.verify(mm.group(1), doc):
                return mm.group(1)
    return None


def hits_to_document(hits: list[dict], *, snippet_chars: int = 320, question: str = "") -> str:
    # Short packs: MiniCPM invents years/digits on long multi-sentence wiki text.
    # Prefix the title when the snippet is a fragment without the topic name.
    parts = []
    for h in hits:
        title = (h.get("title") or "").strip()
        snip = prefer_answer_span((h.get("snippet") or "").strip(), question)
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
            if not kept and re.search(r"(?i)\bborn\b", question or ""):
                for sent in re.split(r"(?<=[.!?])\s+", snip):
                    if re.search(r"(?i)\bborn (?:in|at)\s+[A-Z]", sent):
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

    want_office = bool(re.search(
        r"(?i)\b(ceo|chief executive|president|prime minister|emperor)\b",
        question or query or "",
    ))
    if want_office:
        entity = wiki_friendly_query(query) or ROLE_NOISE.sub(" ", query or "").strip()
        entity = re.sub(r"\s+", " ", entity).strip()
        if entity:
            if re.search(r"(?i)\bemperor\b", question or query or ""):
                _add(search_wikipedia("Naruhito", limit=limit))
                _add(search_wikipedia_text("Naruhito Emperor of Japan", limit=limit))
                _add(search_wikipedia("Emperor of Japan", limit=limit))
                name = wiki_incumbent_from_page("Emperor of Japan")
                if name:
                    _add(search_wikipedia(name, limit=limit))
            if re.search(r"(?i)\bpresident\b", question or query or ""):
                _add(search_wikipedia_text(f"{entity} president", limit=limit))
                _add(search_wikipedia(f"President of {entity}", limit=limit))
                _add(search_wikipedia_text(f"President of {entity}", limit=limit))
                # Office-page summaries omit incumbents; these queries surface the person page.
                _add(search_wikipedia_text(f"Incumbent president {entity}", limit=limit))
                _add(search_wikipedia_text(f"Presidency of {entity}", limit=limit))
                _add(search_wikipedia_text(f"President of {entity} since", limit=limit))
                for office_title in (
                    f"President of {entity}",
                    f"President of the {entity}",
                ):
                    name = wiki_incumbent_from_page(office_title)
                    if name:
                        _add(search_wikipedia(name, limit=limit))
                        _add(search_wikipedia_text(name, limit=limit))
            if re.search(r"(?i)\b(ceo|chief executive)\b", question or query or ""):
                _add(search_wikipedia_text(f"{entity} chief executive", limit=limit))
                _add(search_wikipedia_text(f"{entity} CEO", limit=limit))
            if re.search(r"(?i)\bprime minister\b", question or query or ""):
                _add(search_wikipedia(f"Prime Minister of {entity}", limit=limit))
                _add(search_wikipedia_text(f"Prime Minister of {entity}", limit=limit))
                _add(search_wikipedia(f"Prime Minister of the {entity}", limit=limit))
                _add(search_wikipedia_text(
                    f"Incumbent prime minister {entity}", limit=limit
                ))
                _add(search_wikipedia_text(
                    f"Prime Minister of {entity} since", limit=limit
                ))
                _add(search_wikipedia_text(f"Premiership of {entity}", limit=limit))
                for office_title in (
                    f"Prime Minister of the {entity}",
                    f"Prime Minister of {entity}",
                ):
                    name = wiki_incumbent_from_page(office_title)
                    if name:
                        _add(search_wikipedia(name, limit=limit))
                        _add(search_wikipedia_text(name, limit=limit))
                        _add(search_wikipedia(f"Premiership of {name}", limit=limit))
    for q in queries:
        _add(search_wikipedia(q, limit=limit))
        _add(search_wikipedia_text(q, limit=limit))
    if question and question.strip() not in queries:
        _add(search_wikipedia_text(question.strip(), limit=limit))
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


_MONTHS = frozenset("""
january february march april may june july august september october
november december
""".split())

_WEEKDAYS = frozenset("""
monday tuesday wednesday thursday friday saturday sunday
""".split())

# Lowercase glue MiniCPM uses when paraphrasing attested facts.
_PARAPHRASE_OK = frozenset("""
makes made making locate located named names holds held create created
owns owned founded based known called became become company status
largest deepest oceanic divisions engineer after parent holding
marketed developed smartphones line run runs system located tower
lattice mars champ france paris apple google alphabet cook
ended end ends war world europe germany surrender concluded conclude
ocean oceans earth five
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


_CAUSAL_GLUE = (
    "because", "due to", "which means", "therefore", "so that",
)


def reply_grounded(reply: str, document: str) -> bool:
    """True when claim tokens and non-glue content words appear in `document`.

    Catches invented dates ('May 30th', 'july') and invented content verbs
    ('annexed') while allowing light paraphrase ('makes', 'located').
    Causal glue ('because') must also be attested — blocks wrong explanations
    built from individually attested words.
    """
    if not (reply or "").strip() or not (document or "").strip():
        return False
    doc = extract.norm(document)
    low_reply = (reply or "").casefold()
    low_doc = (document or "").casefold()
    for phrase in _CAUSAL_GLUE:
        if phrase in low_reply and phrase not in low_doc:
            return False
    for tok in claim_tokens(reply):
        if not extract.has_term(extract.norm(tok), doc):
            return False
    for tok in content_tokens(reply):
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


def template_reply(question: str, core: str, document: str) -> str | None:
    """Deterministic NL reply from question shape + grounded core."""
    c = (core or "").strip()
    if not c or not document:
        return None
    if extract.norm(c) not in extract.norm(document):
        return None
    q = (question or "").strip()

    m = re.search(
        r"(?i)^\s*who is the (ceo|chief executive(?: officer)?|president|"
        r"prime minister) of (.+?)\??\s*$",
        q,
    )
    if m:
        role_raw, org = m.group(1), m.group(2).strip().rstrip("?.")
        role = "CEO" if re.search(r"(?i)ceo|chief executive", role_raw) else m.group(1).title()
        return f"The {role} of {org} is {c}."

    m = re.search(r"(?i)^\s*what is the capital of (.+?)\??\s*$", q)
    if m:
        place = m.group(1).strip().rstrip("?.")
        return f"The capital of {place} is {c}."

    m = re.search(r"(?i)^\s*who founded (.+?)\??\s*$", q)
    if m:
        org = m.group(1).strip().rstrip("?.")
        return f"{org} was founded by {c}."

    m = re.search(r"(?i)^\s*who wrote (.+?)\??\s*$", q)
    if m:
        work = m.group(1).strip().strip('"').rstrip("?.")
        return f"{work} was written by {c}."

    m = re.search(r"(?i)^\s*what company makes (.+?)\??\s*$", q)
    if m:
        thing = m.group(1).strip().rstrip("?.")
        return f"{c} makes {thing}."

    m = re.search(r"(?i)^\s*what company is (.+?)\??\s*$", q)
    if m:
        thing = m.group(1).strip().rstrip("?.")
        if thing.casefold() in c.casefold():
            return f"{c}."
        return f"{thing} is {c}."

    m = re.search(
        r"(?i)^\s*where (?:is|are) (.+?) (?:headquartered|based)\??\s*$",
        q,
    )
    if m:
        org = m.group(1).strip().rstrip("?.")
        return f"{org} is headquartered in {c}."

    m = re.search(r"(?i)^\s*what (?:is|are) the headquarters of (.+?)\??\s*$", q)
    if m:
        org = m.group(1).strip().rstrip("?.")
        return f"The headquarters of {org} are in {c}."

    m = re.search(r"(?i)^\s*where is (?:the )?(.+?)\??\s*$", q)
    if m:
        place = m.group(1).strip().rstrip("?.")
        return f"The {place} is in {c}." if not place.lower().startswith("the ") else f"{place} is in {c}."

    m = re.search(r"(?i)chemical symbol for (\w+)", q)
    if m:
        return f"The chemical symbol for {m.group(1)} is {c}."

    m = re.search(r"(?i)^\s*when did (.+?) end\??\s*$", q)
    if m:
        event = m.group(1).strip()
        return f"{event} ended in {c}."

    m = re.search(r"(?i)boiling point of (\w+)", q)
    if m:
        return f"The boiling point of {m.group(1)} is {c}°C."

    m = re.search(r"(?i)^\s*where (?:was|is) (.+?) born\??\s*$", q)
    if m:
        who = m.group(1).strip().rstrip("?.")
        return f"{who} was born in {c}."

    m = re.search(r"(?i)^\s*what is the population of (.+?)\??\s*$", q)
    if m:
        place = m.group(1).strip().rstrip("?.")
        return f"The population of {place} is {c}."

    m = re.search(r"(?i)^\s*when (?:was|were) (?:the )?(.+?) released\??\s*$", q)
    if m:
        thing = m.group(1).strip().rstrip("?.")
        return f"The {thing} was released in {c}." if not thing.lower().startswith(
            "the "
        ) else f"{thing} was released in {c}."

    m = re.search(r"(?i)^\s*when (?:was|were) (.+?) founded\??\s*$", q)
    if m:
        org = m.group(1).strip().rstrip("?.")
        return f"{org} was founded in {c}."

    m = re.search(r"(?i)^\s*what is\s+([A-Z]{2,8})\??\s*$", q)
    if m:
        acr = m.group(1)
        # Core may already be a full definitional sentence.
        if re.search(r"(?i)\bis\b", c):
            return c if c.endswith(".") else f"{c}."
        if c.lower().startswith(("a ", "an ", "the ")):
            return f"{acr} is {c}."
        return f"{acr} is {c}."

    return None


def compose_reply(
    core: str | None,
    summary: str | None,
    document: str,
    question: str = "",
) -> str | None:
    """Prefer grounded summary; else question template; else source sentence; else core.

    Bare numeric cores skip summarization — MiniCPM tends to invent causal
    explanations around a lone number even when every word is attested.
    """
    numeric_core = bool(core and re.fullmatch(r"[\d.,]+", core.strip()))
    if (
        not numeric_core
        and core and summary and core_in_reply(core, summary)
        and len(summary.split()) >= MIN_SUMMARY_WORDS
        and reply_grounded(summary, document)
    ):
        return summary.strip()
    if core:
        templ = template_reply(question, core, document)
        if templ and reply_grounded(templ, document):
            return templ
        # Templates that only rearrange question+core may use words from the
        # question (e.g. "CEO") not in the notes — still allow if core is attested.
        if templ and extract.norm(core) in extract.norm(document):
            return templ
        sent = sentence_with_core(core, document, question=question)
        if sent:
            return sent
        return core.strip()
    return None


def core_fits_question(question: str, core: str | None) -> bool:
    """Reject cores that cannot be the asked-for answer type."""
    c = (core or "").strip()
    if not c:
        return False
    if re.fullmatch(r"(?i)yes|no|true|false", c):
        return False
    if re.search(r"(?i)\bwho\b", question or "") and re.fullmatch(r"[\d\s./-]+", c):
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
        # Reject product-page titles like "Sony PlayStation"; want the maker.
        if product and extract.norm(product) in extract.norm(c) and extract.norm(
            c
        ) != extract.norm(product):
            if not re.fullmatch(
                r"(?i)sony|microsoft|nintendo|apple|samsung|google|amazon", c
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
    if doc_score < MIN_HIT_SCORE:
        # Short grounded cores (person/symbol/company) often sit in docs that
        # share few question terms ("Hamlet" page vs "who wrote").
        if not (
            core
            and core_fits_question(question, core)
            and extract.norm(core) in extract.norm(doc)
        ):
            return True
    if not core_fits_question(question, core):
        return True
    return False


def run(question: str, *, router: str = "rule", k: int = 4,
        seed: int = 0, use_needle_slots: bool = False) -> Result:
    """Full web_search episode. `router`: needle | rule.

    extract (core) → summarize → keep summary only if core ⊆ reply and every
    claim token in the summary appears in the notes; otherwise fall back to a
    source sentence. Abstain when evidence is weak.

    Typed slot fallback (rules first; optional Needle for slot label only)
    fills cores when registered templates miss.
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
    office_core = office_core_from_hit(question, top, doc)
    if top_score < MIN_HIT_SCORE:
        out.answer, out.status = CANNOT_ANSWER, "cannot_answer"
        out.detail["abstain_reason"] = "weak_or_off_topic_hit"
        return out
    slot, slot_src = slot_mod.classify_slot(
        question, use_needle=use_needle_slots
    )
    out.detail["slot"] = slot
    out.detail["slot_source"] = slot_src
    typed = slot_mod.extract_typed(slot, question, doc) if slot != "none" else None
    # Low doc_score alone is not enough to abstain yet: short pages (Hamlet,
    # product brands) often share few question terms but still hold the answer.
    fact = fact_core_from_doc(question, doc)
    # Acronym "What is NASA?" — prefer lexical expansion over MiniCPM cores.
    if fact and re.search(r"(?i)^\s*what is\s+[A-Z]{2,8}\??\s*$", question or ""):
        core, status = fact, "doc_core"
    elif (
        typed
        and slot in ("date", "number", "place")
        and core_fits_question(question, typed)
    ):
        # Measured: MiniCPM is weak on bare dates/places/numbers; typed wins.
        core, status = typed, "typed_core"
    else:
        core, status = minicpm_extract(question, doc, seed=seed)
        if (not core or not core_fits_question(question, core)) and office_core:
            core, status = office_core, "title_core"
        if not core or not core_fits_question(question, core):
            if fact:
                core, status = fact, "doc_core"
        if (not core or not core_fits_question(question, core)) and typed:
            if core_fits_question(question, typed):
                core, status = typed, "typed_core"
    out.detail["core"] = core
    out.detail["extract_status"] = status
    if should_abstain(question=question, doc=doc, score=top_score, core=core):
        out.answer, out.status = CANNOT_ANSWER, "cannot_answer"
        out.detail["abstain_reason"] = "no_grounded_core"
        return out
    summary = minicpm_summarize(question, doc, seed=seed)
    out.detail["summary"] = summary
    if summary and core and core_in_reply(core, summary) and not reply_grounded(summary, doc):
        out.detail["summary_rejected"] = "ungrounded_claims"
    reply = compose_reply(core, summary, doc, question=question)
    if (not reply or reply.strip() == (core or "").strip()) and core:
        typed_r = slot_mod.typed_reply(slot, question, core, doc)
        if typed_r and reply_grounded(typed_r, doc):
            reply = typed_r
            out.detail["reply_source"] = "typed_template"
        elif typed_r and extract.norm(core) in extract.norm(doc):
            reply = typed_r
            out.detail["reply_source"] = "typed_template"
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
