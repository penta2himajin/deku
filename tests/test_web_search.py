import sys
import unittest
from pathlib import Path

from deku import web_search as ws


class TestIntent(unittest.TestCase):
    def test_parse_search_label(self):
        self.assertEqual(ws.parse_intent("INTENT: search"), "search")
        self.assertEqual(ws.parse_intent("search\n"), "search")

    def test_refuse_and_extract(self):
        self.assertEqual(ws.parse_intent("INTENT: refuse"), "refuse")
        self.assertEqual(ws.parse_intent("INTENT: extract"), "extract")

    def test_garbage_is_refuse(self):
        self.assertEqual(ws.parse_intent("I think you should look it up"), "refuse")


class TestQuery(unittest.TestCase):
    def test_parse_query_line(self):
        self.assertEqual(
            ws.parse_query('QUERY: Alphabet Inc company parent Google'),
            "Alphabet Inc company parent Google",
        )

    def test_fallback_strips_intent(self):
        self.assertEqual(ws.parse_query("just some words"), "just some words")


class TestRank(unittest.TestCase):
    def test_port_snippet_wins(self):
        q = "What is the port number?"
        hits = [
            {"title": "Coffee", "snippet": "Coffee is served in the hall.", "url": "a"},
            {"title": "Server", "snippet": "The server listens on port 8765.", "url": "b"},
            {"title": "Retry", "snippet": "Retries are set to 3.", "url": "c"},
        ]
        top = ws.rank_hits(q, hits, k=2)
        self.assertEqual(top[0]["url"], "b")
        self.assertEqual(len(top), 2)

    def test_company_prefers_inc_title(self):
        q = "Who is the CEO of Apple?"
        hits = [
            {"title": "Apple", "snippet": "An apple is a fruit.", "url": "fruit"},
            {"title": "Apple Inc.", "snippet": "Tim Cook is the CEO of Apple Inc.", "url": "corp"},
        ]
        top = ws.rank_hits(q, hits, k=2)
        self.assertEqual(top[0]["url"], "corp")

    def test_lvmh_ceo_prefers_incumbent(self):
        hits = [
            {
                "title": "Jean-Claude Biver",
                "snippet": (
                    "He previously served as the CEO of TAG Heuer. "
                    "From 2014 until 2018 he worked at LVMH."
                ),
                "url": "u1",
            },
            {
                "title": "Bernard Arnault",
                "snippet": (
                    "He is the chairman and chief executive officer (CEO) of LVMH."
                ),
                "url": "u2",
            },
        ]
        top = ws.rank_hits("Who is the current CEO of LVMH?", hits, k=2)
        self.assertEqual(top[0]["title"], "Bernard Arnault")


class TestDocs(unittest.TestCase):
    def test_hits_become_extract_document(self):
        hits = [
            {"title": "Alphabet", "snippet": "Alphabet Inc. is the parent of Google.",
             "url": "https://example.com/a"},
        ]
        doc = ws.hits_to_document(hits)
        self.assertIn("Alphabet Inc.", doc)
        self.assertIn("example.com/a", doc)
        self.assertNotIn("[1]", doc)


class TestRuleIntent(unittest.TestCase):
    def test_who_is_goes_search(self):
        self.assertEqual(ws.rule_intent("What company is Alphabet?"), "search")

    def test_math_refuses_search(self):
        self.assertEqual(ws.rule_intent("What is 2+2?"), "refuse")

    def test_rule_query_strips_syntax(self):
        self.assertIn("Alphabet", ws.rule_query("What company is Alphabet?"))
        self.assertNotIn("What", ws.rule_query("What company is Alphabet?"))

    def test_wiki_friendly_drops_ceo(self):
        self.assertEqual(ws.wiki_friendly_query("CEO Apple"), "Apple")
        self.assertEqual(ws.wiki_friendly_query("Alphabet"), "")


class TestComposeReply(unittest.TestCase):
    def test_core_must_appear_in_summary(self):
        self.assertTrue(ws.core_in_reply("Tim Cook", "Tim Cook is the CEO of Apple."))
        self.assertFalse(ws.core_in_reply("Tim Cook", "John Ternus will lead Apple."))

    def test_summary_kept_when_grounded(self):
        doc = "Tim Cook\nTim Cook is the CEO of Apple since 2011.\nSource: x"
        got = ws.compose_reply(
            "Tim Cook", "Tim Cook is the CEO of Apple.", doc)
        self.assertEqual(got, "Tim Cook is the CEO of Apple.")

    def test_fallback_to_sentence_with_core(self):
        doc = (
            "Tim Cook\n"
            "American executive Tim Cook has served as CEO of Apple since 2011.\n"
            "Source: x"
        )
        got = ws.compose_reply("Tim Cook", "Someone else runs it.", doc)
        self.assertIn("Tim Cook", got)
        self.assertIn("CEO", got)

    def test_fallback_core_alone_if_no_sentence(self):
        doc = "Title\nNo useful body here.\nSource: x"
        self.assertEqual(ws.compose_reply("Paris", "London is capital.", doc), "Paris")

    def test_short_summary_rejected_even_if_core_present(self):
        doc = (
            "IPhone\n"
            "The iPhone is developed and marketed by Apple.\n"
            "Source: x"
        )
        got = ws.compose_reply("Apple", "Apple", doc)
        self.assertIn("Apple", got)
        self.assertGreater(len(got.split()), 2)

    def test_invented_date_fails_sentence_grounding(self):
        doc = (
            "End of World War II in Europe\n"
            "World War II ended in Europe in 1945 with the surrender of Germany.\n"
            "Source: x"
        )
        self.assertFalse(ws.reply_grounded(
            "World War II ended in Europe in 1945, and the war concluded "
            "with the surrender on May 30th.",
            doc,
        ))
        self.assertTrue(ws.reply_grounded(
            "World War II ended in Europe in 1945.",
            doc,
        ))

    def test_compose_drops_ungrounded_summary(self):
        doc = (
            "End of World War II in Europe\n"
            "World War II ended in Europe in 1945 with the surrender of Germany.\n"
            "Source: x"
        )
        bad = (
            "World War II ended in Europe in 1945, and the war concluded "
            "with the surrender on May 30th."
        )
        got = ws.compose_reply("1945", bad, doc)
        self.assertNotIn("May 30", got or "")
        self.assertIn("1945", got or "")

    def test_lowercase_month_is_claim(self):
        doc = "World War II ended in Europe in 1945 with the surrender of Germany."
        self.assertFalse(ws.reply_grounded("World War II ended in july 1945.", doc))

    def test_paraphrase_makes_still_ok(self):
        doc = "The iPhone is a line of smartphones developed and marketed by Apple."
        self.assertTrue(ws.reply_grounded("Apple makes the iPhone.", doc))

    def test_invented_org_rejected(self):
        doc = "Alphabet Inc. is an American holding company and parent of Google."
        self.assertFalse(ws.reply_grounded(
            "Alphabet Inc. is owned by SoftBank Group.", doc))

    def test_ungrounded_content_verb_rejected(self):
        doc = "World War II ended in Europe in 1945 with the surrender of Germany."
        # no month/number invention, but 'annexed' is not in the notes
        self.assertFalse(ws.reply_grounded(
            "World War II annexed Europe in 1945.", doc))

    def test_invented_because_rejected(self):
        doc = (
            "Boiling point\n"
            "Water boils at 100 °C at atmospheric pressure. At lower pressure "
            "the boiling point decreases.\n"
        )
        self.assertFalse(ws.reply_grounded(
            "Water boils at 100°C because the boiling point is lower than "
            "at atmospheric pressure.",
            doc,
        ))

    def test_paraphrase_ok_has_no_entity_names(self):
        entityish = {
            "france", "paris", "apple", "google", "alphabet", "cook",
            "germany", "mars", "europe", "surrender", "lattice", "champ",
        }
        self.assertFalse(entityish & ws._PARAPHRASE_OK)

    def test_question_terms_need_not_repeat_in_notes(self):
        doc = "ExampleCorp\nExampleCorp is a technology company."
        self.assertTrue(ws.reply_grounded(
            "ExampleCorp makes technology products.",
            doc,
            question="What company makes ExampleCorp products?",
        ))


class TestCoreFit(unittest.TestCase):
    def test_who_rejects_bare_year(self):
        self.assertFalse(ws.core_fits_question("Who founded Microsoft?", "1975"))
        self.assertTrue(ws.core_fits_question("Who founded Microsoft?", "Bill Gates"))

    def test_when_wants_a_digit(self):
        self.assertFalse(ws.core_fits_question("When did World War II end?", "Germany"))
        self.assertTrue(ws.core_fits_question("When did World War II end?", "1945"))

    def test_recover_list_answer(self):
        doc = "Paris\nParis is the capital and largest city of France.\n"
        self.assertEqual(
            ws.recover_core("1. Paris", doc, "What is the capital of France?"),
            "Paris",
        )

    def test_recover_name_from_long_ungrounded(self):
        doc = (
            "Family of Bill Gates\n"
            "He co-founded Microsoft. Bill Gates is a business magnate.\n"
        )
        self.assertEqual(
            ws.recover_core(
                "Microsoft was founded by Bill Gates and Melinda.",
                doc,
                "Who founded Microsoft?",
            ),
            "Bill Gates",
        )


class TestAbstain(unittest.TestCase):
    def test_low_score_abstains(self):
        self.assertTrue(ws.should_abstain(question="What ocean is the largest?",
                                          doc="Ocean sunfish\nA large fish.\n",
                                          score=1, core="sunfish"))
        self.assertFalse(ws.should_abstain(
            question="Who is the CEO of Apple?",
            doc="Tim Cook\nCEO of Apple since 2011.\n",
            score=5, core="Tim Cook"))

    def test_doc_must_be_on_topic(self):
        from deku import extract
        q = "Who wrote Romeo and Juliet?"
        doc = "Ocean sunfish\nA large bony fish.\n"
        score = float(extract.term_score(q, doc))
        self.assertTrue(ws.should_abstain(
            question=q, doc=doc, score=score, core="Mola"))


class TestExtraFailures(unittest.TestCase):
    """Held-out failure modes from web_search_live_extra."""

    def test_fact_core_capital_of_france(self):
        doc = (
            "Paris\n"
            "Paris is the capital and largest city of France, with an "
            "estimated city population of 2.04 million.\n"
        )
        self.assertEqual(
            ws.fact_core_from_doc("What is the capital of France?", doc),
            "Paris",
        )

    def test_fact_core_chemical_and_author(self):
        fe = ws.fact_core_from_doc(
            "What is the chemical symbol for iron?",
            "Iron is a chemical element with symbol Fe and atomic number 26.",
        )
        self.assertEqual(fe, "Fe")
        author = ws.fact_core_from_doc(
            "Who wrote Hamlet?",
            "Hamlet — The Tragedy of Hamlet is a play by William Shakespeare.",
        )
        self.assertEqual(author, "William Shakespeare")

    def test_who_wrote_rejects_work_as_author(self):
        self.assertFalse(
            ws.core_fits_question("Who wrote Hamlet?", "Hamlet")
        )
        self.assertTrue(
            ws.core_fits_question("Who wrote Hamlet?", "William Shakespeare")
        )

    def test_hamlet_prefers_play_over_restaurant(self):
        hits = [
            {
                "title": "Hamburger Hamlet",
                "snippet": "Hamburger Hamlet is a restaurant chain.",
                "url": "https://en.wikipedia.org/wiki/Hamburger_Hamlet",
            },
            {
                "title": "Hamlet",
                "snippet": "The Tragedy of Hamlet is a play by William Shakespeare.",
                "url": "https://en.wikipedia.org/wiki/Hamlet",
            },
        ]
        top = ws.rank_hits("Who wrote Hamlet?", hits, k=2)
        self.assertEqual(top[0]["title"], "Hamlet")
        self.assertIn("Shakespeare", top[0]["snippet"])

    def test_apollo11_prefers_exact_mission(self):
        hits = [
            {
                "title": "Apollo 10",
                "snippet": "Apollo 10 was a May 1969 rehearsal for the moon landing.",
                "url": "https://en.wikipedia.org/wiki/Apollo_10",
            },
            {
                "title": "Apollo 11 50th Anniversary commemorative coins",
                "snippet": "Issued in 2019 to commemorate Apollo 11.",
                "url": "https://en.wikipedia.org/wiki/Apollo_11_50th_Anniversary_commemorative_coins",
            },
            {
                "title": "Apollo 11",
                "snippet": "Apollo 11 was the spaceflight that first landed humans on the Moon in 1969.",
                "url": "https://en.wikipedia.org/wiki/Apollo_11",
            },
        ]
        top = ws.rank_hits("When did the Apollo 11 moon landing happen?", hits, k=2)
        self.assertEqual(top[0]["title"], "Apollo 11")

    def test_fact_core_author_stops_at_name(self):
        author = ws.fact_core_from_doc(
            "Who wrote Hamlet?",
            "Hamlet is a tragedy written by William Shakespeare sometime between 1599 and 1601.",
        )
        self.assertEqual(author, "William Shakespeare")

    def test_maker_rejects_product_title_core(self):
        self.assertFalse(
            ws.core_fits_question(
                "What company makes the PlayStation?", "Sony PlayStation"
            )
        )
        self.assertTrue(
            ws.core_fits_question("What company makes the PlayStation?", "Sony")
        )

    def test_grounded_core_skips_low_doc_score_abstain(self):
        doc = (
            "Hamlet (play)\n"
            "Hamlet is a tragedy written by William Shakespeare.\n"
            "Source: https://en.wikipedia.org/wiki/Hamlet_(play)"
        )
        self.assertFalse(
            ws.should_abstain(
                question="Who wrote Hamlet?",
                doc=doc,
                score=8.0,
                core="William Shakespeare",
            )
        )

    def test_playstation_prefers_sony(self):
        hits = [
            {
                "title": "PlayStation",
                "snippet": "PlayStation is a video game brand.",
                "url": "https://en.wikipedia.org/wiki/PlayStation",
            },
            {
                "title": "Sony",
                "snippet": "Sony developed and makes the PlayStation consoles.",
                "url": "https://en.wikipedia.org/wiki/Sony",
            },
        ]
        top = ws.rank_hits("What company makes the PlayStation?", hits, k=2)
        self.assertEqual(top[0]["title"], "Sony")

    def test_pm_prefers_person_over_office(self):
        hits = [
            {
                "title": "Prime Minister of the United Kingdom",
                "snippet": "The prime minister of the United Kingdom is the head of government.",
                "url": "https://en.wikipedia.org/wiki/Prime_Minister_of_the_United_Kingdom",
            },
            {
                "title": "Andy Burnham",
                "snippet": (
                    "Andy Burnham has served as Prime Minister of the "
                    "United Kingdom since 2026."
                ),
                "url": "https://en.wikipedia.org/wiki/Andy_Burnham",
            },
        ]
        top = ws.rank_hits(
            "Who is the prime minister of the United Kingdom?", hits, k=2
        )
        self.assertEqual(top[0]["title"], "Andy Burnham")

    def test_pm_prefers_current_over_former(self):
        hits = [
            {
                "title": "Keir Starmer",
                "snippet": (
                    "Keir Starmer is a former Prime Minister of the "
                    "United Kingdom who served from 2024 to 2026."
                ),
                "url": "https://en.wikipedia.org/wiki/Keir_Starmer",
            },
            {
                "title": "Andy Burnham",
                "snippet": (
                    "Andy Burnham has served as Prime Minister of the "
                    "United Kingdom since July 2026."
                ),
                "url": "https://en.wikipedia.org/wiki/Andy_Burnham",
            },
        ]
        top = ws.rank_hits(
            "Who is the prime minister of the United Kingdom?", hits, k=2
        )
        self.assertEqual(top[0]["title"], "Andy Burnham")

    def test_what_is_acronym_prefers_exact_title(self):
        hits = [
            {
                "title": "NASA Astronaut Corps",
                "snippet": "The NASA Astronaut Corps is a unit of NASA.",
                "url": "u1",
            },
            {
                "title": "NASA",
                "snippet": (
                    "The National Aeronautics and Space Administration "
                    "(NASA) is an independent agency of the U.S. federal government."
                ),
                "url": "u2",
            },
        ]
        top = ws.rank_hits("What is NASA?", hits, k=2)
        self.assertEqual(top[0]["title"], "NASA")
        self.assertGreaterEqual(
            ws.rank_hits_scored("What is NASA?", hits, k=1)[0][0],
            ws.MIN_HIT_SCORE,
        )

    def test_office_core_from_premiership_title(self):
        hit = {
            "title": "Premiership of Andy Burnham",
            "snippet": "Andy Burnham's premiership began in 2026.",
            "url": "https://en.wikipedia.org/wiki/Premiership_of_Andy_Burnham",
        }
        doc = ws.hits_to_document(
            [hit], question="Who is the prime minister of the United Kingdom?"
        )
        self.assertEqual(
            ws.office_core_from_hit(
                "Who is the prime minister of the United Kingdom?", hit, doc
            ),
            "Andy Burnham",
        )

    def test_fact_core_what_is_nasa(self):
        doc = (
            "NASA\n"
            "The National Aeronautics and Space Administration (NASA) is an "
            "independent agency of the U.S. federal government.\n"
            "Source: https://en.wikipedia.org/wiki/NASA"
        )
        core = ws.fact_core_from_doc("What is NASA?", doc)
        self.assertIsNotNone(core)
        self.assertIn("National Aeronautics", core or "")

    def test_expand_hamlet_and_apollo(self):
        qs = ws.expand_search_queries("Who wrote Hamlet?", "Who wrote Hamlet?")
        self.assertIn("Who wrote Hamlet?", qs)
        self.assertNotIn("Shakespeare", " ".join(qs))
        qs2 = ws.expand_search_queries(
            "When did the Apollo 11 moon landing happen?",
            "When did the Apollo 11 moon landing happen?",
        )
        self.assertIn("When did the Apollo 11 moon landing happen?", qs2)


class TestQueryExpand(unittest.TestCase):
    def test_who_wrote_includes_question(self):
        qs = ws.expand_search_queries(
            "Who wrote Romeo and Juliet?", "Who wrote Romeo and Juliet?"
        )
        self.assertIn("Who wrote Romeo and Juliet?", qs)


class TestEnrichAndRank(unittest.TestCase):
    def test_fragment_snip_detected(self):
        self.assertTrue(ws.looks_like_fragment(
            "Because of this, water boils at 100 °C, rounded from scientific"))
        self.assertTrue(ws.looks_like_fragment(
            "lower pressure, has a lower boiling point"))
        self.assertFalse(ws.looks_like_fragment(
            "The boiling point of water is 100 °C at standard pressure."))

    def test_enrich_replaces_office_snippet(self):
        hits = [{
            "title": "President of France",
            "snippet": "The president of France, officially the president of the French Republic (French:",
            "url": "https://en.wikipedia.org/wiki/President_of_France",
        }]
        # Office-page summaries often omit the incumbent; enrich only when
        # the fetched text actually names a person.
        enriched = ws.enrich_hits_for_answer(
            "Who is the president of France?",
            hits,
            summary_fn=lambda t: (
                "The president of France is the head of state. "
                "Emmanuel Macron is the current president of France."
            ),
        )
        self.assertIn("Emmanuel Macron", enriched[0]["snippet"])
        self.assertEqual(enriched[0].get("enriched"), "wiki_summary")

    def test_enrich_skips_office_summary_without_person(self):
        hits = [{
            "title": "President of France",
            "snippet": "The president of France, officially the president of the French Republic.",
            "url": "https://en.wikipedia.org/wiki/President_of_France",
        }]
        enriched = ws.enrich_hits_for_answer(
            "Who is the president of France?",
            hits,
            summary_fn=lambda t: (
                "The president of France is the executive head of state of France."
            ),
        )
        self.assertNotEqual(enriched[0].get("enriched"), "wiki_summary")

    def test_president_prefers_current_over_historical(self):
        hits = [
            {
                "title": "Presidency of Charles de Gaulle",
                "snippet": "Charles de Gaulle's tenure as the 18th president of France began in 1959.",
                "url": "https://en.wikipedia.org/wiki/Presidency_of_Charles_de_Gaulle",
            },
            {
                "title": "Emmanuel Macron",
                "snippet": "Emmanuel Macron has served as President of France since 2017.",
                "url": "https://en.wikipedia.org/wiki/Emmanuel_Macron",
            },
            {
                "title": "Presidency of Emmanuel Macron",
                "snippet": "Emmanuel Macron's presidency began on 14 May 2017.",
                "url": "https://en.wikipedia.org/wiki/Presidency_of_Emmanuel_Macron",
            },
        ]
        top = ws.rank_hits("Who is the president of France?", hits, k=2)
        self.assertIn("Macron", top[0]["title"])

    def test_office_core_overrides_year_extract(self):
        hit = {
            "title": "Presidency of Emmanuel Macron",
            "snippet": "Emmanuel Macron's presidency began on 14 May 2017.",
            "url": "https://en.wikipedia.org/wiki/Presidency_of_Emmanuel_Macron",
        }
        doc = ws.hits_to_document([hit], question="Who is the president of France?")
        self.assertFalse(ws.core_fits_question("Who is the president of France?", "2017"))
        self.assertEqual(
            ws.office_core_from_hit("Who is the president of France?", hit, doc),
            "Emmanuel Macron",
        )

    def test_boiling_prefers_complete_sentence_hit(self):
        hits = [
            {
                "title": "Boiling point",
                "snippet": "Because of this, water boils at 100 °C, rounded from scientific",
                "url": "https://en.wikipedia.org/wiki/Boiling_point",
            },
            {
                "title": "Boiling",
                "snippet": "The boiling point of water is 100 °C or 212 °F at standard pressure.",
                "url": "https://en.wikipedia.org/wiki/Boiling",
            },
        ]
        top = ws.rank_hits("What is the boiling point of water?", hits, k=2)
        self.assertEqual(top[0]["title"], "Boiling")

    def test_boiling_template(self):
        doc = "Water boils at 100 °C at atmospheric pressure."
        got = ws.template_reply(
            "What is the boiling point of water?", "100", doc,
        )
        self.assertEqual(got, "The boiling point of water is 100°C.")

    def test_prefer_boiling_span(self):
        snip = (
            "surface. Transition boiling is unstable. "
            "The boiling point of water is 100 °C or 212 °F, under standard pressure."
        )
        got = ws.prefer_answer_span(snip, "What is the boiling point of water?")
        self.assertTrue(got.startswith("The boiling point of water is"))
        self.assertIn("100", got)
        hits = [
            {"title": "Apple", "snippet": "An apple is a fruit.", "url": "fruit"},
            {"title": "Apple Inc.", "snippet": "Tim Cook is the CEO of Apple Inc.", "url": "corp"},
        ]
        top = ws.rank_hits("Who is the CEO of Apple?", hits, k=2)
        self.assertEqual(top[0]["url"], "corp")


if __name__ == "__main__":
    unittest.main()
