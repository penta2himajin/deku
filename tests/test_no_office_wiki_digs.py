"""Office/wiki dig helpers and closed polity tables are gone."""

from __future__ import annotations

import unittest

from deku import web_search as ws


class TestNoOfficeWikiDigs(unittest.TestCase):
    def test_office_page_title_gone(self):
        self.assertFalse(hasattr(ws, "office_page_title") and callable(
            getattr(ws, "office_page_title", None)
        ) and ws.office_page_title("Who is the pope?") == "Pope")
        # Prefer explicit absence or always-None stub.
        fn = getattr(ws, "office_page_title", None)
        if fn is not None:
            self.assertIsNone(fn("Who is the prime minister of Japan?"))
            self.assertIsNone(fn("Who is the pope?"))

    def test_wiki_incumbent_gone(self):
        self.assertFalse(hasattr(ws, "wiki_incumbent_from_page"))

    def test_wiki_birth_date_gone(self):
        self.assertFalse(hasattr(ws, "wiki_birth_date"))

    def test_wiki_founded_year_gone(self):
        self.assertFalse(hasattr(ws, "wiki_founded_year"))

    def test_wikidata_ceo_name_gone(self):
        self.assertFalse(hasattr(ws, "wikidata_ceo_name"))

    def test_articled_polities_gone(self):
        self.assertFalse(hasattr(ws, "_ARTICLED_POLITIES"))

    def test_non_person_entity_table_gone(self):
        self.assertFalse(hasattr(ws, "_NON_PERSON"))


if __name__ == "__main__":
    unittest.main()
