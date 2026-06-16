import unittest

from neurologybm.queries import build_pmc_query


class QueryTests(unittest.TestCase):
    def test_query_contains_topic_case_license_and_source_clauses(self) -> None:
        query = build_pmc_query("neurology", "training", since="2026/01/01", extra="epilepsy[Title/Abstract]")

        self.assertIn("neurology[MeSH Terms]", query)
        self.assertIn('"case report"[Title]', query)
        self.assertIn("cc_by_license[filter]", query)
        self.assertIn("open_access[filter]", query)
        self.assertIn("2026/01/01:3000/01/01[pmcrdat]", query)
        self.assertIn("(epilepsy[Title/Abstract])", query)

    def test_query_can_include_author_manuscripts(self) -> None:
        query = build_pmc_query("psychiatry", "noncommercial_training", include_author_manuscripts=True)

        self.assertIn("(open_access[filter] OR author_manuscript[filter])", query)
        self.assertIn("cc_by-nc_license[filter]", query)

    def test_query_can_exclude_image_heavy_sources(self) -> None:
        query = build_pmc_query("neurology", "training", text_only=True)

        self.assertIn("NOT (", query)
        self.assertIn('"Teaching NeuroImages"[Title]', query)


if __name__ == "__main__":
    unittest.main()
