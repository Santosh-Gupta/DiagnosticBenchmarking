import unittest

from neurologybm.hf_sources import get_hf_sources


class HfSourceTests(unittest.TestCase):
    def test_priority_one_sources_include_easy_first_downloads(self) -> None:
        sources = {source.key: source for source in get_hf_sources(max_priority=1)}

        self.assertIn("medcase_reasoning", sources)
        self.assertIn("case_report_bench", sources)
        self.assertIn("openmed_multicare_cases", sources)
        self.assertIn("openmed_multicare_articles", sources)
        self.assertIn("rarebench", sources)
        self.assertIn("medmistake", sources)
        self.assertNotIn("pmc_patients", sources)
        self.assertNotIn("diagnosis_arena", sources)

    def test_priority_two_sources_include_failure_mining_leads(self) -> None:
        sources = {source.key: source for source in get_hf_sources(max_priority=2)}

        self.assertIn("diagnosis_arena", sources)
        self.assertEqual(sources["diagnosis_arena"].use_tier, "external_benchmark_failure_mining_license_review")
        self.assertIn("medagents_benchmark", sources)
        self.assertIn("mediq", sources)
        self.assertIn("medeinst", sources)
        self.assertIn("rarearena", sources)

    def test_all_with_priority_dedupes(self) -> None:
        sources = get_hf_sources(["medcase_reasoning", "all", "medcase_reasoning"], max_priority=1)

        self.assertEqual(len({source.key for source in sources}), len(sources))
        self.assertIn("medcase_reasoning", {source.key for source in sources})
        self.assertNotIn("pmc_patients", {source.key for source in sources})

    def test_unknown_source_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_hf_sources(["not_a_dataset"])


if __name__ == "__main__":
    unittest.main()
