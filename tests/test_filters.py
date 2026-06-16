import unittest

from neurologybm.filters import keep_article_metadata


class FilterTests(unittest.TestCase):
    def test_keeps_neurology_text_case(self) -> None:
        keep, reasons = keep_article_metadata(
            {
                "title": "A challenging epilepsy case report",
                "abstract": "A patient presented with seizures and autoimmune encephalitis.",
                "article_type": "case-report",
                "journal": "BMC Neurology",
            }
        )

        self.assertTrue(keep)
        self.assertEqual(reasons, [])

    def test_rejects_image_heavy_case(self) -> None:
        keep, reasons = keep_article_metadata(
            {
                "title": "Teaching NeuroImages: an unusual movement disorder",
                "abstract": "A patient presented with ataxia.",
                "article_type": "case-report",
                "journal": "Neurology",
            }
        )

        self.assertFalse(keep)
        self.assertIn("image_or_radiology_heavy", reasons)

    def test_rejects_weak_incidental_neurology_mention(self) -> None:
        keep, reasons = keep_article_metadata(
            {
                "title": "Case Report: Massive rectal bleeding from stercoral ulcers",
                "abstract": "Stercoral colitis has mainly been described in elderly patients with dementia.",
                "article_type": "case-report",
                "journal": "Frontiers in Gastroenterology",
                "keywords": ["colonoscopy", "rectal bleeding"],
            }
        )

        self.assertFalse(keep)
        self.assertIn("missing_neurology_marker", reasons)

    def test_rejects_abstract_only_biomarker_language(self) -> None:
        keep, reasons = keep_article_metadata(
            {
                "title": "A pediatric pulmonary hypertension case report",
                "abstract": "Biomarkers such as brain natriuretic peptide improved after treatment.",
                "article_type": "case-report",
                "journal": "Frontiers in Pediatrics",
                "keywords": ["pulmonary hypertension", "cardiology"],
            }
        )

        self.assertFalse(keep)
        self.assertIn("missing_neurology_marker", reasons)


if __name__ == "__main__":
    unittest.main()
