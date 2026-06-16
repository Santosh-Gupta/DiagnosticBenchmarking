import unittest
import csv
import json
import tempfile
from pathlib import Path

from neurologybm.split_audit import audit_public_split_row, filter_public_splits_by_audit, summarize_audit_rows


class SplitAuditTests(unittest.TestCase):
    def test_multiple_choice_prompt_requires_repair(self) -> None:
        row = {
            "case_id": "case_mcq",
            "challenge_prompt": "A 50-year-old woman presents with eye pain.\n\nWhat is the diagnosis?\n\n□ a. Cancer\n□ b. Infection",
            "answer_rest": "The answer is infection.",
        }

        audit = audit_public_split_row(row)

        self.assertEqual(audit["decision"], "exclude_or_repair_before_api")
        self.assertIn("multiple_choice_options_in_prompt", audit["issues"])

    def test_non_case_article_requires_repair(self) -> None:
        row = {
            "case_id": "education_article",
            "title": "Clinical reasoning performance among medical students",
            "challenge_prompt": "Clinical reasoning performance among medical students was measured in a survey.",
            "answer_rest": "This was an educational intervention.",
        }

        audit = audit_public_split_row(row)

        self.assertEqual(audit["decision"], "exclude_or_repair_before_api")
        self.assertIn("likely_non_case_article", audit["issues"])

    def test_clean_case_is_ready(self) -> None:
        row = {
            "case_id": "clean",
            "challenge_prompt": (
                "A 44-year-old woman presented with confusion, insomnia, autonomic symptoms, and episodic abnormal "
                "movements. Examination showed fluctuating attention and no clear infectious source. CSF showed "
                "mild lymphocytic pleocytosis and MRI was unrevealing. What is the most likely diagnosis?"
            ),
            "answer_rest": (
                "The diagnosis was autoimmune encephalitis. The key evidence was a subacute neuropsychiatric "
                "syndrome with abnormal movements, inflammatory CSF, and exclusion of infectious causes. The "
                "patient improved with immunotherapy."
            ),
        }

        audit = audit_public_split_row(row)

        self.assertEqual(audit["decision"], "include_ready")
        self.assertEqual(audit["issues"], "")

    def test_summary_counts_issues(self) -> None:
        rows = [
            {"decision": "include_ready", "source_kind": "native", "issues": ""},
            {
                "decision": "exclude_or_repair_before_api",
                "source_kind": "native",
                "issues": "multiple_choice_options_in_prompt,likely_non_case_article",
            },
        ]

        summary = summarize_audit_rows(rows)

        self.assertEqual(summary["decision_counts"]["include_ready"], 1)
        self.assertEqual(summary["issue_counts"]["multiple_choice_options_in_prompt"], 1)

    def test_filter_public_splits_by_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "splits.jsonl"
            audit_csv = root / "audit.csv"
            out = root / "clean.jsonl"
            metadata = root / "clean.csv"
            rows = [
                {"case_id": "keep", "selection_rank": 1, "challenge_prompt": "Prompt", "answer_rest": "Answer"},
                {"case_id": "drop", "selection_rank": 2, "challenge_prompt": "Prompt", "answer_rest": "Answer"},
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            with audit_csv.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["case_id", "decision"])
                writer.writeheader()
                writer.writerow({"case_id": "keep", "decision": "include_ready"})
                writer.writerow({"case_id": "drop", "decision": "exclude_or_repair_before_api"})

            summary = filter_public_splits_by_audit(
                manifest_path=manifest,
                audit_csv_path=audit_csv,
                output_jsonl=out,
                metadata_csv=metadata,
            )

            clean_rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["row_count"], 1)
            self.assertEqual(clean_rows[0]["case_id"], "keep")
            self.assertTrue(metadata.exists())


if __name__ == "__main__":
    unittest.main()
