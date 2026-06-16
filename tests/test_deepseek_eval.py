import tempfile
import unittest
from pathlib import Path

from neurologybm.case_eval import (
    CaseRecord,
    create_default_registry,
    load_case_registry,
    normalize_result_row,
    queue_from_score,
)
from neurologybm.conversion import normalize_conversion_content
from neurologybm.deepseek import assert_private_path, redact_request_payload


class DeepSeekEvalTests(unittest.TestCase):
    def test_registry_round_trip_uses_private_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "docs" / "DO NOT COMMIT TO GITHUB" / "deepseek_eval" / "case_registry.tsv"
            create_default_registry(registry)

            records = load_case_registry(registry)

        self.assertGreaterEqual(len(records), 6)
        self.assertIn("carey2017", {record.case_id for record in records})

    def test_private_path_guard_rejects_public_path(self) -> None:
        with self.assertRaises(ValueError):
            assert_private_path(Path("/tmp/public-results"))

    def test_result_row_and_queue_defaults(self) -> None:
        record = CaseRecord(
            case_id="case_x",
            source_path=Path("docs/DO NOT COMMIT TO GITHUB/text_reading_order/case_x.txt"),
            case_type="ready_challenge",
            prompt_template="closed_book_diagnosis",
        )
        row = normalize_result_row(
            record=record,
            model="deepseek-chat",
            parsed_content={
                "final_diagnosis": "Example diagnosis",
                "etiology": "Example etiology",
                "top_differential": ["A", "B"],
                "recommended_next_step": "Test",
                "confidence": 0.3,
            },
            score_status="fail",
        )

        self.assertEqual(row["next_queue"], "gold_private_benchmark")
        self.assertEqual(queue_from_score("pass", record), "advanced_api_testing")
        self.assertEqual(row["top_differential"], '["A", "B"]')

    def test_redacts_prompt_content_in_request_payload(self) -> None:
        redacted = redact_request_payload({"messages": [{"role": "user", "content": "private case text"}]})

        self.assertNotIn("private case text", str(redacted))
        self.assertIn("<redacted", redacted["messages"][0]["content"])

    def test_conversion_schema_fills_missing_keys(self) -> None:
        content = normalize_conversion_content({"challenge_prompt": "prompt"})

        self.assertEqual(content["challenge_prompt"], "prompt")
        self.assertEqual(content["evidence_map"], [])
        self.assertIn("answer_key", content)


if __name__ == "__main__":
    unittest.main()

