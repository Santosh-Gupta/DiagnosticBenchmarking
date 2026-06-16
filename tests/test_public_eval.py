import json
import tempfile
import unittest
import csv
from pathlib import Path

from neurologybm.public_eval import (
    _completed_case_ids,
    build_judge_prompt,
    build_public_case_prompt,
    load_public_splits,
    merge_public_score_files,
    normalize_judge_content,
    rebuild_public_results_from_raw,
    summarize_public_scores,
    validate_answer_schema,
)


class PublicEvalTests(unittest.TestCase):
    def test_load_public_splits_limit_and_case_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits.jsonl"
            rows = [
                {"case_id": "case_a", "challenge_prompt": "Prompt A", "answer_rest": "Answer A"},
                {"case_id": "case_b", "challenge_prompt": "Prompt B", "answer_rest": "Answer B"},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            selected = load_public_splits(path, limit=1, case_ids={"case_b"})

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["case_id"], "case_b")

    def test_public_prompt_omits_answer_rest(self) -> None:
        row = {
            "challenge_prompt": "A patient presented with fever.",
            "answer_rest": "The diagnosis is hidden disease.",
        }

        _, user_prompt = build_public_case_prompt(row)

        self.assertIn("A patient presented with fever", user_prompt)
        self.assertNotIn("hidden disease", user_prompt)

    def test_judge_prompt_includes_reference_and_model_answer(self) -> None:
        row = {"answer_rest": "Reference diagnosis"}
        _, user_prompt = build_judge_prompt(row, {"final_diagnosis": "Model diagnosis"})

        self.assertIn("Reference diagnosis", user_prompt)
        self.assertIn("Model diagnosis", user_prompt)

    def test_completed_case_ids_excludes_api_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.tsv"
            path.write_text(
                "case_id\treview_status\ncase_a\tneeds_manual_review\ncase_b\tapi_error\n",
                encoding="utf-8",
            )

            completed = _completed_case_ids(path)

        self.assertEqual(completed, {"case_a"})

    def test_validate_answer_schema_rejects_missing_or_bad_fields(self) -> None:
        valid, errors = validate_answer_schema(
            {
                "final_diagnosis": "Diagnosis",
                "etiology": "",
                "top_differential": ["A", "B"],
                "recommended_next_step": "Treat",
                "confidence": 0.7,
                "evidence_summary": ["Clue"],
                "uncertainty_or_missing_information": [],
            }
        )
        self.assertTrue(valid)
        self.assertEqual(errors, [])

        valid, errors = validate_answer_schema({"final_diagnosis": "Diagnosis", "confidence": 2})
        self.assertFalse(valid)
        self.assertIn("missing:etiology", errors)
        self.assertIn("range:confidence", errors)

    def test_normalize_judge_content_rejects_unknown_status(self) -> None:
        normalized = normalize_judge_content(
            {
                "score_status": "great",
                "diagnosis_status": "CORRECT",
                "next_step_status": "wrong",
                "rationale_status": "partial",
                "expected_key_answer": "Expected",
            }
        )

        self.assertEqual(normalized["score_status"], "ungradable")
        self.assertEqual(normalized["diagnosis_status"], "correct")
        self.assertEqual(normalized["next_step_status"], "ungradable")
        self.assertEqual(normalized["rationale_status"], "partial")

    def test_summarize_public_scores(self) -> None:
        metrics = summarize_public_scores(
            [
                {
                    "score_status": "pass",
                    "diagnosis_status": "correct",
                    "next_step_status": "not_applicable",
                    "rationale_status": "correct",
                    "answer_schema_valid": "true",
                },
                {
                    "score_status": "partial",
                    "diagnosis_status": "partial",
                    "next_step_status": "incorrect",
                    "rationale_status": "partial",
                    "answer_schema_valid": "true",
                },
                {
                    "score_status": "fail",
                    "diagnosis_status": "incorrect",
                    "next_step_status": "incorrect",
                    "rationale_status": "incorrect",
                    "answer_schema_valid": "false",
                },
            ]
        )

        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["score_status_counts"]["pass"], 1)
        self.assertEqual(metrics["pass_rate"], 0.3333)
        self.assertEqual(metrics["pass_or_partial_rate"], 0.6667)
        self.assertEqual(metrics["rationale_correct_or_partial_rate"], 0.6667)

    def test_rebuild_public_results_from_raw_preserves_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "splits.jsonl"
            raw = root / "raw.jsonl"
            out = root / "results.tsv"
            manifest.write_text(
                json.dumps(
                    {
                        "case_id": "case_a",
                        "selection_rank": 1,
                        "source_kind": "native",
                        "pmcid": "PMC1",
                        "challenge_prompt": "Prompt",
                        "answer_rest": "Answer",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            raw.write_text(
                json.dumps(
                    {
                        "type": "answer",
                        "case_id": "case_a",
                        "model": "deepseek-test",
                        "parsed_content": {
                            "final_diagnosis": "Disease",
                            "etiology": "Cause",
                            "top_differential": ["Other"],
                            "recommended_next_step": "Treat",
                            "confidence": 0.8,
                            "evidence_summary": ["Key clue"],
                            "uncertainty_or_missing_information": ["Missing test"],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = rebuild_public_results_from_raw(
                split_manifest_path=manifest,
                raw_records_path=raw,
                output_tsv=out,
            )

            with out.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file, delimiter="\t"))
            self.assertEqual(summary["rebuilt_case_count"], 1)
            self.assertEqual(rows[0]["evidence_summary"], '["Key clue"]')
            self.assertEqual(rows[0]["uncertainty_or_missing_information"], '["Missing test"]')

    def test_merge_public_score_files_replaces_later_rows(self) -> None:
        fields = [
            "selection_rank",
            "case_id",
            "source_kind",
            "pmcid",
            "doi",
            "title",
            "journal",
            "license_key",
            "model",
            "final_diagnosis",
            "recommended_next_step",
            "score_status",
            "diagnosis_status",
            "next_step_status",
            "rationale_status",
            "expected_key_answer",
            "expected_next_step",
            "rationale",
            "answer_schema_valid",
            "answer_schema_errors",
            "review_status",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.tsv"
            retry = root / "retry.tsv"
            out = root / "merged.tsv"
            for path, status, review in [
                (first, "ungradable", "judge_api_error"),
                (retry, "fail", "judge_scored_needs_spotcheck"),
            ]:
                with path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t")
                    writer.writeheader()
                    writer.writerow(
                        {
                            "selection_rank": "1",
                            "case_id": "case_a",
                            "score_status": status,
                            "diagnosis_status": "incorrect",
                            "next_step_status": "incorrect",
                            "rationale_status": "incorrect",
                            "answer_schema_valid": "true",
                            "review_status": review,
                        }
                    )

            summary = merge_public_score_files(score_paths=[first, retry], output_tsv=out)

            rows = list(csv.DictReader(out.open(newline="", encoding="utf-8"), delimiter="\t"))
            self.assertEqual(summary["row_count"], 1)
            self.assertEqual(summary["replacement_case_ids"], ["case_a"])
            self.assertEqual(rows[0]["score_status"], "fail")


if __name__ == "__main__":
    unittest.main()
