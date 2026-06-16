import json
import tempfile
import unittest
from pathlib import Path

from neurologybm.public_refine import (
    _review_status_with_reasons,
    _refine_one,
    build_public_refinement_prompt,
    extract_article_text_from_xml,
    normalize_refined_content,
    solvability_probe,
    validate_refinement,
)


class _StubClient:
    """Returns queued parsed_content payloads for successive chat_json calls."""

    def __init__(self, payloads):
        self._payloads = list(payloads)

    def chat_json(self, **kwargs):
        return {"parsed_content": self._payloads.pop(0)}


class PublicRefineTests(unittest.TestCase):
    def test_build_public_refinement_prompt_includes_no_leak_instruction(self) -> None:
        row = {
            "case_id": "case_a",
            "title": "Secret Diagnosis in the Title",
            "challenge_prompt": "A patient presented with confusion.",
            "answer_rest": "The diagnosis was hidden disease.",
        }

        _, user_prompt = build_public_refinement_prompt(row, "Case text")

        self.assertIn("article_title_do_not_leak_into_prompt: Secret Diagnosis", user_prompt)
        self.assertIn("must exclude answer leakage", user_prompt)
        self.assertIn("challenge_prompt", user_prompt)

    def test_prompt_includes_preservation_fabrication_solvability_rules(self) -> None:
        _, user_prompt = build_public_refinement_prompt({"case_id": "c"}, "Case text")
        self.assertIn("PRESERVE THE DISCRIMINATORS", user_prompt)
        self.assertIn("NO FABRICATION", user_prompt)
        self.assertIn("SOLVABILITY MUST MATCH THE QUESTION", user_prompt)
        self.assertIn("fidelity_audit", user_prompt)
        self.assertIn("solvability_audit", user_prompt)

    def test_validate_flags_dropped_discriminator(self) -> None:
        # NPSLE-style defect: the deciding serologies are required but absent from the prompt.
        content = {
            "challenge_prompt": "A 19-year-old woman with acute psychosis and a normal MRI.",
            "answer_key": {"required_findings": ["Positive ANA 1:1280 and anti-dsDNA >300 IU/mL"]},
        }
        reasons = validate_refinement(content, article_text="ANA 1:1280, anti-dsDNA >300 IU/mL")
        self.assertTrue(any(r.startswith("required_finding_absent_from_prompt") for r in reasons))

    def test_validate_flags_unsourced_value(self) -> None:
        # Fabricated/altered value: titer in the prompt does not appear in the source.
        content = {
            "challenge_prompt": "Serology showed ANA 1:2560.",
            "answer_key": {"required_findings": ["ANA 1:2560"]},
        }
        reasons = validate_refinement(content, article_text="The ANA titer was 1:1280.")
        self.assertTrue(any(r.startswith("unsourced_value_in_prompt") for r in reasons))

    def test_validate_clean_case_has_no_reasons(self) -> None:
        content = {
            "challenge_prompt": "ANA was 1:1280 and anti-dsDNA was >300 IU/mL with acute psychosis.",
            "answer_key": {"required_findings": ["ANA 1:1280", "anti-dsDNA >300 IU/mL"]},
        }
        reasons = validate_refinement(content, article_text="ANA 1:1280; anti-dsDNA >300 IU/mL; psychosis.")
        self.assertEqual(reasons, [])

    def test_validate_allows_demographic_numeric_paraphrase(self) -> None:
        content = {
            "challenge_prompt": "A 20-year-old woman had epilepsy.",
            "answer_key": {"required_findings": ["Age 20 years"]},
        }
        reasons = validate_refinement(content, article_text="A 20-year-old woman had epilepsy.")
        self.assertEqual(reasons, [])

    def test_review_status_not_solvable_when_finding_missing(self) -> None:
        content = {
            "challenge_prompt": "Imaging shows a 7 cm renal mass; histopathology pending.",
            "answer_key": {"diagnosis": "Intrarenal neurofibroma", "required_findings": ["S100-positive spindle cells with wavy nuclei"]},
            "leakage_audit": {"has_leakage": False},
            "adequacy_audit": {"is_self_contained": True},
        }
        status, reasons = _review_status_with_reasons(content, article_text="S100-positive spindle cells with wavy nuclei")
        self.assertEqual(status, "not_solvable")

    def test_review_status_needs_fidelity_review_on_self_report(self) -> None:
        content = {
            "challenge_prompt": "ANA 1:1280 with psychosis.",
            "answer_key": {"diagnosis": "NPSLE", "required_findings": ["ANA 1:1280"]},
            "leakage_audit": {"has_leakage": False},
            "adequacy_audit": {"is_self_contained": True},
            "fidelity_audit": {"all_findings_sourced": False},
        }
        status, _ = _review_status_with_reasons(content, article_text="ANA 1:1280")
        self.assertEqual(status, "needs_fidelity_review")

    def test_review_status_needs_fidelity_review_on_possible_polarity_conflict(self) -> None:
        content = {
            "challenge_prompt": "MRI brain showed white matter hyperintense lesions.",
            "answer_key": {"diagnosis": "NPSLE", "required_findings": ["white matter hyperintense lesions"]},
            "leakage_audit": {"has_leakage": False},
            "adequacy_audit": {"is_self_contained": True},
        }
        status, reasons = _review_status_with_reasons(
            content,
            article_text="MRI brain was normal. White matter signal was unremarkable.",
        )
        self.assertEqual(status, "needs_fidelity_review")
        self.assertTrue(any(r.startswith("possible_polarity_conflict") for r in reasons))

    def test_solvability_probe_solvable(self) -> None:
        client = _StubClient([
            {"diagnosis": "Neuropsychiatric SLE", "next_step": "steroids"},
            {"equivalent": True, "rationale": "same entity"},
        ])
        result = solvability_probe(
            client=client, model="m", challenge_prompt="ANA 1:1280, psychosis", expected_diagnosis="NPSLE"
        )
        self.assertTrue(result["is_solvable"])
        self.assertEqual(result["probe_diagnosis"], "Neuropsychiatric SLE")

    def test_solvability_probe_underdetermined(self) -> None:
        # Blind model reaches a different entity -> prompt is underdetermined.
        client = _StubClient([
            {"diagnosis": "Sarcomatoid renal cell carcinoma", "next_step": "nephrectomy"},
            {"equivalent": False, "rationale": "different entity"},
        ])
        result = solvability_probe(
            client=client, model="m", challenge_prompt="7 cm renal mass, histology pending",
            expected_diagnosis="Intrarenal neurofibroma",
        )
        self.assertFalse(result["is_solvable"])

    def test_refine_one_marks_not_solvable_when_probe_fails(self) -> None:
        client = _StubClient([
            {
                "challenge_prompt": "A renal mass is present; histology is pending.",
                "answer_key": {
                    "diagnosis": "Intrarenal neurofibroma",
                    "required_findings": ["renal mass"],
                    "aliases": [],
                    "etiology": "",
                    "next_management_step": "biopsy",
                    "optional_findings": [],
                },
                "evidence_map": [],
                "hypothesis_bank": [],
                "outcome_summary": "",
                "leakage_audit": {"has_leakage": False},
                "adequacy_audit": {"is_self_contained": True},
                "fidelity_audit": {"all_findings_sourced": True, "values_match_source": True},
                "solvability_audit": {"is_solvable_from_prompt": True},
                "source_usage": {},
            },
            {"diagnosis": "Renal cell carcinoma", "next_step": "nephrectomy"},
            {"equivalent": False, "rationale": "different entity"},
        ])

        artifact, raw = _refine_one(
            client=client,
            row={"case_id": "case_probe"},
            model="m",
            dry_run=False,
            temperature=0.0,
            extra_body=None,
            max_article_chars=1000,
            include_article_text=False,
            api_retries=0,
            api_retry_sleep_seconds=0,
            solvability_probe_model="m",
        )

        self.assertEqual(artifact["review_status"], "not_solvable")
        self.assertTrue(any(r.startswith("solvability_probe_failed") for r in artifact["validation_reasons"]))
        self.assertIn("solvability_probe", raw)

    def test_normalize_refined_content_defaults(self) -> None:
        normalized = normalize_refined_content({"challenge_prompt": "Prompt", "answer_key": {"diagnosis": "Dx"}})

        self.assertEqual(normalized["challenge_prompt"], "Prompt")
        self.assertEqual(normalized["answer_key"]["diagnosis"], "Dx")
        self.assertEqual(normalized["evidence_map"], [])
        self.assertEqual(normalized["leakage_audit"], {})

    def test_extract_article_text_from_xml(self) -> None:
        xml = """<article>
          <body>
            <sec><title>Case presentation</title><p>A 20-year-old woman presented with headache.</p></sec>
            <sec><title>Discussion</title><p>The diagnosis was example disease.</p></sec>
          </body>
        </article>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "PMC1.xml"
            path.write_text(xml, encoding="utf-8")

            text = extract_article_text_from_xml(path)

        self.assertIn("## Case presentation", text)
        self.assertIn("20-year-old woman", text)
        self.assertIn("## Discussion", text)


if __name__ == "__main__":
    unittest.main()
