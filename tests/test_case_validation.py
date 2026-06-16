import unittest

from neurologybm.case_validation import deterministic_flags, validate_case, summarize_validations


class CaseValidationTests(unittest.TestCase):
    def test_flags_specific_gene_absent_from_prompt(self) -> None:
        prompt = ("A child with absence seizures and mild cognitive delay; MRI normal. A gene panel was "
                  "sent for sequencing. What is the most likely genetic diagnosis?")
        flags = deterministic_flags(prompt, "SLC6A1-associated neurodevelopmental disorder")
        self.assertTrue(any(f.startswith("specificity_token_absent") for f in flags))
        self.assertIn("specificity_unsupported", flags)
        self.assertIn("deciding_result_withheld", flags)

    def test_no_flag_when_gene_present_in_prompt(self) -> None:
        prompt = "Sequencing revealed a pathogenic SLC6A1 variant c.1648G>A."
        flags = deterministic_flags(prompt, "SLC6A1-associated neurodevelopmental disorder")
        self.assertNotIn("specificity_unsupported", flags)

    def test_no_flag_for_generic_acronym_gold(self) -> None:
        # A clinical/acronym gold (no specific gene token) should not be flagged on token-absence alone.
        flags = deterministic_flags("A patient with progressive ataxia.", "Multiple System Atrophy (MSA)")
        self.assertNotIn("specificity_unsupported", flags)

    def test_validate_case_deterministic_only(self) -> None:
        row = {"case_id": "c1", "challenge_prompt": "gene panel was sent for sequencing; which gene?",
               "answer_key_diagnosis": "KCNMA1-related epilepsy"}
        v = validate_case(row, client=None)
        self.assertIsNone(v.determinable)
        self.assertTrue(v.deterministic_flags)

    def test_summarize(self) -> None:
        from neurologybm.case_validation import CaseValidation
        rows = [CaseValidation(case_id="a", gold_diagnosis="x", determinable=False, suggested_repair="relax_gold"),
                CaseValidation(case_id="b", gold_diagnosis="y", determinable=True)]
        s = summarize_validations(rows)
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["llm_not_determinable"], 1)
        self.assertEqual(s["repair_relax_gold"], 1)

    def test_gold_overspecific_add_result_auto_mends_keeping_gold(self) -> None:
        # An over-specific gold whose subtype discriminator IS sourceable mends via add_result and the
        # fine gold is preserved (the policy: add the result, never relax the gold).
        from neurologybm.case_validation import CaseValidation, mend_row
        v = CaseValidation(case_id="c", gold_diagnosis="ASMAN variant of GBS", determinable=False,
                           broken_class="gold_overspecific", suggested_repair="add_result",
                           prompt_addition="Nerve conduction confirmed sensory involvement (reduced SNAPs).")
        row = {"case_id": "c", "challenge_prompt": "Ascending weakness, areflexia."}
        mended = mend_row(row, v)
        self.assertIsNotNone(mended)
        self.assertIn("sensory involvement", mended["challenge_prompt"])
        self.assertEqual(mended["mend_provenance"]["repair"], "add_result")  # gold untouched

    def test_gold_overspecific_relabel_is_not_auto_applied(self) -> None:
        # relabel_to_parent changes the gold, so it must NOT auto-mend — surfaced for human review only.
        from neurologybm.case_validation import CaseValidation, mend_row, summarize_validations
        v = CaseValidation(case_id="d", gold_diagnosis="autoimmune-thyroid focal CNS disorder",
                           determinable=False, broken_class="gold_overspecific",
                           suggested_repair="relabel_to_parent", gold_relabel="SREAT")
        self.assertIsNone(mend_row({"case_id": "d", "challenge_prompt": "..."}, v))
        s = summarize_validations([v])
        self.assertEqual(s["broken_gold_overspecific"], 1)
        self.assertEqual(s["repair_relabel_to_parent"], 1)


if __name__ == "__main__":
    unittest.main()
