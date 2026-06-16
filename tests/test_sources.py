import unittest

from neurologybm.sources import build_source_query, get_pmc_sources


class SourceTests(unittest.TestCase):
    def test_get_all_sources(self) -> None:
        sources = get_pmc_sources(None)

        self.assertGreater(len(sources), 5)
        self.assertIn("broad_neurology_cases", {source.key for source in sources})

    def test_build_source_query_uses_source_extra(self) -> None:
        source = get_pmc_sources(["bmc_neurology"])[0]
        query = build_source_query(source, "training")

        self.assertIn('"BMC Neurology"[journal]', query)
        self.assertIn("cc_by_license[filter]", query)

    def test_specialty_expansion_sources_are_registered(self) -> None:
        sources = {source.key: source for source in get_pmc_sources(None)}

        self.assertIn("jacc_case_reports", sources)
        self.assertIn("acg_case_reports_journal", sources)
        self.assertIn("idcases", sources)
        self.assertIn("american_journal_ophthalmology_case_reports", sources)
        self.assertIn("medicine_baltimore", sources)
        self.assertIn("american_journal_case_reports", sources)
        self.assertIn("jimd_reports", sources)
        self.assertIn("neuro_oncology_cases", sources)
        self.assertIn("neuro_oncology_advances", sources)
        self.assertIn("neurocase", sources)
        self.assertIn("tremor_hyperkinetic_movements", sources)
        self.assertIn("stroke_vascular_neurology", sources)
        self.assertIn("journal_clinical_neurology", sources)
        self.assertIn("ame_case_reports", sources)
        self.assertIn("nmc_case_report_journal", sources)
        self.assertIn("bmj_neurology_open", sources)
        self.assertIn("neurology_open_access", sources)
        self.assertIn("open_neurology_journal", sources)
        self.assertIn("journal_child_neurology", sources)
        self.assertIn("archives_clinical_medical_case_reports", sources)
        self.assertIn("frontiers_neurology_cpc_collection", sources)
        self.assertIn("frontiers_neuroscience", sources)
        self.assertIn("frontiers_neuroscience_case_reports", sources)
        self.assertIn("frontiers_human_neuroscience", sources)
        self.assertIn("frontiers_aging_neuroscience", sources)
        self.assertIn("frontiers_psychiatry", sources)
        self.assertIn("neurology_main", sources)
        self.assertIn("jcs_cases", sources)
        self.assertIn("jnop_cases", sources)
        self.assertIn("mov_disord_vignettes", sources)
        self.assertIn("headache_journal", sources)
        self.assertIn("neuro_ophthalmology_cases", sources)
        self.assertIn("sleep_neurology_cases", sources)
        self.assertIn("arq_neuro_psiq", sources)
        self.assertIn("ataxia_cerebellar_cases", sources)
        self.assertIn("cerebellum_ataxias", sources)
        self.assertIn("journal_neuropsychiatry", sources)
        self.assertIn("plos_one", sources)
        self.assertIn("peerj", sources)
        self.assertIn("epilepsia_open", sources)
        self.assertIn("molecular_case_studies", sources)
        self.assertIn("orphanet_journal_rare_diseases", sources)
        self.assertIn("journal_peripheral_nervous_system", sources)
        self.assertIn("journal_intensive_care", sources)
        self.assertIn("stroke_vin", sources)
        self.assertIn("radiology_case_reports", sources)
        self.assertIn('"JACC. Case Reports"[journal]', sources["jacc_case_reports"].extra or "")
        self.assertIn(
            '"Stroke and Vascular Neurology"[journal]',
            sources["stroke_vascular_neurology"].extra or "",
        )
        self.assertIn('"BMJ Neurology Open"[journal]', sources["bmj_neurology_open"].extra or "")
        self.assertIn('"Neurology"[journal]', sources["neurology_main"].extra or "")
        self.assertIn('"Epilepsia Open"[journal]', sources["epilepsia_open"].extra or "")
        self.assertIn('"Frontiers in Neuroscience"[journal]', sources["frontiers_neuroscience"].extra or "")
        self.assertIn('"Frontiers in Psychiatry"[journal]', sources["frontiers_psychiatry"].extra or "")


if __name__ == "__main__":
    unittest.main()
