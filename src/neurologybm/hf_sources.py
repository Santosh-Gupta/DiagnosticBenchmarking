"""Curated Hugging Face dataset leads for NeurologyBM collection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HfSource:
    key: str
    repo_id: str
    label: str
    priority: int
    use_tier: str
    expected_license: str | None = None
    default_files: tuple[str, ...] = ()
    notes: str = ""


HF_SOURCES: dict[str, HfSource] = {
    "medcase_reasoning": HfSource(
        key="medcase_reasoning",
        repo_id="zou-lab/MedCaseReasoning",
        label="MedCaseReasoning",
        priority=1,
        use_tier="training_candidate_source_audit",
        expected_license="mit",
        default_files=(
            "data/train-00000-of-00001.parquet",
            "data/val-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
        ),
        notes="Diagnostic QA and reasoning traces from open-access case reports; verify source provenance before training release.",
    ),
    "case_report_bench": HfSource(
        key="case_report_bench",
        repo_id="cxyzhang/caseReportBench_ClinicalDenseExtraction_Benchmark",
        label="CaseReportBench",
        priority=1,
        use_tier="training_candidate_source_audit",
        expected_license="cc-by-4.0",
        default_files=("data/train-00000-of-00001.parquet",),
        notes="Expert-annotated dense information extraction benchmark for rare disease case reports.",
    ),
    "openmed_multicare_cases": HfSource(
        key="openmed_multicare_cases",
        repo_id="OpenMed/multicare-cases",
        label="OpenMed MultiCaRe cases",
        priority=1,
        use_tier="training_candidate_source_audit",
        expected_license="cc-by-4.0",
        default_files=("cases.parquet",),
        notes="Parsed case-level MultiCaRe release; useful bootstrap source after neurology and license filtering.",
    ),
    "openmed_multicare_articles": HfSource(
        key="openmed_multicare_articles",
        repo_id="OpenMed/multicare-articles",
        label="OpenMed MultiCaRe articles",
        priority=1,
        use_tier="training_candidate_source_audit",
        expected_license="cc-by-4.0",
        default_files=("articles.parquet",),
        notes="Parsed article-level MultiCaRe release for PMCID/source metadata joins.",
    ),
    "multicare_original": HfSource(
        key="multicare_original",
        repo_id="mauro-nievoff/MultiCaRe_Dataset",
        label="MultiCaRe original zipped release",
        priority=2,
        use_tier="training_candidate_source_audit",
        expected_license="cc-by-4.0",
        notes="Large ZIP shards; prefer OpenMed parquet mirrors for first-pass collection.",
    ),
    "rarebench": HfSource(
        key="rarebench",
        repo_id="chenxz/RareBench",
        label="RareBench",
        priority=1,
        use_tier="external_benchmark_license_review",
        expected_license="apache-2.0",
        default_files=("data.zip",),
        notes="Rare-disease benchmark/calibration source; audit source provenance before training use.",
    ),
    "medmistake": HfSource(
        key="medmistake",
        repo_id="TheLumos/MedicalMistakeBenchmark",
        label="MedMistake / MedicalMistakeBenchmark",
        priority=1,
        use_tier="failure_mining_open_review",
        expected_license="cc-by-4.0",
        default_files=("medmistake-all.json",),
        notes="Pre-mined medical QA items reported to fail current frontier models; filter for neurology/psychiatry and keep expert-validated subset separate.",
    ),
    "diagnosis_arena": HfSource(
        key="diagnosis_arena",
        repo_id="SII-SPIRAL-MED/DiagnosisArena",
        label="DiagnosisArena",
        priority=2,
        use_tier="external_benchmark_failure_mining_license_review",
        expected_license="mit",
        default_files=("data/test-00000-of-00001.parquet",),
        notes="Hard real-world diagnostic cases with published frontier-model failure rates; adapted from publisher-controlled literature, so use for external evaluation/failure-mining metadata until rights are audited.",
    ),
    "medagents_benchmark": HfSource(
        key="medagents_benchmark",
        repo_id="super-dainiu/medagents-benchmark",
        label="MedAgentsBench hard benchmark",
        priority=2,
        use_tier="external_benchmark_failure_mining_license_review",
        expected_license="mit",
        default_files=(
            "MedQA/test_hard-00000-of-00001.parquet",
            "MedMCQA/test_hard-00000-of-00001.parquet",
            "MedBullets/test_hard-00000-of-00001.parquet",
            "PubMedQA/test_hard-00000-of-00001.parquet",
            "MMLU/test_hard-00000-of-00001.parquet",
            "MMLU-Pro/test_hard-00000-of-00001.parquet",
            "AfrimedQA/test_hard-00000-of-00001.parquet",
            "MedExQA/test_hard-00000-of-00001.parquet",
            "MedXpertQA-R/test_hard-00000-of-00001.parquet",
            "MedXpertQA-U/test_hard-00000-of-00001.parquet",
        ),
        notes="Hard multi-source medical-reasoning benchmark for agent evaluation; audit source benchmark licenses before redistribution or training.",
    ),
    "mediq": HfSource(
        key="mediq",
        repo_id="stellalisy/mediQ",
        label="MediQ",
        priority=2,
        use_tier="interactive_failure_mining_license_review",
        expected_license="cc-by-4.0",
        notes="Interactive question-asking medical reasoning benchmark; useful for dynamic history-taking failures.",
    ),
    "medeinst": HfSource(
        key="medeinst",
        repo_id="zhui711/MedEinst",
        label="MedEinst",
        priority=2,
        use_tier="counterfactual_failure_mining_license_review",
        expected_license="cc-by-4.0",
        notes="Counterfactual differential-diagnosis benchmark targeting Einstellung-effect failures; derived-source chain still needs audit.",
    ),
    "rarearena": HfSource(
        key="rarearena",
        repo_id="THUMedInfo/RareArena",
        label="RareArena",
        priority=2,
        use_tier="noncommercial_external_benchmark",
        expected_license="cc-by-nc-sa-4.0",
        notes="Large rare-disease benchmark; useful for neurogenetics calibration but excluded from default training if noncommercial/share-alike terms apply.",
    ),
    "pmc_patients": HfSource(
        key="pmc_patients",
        repo_id="zhengyun21/PMC-Patients",
        label="PMC-Patients",
        priority=2,
        use_tier="noncommercial_source",
        expected_license="cc-by-nc-sa-4.0",
        notes="Patient summaries from PMC case reports; noncommercial/share-alike license, so exclude from default training.",
    ),
    "pmc_patients_bigbio": HfSource(
        key="pmc_patients_bigbio",
        repo_id="bigbio/pmc_patients",
        label="BigBio PMC-Patients loader",
        priority=2,
        use_tier="noncommercial_source",
        expected_license="cc-by-nc-sa-4.0",
        notes="BigBio loader for PMC-Patients; useful for schema reference but requires arbitrary code execution in HF viewer.",
    ),
    "open_patients": HfSource(
        key="open_patients",
        repo_id="ncbi/Open-Patients",
        label="NCBI Open-Patients",
        priority=2,
        use_tier="external_benchmark_license_review",
        notes="Patient similarity/retrieval benchmark related to PMC-Patients; inspect license and provenance.",
    ),
    "ddxplus": HfSource(
        key="ddxplus",
        repo_id="aai530-group6/ddxplus",
        label="DDXPlus",
        priority=3,
        use_tier="synthetic_control",
        expected_license="cc-by-4.0",
        notes="Synthetic differential diagnosis dataset; useful for infrastructure/control tasks, not primary neurology benchmark.",
    ),
    "symcat_triage": HfSource(
        key="symcat_triage",
        repo_id="cristian-untaru/symcat-medical-triage-dataset",
        label="SymCat medical triage dataset",
        priority=3,
        use_tier="calibration_license_review",
        notes="Symptom-disease prior/calibration source; not a case-report corpus.",
    ),
    "meddialog": HfSource(
        key="meddialog",
        repo_id="bigbio/meddialog",
        label="MedDialog BigBio loader",
        priority=3,
        use_tier="history_taking_license_review",
        expected_license="unknown",
        notes="Patient-doctor dialogue dataset for history-taking and agent workflows; not a diagnosis case-report source.",
    ),
    "nejm_image_challenge_scraped": HfSource(
        key="nejm_image_challenge_scraped",
        repo_id="OctoMed/NEJM-Image-Challenge",
        label="NEJM Image Challenge scraped dataset",
        priority=4,
        use_tier="pointer_only_provenance_risk",
        notes="Third-party scraped NEJM content; do not ingest text/images for release or training without permission.",
    ),
    "neural_medbench": HfSource(
        key="neural_medbench",
        repo_id="Reisen301/Neural-MedBench",
        label="Neural-MedBench",
        priority=4,
        use_tier="gated_external_benchmark",
        expected_license="mit",
        notes="Gated multimodal neurology benchmark; track for comparison and leakage audit, not phase-1 text training.",
    ),
    "medical_case_report_corpus": HfSource(
        key="medical_case_report_corpus",
        repo_id="malteos/medical-case-report-corpus",
        label="Medical case report corpus",
        priority=2,
        use_tier="training_candidate_source_audit",
        expected_license="cc-by-3.0",
        notes="Broad case-report corpus surfaced in the broader HF sweep; audit PMCID/DOI provenance and article-level licenses before training use.",
    ),
    "open_clinical_cases_pubmed": HfSource(
        key="open_clinical_cases_pubmed",
        repo_id="rntc/open-clinical-cases-pubmed",
        label="Open clinical cases PubMed",
        priority=2,
        use_tier="training_candidate_source_audit",
        notes="Large open-clinical-cases lead from HF discovery with missing dataset-card license; inspect source/provenance before any download beyond metadata.",
    ),
    "open_clinical_cases_pubmed_comet": HfSource(
        key="open_clinical_cases_pubmed_comet",
        repo_id="rntc/open-clinical-cases-pubmed-comet",
        label="Open clinical cases PubMed COMET",
        priority=2,
        use_tier="training_candidate_source_audit",
        notes="Companion/variant of open clinical cases PubMed; missing dataset-card license in discovery results, so metadata-only until reviewed.",
    ),
    "differential_diagnosis_dataset": HfSource(
        key="differential_diagnosis_dataset",
        repo_id="shuyuej/Differential-Diagnosis-Dataset",
        label="Differential Diagnosis Dataset",
        priority=2,
        use_tier="calibration_license_review",
        expected_license="apache-2.0",
        notes="Differential-diagnosis dataset surfaced in HF sweep; useful for calibration and routing tasks after provenance review.",
    ),
    "online_patient_cases": HfSource(
        key="online_patient_cases",
        repo_id="jchan58/online_patient_cases",
        label="Online patient cases",
        priority=2,
        use_tier="training_candidate_source_audit",
        expected_license="cc-by-4.0",
        notes="Small online patient case dataset surfaced in HF sweep; audit upstream source rights before treating as training-clean.",
    ),
    "nejm_medqa_diagnostic_reasoning": HfSource(
        key="nejm_medqa_diagnostic_reasoning",
        repo_id="katielink/nejm-medqa-diagnostic-reasoning-dataset",
        label="NEJM MedQA diagnostic reasoning dataset",
        priority=4,
        use_tier="pointer_only_provenance_risk",
        expected_license="cc-by-4.0",
        notes="HF card reports CC BY, but NEJM-derived provenance is high risk; keep pointer/evaluation-metadata only unless rights are resolved.",
    ),
    "neurovascular_diagnostic_reasoning": HfSource(
        key="neurovascular_diagnostic_reasoning",
        repo_id="Maitreyajayaraj/neurovascular_diagnostic_reasoning_v3",
        label="Neurovascular diagnostic reasoning v3",
        priority=3,
        use_tier="synthetic_or_generated_review",
        expected_license="apache-2.0",
        notes="Small neurovascular reasoning dataset surfaced in broader HF sweep; likely generated/synthetic, so useful for infrastructure testing after content review.",
    ),
}


def available_hf_source_keys() -> tuple[str, ...]:
    return tuple(sorted(HF_SOURCES))


def get_hf_sources(keys: list[str] | None = None, *, max_priority: int | None = None) -> list[HfSource]:
    if keys:
        sources = []
        for key in keys:
            if key == "all":
                sources.extend(get_hf_sources(None, max_priority=max_priority))
                continue
            try:
                sources.append(HF_SOURCES[key])
            except KeyError as exc:
                known = ", ".join(available_hf_source_keys())
                raise ValueError(f"Unknown Hugging Face source {key!r}. Known sources: {known}") from exc
        return _dedupe_sources(sources)

    sources = sorted(HF_SOURCES.values(), key=lambda source: (source.priority, source.key))
    if max_priority is not None:
        sources = [source for source in sources if source.priority <= max_priority]
    return sources


def _dedupe_sources(sources: list[HfSource]) -> list[HfSource]:
    seen: set[str] = set()
    deduped: list[HfSource] = []
    for source in sources:
        if source.key in seen:
            continue
        seen.add(source.key)
        deduped.append(source)
    return deduped
