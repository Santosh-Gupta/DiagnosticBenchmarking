"""PMC source registry for neurology text case mining."""

from __future__ import annotations

from dataclasses import dataclass

from .queries import build_pmc_query


@dataclass(frozen=True)
class PmcSource:
    key: str
    label: str
    topic: str = "neurology"
    extra: str | None = None
    text_only: bool = True
    notes: str = ""


PMC_SOURCES: dict[str, PmcSource] = {
    "broad_neurology_cases": PmcSource(
        key="broad_neurology_cases",
        label="Broad PMC neurology case reports",
        notes="High-recall baseline over PMC Open Access case-like articles.",
    ),
    "neuropsychiatry_cases": PmcSource(
        key="neuropsychiatry_cases",
        label="Neuropsychiatry case reports",
        topic="neuropsychiatry",
        notes="Behavioral neurology, autoimmune encephalitis, catatonia, psychosis mimics.",
    ),
    "diagnostic_reasoning_differential_cases": PmcSource(
        key="diagnostic_reasoning_differential_cases",
        label="Reasoning-dense cases with differential diagnosis language",
        extra='(differential[All Fields] OR "differential diagnosis"[All Fields] OR "differential diagnoses"[All Fields])',
        notes="Inspired by MedCaseReasoning's differential-diagnosis filter to enrich for cases with explicit reasoning.",
    ),
    "child_neurology_cases": PmcSource(
        key="child_neurology_cases",
        label="Child and pediatric neurology cases",
        extra='("child"[Title/Abstract] OR pediatric[Title/Abstract] OR paediatric[Title/Abstract] OR infant[Title/Abstract] OR adolescent[Title/Abstract])',
        notes="Pediatric neurology, neurogenetics, seizure syndromes, developmental regression.",
    ),
    "neurogenetics_cases": PmcSource(
        key="neurogenetics_cases",
        label="Neurogenetics and metabolic neurology cases",
        extra='(neurogenetic*[Title/Abstract] OR leukodystrophy[Title/Abstract] OR mitochondrial[Title/Abstract] OR "inborn error"[Title/Abstract] OR "hereditary spastic paraplegia"[Title/Abstract])',
        notes="Rare/genetic diagnostic reasoning subset.",
    ),
    "neuroimmunology_cases": PmcSource(
        key="neuroimmunology_cases",
        label="Neuroimmunology and demyelination cases",
        extra='("autoimmune encephalitis"[Title/Abstract] OR MOGAD[Title/Abstract] OR NMOSD[Title/Abstract] OR "multiple sclerosis"[Title/Abstract] OR demyelinating[Title/Abstract] OR paraneoplastic[Title/Abstract])',
        notes="Autoimmune, paraneoplastic, MS/NMOSD/MOGAD and inflammatory mimics.",
    ),
    "neuro_oncology_cases": PmcSource(
        key="neuro_oncology_cases",
        label="Neuro-oncology diagnostic cases",
        extra='("brain tumor"[Title/Abstract] OR glioma[Title/Abstract] OR glioblastoma[Title/Abstract] OR lymphoma[Title/Abstract] OR meningioma[Title/Abstract] OR leptomeningeal[Title/Abstract] OR "brain metastasis"[Title/Abstract] OR "spinal tumor"[Title/Abstract])',
        notes="Tumor mimics, CNS lymphoma vs demyelination, glioma vs encephalitis, leptomeningeal disease.",
    ),
    "neuromuscular_cases": PmcSource(
        key="neuromuscular_cases",
        label="Neuromuscular cases",
        extra='(neuromuscular[Title/Abstract] OR neuropathy[Title/Abstract] OR myopathy[Title/Abstract] OR myasthenia[Title/Abstract] OR "motor neuron"[Title/Abstract] OR radiculopathy[Title/Abstract] OR plexopathy[Title/Abstract])',
        notes="Peripheral nerve, muscle, NMJ, ALS mimics, radiculoplexus syndromes.",
    ),
    "movement_disorder_cases": PmcSource(
        key="movement_disorder_cases",
        label="Movement disorder cases",
        extra='("movement disorder"[Title/Abstract] OR parkinson*[Title/Abstract] OR dystonia[Title/Abstract] OR chorea[Title/Abstract] OR tremor[Title/Abstract] OR ataxia[Title/Abstract])',
        notes="Movement phenomenology in text-only form.",
    ),
    "ataxia_cerebellar_cases": PmcSource(
        key="ataxia_cerebellar_cases",
        label="Ataxia and cerebellar disorder cases",
        extra='(ataxia[Title/Abstract] OR cerebellar[Title/Abstract] OR "spinocerebellar ataxia"[Title/Abstract] OR "gluten ataxia"[Title/Abstract] OR "episodic ataxia"[Title/Abstract] OR "paraneoplastic cerebellar"[Title/Abstract])',
        notes="Ataxia, cerebellar degeneration, immune/metabolic/genetic ataxia, and cerebellar mimic cases.",
    ),
    "epilepsy_cases": PmcSource(
        key="epilepsy_cases",
        label="Epilepsy and seizure cases",
        extra='(epilepsy[Title/Abstract] OR seizure[Title/Abstract] OR status epilepticus[Title/Abstract] OR "nonconvulsive"[Title/Abstract])',
        notes="Seizure, spell, encephalopathy, and status epilepticus cases.",
    ),
    "vascular_neurology_cases": PmcSource(
        key="vascular_neurology_cases",
        label="Stroke and vascular neurology cases",
        extra='(stroke[Title/Abstract] OR "cerebral infarction"[Title/Abstract] OR hemorrhage[Title/Abstract] OR vasculitis[Title/Abstract] OR moyamoya[Title/Abstract] OR thrombosis[Title/Abstract])',
        notes="Stroke mechanisms, mimics, vasculitis, vascular malformations.",
    ),
    "neuro_ophthalmology_cases": PmcSource(
        key="neuro_ophthalmology_cases",
        label="Neuro-ophthalmology cases",
        extra='("neuro-ophthalmology"[Title/Abstract] OR neuroophthalmology[Title/Abstract] OR "optic neuritis"[Title/Abstract] OR papilledema[Title/Abstract] OR diplopia[Title/Abstract] OR "visual field"[Title/Abstract] OR "internuclear ophthalmoplegia"[Title/Abstract] OR "cranial nerve palsy"[Title/Abstract])',
        notes="Optic neuropathy, papilledema, diplopia, cranial nerve palsy, visual field, and INO localization cases.",
    ),
    "neuro_otology_cases": PmcSource(
        key="neuro_otology_cases",
        label="Neuro-otology and vestibular cases",
        extra='("neuro-otology"[Title/Abstract] OR neurotology[Title/Abstract] OR vestibular[Title/Abstract] OR vertigo[Title/Abstract] OR dizziness[Title/Abstract] OR "Meniere"[Title/Abstract] OR "HINTS"[Title/Abstract])',
        notes="Central vs peripheral vertigo, vestibular syndromes, HINTS reasoning, and neuro-otology mimics.",
    ),
    "sleep_neurology_cases": PmcSource(
        key="sleep_neurology_cases",
        label="Sleep neurology cases",
        extra='(narcolepsy[Title/Abstract] OR parasomnia[Title/Abstract] OR "REM sleep behavior disorder"[Title/Abstract] OR "sleep-related hyperkinetic"[Title/Abstract] OR "nocturnal seizure"[Title/Abstract] OR "sleep disorder"[Title/Abstract])',
        notes="Narcolepsy, parasomnias mimicking seizures, sleep-related hyperkinetic movements, and RBD.",
    ),
    "headache_and_pain_cases": PmcSource(
        key="headache_and_pain_cases",
        label="Headache and neurologic pain cases",
        extra='(headache[Title/Abstract] OR migraine[Title/Abstract] OR "cluster headache"[Title/Abstract] OR "trigeminal autonomic cephalalgia"[Title/Abstract] OR RCVS[Title/Abstract] OR "reversible cerebral vasoconstriction"[Title/Abstract] OR "cerebral venous sinus thrombosis"[Title/Abstract] OR CVST[Title/Abstract] OR "giant cell arteritis"[Title/Abstract])',
        notes="Secondary headache syndromes, RCVS, CVST, TACs, GCA, and headache mimics.",
    ),
    "neuro_critical_care_cases": PmcSource(
        key="neuro_critical_care_cases",
        label="Neurocritical care cases",
        extra='("neurocritical care"[Title/Abstract] OR coma[Title/Abstract] OR "status epilepticus"[Title/Abstract] OR "brain death"[Title/Abstract] OR dysautonomia[Title/Abstract] OR "neurogenic pulmonary edema"[Title/Abstract] OR "hypoxic encephalopathy"[Title/Abstract])',
        notes="Coma, status epilepticus, brain-death mimics, dysautonomia, and critical-care neurology.",
    ),
    "bmc_neurology": PmcSource(
        key="bmc_neurology",
        label="BMC Neurology",
        extra='"BMC Neurology"[journal]',
    ),
    "neurology_main": PmcSource(
        key="neurology_main",
        label="Neurology main journal",
        extra='"Neurology"[journal]',
        notes="Main AAN journal; target Clinical Reasoning, Resident & Fellow, Mystery Case, and other diagnostic sections. Mostly locked/pointer-only unless article-level OA permits.",
    ),
    "jcs_cases": PmcSource(
        key="jcs_cases",
        label="Journal of Clinical Sleep Medicine clinical challenges",
        extra='("Journal of Clinical Sleep Medicine"[journal] OR "J Clin Sleep Med"[journal])',
        notes="Sleep neurology, parasomnias mimicking seizures, narcolepsy, and sleep-related hyperkinetic movement cases; article-level rights required.",
    ),
    "journal_medical_case_reports": PmcSource(
        key="journal_medical_case_reports",
        label="Journal of Medical Case Reports neurology subset",
        extra='"Journal of Medical Case Reports"[journal]',
    ),
    "plos_one": PmcSource(
        key="plos_one",
        label="PLOS ONE neurology subset",
        extra='("PLOS ONE"[journal] OR "PLoS One"[journal])',
        notes="Large CC BY OA mega-journal; filter aggressively for true case reports and neurology-primary narratives.",
    ),
    "peerj": PmcSource(
        key="peerj",
        label="PeerJ neurology subset",
        extra='"PeerJ"[journal]',
        notes="Broad OA journal; smaller neurology case yield but useful as a license-clean supplemental source.",
    ),
    "scientific_reports_neurology": PmcSource(
        key="scientific_reports_neurology",
        label="Scientific Reports neurology subset",
        extra='"Scientific Reports"[journal]',
        notes="Large Nature Portfolio OA journal; filter to clinical case reports/series and avoid basic neuroscience-only records.",
    ),
    "frontiers_neurology": PmcSource(
        key="frontiers_neurology",
        label="Frontiers in Neurology",
        extra='"Frontiers in Neurology"[journal]',
    ),
    "frontiers_neurology_cpc_collection": PmcSource(
        key="frontiers_neurology_cpc_collection",
        label="Frontiers in Neurology CPC and case-report collections",
        extra='("Frontiers in Neurology"[journal] AND ("clinicopathologic"[Title/Abstract] OR "clinicopathological"[Title/Abstract] OR "clinical pathological conference"[Title/Abstract] OR CPC[Title/Abstract] OR "case report"[Title/Abstract]))',
        notes="Reasoning-format subset of Frontiers in Neurology, including CPC-style and dedicated case-report collections.",
    ),
    "frontiers_neuroscience": PmcSource(
        key="frontiers_neuroscience",
        label="Frontiers in Neuroscience",
        extra='"Frontiers in Neuroscience"[journal]',
        notes="Frontiers CC BY neuroscience source; many articles are indexed in PMC, but newer case reports may require direct Frontiers XML/PDF acquisition.",
    ),
    "frontiers_neuroscience_case_reports": PmcSource(
        key="frontiers_neuroscience_case_reports",
        label="Frontiers in Neuroscience case reports",
        extra='("Frontiers in Neuroscience"[journal] AND ("case report"[Title/Abstract] OR "case presentation"[Title/Abstract]))',
        notes="Clinical and neuropsychiatric case reports in Frontiers in Neuroscience; use article-level license metadata and direct Frontiers XML when absent from PMC.",
    ),
    "frontiers_human_neuroscience": PmcSource(
        key="frontiers_human_neuroscience",
        label="Frontiers in Human Neuroscience",
        extra='"Frontiers in Human Neuroscience"[journal]',
        notes="Human cognitive/behavioral neuroscience source; filter for clinically grounded neuropsychiatric case material.",
    ),
    "frontiers_aging_neuroscience": PmcSource(
        key="frontiers_aging_neuroscience",
        label="Frontiers in Aging Neuroscience",
        extra='"Frontiers in Aging Neuroscience"[journal]',
        notes="Dementia, cognitive decline, and neurodegeneration source; useful for neurologic-psychiatric boundary cases after clinical-case filtering.",
    ),
    "frontiers_psychiatry": PmcSource(
        key="frontiers_psychiatry",
        label="Frontiers in Psychiatry",
        extra='"Frontiers in Psychiatry"[journal]',
        notes="Psychiatry and neuropsychiatry source; include case reports and clinically grounded diagnostic-mimic articles with article-level CC BY confirmation.",
    ),
    "case_reports_neurological_medicine": PmcSource(
        key="case_reports_neurological_medicine",
        label="Case Reports in Neurological Medicine",
        extra='"Case Reports in Neurological Medicine"[journal]',
    ),
    "case_reports_neurology": PmcSource(
        key="case_reports_neurology",
        label="Case Reports in Neurology",
        extra='"Case Reports in Neurology"[journal]',
        notes="Mostly noncommercial licenses in PMC; use noncommercial/all_cc profiles for benchmark-only pools.",
    ),
    "clinical_case_reports": PmcSource(
        key="clinical_case_reports",
        label="Clinical Case Reports neurology subset",
        extra='"Clinical Case Reports"[journal]',
        notes="High-volume PMC case-report journal from MedCaseReasoning top-source analysis.",
    ),
    "case_reports_in_medicine": PmcSource(
        key="case_reports_in_medicine",
        label="Case Reports in Medicine neurology subset",
        extra='"Case Reports in Medicine"[journal]',
        notes="General case-report reservoir filtered to neurology.",
    ),
    "medicine_baltimore": PmcSource(
        key="medicine_baltimore",
        label="Medicine (Baltimore) neurology subset",
        extra='("Medicine (Baltimore)"[journal] OR "Medicine"[journal])',
        notes="Very high-volume PMC case-report reservoir; filter aggressively for quality and neurology-primary cases.",
    ),
    "american_journal_case_reports": PmcSource(
        key="american_journal_case_reports",
        label="American Journal of Case Reports neurology subset",
        extra='"American Journal of Case Reports"[journal]',
        notes="High-volume structured case-report source with substantial neurology subset.",
    ),
    "archives_clinical_medical_case_reports": PmcSource(
        key="archives_clinical_medical_case_reports",
        label="Archives of Clinical and Medical Case Reports neurology subset",
        extra='("Archives of Clinical and Medical Case Reports"[journal] OR "Arch Clin Med Case Rep"[journal])',
        notes="Broad OA case-report reservoir with neurology/neuroscience material; use as quality-filter-heavy parser stress-test source.",
    ),
    "asploro_biomedical_case_reports": PmcSource(
        key="asploro_biomedical_case_reports",
        label="Asploro Journal of Biomedical and Clinical Case Reports neurology subset",
        extra='("Asploro Journal of Biomedical and Clinical Case Reports"[journal] OR "Asploro J Biomed Clin Case Rep"[journal])',
        notes="Broad OA case-report reservoir that accepts neurology cases; likely low PMC yield and quality-filter-heavy.",
    ),
    "internal_medicine_japan": PmcSource(
        key="internal_medicine_japan",
        label="Internal Medicine Japan neurology subset",
        extra='("Internal Medicine"[journal] OR "Internal Medicine (Tokyo, Japan)"[journal] OR "Intern Med"[journal])',
        notes="Detailed Japanese internal medicine case reports; often valuable for rare neurologic, neurogenetic, and neuromuscular presentations.",
    ),
    "sage_open_medical_case_reports": PmcSource(
        key="sage_open_medical_case_reports",
        label="SAGE Open Medical Case Reports neurology subset",
        extra='"SAGE Open Medical Case Reports"[journal]',
        notes="General case-report reservoir filtered to neurology.",
    ),
    "international_medical_case_reports_journal": PmcSource(
        key="international_medical_case_reports_journal",
        label="International Medical Case Reports Journal neurology subset",
        extra='"International Medical Case Reports Journal"[journal]',
        notes="Dove Press OA case-report reservoir; article-level license and site terms required.",
    ),
    "clinical_medicine_insights_case_reports": PmcSource(
        key="clinical_medicine_insights_case_reports",
        label="Clinical Medicine Insights: Case Reports neurology subset",
        extra='"Clinical Medicine Insights. Case Reports"[journal]',
        notes="SAGE OA case-report reservoir with structured case presentations.",
    ),
    "jim_high_impact_case_reports": PmcSource(
        key="jim_high_impact_case_reports",
        label="Journal of Investigative Medicine High Impact Case Reports neurology subset",
        extra='("Journal of Investigative Medicine High Impact Case Reports"[journal] OR "J Investig Med High Impact Case Rep"[journal])',
        notes="SAGE OA case-report reservoir; filter to neurology and diagnostic reasoning.",
    ),
    "annals_internal_medicine_clinical_cases": PmcSource(
        key="annals_internal_medicine_clinical_cases",
        label="Annals of Internal Medicine: Clinical Cases neurology subset",
        extra='"Annals of Internal Medicine. Clinical Cases"[journal]',
        notes="Open-access ACP/AHA clinical case journal; useful for neurologic mimics and general diagnostic controls.",
    ),
    "radiology_case_reports": PmcSource(
        key="radiology_case_reports",
        label="Radiology Case Reports neurology subset",
        extra='"Radiology Case Reports"[journal]',
        text_only=False,
        notes="Imaging-first OA case-report reservoir; keep outside default text-only phase unless using licensed clinical text/radiology-report narratives.",
    ),
    "international_journal_surgery_case_reports": PmcSource(
        key="international_journal_surgery_case_reports",
        label="International Journal of Surgery Case Reports neurology subset",
        extra='"International Journal of Surgery Case Reports"[journal]',
        notes="Useful for neurosurgical, spine, vascular, and neurologic mimic cases after filtering.",
    ),
    "journal_surgical_case_reports": PmcSource(
        key="journal_surgical_case_reports",
        label="Journal of Surgical Case Reports neurology subset",
        extra='"Journal of Surgical Case Reports"[journal]',
        notes="Useful for neurosurgical, spine, vascular, and neurologic mimic cases after filtering.",
    ),
    "european_heart_journal_case_reports": PmcSource(
        key="european_heart_journal_case_reports",
        label="European Heart Journal - Case Reports neurology subset",
        extra='"European Heart Journal. Case Reports"[journal]',
        notes="Possible stroke, syncope, embolic, and neuro-cardiology overlap cases.",
    ),
    "jacc_case_reports": PmcSource(
        key="jacc_case_reports",
        label="JACC: Case Reports neurology subset",
        extra='"JACC. Case Reports"[journal]',
        notes="Cardioembolic stroke, syncope, arrhythmia, myocarditis, and neuro-cardiology overlap.",
    ),
    "heart_rhythm_case_reports": PmcSource(
        key="heart_rhythm_case_reports",
        label="HeartRhythm Case Reports neurology subset",
        extra='"HeartRhythm Case Reports"[journal]',
        notes="Syncope, arrhythmia, channelopathy, and neuro-cardiology overlap.",
    ),
    "acg_case_reports_journal": PmcSource(
        key="acg_case_reports_journal",
        label="ACG Case Reports Journal neurology subset",
        extra='"ACG Case Reports Journal"[journal]',
        notes="GI/metabolic/toxic and neuro-gastroenterology overlap after neurology filtering.",
    ),
    "urology_case_reports": PmcSource(
        key="urology_case_reports",
        label="Urology Case Reports neurology subset",
        extra='"Urology Case Reports"[journal]',
        notes="Spine, autonomic, paraneoplastic, infectious, and neurologic mimic overlap.",
    ),
    "case_reports_urology": PmcSource(
        key="case_reports_urology",
        label="Case Reports in Urology neurology subset",
        extra='"Case Reports in Urology"[journal]',
        notes="Autonomic, spinal, paraneoplastic, infectious, and neurologic mimic overlap.",
    ),
    "respirology_case_reports": PmcSource(
        key="respirology_case_reports",
        label="Respirology Case Reports neurology subset",
        extra='"Respirology Case Reports"[journal]',
        notes="Hypoxia, neuromuscular respiratory failure, infection, and ICU-neurology overlap.",
    ),
    "respiratory_medicine_case_reports": PmcSource(
        key="respiratory_medicine_case_reports",
        label="Respiratory Medicine Case Reports neurology subset",
        extra='"Respiratory Medicine Case Reports"[journal]',
        notes="Pulmonary/critical-care overlap, hypoxia, embolic disease, infection, and neurologic mimics.",
    ),
    "jaad_case_reports": PmcSource(
        key="jaad_case_reports",
        label="JAAD Case Reports neurology subset",
        extra='"JAAD Case Reports"[journal]',
        notes="Neurocutaneous, autoimmune, infectious, and paraneoplastic overlap.",
    ),
    "american_journal_ophthalmology_case_reports": PmcSource(
        key="american_journal_ophthalmology_case_reports",
        label="American Journal of Ophthalmology Case Reports neurology subset",
        extra='"American Journal of Ophthalmology Case Reports"[journal]',
        notes="Neuro-ophthalmology, optic neuritis mimics, cranial neuropathy, and visual pathway cases.",
    ),
    "ophthalmology_science": PmcSource(
        key="ophthalmology_science",
        label="Ophthalmology Science neuro-ophthalmology subset",
        extra='"Ophthalmology Science"[journal]',
        notes="Neuro-ophthalmology and retinal/optic pathway overlap after filtering.",
    ),
    "idcases": PmcSource(
        key="idcases",
        label="IDCases neurology subset",
        extra='"IDCases"[journal]',
        notes="Neuroinfectious disease, encephalitis, meningitis, and infectious mimics.",
    ),
    "emerging_infectious_diseases": PmcSource(
        key="emerging_infectious_diseases",
        label="Emerging Infectious Diseases neurology subset",
        extra='"Emerging Infectious Diseases"[journal]',
        notes="CDC infectious disease cases with neurologic complications, outbreak context, or neuroinfectious presentations.",
    ),
    "clinical_infection_practice": PmcSource(
        key="clinical_infection_practice",
        label="Clinical Infection in Practice neurology subset",
        extra='"Clinical Infection in Practice"[journal]',
        notes="Clinical infectious disease cases and neuroinfectious overlap.",
    ),
    "acr_open_rheumatology": PmcSource(
        key="acr_open_rheumatology",
        label="ACR Open Rheumatology neurology subset",
        extra='"ACR Open Rheumatology"[journal]',
        notes="Vasculitis, lupus, autoimmune, and neurologic rheumatology overlap.",
    ),
    "bmj_case_reports": PmcSource(
        key="bmj_case_reports",
        label="BMJ Case Reports neurology subset",
        extra='"BMJ Case Reports"[journal]',
        notes="PMC license varies by article; training profile yields only derivative-friendly CC articles.",
    ),
    "cureus_neurology": PmcSource(
        key="cureus_neurology",
        label="Cureus neurology subset",
        extra='"Cureus"[journal]',
        notes="Large volume, quality variable; later ranking/filtering needed.",
    ),
    "oxford_medical_case_reports": PmcSource(
        key="oxford_medical_case_reports",
        label="Oxford Medical Case Reports neurology subset",
        extra='"Oxford Medical Case Reports"[journal]',
    ),
    "the_neurohospitalist": PmcSource(
        key="the_neurohospitalist",
        label="The Neurohospitalist",
        extra='"The Neurohospitalist"[journal]',
        notes="Small PMC yield under derivative-friendly licenses, but high topical fit.",
    ),
    "practical_neurology": PmcSource(
        key="practical_neurology",
        label="Practical Neurology neurology subset",
        extra='"Practical Neurology"[journal]',
        notes="Small PMC yield; many high-value challenge articles are not OA/training-compatible.",
    ),
    "surgical_neurology_international": PmcSource(
        key="surgical_neurology_international",
        label="Surgical Neurology International",
        extra='"Surgical Neurology International"[journal]',
        notes="Mostly neurosurgical/spine; use downstream quality filters for diagnostic-reasoning value.",
    ),
    "annals_clinical_translational_neurology": PmcSource(
        key="annals_clinical_translational_neurology",
        label="Annals of Clinical and Translational Neurology",
        extra='"Annals of Clinical and Translational Neurology"[journal]',
    ),
    "bmj_neurology_open": PmcSource(
        key="bmj_neurology_open",
        label="BMJ Neurology Open",
        extra='("BMJ Neurology Open"[journal] OR "BMJ Neurol Open"[journal])',
        notes="Fully open-access BMJ neurology journal with clinical neurology and neuroscience content; article-level CC BY vs CC BY-NC split still required.",
    ),
    "neurology_open_access": PmcSource(
        key="neurology_open_access",
        label="Neurology Open Access",
        extra='("Neurology Open Access"[journal] OR "Neurol Open Access"[journal])',
        notes="New AAN fully open-access neurology journal with case reports; check early indexing and article-level license before treating as training-compatible.",
    ),
    "open_neurology_journal": PmcSource(
        key="open_neurology_journal",
        label="The Open Neurology Journal",
        extra='("Open Neurology Journal"[journal] OR "Open Neurol J"[journal])',
        notes="Broad Bentham OA neurology journal with case reports; quality-variable volume supplement.",
    ),
    "journal_neurology_research": PmcSource(
        key="journal_neurology_research",
        label="Journal of Neurology Research",
        extra='("Journal of Neurology Research"[journal] OR "J Neurol Res"[journal])',
        notes="OA neurology case reports and case series; supplemental source for vascular, neuromuscular, and rare presentations.",
    ),
    "neurocase": PmcSource(
        key="neurocase",
        label="Neurocase",
        topic="neuropsychiatry",
        extra='"Neurocase"[journal]',
        notes="Case studies in neuropsychology, neuropsychiatry, and behavioral neurology; mixed access, so expect pointer-only or article-level OA subsets.",
    ),
    "journal_neuropsychiatry": PmcSource(
        key="journal_neuropsychiatry",
        label="Journal of Neuropsychiatry and Clinical Neurosciences",
        topic="neuropsychiatry",
        extra='"Journal of Neuropsychiatry and Clinical Neurosciences"[journal]',
        notes="APA neuropsychiatry journal with case reports and case conferences; mixed access/pointer-only unless article OA permits.",
    ),
    "cerebellum_ataxias": PmcSource(
        key="cerebellum_ataxias",
        label="Cerebellum & Ataxias",
        extra='"Cerebellum & Ataxias"[journal]',
        notes="BMC OA archive dedicated to cerebellar disorders and ataxias; useful for genetic, immune, metabolic, and paraneoplastic ataxia cases.",
    ),
    "seminars_neurology": PmcSource(
        key="seminars_neurology",
        label="Seminars in Neurology",
        extra='"Seminars in Neurology"[journal]',
        notes="Themed review journal with occasional challenging-case issues; likely low PMC OA yield and mostly pointer-only unless article license permits.",
    ),
    "neurologic_clinics": PmcSource(
        key="neurologic_clinics",
        label="Neurologic Clinics",
        extra='"Neurologic Clinics"[journal]',
        notes="Themed clinical issues and case-study volumes; mostly publisher-controlled, but keep as a low-yield PMC/pointer lead.",
    ),
    "movement_disorders_clinical_practice": PmcSource(
        key="movement_disorders_clinical_practice",
        label="Movement Disorders Clinical Practice",
        extra='"Movement Disorders Clinical Practice"[journal]',
        notes="Movement-disorder case reports and clinical reasoning; article licenses vary.",
    ),
    "mov_disord_vignettes": PmcSource(
        key="mov_disord_vignettes",
        label="Movement Disorders clinical vignettes",
        extra='("Movement Disorders"[journal] AND ("clinical vignette"[Title/Abstract] OR "clinical vignettes"[Title/Abstract] OR "video case"[Title/Abstract] OR "case report"[Title/Abstract]))',
        notes="Movement Disorders clinical vignettes and video cases; high diagnostic value but many items are video/image-dependent or publisher-controlled.",
    ),
    "tremor_hyperkinetic_movements": PmcSource(
        key="tremor_hyperkinetic_movements",
        label="Tremor and Other Hyperkinetic Movements",
        extra='"Tremor and Other Hyperkinetic Movements"[journal]',
        notes="Open-access movement-disorders journal with tremor, dystonia, chorea, myoclonus, and other hyperkinetic movement cases.",
    ),
    "epilepsy_behavior_case_reports": PmcSource(
        key="epilepsy_behavior_case_reports",
        label="Epilepsy & Behavior Case Reports / Reports",
        extra='("Epilepsy & Behavior Case Reports"[journal] OR "Epilepsy & Behavior Reports"[journal])',
        notes="Dedicated epilepsy case-report source; often NC/ND, so separate license review is important.",
    ),
    "epilepsy_behavior_reports": PmcSource(
        key="epilepsy_behavior_reports",
        label="Epilepsy & Behavior Reports",
        extra='"Epilepsy & Behavior Reports"[journal]',
        notes="Alias/targeted source for the OA Epilepsy & Behavior case-report companion.",
    ),
    "epilepsy_behavior": PmcSource(
        key="epilepsy_behavior",
        label="Epilepsy & Behavior",
        extra='"Epilepsy & Behavior"[journal]',
        notes="High-volume epilepsy/seizure semiology and behavior journal; mixed access and article-level rights required.",
    ),
    "seizure_journal": PmcSource(
        key="seizure_journal",
        label="Seizure: European Journal of Epilepsy",
        extra='("Seizure"[journal] OR "Seizure: European Journal of Epilepsy"[journal])',
        notes="Seizure mimics, status epilepticus, autoimmune encephalitis with seizures, and epilepsy surgery evaluation; mixed access.",
    ),
    "epilepsia_open": PmcSource(
        key="epilepsia_open",
        label="Epilepsia Open",
        extra='"Epilepsia Open"[journal]',
        notes="ILAE official open-access epilepsy journal; strong source for refractory epilepsy, status epilepticus, and neurophysiology cases.",
    ),
    "multiple_sclerosis_related_disorders": PmcSource(
        key="multiple_sclerosis_related_disorders",
        label="Multiple Sclerosis and Related Disorders",
        extra='"Multiple Sclerosis and Related Disorders"[journal]',
        notes="Demyelinating disease, MOGAD/NMOSD/MS mimics, and neuroimmunology overlap.",
    ),
    "ms_related_disorders": PmcSource(
        key="ms_related_disorders",
        label="Multiple Sclerosis and Related Disorders alias",
        extra='"Multiple Sclerosis and Related Disorders"[journal]',
        notes="Alias for external source-key lists; same target as multiple_sclerosis_related_disorders.",
    ),
    "journal_neuroimmunology": PmcSource(
        key="journal_neuroimmunology",
        label="Journal of Neuroimmunology",
        extra='"Journal of Neuroimmunology"[journal]',
        notes="MOGAD, NMOSD, MS mimics, paraneoplastic disease, neurosarcoidosis, and neuroimmune diagnostic dilemmas; mixed access.",
    ),
    "muscle_nerve": PmcSource(
        key="muscle_nerve",
        label="Muscle & Nerve",
        extra='"Muscle & Nerve"[journal]',
        notes="Neuromuscular cases: neuropathy, myopathy, NMJ, motor neuron mimics. Target Cases of the Month when available.",
    ),
    "journal_peripheral_nervous_system": PmcSource(
        key="journal_peripheral_nervous_system",
        label="Journal of the Peripheral Nervous System",
        extra='("Journal of the Peripheral Nervous System"[journal] OR "J Peripher Nerv Syst"[journal])',
        notes="Peripheral neuropathy, CIDP, hereditary neuropathy, radiculoplexus syndromes, toxic neuropathy, and ALS mimics; mixed access.",
    ),
    "journal_neuromuscular_diseases": PmcSource(
        key="journal_neuromuscular_diseases",
        label="Journal of Neuromuscular Diseases",
        extra='"Journal of Neuromuscular Diseases"[journal]',
        notes="Congenital myopathy, myasthenia, muscular dystrophy, metabolic myopathy, and neuromuscular disease series; mixed access.",
    ),
    "als_ftd": PmcSource(
        key="als_ftd",
        label="Amyotrophic Lateral Sclerosis and Frontotemporal Degeneration",
        extra='("Amyotrophic Lateral Sclerosis and Frontotemporal Degeneration"[journal] OR "Amyotroph Lateral Scler Frontotemporal Degener"[journal])',
        notes="ALS/FTD phenocopies, genetic mimics, motor neuron disease diagnostic dilemmas; mixed access.",
    ),
    "rrnmf_neuromuscular_journal": PmcSource(
        key="rrnmf_neuromuscular_journal",
        label="RRNMF Neuromuscular Journal",
        extra='"RRNMF Neuromuscular Journal"[journal]',
        notes="Neuromuscular-focused open journal with clinic and case reports; check indexing/yield and article license.",
    ),
    "jimd_reports": PmcSource(
        key="jimd_reports",
        label="JIMD Reports",
        extra='"JIMD Reports"[journal]',
        notes="High-value metabolic and genetic disease case reports for neurogenetics and mitochondrial/inborn-error subsets.",
    ),
    "neuropediatrics": PmcSource(
        key="neuropediatrics",
        label="Neuropediatrics",
        extra='"Neuropediatrics"[journal]',
        notes="Pediatric neurology, developmental, neurogenetic, and seizure cases.",
    ),
    "pediatric_neurology": PmcSource(
        key="pediatric_neurology",
        label="Pediatric Neurology",
        extra='"Pediatric Neurology"[journal]',
        notes="Child neurology cases and short clinical reports; hybrid/access varies.",
    ),
    "journal_child_neurology": PmcSource(
        key="journal_child_neurology",
        label="Journal of Child Neurology",
        extra='("Journal of Child Neurology"[journal] OR "J Child Neurol"[journal])',
        notes="Child neurology, developmental regression, pediatric epilepsy, metabolic, and neurogenetic case material; mixed access.",
    ),
    "brain_and_development": PmcSource(
        key="brain_and_development",
        label="Brain and Development",
        extra='("Brain & Development"[journal] OR "Brain and Development"[journal])',
        notes="Japanese Society of Child Neurology journal; pediatric epilepsy, regression, metabolic and neurogenetic cases; hybrid/mixed access.",
    ),
    "journal_pediatric_neurosciences": PmcSource(
        key="journal_pediatric_neurosciences",
        label="Journal of Pediatric Neurosciences",
        extra='"Journal of Pediatric Neurosciences"[journal]',
        notes="Pediatric neurology/neurosurgery cases, including infectious, developmental, seizure, and neurosurgical mimics; article-level license required.",
    ),
    "developmental_medicine_child_neurology": PmcSource(
        key="developmental_medicine_child_neurology",
        label="Developmental Medicine & Child Neurology",
        extra='("Developmental Medicine and Child Neurology"[journal] OR "Dev Med Child Neurol"[journal])',
        notes="Neurodevelopmental disorders, longitudinal diagnostic narratives, and pediatric case series; mixed access.",
    ),
    "neurocritical_care": PmcSource(
        key="neurocritical_care",
        label="Neurocritical Care",
        extra='"Neurocritical Care"[journal]',
        notes="ICU neurology, status epilepticus, stroke, hemorrhage, encephalopathy, and acute diagnostic reasoning.",
    ),
    "journal_intensive_care": PmcSource(
        key="journal_intensive_care",
        label="Journal of Intensive Care neurology subset",
        extra='"Journal of Intensive Care"[journal]',
        notes="BMC OA critical-care journal with neurologic overlap: hypoxic brain injury, septic encephalopathy, PRES, coma, and ICU complications.",
    ),
    "journal_neuro_ophthalmology": PmcSource(
        key="journal_neuro_ophthalmology",
        label="Journal of Neuro-Ophthalmology",
        extra='"Journal of Neuro-Ophthalmology"[journal]',
        notes="Neuro-ophthalmology localization, demyelinating, vascular, compressive, and inflammatory mimics.",
    ),
    "neuro_ophthalmology": PmcSource(
        key="neuro_ophthalmology",
        label="Neuro-Ophthalmology",
        extra='"Neuro-Ophthalmology"[journal]',
        notes="Taylor & Francis neuro-ophthalmology journal; nystagmus, gaze palsy, optic neuropathy, vestibular and visual mimics; mixed access.",
    ),
    "journal_vestibular_research": PmcSource(
        key="journal_vestibular_research",
        label="Journal of Vestibular Research",
        extra='"Journal of Vestibular Research"[journal]',
        notes="Central vertigo, cerebellar stroke mimics, atypical Meniere presentations, and vestibular diagnostic cases; mixed access.",
    ),
    "jnop_cases": PmcSource(
        key="jnop_cases",
        label="Journal of Neuro-Ophthalmology case sections",
        extra='("Journal of Neuro-Ophthalmology"[journal] AND ("clinical challenge"[Title/Abstract] OR "photo essay"[Title/Abstract] OR "case report"[Title/Abstract] OR "case series"[Title/Abstract]))',
        notes="Target Clinical Challenges, Photo Essays, optic neuropathy, cranial nerve palsy, papilledema, and visual localization cases.",
    ),
    "frontiers_neuro_ophthalmology": PmcSource(
        key="frontiers_neuro_ophthalmology",
        label="Frontiers in Neurology neuro-ophthalmology subset",
        extra='("Frontiers in Neurology"[journal] AND ("neuro-ophthalmology"[Title/Abstract] OR neuroophthalmology[Title/Abstract] OR "optic neuritis"[Title/Abstract] OR diplopia[Title/Abstract] OR "cranial nerve"[Title/Abstract]))',
        notes="OA Frontiers neuro-ophthalmology subset for localization-heavy visual pathway and cranial neuropathy cases.",
    ),
    "eneurologicalsci": PmcSource(
        key="eneurologicalsci",
        label="eNeurologicalSci",
        extra='"eNeurologicalSci"[journal]',
        notes="Open-access broad neurology journal with case-like clinical material.",
    ),
    "cerebrovascular_diseases": PmcSource(
        key="cerebrovascular_diseases",
        label="Cerebrovascular Diseases",
        extra='"Cerebrovascular Diseases"[journal]',
        notes="Vascular neurology and stroke diagnostic cases; article licenses vary.",
    ),
    "cerebrovascular_diseases_extra": PmcSource(
        key="cerebrovascular_diseases_extra",
        label="Cerebrovascular Diseases Extra",
        extra='"Cerebrovascular Diseases Extra"[journal]',
        notes="Open-access Karger stroke-medicine sister journal; likely CC BY-NC by default, so split from training pool unless article license permits.",
    ),
    "stroke_vascular_neurology": PmcSource(
        key="stroke_vascular_neurology",
        label="Stroke and Vascular Neurology",
        extra='("Stroke and Vascular Neurology"[journal] OR "Stroke Vasc Neurol"[journal])',
        notes="BMJ open-access vascular neurology journal; mine stroke mechanisms, mimics, vasculitis, and acute neurologic presentations.",
    ),
    "stroke_clinician": PmcSource(
        key="stroke_clinician",
        label="Stroke Clinician",
        extra='"Stroke Clinician"[journal]',
        notes="New open-access neurovascular clinical journal; check PubMed/PMC indexing before expecting automated yield.",
    ),
    "stroke_vin": PmcSource(
        key="stroke_vin",
        label="Stroke: Vascular and Interventional Neurology",
        extra='("Stroke: Vascular and Interventional Neurology"[journal] OR "Stroke Vasc Interv Neurol"[journal])',
        notes="AHA/SVIN open-access vascular/interventional neurology journal; case reports and vascular diagnostic/interventional cases.",
    ),
    "journal_stroke_cerebrovascular": PmcSource(
        key="journal_stroke_cerebrovascular",
        label="Journal of Stroke and Cerebrovascular Diseases",
        extra='("Journal of Stroke and Cerebrovascular Diseases"[journal] OR "J Stroke Cerebrovasc Dis"[journal])',
        notes="Stroke and cerebrovascular case source; full gold OA from 2025 but article-level license remains required.",
    ),
    "headache_journal": PmcSource(
        key="headache_journal",
        label="Headache and Cephalalgia case reports",
        extra='("Headache"[journal] OR "Cephalalgia"[journal] OR "Cephalalgia Reports"[journal])',
        notes="Secondary headache, RCVS, CVST, TACs, GCA, and pain-diagnosis challenge source; article-level rights vary.",
    ),
    "neuropsychiatric_disease_treatment": PmcSource(
        key="neuropsychiatric_disease_treatment",
        label="Neuropsychiatric Disease and Treatment",
        extra='"Neuropsychiatric Disease and Treatment"[journal]',
        notes="Neuropsychiatry, behavioral neurology, epilepsy/psychiatry overlap, and functional-organic boundary cases.",
    ),
    "cognitive_behavioral_neurology": PmcSource(
        key="cognitive_behavioral_neurology",
        label="Cognitive and Behavioral Neurology",
        topic="neuropsychiatry",
        extra='"Cognitive and Behavioral Neurology"[journal]',
        notes="Behavioral neurology, dementia mimics, late-onset neuropsychiatric presentations, and clinicopathologic cases.",
    ),
    "journal_clinical_neurology": PmcSource(
        key="journal_clinical_neurology",
        label="Journal of Clinical Neurology",
        extra='"Journal of Clinical Neurology"[journal]',
        notes="Korean Neurological Association OA/PMC-indexed clinical neurology journal; useful regional neurology source.",
    ),
    "neurology_india": PmcSource(
        key="neurology_india",
        label="Neurology India",
        extra='"Neurology India"[journal]',
        notes="Indian neurology journal with case reports and regional disease presentations; article-level license required.",
    ),
    "annals_indian_academy_neurology": PmcSource(
        key="annals_indian_academy_neurology",
        label="Annals of Indian Academy of Neurology",
        extra='"Annals of Indian Academy of Neurology"[journal]',
        notes="Open-access Indian neurology journal indexed in PMC; likely strong yield for regionally distinctive neurology cases.",
    ),
    "ann_indian_acad_neurol": PmcSource(
        key="ann_indian_acad_neurol",
        label="Annals of Indian Academy of Neurology alias",
        extra='"Annals of Indian Academy of Neurology"[journal]',
        notes="Alias for the regional OA neurology source; useful when following external source-key lists.",
    ),
    "j_clin_neurol_korea": PmcSource(
        key="j_clin_neurol_korea",
        label="Journal of Clinical Neurology Korea alias",
        extra='"Journal of Clinical Neurology"[journal]',
        notes="Alias for Journal of Clinical Neurology (Korean Neurological Association); useful when following external source-key lists.",
    ),
    "arq_neuro_psiq": PmcSource(
        key="arq_neuro_psiq",
        label="Arquivos de Neuro-Psiquiatria",
        extra='("Arquivos de Neuro-Psiquiatria"[journal] OR "Arq Neuropsiquiatr"[journal])',
        notes="Major Latin American open neurology journal; valuable for infectious, tropical, and global neurology cases.",
    ),
    "arquivos_neuro_psiquiatria": PmcSource(
        key="arquivos_neuro_psiquiatria",
        label="Arquivos de Neuro-Psiquiatria alias",
        extra='("Arquivos de Neuro-Psiquiatria"[journal] OR "Arq Neuropsiquiatr"[journal])',
        notes="Alias for external source-key lists; same target as arq_neuro_psiq.",
    ),
    "acta_neurol_scand": PmcSource(
        key="acta_neurol_scand",
        label="Acta Neurologica Scandinavica",
        extra='("Acta Neurologica Scandinavica"[journal] OR "Acta Neurol Scand"[journal])',
        notes="Scandinavian neurology source with case reports and epidemiologic case series; article-level OA/license required.",
    ),
    "canadian_journal_neurological_sciences": PmcSource(
        key="canadian_journal_neurological_sciences",
        label="Canadian Journal of Neurological Sciences",
        extra='"Canadian Journal of Neurological Sciences"[journal]',
        notes="Clinical neuropathological conferences and case reports; mixed access, article-level rights required.",
    ),
    "ame_case_reports": PmcSource(
        key="ame_case_reports",
        label="AME Case Reports neurology subset",
        extra='"AME Case Reports"[journal]',
        notes="Open-access, PMC-indexed cross-specialty case-report journal; many articles are noncommercial/no-derivatives, so license split is important.",
    ),
    "case_reports_clinical_practice": PmcSource(
        key="case_reports_clinical_practice",
        label="Case Reports in Clinical Practice neurology subset",
        extra='"Case Reports in Clinical Practice"[journal]',
        notes="General case-report journal; check PMC indexing, quality, and article-level license before use.",
    ),
    "neuro_oncology": PmcSource(
        key="neuro_oncology",
        label="Neuro-Oncology",
        extra='"Neuro-Oncology"[journal]',
        notes="Neuro-oncology diagnostic overlap, tumor mimics, CNS lymphoma, glioma, and leptomeningeal disease.",
    ),
    "journal_neuro_oncology": PmcSource(
        key="journal_neuro_oncology",
        label="Journal of Neuro-Oncology",
        extra='"Journal of Neuro-Oncology"[journal]',
        notes="CNS tumor cases and tumor-mimic diagnostic overlap.",
    ),
    "neuro_oncology_advances": PmcSource(
        key="neuro_oncology_advances",
        label="Neuro-Oncology Advances",
        extra='"Neuro-Oncology Advances"[journal]',
        notes="Open-access neuro-oncology journal; use for tumor-mimic and neuro-oncology subsets.",
    ),
    "neuro_oncology_practice": PmcSource(
        key="neuro_oncology_practice",
        label="Neuro-Oncology Practice",
        extra='"Neuro-Oncology Practice"[journal]',
        notes="Neuro-oncology practice cases, treatment neurotoxicity, paraneoplastic and tumor-mimic diagnostic challenges; mixed access.",
    ),
    "journal_neuro_oncology_discovery": PmcSource(
        key="journal_neuro_oncology_discovery",
        label="Journal of Neuro-Oncology Discovery",
        extra='"Journal of Neuro-Oncology Discovery"[journal]',
        notes="New full-OA neuro-oncology journal launched in 2026 with case reports; check PubMed/PMC indexing and article licenses.",
    ),
    "case_reports_oncological_medicine": PmcSource(
        key="case_reports_oncological_medicine",
        label="Case Reports in Oncological Medicine neuro-oncology subset",
        extra='"Case Reports in Oncological Medicine"[journal]',
        notes="General oncology case-report reservoir filtered to CNS tumor, paraneoplastic, leptomeningeal, and neurologic complication cases.",
    ),
    "jns_case_lessons": PmcSource(
        key="jns_case_lessons",
        label="Journal of Neurosurgery: Case Lessons",
        extra='("Journal of Neurosurgery. Case Lessons"[journal] OR "J Neurosurg Case Lessons"[journal])',
        notes="Neurosurgery-heavy illustrative cases; useful for tumors, vascular lesions, spine, epilepsy surgery, and neurologic mimics.",
    ),
    "nmc_case_report_journal": PmcSource(
        key="nmc_case_report_journal",
        label="NMC Case Report Journal",
        extra='("NMC Case Report Journal"[journal] OR "NMC Case Rep J"[journal])',
        notes="Open neurosurgical case-report journal; useful for spine, vascular, tumor, and neurosurgical mimic cases after license filtering.",
    ),
    "brain_development_case_reports": PmcSource(
        key="brain_development_case_reports",
        label="Brain and Development Case Reports",
        extra='"Brain and Development Case Reports"[journal]',
        notes="Open-access pediatric neurology case-report journal; check PubMed/PMC indexing and article license before automated import.",
    ),
    "freiburg_neuropathology_case_conference": PmcSource(
        key="freiburg_neuropathology_case_conference",
        label="Freiburg Neuropathology Case Conference",
        extra='("Clinical Neuroradiology"[journal] AND "Freiburg Neuropathology Case Conference"[Title])',
        notes="Structured clinicoradiology-neuropathology cases, often CC BY; text-only phase should avoid image-dependent prompts unless rewritten from licensed clinical text.",
    ),
    "neurology_genetics": PmcSource(
        key="neurology_genetics",
        label="Neurology: Genetics",
        extra='"Neurology: Genetics"[journal]',
    ),
    "orphanet_journal_rare_diseases": PmcSource(
        key="orphanet_journal_rare_diseases",
        label="Orphanet Journal of Rare Diseases",
        extra='"Orphanet Journal of Rare Diseases"[journal]',
        notes="BMC OA rare-disease journal; strong for neurogenetic diagnostic odyssey, leukodystrophy, mitochondrial, and IEM cases.",
    ),
    "orphanet_j_rare_dis": PmcSource(
        key="orphanet_j_rare_dis",
        label="Orphanet Journal of Rare Diseases alias",
        extra='"Orphanet Journal of Rare Diseases"[journal]',
        notes="Alias for external source-key lists; same target as orphanet_journal_rare_diseases.",
    ),
    "molecular_genetics_genomic_medicine": PmcSource(
        key="molecular_genetics_genomic_medicine",
        label="Molecular Genetics & Genomic Medicine",
        extra='("Molecular Genetics & Genomic Medicine"[journal] OR "Mol Genet Genomic Med"[journal])',
        notes="Wiley OA genotype-phenotype and clinical reports; useful for HSP, ataxia, epilepsy genes, and neurodevelopmental disorders.",
    ),
    "molecular_case_studies": PmcSource(
        key="molecular_case_studies",
        label="Cold Spring Harbor Molecular Case Studies",
        extra='("Cold Spring Harbor Molecular Case Studies"[journal] OR "Mol Case Stud"[journal])',
        notes="Open precision-medicine case reports linking variants to phenotypes; strong for monogenic neurologic disease.",
    ),
    "journal_inherited_metabolic_disease": PmcSource(
        key="journal_inherited_metabolic_disease",
        label="Journal of Inherited Metabolic Disease",
        extra='("Journal of Inherited Metabolic Disease"[journal] OR "J Inherit Metab Dis"[journal])',
        notes="Flagship IEM journal with neurologic regression, seizures, movement disorders, and metabolic diagnostic workups; mixed access.",
    ),
    "molecular_genetics_metabolism_reports": PmcSource(
        key="molecular_genetics_metabolism_reports",
        label="Molecular Genetics and Metabolism Reports",
        extra='"Molecular Genetics and Metabolism Reports"[journal]',
        notes="Open-access companion with metabolic case reports and sequence reports; strong for IEM/neurogenetics overlap.",
    ),
    "neurology_neuroimmunology": PmcSource(
        key="neurology_neuroimmunology",
        label="Neurology: Neuroimmunology and Neuroinflammation",
        extra='"Neurology(R) Neuroimmunology & Neuroinflammation"[journal]',
    ),
    "neurology_international": PmcSource(
        key="neurology_international",
        label="Neurology International",
        extra='"Neurology International"[journal]',
    ),
    "behavioural_neurology": PmcSource(
        key="behavioural_neurology",
        label="Behavioural Neurology",
        extra='"Behavioural Neurology"[journal]',
        notes="Cognitive and behavioral neurology subset.",
    ),
    "mayo_clinic_proceedings": PmcSource(
        key="mayo_clinic_proceedings",
        label="Mayo Clinic Proceedings neurology-mimic subset",
        extra='"Mayo Clinic Proceedings"[journal]',
        notes="High-quality clinical vignettes and diagnostic puzzles with neurologic mimics; mostly pointer/mixed access.",
    ),
    "cleveland_clinic_jm": PmcSource(
        key="cleveland_clinic_jm",
        label="Cleveland Clinic Journal of Medicine neurology-mimic subset",
        extra='"Cleveland Clinic Journal of Medicine"[journal]',
        notes="Clinical challenges, board review, and case-based reviews; mixed access/pointer source.",
    ),
    "baylor_proceedings": PmcSource(
        key="baylor_proceedings",
        label="Proceedings of Baylor University Medical Center neurology subset",
        extra='("Proceedings (Baylor University. Medical Center)"[journal] OR "Proc (Bayl Univ Med Cent)"[journal])',
        notes="General case reports with occasional high-yield neurology and internal-medicine mimic cases.",
    ),
    "yale_jbm": PmcSource(
        key="yale_jbm",
        label="Yale Journal of Biology and Medicine neurology subset",
        extra='"Yale Journal of Biology and Medicine"[journal]',
        notes="Open journal with occasional clinical case reports and images; filter to neurology and clinical reasoning.",
    ),
}


def available_pmc_source_keys() -> tuple[str, ...]:
    return tuple(sorted(PMC_SOURCES))


def get_pmc_sources(keys: list[str] | None = None) -> list[PmcSource]:
    if not keys:
        sources = list(PMC_SOURCES.values())
        return [source for source in sources if source.key != "broad_neurology_cases"] + [
            source for source in sources if source.key == "broad_neurology_cases"
        ]

    sources: list[PmcSource] = []
    for key in keys:
        if key == "all":
            sources.extend(get_pmc_sources(None))
            continue
        try:
            sources.append(PMC_SOURCES[key])
        except KeyError as exc:
            known = ", ".join(available_pmc_source_keys())
            raise ValueError(f"Unknown source {key!r}. Known sources: {known}") from exc
    return _dedupe_sources(sources)


def build_source_query(
    source: PmcSource,
    license_profile: str,
    *,
    include_author_manuscripts: bool = False,
    since: str | None = None,
    until: str | None = None,
    text_only: bool | None = None,
) -> str:
    return build_pmc_query(
        source.topic,
        license_profile,
        include_author_manuscripts=include_author_manuscripts,
        since=since,
        until=until,
        extra=source.extra,
        text_only=source.text_only if text_only is None else text_only,
    )


def _dedupe_sources(sources: list[PmcSource]) -> list[PmcSource]:
    seen: set[str] = set()
    deduped: list[PmcSource] = []
    for source in sources:
        if source.key in seen:
            continue
        seen.add(source.key)
        deduped.append(source)
    return deduped
