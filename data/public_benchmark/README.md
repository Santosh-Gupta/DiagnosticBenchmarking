# Neuro/Psych Diagnostic Case-Challenge Benchmark

`neuro_psych_cases.jsonl` — **349 validated neurology / psychiatry diagnostic case challenges**, each
derived from a single open-access (**CC-BY**) published case report. Each case presents a redacted
clinical vignette; the task is to produce the most likely diagnosis (and best next step).

## Provenance & license

Every case is transformed from one **CC-BY**-licensed article (no NonCommercial or ShareAlike
sources are included). The benchmark text is a transformation of that article; under CC-BY you may use,
share, and adapt it **with attribution**. Per-case attribution is in the `provenance` field
(`doi`, `pmcid`, `title`, `journal`) — cite the original article when using a case. The dataset as a
whole is released under **CC-BY 4.0**.

## Quality control

Every case passed the repository's `validate-cases` determinacy audit (see
`src/neurologybm/case_validation.py` and `docs/next_batch_creation_guide.md`):

- Cases whose gold diagnosis was **not determinable from the prompt alone** were either repaired by
  appending the source's deciding result **verbatim** (the gold diagnosis is never relaxed) or
  excluded. **65** of the 349 cases carry a `mend_provenance` field recording exactly which result was
  added, for full transparency.
- Cases that were under-determined with no sourceable discriminator, whose gold was not a clinical
  diagnosis, whose prompt refuted the gold, or whose gold was finer than the prompt supports
  (`gold_overspecific`) were **dropped** and are not in this file.

## Schema (one JSON object per line)

| field | meaning |
| --- | --- |
| `case_id` | stable id (`transformed_PMC…`) |
| `challenge_prompt` | the redacted clinical vignette + the question |
| `gold_diagnosis` | the reference diagnosis (string; convenience copy of `answer_key.diagnosis`) |
| `answer_key` | `{diagnosis, aliases, etiology, next_management_step, required_findings, optional_findings}` |
| `provenance` | `{doi, pmcid, title, journal}` — the source article (attribution) |
| `license_key` / `license_tier` | `cc_by` / `public_training_compatible_holdout` |
| `source_kind` | `transformed_case_report` |
| `mend_provenance` | *(present on 65 cases)* the source-derived result appended to make the gold determinable |

## Suggested scoring

Models emit a top-5 ranked differential; score **pass@1 … pass@5** against `gold_diagnosis` with an
LLM judge for semantic equivalence (a literal string match under-counts valid paraphrases). Conjunctive
golds ("A with B") should credit a candidate that names all components.
