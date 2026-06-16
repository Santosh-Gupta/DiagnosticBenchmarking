# Case-Challenge Quality Guardrails (refinement pipeline)

Added after two refined challenges were found defective during downstream evaluation.
Both had passed the old refinement as `refined_needs_spotcheck` — the model's own
`adequacy_audit.is_self_contained` was `True` in both cases, so self-audit alone is not enough.

## What went wrong (the two cases)

- **PMC12581184 (NPSLE).** The refiner **dropped the deciding serologies** (ANA 1:1280, anti-dsDNA
  >300 IU/mL), **fabricated** an MRI finding (subcortical white-matter hyperintensities) that the
  source said was negative, and **altered** the CSF numbers. Result: the prompt's best answer was
  anti-NMDA encephalitis — a defensible *wrong* answer — because the lupus evidence was gone.
- **PMC11066795 (intrarenal neurofibroma).** The refiner asked for "the pathological diagnosis and
  confirmatory IHC" but left histology **"pending"** in the prompt. The deciding pathology
  (S100-positive spindle cells with serpentine wavy nuclei) was withheld, so the question was
  unanswerable from the prompt.

## The three failure modes a refiner must be guarded against

1. **Omission** — dropping the finding that discriminates the answer from its mimics.
2. **Fabrication / value drift** — inventing or altering findings not in the source.
3. **Unsolvability** — asking for a determination (often a final pathological/microbiological
   diagnosis) whose deciding result is not present in the prompt.

## What changed in `src/neurologybm/public_refine.py`

**Prompt (`build_public_refinement_prompt`)** now carries four CRITICAL RULES: preserve all
discriminators verbatim; no fabrication and exact numeric values; solvability must match the
question (include the deciding result, or reframe to the pre-result decision — never ask for a
diagnosis whose defining pathology is absent); and `required_findings` must literally appear in the
prompt. Two new self-audit objects are required in the output: `fidelity_audit` and
`solvability_audit`.

**Deterministic validation (`validate_refinement`, independent of the model's self-audit):**
- *Omission check* — every `answer_key.required_findings` item must have token overlap with
  `challenge_prompt`; an absent deciding finding is flagged.
- *Value-fidelity check* — distinctive numeric values in the prompt (titers, comparators, decimals,
  3+ digit integers) must appear in the source article; unsourced values are flagged (catches both
  fabricated and altered numbers).

**New `review_status` values** (downstream filtering should treat these as NOT benchmark-ready
without human review):
- `not_solvable` — self-reported or detected: the question can't be answered from the prompt.
- `needs_fidelity_review` — self-reported or detected fabrication / value drift.
- (existing: `needs_leakage_review`, `not_self_contained`, `invalid_refinement`,
  `refined_needs_spotcheck`.)

Each refined record now also carries a `validation_reasons` list for triage.

## Solvability probe (the `validate-cases` gate)

The strongest check is an independent LLM pass given ONLY the `challenge_prompt` (no answer): if a
strong model with full reasoning cannot reach `answer_key.diagnosis` (judged by clinical
equivalence), the prompt is underdetermined and should go to human review. This separates "hard for
the model" from "impossible from the text" — the distinction both defective cases failed. It costs
one extra call per case and gates the final benchmark export. See
[next_batch_creation_guide.md](next_batch_creation_guide.md).
