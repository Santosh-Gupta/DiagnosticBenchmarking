# Creating the next case-challenge batch — turnkey guide

For the agent generating the next dev/test batch. The pipeline has a **validation/mend gate** that
catches the defect classes which silently inflated prior benchmarks (cases the solver "failed" only
because they were unanswerable). Follow these steps and the batch ships clean.

## The end-to-end pipeline

1. **Harvest** public CC-BY PMC case reports: `neurologybm harvest` / `harvest-sources`.
   XML extractors recurse into nested raw-XML folders.
2. **Refine** into challenge/answer splits: `neurologybm refine-public-challenges …`. The refiner
   runs a deterministic `specificity_unsupported` check (see below) and solvability / fidelity /
   leakage audits. Heed `review_status` and `validation_reasons`.
3. **Select** the batch (license = CC BY; bucket as desired).
4. **VALIDATE + MEND (the required gate):**
   ```
   neurologybm validate-cases --manifest <selected_manifest>.jsonl \
       --out data/pmc/processed/case_validation/<batch>_<date> --mend --concurrency 8
   ```
   This reads the **local source XML** for each case and, per case, decides whether the gold is
   determinable from the prompt. It writes `validation.jsonl`, `summary.json`, and `mended_manifest.jsonl`.
5. **Review the summary**, apply the policy below, and ship `mended_manifest.jsonl` as the batch.

## The four broken-case classes the validator flags (and what to do)

| broken_class | meaning | repair |
| --- | --- | --- |
| `under_determined` | gold needs a withheld result (specific gene/antibody/pathology; "panel sent", "biopsy pending") | **add_result** — the deciding result is appended VERBATIM from the source, gold UNCHANGED (auto-applied with `--mend`). ~27% of recent batches needed this. |
| `gold_not_a_diagnosis` | source is a research/methods/mechanism paper; "gold" is an experimental construct, not a clinical diagnosis | **drop** the case (not a valid diagnostic challenge) |
| `prompt_refutes_gold` | the prompt contains a result that refutes the gold (e.g. "MRI: no recurrence" vs gold "recurrence") | **drop** or reframe; the gold is not the best answer to the prompt as written |
| `gold_overspecific` | gold pins a finer granularity than the prompt supports vs a standard PARENT entity — a named subtype/variant with no discriminator in the prompt (e.g. gold "ASMAN variant of GBS", prompt equally fits "AMAN"), a present/absent qualifier the prompt never raises (gold "… *without* encephalopathy", prompt silent on mental status), or a bespoke label where a recognized parent fits (gold "autoimmune-thyroid focal CNS disorder" vs standard SREAT). A correct reasoner lands on the parent and is scored wrong on a distinction the prompt does not contain. | **add_result** preferred — append the source's subtype/qualifier discriminator so the FINE gold becomes determinable, gold UNCHANGED (auto-applied with `--mend`). Else `relabel_to_parent` (the validator names the parent in `gold_relabel`) — surfaced for human review, **not** auto-applied since it changes the gold — or **drop**. |

Also: `addition_is_vus=true` flags a mend whose only added result is a **Variant of Uncertain
Significance** — a red herring (it pushes solvers to the wrong gene). The validator does NOT auto-mend
these; review by hand (usually reframe or drop).

## The non-negotiable quality rule

**A case is usable only if every discriminator needed to reach the published gold is present in the
prompt** — and the gold is an actual clinical diagnosis. Never relax the gold to make a case "pass";
preserve clinical truth. Fix by adding the source's deciding result, or drop the case.

## Creator-side check already in the refiner

`public_refine.validate_refinement` emits `specificity_unsupported` when the answer_key names a
specific gene/antibody/molecular entity whose token is absent from the prompt — so most
under-determined cases are caught at refinement time. Treat that flag as "add the result or set the
gold at the determinable level (with the confirmatory test as the next step)."

## Evidence this matters

Re-validating the second-/third-100 dev sets: ~54/200 (27%) were under-determined and mended
(add_result, 0 gold relaxed); plus a handful of `gold_not_a_diagnosis` / `prompt_refutes_gold` found in
the residual deep-dive. Cleaning these moved the measured end-to-end solve-rate from a misleading ~15%
to 88–92%. **Run `validate-cases` before publishing any batch.**
