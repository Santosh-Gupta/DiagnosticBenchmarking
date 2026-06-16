# DiagnosticBenchmarking

A neurology / psychiatry **clinical-diagnosis benchmark** built from open-access (CC-BY) published
case reports, plus the pipeline that constructs and quality-controls it.

Each benchmark item presents a redacted clinical vignette and asks for the most likely diagnosis (and
best next step). Cases are derived from peer-reviewed case reports where the diagnosis was
established, and every case is gated by a determinacy audit so the answer is reachable from the prompt
alone — not "failed" simply because it was unanswerable.

## The benchmark

[`data/public_benchmark/neuro_psych_cases.jsonl`](data/public_benchmark/neuro_psych_cases.jsonl) —
**349 validated CC-BY case challenges**. See
[`data/public_benchmark/README.md`](data/public_benchmark/README.md) for the schema, per-case
attribution/provenance, license terms, and quality-control summary.

Suggested scoring: models emit a top-5 ranked differential; score **pass@1 … pass@5** against
`gold_diagnosis` with an LLM judge for semantic equivalence.

## How cases are built and validated

- [docs/public/next_batch_creation_guide.md](docs/public/next_batch_creation_guide.md) — the turnkey
  harvest → refine → select → **validate/mend** → ship pipeline, and the four broken-case classes the
  validator catches.
- [docs/public/case_challenge_quality_guardrails.md](docs/public/case_challenge_quality_guardrails.md)
  — the refinement guardrails against omission, fabrication, and unsolvability.

The validation gate (`src/neurologybm/case_validation.py`) checks, per case, whether the gold is
determinable from the prompt; it repairs under-determined cases by appending the source's deciding
result **verbatim** (the gold is never relaxed) or drops cases that cannot be made fair.

## License

- **Benchmark data:** CC-BY 4.0; each case attributes its source article via the `provenance` field.
  Only strictly `cc_by` sources are included (no NonCommercial / ShareAlike / NoDerivs).
- **Code:** see `LICENSE` (or repository settings).

## Quick start

Requires Python 3.11+.

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

Build/validate a new batch:

```bash
neurologybm harvest --topic neurology --license-profile training --limit 25 --email you@example.com
neurologybm refine-public-challenges --model <model> --run
neurologybm validate-cases --manifest <selected_manifest>.jsonl \
    --out data/pmc/processed/case_validation/<batch>_<date> --mend --concurrency 8
```

Harvested full text and intermediate working data are gitignored (large and license-sensitive); only
the vetted CC-BY benchmark export under `data/public_benchmark/` is published.
