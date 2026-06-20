# Feedback to the batch generator — from the harness side (2026-06-19)

Grounded in deep-dives of harness failures across waves 8–12. Ordered by impact. The pipeline is
fundamentally sound; these are targeted fixes that would raise usable yield and gold trustworthiness.

## Guiding principle (project owner): no regex for *semantic* decisions — use the LLM
Any choice that requires *understanding the clinical text* — where the diagnosis starts, what counts as
a discriminator, whether the gold is reachable — must be made by an **LLM call**, never a regex. Regex is
fine for mechanical parsing (XML section extraction, "year-old" detection, vet/non-human filtering), but
the moment a decision depends on meaning, it belongs to the model. The current `transform_extract.py`
violates this for the most important decision in the whole pipeline (see #1).

---

## 1. (BIGGEST) REMOVE the regex truncation — let an LLM construct the challenge
**The bug.** `transform_extract.py` decides what to withhold with ONE regex, `DIAGNOSIS_LEAK_RE`
(`transform_extract.py:21-26`), in `_truncate_before_leak` (`:176`): it cuts everything from the first
match onward. The offending clause:
```
revealed (?:a|an)?\s*[A-Z][A-Za-z -]{3,}
```
matches any *"revealed/showed a Capitalized-word"* — which is almost always a **workup result**, not the
diagnosis: "MRI revealed Bilateral…", "labs revealed a Markedly elevated methylmalonic acid…", "CSF
revealed Neutrophilic pleocytosis…". So the **deciding finding gets deleted from the prompt** and dumped
into `answer_rest`. This is literally how cblC lost its MMA and SPNM lost its CSF differential. The regex
confuses "revealed a *finding*" with "revealed the *diagnosis*". (Note: pre-API regex cut, NOT a token
limit — raising max_tokens does nothing.)

**The fix — delete the regex split; make this an LLM step.** Feed an LLM the **full pre-diagnosis case +
workup** (from the XML sections) and have it produce the challenge directly: include every finding a
clinician needs, withhold only the diagnosis name / definitive treatment / outcome. The refine prompt
already has the right instructions (`public_refine.py:162-188`: "carry every discriminator; add the
result from the article if reported") — it was just being fed a regex-mangled prompt + a length-capped
article. Concretely:
- **Stop pre-truncating with the regex.** Pass the full case text to the constructor LLM; let *it* decide
  the withholding boundary semantically.
- **Don't cap the article below the workup/results section** (`--max-article-chars`) — the deciding lab
  is often deep in the text; if the model can't see it, it can't carry it.
- **Make discriminator-preservation a hard, checked step:** "name the single finding that separates this
  diagnosis from its nearest mimic; confirm it appears verbatim in `challenge_prompt`; if absent, add it
  from source." `required_findings` already half-does this — enforce it as a gate, not a hint.

Concrete cases this broke: **cblC (PMC12173315)** kept homocysteine, dropped MMA → indistinguishable from
CBS homocystinuria (harness ranked gold #3; after re-adding MMA from source, #1). **SPNM (PMC8769864)**
dropped the CSF neutrophil differential. **Aconitine / hyperthermia:** precipitant present, decisive
workup absent.

## 2. Gold granularity — NEVER alter the paper's gold; fix the prompt or drop
**Hard rule (project owner): never change/relabel the diagnosis the paper concluded.** The gold is the
paper's stated diagnosis, verbatim. So for an over-specific gold the model keeps missing on
("Animated picture syndrome", "ASMAN variant of GBS", "Fragile X *with epileptiform EEG epilepsy*"),
the fix is NOT to relabel it to a parent — it is to **`add_result` the discriminating finding from the
source** so the specific gold becomes fairly reachable, or **drop** the case if the paper doesn't contain
that discriminator. Same as the no-relax-gold rule everywhere else.
- **Deprecate `relabel_to_parent`** in `validate-cases` — it changes the paper's gold and violates this
  rule. Use `add_result` (source-grounded, gold unchanged) or `drop` instead.
- "Over-vague" golds (e.g. "mild intellectual disability", a severity descriptor) are a *selection*
  question, not a relabel one: if the paper's actual diagnostic conclusion is the etiology, use that;
  if the paper genuinely only establishes severity, the case is a poor diagnostic challenge — drop it.
  Never invent a more specific gold than the paper supports, and never generalize the paper's specific one.

## 3. Ship VALIDATED, not raw — bake `validate-cases --mend` into generation
Every recent batch arrived raw (prompt + `answer_rest`, no structured gold), and the harness side has
been running refine → `validate-cases --mend` each time. Move that INTO the generator so batches ship
already gold-bearing and mended. The validator's finding-level determinacy check (added 2026-06-17)
catches the #1 issue and `add_result`-mends it from source (gold preserved) — run it as the last
generation step so every shipped case is solvable-from-prompt or explicitly dropped.

## 4. Fix the Pro `Response ended prematurely` drops — keep using Pro
Pro is cheap; keep using it. This is a **known DeepSeek issue** with a real fix. The client uses a
**non-streaming** `requests.post(...)` (`deepseek.py:111`). On a long `deepseek-reasoner` (Pro)
generation the TCP connection sits **idle** while the server reasons for 1–2 min, and a load
balancer/proxy drops it before the body arrives → `urllib3 ProtocolError: Response ended prematurely`.
It is NOT context size. Per DeepSeek's FAQ: *"To prevent TCP connection timeout, DeepSeek continuously
returns SSE keep-alive comments for **streaming** requests."*
**Primary fix: switch the Pro path to `stream=True`** and accumulate SSE chunks (handle keep-alive empty
lines) — the keep-alive traffic holds the connection open through the long generation. Keep
retry-with-backoff (now in `deepseek.py`) as a backstop and raise the read timeout (reasoner > 120 s).
Sources: DeepSeek API FAQ (api-docs.deepseek.com/faq); deepseek-ai/DeepSeek-R1 issue #314.
Separately, Pro is *slow* on heavy refinement input (~100 s/case on 30 K chars) — a latency tradeoff,
not a failure. Caveat: for *gold extraction specifically*, Flash produced identical golds to Pro on every
overlap (FLS; MERS; …), so Flash is a fine fast fallback when Pro is flaky — not a mandated replacement.

## 5. Do NOT filter by topic — quality only
We want all cases; occasional out-of-field aids generalization. So the earlier "tighten the neuro/psych
filter / drop the pulmonary case" advice is **withdrawn** — field diversity is a feature. The only
selection filters are **quality** (solvable, a real clinical diagnosis as gold), never topic. Two quality
items still stand and are LLM judgments: reviews/overviews used as case reports (e.g. catatonia "Clinical
Overview" PMC8628989) and research/methods constructs as "gold" (`gold_not_a_diagnosis`).

## 6. Remove the `not_solvable` AUTO-DROP entirely — route everything to `validate-cases`
There should be no `not_solvable` drop at all. That status is just the refiner's *Flash blind-probe*
failing, and it conflates **(a) missing info from source** (→ mend it, or it's a genuine hard IR case)
and **(b) the system isn't good enough yet** (→ exactly the valuable hard case the benchmark exists to
contain). Dropping it discards both — 30–43 % of recent batches. **Send every gold-bearing case
(including `not_solvable`) through `validate-cases`**, the correct gate (broken→drop/mend vs
hard-but-fair→keep). *Proven on twelfth-wave: routing the full set recovered **89 usable cases vs 49**,
and 21 of the `not_solvable` cases were just missing a discriminator the validator added from source,
then solved fine.* Harness side has adopted this too.

## 7. Existing/shipped cases do NOT need regeneration — a cheap re-validation sweep fixes them
Because every case row retains `xml_path` to the full source article, **no information is ever
permanently lost** — any LLM pass can re-read the source and restore a dropped discriminator. So the
regex damage in already-built cases (incl. the public 349) is fixed by a **one-time `validate-cases`
re-sweep** (re-reads source, `add_result`-mends, gold preserved), not by rebuilding. Empirically the
discriminator-loss hit ~25 % of cases (tenth-wave: 16/64 mended); the rest were already fine. Run the
improved finding-level `validate-cases --mend` over each shipped manifest once and re-ship the mended
output.

---

## Action checklist for the generator agent
1. **Delete the regex truncation in `transform_extract.py`; construct the challenge with an LLM** from the
   full pre-diagnosis case text (principle: no regex for semantic decisions). [#1]
2. **Switch the Pro/DeepSeek client to `stream=True`** (SSE keep-alive) so Pro stops dropping. [#4]
3. **Make `validate-cases --mend` the final generation step** so batches ship validated, not raw. [#3]
4. **Remove the `not_solvable` auto-drop**; route all gold-bearing cases through `validate-cases`. [#6]
5. **Gold integrity:** keep the paper's gold verbatim — fix over-specific golds with `add_result` from
   source (never relabel), drop severity-descriptor-only cases. **Deprecate `relabel_to_parent`.** [#2]
6. **Stop topic-filtering**; keep field diversity. Keep the review/non-case and `gold_not_a_diagnosis`
   quality filters. [#5]
7. **One-time `validate-cases` re-sweep over already-shipped cases** (incl. public 349) — re-mend from
   source; no regeneration. [#7]

## For the NEXT set (now that the regex is gone)
- The pipeline is now LLM-driven + auto-validated, so the next batch should arrive much cleaner. Before
  shipping the full 100, **spot-check ~5 cases that the LLM construction actually preserved the
  discriminator** (compare challenge_prompt vs source for the deciding finding) — the regex removal is
  new, verify it behaves.
- **Ship the refined + validated artifact, not the raw candidate pool** (raw is now explicitly
  `candidate_requires_llm_refinement`).
- **Deprecate `relabel_to_parent`** (see #2) — never alter the paper's gold.
- You have ~474 candidates available, so a full 100 is fine. Strict CC-BY only.

## Keep doing
- **Strict CC-BY licensing.** Do NOT introduce an NC/benchmark-only pool: the public benchmark is now
  MIT-code + CC-BY-data (commercially usable), and a mixed pool breaks that boundary. For bigger batches,
  harvest a wider CC-BY OA tranche rather than relaxing the license.
- The refiner self-audit statuses (`needs_fidelity_review`, `needs_leakage_review`, …) — useful; keep.
- The corrected exclusion logic (exclude only shipped artifacts, not intermediate manifests) — good fix
  that unlocked the 474-candidate pool.
