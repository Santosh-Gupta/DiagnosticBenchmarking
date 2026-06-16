"""Validate EXISTING case challenges for solvability/determinacy defects.

Motivation (from ClinicalHarness, 2026-06-15): a large fraction of "hard" benchmark failures are not
model failures but BROKEN challenges — the prompt asks for a specific gene / antibody / pathology whose
defining result is withheld (e.g. "a gene panel was SENT for sequencing" then "what is the genetic
diagnosis?"), and that specific entity is not inferable from a phenotype shared with near-neighbors. A
case is only usable if every discriminator needed to reach the gold diagnosis is present in the prompt.

This module flags such cases and proposes a repair. It complements `public_refine.py` (which validates
at refinement time); this runs over ALREADY-built challenge manifests to find and fix existing defects.

Two layers:
  1. Deterministic specificity check (no API): if the gold names a specific gene/antibody/molecular
     token that is absent from the prompt, flag `specificity_unsupported`.
  2. LLM determinacy check (DeepSeek): is the gold reachable from the prompt alone? If it needs a
     withheld result, name the missing discriminator and suggest a repair.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .deepseek import DeepSeekClient

# Distinctive specific-entity tokens: gene-symbol-like (KCNMA1, SLC6A1, ATP1A3, DJ-1, SPG4), antibody
# names (anti-NMDAR, CASPR2, DPPX, MOG, AQP4), and variant notations (c.123, p.Arg). Generic clinical
# acronyms are excluded — their absence from a prompt is not a determinacy defect.
_GENE_LIKE = re.compile(r"\b([A-Z][A-Z0-9]{2,}\d?[A-Z]?|[A-Z]{2,}\d)\b")
_ANTIBODY = re.compile(r"\b(?:anti-)?(?:NMDAR|NMDA|CASPR2|LGI1|GABA[AB]?|AMPA|GAD65|DPPX|MOG|AQP4|GlyR|IgLON5)\b", re.I)
_VARIANT = re.compile(r"\bc\.\d|\bp\.[A-Z][a-z]{2}\d")
_SPECIFICITY_CONTEXT = re.compile(
    r"\b(gene|genetic|variant|mutation|sequenc|exome|panel|antibody|immunohistochem|IHC|biopsy|titer|titre)\b", re.I
)
_WITHHELD = re.compile(
    r"\b(sent for|pending|await\w*|was sent|ordered|to be (?:determined|performed)|will be|results? not\b|under (?:investigation|review))\b",
    re.I,
)
# Generic acronyms that are not "specific entity" tokens (their absence from a prompt is fine).
_GENERIC_ACRONYMS = frozenset(
    {"MRI", "CT", "EEG", "CSF", "PCR", "ALS", "MS", "ADHD", "IUD", "DNA", "RNA", "CNS", "ICU", "NICU",
     "PET", "FDG", "ESR", "CRP", "MCI", "MMO", "PNS", "MSA", "MERS", "SREAT", "VLOSLP", "IAD", "RVCL",
     "NPH", "GTN", "TB", "HSV", "SLE"}
)


@dataclass(frozen=True)
class CaseValidation:
    case_id: str
    gold_diagnosis: str
    determinable: bool | None  # None when LLM check not run
    deterministic_flags: tuple[str, ...] = field(default_factory=tuple)
    missing_discriminators: tuple[str, ...] = field(default_factory=tuple)
    broken_class: str | None = None  # ok | under_determined | gold_not_a_diagnosis | prompt_refutes_gold | gold_overspecific
    suggested_repair: str | None = None  # add_result | reframe_next_step | relabel_to_parent | drop | relax_gold | none
    repair_detail: str | None = None
    prompt_addition: str | None = None  # for add_result: exact sentence(s) to append to the prompt
    gold_relabel: str | None = None  # for relabel_to_parent: standard parent diagnosis (non-auto; surfaced for review)
    addition_is_vus: bool = False  # True if the only added result is a VUS (red herring; do not auto-mend)
    error: str | None = None


def _specific_tokens(gold: str) -> set[str]:
    tokens: set[str] = set()
    for m in _GENE_LIKE.findall(gold):
        if m not in _GENERIC_ACRONYMS and len(m) >= 3:
            tokens.add(m)
    tokens.update(m if isinstance(m, str) else m[0] for m in _ANTIBODY.findall(gold))
    return {t for t in tokens if t}


def deterministic_flags(challenge_prompt: str, gold_diagnosis: str) -> list[str]:
    """Flags computable without an API call. The key one: the gold names a specific gene/antibody/
    molecular entity that does not appear in the prompt, AND the prompt frames the answer as requiring
    such a result (or states the result is withheld) — i.e. the gold is not determinable as written."""
    flags: list[str] = []
    prompt_l = challenge_prompt.lower()
    specific = {t for t in _specific_tokens(gold_diagnosis) if t.lower() not in prompt_l}
    if specific:
        flags.append("specificity_token_absent:" + ",".join(sorted(specific))[:80])
        if _SPECIFICITY_CONTEXT.search(gold_diagnosis) or _SPECIFICITY_CONTEXT.search(challenge_prompt):
            flags.append("specificity_unsupported")  # specific entity asked for, defining token absent
    if _WITHHELD.search(challenge_prompt) and _SPECIFICITY_CONTEXT.search(challenge_prompt):
        flags.append("deciding_result_withheld")  # "panel sent / biopsy pending" pattern
    return flags


def build_determinacy_prompt(row: dict[str, Any], source_excerpt: str) -> tuple[str, str]:
    system = (
        "You audit whether a clinical case-challenge is SOLVABLE as written. Return only JSON. No hidden "
        "chain-of-thought."
    )
    user = f"""A benchmark case-challenge presents a vignette and is scored against a fixed gold diagnosis.
Decide whether the gold is DETERMINABLE from the challenge prompt ALONE (plus general medical knowledge
+ literature) — i.e. every discriminator needed to reach THIS specific gold is present in the prompt.

A case is BROKEN (unusable) for any of these reasons:
- UNDER_DETERMINED: reaching the gold requires a result the prompt withholds — a specific gene when only
  "a gene panel was sent" is stated, a specific antibody when no serology is given, a specific pathology
  when the biopsy is "pending". Near-neighbors (DJ-1 vs PRKN vs PINK1; one antibody subtype vs another)
  are not distinguishable without that result.
- GOLD_NOT_A_DIAGNOSIS: the gold is not a clinical diagnosis a clinician would make — it is a research
  construct, an experimental/mechanistic finding, or an intraoperative mapping result (e.g. "hemispheric
  dissociation of anxiety," "atypical right-hemisphere language representation"). These come from
  research/methods papers, not diagnostic case reports.
- PROMPT_REFUTES_GOLD: the prompt contains a result that REFUTES the gold — a negative/normal test for
  the gold entity (e.g. gold "tumor recurrence" but prompt says "MRI shows no recurrence"), so the gold
  is not the best answer to the prompt as written.
- GOLD_OVERSPECIFIC: the gold pins a finer GRANULARITY than the prompt can fairly support against a
  standard PARENT entity — a named subtype/variant whose discriminator is absent (e.g. gold "ASMAN
  variant of GBS" but the prompt's features are equally an "AMAN variant"; gold "X type 9" with no
  type-distinguishing result), a present/absent qualifier the prompt never raises (e.g. gold "... WITHOUT
  encephalopathy" but the prompt never discusses mental status), or a bespoke/non-standard label when a
  recognized parent diagnosis fits the prompt equally (e.g. gold "autoimmune-thyroid focal CNS disorder"
  where SREAT is the standard entity the features support). The model lands on the correct PARENT and is
  scored wrong on a distinction the prompt does not contain.
Set determinable=false for ALL of these and choose the repair accordingly (drop for GOLD_NOT_A_DIAGNOSIS
and usually PROMPT_REFUTES_GOLD; add_result/reframe for UNDER_DETERMINED; add_result preferred for
GOLD_OVERSPECIFIC, else relabel_to_parent/drop).

REPAIR POLICY — preserve clinical truth, keep the gold diagnosis as published:
- STRONGLY PREFER "add_result": add the deciding result to the challenge_prompt VERBATIM from the source
  (the sequencing variant, the antibody serology, the IHC/biopsy result) so the gold becomes reachable.
  This keeps the case clinically true and the gold unchanged.
- "reframe_next_step": only if the source itself reached the gold only AFTER a test not in the vignette
  and you cannot add a concrete result — reframe the asked question to that next diagnostic step.
- "relabel_to_parent": ONLY for GOLD_OVERSPECIFIC when the source cannot supply a discriminator to make
  the fine gold determinable — name the standard PARENT diagnosis the prompt DOES support in
  "gold_relabel". This is surfaced for human review and is NOT auto-applied (it changes the gold).
- "drop": if the case cannot be made determinable without inventing data.
- Do NOT relax/generalize the gold diagnosis (that sacrifices clinical accuracy). "relax_gold" is a last
  resort only when the SOURCE PAPER itself explicitly could not specify the entity. For GOLD_OVERSPECIFIC
  STRONGLY PREFER add_result (add the subtype/qualifier discriminator from the source so the FINE gold
  becomes determinable, keeping it unchanged); use relabel_to_parent only if the source has no such
  discriminator.

Return JSON:
{{
  "determinable": true|false,
  "broken_class": "ok | under_determined | gold_not_a_diagnosis | prompt_refutes_gold | gold_overspecific",
  "missing_discriminators": ["the specific result(s) needed but absent from the prompt"],
  "suggested_repair": "add_result | reframe_next_step | relabel_to_parent | drop | relax_gold | none",
  "repair_detail": "concrete instruction; for add_result, the EXACT sentence to insert into the prompt, copied/derived from the source (e.g. 'Genetic testing revealed compound heterozygous DJ-1 (PARK7) variants: exon 6 deletion and c.242dup.')",
  "prompt_addition": "<for add_result: the exact sentence(s) to append to challenge_prompt, else null>",
  "gold_relabel": "<for relabel_to_parent: the standard parent diagnosis the prompt supports, else null>",
  "addition_is_vus": true|false  // true if the only added result is a Variant of Uncertain Significance (a red herring, not confirmatory) -> prefer reframe/drop
}}

gold_diagnosis: {row.get('answer_key_diagnosis') or row.get('gold')}
challenge_prompt:
{row.get('challenge_prompt','')}

source_excerpt (the original paper, for repair via add_result):
{source_excerpt[:3000]}
"""
    return system, user


def validate_case(
    row: dict[str, Any],
    *,
    client: DeepSeekClient | None = None,
    model: str | None = None,
    source_excerpt: str = "",
) -> CaseValidation:
    case_id = str(row.get("case_id", ""))
    gold = row.get("answer_key_diagnosis") or row.get("gold") or _gold_from_answer_rest(row)
    det = deterministic_flags(row.get("challenge_prompt", ""), gold or "")
    if client is None:
        # deterministic-only: 'determinable' unknown; flag specificity defects
        return CaseValidation(case_id=case_id, gold_diagnosis=gold or "", determinable=None,
                              deterministic_flags=tuple(det))
    system, user = build_determinacy_prompt({**row, "gold": gold}, source_excerpt)
    try:
        raw = client.chat_json(model=model or "deepseek-v4-flash", system_prompt=system, user_prompt=user, temperature=0.0)
        p = raw.get("parsed_content", {}) if isinstance(raw, dict) else {}
    except Exception as exc:  # noqa: BLE001
        return CaseValidation(case_id=case_id, gold_diagnosis=gold or "", determinable=None,
                              deterministic_flags=tuple(det), error=str(exc))
    md = p.get("missing_discriminators")
    return CaseValidation(
        case_id=case_id,
        gold_diagnosis=gold or "",
        determinable=bool(p.get("determinable")) if "determinable" in p else None,
        broken_class=p.get("broken_class") if isinstance(p.get("broken_class"), str) else None,
        deterministic_flags=tuple(det),
        missing_discriminators=tuple(s for s in md if isinstance(s, str)) if isinstance(md, list) else (),
        suggested_repair=p.get("suggested_repair") if isinstance(p.get("suggested_repair"), str) else None,
        repair_detail=p.get("repair_detail") if isinstance(p.get("repair_detail"), str) else None,
        prompt_addition=p.get("prompt_addition") if isinstance(p.get("prompt_addition"), str) and p.get("prompt_addition").strip() else None,
        gold_relabel=p.get("gold_relabel") if isinstance(p.get("gold_relabel"), str) and p.get("gold_relabel").strip() else None,
        addition_is_vus=bool(p.get("addition_is_vus")),
    )


def mend_row(row: dict[str, Any], validation: CaseValidation) -> dict[str, Any] | None:
    """Apply an add_result repair: append the source-derived deciding result to challenge_prompt.

    Returns a NEW manifest row with the mended prompt (and a provenance marker), or None if the case
    needs a different repair (reframe/relabel/drop/relax) that is not a safe automatic prompt edit.
    Never changes the gold diagnosis — including GOLD_OVERSPECIFIC, which auto-mends ONLY via add_result
    (adding the subtype/qualifier discriminator so the fine gold becomes determinable); its
    relabel_to_parent fallback returns None here and is surfaced for human review.
    """
    if validation.suggested_repair != "add_result" or not validation.prompt_addition:
        return None
    if validation.addition_is_vus:
        return None  # a VUS is a red herring, not a confirmatory result — do not auto-mend with it
    prompt = row.get("challenge_prompt", "")
    addition = validation.prompt_addition.strip()
    if addition.lower() in prompt.lower():
        return None  # already present; nothing to mend
    mended = dict(row)
    mended["challenge_prompt"] = prompt.rstrip() + "\n\n" + addition
    mended["mend_provenance"] = {
        "repair": "add_result",
        "added": addition,
        "missing_discriminators": list(validation.missing_discriminators),
    }
    return mended


def _gold_from_answer_rest(row: dict[str, Any]) -> str | None:
    # Support both manifest schemas: ``answer_rest`` (JSON string) and a nested ``answer_key`` object.
    # The eighth-wave batch used answer_key, so a row.get("answer_rest")-only read returned an empty
    # gold and the validator judged determinacy without the actual target diagnosis.
    ar = row.get("answer_rest")
    if isinstance(ar, str) and ar.strip():
        try:
            d = json.loads(ar)
            if isinstance(d, dict) and isinstance(d.get("diagnosis"), str):
                return d["diagnosis"]
        except Exception:  # noqa: BLE001
            pass
    ak = row.get("answer_key")
    if isinstance(ak, dict) and isinstance(ak.get("diagnosis"), str):
        return ak["diagnosis"]
    return None


def _source_text(row: dict[str, Any]) -> str:
    from .public_refine import extract_article_text_from_xml
    xml = row.get("source_xml_path")
    if xml and Path(xml).exists():
        try:
            return extract_article_text_from_xml(Path(xml), max_chars=20000)
        except Exception:  # noqa: BLE001
            return ""
    return ""


def run_case_validation(
    *,
    client: DeepSeekClient | None,
    manifest_path: str | Path,
    out_dir: str | Path,
    model: str | None = None,
    mend: bool = False,
    concurrency: int = 8,
    limit: int | None = None,
) -> dict[str, Any]:
    """Validate (and optionally mend) every case in a manifest. Writes a validation JSONL, a mended
    manifest (add_result repairs applied, gold unchanged), and a summary. Deterministic-only when
    client is None."""
    from .concurrency import run_ordered_concurrent

    rows = [json.loads(l) for l in Path(manifest_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit is not None:
        rows = rows[:limit]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _validate(row: dict[str, Any]) -> tuple[dict[str, Any], CaseValidation]:
        v = validate_case(row, client=client, model=model, source_excerpt=_source_text(row) if client else "")
        return row, v

    pairs = list(run_ordered_concurrent(rows, _validate, concurrency=concurrency if client else 1, request_spacing_seconds=0.0))

    validations = [v for _, v in pairs]
    mended_rows: list[dict[str, Any]] = []
    mend_count = 0
    for row, v in pairs:
        m = mend_row(row, v) if mend else None
        if m is not None:
            mended_rows.append(m)
            mend_count += 1
        else:
            mended_rows.append(row)

    (out / "validation.jsonl").write_text(
        "\n".join(json.dumps(_validation_to_dict(v), ensure_ascii=False) for v in validations) + "\n", encoding="utf-8"
    )
    if mend:
        (out / "mended_manifest.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in mended_rows) + "\n", encoding="utf-8"
        )
    summary = {**summarize_validations(validations), "mended": mend_count, "manifest": str(manifest_path)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _validation_to_dict(v: CaseValidation) -> dict[str, Any]:
    return {
        "case_id": v.case_id, "gold_diagnosis": v.gold_diagnosis, "determinable": v.determinable,
        "broken_class": v.broken_class, "deterministic_flags": list(v.deterministic_flags),
        "missing_discriminators": list(v.missing_discriminators),
        "suggested_repair": v.suggested_repair, "repair_detail": v.repair_detail,
        "prompt_addition": v.prompt_addition, "gold_relabel": v.gold_relabel,
        "addition_is_vus": v.addition_is_vus, "error": v.error,
    }


def summarize_validations(rows: list[CaseValidation]) -> dict[str, int]:
    out = {"total": len(rows)}
    out["deterministic_flagged"] = sum(1 for r in rows if r.deterministic_flags)
    out["llm_not_determinable"] = sum(1 for r in rows if r.determinable is False)
    for repair in ("add_result", "relax_gold", "reframe_next_step", "relabel_to_parent", "drop"):
        out[f"repair_{repair}"] = sum(1 for r in rows if r.suggested_repair == repair)
    for bc in ("under_determined", "gold_not_a_diagnosis", "prompt_refutes_gold", "gold_overspecific"):
        out[f"broken_{bc}"] = sum(1 for r in rows if r.broken_class == bc)
    out["vus_additions"] = sum(1 for r in rows if r.addition_is_vus)
    return out
