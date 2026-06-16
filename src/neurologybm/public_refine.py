"""DeepSeek-assisted refinement of public case-report challenge splits."""

from __future__ import annotations

import csv
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .case_eval import new_run_id
from .concurrency import run_ordered_concurrent
from .deepseek import DeepSeekClient
from .public_eval import load_public_splits


REFINED_FIELDS = [
    "case_id",
    "source_kind",
    "pmcid",
    "doi",
    "title",
    "journal",
    "license_key",
    "license_tier",
    "model",
    "challenge_prompt",
    "answer_key",
    "evidence_map",
    "hypothesis_bank",
    "outcome_summary",
    "leakage_audit",
    "adequacy_audit",
    "fidelity_audit",
    "solvability_audit",
    "solvability_probe",
    "source_usage",
    "validation_reasons",
    "review_status",
]

REFINED_SCHEMA_KEYS = [
    "challenge_prompt",
    "answer_key",
    "evidence_map",
    "hypothesis_bank",
    "outcome_summary",
    "leakage_audit",
    "adequacy_audit",
    "fidelity_audit",
    "solvability_audit",
    "source_usage",
]


def run_public_refinement(
    *,
    client: DeepSeekClient,
    manifest_path: Path,
    out_root: Path,
    model: str,
    dry_run: bool,
    temperature: float,
    case_ids: set[str] | None = None,
    limit: int | None = None,
    extra_body: dict[str, Any] | None = None,
    resume_results: Path | None = None,
    max_article_chars: int = 45000,
    include_article_text: bool = True,
    api_retries: int = 2,
    api_retry_sleep_seconds: float = 5.0,
    solvability_probe_model: str | None = None,
    concurrency: int = 1,
    request_spacing_seconds: float = 0.0,
) -> dict[str, Any]:
    rows = load_public_splits(manifest_path, case_ids=case_ids, limit=limit)
    completed = _completed_case_ids(resume_results)
    if completed:
        rows = [row for row in rows if str(row.get("case_id")) not in completed]
    if not rows:
        raise ValueError("No public rows selected for refinement.")

    run_id = new_run_id()
    run_dir = out_root / f"public_refine_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    refined_path = run_dir / "refined_cases.jsonl"
    tsv_path = run_dir / "refined_cases.tsv"
    raw_path = run_dir / "raw_api_records.jsonl"

    with refined_path.open("w", encoding="utf-8") as jsonl_file, tsv_path.open(
        "w", newline="", encoding="utf-8"
    ) as tsv_file, raw_path.open("w", encoding="utf-8") as raw_file:
        writer = csv.DictWriter(tsv_file, fieldnames=REFINED_FIELDS, delimiter="\t")
        writer.writeheader()

        def worker(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            return _refine_one(
                client=client,
                row=row,
                model=model,
                dry_run=dry_run,
                temperature=temperature,
                extra_body=extra_body,
                max_article_chars=max_article_chars,
                include_article_text=include_article_text,
                api_retries=api_retries,
                api_retry_sleep_seconds=api_retry_sleep_seconds,
                solvability_probe_model=solvability_probe_model,
            )

        for artifact, raw_record in run_ordered_concurrent(
            rows,
            worker,
            concurrency=concurrency,
            request_spacing_seconds=request_spacing_seconds,
        ):
            jsonl_file.write(json.dumps(artifact, ensure_ascii=False, sort_keys=True) + "\n")
            jsonl_file.flush()
            writer.writerow(_tsv_row(artifact))
            tsv_file.flush()
            raw_file.write(json.dumps(raw_record, ensure_ascii=False, sort_keys=True) + "\n")
            raw_file.flush()

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "input_manifest": str(manifest_path),
        "selected_case_count": len(rows),
        "skipped_completed_case_count": len(completed),
        "resume_results": str(resume_results) if resume_results else None,
        "model": model,
        "temperature": temperature,
        "extra_body": extra_body or {},
        "max_article_chars": max_article_chars,
        "include_article_text": include_article_text,
        "api_retries": api_retries,
        "api_retry_sleep_seconds": api_retry_sleep_seconds,
        "solvability_probe_model": solvability_probe_model,
        "concurrency": concurrency,
        "request_spacing_seconds": request_spacing_seconds,
        "refined_path": str(refined_path),
        "tsv_path": str(tsv_path),
        "raw_path": str(raw_path),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"run_dir": str(run_dir), **manifest}


def build_public_refinement_prompt(row: dict[str, Any], article_text: str) -> tuple[str, str]:
    system_prompt = (
        "You convert public medical case reports into benchmark-ready diagnostic or management challenges. "
        "Return only valid JSON. Do not include hidden chain-of-thought. "
        "Use concise evidence maps, not private reasoning traces."
    )
    user_prompt = f"""Rewrite this public case into a benchmark item.

Goal:
- The challenge_prompt must contain all clinical starting information needed for a strong clinician/model to diagnose the case or choose the next step.
- The challenge_prompt must exclude answer leakage: no final diagnosis, final etiology, article title diagnosis, definitive treatment response, outcome after diagnosis, or publication/discussion wording that gives away the answer.
- You may include tests, imaging descriptions, pathology descriptions, or intermediate workup results only if they were available before the final diagnosis/etiology was established.
- If the original article is not suitable as a self-contained text-only benchmark, say so in adequacy_audit.
- Keep the output compact. challenge_prompt may be as long as needed, but all other fields must be concise.
- Limit required_findings to 8 items, optional_findings to 8 items, evidence_map to 8 items, and hypothesis_bank to 10 items.
- Do not include references, citations, long quotes, full discussion text, or long outcome narrative.

CRITICAL RULES (these caused real benchmark defects when violated):
1. PRESERVE THE DISCRIMINATORS. Every objective finding needed to distinguish the answer from its
   closest mimics — lab values, serologies and titers (e.g. ANA 1:1280, anti-dsDNA >300 IU/mL),
   immunohistochemistry/markers, cytogenetics, microbiology, gross/microscopic pathology
   descriptors, and key imaging findings — MUST be carried into challenge_prompt in full clinical
   substance. Do NOT summarize away or omit the finding that makes the diagnosis decidable. A
   prompt that omits the deciding finding is worse than useless: it produces a defensible WRONG
   answer.
2. NO FABRICATION, EXACT VALUES. Include ONLY findings actually present in the source article.
   Never invent, infer, or alter a finding. Copy numeric values (titers, counts, sizes, dates)
   exactly; do not round or change them. If the source says imaging/a test was negative or normal,
   do NOT report it as abnormal (and vice versa). Inventing a finding is a critical defect.
3. SOLVABILITY MUST MATCH THE QUESTION. challenge_prompt + general medical knowledge must suffice
   to reach answer_key.diagnosis (or the asked next step). If reaching the answer requires a result
   that is NOT in the case's pre-answer workup — e.g. histopathology/IHC that the source only
   described as "pending" — then either (a) include the actual result if the article reports it
   later, or (b) REFRAME the question to the appropriate pre-result decision (most likely
   diagnosis from imaging / next best diagnostic test). NEVER ask for a final pathological or
   microbiological diagnosis when the defining pathology/microbiology is absent from the prompt.
4. required_findings must list the specific findings in challenge_prompt that drive the diagnosis,
   and every item listed there MUST literally appear in challenge_prompt.

Return JSON with exactly these top-level keys:
- challenge_prompt: string
- answer_key: object with diagnosis, aliases, etiology, next_management_step, required_findings, optional_findings
- evidence_map: array of concise evidence-to-answer summaries
- hypothesis_bank: array of plausible alternative diagnoses/actions to explore
- outcome_summary: concise final course/outcome, excluded from challenge_prompt
- leakage_audit: object with has_leakage boolean, suspected_leakage_terms array, explanation string
- adequacy_audit: object with is_self_contained boolean, missing_starting_info array, transformation_notes array
- fidelity_audit: object with all_findings_sourced boolean, unsourced_or_altered_findings array (any statement in challenge_prompt not directly supported by the source), values_match_source boolean
- solvability_audit: object with is_solvable_from_prompt boolean, required_results_missing_from_prompt array (deciding results the question needs but the prompt lacks), question_matches_available_evidence boolean
- source_usage: object with used_existing_prompt boolean, used_answer_rest boolean, used_full_article boolean

Case metadata:
case_id: {row.get("case_id")}
source_kind: {row.get("source_kind")}
pmcid: {row.get("pmcid")}
doi: {row.get("doi")}
journal: {row.get("journal")}
article_title_do_not_leak_into_prompt: {row.get("title")}

Existing crude challenge prompt:
{row.get("challenge_prompt", "")}

Existing answer/discussion material:
{row.get("answer_rest", "")}

Full article text extracted from XML:
{article_text}
"""
    return system_prompt, user_prompt


def normalize_refined_content(content: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key in REFINED_SCHEMA_KEYS:
        if key in {"evidence_map", "hypothesis_bank"}:
            output[key] = content.get(key) if isinstance(content.get(key), list) else []
        elif key in {"answer_key", "leakage_audit", "adequacy_audit", "fidelity_audit", "solvability_audit", "source_usage"}:
            output[key] = content.get(key) if isinstance(content.get(key), dict) else {}
        else:
            output[key] = content.get(key, "") if isinstance(content.get(key, ""), str) else ""
    return output


def extract_article_text_from_xml(xml_path: Path, *, max_chars: int = 45000) -> str:
    root = ET.fromstring(xml_path.read_bytes())
    article = _first_by_local_name(root, "article")
    if article is None:
        return ""
    body = _first_child_by_local_name(article, "body")
    if body is None:
        return ""
    chunks = []
    for section in body.iter():
        if _local_name(section.tag) == "sec":
            title = _section_title(section)
            if title:
                chunks.append(f"\n## {title}\n")
            paragraphs = []
            for child in list(section):
                if _local_name(child.tag) == "p":
                    text = _clean_text(child)
                    if text:
                        paragraphs.append(text)
            if paragraphs:
                chunks.append("\n\n".join(paragraphs))
    text = "\n\n".join(chunks)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[TRUNCATED_FOR_MODEL_CONTEXT]"
    return text


def _refine_one(
    *,
    client: DeepSeekClient,
    row: dict[str, Any],
    model: str,
    dry_run: bool,
    temperature: float,
    extra_body: dict[str, Any] | None,
    max_article_chars: int,
    include_article_text: bool,
    api_retries: int,
    api_retry_sleep_seconds: float,
    solvability_probe_model: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    xml_path = Path(str(row.get("source_xml_path") or ""))
    article_text = (
        extract_article_text_from_xml(xml_path, max_chars=max_article_chars)
        if include_article_text and xml_path.exists()
        else ""
    )
    system_prompt, user_prompt = build_public_refinement_prompt(row, article_text)
    if dry_run:
        content = normalize_refined_content({})
        raw_record = {
            "type": "refine",
            "dry_run": True,
            "case_id": row.get("case_id"),
            "model": model,
            "system_prompt_chars": len(system_prompt),
            "user_prompt_chars": len(user_prompt),
            "article_text_chars": len(article_text),
        }
        review_status = "dry_run"
        validation_reasons: list[str] = []
        probe_result: dict[str, Any] = {}
    else:
        errors = []
        probe_result = {}
        try:
            for attempt in range(api_retries + 1):
                try:
                    raw_record = client.chat_json(
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        extra_body=extra_body,
                    )
                    raw_record["attempt"] = attempt + 1
                    break
                except Exception as exc:  # noqa: BLE001 - record transient API failures per case.
                    errors.append(str(exc))
                    if attempt >= api_retries:
                        raise
                    time.sleep(api_retry_sleep_seconds * (attempt + 1))
            raw_record["type"] = "refine"
            raw_record["case_id"] = row.get("case_id")
            if errors:
                raw_record["retry_errors"] = errors
            content = normalize_refined_content(raw_record.get("parsed_content", {}))
            review_status, validation_reasons = _review_status_with_reasons(content, article_text)
            if solvability_probe_model and review_status == "refined_needs_spotcheck":
                diagnosis = ""
                answer_key = content.get("answer_key")
                if isinstance(answer_key, dict) and isinstance(answer_key.get("diagnosis"), str):
                    diagnosis = answer_key["diagnosis"]
                if diagnosis and content.get("challenge_prompt"):
                    try:
                        probe_result = solvability_probe(
                            client=client,
                            model=solvability_probe_model,
                            challenge_prompt=str(content["challenge_prompt"]),
                            expected_diagnosis=diagnosis,
                            temperature=0.0,
                        )
                        raw_record["solvability_probe"] = probe_result
                        if probe_result.get("is_solvable") is False:
                            validation_reasons.append(
                                "solvability_probe_failed:"
                                + str(probe_result.get("probe_diagnosis", ""))[:80]
                            )
                            review_status = "not_solvable"
                    except Exception as exc:  # noqa: BLE001 - preserve refinement output; flag for review.
                        probe_result = {"error": str(exc)}
                        raw_record["solvability_probe"] = probe_result
                        validation_reasons.append("solvability_probe_error:" + str(exc)[:120])
                        review_status = "needs_probe_review"
        except Exception as exc:  # noqa: BLE001 - preserve long-run progress and record per-case failure.
            content = normalize_refined_content({})
            raw_record = {
                "type": "error",
                "case_id": row.get("case_id"),
                "model": model,
                "error": str(exc),
                "retry_errors": errors,
            }
            review_status = "api_error"
            validation_reasons = []
            probe_result = {}

    artifact = {
        "case_id": row.get("case_id"),
        "source_kind": row.get("source_kind"),
        "pmcid": row.get("pmcid"),
        "doi": row.get("doi"),
        "title": row.get("title"),
        "journal": row.get("journal"),
        "license_key": row.get("license_key"),
        "license_tier": row.get("license_tier"),
        "source_xml_path": row.get("source_xml_path"),
        "model": model,
        "review_status": review_status,
        "validation_reasons": validation_reasons,
        "solvability_probe": probe_result,
        "article_text_chars_used": len(article_text),
        **content,
    }
    return artifact, raw_record


def _review_status(content: dict[str, Any], article_text: str = "") -> str:
    return _review_status_with_reasons(content, article_text)[0]


def _review_status_with_reasons(content: dict[str, Any], article_text: str = "") -> tuple[str, list[str]]:
    """Decide review status and collect machine-checkable validation reasons.

    Beyond the model's own audits (which wrongly passed both known-defective cases), this runs
    deterministic checks against the SOURCE article: every answer_key.required_finding must appear
    in challenge_prompt (catches dropped discriminators), and distinctive numeric values in the
    prompt must appear in the source (catches fabricated/altered findings).
    """

    reasons = validate_refinement(content, article_text)
    if not content.get("challenge_prompt") or not content.get("answer_key"):
        return "invalid_refinement", reasons

    leakage = content.get("leakage_audit") if isinstance(content.get("leakage_audit"), dict) else {}
    adequacy = content.get("adequacy_audit") if isinstance(content.get("adequacy_audit"), dict) else {}
    fidelity = content.get("fidelity_audit") if isinstance(content.get("fidelity_audit"), dict) else {}
    solvability = content.get("solvability_audit") if isinstance(content.get("solvability_audit"), dict) else {}

    if leakage.get("has_leakage") is True:
        return "needs_leakage_review", reasons
    if adequacy.get("is_self_contained") is False:
        return "not_self_contained", reasons
    # Self-reported or deterministically-detected solvability problems (asked-question can't be
    # answered from the prompt -- the intrarenal-neurofibroma defect).
    if solvability.get("is_solvable_from_prompt") is False or any(
        r.startswith(("not_solvable", "required_finding_absent")) for r in reasons
    ):
        return "not_solvable", reasons
    # Self-reported or deterministically-detected fabrication/value drift (the NPSLE defect:
    # dropped serologies + fabricated MRI + altered CSF numbers).
    if (
        fidelity.get("all_findings_sourced") is False
        or fidelity.get("values_match_source") is False
        or any(r.startswith(("unsourced_value", "possible_polarity_conflict")) for r in reasons)
    ):
        return "needs_fidelity_review", reasons
    return "refined_needs_spotcheck", reasons


_DISTINCTIVE_VALUE_RE = re.compile(
    r"\d+\s*:\s*\d+"          # titers e.g. 1:1280
    r"|[<>]\s*\d[\d,.]*"      # comparators e.g. >300
    r"|\d[\d,]*\.\d+"         # decimals e.g. 52.1
    r"|\b\d{3,}\b"            # 3+ digit integers e.g. 1280
)


def validate_refinement(content: dict[str, Any], article_text: str = "") -> list[str]:
    """Machine-checkable defects, independent of the model's self-audit."""

    reasons: list[str] = []
    prompt = content.get("challenge_prompt") or ""
    if not isinstance(prompt, str) or not prompt:
        return reasons
    answer_key = content.get("answer_key") if isinstance(content.get("answer_key"), dict) else {}

    # (1) Omission: each required finding must be reflected in the prompt.
    required = answer_key.get("required_findings")
    if isinstance(required, list):
        prompt_tokens = _content_tokens(prompt)
        for finding in required:
            if not isinstance(finding, str) or not finding.strip():
                continue
            finding_tokens = _content_tokens(finding)
            if not finding_tokens:
                continue
            overlap = len(finding_tokens & prompt_tokens) / len(finding_tokens)
            if overlap < 0.34 and not _numeric_paraphrase_present(finding, prompt):
                # The deciding finding is essentially absent from the prompt. Numeric demographic
                # paraphrases like "Age 20 years" vs "20-year-old" are allowed because they are
                # not the dropped-discriminator failure mode this guard is meant to catch.
                reasons.append(f"required_finding_absent_from_prompt:{finding[:60]}")

    # (2) Fabrication / value drift: distinctive numeric values in the prompt must be in the source.
    if article_text:
        source_digits = {re.sub(r"[^0-9]", "", m) for m in _DISTINCTIVE_VALUE_RE.findall(article_text)}
        for value in _DISTINCTIVE_VALUE_RE.findall(prompt):
            digits = re.sub(r"[^0-9]", "", value)
            if digits and digits not in source_digits:
                reasons.append(f"unsourced_value_in_prompt:{value.strip()[:24]}")
        reasons.extend(_possible_polarity_conflicts(prompt, article_text))

    # (3) Specificity-determinacy: if the diagnosis names a specific gene/antibody/molecular entity,
    # the prompt must contain the result that identifies it (sequencing variant, antibody serology,
    # IHC). Otherwise the challenge asks for an entity not inferable from a phenotype shared with
    # near-neighbors (the broken-case pattern from ClinicalHarness 2026-06-15). Reward the determinable
    # level instead. Lightweight token check: a distinctive gene-symbol / antibody token in the
    # diagnosis that is absent from the prompt is flagged.
    diagnosis = answer_key.get("diagnosis") if isinstance(answer_key.get("diagnosis"), str) else ""
    if diagnosis:
        for token in _SPECIFICITY_TOKEN_RE.findall(diagnosis):
            if len(token) >= 3 and token not in _SPECIFICITY_GENERIC and token.lower() not in prompt.lower():
                reasons.append(f"specificity_unsupported:{token}")

    return reasons


# Distinctive specific-entity tokens (gene symbols like SLC6A1/ATP1A3/KCNMA1, antibody names) whose
# absence from the prompt means the named diagnosis is not determinable from the vignette.
_SPECIFICITY_TOKEN_RE = re.compile(r"\b([A-Z][A-Z0-9]{2,}\d?[A-Z]?|[A-Z]{2,}\d)\b")
_SPECIFICITY_GENERIC = frozenset(
    {"MRI", "CT", "EEG", "CSF", "PCR", "ALS", "ADHD", "DNA", "RNA", "CNS", "ICU", "NICU", "PET", "FDG",
     "ESR", "CRP", "MCI", "MSA", "MERS", "HSV", "SLE", "TB", "NPH", "IUD"}
)


_TEST_NAME_RE = re.compile(
    r"\b(MRI|CT|EEG|EMG|CSF|lumbar puncture|LP|ANA|anti[- ]dsDNA|biopsy|histolog\w*|IHC|immunohistochem\w*)\b",
    re.I,
)
_NORMAL_NEGATIVE_RE = re.compile(r"\b(normal|negative|unremarkable|without|no evidence of|no abnormal)\b", re.I)
_ABNORMAL_RE = re.compile(
    r"\b(abnormal|positive|elevated|increased|decreased|hyperintens\w*|hypointens\w*|lesion|enhanc\w*|pleocytosis|oligoclonal|spindle|atypia|mass|infiltrat\w*)\b",
    re.I,
)


def _possible_polarity_conflicts(prompt: str, article_text: str) -> list[str]:
    """Flag possible normal/negative source tests rewritten as abnormal prompt findings.

    This is a review flag, not proof of fabrication. Articles can report multiple serial studies,
    but this catches high-risk drift such as a normal MRI becoming a white-matter-lesion clue.
    """

    source_sentences = _sentences(article_text)
    prompt_sentences = _sentences(prompt)
    conflicts: list[str] = []
    for prompt_sentence in prompt_sentences:
        if not (_ABNORMAL_RE.search(prompt_sentence) and _TEST_NAME_RE.search(prompt_sentence)):
            continue
        tests = {match.group(0).lower() for match in _TEST_NAME_RE.finditer(prompt_sentence)}
        for test in tests:
            test_source_sentences = [s for s in source_sentences if test in s.lower()]
            if not test_source_sentences:
                continue
            has_negative_source = any(_NORMAL_NEGATIVE_RE.search(s) for s in test_source_sentences)
            has_abnormal_source = any(_ABNORMAL_RE.search(s) for s in test_source_sentences)
            if has_negative_source and not has_abnormal_source:
                conflicts.append(f"possible_polarity_conflict:{test}:{prompt_sentence[:80]}")
                break
    return conflicts[:4]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


_VALIDATION_STOPWORDS = frozenset(
    {
        "a", "an", "and", "of", "the", "with", "without", "for", "to", "in", "on", "was", "were",
        "is", "are", "had", "has", "no", "not", "patient", "showed", "revealed", "positive",
        "negative", "normal", "elevated", "test", "testing", "level", "levels", "result", "results",
    }
)


def _content_tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {t for t in raw if len(t) > 2 and t not in _VALIDATION_STOPWORDS}


def _numeric_paraphrase_present(finding: str, prompt: str) -> bool:
    finding_numbers = {re.sub(r"[^0-9]", "", n) for n in re.findall(r"\d+(?:\.\d+)?", finding)}
    finding_numbers.discard("")
    if not finding_numbers:
        return False
    prompt_numbers = {re.sub(r"[^0-9]", "", n) for n in re.findall(r"\d+(?:\.\d+)?", prompt)}
    if not finding_numbers <= prompt_numbers:
        return False
    # Only relax when the finding is mostly demographic/time wording, not a lab/pathology clue.
    nonnumeric_tokens = _content_tokens(re.sub(r"\d+(?:\.\d+)?", " ", finding))
    demographic_tokens = {
        "age", "year", "years", "old", "male", "female", "man", "woman", "boy", "girl",
        "month", "months", "day", "days", "week", "weeks",
    }
    return nonnumeric_tokens <= demographic_tokens


def solvability_probe(
    *,
    client: DeepSeekClient,
    model: str,
    challenge_prompt: str,
    expected_diagnosis: str,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Independent check that the answer is reachable from the prompt alone.

    Gives a strong model ONLY the challenge_prompt (no answer key), asks for the most likely
    diagnosis and next step, then asks it to judge whether its blind answer is clinically equivalent
    to the expected diagnosis. If a capable model cannot reach the answer from the prompt, the prompt
    is underdetermined (the intrarenal-neurofibroma / NPSLE failure mode) and should go to human
    review rather than into the benchmark. Returns {probe_diagnosis, is_solvable, rationale}.

    This is the strongest available guardrail and is intended to gate the final benchmark export; it
    costs one to two model calls per case, so it is opt-in rather than part of every refinement.
    """

    answer = client.chat_json(
        model=model,
        system_prompt=(
            "You are a careful diagnostician taking a closed-book exam. Use only the information in "
            "the case. Return only JSON."
        ),
        user_prompt=(
            "Given only this case, what is the single most likely diagnosis and the next best step? "
            'Return JSON: {"diagnosis": "...", "next_step": "..."}.\n\nCase:\n' + challenge_prompt
        ),
        temperature=temperature,
    )
    probe_dx = ""
    parsed = answer.get("parsed_content") if isinstance(answer, dict) else None
    if isinstance(parsed, dict) and isinstance(parsed.get("diagnosis"), str):
        probe_dx = parsed["diagnosis"]

    verdict = client.chat_json(
        model=model,
        system_prompt=(
            "You are a strict diagnostic-equivalence judge. Two phrasings are equivalent if they name "
            "the same disease entity (allow missing qualifiers/synonyms); different species, genetic "
            "subtype, lineage, or entity are NOT equivalent. Return only JSON."
        ),
        user_prompt=(
            'Return JSON {"equivalent": true|false, "rationale": "..."}.\n'
            f"answer_key_diagnosis: {expected_diagnosis}\n"
            f"blind_probe_diagnosis: {probe_dx}\n"
        ),
        temperature=0.0,
    )
    vparsed = verdict.get("parsed_content") if isinstance(verdict, dict) else None
    equivalent = bool(vparsed.get("equivalent")) if isinstance(vparsed, dict) else False
    rationale = vparsed.get("rationale") if isinstance(vparsed, dict) and isinstance(vparsed.get("rationale"), str) else ""
    return {"probe_diagnosis": probe_dx, "is_solvable": equivalent, "rationale": rationale}


def _completed_case_ids(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row.get("case_id")) for row in rows if row.get("case_id") and row.get("review_status") != "api_error"}


def _tsv_row(artifact: dict[str, Any]) -> dict[str, str]:
    return {
        field: _stringify(artifact.get(field, ""))
        for field in REFINED_FIELDS
    }


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_by_local_name(element: ET.Element, name: str) -> ET.Element | None:
    if _local_name(element.tag) == name:
        return element
    for child in element.iter():
        if _local_name(child.tag) == name:
            return child
    return None


def _first_child_by_local_name(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    for child in list(element):
        if _local_name(child.tag) == name:
            return child
    return None


def _section_title(section: ET.Element) -> str:
    title = _first_child_by_local_name(section, "title")
    return _clean_text(title) if title is not None else ""


def _clean_text(element: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
