"""DeepSeek evaluation over public case challenge / answer split manifests."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .case_eval import new_run_id
from .concurrency import run_ordered_concurrent
from .deepseek import DeepSeekClient


PUBLIC_EVAL_RESULT_FIELDS = [
    "selection_rank",
    "case_id",
    "source_kind",
    "pmcid",
    "doi",
    "title",
    "journal",
    "license_key",
    "model",
    "final_diagnosis",
    "etiology",
    "top_differential",
    "recommended_next_step",
    "confidence",
    "evidence_summary",
    "uncertainty_or_missing_information",
    "judge_score_status",
    "judge_rationale",
    "review_status",
]

PUBLIC_SCORE_FIELDS = [
    "selection_rank",
    "case_id",
    "source_kind",
    "pmcid",
    "doi",
    "title",
    "journal",
    "license_key",
    "model",
    "final_diagnosis",
    "recommended_next_step",
    "score_status",
    "diagnosis_status",
    "next_step_status",
    "rationale_status",
    "expected_key_answer",
    "expected_next_step",
    "rationale",
    "answer_schema_valid",
    "answer_schema_errors",
    "review_status",
]

VALID_SCORE_STATUSES = {"pass", "partial", "fail", "ungradable"}
VALID_COMPONENT_STATUSES = {"correct", "partial", "incorrect", "not_applicable", "ungradable"}
ANSWER_REQUIRED_KEYS = {
    "final_diagnosis": str,
    "etiology": str,
    "top_differential": list,
    "recommended_next_step": str,
    "confidence": (int, float),
    "evidence_summary": list,
    "uncertainty_or_missing_information": list,
}


def load_public_splits(path: Path, *, limit: int | None = None, case_ids: set[str] | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if case_ids:
        rows = [row for row in rows if str(row.get("case_id")) in case_ids]
    if limit is not None:
        rows = rows[:limit]
    return rows


def build_public_case_prompt(row: dict[str, Any]) -> tuple[str, str]:
    system_prompt = (
        "You are evaluating a public medical case challenge. Return only valid JSON. "
        "Do not search for the article, DOI, title, PMCID, or publication. "
        "Do not provide hidden chain-of-thought; provide concise evidence summaries only."
    )
    user_prompt = f"""Based only on the clinical challenge below, identify the most likely diagnosis or etiology and the most appropriate next diagnostic or treatment step if applicable.

Return JSON with exactly these keys:
- final_diagnosis: string
- etiology: string
- top_differential: array of strings
- recommended_next_step: string
- confidence: number from 0 to 1
- evidence_summary: array of concise strings
- uncertainty_or_missing_information: array of concise strings

Clinical challenge:
{row["challenge_prompt"]}
"""
    return system_prompt, user_prompt


def build_judge_prompt(row: dict[str, Any], model_answer: dict[str, Any]) -> tuple[str, str]:
    system_prompt = (
        "You grade medical case challenge answers against a reference answer. Return only valid JSON. "
        "Be strict about the final diagnosis or key management answer, but allow synonymous disease names. "
        "Do not use outside knowledge or the article title as the answer key; grade only against the reference answer."
    )
    user_prompt = f"""Grade whether the model answer matches the reference answer.

Return JSON with exactly these keys:
- score_status: one of "pass", "partial", "fail", "ungradable"
- diagnosis_status: one of "correct", "partial", "incorrect", "not_applicable", "ungradable"
- next_step_status: one of "correct", "partial", "incorrect", "not_applicable", "ungradable"
- rationale_status: one of "correct", "partial", "incorrect", "not_applicable", "ungradable"
- rationale: concise explanation
- expected_key_answer: string
- expected_next_step: string
- model_key_answer: string
- rationale_errors: array of concise strings for unsupported or wrong reasoning, empty if none

Reference answer/rest:
{row["answer_rest"]}

Model answer:
{json.dumps(model_answer, ensure_ascii=False, sort_keys=True)}
"""
    return system_prompt, user_prompt


def run_public_deepseek_eval(
    *,
    client: DeepSeekClient,
    manifest_path: Path,
    out_root: Path,
    model: str,
    limit: int | None,
    case_ids: set[str] | None,
    dry_run: bool,
    temperature: float,
    judge: bool,
    judge_model: str | None,
    extra_body: dict[str, Any] | None = None,
    resume_results: Path | None = None,
    concurrency: int = 1,
    request_spacing_seconds: float = 0.0,
) -> dict[str, Any]:
    rows = load_public_splits(manifest_path, limit=limit, case_ids=case_ids)
    completed_case_ids = _completed_case_ids(resume_results)
    if completed_case_ids:
        rows = [row for row in rows if str(row.get("case_id")) not in completed_case_ids]
    if not rows:
        raise ValueError("No public case split rows selected.")

    run_id = new_run_id()
    run_dir = out_root / f"deepseek_public_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    judge_model_name = judge_model or model
    results_path = run_dir / "results.tsv"
    raw_path = run_dir / "raw_api_records.jsonl"

    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PUBLIC_EVAL_RESULT_FIELDS, delimiter="\t")
        writer.writeheader()
        with raw_path.open("w", encoding="utf-8") as raw_file:
            def worker(row: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
                return _evaluate_one_public_case(
                    client=client,
                    row=row,
                    model=model,
                    dry_run=dry_run,
                    temperature=temperature,
                    judge=judge,
                    judge_model_name=judge_model_name,
                    extra_body=extra_body,
                )

            for result_row, raw_records in run_ordered_concurrent(
                rows,
                worker,
                concurrency=concurrency,
                request_spacing_seconds=request_spacing_seconds,
            ):
                writer.writerow(result_row)
                file.flush()
                for record in raw_records:
                    raw_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                raw_file.flush()
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "input_manifest": str(manifest_path),
        "selected_case_count": len(rows),
        "skipped_completed_case_count": len(completed_case_ids),
        "resume_results": str(resume_results) if resume_results else None,
        "model": model,
        "judge": judge,
        "judge_model": judge_model_name if judge else None,
        "temperature": temperature,
        "extra_body": extra_body or {},
        "concurrency": concurrency,
        "request_spacing_seconds": request_spacing_seconds,
        "results_path": str(results_path),
        "raw_path": str(raw_path),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"run_dir": str(run_dir), **manifest}


def score_public_deepseek_results(
    *,
    client: DeepSeekClient,
    split_manifest_path: Path,
    results_path: Path,
    out_root: Path,
    judge_model: str,
    dry_run: bool,
    temperature: float,
    extra_body: dict[str, Any] | None = None,
    limit: int | None = None,
    case_ids: set[str] | None = None,
    resume_scores: Path | None = None,
    concurrency: int = 1,
    request_spacing_seconds: float = 0.0,
) -> dict[str, Any]:
    split_rows = {
        str(row.get("case_id")): row
        for row in load_public_splits(split_manifest_path, case_ids=case_ids)
    }
    result_rows = load_public_results(results_path, limit=limit, case_ids=case_ids)
    completed_case_ids = _completed_case_ids(resume_scores)
    if completed_case_ids:
        result_rows = [row for row in result_rows if row.get("case_id") not in completed_case_ids]
    if not result_rows:
        raise ValueError("No public result rows selected for scoring.")

    missing = sorted({row.get("case_id", "") for row in result_rows} - set(split_rows))
    if missing:
        raise ValueError(f"Results contain case IDs missing from split manifest: {missing[:10]}")

    run_id = new_run_id()
    run_dir = out_root / f"deepseek_public_scores_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    scores_path = run_dir / "scores.tsv"
    raw_path = run_dir / "raw_judge_records.jsonl"

    with scores_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PUBLIC_SCORE_FIELDS, delimiter="\t")
        writer.writeheader()
        with raw_path.open("w", encoding="utf-8") as raw_file:
            def worker(result_row: dict[str, str]) -> tuple[dict[str, str], dict[str, Any]]:
                split_row = split_rows[result_row["case_id"]]
                return _score_one_public_result(
                    client=client,
                    split_row=split_row,
                    result_row=result_row,
                    judge_model=judge_model,
                    dry_run=dry_run,
                    temperature=temperature,
                    extra_body=extra_body,
                )

            for score_row, raw_record in run_ordered_concurrent(
                result_rows,
                worker,
                concurrency=concurrency,
                request_spacing_seconds=request_spacing_seconds,
            ):
                writer.writerow(score_row)
                file.flush()
                raw_file.write(json.dumps(raw_record, ensure_ascii=False, sort_keys=True) + "\n")
                raw_file.flush()

    score_rows = load_public_scores(scores_path)
    metrics = summarize_public_scores(score_rows)
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "split_manifest": str(split_manifest_path),
        "results_path": str(results_path),
        "selected_case_count": len(result_rows),
        "skipped_completed_case_count": len(completed_case_ids),
        "resume_scores": str(resume_scores) if resume_scores else None,
        "judge_model": judge_model,
        "temperature": temperature,
        "extra_body": extra_body or {},
        "concurrency": concurrency,
        "request_spacing_seconds": request_spacing_seconds,
        "scores_path": str(scores_path),
        "raw_path": str(raw_path),
        "metrics": metrics,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"run_dir": str(run_dir), **manifest}


def load_public_results(path: Path, *, limit: int | None = None, case_ids: set[str] | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    if case_ids:
        rows = [row for row in rows if row.get("case_id") in case_ids]
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_public_scores(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def rebuild_public_results_from_raw(
    *,
    split_manifest_path: Path,
    raw_records_path: Path,
    output_tsv: Path,
    case_ids: set[str] | None = None,
) -> dict[str, Any]:
    split_rows = {
        str(row.get("case_id")): row
        for row in load_public_splits(split_manifest_path, case_ids=case_ids)
    }
    raw_by_case: dict[str, dict[str, Any]] = {}
    for line in raw_records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") != "answer":
            continue
        case_id = str(record.get("case_id") or "")
        if case_id and case_id in split_rows:
            raw_by_case[case_id] = record

    missing_case_ids = sorted(set(split_rows) - set(raw_by_case))
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PUBLIC_EVAL_RESULT_FIELDS, delimiter="\t")
        writer.writeheader()
        for case_id, row in split_rows.items():
            raw_record = raw_by_case.get(case_id)
            if not raw_record:
                continue
            writer.writerow(_result_row(row, raw_record.get("model", ""), raw_record.get("parsed_content", {}), {}))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_manifest": str(split_manifest_path),
        "raw_records": str(raw_records_path),
        "output_tsv": str(output_tsv),
        "selected_case_count": len(split_rows),
        "rebuilt_case_count": len(split_rows) - len(missing_case_ids),
        "missing_case_count": len(missing_case_ids),
        "missing_case_ids": missing_case_ids,
        "fieldnames": PUBLIC_EVAL_RESULT_FIELDS,
    }
    summary_path = output_tsv.with_name(output_tsv.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def merge_public_score_files(
    *,
    score_paths: list[Path],
    output_tsv: Path,
) -> dict[str, Any]:
    merged_by_case: dict[str, dict[str, str]] = {}
    replacements: list[str] = []
    for score_path in score_paths:
        for row in load_public_scores(score_path):
            case_id = row.get("case_id", "")
            if not case_id:
                continue
            if case_id in merged_by_case:
                replacements.append(case_id)
            normalized_row = dict(row)
            if (
                normalized_row.get("score_status") == "ungradable"
                and normalized_row.get("review_status") == "judge_api_error"
                and not _looks_like_api_error(normalized_row.get("rationale", ""))
            ):
                normalized_row["review_status"] = "judge_scored_needs_spotcheck"
            merged_by_case[case_id] = normalized_row

    rows = sorted(
        merged_by_case.values(),
        key=lambda row: int(row.get("selection_rank") or 0),
    )
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PUBLIC_SCORE_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PUBLIC_SCORE_FIELDS})

    metrics = summarize_public_scores(rows)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "score_paths": [str(path) for path in score_paths],
        "output_tsv": str(output_tsv),
        "row_count": len(rows),
        "replacement_case_ids": replacements,
        "metrics": metrics,
    }
    metrics_path = output_tsv.with_name(output_tsv.stem + "_metrics.json")
    summary_path = output_tsv.with_name(output_tsv.stem + "_summary.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["metrics_path"] = str(metrics_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def summarize_public_scores(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)

    def counts(field: str) -> dict[str, int]:
        output: dict[str, int] = {}
        for row in rows:
            value = row.get(field, "") or ""
            output[value] = output.get(value, 0) + 1
        return dict(sorted(output.items()))

    score_counts = counts("score_status")
    diagnosis_counts = counts("diagnosis_status")
    next_step_counts = counts("next_step_status")
    rationale_counts = counts("rationale_status")
    schema_counts = counts("answer_schema_valid")

    def rate(numerator: int) -> float:
        return round(numerator / total, 4) if total else 0.0

    pass_count = score_counts.get("pass", 0)
    partial_count = score_counts.get("partial", 0)
    fail_count = score_counts.get("fail", 0)
    ungradable_count = score_counts.get("ungradable", 0)
    diagnosis_correct = diagnosis_counts.get("correct", 0)
    diagnosis_partial = diagnosis_counts.get("partial", 0)
    next_step_correct = next_step_counts.get("correct", 0)
    next_step_partial = next_step_counts.get("partial", 0)
    rationale_correct = rationale_counts.get("correct", 0)
    rationale_partial = rationale_counts.get("partial", 0)
    return {
        "total": total,
        "score_status_counts": score_counts,
        "diagnosis_status_counts": diagnosis_counts,
        "next_step_status_counts": next_step_counts,
        "rationale_status_counts": rationale_counts,
        "answer_schema_valid_counts": schema_counts,
        "pass_rate": rate(pass_count),
        "pass_or_partial_rate": rate(pass_count + partial_count),
        "fail_rate": rate(fail_count),
        "ungradable_rate": rate(ungradable_count),
        "diagnosis_correct_rate": rate(diagnosis_correct),
        "diagnosis_correct_or_partial_rate": rate(diagnosis_correct + diagnosis_partial),
        "next_step_correct_rate": rate(next_step_correct),
        "next_step_correct_or_partial_rate": rate(next_step_correct + next_step_partial),
        "rationale_correct_rate": rate(rationale_correct),
        "rationale_correct_or_partial_rate": rate(rationale_correct + rationale_partial),
    }


def _evaluate_one_public_case(
    *,
    client: DeepSeekClient,
    row: dict[str, Any],
    model: str,
    dry_run: bool,
    temperature: float,
    judge: bool,
    judge_model_name: str,
    extra_body: dict[str, Any] | None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    system_prompt, user_prompt = build_public_case_prompt(row)
    raw_records = []
    if dry_run:
        parsed_content = _empty_answer()
        raw_record = {
            "case_id": row.get("case_id"),
            "dry_run": True,
            "model": model,
            "system_prompt_chars": len(system_prompt),
            "user_prompt_chars": len(user_prompt),
        }
        judge_content = {}
    else:
        try:
            raw_record = client.chat_json(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                extra_body=extra_body,
            )
            raw_record["case_id"] = row.get("case_id")
            parsed_content = raw_record.get("parsed_content", {})
            judge_content = {}
            raw_records.append({"type": "answer", **raw_record})
            if judge:
                judge_system, judge_user = build_judge_prompt(row, parsed_content)
                judge_record = client.chat_json(
                    model=judge_model_name,
                    system_prompt=judge_system,
                    user_prompt=judge_user,
                    temperature=0.0,
                    extra_body=extra_body,
                )
                judge_record["case_id"] = row.get("case_id")
                judge_record["judge_for_model"] = model
                judge_content = judge_record.get("parsed_content", {})
                raw_records.append({"type": "judge", **judge_record})
        except Exception as exc:  # noqa: BLE001 - preserve long-run progress and record per-case failure.
            parsed_content = _empty_answer()
            judge_content = {}
            raw_records.append(
                {
                    "type": "error",
                    "case_id": row.get("case_id"),
                    "model": model,
                    "error": str(exc),
                }
            )
            result = _result_row(row, model, parsed_content, judge_content)
            result["review_status"] = "api_error"
            result["judge_rationale"] = str(exc)
            return result, raw_records

    if dry_run:
        raw_records.append({"type": "answer", **raw_record})
    return _result_row(row, model, parsed_content, judge_content), raw_records


def _score_one_public_result(
    *,
    client: DeepSeekClient,
    split_row: dict[str, Any],
    result_row: dict[str, str],
    judge_model: str,
    dry_run: bool,
    temperature: float,
    extra_body: dict[str, Any] | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    model_answer = _model_answer_from_result_row(result_row)
    schema_valid, schema_errors = validate_answer_schema(model_answer)
    if dry_run:
        judge_content = {
            "score_status": "ungradable",
            "diagnosis_status": "ungradable",
            "next_step_status": "ungradable",
            "rationale": "Dry run; no judge call made.",
            "expected_key_answer": "",
            "expected_next_step": "",
        }
        raw_record = {
            "type": "judge",
            "dry_run": True,
            "case_id": result_row.get("case_id"),
            "model": judge_model,
        }
    else:
        judge_system, judge_user = build_judge_prompt(split_row, model_answer)
        try:
            raw_record = client.chat_json(
                model=judge_model,
                system_prompt=judge_system,
                user_prompt=judge_user,
                temperature=temperature,
                extra_body=extra_body,
            )
            raw_record["type"] = "judge"
            raw_record["case_id"] = result_row.get("case_id")
            raw_record["judge_for_model"] = result_row.get("model")
            judge_content = raw_record.get("parsed_content", {})
        except Exception as exc:  # noqa: BLE001 - preserve long-run progress and record per-case failure.
            judge_content = {
                "score_status": "ungradable",
                "diagnosis_status": "ungradable",
                "next_step_status": "ungradable",
                "rationale": str(exc),
                "expected_key_answer": "",
                "expected_next_step": "",
            }
            raw_record = {
                "type": "error",
                "case_id": result_row.get("case_id"),
                "model": judge_model,
                "error": str(exc),
            }

    normalized_judge = normalize_judge_content(judge_content)
    return _score_row(result_row, normalized_judge, schema_valid, schema_errors), raw_record


def _completed_case_ids(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        return {
            row["case_id"]
            for row in reader
            if row.get("case_id") and row.get("review_status") not in {"api_error", "judge_api_error"}
        }


def _model_answer_from_result_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "final_diagnosis": row.get("final_diagnosis", ""),
        "etiology": row.get("etiology", ""),
        "top_differential": _parse_jsonish(row.get("top_differential", ""), fallback=[]),
        "recommended_next_step": row.get("recommended_next_step", ""),
        "confidence": _parse_jsonish(row.get("confidence", ""), fallback=row.get("confidence", "")),
        "evidence_summary": _parse_jsonish(row.get("evidence_summary", ""), fallback=[]),
        "uncertainty_or_missing_information": _parse_jsonish(
            row.get("uncertainty_or_missing_information", ""),
            fallback=[],
        ),
    }


def validate_answer_schema(model_answer: dict[str, Any]) -> tuple[bool, list[str]]:
    errors = []
    for key, expected_type in ANSWER_REQUIRED_KEYS.items():
        if key not in model_answer:
            errors.append(f"missing:{key}")
            continue
        value = model_answer[key]
        if isinstance(expected_type, tuple):
            valid = isinstance(value, expected_type)
        else:
            valid = isinstance(value, expected_type)
        if not valid:
            errors.append(f"type:{key}")
    confidence = model_answer.get("confidence")
    if isinstance(confidence, int | float) and not 0 <= confidence <= 1:
        errors.append("range:confidence")
    return not errors, errors


def normalize_judge_content(content: dict[str, Any]) -> dict[str, str]:
    score_status = str(content.get("score_status", "ungradable")).lower()
    if score_status not in VALID_SCORE_STATUSES:
        score_status = "ungradable"
    return {
        "score_status": score_status,
        "diagnosis_status": _component_status(content.get("diagnosis_status", "ungradable")),
        "next_step_status": _component_status(content.get("next_step_status", "ungradable")),
        "rationale_status": _component_status(content.get("rationale_status", "ungradable")),
        "rationale": _stringify(content.get("rationale", "")),
        "expected_key_answer": _stringify(
            content.get("expected_key_answer", content.get("expected_diagnosis_or_key_answer", ""))
        ),
        "expected_next_step": _stringify(content.get("expected_next_step", "")),
    }


def _score_row(
    result_row: dict[str, str],
    judge_content: dict[str, str],
    schema_valid: bool,
    schema_errors: list[str],
) -> dict[str, str]:
    review_status = "judge_scored_needs_spotcheck"
    if judge_content["score_status"] == "ungradable" and _looks_like_api_error(judge_content.get("rationale", "")):
        review_status = "judge_api_error"
    return {
        "selection_rank": result_row.get("selection_rank", ""),
        "case_id": result_row.get("case_id", ""),
        "source_kind": result_row.get("source_kind", ""),
        "pmcid": result_row.get("pmcid", ""),
        "doi": result_row.get("doi", ""),
        "title": result_row.get("title", ""),
        "journal": result_row.get("journal", ""),
        "license_key": result_row.get("license_key", ""),
        "model": result_row.get("model", ""),
        "final_diagnosis": result_row.get("final_diagnosis", ""),
        "recommended_next_step": result_row.get("recommended_next_step", ""),
        "score_status": judge_content["score_status"],
        "diagnosis_status": judge_content["diagnosis_status"],
        "next_step_status": judge_content["next_step_status"],
        "rationale_status": judge_content["rationale_status"],
        "expected_key_answer": judge_content.get("expected_key_answer", ""),
        "expected_next_step": judge_content.get("expected_next_step", ""),
        "rationale": judge_content.get("rationale", ""),
        "answer_schema_valid": str(schema_valid).lower(),
        "answer_schema_errors": ",".join(schema_errors),
        "review_status": review_status,
    }


def _parse_jsonish(value: str, *, fallback: Any) -> Any:
    if value == "":
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _looks_like_api_error(text: str) -> bool:
    lower = text.lower()
    return any(
        marker in lower
        for marker in (
            "deepseek api",
            "api http",
            "http ",
            "incomplete chunked response",
            "timed out",
            "timeout",
            "connection reset",
        )
    )


def _component_status(value: Any) -> str:
    status = str(value).lower()
    return status if status in VALID_COMPONENT_STATUSES else "ungradable"


def _result_row(
    row: dict[str, Any],
    model: str,
    parsed_content: dict[str, Any],
    judge_content: dict[str, Any],
) -> dict[str, str]:
    return {
        "selection_rank": _stringify(row.get("selection_rank", "")),
        "case_id": _stringify(row.get("case_id", "")),
        "source_kind": _stringify(row.get("source_kind", "")),
        "pmcid": _stringify(row.get("pmcid", "")),
        "doi": _stringify(row.get("doi", "")),
        "title": _stringify(row.get("title", "")),
        "journal": _stringify(row.get("journal", "")),
        "license_key": _stringify(row.get("license_key", "")),
        "model": model,
        "final_diagnosis": _stringify(parsed_content.get("final_diagnosis", "")),
        "etiology": _stringify(parsed_content.get("etiology", "")),
        "top_differential": _stringify(parsed_content.get("top_differential", "")),
        "recommended_next_step": _stringify(parsed_content.get("recommended_next_step", "")),
        "confidence": _stringify(parsed_content.get("confidence", "")),
        "evidence_summary": _stringify(parsed_content.get("evidence_summary", "")),
        "uncertainty_or_missing_information": _stringify(parsed_content.get("uncertainty_or_missing_information", "")),
        "judge_score_status": _stringify(judge_content.get("score_status", "")),
        "judge_rationale": _stringify(judge_content.get("rationale", "")),
        "review_status": "needs_manual_review" if not judge_content else "judge_scored_needs_spotcheck",
    }


def _empty_answer() -> dict[str, Any]:
    return {
        "final_diagnosis": "",
        "etiology": "",
        "top_differential": [],
        "recommended_next_step": "",
        "confidence": "",
        "evidence_summary": [],
        "uncertainty_or_missing_information": [],
    }


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
