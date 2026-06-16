"""Private case registry, prompt, and result helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .deepseek import DEFAULT_PRIVATE_ROOT, assert_private_path


REGISTRY_FIELDS = [
    "case_id",
    "source_path",
    "case_type",
    "prompt_template",
    "answer_key",
    "prior_status",
    "next_queue",
    "notes",
]

RESULT_FIELDS = [
    "case_id",
    "source_path",
    "case_type",
    "model",
    "prompt_template",
    "final_diagnosis",
    "etiology",
    "top_differential",
    "recommended_next_step",
    "confidence",
    "score_status",
    "next_queue",
]

VALID_SCORE_STATUSES = {"pass", "partial", "fail", "ungradable"}
VALID_NEXT_QUEUES = {"advanced_api_testing", "conversion_needed", "gold_private_benchmark", "retire"}


DEFAULT_CASES = [
    (
        "carey2017",
        "docs/DO NOT COMMIT TO GITHUB/challenge_prompts/carey2017_prompt_only/challenge.txt",
        "management_challenge",
        "management_next_step",
        "fail",
        "gold_private_benchmark",
        "Challenge-style what-to-do-next item; DeepSeek Instant + search failed.",
    ),
    (
        "case_12_2023_weakness_myalgia",
        "docs/DO NOT COMMIT TO GITHUB/text_reading_order/Case-12-2023-A-44-Year-Old-Woman-with-Muscle-Weakness-and-Myalgia.txt",
        "non_challenge_but_deepseek_failed",
        "closed_book_diagnosis",
        "fail",
        "conversion_needed",
        "Failed closed-book, search-assisted, and augmented retrieval prompt; needs clean challenge construction.",
    ),
    (
        "case_7_2024_alternating_sixth_nerve_palsy",
        "docs/DO NOT COMMIT TO GITHUB/text_reading_order/Case-7-2024-A-67-Year-Old-Woman-with-Alternating-Sixth-Cranial-Nerve-Palsy.txt",
        "non_challenge_but_deepseek_failed",
        "closed_book_diagnosis",
        "fail",
        "conversion_needed",
        "Failed closed-book, search-assisted, and augmented retrieval prompt; needs clean challenge construction.",
    ),
    (
        "case_3_2021_transient_vision_loss",
        "docs/DO NOT COMMIT TO GITHUB/text_reading_order/Case-3-2021-A-48-Year-Old-Man-with-Transient-Vision-Loss.txt",
        "non_challenge_but_deepseek_failed",
        "closed_book_diagnosis",
        "fail",
        "conversion_needed",
        "Failed closed-book, search-assisted, and augmented retrieval prompt; needs clean challenge construction.",
    ),
    (
        "hogan2019",
        "docs/DO NOT COMMIT TO GITHUB/text_reading_order/hogan2019.txt",
        "passed_light_enqueue_advanced",
        "closed_book_diagnosis",
        "search_rescued",
        "advanced_api_testing",
        "Initially failed, then passed with search; useful retrieval-pipeline item.",
    ),
    (
        "acute_myelopathy_cutaneous_papules",
        "docs/DO NOT COMMIT TO GITHUB/text_reading_order/Acute-Myelopathy-in-a-Man-With-Cutaneous-Papules.txt",
        "passed_light_enqueue_advanced",
        "closed_book_diagnosis",
        "search_rescued",
        "advanced_api_testing",
        "Initially failed, then passed with search; useful retrieval-pipeline item.",
    ),
]


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    source_path: Path
    case_type: str
    prompt_template: str
    answer_key: str = ""
    prior_status: str = ""
    next_queue: str = "conversion_needed"
    notes: str = ""


def default_private_root() -> Path:
    return DEFAULT_PRIVATE_ROOT


def ensure_private_workspace(private_root: Path | None = None) -> dict[str, Path]:
    root = private_root or default_private_root()
    assert_private_path(root)
    paths = {
        "root": root,
        "eval": root / "deepseek_eval",
        "runs": root / "deepseek_eval" / "runs",
        "registry": root / "deepseek_eval" / "case_registry.tsv",
        "conversion": root / "case_conversion",
        "conversion_runs": root / "case_conversion" / "runs",
    }
    for key, path in paths.items():
        if key == "registry":
            continue
        path.mkdir(parents=True, exist_ok=True)
    return paths


def create_default_registry(path: Path, *, force: bool = False) -> Path:
    assert_private_path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REGISTRY_FIELDS, delimiter="\t")
        writer.writeheader()
        for case_id, source_path, case_type, prompt_template, prior_status, next_queue, notes in DEFAULT_CASES:
            writer.writerow(
                {
                    "case_id": case_id,
                    "source_path": source_path,
                    "case_type": case_type,
                    "prompt_template": prompt_template,
                    "answer_key": "",
                    "prior_status": prior_status,
                    "next_queue": next_queue,
                    "notes": notes,
                }
            )
    return path


def load_case_registry(path: Path) -> list[CaseRecord]:
    if not path.exists():
        raise ValueError(f"Case registry not found: {path}")
    records: list[CaseRecord] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        missing = set(REGISTRY_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Registry missing fields: {sorted(missing)}")
        for row in reader:
            records.append(
                CaseRecord(
                    case_id=row["case_id"],
                    source_path=Path(row["source_path"]),
                    case_type=row["case_type"],
                    prompt_template=row["prompt_template"],
                    answer_key=row.get("answer_key", ""),
                    prior_status=row.get("prior_status", ""),
                    next_queue=row.get("next_queue", "conversion_needed"),
                    notes=row.get("notes", ""),
                )
            )
    return records


def build_case_prompt(record: CaseRecord, case_text: str) -> tuple[str, str]:
    system_prompt = (
        "You are evaluating a published medical case for diagnosis or management. "
        "Return only valid JSON. Do not look up the publication title or DOI. "
        "Do not provide hidden chain-of-thought; provide concise clinical evidence summaries."
    )
    if record.prompt_template == "management_next_step":
        task = (
            "Identify the best next diagnostic or management step, the most likely diagnosis or etiology "
            "if knowable, and the key differentials."
        )
    else:
        task = "Identify the most likely diagnosis or etiology and the key differentials."
    user_prompt = f"""{task}

Return JSON with these keys:
- final_diagnosis: string
- etiology: string
- top_differential: array of strings
- recommended_next_step: string
- confidence: number from 0 to 1
- evidence_summary: array of concise strings
- uncertainty_or_missing_information: array of concise strings

Case material:
{case_text}
"""
    return system_prompt, user_prompt


def normalize_result_row(
    *,
    record: CaseRecord,
    model: str,
    parsed_content: dict[str, Any],
    score_status: str = "ungradable",
    next_queue: str | None = None,
) -> dict[str, str]:
    if score_status not in VALID_SCORE_STATUSES:
        raise ValueError(f"Invalid score status: {score_status}")
    queue = next_queue or queue_from_score(score_status, record)
    if queue not in VALID_NEXT_QUEUES:
        raise ValueError(f"Invalid next queue: {queue}")
    return {
        "case_id": record.case_id,
        "source_path": str(record.source_path),
        "case_type": record.case_type,
        "model": model,
        "prompt_template": record.prompt_template,
        "final_diagnosis": _stringify(parsed_content.get("final_diagnosis", "")),
        "etiology": _stringify(parsed_content.get("etiology", "")),
        "top_differential": _stringify(parsed_content.get("top_differential", "")),
        "recommended_next_step": _stringify(parsed_content.get("recommended_next_step", "")),
        "confidence": _stringify(parsed_content.get("confidence", "")),
        "score_status": score_status,
        "next_queue": queue,
    }


def queue_from_score(score_status: str, record: CaseRecord) -> str:
    if score_status in {"pass", "partial"}:
        return "advanced_api_testing"
    if record.case_type in {"management_challenge", "ready_challenge"}:
        return "gold_private_benchmark"
    if record.case_type in {"case_study_needs_transformation", "non_challenge_but_deepseek_failed"}:
        return "conversion_needed"
    return "retire"


def write_run_outputs(
    *,
    run_dir: Path,
    result_rows: list[dict[str, str]],
    raw_records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    assert_private_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "results.tsv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(result_rows)
    with (run_dir / "raw_api_records.jsonl").open("w", encoding="utf-8") as file:
        for record in raw_records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
