"""Case-study to challenge conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .case_eval import CaseRecord, new_run_id
from .deepseek import assert_private_path


CONVERSION_SCHEMA_KEYS = [
    "challenge_prompt",
    "answer_key",
    "evidence_map",
    "hypothesis_bank",
    "outcome_summary",
]


def build_conversion_prompt(record: CaseRecord, case_text: str, comment_text: str = "") -> tuple[str, str]:
    system_prompt = (
        "You convert medical case material into private benchmark/training artifacts. "
        "Return only valid JSON. Do not include hidden chain-of-thought. "
        "Use concise evidence maps and differential-path summaries instead."
    )
    comment_section = f"\nReader comments or hypothesis bank material:\n{comment_text}\n" if comment_text else ""
    user_prompt = f"""Create a private diagnostic or management challenge from this case material.

Return JSON with exactly these top-level keys:
- challenge_prompt: starting case material only, with no answer leakage
- answer_key: object with diagnosis, etiology, aliases, and next_management_step
- evidence_map: array of concise evidence-to-conclusion summaries
- hypothesis_bank: array of possible hypotheses or next actions to explore
- outcome_summary: concise final clinical course and outcome

Case ID: {record.case_id}
Case type: {record.case_type}

Case material:
{case_text}
{comment_section}
"""
    return system_prompt, user_prompt


def normalize_conversion_content(content: dict[str, Any]) -> dict[str, Any]:
    return {key: content.get(key, "" if key != "hypothesis_bank" and key != "evidence_map" else []) for key in CONVERSION_SCHEMA_KEYS}


def write_conversion_artifact(
    *,
    out_root: Path,
    record: CaseRecord,
    model: str,
    content: dict[str, Any],
    raw_record: dict[str, Any],
    dry_run: bool,
) -> Path:
    assert_private_path(out_root)
    case_dir = out_root / record.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "case_id": record.case_id,
        "source_path": str(record.source_path),
        "case_type": record.case_type,
        "model": model,
        "dry_run": dry_run,
        "artifact_schema": CONVERSION_SCHEMA_KEYS,
        **normalize_conversion_content(content),
    }
    (case_dir / "conversion.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (case_dir / "raw_api_record.json").write_text(json.dumps(raw_record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return case_dir


def conversion_run_dir(out_root: Path, model_tier: str) -> Path:
    return out_root / "runs" / f"{model_tier}_{new_run_id()}"

