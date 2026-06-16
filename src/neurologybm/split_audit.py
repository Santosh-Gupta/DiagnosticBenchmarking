"""Audit public case challenge split manifests before API evaluation."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_FIELDS = [
    "selection_rank",
    "case_id",
    "source_kind",
    "pmcid",
    "doi",
    "title",
    "journal",
    "license_key",
    "decision",
    "issue_count",
    "issues",
    "prompt_char_count",
    "answer_rest_char_count",
]

MULTIPLE_CHOICE_OPTION_RE = re.compile(
    r"(?im)(^|\n)\s*(?:[□☐☑]\s*)?(?:\(?[a-e]\)?[.)]|[A-E][.)])\s+[^\n]{2,}"
)
MULTIPLE_CHOICE_INSTRUCTION_RE = re.compile(
    r"(?i)\b(select one|select all|tick all|choose the|which of the following|multiple choice|answer choices?)\b"
)
QUESTION_ONLY_RE = re.compile(r"(?i)\b(what is your diagnosis|what is the diagnosis|what is the next step|what would you do next)\b")
NON_CASE_ARTICLE_RE = re.compile(
    r"(?i)\b(cross-sectional study|survey|curriculum|medical students|clinical reasoning performance|"
    r"automated feedback|virtual patients|cognitive load|educational intervention|scoping review)\b"
)
HUMAN_CASE_RE = re.compile(
    r"(?i)\b(patient|man|woman|male|female|boy|girl|child|infant|year-old|presented|admitted|history|examination)\b"
)
IMAGE_REFERENCE_RE = re.compile(
    r"(?i)\b(fig(?:ure)?\.?\s*\d+|shown in fig|shown in figure|see fig|see figure|"
    r"image shows|photograph shows|radiograph shows|as shown)\b"
)
ANSWER_LEAK_RE = re.compile(
    r"(?i)\b(final diagnosis|confirmed diagnosis|was diagnosed with|were diagnosed with|"
    r"diagnosis was|diagnosis is|we diagnosed|ultimately diagnosed)\b"
)


def audit_public_splits(
    *,
    manifest_path: Path,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit_rows = [audit_public_split_row(row) for row in rows]
    summary = summarize_audit_rows(audit_rows)
    summary.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path),
        }
    )
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{manifest_path.stem}_audit.csv"
        json_path = out_dir / f"{manifest_path.stem}_audit_summary.json"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=AUDIT_FIELDS)
            writer.writeheader()
            writer.writerows(audit_rows)
        summary["audit_csv"] = str(csv_path)
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["audit_summary"] = str(json_path)
    return summary


def filter_public_splits_by_audit(
    *,
    manifest_path: Path,
    audit_csv_path: Path,
    output_jsonl: Path,
    metadata_csv: Path | None = None,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit_rows = {
        row["case_id"]: row
        for row in csv.DictReader(audit_csv_path.open(newline="", encoding="utf-8"))
    }
    clean_rows = []
    excluded_rows = []
    for row in rows:
        audit_row = audit_rows.get(str(row.get("case_id")), {})
        if audit_row.get("decision") == "include_ready":
            clean_row = dict(row)
            clean_row["clean_selection_rank"] = len(clean_rows) + 1
            clean_row["parent_manifest"] = str(manifest_path)
            clean_row["audit_decision"] = "include_ready"
            clean_rows.append(clean_row)
        else:
            excluded_rows.append(row)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in clean_rows),
        encoding="utf-8",
    )
    if metadata_csv:
        metadata_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "clean_selection_rank",
            "selection_rank",
            "case_id",
            "source_kind",
            "status",
            "pmcid",
            "doi",
            "title",
            "journal",
            "license_key",
            "license_tier",
            "challenge_prompt_char_count",
            "answer_rest_char_count",
            "source_xml_path",
        ]
        with metadata_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in clean_rows:
                writer.writerow({field: row.get(field, "") for field in fields})

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_from": str(manifest_path),
        "audit_csv": str(audit_csv_path),
        "output": str(output_jsonl),
        "metadata_csv": str(metadata_csv) if metadata_csv else None,
        "row_count": len(clean_rows),
        "excluded_count": len(excluded_rows),
        "excluded_case_ids": [str(row.get("case_id", "")) for row in excluded_rows],
    }
    summary_path = output_jsonl.with_name(output_jsonl.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def audit_public_split_row(row: dict[str, Any]) -> dict[str, str]:
    prompt = str(row.get("challenge_prompt") or "")
    answer = str(row.get("answer_rest") or "")
    issues = []
    if MULTIPLE_CHOICE_OPTION_RE.search(prompt):
        issues.append("multiple_choice_options_in_prompt")
    if MULTIPLE_CHOICE_INSTRUCTION_RE.search(prompt):
        issues.append("multiple_choice_instruction_in_prompt")
    if IMAGE_REFERENCE_RE.search(prompt):
        issues.append("image_or_figure_reference_in_prompt")
    if ANSWER_LEAK_RE.search(prompt):
        issues.append("possible_answer_leak_in_prompt")
    if NON_CASE_ARTICLE_RE.search(prompt) or NON_CASE_ARTICLE_RE.search(str(row.get("title") or "")):
        issues.append("likely_non_case_article")
    if not HUMAN_CASE_RE.search(prompt):
        issues.append("not_clearly_human_case_prompt")
    if not answer.strip():
        issues.append("missing_answer_rest")
    if len(prompt) < 250:
        issues.append("prompt_too_short_for_benchmark")
    if len(answer) < 120:
        issues.append("answer_rest_too_short_for_benchmark")

    decision = "include_ready"
    hard_rejects = {
        "multiple_choice_options_in_prompt",
        "multiple_choice_instruction_in_prompt",
        "likely_non_case_article",
        "missing_answer_rest",
        "not_clearly_human_case_prompt",
        "possible_answer_leak_in_prompt",
    }
    if any(issue in hard_rejects for issue in issues):
        decision = "exclude_or_repair_before_api"
    elif issues:
        decision = "manual_review_before_api"

    return {
        "selection_rank": _stringify(row.get("selection_rank", "")),
        "case_id": _stringify(row.get("case_id", "")),
        "source_kind": _stringify(row.get("source_kind", "")),
        "pmcid": _stringify(row.get("pmcid", "")),
        "doi": _stringify(row.get("doi", "")),
        "title": _stringify(row.get("title", "")),
        "journal": _stringify(row.get("journal", "")),
        "license_key": _stringify(row.get("license_key", "")),
        "decision": decision,
        "issue_count": str(len(issues)),
        "issues": ",".join(issues),
        "prompt_char_count": str(len(prompt)),
        "answer_rest_char_count": str(len(answer)),
    }


def summarize_audit_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    decisions = Counter(row["decision"] for row in rows)
    issue_counts: Counter[str] = Counter()
    source_decisions: dict[str, Counter[str]] = {}
    for row in rows:
        source_decisions.setdefault(row["source_kind"], Counter())[row["decision"]] += 1
        for issue in row["issues"].split(","):
            if issue:
                issue_counts[issue] += 1
    return {
        "total": len(rows),
        "decision_counts": dict(sorted(decisions.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "source_decision_counts": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(source_decisions.items())
        },
    }


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
