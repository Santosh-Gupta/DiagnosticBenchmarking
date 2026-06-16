"""Create public challenge/answer splits from license-compatible case reports."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metadata import extract_article_metadata


CASE_TITLE_RE = re.compile(
    r"\b(case|case presentation|case report|clinical presentation|patient presentation|history|clinical history)\b",
    re.I,
)
ANSWER_TITLE_RE = re.compile(r"\b(discussion|diagnosis|conclusion|treatment|outcome|follow-up|case resolution)\b", re.I)
DIAGNOSIS_LEAK_RE = re.compile(
    r"\b(was diagnosed (?:with|as)|were diagnosed (?:with|as)|diagnosis (?:was|is|of)|"
    r"final diagnosis|confirmed diagnosis|we diagnosed|was treated with|treatment was started|"
    r"ultimately diagnosed|revealed (?:a|an)?\s*[A-Z][A-Za-z -]{3,})\b",
    re.I,
)
IMAGE_REQUIRED_RE = re.compile(
    r"\b(fig(?:ure)?\.?\s*\d+|shown in fig|shown in figure|see fig|see figure|"
    r"image shows|photograph shows|radiograph shows|as shown)\b",
    re.I,
)
VET_RE = re.compile(r"\b(dog|cat|bull|cow|calf|horse|equine|canine|feline|veterinary|limousin bull)\b", re.I)
HUMAN_CLINICAL_RE = re.compile(
    r"\b(patient|man|woman|male|female|boy|girl|child|infant|year-old|presented|admitted|history|examination)\b",
    re.I,
)
MIN_PROMPT_CHARS = 300
MAX_PROMPT_CHARS = 12000
MIN_ANSWER_CHARS = 160


def extract_transformed_challenges_from_xml_dir(
    *,
    xml_dir: Path,
    output_jsonl: Path,
    source_metadata_jsonl: Path | None = None,
) -> dict[str, Any]:
    route_by_pmcid = _load_route_metadata(source_metadata_jsonl) if source_metadata_jsonl else {}
    rows = []
    for xml_path in sorted(xml_dir.rglob("PMC*.xml")):
        xml_bytes = xml_path.read_bytes()
        metadata = extract_article_metadata(xml_bytes)
        pmcid = metadata.get("pmcid") or xml_path.stem
        row = extract_transformed_challenge(xml_bytes)
        row.update(
            {
                "case_id": f"transformed_{pmcid}",
                "source_kind": "transformed_case_report",
                "pmcid": pmcid,
                "doi": metadata.get("doi"),
                "title": metadata.get("title"),
                "journal": metadata.get("journal"),
                "license_key": metadata.get("license_key"),
                "license_tier": _license_tier(metadata.get("license_key")),
                "xml_path": str(xml_path),
                "route_source": route_by_pmcid.get(str(pmcid), {}).get("route_source", {}),
            }
        )
        rows.append(row)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = summarize_transformed_challenges(rows)
    summary.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "xml_dir": str(xml_dir),
            "source_metadata": str(source_metadata_jsonl) if source_metadata_jsonl else None,
            "output": str(output_jsonl),
        }
    )
    output_jsonl.with_name(output_jsonl.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def extract_transformed_challenge(xml_bytes: bytes) -> dict[str, Any]:
    root = ET.fromstring(xml_bytes)
    article = _first_by_local_name(root, "article")
    if article is None:
        return _row("", "", "reject_no_article", ["no_article"])

    title = _article_title(article)
    if VET_RE.search(title):
        return _row("", "", "reject_nonhuman_vet", ["nonhuman_vet"])

    sections = _top_level_sections(article)
    case_sections = [(section_title, text) for section_title, text in sections if CASE_TITLE_RE.search(section_title)]
    if not case_sections:
        case_sections = [(section_title, text) for section_title, text in sections if HUMAN_CLINICAL_RE.search(text)]
    if not case_sections:
        return _row("", "", "reject_no_case_section", ["no_case_section"])

    case_text = "\n\n".join(text for _, text in case_sections)
    if VET_RE.search(case_text):
        return _row("", "", "reject_nonhuman_vet", ["nonhuman_vet"])
    prompt_source = _truncate_before_leak(case_text)
    reasons = []
    transformation_notes = []
    if len(prompt_source) < len(case_text):
        transformation_notes.append("truncated_before_diagnosis_or_treatment_marker")
    challenge_prompt = _build_prompt(prompt_source)

    answer_sections = [(section_title, text) for section_title, text in sections if ANSWER_TITLE_RE.search(section_title)]
    answer_rest = _build_answer(title, answer_sections, sections)

    reasons.extend(_quality_reasons(challenge_prompt, answer_rest))
    status = "definitive_human_text_transformed" if not reasons else "needs_review"
    return _row(challenge_prompt, answer_rest, status, reasons, transformation_notes)


def summarize_transformed_challenges(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "status_counts": dict(Counter(row.get("status") for row in rows)),
        "license_counts": dict(Counter(row.get("license_key") for row in rows)),
        "definitive_human_text_transformed_count": sum(
            1 for row in rows if row.get("status") == "definitive_human_text_transformed"
        ),
    }


def _build_prompt(case_text: str) -> str:
    return (
        "Here is a published clinical case. Based only on the presentation below, "
        "what is the most likely diagnosis or etiology, and what next diagnostic or treatment step is most appropriate?\n\n"
        f"{case_text.strip()}"
    )


def _build_answer(title: str, answer_sections: list[tuple[str, str]], all_sections: list[tuple[str, str]]) -> str:
    parts = []
    if title:
        parts.append(f"Article title / answer clue: {title}")
    if answer_sections:
        for section_title, text in answer_sections[:4]:
            parts.append(f"{section_title}\n{text}")
    else:
        for section_title, text in all_sections[-3:]:
            parts.append(f"{section_title}\n{text}")
    return "\n\n".join(part.strip() for part in parts if part.strip())


def _quality_reasons(prompt: str, answer: str) -> list[str]:
    reasons = []
    if len(prompt) < MIN_PROMPT_CHARS:
        reasons.append("prompt_too_short")
    if len(prompt) > MAX_PROMPT_CHARS:
        reasons.append("prompt_too_long")
    if len(answer) < MIN_ANSWER_CHARS:
        reasons.append("answer_too_short")
    if IMAGE_REQUIRED_RE.search(prompt):
        reasons.append("image_dependent_prompt")
    if VET_RE.search(prompt):
        reasons.append("nonhuman_vet")
    if not HUMAN_CLINICAL_RE.search(prompt):
        reasons.append("not_clearly_human_clinical")
    return reasons


def _truncate_before_leak(text: str) -> str:
    match = DIAGNOSIS_LEAK_RE.search(text)
    if not match:
        return text.strip()
    truncated = text[: match.start()].strip()
    last_period = truncated.rfind(".")
    if last_period > 0 and len(truncated) - last_period < 40:
        truncated = truncated[: last_period + 1]
    return truncated.strip()


def _row(
    challenge_prompt: str,
    answer_rest: str,
    status: str,
    reasons: list[str],
    transformation_notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "review_reasons": reasons,
        "transformation_notes": transformation_notes or [],
        "challenge_prompt": challenge_prompt,
        "answer_rest": answer_rest,
        "challenge_prompt_char_count": len(challenge_prompt),
        "answer_rest_char_count": len(answer_rest),
    }


def _top_level_sections(article: ET.Element) -> list[tuple[str, str]]:
    body = _first_child_by_local_name(article, "body")
    if body is None:
        return []
    sections = []
    for child in list(body):
        if _local_name(child.tag) != "sec":
            continue
        section_title = _section_title(child) or "Untitled section"
        text_parts = []
        for element in child.iter():
            if _local_name(element.tag) == "p":
                text = _clean_text(element)
                if text:
                    text_parts.append(text)
        text = "\n\n".join(text_parts)
        if text:
            sections.append((section_title, text))
    return sections


def _load_route_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    if not path or not path.exists():
        return rows
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                pmcid = row.get("pmcid")
                if pmcid:
                    rows[str(pmcid)] = row
    return rows


def _license_tier(license_key: str | None) -> str:
    if license_key in {"cc0", "cc_by", "cc_by_sa"}:
        return "public_training_compatible_holdout"
    if license_key in {"cc_by_nc", "cc_by_nc_sa"}:
        return "public_benchmark_noncommercial"
    return "public_license_review_required"


def _article_title(article: ET.Element) -> str:
    front = _first_child_by_local_name(article, "front")
    meta = _first_child_by_local_name(front, "article-meta") if front is not None else None
    title_group = _first_child_by_local_name(meta, "title-group") if meta is not None else None
    title = _first_child_by_local_name(title_group, "article-title") if title_group is not None else None
    return _clean_text(title) if title is not None else ""


def _section_title(section: ET.Element) -> str:
    title = _first_child_by_local_name(section, "title")
    return _clean_text(title) if title is not None else ""


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


def _clean_text(element: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()
