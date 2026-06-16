"""Challenge prompt extraction from JATS/OAI XML."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metadata import extract_article_metadata


QUESTION_RE = re.compile(
    r"(?P<question>(what\s+is\s+your\s+diagnosis|what\s+is\s+the\s+diagnosis|"
    r"what\s+is\s+your\s+differential\s+diagnosis|spot\s+diagnosis|photo\s+quiz|"
    r"what\s+would\s+you\s+do\s+next|what\s+is\s+the\s+next\s+step|"
    r"which\s+of\s+the\s+following|what\s+is\s+shown|what\s+does\s+the\s+image\s+show)[^?\.]*[?\.]?)",
    re.IGNORECASE,
)
ANSWER_TITLE_RE = re.compile(
    r"^(answer|diagnosis|discussion|comment|explanation|final diagnosis|"
    r"case discussion|what was your diagnosis|photo quiz answer)\b",
    re.IGNORECASE,
)
ANSWER_PARAGRAPH_HEADING_RE = re.compile(
    r"^(answers?|diagnosis|discussion|explanation|case fate|case resolution)\s*:?\s*$",
    re.IGNORECASE,
)
IMAGE_DEPENDENT_RE = re.compile(r"\b(fig\.?|figure|image|photograph|radiograph|ct|mri|histolog|microscop|smear)\b", re.I)


def extract_prompt_candidates_from_xml_dir(
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
        route_metadata = route_by_pmcid.get(str(pmcid), {})
        row = extract_prompt_candidate(xml_bytes)
        row.update(
            {
                "pmcid": pmcid,
                "doi": metadata.get("doi"),
                "title": metadata.get("title"),
                "journal": metadata.get("journal"),
                "license_key": metadata.get("license_key"),
                "xml_path": str(xml_path),
                "route_source": route_metadata.get("route_source", {}),
            }
        )
        rows.append(row)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = summarize_prompt_candidates(rows)
    summary.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "xml_dir": str(xml_dir),
            "source_metadata": str(source_metadata_jsonl) if source_metadata_jsonl else None,
            "output": str(output_jsonl),
            "note": "Prototype extraction. Review prompt/answer boundaries and image dependence before model benchmarking.",
        }
    )
    summary_path = output_jsonl.with_name(output_jsonl.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def extract_prompt_candidate(xml_bytes: bytes) -> dict[str, Any]:
    root = ET.fromstring(xml_bytes)
    article = _first_by_local_name(root, "article")
    if article is None:
        return _empty_row("no_article")

    blocks = _article_blocks(article)
    if not blocks:
        return _empty_row("no_text_blocks")

    prompt_blocks: list[str] = []
    answer_blocks: list[str] = []
    method = "no_clear_boundary"
    confidence = "low"
    in_answer = False
    saw_question = False

    for kind, text in blocks:
        if not text:
            continue
        if kind == "title" and ANSWER_TITLE_RE.search(text):
            in_answer = True
            method = "section_boundary"
            confidence = "high" if saw_question else "medium"
            answer_blocks.append(text)
            continue
        if kind == "p" and ANSWER_PARAGRAPH_HEADING_RE.search(text):
            in_answer = True
            method = "section_boundary"
            confidence = "high" if saw_question else "medium"
            answer_blocks.append(text)
            continue
        if not in_answer:
            prompt_blocks.append(text)
            if QUESTION_RE.search(text):
                saw_question = True
        else:
            answer_blocks.append(text)

    if not answer_blocks:
        joined = "\n\n".join(prompt_blocks)
        match = QUESTION_RE.search(joined)
        if match:
            split_at = match.end()
            prompt_text = joined[:split_at].strip()
            answer_text = joined[split_at:].strip()
            method = "single_paragraph_question_split" if answer_text else "question_only_no_answer_boundary"
            confidence = "medium" if answer_text else "low"
        else:
            prompt_text = joined.strip()
            answer_text = ""
    else:
        prompt_text = "\n\n".join(prompt_blocks).strip()
        answer_text = "\n\n".join(answer_blocks).strip()

    return {
        "prompt_candidate": prompt_text,
        "answer_rest_candidate": answer_text,
        "prompt_char_count": len(prompt_text),
        "answer_rest_char_count": len(answer_text),
        "extraction_method": method,
        "extraction_confidence": confidence,
        "likely_image_dependent": "yes" if IMAGE_DEPENDENT_RE.search(prompt_text) else "no",
        "figure_count": _count_by_local_name(article, "fig"),
        "ready_without_review": bool(prompt_text and answer_text and confidence == "high" and not IMAGE_DEPENDENT_RE.search(prompt_text)),
    }


def summarize_prompt_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "confidence_counts": dict(Counter(str(row.get("extraction_confidence")) for row in rows)),
        "method_counts": dict(Counter(str(row.get("extraction_method")) for row in rows)),
        "image_dependent_count": sum(1 for row in rows if row.get("likely_image_dependent") == "yes"),
        "answer_present_count": sum(1 for row in rows if row.get("answer_rest_candidate")),
        "prompt_present_count": sum(1 for row in rows if row.get("prompt_candidate")),
        "ready_without_review_count": sum(1 for row in rows if row.get("ready_without_review")),
        "needs_manual_review_count": sum(1 for row in rows if not row.get("ready_without_review")),
    }


def _article_blocks(article: ET.Element) -> list[tuple[str, str]]:
    body = _first_child_by_local_name(article, "body")
    if body is None:
        return []
    blocks: list[tuple[str, str]] = []
    for element in body.iter():
        local = _local_name(element.tag)
        if local == "title":
            text = _clean_text(element)
            if text:
                blocks.append(("title", text))
        elif local == "p":
            text = _clean_text(element)
            if text:
                blocks.append(("p", text))
    return blocks


def _load_route_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    if not path or not path.exists():
        return rows
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            pmcid = row.get("pmcid")
            if pmcid:
                rows[str(pmcid)] = row
    return rows


def _empty_row(reason: str) -> dict[str, Any]:
    return {
        "prompt_candidate": "",
        "answer_rest_candidate": "",
        "prompt_char_count": 0,
        "answer_rest_char_count": 0,
        "extraction_method": reason,
        "extraction_confidence": "low",
        "likely_image_dependent": "unknown",
        "figure_count": 0,
        "ready_without_review": False,
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


def _first_child_by_local_name(element: ET.Element, name: str) -> ET.Element | None:
    for child in list(element):
        if _local_name(child.tag) == name:
            return child
    return None


def _count_by_local_name(element: ET.Element, name: str) -> int:
    return sum(1 for child in element.iter() if _local_name(child.tag) == name)


def _clean_text(element: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()
