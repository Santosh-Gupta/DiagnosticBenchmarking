"""Post-download filters for neurology text case candidates."""

from __future__ import annotations

from typing import Any


NEUROLOGY_MARKERS: tuple[str, ...] = (
    "neurolog",
    "neuropsychiatr",
    "brain",
    "cerebral",
    "cerebell",
    "spinal cord",
    "myel",
    "stroke",
    "seizure",
    "epilep",
    "encephal",
    "mening",
    "dementia",
    "cognitive",
    "aphasia",
    "ataxia",
    "parkinson",
    "movement disorder",
    "dystonia",
    "chorea",
    "tremor",
    "neuropathy",
    "polyneuropathy",
    "myopathy",
    "myasthen",
    "als",
    "amyotrophic",
    "multiple sclerosis",
    "demyel",
    "moyamoya",
    "migraine",
    "headache",
    "cranial nerve",
    "optic neur",
    "neuro-ophthalm",
    "neurogenetic",
    "leukodystrophy",
    "mitochondrial",
    "neuromuscular",
    "radicul",
    "plexopathy",
    "myelopathy",
    "coma",
    "altered mental status",
    "catatonia",
    "psychosis",
)

IMAGE_HEAVY_MARKERS: tuple[str, ...] = (
    "teaching neuroimage",
    "teaching video neuroimage",
    "neuroimages",
    "neuroimage",
    "video neuroimages",
    "image challenge",
    "case of the week",
    "radiology",
    "neuroradiology",
    "radiopaedia",
    "eurorad",
    "medpix",
)


def keep_article_metadata(
    metadata: dict[str, Any],
    *,
    strict_neurology: bool = True,
    text_only: bool = True,
    case_only: bool = True,
) -> tuple[bool, list[str]]:
    """Return whether parsed article metadata should stay in the corpus."""

    reasons: list[str] = []
    haystack = _metadata_haystack(metadata)
    title_and_journal = _metadata_title_journal(metadata)

    if strict_neurology and not _has_neurology_signal(metadata):
        reasons.append("missing_neurology_marker")

    if text_only and any(marker in title_and_journal for marker in IMAGE_HEAVY_MARKERS):
        reasons.append("image_or_radiology_heavy")

    if case_only and not _looks_like_case(metadata, haystack):
        reasons.append("not_case_like")

    return not reasons, reasons


def _metadata_haystack(metadata: dict[str, Any]) -> str:
    fields: list[str] = []
    for key in ("title", "abstract", "journal", "article_type", "license_text"):
        value = metadata.get(key)
        if isinstance(value, str):
            fields.append(value)
    for key in ("keywords", "subjects", "section_titles"):
        value = metadata.get(key)
        if isinstance(value, list):
            fields.extend(str(item) for item in value)
    return " ".join(fields).lower()


def _primary_neurology_text(metadata: dict[str, Any]) -> str:
    fields: list[str] = []
    for key in ("title", "journal", "article_type"):
        value = metadata.get(key)
        if isinstance(value, str):
            fields.append(value)
    for key in ("keywords", "subjects"):
        value = metadata.get(key)
        if isinstance(value, list):
            fields.extend(str(item) for item in value)
    return " ".join(fields).lower()


def _has_neurology_signal(metadata: dict[str, Any]) -> bool:
    primary = _primary_neurology_text(metadata)
    return any(marker in primary for marker in NEUROLOGY_MARKERS)


def _metadata_title_journal(metadata: dict[str, Any]) -> str:
    values = [metadata.get("title") or "", metadata.get("journal") or ""]
    return " ".join(str(value) for value in values).lower()


def _looks_like_case(metadata: dict[str, Any], haystack: str) -> bool:
    article_type = str(metadata.get("article_type") or "").lower()
    if "case" in article_type:
        return True
    if "case report" in haystack or "case presentation" in haystack or "case series" in haystack:
        return True
    return False
