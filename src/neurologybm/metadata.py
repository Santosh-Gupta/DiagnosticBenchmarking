"""Metadata extraction from PMC OAI/JATS XML."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

from .licenses import is_allowed_by_profile, license_key_from_href, normalize_license_href


CC_URL_RE = re.compile(r"https?://creativecommons\.org/(?:licenses|publicdomain)/[^\s<)]+", re.I)
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


def extract_article_metadata(xml_bytes: bytes, *, license_profile: str | None = None) -> dict[str, Any]:
    """Extract compact article metadata from an OAI-PMH JATS XML response."""

    root = ET.fromstring(xml_bytes)
    article = _first_by_local_name(root, "article")
    if article is None:
        return {
            "oai_error": _text_at(root, "error"),
        }

    license_el = _first_by_local_name(article, "license")
    license_href = _license_href(license_el)
    normalized_href = normalize_license_href(license_href)
    metadata: dict[str, Any] = {
        "pmcid": _article_id(article, ("pmcid", "pmc", "pmcaid")),
        "pmid": _article_id(article, "pmid"),
        "doi": _article_id(article, "doi"),
        "title": _text_path(article, ["front", "article-meta", "title-group", "article-title"]),
        "journal": _text_path(article, ["front", "journal-meta", "journal-title-group", "journal-title"]),
        "publisher": _text_path(article, ["front", "journal-meta", "publisher", "publisher-name"]),
        "publication_date": _publication_date(article),
        "article_type": article.attrib.get("article-type"),
        "subjects": _subjects(article),
        "keywords": _keywords(article),
        "abstract": _abstract(article),
        "license_href": normalized_href,
        "license_key": license_key_from_href(normalized_href),
        "license_text": _clean_whitespace(_text_content(license_el)) if license_el is not None else None,
        "section_titles": _section_titles(article),
    }
    if metadata["pmcid"] and not str(metadata["pmcid"]).startswith("PMC"):
        metadata["pmcid"] = f"PMC{metadata['pmcid']}"
    if license_profile:
        metadata["allowed_by_profile"] = is_allowed_by_profile(normalized_href, license_profile)
    return metadata


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children_by_local_name(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _local_name(child.tag) == name]


def _first_child_by_local_name(element: ET.Element, name: str) -> ET.Element | None:
    for child in list(element):
        if _local_name(child.tag) == name:
            return child
    return None


def _first_by_local_name(element: ET.Element, name: str) -> ET.Element | None:
    if _local_name(element.tag) == name:
        return element
    for child in element.iter():
        if _local_name(child.tag) == name:
            return child
    return None


def _text_path(element: ET.Element, path: list[str]) -> str | None:
    current = element
    for part in path:
        next_element = _first_child_by_local_name(current, part)
        if next_element is None:
            return None
        current = next_element
    return _clean_whitespace(_text_content(current))


def _text_at(element: ET.Element, name: str) -> str | None:
    child = _first_by_local_name(element, name)
    if child is None:
        return None
    return _clean_whitespace(_text_content(child))


def _text_content(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext())


def _clean_whitespace(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned or None


def _article_id(article: ET.Element, pub_id_type: str | tuple[str, ...]) -> str | None:
    allowed_types = (pub_id_type,) if isinstance(pub_id_type, str) else pub_id_type
    front = _first_child_by_local_name(article, "front")
    if front is None:
        return None
    meta = _first_child_by_local_name(front, "article-meta")
    if meta is None:
        return None
    for article_id in _children_by_local_name(meta, "article-id"):
        if article_id.attrib.get("pub-id-type") in allowed_types:
            return _clean_whitespace(_text_content(article_id))
    return None


def _publication_date(article: ET.Element) -> str | None:
    front = _first_child_by_local_name(article, "front")
    meta = _first_child_by_local_name(front, "article-meta") if front is not None else None
    if meta is None:
        return None
    pub_dates = _children_by_local_name(meta, "pub-date")
    if not pub_dates:
        return None
    preferred = next((date for date in pub_dates if date.attrib.get("pub-type") == "epub"), pub_dates[0])
    year = _text_path(preferred, ["year"])
    month = _text_path(preferred, ["month"])
    day = _text_path(preferred, ["day"])
    if not year:
        return None
    parts = [year]
    if month:
        parts.append(month.zfill(2))
    if day:
        parts.append(day.zfill(2))
    return "-".join(parts)


def _subjects(article: ET.Element) -> list[str]:
    subjects: list[str] = []
    categories = _first_by_local_name(article, "article-categories")
    if categories is None:
        return subjects
    for subject in categories.iter():
        if _local_name(subject.tag) == "subject":
            text = _clean_whitespace(_text_content(subject))
            if text:
                subjects.append(text)
    return sorted(set(subjects))


def _keywords(article: ET.Element) -> list[str]:
    values: list[str] = []
    for element in article.iter():
        if _local_name(element.tag) == "kwd":
            text = _clean_whitespace(_text_content(element))
            if text:
                values.append(text)
    return values


def _abstract(article: ET.Element) -> str | None:
    front = _first_child_by_local_name(article, "front")
    meta = _first_child_by_local_name(front, "article-meta") if front is not None else None
    abstract = _first_child_by_local_name(meta, "abstract") if meta is not None else None
    return _clean_whitespace(_text_content(abstract)) if abstract is not None else None


def _section_titles(article: ET.Element) -> list[str]:
    titles: list[str] = []
    for element in article.iter():
        if _local_name(element.tag) == "sec":
            title = _first_child_by_local_name(element, "title")
            text = _clean_whitespace(_text_content(title))
            if text:
                titles.append(text)
    return titles


def _license_href(license_el: ET.Element | None) -> str | None:
    if license_el is None:
        return None
    direct_href = _attribute_by_local_name(license_el, "href")
    if direct_href:
        return direct_href

    for element in license_el.iter():
        if _local_name(element.tag) == "license_ref":
            text = _clean_whitespace(_text_content(element))
            if text:
                return text
        nested_href = _attribute_by_local_name(element, "href")
        if nested_href and "creativecommons.org" in nested_href.lower():
            return nested_href

    text = _text_content(license_el)
    match = CC_URL_RE.search(text)
    if match:
        return match.group(0)
    return None


def _attribute_by_local_name(element: ET.Element, name: str) -> str | None:
    if name == "href" and XLINK_HREF in element.attrib:
        return element.attrib[XLINK_HREF]
    for key, value in element.attrib.items():
        if _local_name(key) == name:
            return value
    return None
