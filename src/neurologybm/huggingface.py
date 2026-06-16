"""Minimal Hugging Face dataset collection helpers."""

from __future__ import annotations

import json
import ssl
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import HTTPSHandler, Request, build_opener

from .hf_sources import HfSource


HF_BASE_URL = "https://huggingface.co"
USER_AGENT = "NeurologyBM/0.1 (+https://github.com/Santosh-Gupta)"


class HuggingFaceClient:
    """Tiny stdlib Hugging Face client for metadata and direct file downloads."""

    def __init__(self, *, verify_tls: bool = True, timeout_seconds: float = 60.0) -> None:
        self.timeout_seconds = timeout_seconds
        context = None if verify_tls else ssl._create_unverified_context()  # noqa: S323 - explicit CLI escape hatch.
        self.opener = build_opener(HTTPSHandler(context=context)) if context else build_opener()

    def dataset_info(self, repo_id: str) -> dict[str, Any]:
        path = f"/api/datasets/{quote(repo_id, safe='/')}"
        return self._json(path)

    def search_datasets(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        path = "/api/datasets?" + urlencode({"search": query, "limit": str(limit), "full": "true"})
        payload = self._json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected Hugging Face search payload for {query!r}")
        return payload

    def head_file(self, repo_id: str, file_path: str) -> dict[str, Any]:
        request = Request(self._resolve_url(repo_id, file_path), method="HEAD", headers=_headers())
        try:
            with self.opener.open(request, timeout=self.timeout_seconds) as response:
                return {
                    "path": file_path,
                    "status": getattr(response, "status", 200),
                    "content_length": _int_header(response.headers.get("Content-Length")),
                    "content_type": response.headers.get("Content-Type"),
                    "etag": response.headers.get("ETag"),
                }
        except HTTPError as exc:
            return {"path": file_path, "status": exc.code, "error": str(exc)}
        except URLError as exc:
            return {"path": file_path, "status": None, "error": str(exc)}

    def download_file(self, repo_id: str, file_path: str, dest: Path, *, force: bool = False) -> dict[str, Any]:
        dest = dest / _safe_relative_path(file_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not force:
            return {
                "path": file_path,
                "downloaded": False,
                "skipped_reason": "exists",
                "local_path": str(dest),
                "bytes": dest.stat().st_size,
            }

        request = Request(self._resolve_url(repo_id, file_path), headers=_headers())
        with self.opener.open(request, timeout=self.timeout_seconds) as response:
            byte_count = 0
            with dest.open("wb") as out_file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    byte_count += len(chunk)
        return {"path": file_path, "downloaded": True, "local_path": str(dest), "bytes": byte_count}

    def download_readme(self, repo_id: str, dest: Path, *, force: bool = False) -> dict[str, Any]:
        try:
            return self.download_file(repo_id, "README.md", dest, force=force)
        except HTTPError as exc:
            if exc.code == 404:
                return {"path": "README.md", "downloaded": False, "skipped_reason": "missing"}
            raise

    def _json(self, path: str) -> dict[str, Any] | list[dict[str, Any]]:
        request = Request(HF_BASE_URL + path, headers=_headers())
        with self.opener.open(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _resolve_url(self, repo_id: str, file_path: str) -> str:
        encoded_repo = quote(repo_id, safe="/")
        encoded_file = "/".join(quote(part) for part in _safe_relative_path(file_path).parts)
        return f"{HF_BASE_URL}/datasets/{encoded_repo}/resolve/main/{encoded_file}"


def collect_hf_source(
    client: HuggingFaceClient,
    source: HfSource,
    out: Path,
    *,
    download_files: bool = False,
    max_file_bytes: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Collect metadata and optionally configured files for one HF source."""

    repo_dir = out / "raw" / repo_dir_name(source.repo_id)
    metadata_dir = out / "metadata" / "repos"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    repo_dir.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {
        **asdict(source),
        "dataset_url": f"{HF_BASE_URL}/datasets/{source.repo_id}",
        "repo_dir": str(repo_dir),
        "errors": [],
        "files": [],
    }

    try:
        info = client.dataset_info(source.repo_id)
    except Exception as exc:  # noqa: BLE001 - keep batch collection moving.
        row["errors"].append({"stage": "dataset_info", "error": str(exc)})
        _write_json(metadata_dir / f"{source.key}.json", row)
        return row

    siblings = _siblings(info)
    observed_license = license_from_info(info)
    info_path = metadata_dir / f"{source.key}.json"
    _write_json(info_path, info)

    row.update(
        {
            "metadata_path": str(info_path),
            "observed_license": observed_license,
            "license_matches_expected": _licenses_match(observed_license, source.expected_license),
            "gated": bool(info.get("gated")),
            "private": bool(info.get("private")),
            "downloads": info.get("downloads"),
            "likes": info.get("likes"),
            "last_modified": info.get("lastModified"),
            "siblings_count": len(siblings),
            "siblings_sample": siblings[:50],
        }
    )

    try:
        row["readme"] = client.download_readme(source.repo_id, repo_dir, force=force)
    except Exception as exc:  # noqa: BLE001 - README is audit support, not fatal.
        row["errors"].append({"stage": "readme", "error": str(exc)})

    for file_path in source.default_files:
        file_row = client.head_file(source.repo_id, file_path)
        size = file_row.get("content_length")
        if isinstance(size, int) and max_file_bytes is not None and size > max_file_bytes:
            file_row.update({"downloaded": False, "skipped_reason": "too_large"})
            row["files"].append(file_row)
            continue
        if file_row.get("status") not in (200, 302, 303, 307, 308):
            file_row.update({"downloaded": False, "skipped_reason": "head_failed"})
            row["files"].append(file_row)
            continue
        if not download_files:
            file_row.update({"downloaded": False, "skipped_reason": "metadata_only"})
            row["files"].append(file_row)
            continue
        try:
            file_row.update(client.download_file(source.repo_id, file_path, repo_dir, force=force))
        except Exception as exc:  # noqa: BLE001 - preserve manifest progress.
            file_row.update({"downloaded": False, "error": str(exc)})
            row["errors"].append({"stage": "download", "path": file_path, "error": str(exc)})
        row["files"].append(file_row)

    return row


def discover_hf_datasets(
    client: HuggingFaceClient,
    queries: list[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    results = []
    for query in queries:
        try:
            matches = client.search_datasets(query, limit=limit)
        except Exception as exc:  # noqa: BLE001 - discovery should continue.
            results.append({"query": query, "error": str(exc), "matches": []})
            continue
        results.append({"query": query, "matches": [_compact_dataset(item) for item in matches]})
    return results


def repo_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def license_from_info(info: dict[str, Any]) -> str | None:
    card_data = info.get("cardData")
    if isinstance(card_data, dict):
        license_value = card_data.get("license")
        if isinstance(license_value, str):
            return license_value.lower()
        if isinstance(license_value, list) and license_value:
            return ",".join(str(item).lower() for item in license_value)

    for tag in info.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1].lower()
    return None


def _compact_dataset(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id") or item.get("_id"),
        "author": item.get("author"),
        "downloads": item.get("downloads"),
        "likes": item.get("likes"),
        "gated": item.get("gated"),
        "private": item.get("private"),
        "last_modified": item.get("lastModified"),
        "license": license_from_info(item),
        "tags": [tag for tag in item.get("tags") or [] if isinstance(tag, str)][:30],
    }


def _siblings(info: dict[str, Any]) -> list[str]:
    siblings = []
    for sibling in info.get("siblings") or []:
        if isinstance(sibling, dict) and isinstance(sibling.get("rfilename"), str):
            siblings.append(sibling["rfilename"])
    return sorted(siblings)


def _licenses_match(observed: str | None, expected: str | None) -> bool | None:
    if expected is None:
        return None
    if observed is None:
        return False
    return observed.lower() == expected.lower()


def _safe_relative_path(file_path: str) -> Path:
    path = Path(file_path)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Unsafe Hugging Face file path: {file_path!r}")
    return path


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT}


def _int_header(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
