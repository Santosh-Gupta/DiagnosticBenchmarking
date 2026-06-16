"""Small stdlib HTTP client for NCBI services."""

from __future__ import annotations

import gzip
import json
import ssl
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_OAI_BASE = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"


@dataclass(frozen=True)
class NcbiConfig:
    tool: str = "NeurologyBM"
    email: str | None = None
    api_key: str | None = None
    verify_tls: bool = True
    min_interval_seconds: float = 0.34
    retries: int = 3


class NcbiClient:
    """Rate-limited HTTP wrapper for E-Utilities and PMC OAI-PMH."""

    def __init__(self, config: NcbiConfig) -> None:
        self.config = config
        self._last_request_at = 0.0
        self._ssl_context = None if config.verify_tls else ssl._create_unverified_context()

    def get_json(self, url: str, params: dict[str, str | int]) -> dict[str, Any]:
        data = self.get_bytes(url, params, accept_compression=False)
        return json.loads(data.decode("utf-8"))

    def get_bytes(
        self,
        url: str,
        params: dict[str, str | int],
        *,
        accept_compression: bool = True,
        add_ncbi_params: bool = True,
    ) -> bytes:
        query = dict(params)
        if add_ncbi_params:
            query.setdefault("tool", self.config.tool)
            if self.config.email:
                query.setdefault("email", self.config.email)
            if self.config.api_key:
                query.setdefault("api_key", self.config.api_key)

        full_url = url + "?" + urlencode(query)
        headers = {
            "User-Agent": self._user_agent(),
        }
        if accept_compression:
            headers["Accept-Encoding"] = "gzip, deflate"

        last_error: Exception | None = None
        for attempt in range(self.config.retries):
            self._respect_rate_limit()
            try:
                request = Request(full_url, headers=headers)
                with urlopen(request, timeout=60, context=self._ssl_context) as response:
                    data = response.read()
                    encoding = response.headers.get("Content-Encoding", "")
                return _decompress_if_needed(data, encoding)
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                if attempt + 1 == self.config.retries:
                    break
                time.sleep(2**attempt)

        assert last_error is not None
        raise last_error

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait = self.config.min_interval_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _user_agent(self) -> str:
        if self.config.email:
            return f"{self.config.tool}/0.1 (mailto:{self.config.email})"
        return f"{self.config.tool}/0.1"


def _decompress_if_needed(data: bytes, encoding: str) -> bytes:
    if encoding.lower() == "gzip" or data.startswith(b"\x1f\x8b"):
        return gzip.decompress(data)
    return data


def esearch(
    client: NcbiClient,
    term: str,
    *,
    retmax: int,
    retstart: int = 0,
) -> dict[str, Any]:
    """Run PMC ESearch and return the raw JSON response."""

    return client.get_json(
        f"{EUTILS_BASE}/esearch.fcgi",
        {
            "db": "pmc",
            "retmode": "json",
            "retmax": retmax,
            "retstart": retstart,
            "sort": "pub+date",
            "term": term,
        },
    )


def collect_pmcids(client: NcbiClient, term: str, *, limit: int, page_size: int = 100) -> tuple[list[str], int]:
    """Collect PMC numeric ids for an ESearch term."""

    ids: list[str] = []
    total = 0
    retstart = 0
    while len(ids) < limit:
        batch_size = min(page_size, limit - len(ids))
        response = esearch(client, term, retmax=batch_size, retstart=retstart)
        result = response.get("esearchresult", {})
        if retstart == 0:
            total = int(result.get("count", "0"))
        batch = [str(item) for item in result.get("idlist", [])]
        if not batch:
            break
        ids.extend(batch)
        retstart += len(batch)
    return ids, total


def fetch_oai_full_text_xml(client: NcbiClient, pmc_numeric_id: str) -> bytes:
    """Fetch full-text JATS XML wrapped in an OAI-PMH response."""

    identifier = f"oai:pubmedcentral.nih.gov:{pmc_numeric_id.removeprefix('PMC')}"
    return client.get_bytes(
        PMC_OAI_BASE,
        {
            "verb": "GetRecord",
            "identifier": identifier,
            "metadataPrefix": "pmc",
        },
        add_ncbi_params=False,
    )
