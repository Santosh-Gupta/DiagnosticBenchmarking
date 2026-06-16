"""Search query construction for PMC neurology and psychiatry case reports."""

from __future__ import annotations

from dataclasses import dataclass

from .licenses import license_filter_query


@dataclass(frozen=True)
class Topic:
    key: str
    terms: tuple[str, ...]


TOPICS: dict[str, Topic] = {
    "neurology": Topic(
        "neurology",
        (
            "neurology[MeSH Terms]",
            "neurolog*[Title/Abstract]",
            "brain[Title/Abstract]",
            '"spinal cord"[Title/Abstract]',
            "epilepsy[Title/Abstract]",
            "seizure[Title/Abstract]",
            "stroke[Title/Abstract]",
            "encephalitis[Title/Abstract]",
            "meningitis[Title/Abstract]",
            "dementia[Title/Abstract]",
            "ataxia[Title/Abstract]",
            "neuropathy[Title/Abstract]",
            "myopathy[Title/Abstract]",
            '"movement disorder"[Title/Abstract]',
            '"multiple sclerosis"[Title/Abstract]',
            "Parkinson[Title/Abstract]",
        ),
    ),
    "psychiatry": Topic(
        "psychiatry",
        (
            "psychiatry[MeSH Terms]",
            "psychiatr*[Title/Abstract]",
            "psychosis[Title/Abstract]",
            "depression[Title/Abstract]",
            "mania[Title/Abstract]",
            "bipolar[Title/Abstract]",
            "catatonia[Title/Abstract]",
            "schizophrenia[Title/Abstract]",
            "neuropsychiatr*[Title/Abstract]",
        ),
    ),
    "neuropsychiatry": Topic(
        "neuropsychiatry",
        (
            "neuropsychiatr*[Title/Abstract]",
            '"behavioral neurology"[Title/Abstract]',
            '"autoimmune encephalitis"[Title/Abstract]',
            '"limbic encephalitis"[Title/Abstract]',
            "catatonia[Title/Abstract]",
            "psychosis[Title/Abstract]",
        ),
    ),
}


CASE_REPORT_TERMS: tuple[str, ...] = (
    '"case reports"[All Fields]',
    '"case report"[Title]',
    '"case presentation"[Title]',
    '"case series"[Title]',
)

TEXT_ONLY_EXCLUSION_TERMS: tuple[str, ...] = (
    '"Teaching NeuroImages"[Title]',
    '"Teaching Video NeuroImages"[Title]',
    '"NeuroImages"[Title]',
    '"NeuroImage"[Title]',
    '"Video NeuroImages"[Title]',
    '"Image Challenge"[Title]',
    '"Images in"[Title]',
    '"case of the week"[Title]',
    '"radiology"[Journal]',
    '"neuroradiology"[Journal]',
)


def available_topics() -> tuple[str, ...]:
    return tuple(sorted(TOPICS))


def topic_clause(topic: str) -> str:
    try:
        terms = TOPICS[topic].terms
    except KeyError as exc:
        known = ", ".join(available_topics())
        raise ValueError(f"Unknown topic {topic!r}. Known topics: {known}") from exc
    return "(" + " OR ".join(terms) + ")"


def case_report_clause() -> str:
    return "(" + " OR ".join(CASE_REPORT_TERMS) + ")"


def source_clause(include_author_manuscripts: bool = False) -> str:
    if include_author_manuscripts:
        return "(open_access[filter] OR author_manuscript[filter])"
    return "open_access[filter]"


def text_only_exclusion_clause() -> str:
    return "(" + " OR ".join(TEXT_ONLY_EXCLUSION_TERMS) + ")"


def build_pmc_query(
    topic: str,
    license_profile: str,
    *,
    include_author_manuscripts: bool = False,
    since: str | None = None,
    until: str | None = None,
    extra: str | None = None,
    text_only: bool = False,
) -> str:
    """Build a PMC ESearch query for licensed case-report candidates."""

    clauses = [
        topic_clause(topic),
        case_report_clause(),
        license_filter_query(license_profile),
        source_clause(include_author_manuscripts),
    ]

    if since and until:
        clauses.append(f"{since}:{until}[pmcrdat]")
    elif since:
        clauses.append(f"{since}:3000/01/01[pmcrdat]")
    elif until:
        clauses.append(f"1800/01/01:{until}[pmcrdat]")

    if extra:
        clauses.append(f"({extra})")

    query = " AND ".join(clauses)
    if text_only:
        query = f"{query} NOT {text_only_exclusion_clause()}"
    return query
