"""Command line entry points for the NeurologyBM data pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError

from .case_eval import (
    build_case_prompt,
    create_default_registry,
    ensure_private_workspace,
    load_case_registry,
    new_run_id,
    normalize_result_row,
    write_run_outputs,
)
from .comments import extract_discussion_texts
from .conversion import build_conversion_prompt, conversion_run_dir, write_conversion_artifact
from .deepseek import DeepSeekClient, DeepSeekConfig, assert_private_path
from .filters import keep_article_metadata
from .hf_sources import get_hf_sources
from .huggingface import HuggingFaceClient, collect_hf_source, discover_hf_datasets
from .licenses import LICENSE_PROFILES
from .metadata import extract_article_metadata
from .ncbi import NcbiClient, NcbiConfig, collect_pmcids, esearch, fetch_oai_full_text_xml
from .prompt_extract import extract_prompt_candidates_from_xml_dir
from .public_eval import (
    merge_public_score_files,
    rebuild_public_results_from_raw,
    score_public_deepseek_results,
    run_public_deepseek_eval,
)
from .public_refine import run_public_refinement
from .queries import available_topics, build_pmc_query
from .sources import build_source_query, get_pmc_sources
from .split_audit import audit_public_splits, filter_public_splits_by_audit
from .transform_extract import extract_transformed_challenges_from_xml_dir


DEFAULT_HF_DISCOVERY_QUERIES = [
    "MedCaseReasoning",
    "case report benchmark medical",
    "PMC patients",
    "MultiCaRe clinical case",
    "RareBench medical",
    "neurology benchmark diagnosis",
    "JAMA Clinical Challenge",
    "NEJM Image Challenge",
    "DDXPlus diagnosis",
]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, URLError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurologybm",
        description="License-aware PMC harvesting utilities for neurology case-report corpora.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="Print the PMC ESearch query for a topic/profile.")
    add_query_args(query_parser)
    query_parser.set_defaults(func=cmd_query)

    sources_parser = subparsers.add_parser("sources", help="List configured PMC source buckets.")
    sources_parser.set_defaults(func=cmd_sources)

    hf_sources_parser = subparsers.add_parser("hf-sources", help="List curated Hugging Face dataset leads.")
    hf_sources_parser.add_argument("--sources", nargs="*", default=None, help="HF source keys. Omit for all.")
    hf_sources_parser.add_argument("--max-priority", type=int, help="Only include sources at or below this priority.")
    hf_sources_parser.set_defaults(func=cmd_hf_sources)

    hf_discover_parser = subparsers.add_parser("hf-discover", help="Search Hugging Face for dataset leads.")
    add_hf_client_args(hf_discover_parser)
    hf_discover_parser.add_argument("--queries", nargs="*", default=None, help="Search strings. Defaults to medical case leads.")
    hf_discover_parser.add_argument("--limit", type=int, default=10, help="Maximum matches per query.")
    hf_discover_parser.add_argument("--out", type=Path, default=Path("data/huggingface"), help="Output directory.")
    hf_discover_parser.set_defaults(func=cmd_hf_discover)

    hf_collect_parser = subparsers.add_parser("hf-collect", help="Collect Hugging Face dataset metadata and files.")
    add_hf_client_args(hf_collect_parser)
    hf_collect_parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="HF source keys to collect. Omit for priority-1 sources or pass 'all'.",
    )
    hf_collect_parser.add_argument("--max-priority", type=int, default=1, help="Priority cutoff when --sources is omitted or all.")
    hf_collect_parser.add_argument("--out", type=Path, default=Path("data/huggingface"), help="Output directory.")
    hf_collect_parser.add_argument("--download-files", action="store_true", help="Download configured default data files.")
    hf_collect_parser.add_argument("--max-file-mb", type=int, default=120, help="Skip configured files larger than this size.")
    hf_collect_parser.add_argument("--force", action="store_true", help="Redownload files that already exist.")
    hf_collect_parser.set_defaults(func=cmd_hf_collect)

    plan_parser = subparsers.add_parser("plan", help="Count matching PMC records for source buckets.")
    add_batch_args(plan_parser)
    add_ncbi_args(plan_parser)
    plan_parser.add_argument("--format", choices=("json", "table"), default="table")
    plan_parser.set_defaults(func=cmd_plan)

    harvest_parser = subparsers.add_parser("harvest", help="Search PMC and download OAI full-text XML.")
    add_query_args(harvest_parser)
    add_harvest_args(harvest_parser)
    harvest_parser.add_argument("--limit", type=int, default=25, help="Maximum number of articles to download.")
    harvest_parser.add_argument("--page-size", type=int, default=100, help="ESearch page size.")
    harvest_parser.set_defaults(func=cmd_harvest)

    batch_parser = subparsers.add_parser("harvest-sources", help="Harvest configured PMC source buckets.")
    add_batch_args(batch_parser)
    add_harvest_args(batch_parser)
    batch_parser.add_argument("--per-source-limit", type=int, default=50)
    batch_parser.add_argument("--global-limit", type=int, default=500)
    batch_parser.add_argument("--page-size", type=int, default=100)
    batch_parser.set_defaults(func=cmd_harvest_sources)

    route_parser = subparsers.add_parser("download-routes", help="Download PMC XML from routed case-challenge CSVs.")
    add_ncbi_args(route_parser)
    route_parser.add_argument(
        "--routes-csv",
        type=Path,
        default=Path("data/pmc/metadata/pmc_case_challenge_download_routes.csv"),
        help="Input CSV produced by the case-challenge route audit.",
    )
    route_parser.add_argument("--route", required=True, help="Route name, e.g. benchmark_ready_holdout.")
    route_parser.add_argument(
        "--specialty",
        choices=("neurology_psychiatry_intersection", "neurology", "psychiatry", "rest_or_uncertain"),
        help="Optional specialty bucket filter.",
    )
    route_parser.add_argument(
        "--license-class",
        choices=("training_compatible", "noncommercial_benchmark_only", "no_derivatives_pointer_only"),
        help="Optional license tier filter.",
    )
    route_parser.add_argument("--limit", type=int, default=25, help="Maximum route rows to download.")
    route_parser.add_argument("--out", type=Path, default=Path("data/pmc"), help="Output directory.")
    route_parser.add_argument("--force", action="store_true", help="Redownload XML files that already exist.")
    route_parser.add_argument(
        "--allow-pointer-only",
        action="store_true",
        help="Allow downloading no-derivatives pointer-only rows for private/internal work.",
    )
    route_parser.set_defaults(func=cmd_download_routes)

    deepseek_init_parser = subparsers.add_parser(
        "deepseek-init",
        help="Create private DeepSeek evaluation/conversion directories and a seed case registry.",
    )
    deepseek_init_parser.add_argument(
        "--private-root",
        type=Path,
        default=Path("docs/DO NOT COMMIT TO GITHUB"),
        help="Private gitignored root for case text and API traces.",
    )
    deepseek_init_parser.add_argument("--force-registry", action="store_true", help="Overwrite the seed registry.")
    deepseek_init_parser.set_defaults(func=cmd_deepseek_init)

    eval_parser = subparsers.add_parser(
        "eval-deepseek",
        help="Evaluate private case prompts against DeepSeek Light/Pro or emit a dry-run manifest.",
    )
    eval_parser.add_argument("--model-tier", choices=("light", "pro"), default="light")
    eval_parser.add_argument(
        "--registry",
        type=Path,
        default=Path("docs/DO NOT COMMIT TO GITHUB/deepseek_eval/case_registry.tsv"),
    )
    eval_parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/DO NOT COMMIT TO GITHUB/deepseek_eval/runs"),
        help="Private output directory for run artifacts.",
    )
    eval_parser.add_argument("--case-id", action="append", help="Limit to one or more case IDs.")
    eval_parser.add_argument("--dry-run", action="store_true", help="Do not call DeepSeek; write prompt/run manifest only.")
    eval_parser.add_argument("--run", action="store_true", help="Allow real API calls. Requires DEEPSEEK_API_KEY.")
    eval_parser.add_argument("--temperature", type=float, default=0.0)
    eval_parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    eval_parser.add_argument("--light-model", default=os.getenv("DEEPSEEK_LIGHT_MODEL", "deepseek-chat"))
    eval_parser.add_argument("--pro-model", default=os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-reasoner"))
    eval_parser.add_argument("--timeout", type=float, default=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120")))
    eval_parser.set_defaults(func=cmd_eval_deepseek)

    public_eval_parser = subparsers.add_parser(
        "eval-public-deepseek",
        help="Evaluate public case challenge/answer splits against a DeepSeek-compatible chat model.",
    )
    public_eval_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/pmc/processed/public_case_challenge_splits/public_case_challenge_answer_splits_100_20260609.jsonl"),
        help="Public JSONL manifest with challenge_prompt and answer_rest fields.",
    )
    public_eval_parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/pmc/processed/public_case_challenge_splits/deepseek_runs"),
        help="Output directory for public eval run artifacts.",
    )
    public_eval_parser.add_argument("--model", default=os.getenv("DEEPSEEK_PUBLIC_MODEL", "deepseek-reasoner"))
    public_eval_parser.add_argument("--judge-model", default=os.getenv("DEEPSEEK_JUDGE_MODEL"))
    public_eval_parser.add_argument("--judge", action="store_true", help="Use DeepSeek to grade answers against answer_rest.")
    public_eval_parser.add_argument("--limit", type=int, help="Limit to the first N cases.")
    public_eval_parser.add_argument("--case-id", action="append", help="Evaluate one or more case IDs.")
    public_eval_parser.add_argument(
        "--resume-from-results",
        type=Path,
        help="Optional previous results.tsv; successful case_ids are skipped in the new run.",
    )
    public_eval_parser.add_argument("--dry-run", action="store_true", help="Write run manifest without API calls.")
    public_eval_parser.add_argument("--run", action="store_true", help="Allow real API calls. Requires DEEPSEEK_API_KEY.")
    public_eval_parser.add_argument("--temperature", type=float, default=0.0)
    public_eval_parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    public_eval_parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Optional private file containing the DeepSeek API key. The key is not written to manifests.",
    )
    public_eval_parser.add_argument("--timeout", type=float, default=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120")))
    public_eval_parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("DEEPSEEK_CONCURRENCY", "1")),
        help="Maximum concurrent case workers. DeepSeek v4-pro currently allows 500, v4-flash 2500.",
    )
    public_eval_parser.add_argument(
        "--request-spacing-seconds",
        type=float,
        default=float(os.getenv("DEEPSEEK_REQUEST_SPACING_SECONDS", "0")),
        help="Minimum delay between starting case workers; useful for future RPM/TPM-limited providers.",
    )
    public_eval_parser.add_argument(
        "--extra-body-json",
        help="Optional JSON object merged into the chat completion payload for provider-specific options.",
    )
    public_eval_parser.set_defaults(func=cmd_eval_public_deepseek)

    public_score_parser = subparsers.add_parser(
        "score-public-deepseek",
        help="Score an existing public DeepSeek result TSV against public answer sections.",
    )
    public_score_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/pmc/processed/public_case_challenge_splits/public_case_challenge_answer_splits_100_20260609.jsonl"),
        help="Public JSONL manifest with answer_rest fields.",
    )
    public_score_parser.add_argument("--results", type=Path, required=True, help="Existing eval-public-deepseek results.tsv.")
    public_score_parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/pmc/processed/public_case_challenge_splits/deepseek_runs"),
        help="Output directory for scoring artifacts.",
    )
    public_score_parser.add_argument("--judge-model", default=os.getenv("DEEPSEEK_JUDGE_MODEL", "deepseek-v4-flash"))
    public_score_parser.add_argument("--limit", type=int, help="Limit to the first N result rows.")
    public_score_parser.add_argument("--case-id", action="append", help="Score one or more case IDs.")
    public_score_parser.add_argument(
        "--resume-from-scores",
        type=Path,
        help="Optional previous scores.tsv; completed case_ids are skipped in the new run.",
    )
    public_score_parser.add_argument("--dry-run", action="store_true", help="Write scoring manifest without API calls.")
    public_score_parser.add_argument("--run", action="store_true", help="Allow real judge API calls. Requires DEEPSEEK_API_KEY.")
    public_score_parser.add_argument("--temperature", type=float, default=0.0)
    public_score_parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    public_score_parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Optional private file containing the DeepSeek API key. The key is not written to manifests.",
    )
    public_score_parser.add_argument("--timeout", type=float, default=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120")))
    public_score_parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("DEEPSEEK_CONCURRENCY", "1")),
        help="Maximum concurrent scoring workers.",
    )
    public_score_parser.add_argument(
        "--request-spacing-seconds",
        type=float,
        default=float(os.getenv("DEEPSEEK_REQUEST_SPACING_SECONDS", "0")),
        help="Minimum delay between starting scoring workers; useful for future RPM/TPM-limited providers.",
    )
    public_score_parser.add_argument(
        "--extra-body-json",
        help="Optional JSON object merged into the chat completion payload for provider-specific options.",
    )
    public_score_parser.set_defaults(func=cmd_score_public_deepseek)

    public_audit_parser = subparsers.add_parser(
        "audit-public-splits",
        help="Audit public case challenge splits for multiple-choice leakage and non-case rows before API evaluation.",
    )
    public_audit_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/pmc/processed/public_case_challenge_splits/public_case_challenge_answer_splits_100_20260609.jsonl"),
    )
    public_audit_parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/pmc/processed/public_case_challenge_splits/audits"),
    )
    public_audit_parser.set_defaults(func=cmd_audit_public_splits)

    public_filter_parser = subparsers.add_parser(
        "filter-public-splits",
        help="Write a clean public split manifest containing only rows accepted by an audit CSV.",
    )
    public_filter_parser.add_argument("--manifest", type=Path, required=True)
    public_filter_parser.add_argument("--audit-csv", type=Path, required=True)
    public_filter_parser.add_argument("--out", type=Path, required=True, help="Output clean JSONL manifest.")
    public_filter_parser.add_argument("--metadata-csv", type=Path, help="Optional metadata CSV output.")
    public_filter_parser.set_defaults(func=cmd_filter_public_splits)

    public_rebuild_parser = subparsers.add_parser(
        "rebuild-public-results",
        help="Rebuild a public DeepSeek results TSV from raw API records, preserving structured evidence fields.",
    )
    public_rebuild_parser.add_argument("--manifest", type=Path, required=True)
    public_rebuild_parser.add_argument("--raw-records", type=Path, required=True)
    public_rebuild_parser.add_argument("--out", type=Path, required=True)
    public_rebuild_parser.add_argument("--case-id", action="append", help="Rebuild one or more case IDs.")
    public_rebuild_parser.set_defaults(func=cmd_rebuild_public_results)

    public_merge_scores_parser = subparsers.add_parser(
        "merge-public-scores",
        help="Merge score TSVs from smoke/resume/retry runs; later files replace earlier case IDs.",
    )
    public_merge_scores_parser.add_argument("--scores", type=Path, nargs="+", required=True)
    public_merge_scores_parser.add_argument("--out", type=Path, required=True)
    public_merge_scores_parser.set_defaults(func=cmd_merge_public_scores)

    public_refine_parser = subparsers.add_parser(
        "refine-public-challenges",
        help="Use DeepSeek to rewrite public case splits into self-contained, leak-free benchmark artifacts.",
    )
    public_refine_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/pmc/processed/public_case_challenge_splits/public_case_challenge_answer_splits_87_clean_20260610.jsonl"),
    )
    public_refine_parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/pmc/processed/public_case_challenge_splits/refined"),
    )
    public_refine_parser.add_argument("--model", default=os.getenv("DEEPSEEK_REFINE_MODEL", "deepseek-v4-pro"))
    public_refine_parser.add_argument("--case-id", action="append", help="Refine one or more case IDs.")
    public_refine_parser.add_argument("--case-id-file", type=Path, help="Newline-delimited case IDs to refine.")
    public_refine_parser.add_argument("--resume-from-results", type=Path, help="Previous refined_cases.jsonl to skip.")
    public_refine_parser.add_argument("--limit", type=int, help="Limit selected rows.")
    public_refine_parser.add_argument("--dry-run", action="store_true")
    public_refine_parser.add_argument("--run", action="store_true")
    public_refine_parser.add_argument("--temperature", type=float, default=0.0)
    public_refine_parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    public_refine_parser.add_argument("--api-key-file", type=Path)
    public_refine_parser.add_argument("--timeout", type=float, default=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120")))
    public_refine_parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("DEEPSEEK_CONCURRENCY", "1")),
        help="Maximum concurrent refinement workers. Start below provider limits and raise after smoke tests.",
    )
    public_refine_parser.add_argument(
        "--request-spacing-seconds",
        type=float,
        default=float(os.getenv("DEEPSEEK_REQUEST_SPACING_SECONDS", "0")),
        help="Minimum delay between starting refinement workers; useful for future RPM/TPM-limited providers.",
    )
    public_refine_parser.add_argument("--max-article-chars", type=int, default=45000)
    public_refine_parser.add_argument(
        "--no-article-text",
        action="store_true",
        help="Use only existing prompt and answer/discussion material, without full XML body context.",
    )
    public_refine_parser.add_argument("--api-retries", type=int, default=2)
    public_refine_parser.add_argument("--api-retry-sleep", type=float, default=5.0)
    public_refine_parser.add_argument(
        "--solvability-probe-model",
        help=(
            "Optional model for a blind re-solve gate. Clean-looking refinements that the probe "
            "cannot solve are marked not_solvable."
        ),
    )
    public_refine_parser.add_argument("--extra-body-json", help="Optional JSON object merged into the chat completion payload.")
    public_refine_parser.set_defaults(func=cmd_refine_public_challenges)

    comments_parser = subparsers.add_parser(
        "extract-comments",
        help="Extract visible text from private downloaded case-discussion HTML pages.",
    )
    comments_parser.add_argument(
        "--raw-html-dir",
        type=Path,
        default=Path("docs/DO NOT COMMIT TO GITHUB/nejm_case_challenge_discussions/raw_html"),
    )
    comments_parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/DO NOT COMMIT TO GITHUB/nejm_case_challenge_discussions/extracted_text"),
    )
    comments_parser.set_defaults(func=cmd_extract_comments)

    convert_parser = subparsers.add_parser(
        "convert-case",
        help="Convert private case studies into challenge/evidence-map artifacts with DeepSeek Pro or dry-run.",
    )
    convert_parser.add_argument(
        "--registry",
        type=Path,
        default=Path("docs/DO NOT COMMIT TO GITHUB/deepseek_eval/case_registry.tsv"),
    )
    convert_parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/DO NOT COMMIT TO GITHUB/case_conversion"),
        help="Private output directory for conversion artifacts.",
    )
    convert_parser.add_argument("--case-id", action="append", help="Limit to one or more case IDs.")
    convert_parser.add_argument("--model-tier", choices=("pro", "light"), default="pro")
    convert_parser.add_argument("--dry-run", action="store_true", help="Do not call DeepSeek; write planned artifacts only.")
    convert_parser.add_argument("--run", action="store_true", help="Allow real API calls. Requires DEEPSEEK_API_KEY.")
    convert_parser.add_argument("--temperature", type=float, default=0.0)
    convert_parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    convert_parser.add_argument("--light-model", default=os.getenv("DEEPSEEK_LIGHT_MODEL", "deepseek-chat"))
    convert_parser.add_argument("--pro-model", default=os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-reasoner"))
    convert_parser.add_argument("--timeout", type=float, default=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120")))
    convert_parser.set_defaults(func=cmd_convert_case)

    extract_prompts_parser = subparsers.add_parser(
        "extract-prompt-candidates",
        help="Extract challenge prompt/answer candidates from downloaded JATS/OAI XML.",
    )
    extract_prompts_parser.add_argument("--xml-dir", type=Path, required=True, help="Directory containing PMC*.xml files.")
    extract_prompts_parser.add_argument(
        "--source-metadata",
        type=Path,
        help="Optional route metadata JSONL to attach route/license context.",
    )
    extract_prompts_parser.add_argument("--out", type=Path, required=True, help="Output JSONL for prompt candidates.")
    extract_prompts_parser.set_defaults(func=cmd_extract_prompt_candidates)

    transform_prompts_parser = subparsers.add_parser(
        "extract-transformed-challenges",
        help="Create challenge/answer splits from public case-report XML using deterministic section extraction.",
    )
    transform_prompts_parser.add_argument("--xml-dir", type=Path, required=True, help="Directory containing PMC*.xml files.")
    transform_prompts_parser.add_argument(
        "--source-metadata",
        type=Path,
        help="Optional route metadata JSONL to attach route/license context.",
    )
    transform_prompts_parser.add_argument("--out", type=Path, required=True, help="Output JSONL for transformed challenges.")
    transform_prompts_parser.set_defaults(func=cmd_extract_transformed_challenges)

    validate_parser = subparsers.add_parser(
        "validate-cases",
        help="Validate (and optionally mend) a challenge manifest for determinacy/solvability defects "
             "(under-determined / gold-not-a-diagnosis / prompt-refutes-gold).",
    )
    validate_parser.add_argument("--manifest", type=Path, required=True, help="Challenge manifest JSONL.")
    validate_parser.add_argument("--out", type=Path, required=True, help="Output dir: validation.jsonl, mended_manifest.jsonl, summary.json.")
    validate_parser.add_argument("--model", default=os.getenv("DEEPSEEK_VALIDATE_MODEL", "deepseek-v4-flash"))
    validate_parser.add_argument("--mend", action="store_true", help="Apply add_result mends (gold unchanged) -> mended_manifest.jsonl.")
    validate_parser.add_argument("--deterministic-only", action="store_true", help="Specificity flags only, no API.")
    validate_parser.add_argument("--concurrency", type=int, default=8)
    validate_parser.add_argument("--limit", type=int)
    validate_parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    validate_parser.add_argument("--api-key-file", type=Path)
    validate_parser.add_argument("--timeout", type=float, default=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120")))
    validate_parser.set_defaults(func=cmd_validate_cases)

    return parser


def cmd_validate_cases(args: argparse.Namespace) -> int:
    from .case_validation import run_case_validation
    client = None
    if not args.deterministic_only:
        config = DeepSeekConfig(
            api_key=_api_key_from_env_or_file(args.api_key_file),
            base_url=args.base_url,
            timeout_seconds=args.timeout,
        )
        client = DeepSeekClient(config)
    summary = run_case_validation(
        client=client, manifest_path=args.manifest, out_dir=args.out, model=args.model,
        mend=args.mend, concurrency=args.concurrency, limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


def add_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--topic", choices=available_topics(), default="neurology")
    parser.add_argument("--license-profile", choices=sorted(LICENSE_PROFILES), default="training")
    parser.add_argument(
        "--include-author-manuscripts",
        action="store_true",
        help="Include author manuscripts in addition to the PMC Open Access Subset.",
    )
    parser.add_argument("--since", help="Earliest PMC release date, YYYY/MM/DD.")
    parser.add_argument("--until", help="Latest PMC release date, YYYY/MM/DD.")
    parser.add_argument("--extra", help="Additional raw PMC query clause to AND with the generated query.")
    parser.add_argument("--text-only", action="store_true", help="Exclude image/video/radiology-heavy sources.")


def add_batch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="PMC source keys to use. Omit or pass 'all' for every configured source.",
    )
    parser.add_argument("--license-profile", choices=sorted(LICENSE_PROFILES), default="training")
    parser.add_argument(
        "--include-author-manuscripts",
        action="store_true",
        help="Include author manuscripts in addition to the PMC Open Access Subset.",
    )
    parser.add_argument("--since", help="Earliest PMC release date, YYYY/MM/DD.")
    parser.add_argument("--until", help="Latest PMC release date, YYYY/MM/DD.")
    parser.add_argument("--include-imaging", action="store_true", help="Do not apply text-only exclusions.")


def add_ncbi_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--email", default=os.getenv("NCBI_EMAIL"), help="Contact email for NCBI requests.")
    parser.add_argument("--api-key", default=os.getenv("NCBI_API_KEY"), help="Optional NCBI API key.")
    parser.add_argument("--tool", default="NeurologyBM", help="NCBI tool name.")
    parser.add_argument("--sleep", type=float, default=0.34, help="Minimum seconds between requests.")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification. Intended only for broken local certificate stores.",
    )


def add_hf_client_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification. Intended only for broken local certificate stores.",
    )


def add_harvest_args(parser: argparse.ArgumentParser) -> None:
    add_ncbi_args(parser)
    parser.add_argument("--out", type=Path, default=Path("data/pmc"), help="Output directory.")
    parser.add_argument("--force", action="store_true", help="Redownload XML files that already exist.")
    parser.add_argument(
        "--keep-rejected",
        action="store_true",
        help="Keep raw XML for records rejected by post-download metadata filters.",
    )
    parser.add_argument(
        "--no-strict-neurology",
        action="store_true",
        help="Do not require parsed metadata to contain a neurology marker.",
    )
    parser.add_argument("--allow-non-case", action="store_true", help="Do not require parsed metadata to be case-like.")


def cmd_query(args: argparse.Namespace) -> int:
    print(_query_from_args(args))
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    rows = []
    for source in get_pmc_sources(None):
        rows.append(
            {
                "key": source.key,
                "label": source.label,
                "topic": source.topic,
                "extra": source.extra,
                "text_only": source.text_only,
                "notes": source.notes,
            }
        )
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def cmd_hf_sources(args: argparse.Namespace) -> int:
    sources = get_hf_sources(args.sources, max_priority=args.max_priority)
    rows = [asdict(source) for source in sources]
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def cmd_hf_discover(args: argparse.Namespace) -> int:
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    queries = args.queries or DEFAULT_HF_DISCOVERY_QUERIES
    client = _hf_client_from_args(args)
    out: Path = args.out
    manifest_dir = out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = discover_hf_datasets(client, queries, limit=args.limit)
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "queries": queries,
        "limit": args.limit,
        "source_service": "Hugging Face dataset API",
        "results": results,
    }
    manifest_path = manifest_dir / f"hf_discovery_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def cmd_hf_collect(args: argparse.Namespace) -> int:
    if args.max_file_mb < 0:
        raise ValueError("--max-file-mb must be non-negative")

    client = _hf_client_from_args(args)
    out: Path = args.out
    manifest_dir = out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    sources = get_hf_sources(args.sources, max_priority=args.max_priority)
    max_file_bytes = args.max_file_mb * 1024 * 1024
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source_runs = [
        collect_hf_source(
            client,
            source,
            out,
            download_files=args.download_files,
            max_file_bytes=max_file_bytes,
            force=args.force,
        )
        for source in sources
    ]

    downloaded_file_count = 0
    downloaded_bytes = 0
    skipped_file_count = 0
    error_count = 0
    for row in source_runs:
        error_count += len(row.get("errors") or [])
        for file_row in row.get("files") or []:
            if file_row.get("downloaded"):
                downloaded_file_count += 1
                downloaded_bytes += int(file_row.get("bytes") or 0)
            else:
                skipped_file_count += 1

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_service": "Hugging Face dataset API and dataset file resolver",
        "download_files": args.download_files,
        "max_file_mb": args.max_file_mb,
        "sources_requested": [source.key for source in sources],
        "sources": source_runs,
        "source_count": len(source_runs),
        "downloaded_file_count": downloaded_file_count,
        "downloaded_bytes": downloaded_bytes,
        "skipped_file_count": skipped_file_count,
        "error_count": error_count,
    }
    manifest_path = manifest_dir / f"hf_collect_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 1 if error_count else 0


def cmd_plan(args: argparse.Namespace) -> int:
    client = _client_from_args(args)
    rows = []
    for source in get_pmc_sources(args.sources):
        query = build_source_query(
            source,
            args.license_profile,
            include_author_manuscripts=args.include_author_manuscripts,
            since=args.since,
            until=args.until,
            text_only=not args.include_imaging,
        )
        result = esearch(client, query, retmax=0)
        count = int(result.get("esearchresult", {}).get("count", "0"))
        rows.append({"source": source.key, "label": source.label, "count": count, "query": query})

    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        width = max(len(row["source"]) for row in rows) if rows else 6
        for row in rows:
            print(f"{row['source']:<{width}}  {row['count']:>8}  {row['label']}")
    return 0


def cmd_harvest(args: argparse.Namespace) -> int:
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")

    out: Path = args.out
    raw_dir, manifest_dir, metadata_dir = _prepare_output_dirs(out)

    query = _query_from_args(args)
    client = _client_from_args(args)
    pmc_ids, total_matches = collect_pmcids(client, query, limit=args.limit, page_size=args.page_size)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata_path = metadata_dir / f"articles_{run_id}.jsonl"
    rejected_path = metadata_dir / f"rejected_{run_id}.jsonl"
    stats = _harvest_articles(
        client,
        pmc_ids,
        raw_dir=raw_dir,
        metadata_path=metadata_path,
        rejected_path=rejected_path,
        license_profile=args.license_profile,
        force=args.force,
        keep_rejected=args.keep_rejected,
        strict_neurology=not args.no_strict_neurology,
        text_only=args.text_only,
        case_only=not args.allow_non_case,
        source_key=None,
    )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "topic": args.topic,
        "license_profile": args.license_profile,
        "include_author_manuscripts": args.include_author_manuscripts,
        "limit": args.limit,
        "total_matches": total_matches,
        "pmcids": [f"PMC{item.removeprefix('PMC')}" for item in pmc_ids],
        "metadata_path": str(metadata_path),
        "rejected_path": str(rejected_path),
        "source_services": [
            "NCBI E-Utilities ESearch",
            "PMC OAI-PMH GetRecord metadataPrefix=pmc",
        ],
        **stats,
    }
    manifest_path = manifest_dir / f"run_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 1 if stats["failed"] else 0


def cmd_harvest_sources(args: argparse.Namespace) -> int:
    if args.per_source_limit < 1:
        raise ValueError("--per-source-limit must be at least 1")
    if args.global_limit < 1:
        raise ValueError("--global-limit must be at least 1")

    raw_dir, manifest_dir, metadata_dir = _prepare_output_dirs(args.out)
    client = _client_from_args(args)
    sources = get_pmc_sources(args.sources)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata_path = metadata_dir / f"articles_{run_id}.jsonl"
    rejected_path = metadata_dir / f"rejected_{run_id}.jsonl"
    seen_pmcids: set[str] = set()
    aggregate = _empty_stats()
    source_runs = []

    for source in sources:
        remaining = args.global_limit - int(aggregate["kept"])
        if remaining <= 0:
            break
        limit = min(args.per_source_limit, remaining)
        query = build_source_query(
            source,
            args.license_profile,
            include_author_manuscripts=args.include_author_manuscripts,
            since=args.since,
            until=args.until,
            text_only=not args.include_imaging,
        )
        pmc_ids, total_matches = collect_pmcids(client, query, limit=limit, page_size=args.page_size)
        stats = _harvest_articles(
            client,
            pmc_ids,
            raw_dir=raw_dir,
            metadata_path=metadata_path,
            rejected_path=rejected_path,
            license_profile=args.license_profile,
            force=args.force,
            keep_rejected=args.keep_rejected,
            strict_neurology=not args.no_strict_neurology,
            text_only=not args.include_imaging,
            case_only=not args.allow_non_case,
            source_key=source.key,
            seen_pmcids=seen_pmcids,
        )
        _merge_stats(aggregate, stats)
        source_runs.append(
            {
                "source": source.key,
                "label": source.label,
                "query": query,
                "total_matches": total_matches,
                "requested_limit": limit,
                **stats,
            }
        )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "license_profile": args.license_profile,
        "include_author_manuscripts": args.include_author_manuscripts,
        "include_imaging": args.include_imaging,
        "per_source_limit": args.per_source_limit,
        "global_limit": args.global_limit,
        "metadata_path": str(metadata_path),
        "rejected_path": str(rejected_path),
        "sources": source_runs,
        "source_services": [
            "NCBI E-Utilities ESearch",
            "PMC OAI-PMH GetRecord metadataPrefix=pmc",
        ],
        **aggregate,
    }
    manifest_path = manifest_dir / f"batch_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 1 if aggregate["failed"] else 0


def cmd_download_routes(args: argparse.Namespace) -> int:
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if not args.routes_csv.exists():
        raise ValueError(f"Route CSV not found: {args.routes_csv}")

    rows = _select_route_rows(
        args.routes_csv,
        route=args.route,
        specialty=args.specialty,
        license_class=args.license_class,
        limit=args.limit,
        allow_pointer_only=args.allow_pointer_only,
    )
    client = _client_from_args(args)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = args.out / "raw" / "xml" / args.route
    manifest_dir = args.out / "manifests"
    metadata_dir = args.out / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = metadata_dir / f"route_{args.route}_{run_id}.jsonl"
    manifest_path = manifest_dir / f"route_{args.route}_{run_id}.json"
    stats = _download_route_articles(client, rows, raw_dir=raw_dir, metadata_path=metadata_path, force=args.force)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "routes_csv": str(args.routes_csv),
        "route": args.route,
        "specialty": args.specialty,
        "license_class": args.license_class,
        "limit": args.limit,
        "allow_pointer_only": args.allow_pointer_only,
        "selected_rows": len(rows),
        "metadata_path": str(metadata_path),
        "raw_dir": str(raw_dir),
        "source_services": [
            "PMC OAI-PMH GetRecord metadataPrefix=pmc",
        ],
        **stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 1 if stats["failed"] else 0


def cmd_deepseek_init(args: argparse.Namespace) -> int:
    paths = ensure_private_workspace(args.private_root)
    registry_path = create_default_registry(paths["registry"], force=args.force_registry)
    manifest = {
        "private_root": str(paths["root"]),
        "deepseek_eval_dir": str(paths["eval"]),
        "deepseek_runs_dir": str(paths["runs"]),
        "case_conversion_dir": str(paths["conversion"]),
        "case_conversion_runs_dir": str(paths["conversion_runs"]),
        "registry": str(registry_path),
        "registry_created_or_present": registry_path.exists(),
    }
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def cmd_eval_deepseek(args: argparse.Namespace) -> int:
    if not args.dry_run and not args.run:
        raise ValueError("Pass --dry-run to prepare artifacts or --run to make DeepSeek API calls.")
    if args.dry_run and args.run:
        raise ValueError("Use only one of --dry-run or --run.")
    assert_private_path(args.out)

    records = load_case_registry(args.registry)
    if args.case_id:
        wanted = set(args.case_id)
        records = [record for record in records if record.case_id in wanted]
        missing = wanted - {record.case_id for record in records}
        if missing:
            raise ValueError(f"Case IDs not found in registry: {sorted(missing)}")
    if not records:
        raise ValueError("No case records selected.")

    config = DeepSeekConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=args.base_url,
        light_model=args.light_model,
        pro_model=args.pro_model,
        timeout_seconds=args.timeout,
    )
    model = config.model_for_tier(args.model_tier)
    client = DeepSeekClient(config)
    run_id = new_run_id()
    run_dir = args.out / f"{args.model_tier}_{run_id}"
    result_rows = []
    raw_records = []

    for record in records:
        if not record.source_path.exists():
            raise ValueError(f"Case source not found for {record.case_id}: {record.source_path}")
        case_text = record.source_path.read_text(encoding="utf-8", errors="replace")
        system_prompt, user_prompt = build_case_prompt(record, case_text)
        if args.dry_run:
            parsed_content = {
                "final_diagnosis": "",
                "etiology": "",
                "top_differential": [],
                "recommended_next_step": "",
                "confidence": "",
            }
            raw_record = {
                "case_id": record.case_id,
                "dry_run": True,
                "model": model,
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
                "source_path": str(record.source_path),
                "prompt_template": record.prompt_template,
            }
        else:
            raw_record = client.chat_json(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
            )
            parsed_content = raw_record.get("parsed_content", {})
            raw_record["case_id"] = record.case_id
            raw_record["source_path"] = str(record.source_path)

        result_rows.append(
            normalize_result_row(
                record=record,
                model=model,
                parsed_content=parsed_content,
                score_status="ungradable",
            )
        )
        raw_records.append(raw_record)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "model_tier": args.model_tier,
        "model": model,
        "registry": str(args.registry),
        "selected_case_count": len(records),
        "case_ids": [record.case_id for record in records],
        "temperature": args.temperature,
        "base_url": args.base_url,
        "raw_prompts_and_outputs_private": True,
    }
    write_run_outputs(run_dir=run_dir, result_rows=result_rows, raw_records=raw_records, manifest=manifest)
    print(json.dumps({"run_dir": str(run_dir), **manifest}, indent=2, sort_keys=True))
    return 0


def cmd_eval_public_deepseek(args: argparse.Namespace) -> int:
    if not args.dry_run and not args.run:
        raise ValueError("Pass --dry-run to prepare artifacts or --run to make DeepSeek API calls.")
    if args.dry_run and args.run:
        raise ValueError("Use only one of --dry-run or --run.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    _validate_concurrency_args(args)
    if not args.manifest.exists():
        raise ValueError(f"Public split manifest not found: {args.manifest}")
    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else None
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("--extra-body-json must be a JSON object")

    config = DeepSeekConfig(
        api_key=_api_key_from_env_or_file(args.api_key_file),
        base_url=args.base_url,
        timeout_seconds=args.timeout,
    )
    client = DeepSeekClient(config)
    manifest = run_public_deepseek_eval(
        client=client,
        manifest_path=args.manifest,
        out_root=args.out,
        model=args.model,
        limit=args.limit,
        case_ids=set(args.case_id or []) or None,
        dry_run=args.dry_run,
        temperature=args.temperature,
        judge=args.judge,
        judge_model=args.judge_model,
        extra_body=extra_body,
        resume_results=args.resume_from_results,
        concurrency=args.concurrency,
        request_spacing_seconds=args.request_spacing_seconds,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def cmd_score_public_deepseek(args: argparse.Namespace) -> int:
    if not args.dry_run and not args.run:
        raise ValueError("Pass --dry-run to prepare artifacts or --run to make DeepSeek API calls.")
    if args.dry_run and args.run:
        raise ValueError("Use only one of --dry-run or --run.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    _validate_concurrency_args(args)
    if not args.manifest.exists():
        raise ValueError(f"Public split manifest not found: {args.manifest}")
    if not args.results.exists():
        raise ValueError(f"Public results TSV not found: {args.results}")
    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else None
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("--extra-body-json must be a JSON object")

    config = DeepSeekConfig(
        api_key=_api_key_from_env_or_file(args.api_key_file),
        base_url=args.base_url,
        timeout_seconds=args.timeout,
    )
    client = DeepSeekClient(config)
    manifest = score_public_deepseek_results(
        client=client,
        split_manifest_path=args.manifest,
        results_path=args.results,
        out_root=args.out,
        judge_model=args.judge_model,
        dry_run=args.dry_run,
        temperature=args.temperature,
        extra_body=extra_body,
        limit=args.limit,
        case_ids=set(args.case_id or []) or None,
        resume_scores=args.resume_from_scores,
        concurrency=args.concurrency,
        request_spacing_seconds=args.request_spacing_seconds,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def cmd_audit_public_splits(args: argparse.Namespace) -> int:
    if not args.manifest.exists():
        raise ValueError(f"Public split manifest not found: {args.manifest}")
    summary = audit_public_splits(manifest_path=args.manifest, out_dir=args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_filter_public_splits(args: argparse.Namespace) -> int:
    if not args.manifest.exists():
        raise ValueError(f"Public split manifest not found: {args.manifest}")
    if not args.audit_csv.exists():
        raise ValueError(f"Audit CSV not found: {args.audit_csv}")
    summary = filter_public_splits_by_audit(
        manifest_path=args.manifest,
        audit_csv_path=args.audit_csv,
        output_jsonl=args.out,
        metadata_csv=args.metadata_csv,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_rebuild_public_results(args: argparse.Namespace) -> int:
    if not args.manifest.exists():
        raise ValueError(f"Public split manifest not found: {args.manifest}")
    if not args.raw_records.exists():
        raise ValueError(f"Raw records JSONL not found: {args.raw_records}")
    summary = rebuild_public_results_from_raw(
        split_manifest_path=args.manifest,
        raw_records_path=args.raw_records,
        output_tsv=args.out,
        case_ids=set(args.case_id or []) or None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_merge_public_scores(args: argparse.Namespace) -> int:
    missing = [path for path in args.scores if not path.exists()]
    if missing:
        raise ValueError(f"Score TSV not found: {missing[0]}")
    summary = merge_public_score_files(score_paths=args.scores, output_tsv=args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_refine_public_challenges(args: argparse.Namespace) -> int:
    if not args.dry_run and not args.run:
        raise ValueError("Pass --dry-run to prepare artifacts or --run to make DeepSeek API calls.")
    if args.dry_run and args.run:
        raise ValueError("Use only one of --dry-run or --run.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.max_article_chars < 1000:
        raise ValueError("--max-article-chars must be at least 1000")
    if args.api_retries < 0:
        raise ValueError("--api-retries must be at least 0")
    if args.api_retry_sleep < 0:
        raise ValueError("--api-retry-sleep must be non-negative")
    _validate_concurrency_args(args)
    if not args.manifest.exists():
        raise ValueError(f"Public split manifest not found: {args.manifest}")
    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else None
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("--extra-body-json must be a JSON object")

    case_ids = set(args.case_id or [])
    if args.case_id_file:
        if not args.case_id_file.exists():
            raise ValueError(f"Case ID file not found: {args.case_id_file}")
        case_ids.update(line.strip() for line in args.case_id_file.read_text(encoding="utf-8").splitlines() if line.strip())

    config = DeepSeekConfig(
        api_key=_api_key_from_env_or_file(args.api_key_file),
        base_url=args.base_url,
        timeout_seconds=args.timeout,
    )
    manifest = run_public_refinement(
        client=DeepSeekClient(config),
        manifest_path=args.manifest,
        out_root=args.out,
        model=args.model,
        dry_run=args.dry_run,
        temperature=args.temperature,
        case_ids=case_ids or None,
        limit=args.limit,
        extra_body=extra_body,
        resume_results=args.resume_from_results,
        max_article_chars=args.max_article_chars,
        include_article_text=not args.no_article_text,
        api_retries=args.api_retries,
        api_retry_sleep_seconds=args.api_retry_sleep,
        solvability_probe_model=args.solvability_probe_model,
        concurrency=args.concurrency,
        request_spacing_seconds=args.request_spacing_seconds,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _validate_concurrency_args(args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.request_spacing_seconds < 0:
        raise ValueError("--request-spacing-seconds must be non-negative")


def _api_key_from_env_or_file(api_key_file: Path | None) -> str | None:
    if api_key_file:
        if not api_key_file.exists():
            raise ValueError(f"API key file not found: {api_key_file}")
        return api_key_file.read_text(encoding="utf-8").strip()
    return os.getenv("DEEPSEEK_API_KEY")


def cmd_extract_comments(args: argparse.Namespace) -> int:
    if not args.raw_html_dir.exists():
        raise ValueError(f"Raw HTML directory not found: {args.raw_html_dir}")
    rows = extract_discussion_texts(args.raw_html_dir, args.out)
    manifest_path = args.out / "manifest.json"
    assert_private_path(manifest_path)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_html_dir": str(args.raw_html_dir),
        "out": str(args.out),
        "file_count": len(rows),
        "files": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def cmd_convert_case(args: argparse.Namespace) -> int:
    if not args.dry_run and not args.run:
        raise ValueError("Pass --dry-run to prepare artifacts or --run to make DeepSeek API calls.")
    if args.dry_run and args.run:
        raise ValueError("Use only one of --dry-run or --run.")
    assert_private_path(args.out)

    records = load_case_registry(args.registry)
    if args.case_id:
        wanted = set(args.case_id)
        records = [record for record in records if record.case_id in wanted]
        missing = wanted - {record.case_id for record in records}
        if missing:
            raise ValueError(f"Case IDs not found in registry: {sorted(missing)}")
    else:
        records = [record for record in records if record.next_queue == "conversion_needed"]
    if not records:
        raise ValueError("No case records selected for conversion.")

    config = DeepSeekConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=args.base_url,
        light_model=args.light_model,
        pro_model=args.pro_model,
        timeout_seconds=args.timeout,
    )
    model = config.model_for_tier(args.model_tier)
    client = DeepSeekClient(config)
    run_dir = conversion_run_dir(args.out, args.model_tier)
    assert_private_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dirs = []

    for record in records:
        if not record.source_path.exists():
            raise ValueError(f"Case source not found for {record.case_id}: {record.source_path}")
        case_text = record.source_path.read_text(encoding="utf-8", errors="replace")
        system_prompt, user_prompt = build_conversion_prompt(record, case_text)
        if args.dry_run:
            parsed_content = {
                "challenge_prompt": "",
                "answer_key": {},
                "evidence_map": [],
                "hypothesis_bank": [],
                "outcome_summary": "",
            }
            raw_record = {
                "case_id": record.case_id,
                "dry_run": True,
                "model": model,
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
                "source_path": str(record.source_path),
            }
        else:
            raw_record = client.chat_json(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
            )
            parsed_content = raw_record.get("parsed_content", {})
            raw_record["case_id"] = record.case_id
            raw_record["source_path"] = str(record.source_path)
        artifact_dir = write_conversion_artifact(
            out_root=run_dir,
            record=record,
            model=model,
            content=parsed_content,
            raw_record=raw_record,
            dry_run=args.dry_run,
        )
        artifact_dirs.append(str(artifact_dir))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "model_tier": args.model_tier,
        "model": model,
        "registry": str(args.registry),
        "selected_case_count": len(records),
        "case_ids": [record.case_id for record in records],
        "artifact_dirs": artifact_dirs,
        "raw_prompts_and_outputs_private": True,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), **manifest}, indent=2, sort_keys=True))
    return 0


def cmd_extract_prompt_candidates(args: argparse.Namespace) -> int:
    if not args.xml_dir.exists():
        raise ValueError(f"XML directory not found: {args.xml_dir}")
    summary = extract_prompt_candidates_from_xml_dir(
        xml_dir=args.xml_dir,
        output_jsonl=args.out,
        source_metadata_jsonl=args.source_metadata,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_extract_transformed_challenges(args: argparse.Namespace) -> int:
    if not args.xml_dir.exists():
        raise ValueError(f"XML directory not found: {args.xml_dir}")
    summary = extract_transformed_challenges_from_xml_dir(
        xml_dir=args.xml_dir,
        output_jsonl=args.out,
        source_metadata_jsonl=args.source_metadata,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _select_route_rows(
    routes_csv: Path,
    *,
    route: str,
    specialty: str | None,
    license_class: str | None,
    limit: int,
    allow_pointer_only: bool,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    skipped_pointer_only = 0
    with routes_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("download_route") != route:
                continue
            if specialty and row.get("specialty_bucket") != specialty:
                continue
            if license_class and row.get("tier") != license_class:
                continue
            if row.get("tier") == "no_derivatives_pointer_only" and not allow_pointer_only:
                skipped_pointer_only += 1
                continue
            if not row.get("pmcid"):
                continue
            selected.append(row)
            if len(selected) >= limit:
                break

    if not selected:
        hint = ""
        if skipped_pointer_only:
            hint = " Matching rows were pointer-only; pass --allow-pointer-only for private/internal downloads."
        raise ValueError(
            f"No route rows selected for route={route!r}, specialty={specialty!r}, "
            f"license_class={license_class!r}.{hint}"
        )
    return selected


def _download_route_articles(
    client: NcbiClient,
    rows: list[dict[str, str]],
    *,
    raw_dir: Path,
    metadata_path: Path,
    force: bool,
) -> dict[str, object]:
    downloaded = 0
    skipped_existing = 0
    kept = 0
    failed: list[dict[str, str]] = []

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for row in rows:
            pmcid = str(row["pmcid"])
            numeric_id = pmcid.removeprefix("PMC")
            xml_path = raw_dir / f"{pmcid}.xml"
            try:
                if xml_path.exists() and not force:
                    xml_bytes = xml_path.read_bytes()
                    skipped_existing += 1
                else:
                    xml_bytes = fetch_oai_full_text_xml(client, numeric_id)
                    xml_path.write_bytes(xml_bytes)
                    downloaded += 1

                metadata = extract_article_metadata(xml_bytes)
                metadata["pmcid"] = metadata.get("pmcid") or pmcid
                metadata["xml_path"] = str(xml_path)
                metadata["route_source"] = {
                    key: row.get(key)
                    for key in (
                        "download_route",
                        "specialty_bucket",
                        "tier",
                        "challenge_confidence",
                        "license_keys",
                        "matched_patterns",
                        "doi",
                        "pmid",
                        "pmc_url",
                        "public_text_policy",
                        "holdout_from_training",
                    )
                }
                metadata["keep"] = True
                metadata["reject_reasons"] = []
                metadata_file.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
                kept += 1
            except Exception as exc:  # noqa: BLE001 - continue route downloads and preserve exact failure.
                failed.append({"pmcid": pmcid, "error": str(exc)})

    return {
        "downloaded": downloaded,
        "skipped_existing": skipped_existing,
        "kept": kept,
        "failed": failed,
    }


def _harvest_articles(
    client: NcbiClient,
    pmc_ids: list[str],
    *,
    raw_dir: Path,
    metadata_path: Path,
    rejected_path: Path,
    license_profile: str,
    force: bool,
    keep_rejected: bool,
    strict_neurology: bool,
    text_only: bool,
    case_only: bool,
    source_key: str | None,
    seen_pmcids: set[str] | None = None,
) -> dict[str, object]:
    downloaded = 0
    skipped_existing = 0
    skipped_duplicate = 0
    kept = 0
    rejected = 0
    failed: list[dict[str, str]] = []

    with (
        metadata_path.open("a", encoding="utf-8") as metadata_file,
        rejected_path.open("a", encoding="utf-8") as rejected_file,
    ):
        for numeric_id in pmc_ids:
            pmcid = f"PMC{numeric_id.removeprefix('PMC')}"
            if seen_pmcids is not None and pmcid in seen_pmcids:
                skipped_duplicate += 1
                continue
            if seen_pmcids is not None:
                seen_pmcids.add(pmcid)

            xml_path = raw_dir / f"{pmcid}.xml"
            try:
                if xml_path.exists() and not force:
                    xml_bytes = xml_path.read_bytes()
                    skipped_existing += 1
                else:
                    xml_bytes = fetch_oai_full_text_xml(client, numeric_id)
                    xml_path.write_bytes(xml_bytes)
                    downloaded += 1

                metadata = extract_article_metadata(xml_bytes, license_profile=license_profile)
                metadata["pmcid"] = metadata.get("pmcid") or pmcid
                metadata["xml_path"] = str(xml_path)
                if source_key:
                    metadata["source_key"] = source_key
                keep, reject_reasons = keep_article_metadata(
                    metadata,
                    strict_neurology=strict_neurology,
                    text_only=text_only,
                    case_only=case_only,
                )
                metadata["keep"] = keep
                metadata["reject_reasons"] = reject_reasons
                if keep:
                    kept += 1
                    metadata_file.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
                else:
                    rejected += 1
                    rejected_file.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
                    if not keep_rejected:
                        xml_path.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001 - preserve harvest progress and log exact article failures.
                failed.append({"pmcid": pmcid, "error": str(exc)})

    return {
        "downloaded": downloaded,
        "skipped_existing": skipped_existing,
        "skipped_duplicate": skipped_duplicate,
        "kept": kept,
        "rejected": rejected,
        "failed": failed,
    }


def _client_from_args(args: argparse.Namespace) -> NcbiClient:
    if not args.email:
        print(
            "warning: set --email or NCBI_EMAIL so NCBI can contact you about automated traffic.",
            file=sys.stderr,
        )
    return NcbiClient(
        NcbiConfig(
            tool=args.tool,
            email=args.email,
            api_key=args.api_key,
            verify_tls=not args.insecure,
            min_interval_seconds=args.sleep,
        )
    )


def _hf_client_from_args(args: argparse.Namespace) -> HuggingFaceClient:
    return HuggingFaceClient(verify_tls=not args.insecure, timeout_seconds=args.timeout)


def _prepare_output_dirs(out: Path) -> tuple[Path, Path, Path]:
    raw_dir = out / "raw" / "xml"
    manifest_dir = out / "manifests"
    metadata_dir = out / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, manifest_dir, metadata_dir


def _empty_stats() -> dict[str, object]:
    return {
        "downloaded": 0,
        "skipped_existing": 0,
        "skipped_duplicate": 0,
        "kept": 0,
        "rejected": 0,
        "failed": [],
    }


def _merge_stats(aggregate: dict[str, object], stats: dict[str, object]) -> None:
    for key in ("downloaded", "skipped_existing", "skipped_duplicate", "kept", "rejected"):
        aggregate[key] = int(aggregate[key]) + int(stats[key])
    aggregate["failed"] = list(aggregate["failed"]) + list(stats["failed"])


def _query_from_args(args: argparse.Namespace) -> str:
    return build_pmc_query(
        args.topic,
        args.license_profile,
        include_author_manuscripts=args.include_author_manuscripts,
        since=args.since,
        until=args.until,
        extra=args.extra,
        text_only=args.text_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
