from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from filter_lcqa_by_tokens import update_lengths  # noqa: E402
from longcontext.benchmark_guard import expand_heldout_benchmarks, mark_training_eligibility  # noqa: E402
from longcontext.length import BUCKET_ORDER, get_length_bucket, is_main_long_context_bucket  # noqa: E402
from longcontext.schema import LCQASample  # noqa: E402


CLEANING_STATUSES = (
    "clean_ready",
    "needs_teacher_label",
    "needs_query_generation",
    "needs_answer_relabel",
    "benchmark_or_eval_only",
    "drop_or_review",
)

QUEUE_FILES = {
    "teacher_current": "teacher_queue_le_256k.jsonl",
    "teacher_long": "teacher_queue_256k_900k.jsonl",
    "benchmark_or_eval_only": "benchmark_or_eval_only.jsonl",
    "needs_query_generation": "needs_query_generation.jsonl",
    "drop_or_review": "drop_or_review.jsonl",
    "clean_ready": "clean_ready.jsonl",
    "needs_answer_relabel": "needs_answer_relabel.jsonl",
}

BAD_QUERY_PATTERNS = (
    re.compile(r"^\s*(n/?a|none|null|unknown|no question)\s*$", re.I),
    re.compile(r"^\s*(question|query|instruction)\s*:?\s*$", re.I),
    re.compile(r"^\s*\{+\s*(question|query|instruction)\s*\}+\s*$", re.I),
    re.compile(r"^\s*<\s*(question|query|instruction)\s*>\s*$", re.I),
)


def open_jsonl_handles(output_dir: Path) -> dict[str, Any]:
    handles = {}
    for key, filename in QUEUE_FILES.items():
        path = output_dir / "queues" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        handles[key] = path.open("w", encoding="utf-8", newline="\n")
    return handles


def close_handles(handles: dict[str, Any]) -> None:
    for handle in handles.values():
        handle.close()


def write_jsonl_row(handle, sample: LCQASample) -> None:
    handle.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")


def iter_jsonl_with_errors(path: Path) -> Iterator[tuple[int, dict[str, Any] | None, str | None]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield line_no, None, f"json_decode_error:{exc.msg}"


def stable_hash(*parts: str) -> str:
    payload = "\n".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def query_text(sample: LCQASample) -> str:
    parts = []
    if sample.input.instruction:
        parts.append(sample.input.instruction)
    parts.append(sample.input.question or "")
    if sample.input.choices:
        parts.extend(f"{choice.label}. {choice.text}" for choice in sample.input.choices)
    return "\n\n".join(part for part in parts if part)


def looks_garbled(text: str) -> bool:
    if not text:
        return False
    replacement = text.count("\ufffd")
    control = sum(1 for ch in text if ord(ch) < 32 and ch not in "\t\n\r")
    printable = max(len(text), 1)
    return (replacement + control) / printable > 0.05


def query_issues(sample: LCQASample) -> list[str]:
    query = query_text(sample).strip()
    if not query:
        return ["missing_query"]
    issues = []
    if len(query) < 8:
        issues.append("query_too_short")
    if any(pattern.match(query) for pattern in BAD_QUERY_PATTERNS):
        issues.append("bad_template_query")
    if looks_garbled(query):
        issues.append("garbled_query")
    if len(set(query)) <= 3 and len(query) >= 8:
        issues.append("low_character_diversity_query")
    return issues


def answer_issues(sample: LCQASample) -> list[str]:
    answer = (sample.output.answer or "").strip()
    references = [item for item in sample.output.reference_answers if str(item).strip()]
    has_answer = bool(answer or references or sample.output.label)
    issues = []
    sample.quality.metadata["has_reference_answer"] = has_answer
    if not has_answer:
        return issues

    if looks_garbled(answer):
        issues.append("garbled_answer")
    if sample.input.choices:
        labels = {choice.label for choice in sample.input.choices}
        answer_label = (sample.output.label or answer).strip()
        if answer_label and answer_label in labels:
            return issues
        if sample.output.answer_type == "multiple_choice":
            issues.append("multiple_choice_label_mismatch")
    return issues


def source_key(sample: LCQASample) -> str:
    return "|".join(
        str(value or "")
        for value in (
            sample.source.dataset,
            sample.source.subset,
            sample.source.split,
        )
    )


def source_doc_group(sample: LCQASample) -> str:
    doc_id = sample.source.doc_id or sample.source.original_id or ""
    context_hash = stable_hash(compact_text(sample.document.context)[:10_000])[:16]
    return "|".join(
        str(value or "")
        for value in (
            sample.source.dataset,
            sample.source.subset,
            doc_id or context_hash,
        )
    )


def ensure_bucket(sample: LCQASample) -> str:
    if sample.length.input_tokens is None:
        approx_tokens = len((sample.document.context + "\n" + query_text(sample)).split())
        sample.length.input_tokens = approx_tokens
        sample.length.length_bucket = get_length_bucket(approx_tokens)
        sample.quality.metadata["length_note"] = "approximated_by_whitespace_split"
    if not sample.length.length_bucket:
        sample.length.length_bucket = get_length_bucket(sample.length.input_tokens or 0)
    return sample.length.length_bucket


def set_cleaning_metadata(
    sample: LCQASample,
    *,
    status: str,
    reasons: list[str],
    hashes: dict[str, str],
) -> None:
    sample.quality.metadata["cleaning_status"] = status
    sample.quality.metadata["cleaning_reasons"] = reasons
    sample.quality.metadata["dedup_hashes"] = hashes
    sample.quality.metadata["license_status"] = (
        "license_unknown" if not sample.source.license else "license_present"
    )


def classify_sample(
    sample: LCQASample,
    *,
    seen_exact_global: set[str],
    seen_exact_by_source: dict[str, set[str]],
    group_counts: Counter,
    max_per_source_doc_group: int,
) -> tuple[str, list[str], dict[str, str]]:
    reasons: list[str] = []
    context = sample.document.context or ""
    query = query_text(sample)
    bucket = ensure_bucket(sample)

    context_hash = stable_hash(compact_text(context))
    exact_hash = stable_hash(compact_text(context), compact_text(query))
    hashes = {"context_hash": context_hash, "context_query_hash": exact_hash}

    if not context.strip():
        reasons.append("empty_context")
        return "drop_or_review", reasons, hashes

    if not is_main_long_context_bucket(bucket):
        reasons.append(f"out_of_target_length:{bucket}")
        return "drop_or_review", reasons, hashes

    q_issues = query_issues(sample)
    reasons.extend(q_issues)
    if "missing_query" in q_issues:
        return "needs_query_generation", reasons, hashes
    if any(issue in q_issues for issue in ("bad_template_query", "garbled_query")):
        return "drop_or_review", reasons, hashes

    by_source = seen_exact_by_source.setdefault(source_key(sample), set())
    if exact_hash in by_source:
        reasons.append("duplicate_within_source")
        return "drop_or_review", reasons, hashes
    if exact_hash in seen_exact_global:
        reasons.append("duplicate_across_sources")
        return "drop_or_review", reasons, hashes

    group = source_doc_group(sample)
    group_counts[group] += 1
    hashes["source_doc_group"] = group
    if max_per_source_doc_group > 0 and group_counts[group] > max_per_source_doc_group:
        reasons.append("source_doc_group_downsampled")
        return "drop_or_review", reasons, hashes

    seen_exact_global.add(exact_hash)
    by_source.add(exact_hash)

    ans_issues = answer_issues(sample)
    reasons.extend(ans_issues)
    if ans_issues:
        return "needs_answer_relabel", reasons, hashes

    if sample.quality.training_eligible is False:
        reasons.append(sample.quality.training_exclusion_reason or "training_ineligible")
        return "benchmark_or_eval_only", reasons, hashes
    if sample.quality.contamination_risk == "high":
        reasons.append("high_contamination_risk")
        return "benchmark_or_eval_only", reasons, hashes

    if not sample.source.license:
        reasons.append("license_unknown")

    if not sample.quality.metadata.get("has_reference_answer"):
        reasons.append("missing_reference_answer")
        return "needs_teacher_label", reasons, hashes

    return "clean_ready", reasons, hashes


def queue_key_for(sample: LCQASample, status: str) -> str:
    if status in {"benchmark_or_eval_only", "needs_query_generation", "drop_or_review", "clean_ready"}:
        return status
    if status == "needs_answer_relabel":
        return "needs_answer_relabel"
    bucket = sample.length.length_bucket or ""
    if bucket in {"32K-64K", "64K-128K", "128K-256K"}:
        return "teacher_current"
    return "teacher_long"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown_report(summary: dict[str, Any], source_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# LCQA Cleaning Report",
        "",
        "## Overview",
        "",
        f"- Input files: {summary['input_files']}",
        f"- Parsed samples: {summary['parsed_samples']}",
        f"- Parse/validation errors: {summary['invalid_rows']}",
        f"- Final samples: {summary['final_samples']}",
        f"- Training eligible: {summary['training_eligible']}",
        f"- Training ineligible: {summary['training_ineligible']}",
        f"- Needs teacher label: {summary['status_counts'].get('needs_teacher_label', 0)}",
        "",
        "## Status Counts",
        "",
    ]
    for status in CLEANING_STATUSES:
        lines.append(f"- {status}: {summary['status_counts'].get(status, 0)}")
    lines.extend(["", "## Bucket Counts", ""])
    for bucket in BUCKET_ORDER:
        lines.append(f"- {bucket}: {summary['bucket_counts'].get(bucket, 0)}")
    lines.extend(["", "## License Missing / Risk Sources", ""])
    risky = [
        row
        for row in source_rows
        if row.get("license_unknown") or row.get("contamination_high") or row.get("training_ineligible")
    ][:50]
    if not risky:
        lines.append("- None")
    for row in risky:
        lines.append(
            "- {dataset} | subset={subset} | split={split} | kept={samples} | "
            "license_unknown={license_unknown} | contamination_high={contamination_high} | "
            "training_ineligible={training_ineligible}".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean, deduplicate, quality-tag, and queue LCQA samples after normalization/filtering."
    )
    parser.add_argument(
        "--input-root",
        default="data/filtered",
        help="Root containing filtered LCQA JSONL files.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Glob pattern under input root. Can be repeated. Defaults to filtered_32k_900k.jsonl.",
    )
    parser.add_argument("--output-dir", default="data/cleaned")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--recompute-lengths", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--max-per-source-doc-group", type=int, default=20)
    parser.add_argument(
        "--heldout-benchmark",
        action="append",
        default=None,
        help="Held-out benchmark aliases. Defaults to longbench,longbench_v2,infinitebench,loogle.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    patterns = args.pattern or ["filtered_32k_900k.jsonl"]
    inputs: list[Path] = []
    for pattern in patterns:
        inputs.extend(sorted(input_root.glob(f"**/{pattern}")))
    inputs = sorted(set(inputs))
    if not inputs:
        raise SystemExit(f"No input files found under {input_root} for patterns: {patterns}")

    tokenizer = None
    if args.recompute_lengths:
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency: transformers") from exc
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    heldout_aliases = expand_heldout_benchmarks(
        args.heldout_benchmark or ["longbench", "longbench_v2", "infinitebench", "loogle"]
    )

    seen_exact_global: set[str] = set()
    seen_exact_by_source: dict[str, set[str]] = {}
    group_counts: Counter = Counter()
    status_counts: Counter = Counter()
    bucket_counts: Counter = Counter()
    source_stats: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    error_rows: list[dict[str, Any]] = []
    total_rows = 0
    parsed_samples = 0
    final_samples = 0
    training_eligible = 0
    training_ineligible = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = output_dir / "cleaned_all.lcqa.jsonl"
    handles = open_jsonl_handles(output_dir)

    file_iter = inputs
    if args.progress:
        try:
            from tqdm import tqdm

            file_iter = tqdm(inputs, desc="Cleaning LCQA files", unit="file")
        except ModuleNotFoundError:
            print("tqdm is not installed; continuing without a progress bar.", file=sys.stderr)

    try:
        with cleaned_path.open("w", encoding="utf-8", newline="\n") as cleaned_f:
            stop = False
            for input_path in file_iter:
                if stop:
                    break
                for line_no, row, error in iter_jsonl_with_errors(input_path):
                    if args.limit is not None and total_rows >= args.limit:
                        stop = True
                        break
                    total_rows += 1
                    if error or row is None:
                        error_rows.append({"path": str(input_path), "line": line_no, "error": error})
                        continue
                    try:
                        sample = LCQASample.model_validate(row)
                    except Exception as exc:
                        error_rows.append(
                            {
                                "path": str(input_path),
                                "line": line_no,
                                "error": f"validation_error:{type(exc).__name__}: {exc}",
                            }
                        )
                        continue

                    parsed_samples += 1
                    if tokenizer is not None:
                        sample = update_lengths(sample, args.tokenizer, tokenizer)
                    sample = mark_training_eligibility(sample, heldout_aliases)
                    status, reasons, hashes = classify_sample(
                        sample,
                        seen_exact_global=seen_exact_global,
                        seen_exact_by_source=seen_exact_by_source,
                        group_counts=group_counts,
                        max_per_source_doc_group=args.max_per_source_doc_group,
                    )
                    set_cleaning_metadata(sample, status=status, reasons=reasons, hashes=hashes)

                    status_counts[status] += 1
                    bucket_counts[sample.length.length_bucket or "unknown"] += 1
                    key = (
                        sample.source.dataset or "",
                        sample.source.subset or "",
                        sample.source.split or "",
                    )
                    stats = source_stats[key]
                    stats["samples"] += 1
                    stats[status] += 1
                    stats[sample.length.length_bucket or "unknown"] += 1
                    if not sample.source.license:
                        stats["license_unknown"] += 1
                    if sample.quality.contamination_risk == "high":
                        stats["contamination_high"] += 1
                    if sample.quality.training_eligible is True:
                        training_eligible += 1
                        stats["training_eligible"] += 1
                    elif sample.quality.training_eligible is False:
                        training_ineligible += 1
                        stats["training_ineligible"] += 1

                    write_jsonl_row(cleaned_f, sample)
                    write_jsonl_row(handles[queue_key_for(sample, status)], sample)
                    final_samples += 1
    finally:
        close_handles(handles)

    source_rows: list[dict[str, Any]] = []
    for (dataset, subset, split), stats in sorted(source_stats.items()):
        row = {
            "dataset": dataset,
            "subset": subset,
            "split": split,
            "samples": stats["samples"],
            "training_eligible": stats["training_eligible"],
            "training_ineligible": stats["training_ineligible"],
            "license_unknown": stats["license_unknown"],
            "contamination_high": stats["contamination_high"],
            **{status: stats[status] for status in CLEANING_STATUSES},
            **{bucket: stats[bucket] for bucket in BUCKET_ORDER},
        }
        source_rows.append(row)

    summary = {
        "input_root": str(input_root),
        "patterns": patterns,
        "input_files": len(inputs),
        "output_dir": str(output_dir),
        "cleaned_all": str(cleaned_path),
        "parsed_samples": parsed_samples,
        "invalid_rows": len(error_rows),
        "final_samples": final_samples,
        "training_eligible": training_eligible,
        "training_ineligible": training_ineligible,
        "heldout_benchmark_aliases": sorted(heldout_aliases),
        "status_counts": {status: status_counts.get(status, 0) for status in CLEANING_STATUSES},
        "bucket_counts": {bucket: bucket_counts.get(bucket, 0) for bucket in BUCKET_ORDER},
        "main_long_context_total": sum(
            count for bucket, count in bucket_counts.items() if is_main_long_context_bucket(bucket)
        ),
        "queues": {key: str(output_dir / "queues" / filename) for key, filename in QUEUE_FILES.items()},
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "invalid_rows.json").write_text(
        json.dumps(error_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    source_fieldnames = [
        "dataset",
        "subset",
        "split",
        "samples",
        "training_eligible",
        "training_ineligible",
        "license_unknown",
        "contamination_high",
        *CLEANING_STATUSES,
        *BUCKET_ORDER,
    ]
    write_csv(output_dir / "source_quality_summary.csv", source_rows, source_fieldnames)
    (output_dir / "cleaning_report.md").write_text(
        build_markdown_report(summary, source_rows),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
