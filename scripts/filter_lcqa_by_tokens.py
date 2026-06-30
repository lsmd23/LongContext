from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longcontext.io import read_jsonl
from longcontext.benchmark_guard import expand_heldout_benchmarks, mark_training_eligibility
from longcontext.length import BUCKET_ORDER, get_length_bucket, is_main_long_context_bucket
from longcontext.schema import LCQASample


def build_question_text(sample: LCQASample) -> str:
    parts = []
    if sample.input.instruction:
        parts.append(sample.input.instruction)
    parts.append(sample.input.question)
    if sample.input.choices:
        parts.extend(f"{choice.label}. {choice.text}" for choice in sample.input.choices)
    return "\n\n".join(part for part in parts if part)


def build_input_text(sample: LCQASample) -> str:
    return "\n\n".join(part for part in (sample.document.context, build_question_text(sample)) if part)


def build_text(sample: LCQASample, token_field: str) -> str:
    if token_field == "question_text":
        return build_question_text(sample)
    if token_field == "context":
        return sample.document.context
    if token_field == "answer":
        return sample.output.answer
    if token_field in {"input", "full_question", "prompt"}:
        return build_input_text(sample)
    raise ValueError(f"Unsupported token field: {token_field}")


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def update_lengths(
    sample: LCQASample,
    tokenizer_name: str,
    tokenizer,
) -> LCQASample:
    sample.length.tokenizer = tokenizer_name
    sample.length.context_tokens = count_tokens(tokenizer, sample.document.context)
    sample.length.question_tokens = count_tokens(tokenizer, build_question_text(sample))
    sample.length.input_tokens = sample.length.context_tokens + sample.length.question_tokens
    sample.length.prompt_tokens = count_tokens(tokenizer, build_input_text(sample))
    sample.length.answer_tokens = count_tokens(tokenizer, sample.output.answer)
    sample.length.length_bucket = get_length_bucket(sample.length.input_tokens)
    return sample


def update_running_stats(stats: dict[str, dict], sample: LCQASample) -> None:
    bucket = sample.length.length_bucket or "unknown"
    bucket_stats = stats.setdefault(
        bucket,
        {
            "sample_count": 0,
            "input_sum": 0,
            "min_input_tokens": None,
            "max_input_tokens": None,
            "sources": Counter(),
            "domains": Counter(),
        },
    )
    input_tokens = sample.length.input_tokens or 0
    bucket_stats["sample_count"] += 1
    bucket_stats["input_sum"] += input_tokens
    bucket_stats["min_input_tokens"] = (
        input_tokens
        if bucket_stats["min_input_tokens"] is None
        else min(bucket_stats["min_input_tokens"], input_tokens)
    )
    bucket_stats["max_input_tokens"] = (
        input_tokens
        if bucket_stats["max_input_tokens"] is None
        else max(bucket_stats["max_input_tokens"], input_tokens)
    )
    bucket_stats["sources"][sample.source.dataset] += 1
    if sample.task.domain:
        bucket_stats["domains"][sample.task.domain] += 1


def write_bucket_stats_csv(path: str | Path, bucket_stats: dict[str, dict], total: int) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bucket",
        "sample_count",
        "percentage",
        "avg_input_tokens",
        "min_input_tokens",
        "max_input_tokens",
        "main_source_datasets",
        "main_domains",
        "teacher_success_rate",
        "failure_rate",
        "notes",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for bucket in BUCKET_ORDER:
            stats = bucket_stats.get(bucket)
            sample_count = stats["sample_count"] if stats else 0
            writer.writerow(
                {
                    "bucket": bucket,
                    "sample_count": sample_count,
                    "percentage": f"{(sample_count / total * 100):.4f}" if total else "0.0000",
                    "avg_input_tokens": round(stats["input_sum"] / sample_count, 2)
                    if stats and sample_count
                    else "",
                    "min_input_tokens": stats["min_input_tokens"] if stats else "",
                    "max_input_tokens": stats["max_input_tokens"] if stats else "",
                    "main_source_datasets": "; ".join(
                        f"{name}:{count}" for name, count in stats["sources"].most_common(5)
                    )
                    if stats
                    else "",
                    "main_domains": "; ".join(
                        f"{name}:{count}" for name, count in stats["domains"].most_common(5)
                    )
                    if stats
                    else "",
                    "teacher_success_rate": "",
                    "failure_rate": "",
                    "notes": "",
                }
            )


def preview_text(text: str, limit: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text[:limit]


def write_review_samples(path: str | Path, samples_by_bucket: dict[str, list[LCQASample]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Bucket Review Samples", ""]
    for bucket in BUCKET_ORDER:
        samples = samples_by_bucket.get(bucket, [])
        if not samples:
            continue
        lines.extend([f"## {bucket}", ""])
        for sample in samples:
            context = sample.document.context or ""
            answer = sample.output.label or sample.output.answer
            lines.extend(
                [
                    f"### {sample.id}",
                    "",
                    f"- dataset_name: {sample.source.dataset}",
                    f"- subset: {sample.source.subset or ''}",
                    f"- length_bucket: {sample.length.length_bucket}",
                    f"- input_tokens: {sample.length.input_tokens}",
                    f"- question: {build_question_text(sample)}",
                    f"- answer preview: {preview_text(answer, 500)}",
                    "",
                    "**context head**",
                    "",
                    "```text",
                    preview_text(context, 1000),
                    "```",
                    "",
                    "**context tail**",
                    "",
                    "```text",
                    context[-1000:] if len(context) > 1000 else context,
                    "```",
                    "",
                ]
            )
    output.write_text("\n".join(lines), encoding="utf-8")


def maybe_keep_review_sample(
    samples_by_bucket: dict[str, list[LCQASample]],
    sample: LCQASample,
    samples_per_bucket: int,
    rng: random.Random,
    seen_by_bucket: Counter,
) -> None:
    if samples_per_bucket <= 0:
        return
    bucket = sample.length.length_bucket
    if not bucket:
        return
    seen_by_bucket[bucket] += 1
    bucket_samples = samples_by_bucket.setdefault(bucket, [])
    if len(bucket_samples) < samples_per_bucket:
        bucket_samples.append(sample.model_copy(deep=True))
        return
    replace_index = rng.randrange(seen_by_bucket[bucket])
    if replace_index < samples_per_bucket:
        bucket_samples[replace_index] = sample.model_copy(deep=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize, bucket, and filter LCQA JSONL by input length.")
    parser.add_argument("input")
    parser.add_argument("output", help="Filtered 32K-900K LCQA JSONL output path.")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--token-field",
        choices=("input", "full_question", "question_text", "context", "prompt", "answer"),
        default="input",
        help=(
            "Deprecated compatibility option. Filtering now follows length.input_tokens "
            "from context + question + choices."
        ),
    )
    parser.add_argument("--min-tokens", type=int, default=32_768)
    parser.add_argument("--max-tokens", type=int, default=900_000)
    parser.add_argument("--stats-output", default=None)
    parser.add_argument(
        "--tokens-output",
        default=None,
        help="JSONL with every sample plus token fields and length_bucket. Defaults to samples_with_tokens.jsonl beside output.",
    )
    parser.add_argument(
        "--bucket-stats-output",
        default=None,
        help="CSV with per-bucket sample counts and input token stats. Defaults to length_bucket_stats.csv beside output.",
    )
    parser.add_argument(
        "--review-output",
        default=None,
        help="Markdown file with stratified samples for manual review. Defaults to bucket_review_samples.md beside output.",
    )
    parser.add_argument("--review-samples-per-bucket", type=int, default=5)
    parser.add_argument("--review-seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N rows.")
    parser.add_argument("--progress", action="store_true", help="Show a progress bar.")
    parser.add_argument(
        "--heldout-benchmark",
        action="append",
        default=None,
        help=(
            "Benchmark source reserved for evaluation and excluded from training eligibility. "
            "Can be repeated or comma-separated. Defaults to longbench_v2."
        ),
    )
    parser.add_argument(
        "--compute-all-lengths",
        action="store_true",
        help="Deprecated no-op. All length fields are now computed to support bucketed processing.",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: transformers\n"
            "Install project dependencies in the active Python environment:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    heldout_aliases = expand_heldout_benchmarks(args.heldout_benchmark or ["longbench_v2"])

    total = 0
    kept = 0
    training_eligible = 0
    training_ineligible = 0
    min_seen = None
    max_seen = None
    token_sum = 0
    bucket_stats: dict[str, dict] = {}
    kept_bucket_counts = Counter()
    review_samples: dict[str, list[LCQASample]] = {}
    review_seen = Counter()
    rng = random.Random(args.review_seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_path.parent
    tokens_output_path = Path(args.tokens_output) if args.tokens_output else output_dir / "samples_with_tokens.jsonl"
    bucket_stats_output = (
        Path(args.bucket_stats_output) if args.bucket_stats_output else output_dir / "length_bucket_stats.csv"
    )
    review_output = Path(args.review_output) if args.review_output else output_dir / "bucket_review_samples.md"
    tokens_file = None

    row_iter = read_jsonl(args.input)
    if args.progress:
        try:
            from tqdm import tqdm

            row_iter = tqdm(row_iter, desc=f"Filtering {Path(args.input).parent.name}", unit="row")
        except ModuleNotFoundError:
            print("tqdm is not installed; continuing without a progress bar.", file=sys.stderr)

    try:
        if tokens_output_path:
            tokens_output_path.parent.mkdir(parents=True, exist_ok=True)
            tokens_file = tokens_output_path.open("w", encoding="utf-8", newline="\n")

        with output_path.open("w", encoding="utf-8", newline="\n") as f:
            for row in row_iter:
                if args.limit is not None and total >= args.limit:
                    break
                total += 1
                sample = LCQASample.model_validate(row)
                sample = update_lengths(sample, args.tokenizer, tokenizer)
                sample = mark_training_eligibility(sample, heldout_aliases)
                token_count = sample.length.input_tokens or 0

                min_seen = token_count if min_seen is None else min(min_seen, token_count)
                max_seen = token_count if max_seen is None else max(max_seen, token_count)
                token_sum += token_count
                update_running_stats(bucket_stats, sample)

                if tokens_file:
                    tokens_file.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")

                if args.min_tokens <= token_count <= args.max_tokens:
                    sample.quality.status = "filtered"
                    f.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")
                    kept += 1
                    if sample.quality.training_eligible:
                        training_eligible += 1
                    else:
                        training_ineligible += 1
                    if sample.length.length_bucket:
                        kept_bucket_counts[sample.length.length_bucket] += 1
                    maybe_keep_review_sample(
                        review_samples,
                        sample,
                        args.review_samples_per_bucket,
                        rng,
                        review_seen,
                    )
    finally:
        if tokens_file:
            tokens_file.close()

    stats = {
        "input": args.input,
        "output": args.output,
        "tokens_output": str(tokens_output_path),
        "bucket_stats_output": str(bucket_stats_output),
        "review_output": str(review_output),
        "tokenizer": args.tokenizer,
        "token_field": args.token_field,
        "length_field": "input_tokens",
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "total": total,
        "kept": kept,
        "dropped": total - kept,
        "heldout_benchmark_aliases": sorted(heldout_aliases),
        "training_eligible_kept": training_eligible,
        "training_ineligible_kept": training_ineligible,
        "min_seen": min_seen,
        "max_seen": max_seen,
        "avg_seen": token_sum / total if total else None,
        "bucket_counts": {bucket: bucket_stats.get(bucket, {}).get("sample_count", 0) for bucket in BUCKET_ORDER},
        "kept_bucket_counts": {bucket: kept_bucket_counts.get(bucket, 0) for bucket in BUCKET_ORDER},
        "main_long_context_kept": sum(
            count for bucket, count in kept_bucket_counts.items() if is_main_long_context_bucket(bucket)
        ),
    }

    if args.stats_output:
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_bucket_stats_csv(bucket_stats_output, bucket_stats, total)
    write_review_samples(review_output, review_samples)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
