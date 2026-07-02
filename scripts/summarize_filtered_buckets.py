from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from filter_lcqa_by_tokens import (  # noqa: E402
    maybe_keep_review_sample,
    update_running_stats,
    write_bucket_stats_csv,
    write_review_samples,
)
from longcontext.io import read_jsonl
from longcontext.length import BUCKET_ORDER, get_length_bucket, is_main_long_context_bucket
from longcontext.schema import LCQASample


def ensure_bucket(sample: LCQASample) -> tuple[LCQASample, bool]:
    input_tokens = sample.length.input_tokens
    if input_tokens is None:
        raise ValueError(f"Sample {sample.id} is missing length.input_tokens")
    recomputed = False
    if not sample.length.length_bucket:
        sample.length.length_bucket = get_length_bucket(input_tokens)
        recomputed = True
    return sample, recomputed


def source_name_for(input_path: Path, filtered_root: Path) -> str:
    return str(input_path.parent.relative_to(filtered_root)).replace("\\", "/")


def top_level_source(source: str) -> str:
    return source.split("/", 1)[0]


def init_source_bucket_stats() -> dict[str, object]:
    return {
        "sample_count": 0,
        "input_sum": 0,
        "min_input_tokens": None,
        "max_input_tokens": None,
        "training_eligible": 0,
        "training_ineligible": 0,
        "training_eligible_missing": 0,
    }


def update_source_bucket_stats(stats: dict[str, object], sample: LCQASample) -> None:
    input_tokens = sample.length.input_tokens or 0
    stats["sample_count"] = int(stats["sample_count"]) + 1
    stats["input_sum"] = int(stats["input_sum"]) + input_tokens
    stats["min_input_tokens"] = (
        input_tokens
        if stats["min_input_tokens"] is None
        else min(int(stats["min_input_tokens"]), input_tokens)
    )
    stats["max_input_tokens"] = (
        input_tokens
        if stats["max_input_tokens"] is None
        else max(int(stats["max_input_tokens"]), input_tokens)
    )
    if sample.quality.training_eligible is True:
        stats["training_eligible"] = int(stats["training_eligible"]) + 1
    elif sample.quality.training_eligible is False:
        stats["training_ineligible"] = int(stats["training_ineligible"]) + 1
    else:
        stats["training_eligible_missing"] = int(stats["training_eligible_missing"]) + 1


def write_by_source_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "top_level_source",
        "bucket",
        "sample_count",
        "training_eligible",
        "training_ineligible",
        "training_eligible_missing",
        "avg_input_tokens",
        "min_input_tokens",
        "max_input_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate length_bucket stats from filtered LCQA JSONL files."
    )
    parser.add_argument("--filtered-root", default="data/filtered")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Summary output directory. Defaults to <filtered-root>/summary.",
    )
    parser.add_argument(
        "--pattern",
        default="filtered_32k_900k.jsonl",
        help="Filename pattern for filtered JSONL files to include.",
    )
    parser.add_argument("--review-samples-per-bucket", type=int, default=10)
    parser.add_argument("--review-seed", type=int, default=13)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    filtered_root = Path(args.filtered_root)
    output_dir = Path(args.output_dir) if args.output_dir else filtered_root / "summary"
    inputs = sorted(filtered_root.glob(f"**/{args.pattern}"))
    if not inputs:
        raise SystemExit(f"No files matching {args.pattern!r} under {filtered_root}")

    global_stats: dict[str, dict] = {}
    eligible_stats: dict[str, dict] = {}
    ineligible_stats: dict[str, dict] = {}
    by_source: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    review_samples: dict[str, list[LCQASample]] = {}
    review_seen = Counter()
    rng = random.Random(args.review_seed)

    total = 0
    training_eligible = 0
    training_ineligible = 0
    training_eligible_missing = 0
    files_processed = 0
    missing_bucket = 0

    file_iter: list[Path] = inputs
    if args.progress:
        try:
            from tqdm import tqdm

            file_iter = tqdm(inputs, desc="Summarizing filtered files", unit="file")
        except ModuleNotFoundError:
            print("tqdm is not installed; continuing without a progress bar.", file=sys.stderr)

    for input_path in file_iter:
        files_processed += 1
        source = source_name_for(input_path, filtered_root)
        top_level = top_level_source(source)
        for row in read_jsonl(input_path):
            sample, recomputed = ensure_bucket(LCQASample.model_validate(row))
            if recomputed:
                missing_bucket += 1
            total += 1
            bucket = sample.length.length_bucket or "unknown"

            update_running_stats(global_stats, sample)
            source_buckets = by_source.setdefault((source, top_level), {})
            bucket_stats = source_buckets.setdefault(bucket, init_source_bucket_stats())
            update_source_bucket_stats(bucket_stats, sample)

            if sample.quality.training_eligible is True:
                training_eligible += 1
                update_running_stats(eligible_stats, sample)
            elif sample.quality.training_eligible is False:
                training_ineligible += 1
                update_running_stats(ineligible_stats, sample)
            else:
                training_eligible_missing += 1

            maybe_keep_review_sample(
                review_samples,
                sample,
                args.review_samples_per_bucket,
                rng,
                review_seen,
            )

    by_source_rows: list[dict[str, object]] = []
    for (source, top_level), bucket_map in sorted(by_source.items()):
        for bucket in BUCKET_ORDER:
            stats = bucket_map.get(bucket)
            if not stats or not stats["sample_count"]:
                continue
            count = int(stats["sample_count"])
            by_source_rows.append(
                {
                    "source": source,
                    "top_level_source": top_level,
                    "bucket": bucket,
                    "sample_count": count,
                    "training_eligible": stats["training_eligible"],
                    "training_ineligible": stats["training_ineligible"],
                    "training_eligible_missing": stats["training_eligible_missing"],
                    "avg_input_tokens": round(int(stats["input_sum"]) / count, 2),
                    "min_input_tokens": stats["min_input_tokens"],
                    "max_input_tokens": stats["max_input_tokens"],
                }
            )

    bucket_counts = {bucket: global_stats.get(bucket, {}).get("sample_count", 0) for bucket in BUCKET_ORDER}
    eligible_bucket_counts = {
        bucket: eligible_stats.get(bucket, {}).get("sample_count", 0) for bucket in BUCKET_ORDER
    }
    ineligible_bucket_counts = {
        bucket: ineligible_stats.get(bucket, {}).get("sample_count", 0) for bucket in BUCKET_ORDER
    }

    summary = {
        "filtered_root": str(filtered_root),
        "pattern": args.pattern,
        "files_processed": files_processed,
        "total_samples": total,
        "training_eligible": training_eligible,
        "training_ineligible": training_ineligible,
        "training_eligible_missing": training_eligible_missing,
        "missing_length_bucket_recomputed": missing_bucket,
        "bucket_counts": bucket_counts,
        "training_eligible_bucket_counts": eligible_bucket_counts,
        "training_ineligible_bucket_counts": ineligible_bucket_counts,
        "main_long_context_total": sum(
            count for bucket, count in bucket_counts.items() if is_main_long_context_bucket(bucket)
        ),
        "outputs": {
            "length_bucket_stats": str(output_dir / "length_bucket_stats.csv"),
            "training_eligible_bucket_stats": str(output_dir / "training_eligible_bucket_stats.csv"),
            "length_bucket_by_source": str(output_dir / "length_bucket_by_source.csv"),
            "bucket_review_samples": str(output_dir / "bucket_review_samples.md"),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_bucket_stats_csv(output_dir / "length_bucket_stats.csv", global_stats, total)
    write_bucket_stats_csv(output_dir / "training_eligible_bucket_stats.csv", eligible_stats, training_eligible)
    write_by_source_csv(output_dir / "length_bucket_by_source.csv", by_source_rows)
    write_review_samples(output_dir / "bucket_review_samples.md", review_samples)
    (output_dir / "bucket_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
