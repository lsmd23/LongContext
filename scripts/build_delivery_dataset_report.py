from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longcontext.io import read_jsonl
from longcontext.length import BUCKET_ORDER
from longcontext.schema import LCQASample


LONG_CONTEXT_BUCKETS = [bucket for bucket in BUCKET_ORDER if bucket not in {"<32K", ">900K"}]


def source_key_for(sample: LCQASample, input_path: Path, filtered_root: Path) -> tuple[str, str, str]:
    dataset = sample.source.dataset or input_path.parent.name
    subset = sample.source.subset or ""
    split = sample.source.split or ""
    return dataset, subset, split


def source_path_for(input_path: Path, filtered_root: Path) -> str:
    return str(input_path.parent.relative_to(filtered_root)).replace("\\", "/")


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidate_sources(path: Path | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    manifest: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path or not path.exists():
        return manifest
    for row in read_jsonl(path):
        key = (
            str(row.get("dataset") or ""),
            str(row.get("subset") or ""),
            str(row.get("split") or ""),
        )
        manifest[key] = row
    return manifest


def find_manifest_match(
    manifest: dict[tuple[str, str, str], dict[str, Any]],
    dataset: str,
    subset: str,
    split: str,
) -> dict[str, Any]:
    for key in (
        (dataset, subset, split),
        (dataset, subset, ""),
        (dataset, "", split),
        (dataset, "", ""),
    ):
        if key in manifest:
            return manifest[key]
    return {}


def truncate_text(text: str, limit: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    half = max(limit // 2, 1)
    return text[:half] + "\n\n[... truncated ...]\n\n" + text[-(limit - half) :]


def sample_preview(sample: LCQASample, max_chars: int) -> dict[str, Any]:
    return {
        "id": sample.id,
        "length_bucket": sample.length.length_bucket,
        "input_tokens": sample.length.input_tokens,
        "task_family": sample.task.family,
        "task_type": sample.task.type,
        "question": truncate_text(sample.input.question, max_chars),
        "context_head_tail": truncate_text(sample.document.context, max_chars),
        "has_reference_answer": bool(sample.output.answer or sample.output.reference_answers),
        "training_eligible": sample.quality.training_eligible,
    }


def load_hf_card(dataset: str) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError:
        return {"hf_card_error": "huggingface_hub is not installed"}

    try:
        info = HfApi().dataset_info(dataset)
    except Exception as exc:
        return {"hf_card_error": f"{type(exc).__name__}: {exc}"}

    card_data = getattr(info, "card_data", None)
    description = ""
    if isinstance(card_data, dict):
        value = card_data.get("description") or card_data.get("pretty_name") or ""
        description = str(value)
    return {
        "hf_url": f"https://huggingface.co/datasets/{dataset}",
        "license": getattr(info, "license", None),
        "tags": list(getattr(info, "tags", None) or []),
        "card_description": description,
    }


def heuristic_quality(row: dict[str, Any]) -> dict[str, Any]:
    total = int(row["kept_samples"])
    with_answer = int(row["with_reference_answer"])
    eligible = int(row["training_eligible"])
    missing_eligible = int(row["training_eligible_missing"])
    long_tail = int(row.get("256K-512K", 0)) + int(row.get("512K-900K", 0))
    answer_ratio = with_answer / total if total else 0
    eligible_ratio = eligible / total if total else 0
    long_tail_ratio = long_tail / total if total else 0

    score = 0.35 * answer_ratio + 0.35 * eligible_ratio + 0.30 * min(long_tail_ratio * 4, 1.0)
    if row.get("contamination_risk") == "high":
        score *= 0.5
    elif row.get("contamination_risk") == "medium":
        score *= 0.8

    if eligible == 0 and missing_eligible:
        recommended_use = "needs_training_eligibility_marking"
    elif eligible == 0:
        recommended_use = "reference_only"
    elif row.get("contamination_risk") == "high":
        recommended_use = "review_before_train"
    elif score >= 0.65:
        recommended_use = "train_candidate"
    else:
        recommended_use = "teacher_label_candidate"

    return {
        "overall_score": round(score, 3),
        "recommended_use": recommended_use,
        "rationale": (
            f"answer_ratio={answer_ratio:.2f}; eligible_ratio={eligible_ratio:.2f}; "
            f"missing_eligible={missing_eligible}; "
            f"long_tail_ratio={long_tail_ratio:.2f}; contamination={row.get('contamination_risk') or 'unknown'}"
        ),
    }


def build_quality_prompt(
    row: dict[str, Any],
    card: dict[str, Any],
    samples: list[dict[str, Any]],
) -> str:
    payload = {
        "source_summary": row,
        "hf_card": card,
        "stratified_samples": samples,
    }
    return (
        "You are reviewing long-context training data sources. "
        "Assess this dataset source for teacher labeling and training suitability. "
        "Return compact JSON with keys: task_type, context_quality, query_clarity, "
        "answer_availability, long_context_dependency, contamination_risk, "
        "teacher_labeling_priority, recommended_use, overall_score, evidence.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def call_openai_quality_model(
    *,
    model: str,
    base_url: str | None,
    prompt: str,
    max_output_tokens: int,
    timeout_seconds: float | None,
) -> str:
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {}
    if base_url:
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized += "/v1"
        client_kwargs["base_url"] = normalized
    client = OpenAI(**client_kwargs)
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
        timeout=timeout_seconds,
    )
    return response.output_text


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], quality_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(int(row["kept_samples"]) for row in rows)
    eligible = sum(int(row["training_eligible"]) for row in rows)
    ineligible = sum(int(row["training_ineligible"]) for row in rows)
    missing_eligible = sum(int(row["training_eligible_missing"]) for row in rows)
    lines = [
        "# Dataset Delivery Report",
        "",
        "## Overview",
        "",
        f"- Sources: {len(rows)}",
        f"- Filtered samples: {total}",
        f"- Training eligible: {eligible}",
        f"- Training ineligible: {ineligible}",
        f"- Missing training eligibility: {missing_eligible}",
        "",
        "## Bucket Distribution By Source",
        "",
        "| dataset | subset | split | kept | eligible | 32K-64K | 64K-128K | 128K-256K | 256K-512K | 512K-900K | recommended_use |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    quality_by_key = {
        (row["dataset"], row["subset"], row["split"]): row for row in quality_rows
    }
    for row in rows:
        quality = quality_by_key.get((row["dataset"], row["subset"], row["split"]), {})
        lines.append(
            "| {dataset} | {subset} | {split} | {kept_samples} | {training_eligible} | "
            "{b32} | {b64} | {b128} | {b256} | {b512} | {recommended_use} |".format(
                dataset=row["dataset"],
                subset=row["subset"],
                split=row["split"],
                kept_samples=row["kept_samples"],
                training_eligible=row["training_eligible"],
                b32=row.get("32K-64K", 0),
                b64=row.get("64K-128K", 0),
                b128=row.get("128K-256K", 0),
                b256=row.get("256K-512K", 0),
                b512=row.get("512K-900K", 0),
                recommended_use=quality.get("recommended_use", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Build delivery-ready source, bucket, and quality reports from filtered LCQA data."
    )
    parser.add_argument("--filtered-root", default="data/filtered")
    parser.add_argument("--output-dir", default="outputs/delivery_report")
    parser.add_argument("--pattern", default="filtered_32k_900k.jsonl")
    parser.add_argument("--candidate-sources", default=None)
    parser.add_argument("--samples-per-bucket", type=int, default=2)
    parser.add_argument("--sample-preview-chars", type=int, default=1200)
    parser.add_argument(
        "--hf-card-mode",
        choices=("none", "manifest", "live"),
        default="manifest",
        help="Use no HF cards, manifest metadata only, or live Hugging Face dataset_info calls.",
    )
    parser.add_argument(
        "--quality-mode",
        choices=("none", "heuristic", "prompt-only", "openai"),
        default="heuristic",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_ENDPOINT"))
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--limit-sources", type=int, default=None)
    args = parser.parse_args()

    filtered_root = Path(args.filtered_root)
    output_dir = Path(args.output_dir)
    candidate_sources_path = (
        Path(args.candidate_sources)
        if args.candidate_sources
        else filtered_root.parent / "discovery" / "hf_candidates" / "candidate_sources.jsonl"
    )
    manifest = load_candidate_sources(candidate_sources_path)
    inputs = sorted(filtered_root.glob(f"**/{args.pattern}"))
    if not inputs:
        raise SystemExit(f"No files matching {args.pattern!r} under {filtered_root}")

    source_stats: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_samples: dict[tuple[str, str, str], dict[str, list[LCQASample]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for input_path in inputs:
        if input_path.stat().st_size == 0:
            continue
        source_path = source_path_for(input_path, filtered_root)
        file_stats = safe_read_json(input_path.parent / "filter_stats.json")
        for row in read_jsonl(input_path):
            sample = LCQASample.model_validate(row)
            dataset, subset, split = source_key_for(sample, input_path, filtered_root)
            key = (dataset, subset, split)
            manifest_row = find_manifest_match(manifest, dataset, subset, split)
            stats = source_stats.setdefault(
                key,
                {
                    "dataset": dataset,
                    "subset": subset,
                    "split": split,
                    "source_path": source_path,
                    "hf_url": manifest_row.get("url")
                    or manifest_row.get("hf_url")
                    or (f"https://huggingface.co/datasets/{dataset}" if "/" in dataset else ""),
                    "license": manifest_row.get("license") or sample.source.license or "",
                    "candidate_type": manifest_row.get("candidate_type")
                    or sample.task.subtype
                    or "",
                    "task_family": sample.task.family,
                    "task_type": sample.task.type,
                    "contamination_risk": sample.quality.contamination_risk or "",
                    "training_exclusion_reason": sample.quality.training_exclusion_reason or "",
                    "raw_total_seen": file_stats.get("total", ""),
                    "raw_kept_seen": file_stats.get("kept", ""),
                    "kept_samples": 0,
                    "training_eligible": 0,
                    "training_ineligible": 0,
                    "training_eligible_missing": 0,
                    "with_reference_answer": 0,
                    **{bucket: 0 for bucket in BUCKET_ORDER},
                },
            )
            stats["kept_samples"] += 1
            bucket = sample.length.length_bucket or "unknown"
            if bucket in BUCKET_ORDER:
                stats[bucket] += 1
            if sample.quality.training_eligible is True:
                stats["training_eligible"] += 1
            elif sample.quality.training_eligible is False:
                stats["training_ineligible"] += 1
            else:
                stats["training_eligible_missing"] += 1
            if sample.output.answer or sample.output.reference_answers:
                stats["with_reference_answer"] += 1
            bucket_samples = source_samples[key][bucket]
            if len(bucket_samples) < args.samples_per_bucket:
                bucket_samples.append(sample.model_copy(deep=True))

    rows = sorted(source_stats.values(), key=lambda item: (item["dataset"], item["subset"], item["split"]))
    if args.limit_sources is not None:
        rows = rows[: args.limit_sources]

    cards: dict[tuple[str, str, str], dict[str, Any]] = {}
    prompts: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for row in rows:
        key = (row["dataset"], row["subset"], row["split"])
        manifest_row = find_manifest_match(manifest, *key)
        if args.hf_card_mode == "live" and "/" in row["dataset"]:
            card = load_hf_card(row["dataset"])
        elif args.hf_card_mode == "manifest":
            card = {
                "hf_url": row["hf_url"],
                "license": row["license"],
                "tags": manifest_row.get("tags", []),
                "card_description": manifest_row.get("description", ""),
            }
        else:
            card = {}
        cards[key] = card
        samples = [
            sample_preview(sample, args.sample_preview_chars)
            for bucket in LONG_CONTEXT_BUCKETS
            for sample in source_samples[key].get(bucket, [])
        ]
        prompt = build_quality_prompt(row, card, samples)
        prompts.append({"dataset": row["dataset"], "subset": row["subset"], "split": row["split"], "prompt": prompt})

        quality: dict[str, Any]
        if args.quality_mode == "none":
            quality = {}
        elif args.quality_mode in {"heuristic", "prompt-only"}:
            quality = heuristic_quality(row)
            if args.quality_mode == "prompt-only":
                quality["model_prompt_path"] = "quality_prompts.jsonl"
        else:
            if not os.getenv("OPENAI_API_KEY"):
                raise SystemExit("OPENAI_API_KEY is required for --quality-mode openai")
            text = call_openai_quality_model(
                model=args.model,
                base_url=args.base_url,
                prompt=prompt,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout_seconds,
            )
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = {"model_raw": text, "parse_error": "not_json"}
            quality = parsed

        quality_rows.append(
            {
                "dataset": row["dataset"],
                "subset": row["subset"],
                "split": row["split"],
                "quality_mode": args.quality_mode,
                **quality,
            }
        )

    source_fieldnames = [
        "dataset",
        "subset",
        "split",
        "source_path",
        "hf_url",
        "license",
        "candidate_type",
        "task_family",
        "task_type",
        "contamination_risk",
        "training_exclusion_reason",
        "raw_total_seen",
        "raw_kept_seen",
        "kept_samples",
        "training_eligible",
        "training_ineligible",
        "training_eligible_missing",
        "with_reference_answer",
        *BUCKET_ORDER,
    ]
    quality_fieldnames = sorted({key for row in quality_rows for key in row.keys()})

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "source_bucket_distribution.csv", rows, source_fieldnames)
    write_csv(output_dir / "source_quality_assessment.csv", quality_rows, quality_fieldnames)
    write_markdown(output_dir / "delivery_report.md", rows, quality_rows)
    (output_dir / "source_bucket_distribution.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "source_quality_assessment.json").write_text(
        json.dumps(quality_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "quality_prompts.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in prompts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "filtered_root": str(filtered_root),
        "candidate_sources": str(candidate_sources_path) if candidate_sources_path.exists() else None,
        "sources": len(rows),
        "samples": sum(int(row["kept_samples"]) for row in rows),
        "training_eligible": sum(int(row["training_eligible"]) for row in rows),
        "training_ineligible": sum(int(row["training_ineligible"]) for row in rows),
        "training_eligible_missing": sum(
            int(row["training_eligible_missing"]) for row in rows
        ),
        "outputs": {
            "source_bucket_distribution": str(output_dir / "source_bucket_distribution.csv"),
            "source_quality_assessment": str(output_dir / "source_quality_assessment.csv"),
            "quality_prompts": str(output_dir / "quality_prompts.jsonl"),
            "delivery_report": str(output_dir / "delivery_report.md"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
