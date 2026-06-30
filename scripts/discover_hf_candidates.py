from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
from typing import Any

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longcontext.benchmark_guard import expand_heldout_benchmarks, mark_training_eligibility
from longcontext.discovery import decision_to_json, make_lcqa_sample, sample_preview


DEFAULT_SEARCH_TERMS = (
    "long context",
    "long document",
    "document QA",
    "multi document QA",
    "text2text",
    "summarization",
    "instruction",
    "reading comprehension",
    "book",
    "paper",
    "code",
)

DEFAULT_SPLIT_PRIORITY = ("train", "validation", "dev", "test")


def write_jsonl_row(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def safe_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "__" for ch in value)


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def discover_dataset_names(api: HfApi, search_terms: list[str], max_datasets_per_term: int) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for term in search_terms:
        for info in islice(api.list_datasets(search=term), max_datasets_per_term):
            if info.id in seen:
                continue
            seen.add(info.id)
            names.append(info.id)
    return names


def safe_dataset_info(api: HfApi, dataset_name: str):
    try:
        return api.dataset_info(dataset_name)
    except Exception:
        return None


def dataset_tags_and_license(info) -> tuple[list[str], str | None]:
    if info is None:
        return [], None
    tags = list(getattr(info, "tags", None) or [])
    card_data = getattr(info, "card_data", None)
    license_name = None
    if isinstance(card_data, dict):
        value = card_data.get("license")
        if isinstance(value, list):
            license_name = ",".join(str(item) for item in value)
        elif value:
            license_name = str(value)
    return tags, license_name


def safe_configs(dataset_name: str, max_configs: int) -> list[str | None]:
    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception:
        return [None]
    if not configs:
        return [None]
    return configs[:max_configs]


def safe_splits(dataset_name: str, config: str | None) -> list[str]:
    try:
        splits = get_dataset_split_names(dataset_name, config) if config else get_dataset_split_names(dataset_name)
    except Exception:
        return ["train"]
    return list(splits) or ["train"]


def choose_split(splits: list[str], requested_split: str | None) -> str:
    if requested_split and requested_split in splits:
        return requested_split
    for split in DEFAULT_SPLIT_PRIORITY:
        if split in splits:
            return split
    return splits[0]


def iter_streaming_rows(dataset_name: str, config: str | None, split: str, limit: int):
    kwargs = {"split": split, "streaming": True, "trust_remote_code": False}
    if config:
        dataset = load_dataset(dataset_name, config, **kwargs)
    else:
        dataset = load_dataset(dataset_name, **kwargs)
    return islice(dataset, limit)


def normalized_candidate_path(root: Path, dataset_name: str, subset: str, split: str) -> Path:
    dataset_dir = safe_path_component(dataset_name)
    subset_dir = safe_path_component(subset)
    split_name = safe_path_component(split)
    return root / dataset_dir / subset_dir / f"{split_name}.lcqa.jsonl"


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Discover broad Hugging Face long-context candidates and write LCQA samples."
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated HF dataset ids. If omitted, search terms are used.",
    )
    parser.add_argument(
        "--search-terms",
        default=",".join(DEFAULT_SEARCH_TERMS),
        help="Comma-separated HF search terms used when --datasets is omitted.",
    )
    parser.add_argument("--max-datasets-per-term", type=int, default=5)
    parser.add_argument("--max-configs-per-dataset", type=int, default=2)
    parser.add_argument("--sample-rows", type=int, default=20)
    parser.add_argument("--split", default=None)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--min-tokens", type=int, default=32_768)
    parser.add_argument("--max-tokens", type=int, default=900_000)
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
        "--output-root",
        default=os.getenv("LONGCONTEXT_DATA_ROOT", "data"),
        help="Base data directory. Discovery files are written under <output-root>/discovery/hf_candidates.",
    )
    parser.add_argument(
        "--normalized-output-root",
        default=None,
        help=(
            "Normalized-compatible LCQA output root. Defaults to <output-root>/normalized, "
            "matching the rest of the pipeline."
        ),
    )
    parser.add_argument(
        "--no-normalized-output",
        action="store_true",
        help="Do not write per-source normalized-compatible LCQA files.",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: transformers") from exc

    api = HfApi()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    heldout_aliases = expand_heldout_benchmarks(args.heldout_benchmark or ["longbench_v2"])
    dataset_names = parse_csv(args.datasets) or discover_dataset_names(
        api,
        parse_csv(args.search_terms),
        args.max_datasets_per_term,
    )

    output_root = Path(args.output_root) / "discovery" / "hf_candidates"
    output_root.mkdir(parents=True, exist_ok=True)
    normalized_output_root = (
        Path(args.normalized_output_root)
        if args.normalized_output_root
        else Path(args.output_root) / "normalized"
    )
    candidates_path = output_root / "candidates_32k_900k.lcqa.jsonl"
    all_samples_path = output_root / "samples_with_tokens.lcqa.jsonl"
    decisions_path = output_root / "discovery_decisions.jsonl"
    sources_manifest_path = output_root / "candidate_sources.jsonl"
    summary_path = output_root / "summary.json"

    counters = Counter()
    dataset_counters: dict[str, Counter] = {}
    normalized_handles = {}
    source_manifest_keys = set()

    with ExitStack() as stack:
        candidates_f = stack.enter_context(candidates_path.open("w", encoding="utf-8", newline="\n"))
        all_samples_f = stack.enter_context(all_samples_path.open("w", encoding="utf-8", newline="\n"))
        decisions_f = stack.enter_context(decisions_path.open("w", encoding="utf-8", newline="\n"))
        sources_manifest_f = stack.enter_context(sources_manifest_path.open("w", encoding="utf-8", newline="\n"))

        for dataset_name in dataset_names:
            info = safe_dataset_info(api, dataset_name)
            tags, license_name = dataset_tags_and_license(info)
            dataset_counter = dataset_counters.setdefault(dataset_name, Counter())
            configs = safe_configs(dataset_name, args.max_configs_per_dataset)
            for config in configs:
                splits = safe_splits(dataset_name, config)
                split = choose_split(splits, args.split)
                subset = config or "default"
                try:
                    rows = iter_streaming_rows(dataset_name, config, split, args.sample_rows)
                    for row_index, row in enumerate(rows):
                        sample, decision = make_lcqa_sample(
                            dict(row),
                            dataset_name=dataset_name,
                            subset=subset,
                            split=split,
                            row_index=row_index,
                            tokenizer_name=args.tokenizer,
                            tokenizer=tokenizer,
                            tags=tags,
                            license_name=license_name,
                        )
                        decision_row = decision_to_json(decision, dataset_name, subset, row_index)
                        if decision.status != "candidate":
                            decision_row["row_preview"] = sample_preview(dict(row), max_chars=1_000)
                        write_jsonl_row(decisions_f, decision_row)
                        counters[decision.status] += 1
                        counters[decision.candidate_type] += 1
                        dataset_counter[decision.status] += 1
                        dataset_counter[decision.candidate_type] += 1

                        if not sample:
                            continue
                        sample = mark_training_eligibility(sample, heldout_aliases)
                        write_jsonl_row(all_samples_f, sample.model_dump(mode="json"))
                        input_tokens = sample.length.input_tokens or 0
                        if args.min_tokens <= input_tokens <= args.max_tokens:
                            sample.quality.status = "filtered"
                            write_jsonl_row(candidates_f, sample.model_dump(mode="json"))
                            counters["kept_32k_900k"] += 1
                            dataset_counter["kept_32k_900k"] += 1
                            if sample.quality.training_eligible:
                                counters["training_eligible_kept"] += 1
                                dataset_counter["training_eligible_kept"] += 1
                            else:
                                counters["training_ineligible_kept"] += 1
                                dataset_counter["training_ineligible_kept"] += 1
                            manifest_key = (dataset_name, subset, split)
                            normalized_path = normalized_candidate_path(
                                normalized_output_root, dataset_name, subset, split
                            )
                            if not args.no_normalized_output:
                                handle = normalized_handles.get(normalized_path)
                                if handle is None:
                                    normalized_path.parent.mkdir(parents=True, exist_ok=True)
                                    handle = stack.enter_context(
                                        normalized_path.open("w", encoding="utf-8", newline="\n")
                                    )
                                    normalized_handles[normalized_path] = handle
                                write_jsonl_row(handle, sample.model_dump(mode="json"))

                            if manifest_key not in source_manifest_keys:
                                source_manifest_keys.add(manifest_key)
                                write_jsonl_row(
                                    sources_manifest_f,
                                    {
                                        "dataset": dataset_name,
                                        "subset": subset,
                                        "split": split,
                                        "config": None if subset == "default" else subset,
                                        "license": license_name,
                                        "tags": tags,
                                        "normalized_candidates_path": str(normalized_path),
                                        "candidate_type": sample.task.subtype,
                                        "contamination_risk": sample.quality.contamination_risk,
                                        "training_eligible": sample.quality.training_eligible,
                                        "training_exclusion_reason": sample.quality.training_exclusion_reason,
                                    },
                                )
                except Exception as exc:
                    error_row = {
                        "dataset": dataset_name,
                        "subset": subset,
                        "split": split,
                        "status": "load_failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                    write_jsonl_row(decisions_f, error_row)
                    counters["load_failed"] += 1
                    dataset_counter["load_failed"] += 1

    summary = {
        "datasets_scanned": len(dataset_names),
        "tokenizer": args.tokenizer,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "heldout_benchmark_aliases": sorted(heldout_aliases),
        "outputs": {
            "candidates": str(candidates_path),
            "samples_with_tokens": str(all_samples_path),
            "decisions": str(decisions_path),
            "candidate_sources": str(sources_manifest_path),
            "normalized_output_root": None
            if args.no_normalized_output
            else str(normalized_output_root),
        },
        "counts": dict(counters),
        "dataset_counts": {name: dict(counter) for name, counter in dataset_counters.items()},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
