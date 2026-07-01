from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from longcontext.io import write_jsonl
from label_lcqa_openai import iter_labeled_rows, load_config


def discover_inputs(filtered_root: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in filtered_root.rglob(pattern)
        if path.is_file() and path.suffix == ".jsonl" and path.stat().st_size > 0
    )


def has_training_eligible_true(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("quality") or {}).get("training_eligible") is True:
                return True
    return False


def output_path_for(input_path: Path, filtered_root: Path, labeled_root: Path) -> Path:
    relative = input_path.relative_to(filtered_root)
    return labeled_root / relative


def build_single_file_args(args: argparse.Namespace, input_path: Path, output_path: Path):
    return SimpleNamespace(
        input=str(input_path),
        output=str(output_path),
        config=args.config,
        model=args.model,
        base_url=args.base_url,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        backoff_seconds=args.backoff_seconds,
        limit=args.limit_per_file,
        max_context_chars=args.max_context_chars,
        include_training_ineligible=args.include_training_ineligible,
        require_training_eligible=args.require_training_eligible,
        dry_run=args.dry_run,
        fail_on_error=args.fail_on_error,
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Batch label filtered LCQA JSONL files with an OpenAI-compatible endpoint."
    )
    parser.add_argument("--filtered-root", default="data/filtered")
    parser.add_argument("--labeled-root", default="data/labeled")
    parser.add_argument("--pattern", default="*32k_900k*.lcqa.jsonl")
    parser.add_argument("--config", default="configs/labeling/openai_teacher_example.yaml")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--backoff-seconds", type=float)
    parser.add_argument("--limit-per-file", type=int)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-context-chars", type=int)
    parser.add_argument("--include-training-ineligible", action="store_true")
    parser.add_argument(
        "--require-training-eligible",
        action="store_true",
        help="Only label files and samples explicitly marked quality.training_eligible=true.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    filtered_root = Path(args.filtered_root)
    labeled_root = Path(args.labeled_root)
    if not filtered_root.exists():
        raise SystemExit(f"Filtered root does not exist: {filtered_root}")

    inputs = discover_inputs(filtered_root, args.pattern)
    if args.require_training_eligible:
        inputs = [path for path in inputs if has_training_eligible_true(path)]
    if args.max_files is not None:
        inputs = inputs[: args.max_files]
    if not inputs:
        raise SystemExit(
            f"No non-empty LCQA JSONL files matched {args.pattern!r} under {filtered_root}"
        )

    config = load_config(args.config)
    print(f"Discovered {len(inputs)} non-empty filtered file(s).")
    for index, input_path in enumerate(inputs, start=1):
        output_path = output_path_for(input_path, filtered_root, labeled_root)
        print(f"[{index}/{len(inputs)}] {input_path} -> {output_path}")
        single_args = build_single_file_args(args, input_path, output_path)
        rows = iter_labeled_rows(single_args, config)
        write_jsonl(output_path, rows)

    print(f"Batch labeling complete. Labeled root: {labeled_root}")


if __name__ == "__main__":
    main()
