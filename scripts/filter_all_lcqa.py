from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def output_path_for(input_path: Path, normalized_root: Path, filtered_root: Path) -> Path:
    relative = input_path.relative_to(normalized_root)
    output_dir = filtered_root / relative.parent
    return output_dir / "filtered_32k_900k.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter every LCQA JSONL file under a normalized data root.")
    parser.add_argument("--normalized-root", default="data/normalized")
    parser.add_argument("--filtered-root", default="data/filtered")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--min-tokens", type=int, default=32_768)
    parser.add_argument("--max-tokens", type=int, default=900_000)
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N rows per file.")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    normalized_root = Path(args.normalized_root)
    filtered_root = Path(args.filtered_root)
    inputs = sorted(normalized_root.glob("**/*.lcqa.jsonl"))
    if not inputs:
        raise SystemExit(f"No LCQA files found under {normalized_root}")

    script_path = Path(__file__).with_name("filter_lcqa_by_tokens.py")
    for input_path in inputs:
        output_path = output_path_for(input_path, normalized_root, filtered_root)
        stats_path = output_path.parent / "filter_stats.json"
        cmd = [
            sys.executable,
            str(script_path),
            str(input_path),
            str(output_path),
            "--tokenizer",
            args.tokenizer,
            "--min-tokens",
            str(args.min_tokens),
            "--max-tokens",
            str(args.max_tokens),
            "--stats-output",
            str(stats_path),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.progress:
            cmd.append("--progress")

        print(f"Filtering {input_path} -> {output_path}", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
