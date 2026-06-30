from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longcontext.io import read_jsonl, write_jsonl
from longcontext.render import render_context_qa
from longcontext.schema import LCQASample


def iter_rendered_rows(input_path: str, include_training_ineligible: bool, stats: dict[str, int]):
    for row in read_jsonl(input_path):
        sample = LCQASample.model_validate(row)
        if sample.quality.training_eligible is False and not include_training_ineligible:
            stats["skipped_training_ineligible"] += 1
            continue
        stats["rendered"] += 1
        yield render_context_qa(sample)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render LCQA JSONL into SFT messages JSONL.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--include-training-ineligible",
        action="store_true",
        help="Render samples even when quality.training_eligible is false.",
    )
    args = parser.parse_args()

    stats = {"rendered": 0, "skipped_training_ineligible": 0}
    rows = iter_rendered_rows(args.input, args.include_training_ineligible, stats)
    write_jsonl(args.output, rows)
    print(
        f"Wrote SFT data to {args.output}; "
        f"rendered={stats['rendered']}, "
        f"skipped_training_ineligible={stats['skipped_training_ineligible']}"
    )


if __name__ == "__main__":
    main()
