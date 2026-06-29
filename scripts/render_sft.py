from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longcontext.io import read_jsonl, write_jsonl
from longcontext.render import render_context_qa
from longcontext.schema import LCQASample


def main() -> None:
    parser = argparse.ArgumentParser(description="Render LCQA JSONL into SFT messages JSONL.")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    rows = (render_context_qa(LCQASample.model_validate(row)) for row in read_jsonl(args.input))
    write_jsonl(args.output, rows)
    print(f"Wrote SFT data to {args.output}")


if __name__ == "__main__":
    main()
