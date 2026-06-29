from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longcontext.io import read_jsonl
from longcontext.schema import LCQASample


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LCQA JSONL files.")
    parser.add_argument("path")
    args = parser.parse_args()

    total = 0
    for total, row in enumerate(read_jsonl(args.path), start=1):
        try:
            LCQASample.model_validate(row)
        except ValidationError as exc:
            raise SystemExit(f"Validation failed at row {total}:\n{exc}") from exc

    print(f"OK: {total} LCQA samples validated.")


if __name__ == "__main__":
    main()
