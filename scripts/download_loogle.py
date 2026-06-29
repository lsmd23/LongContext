from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv


DATASET_NAME = "bigai-nlco/LooGLE"
SUBSETS = ("shortdep_qa", "shortdep_cloze", "longdep_qa", "summarization")
DEFAULT_SPLIT = "test"


def download_subset(subset: str, split: str, output_root: Path) -> None:
    dataset = load_dataset(DATASET_NAME, subset, split=split)

    subset_dir = output_root / subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = subset_dir / f"{split}.jsonl"
    dataset.to_json(str(jsonl_path), force_ascii=False, lines=True)

    metadata = {
        "dataset": DATASET_NAME,
        "subset": subset,
        "split": split,
        "num_rows": len(dataset),
        "columns": list(dataset.column_names),
        "jsonl_path": str(jsonl_path),
    }
    metadata_path = subset_dir / f"{split}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(dataset)} rows to {jsonl_path}")
    print(f"Wrote metadata to {metadata_path}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Download raw LooGLE subsets from Hugging Face.")
    parser.add_argument(
        "--subset",
        choices=(*SUBSETS, "all"),
        default="all",
        help="LooGLE subset to download.",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--output-root",
        default=os.getenv("LONGCONTEXT_DATA_ROOT", "data"),
        help="Base data directory. Raw files are written under <output-root>/raw/loogle.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root) / "raw" / "loogle"
    subsets = SUBSETS if args.subset == "all" else (args.subset,)

    for subset in subsets:
        download_subset(subset=subset, split=args.split, output_root=output_root)


if __name__ == "__main__":
    main()
