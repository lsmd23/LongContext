from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections.abc import Iterable
from itertools import islice
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longcontext.io import read_jsonl, write_jsonl
from longcontext.schema import LCQASample


DEFAULT_SUBSETS = ("shortdep_qa", "shortdep_cloze", "longdep_qa", "summarization")


def normalize_row(row: dict[str, Any], subset: str, split: str) -> dict[str, Any]:
    raw_id = str(row.get("id") or "")
    doc_id = row.get("doc_id")
    sample_id = f"loogle.{subset}.{raw_id}" if raw_id else f"loogle.{subset}.{doc_id}"

    evidence = row.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]

    answer = row.get("answer")
    if isinstance(answer, list):
        answer_text = "\n".join(str(item) for item in answer)
        reference_answers = [str(item) for item in answer]
    else:
        answer_text = "" if answer is None else str(answer)
        reference_answers = [answer_text] if answer_text else []

    sample = LCQASample(
        id=sample_id,
        source={
            "dataset": "bigai-nlco/LooGLE",
            "subset": subset,
            "split": split,
            "original_id": raw_id or None,
            "doc_id": str(doc_id) if doc_id is not None else None,
            "url": "https://huggingface.co/datasets/bigai-nlco/LooGLE",
        },
        task={
            "family": "qa" if "qa" in subset else subset,
            "type": "long_context_qa" if "qa" in subset else f"long_context_{subset}",
            "subtype": str(row.get("task") or subset),
        },
        document={
            "title": row.get("title"),
            "context": str(row.get("context") or ""),
            "context_type": "document",
        },
        input={
            "question": str(row.get("question") or ""),
            "choices": None,
        },
        output={
            "answer": answer_text,
            "answer_type": "free_form",
            "label": None,
            "reference_answers": reference_answers,
        },
        evidence=[{"text": str(item), "source": "provided"} for item in evidence],
        quality={"status": "raw", "annotator": "original", "verified": False},
        raw=row,
    )
    return sample.model_dump(mode="json")


def maybe_limit(rows: Iterable[dict[str, Any]], limit: int | None) -> Iterable[dict[str, Any]]:
    if limit is None:
        return rows
    return islice(rows, limit)


def normalize_subset(
    raw_root: Path,
    output_root: Path,
    subset: str,
    split: str,
    limit: int | None,
) -> Path:
    input_path = raw_root / subset / f"{split}.jsonl"
    output_path = output_root / subset / f"{split}.lcqa.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"Raw file not found: {input_path}")

    raw_rows = maybe_limit(read_jsonl(input_path), limit)
    rows = (normalize_row(row, subset=subset, split=split) for row in raw_rows)
    write_jsonl(output_path, rows)
    return output_path


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Normalize raw LooGLE JSONL files to LCQA.")
    parser.add_argument("--raw-root", default="data/raw/loogle")
    parser.add_argument("--output-root", default="data/normalized/loogle")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="Only normalize the first N rows.")
    parser.add_argument(
        "--subset",
        choices=(*DEFAULT_SUBSETS, "all"),
        default="all",
        help="LooGLE subset to normalize.",
    )
    args = parser.parse_args()

    subsets = DEFAULT_SUBSETS if args.subset == "all" else (args.subset,)
    for subset in subsets:
        output_path = normalize_subset(
            raw_root=Path(args.raw_root),
            output_root=Path(args.output_root),
            subset=subset,
            split=args.split,
            limit=args.limit,
        )
        print(f"Wrote normalized LCQA data to {output_path}")


if __name__ == "__main__":
    main()
