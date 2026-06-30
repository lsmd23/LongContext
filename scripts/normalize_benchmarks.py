from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longcontext.io import read_jsonl, write_jsonl
from longcontext.schema import LCQASample


DATASETS = ("zai_longbench", "zai_longbench_v2", "infinitebench")


def maybe_limit(rows: Iterable[dict[str, Any]], limit: int | None) -> Iterable[dict[str, Any]]:
    if limit is None:
        return rows
    return islice(rows, limit)


def stringify_answer(answer: Any) -> tuple[str, list[str]]:
    if answer is None:
        return "", []
    if isinstance(answer, list):
        references = [str(item) for item in answer]
        return "\n".join(references), references
    text = str(answer)
    return text, [text] if text else []


def normalize_choices(options: Any) -> list[dict[str, str]] | None:
    if not options:
        return None
    if isinstance(options, dict):
        return [{"label": str(label), "text": str(text)} for label, text in options.items()]
    if isinstance(options, list):
        choices = []
        for index, item in enumerate(options):
            label = chr(ord("A") + index) if index < 26 else str(index)
            choices.append({"label": label, "text": str(item)})
        return choices
    return None


def task_family(subset: str) -> str:
    if any(name in subset for name in ("sum", "report", "news", "qmsum", "vcsum", "samsum")):
        return "summarization"
    if any(name in subset for name in ("code", "lcc", "repobench")):
        return "code"
    if any(name in subset for name in ("math", "number")):
        return "math"
    if any(name in subset for name in ("retrieval", "passage", "passkey", "kv")):
        return "retrieval"
    return "qa"


def normalize_longbench(row: dict[str, Any], subset: str, split: str) -> dict[str, Any]:
    raw_id = str(row.get("_id") or row.get("id") or "")
    sample_id = f"zai_longbench.{subset}.{raw_id}" if raw_id else f"zai_longbench.{subset}"
    answer_text, reference_answers = stringify_answer(row.get("answers") or row.get("answer"))
    choices = normalize_choices(row.get("all_classes"))
    family = task_family(subset)

    sample = LCQASample(
        id=sample_id,
        source={
            "dataset": "zai-org/LongBench",
            "subset": subset,
            "split": split,
            "original_id": raw_id or None,
            "url": "https://huggingface.co/datasets/zai-org/LongBench",
        },
        task={
            "family": family,
            "type": f"long_context_{family}",
            "subtype": str(row.get("dataset") or subset),
        },
        document={
            "context": str(row.get("context") or ""),
            "context_type": "document",
            "language": row.get("language"),
            "metadata": {"source_length": row.get("length")},
        },
        input={"question": str(row.get("input") or ""), "choices": choices},
        output={
            "answer": answer_text,
            "answer_type": "free_form",
            "label": None,
            "reference_answers": reference_answers,
        },
        quality={"status": "raw", "annotator": "original", "verified": False},
        raw=row,
    )
    return sample.model_dump(mode="json")


def normalize_longbench_v2(row: dict[str, Any], subset: str, split: str) -> dict[str, Any]:
    raw_id = str(row.get("_id") or row.get("id") or "")
    choices = normalize_choices([row.get(f"choice_{label}") for label in ("A", "B", "C", "D")])
    answer_text, reference_answers = stringify_answer(row.get("answer"))
    sample = LCQASample(
        id=f"zai_longbench_v2.{subset}.{raw_id}" if raw_id else f"zai_longbench_v2.{subset}",
        source={
            "dataset": "zai-org/LongBench-v2",
            "subset": subset,
            "split": split,
            "original_id": raw_id or None,
            "url": "https://huggingface.co/datasets/zai-org/LongBench-v2",
        },
        task={
            "family": "qa",
            "type": "long_context_qa",
            "subtype": subset,
            "domain": row.get("domain"),
            "sub_domain": row.get("sub_domain"),
            "difficulty": row.get("difficulty"),
        },
        document={
            "context": str(row.get("context") or ""),
            "context_type": "document",
            "metadata": {"source_length": row.get("length")},
        },
        input={"question": str(row.get("question") or ""), "choices": choices},
        output={
            "answer": answer_text,
            "answer_type": "multiple_choice",
            "label": answer_text or None,
            "reference_answers": reference_answers,
        },
        quality={"status": "raw", "annotator": "original", "verified": False},
        raw=row,
    )
    return sample.model_dump(mode="json")


def normalize_infinitebench(row: dict[str, Any], subset: str, split: str) -> dict[str, Any]:
    raw_id = str(row.get("id") or "")
    choices = normalize_choices(row.get("options"))
    answer_text, reference_answers = stringify_answer(row.get("answer"))
    family = task_family(subset)
    sample = LCQASample(
        id=f"infinitebench.{subset}.{raw_id}" if raw_id else f"infinitebench.{subset}",
        source={
            "dataset": "xinrongzhang2022/InfiniteBench",
            "subset": subset,
            "split": split,
            "original_id": raw_id or None,
            "url": "https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench",
        },
        task={"family": family, "type": f"long_context_{family}", "subtype": subset},
        document={"context": str(row.get("context") or ""), "context_type": "document"},
        input={"question": str(row.get("input") or row.get("question") or ""), "choices": choices},
        output={
            "answer": answer_text,
            "answer_type": "multiple_choice" if choices else "free_form",
            "label": None,
            "reference_answers": reference_answers,
        },
        quality={"status": "raw", "annotator": "original", "verified": False},
        raw=row,
    )
    return sample.model_dump(mode="json")


def normalize_row(row: dict[str, Any], dataset: str, subset: str, split: str) -> dict[str, Any]:
    if dataset == "zai_longbench":
        return normalize_longbench(row, subset, split)
    if dataset == "zai_longbench_v2":
        return normalize_longbench_v2(row, subset, split)
    if dataset == "infinitebench":
        return normalize_infinitebench(row, subset, split)
    raise ValueError(f"Unsupported dataset: {dataset}")


def find_subsets(raw_root: Path, dataset: str) -> list[str]:
    dataset_root = raw_root / dataset
    if not dataset_root.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {dataset_root}")
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir())


def normalize_subset(
    raw_root: Path,
    output_root: Path,
    dataset: str,
    subset: str,
    split: str,
    limit: int | None,
) -> Path:
    input_path = raw_root / dataset / subset / f"{split}.jsonl"
    output_path = output_root / dataset / subset / f"{split}.lcqa.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Raw file not found: {input_path}")

    raw_rows = maybe_limit(read_jsonl(input_path), limit)
    rows = (normalize_row(row, dataset=dataset, subset=subset, split=split) for row in raw_rows)
    write_jsonl(output_path, rows)
    return output_path


def parse_csv(value: str | None) -> list[str] | None:
    if not value or value == "all":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Normalize LongBench/InfiniteBench raw JSONL files to LCQA.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-root", default="data/normalized")
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--subset", default="all", help="Subset/task name or comma-separated names.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="Only normalize the first N rows per subset.")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)

    for dataset in datasets:
        subsets = parse_csv(args.subset) or find_subsets(raw_root, dataset)
        for subset in subsets:
            output_path = normalize_subset(
                raw_root=raw_root,
                output_root=output_root,
                dataset=dataset,
                subset=subset,
                split=args.split,
                limit=args.limit,
            )
            print(f"Wrote normalized LCQA data to {output_path}")


if __name__ == "__main__":
    main()
