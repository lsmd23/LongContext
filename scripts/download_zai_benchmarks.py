from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download


LONGBENCH_REPO = "zai-org/LongBench"
LONGBENCH_V2_REPO = "zai-org/LongBench-v2"
INFINITEBENCH_REPO = "xinrongzhang2022/InfiniteBench"

LONGBENCH_DATASET = "zai_longbench"
LONGBENCH_V2_DATASET = "zai_longbench_v2"
INFINITEBENCH_DATASET = "infinitebench"

INFINITEBENCH_TASKS = (
    "passkey",
    "kv_retrieval",
    "number_string",
    "code_run",
    "code_debug",
    "math_find",
    "math_calc",
    "longdialogue_qa_eng",
    "longbook_qa_eng",
    "longbook_sum_eng",
    "longbook_choice_eng",
    "longbook_qa_chn",
)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
    return total


def iter_json_or_jsonl(path: Path) -> Iterable[dict]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        yield from data
        return
    if isinstance(data, dict):
        rows = data.get("data") or data.get("rows")
        if isinstance(rows, list):
            yield from rows
            return
    raise ValueError(f"Unsupported JSON structure: {path}")


def write_metadata(path: Path, metadata: dict) -> None:
    metadata_path = path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote metadata to {metadata_path}")


def download_longbench(output_root: Path, split: str, subsets: set[str] | None) -> None:
    zip_path = Path(hf_hub_download(LONGBENCH_REPO, "data.zip", repo_type="dataset"))
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.startswith("data/") and name.endswith(".jsonl")]
        for member in members:
            subset = Path(member).stem
            if subsets and subset not in subsets:
                continue
            output_path = output_root / LONGBENCH_DATASET / subset / f"{split}.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, output_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            num_rows = sum(1 for _ in output_path.open("r", encoding="utf-8"))
            metadata = {
                "dataset": LONGBENCH_REPO,
                "local_dataset": LONGBENCH_DATASET,
                "subset": subset,
                "split": split,
                "num_rows": num_rows,
                "source_file": member,
                "jsonl_path": str(output_path),
            }
            write_metadata(output_path, metadata)
            print(f"Wrote {num_rows} rows to {output_path}")


def download_longbench_v2(output_root: Path, split: str) -> None:
    data_path = Path(hf_hub_download(LONGBENCH_V2_REPO, "data.json", repo_type="dataset"))
    output_path = output_root / LONGBENCH_V2_DATASET / "main" / f"{split}.jsonl"
    num_rows = write_jsonl(output_path, iter_json_or_jsonl(data_path))
    metadata = {
        "dataset": LONGBENCH_V2_REPO,
        "local_dataset": LONGBENCH_V2_DATASET,
        "subset": "main",
        "split": split,
        "hf_split": "train",
        "num_rows": num_rows,
        "source_file": "data.json",
        "jsonl_path": str(output_path),
    }
    write_metadata(output_path, metadata)
    print(f"Wrote {num_rows} rows to {output_path}")


def download_infinitebench(output_root: Path, split: str, tasks: set[str] | None) -> None:
    selected_tasks = INFINITEBENCH_TASKS if tasks is None else tuple(task for task in INFINITEBENCH_TASKS if task in tasks)
    for task in selected_tasks:
        source_path = Path(hf_hub_download(INFINITEBENCH_REPO, f"{task}.jsonl", repo_type="dataset"))
        output_path = output_root / INFINITEBENCH_DATASET / task / f"{split}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, output_path)
        num_rows = sum(1 for _ in output_path.open("r", encoding="utf-8"))
        metadata = {
            "dataset": INFINITEBENCH_REPO,
            "local_dataset": INFINITEBENCH_DATASET,
            "subset": task,
            "split": split,
            "hf_split": "train",
            "num_rows": num_rows,
            "source_file": f"{task}.jsonl",
            "jsonl_path": str(output_path),
        }
        write_metadata(output_path, metadata)
        print(f"Wrote {num_rows} rows to {output_path}")


def parse_csv_set(value: str | None) -> set[str] | None:
    if not value or value == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Download ZAI LongBench and InfiniteBench raw data.")
    parser.add_argument(
        "--dataset",
        choices=("longbench", "longbench_v2", "infinitebench", "all"),
        default="all",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--subsets",
        default="all",
        help="Comma-separated LongBench subsets. Only applies to --dataset longbench/all.",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated InfiniteBench tasks. Only applies to --dataset infinitebench/all.",
    )
    parser.add_argument(
        "--output-root",
        default=os.getenv("LONGCONTEXT_DATA_ROOT", "data"),
        help="Base data directory. Raw files are written under <output-root>/raw.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root) / "raw"
    datasets = ("longbench", "longbench_v2", "infinitebench") if args.dataset == "all" else (args.dataset,)

    if "longbench" in datasets:
        download_longbench(output_root, split=args.split, subsets=parse_csv_set(args.subsets))
    if "longbench_v2" in datasets:
        download_longbench_v2(output_root, split=args.split)
    if "infinitebench" in datasets:
        download_infinitebench(output_root, split=args.split, tasks=parse_csv_set(args.tasks))


if __name__ == "__main__":
    main()
