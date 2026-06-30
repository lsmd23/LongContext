#!/usr/bin/env python3
"""Summarize multiple EvalScope runs into CSV and Markdown reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

SUMMARY_FIELDS = [
    "run_id",
    "run_type",
    "benchmark",
    "subset",
    "limit",
    "model",
    "accuracy",
    "total",
    "success",
    "failed",
    "evalscope_version",
    "git_commit",
    "started_at",
    "finished_at",
    "notes",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "parsed" / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def collect_run_row(run_dir: Path) -> dict[str, Any] | None:
    if not run_dir.is_dir() or run_dir.name.startswith("."):
        return None
    manifest = _load_yaml(run_dir / "run_manifest.yaml")
    metrics = _load_metrics(run_dir)
    if not manifest and not metrics:
        return None

    timing = manifest.get("timing", {})
    packages = manifest.get("packages", {})
    git_info = manifest.get("git", {})
    experiment = manifest.get("experiment", {})

    notes = ""
    blocking = run_dir / "reports" / "blocking_issue.md"
    if blocking.exists():
        notes = "blocking_issue"

    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "run_type": experiment.get("run_type"),
        "benchmark": metrics.get("benchmark") or manifest.get("benchmark", {}).get("name"),
        "subset": metrics.get("subset") or manifest.get("benchmark", {}).get("subset"),
        "limit": metrics.get("limit") if metrics.get("limit") is not None else manifest.get("benchmark", {}).get("limit"),
        "model": metrics.get("model") or manifest.get("model", {}).get("name"),
        "accuracy": metrics.get("accuracy"),
        "total": metrics.get("total"),
        "success": metrics.get("success"),
        "failed": metrics.get("failed"),
        "evalscope_version": packages.get("evalscope"),
        "git_commit": git_info.get("commit"),
        "started_at": timing.get("started_at"),
        "finished_at": timing.get("finished_at"),
        "notes": notes,
    }


def summarize_runs(runs_root: Path) -> tuple[Path, Path]:
    rows: list[dict[str, Any]] = []
    for child in sorted(runs_root.iterdir()):
        row = collect_run_row(child)
        if row:
            rows.append(row)

    csv_path = runs_root / "summary.csv"
    md_path = runs_root / "summary.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})

    lines = [
        "# EvalScope Run Summary",
        "",
        f"Total runs: {len(rows)}",
        "",
        "| Run ID | Type | Benchmark | Model | Accuracy | Total | Failed | Started |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {run_type} | {benchmark} | {model} | {accuracy} | {total} | {failed} | {started_at} |".format(
                **{key: row.get(key, "") for key in SUMMARY_FIELDS}
            )
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize EvalScope experiment runs.")
    parser.add_argument(
        "--runs-root",
        default="experiments/evalscope_long_context/runs",
        help="Root directory containing individual run folders.",
    )
    args = parser.parse_args()
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = Path.cwd() / runs_root
    runs_root = runs_root.resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    csv_path, md_path = summarize_runs(runs_root)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
