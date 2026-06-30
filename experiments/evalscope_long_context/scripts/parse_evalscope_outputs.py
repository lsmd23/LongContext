#!/usr/bin/env python3
"""Parse EvalScope raw outputs into unified project result structure."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

ACC_KEYS = ("accuracy", "acc", "score")
COUNT_KEYS = ("total", "count", "num", "samples")


def _load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.resolved.yaml"
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _find_json_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.json"))


def _find_jsonl_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.jsonl"))


def _extract_metric_from_obj(obj: Any) -> dict[str, float | int | None]:
    found: dict[str, float | int | None] = {
        "accuracy": None,
        "total": None,
        "success": None,
        "failed": None,
    }

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_lower = str(key).lower()
                if key_lower in ACC_KEYS and isinstance(value, (int, float)):
                    found["accuracy"] = float(value)
                if key_lower in COUNT_KEYS and isinstance(value, (int, float)):
                    if found["total"] is None:
                        found["total"] = int(value)
                if key_lower in ("success", "correct") and isinstance(value, (int, float)):
                    found["success"] = int(value)
                if key_lower in ("failed", "failures", "error") and isinstance(value, (int, float)):
                    found["failed"] = int(value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return found


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            yield line_no, line


def _read_report_metrics(raw_dir: Path) -> tuple[dict[str, float | int | None], Path | None]:
    reports_dir = raw_dir / "reports"
    for json_path in _find_json_files(reports_dir):
        try:
            data = _read_json(json_path)
        except (json.JSONDecodeError, OSError):
            continue
        metrics = _extract_metric_from_obj(data)
        if metrics["accuracy"] is not None or metrics["total"] is not None:
            return metrics, json_path
    return {"accuracy": None, "total": None, "success": None, "failed": None}, None


def _parse_predictions(raw_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []
    predictions_dir = raw_dir / "predictions"

    for jsonl_path in _find_jsonl_files(predictions_dir):
        try:
            for line_no, line in _iter_jsonl(jsonl_path):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    failed_cases.append(
                        {
                            "sample_id": f"{jsonl_path.name}:{line_no}",
                            "failure_type": "parse_error",
                            "message": str(exc),
                            "raw_output_preview": line[:500],
                        }
                    )
                    continue
                predictions.append(_normalize_prediction(row))
        except OSError as exc:
            failed_cases.append(
                {
                    "sample_id": str(jsonl_path),
                    "failure_type": "parse_error",
                    "message": str(exc),
                    "raw_output_preview": None,
                }
            )

    reviews_dir = raw_dir / "reviews"
    for jsonl_path in _find_jsonl_files(reviews_dir):
        try:
            for line_no, line in _iter_jsonl(jsonl_path):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _is_failed_review(row):
                    failed_cases.append(_normalize_failed_case(row, jsonl_path, line_no))
        except OSError:
            continue

    return predictions, failed_cases


def _normalize_prediction(row: dict[str, Any]) -> dict[str, Any]:
    sample_id = row.get("index", row.get("id", row.get("sample_id")))
    gold = row.get("target", row.get("gold", row.get("answer")))
    prediction_raw = row.get("model_output", row.get("prediction", row.get("response")))
    if isinstance(prediction_raw, dict):
        prediction_raw = prediction_raw.get("content") or json.dumps(prediction_raw, ensure_ascii=False)
    prediction_parsed = _extract_choice_letter(prediction_raw)
    is_correct = None
    if gold is not None and prediction_parsed is not None:
        is_correct = str(gold).strip().upper() == str(prediction_parsed).strip().upper()

    question_preview = None
    for key in ("origin_prompt", "raw_input", "input", "question"):
        value = row.get(key)
        if value:
            question_preview = str(value)[:200]
            break

    return {
        "sample_id": sample_id,
        "question_preview": question_preview,
        "gold": gold,
        "prediction_raw": prediction_raw,
        "prediction_parsed": prediction_parsed,
        "is_correct": is_correct,
        "latency_seconds": row.get("latency", row.get("latency_seconds")),
        "error": row.get("error"),
    }


def _extract_choice_letter(text: Any) -> str | None:
    if text is None:
        return None
    text = str(text).strip()
    match = re.search(r"ANSWER:\s*([A-D])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-D])\b", text[-20:])
    if match:
        return match.group(1).upper()
    if len(text) == 1 and text.upper() in {"A", "B", "C", "D"}:
        return text.upper()
    return None


def _is_failed_review(row: dict[str, Any]) -> bool:
    if row.get("error"):
        return True
    score = row.get("score")
    if score is not None and score in (-9999999, -1):
        return True
    review = row.get("review")
    if isinstance(review, dict) and review.get("error"):
        return True
    return False


def _normalize_failed_case(row: dict[str, Any], path: Path, line_no: int) -> dict[str, Any]:
    failure_type = "unknown"
    message = row.get("error")
    if message:
        text = str(message).lower()
        if "timeout" in text:
            failure_type = "timeout"
        elif "api" in text:
            failure_type = "api_error"
        else:
            failure_type = "parse_error"
    elif row.get("score") is not None:
        failure_type = "invalid_answer"

    preview = row.get("model_output", row.get("prediction"))
    if isinstance(preview, dict):
        preview = json.dumps(preview, ensure_ascii=False)
    if preview is not None:
        preview = str(preview)[:500]

    return {
        "sample_id": row.get("index", row.get("id", f"{path.name}:{line_no}")),
        "failure_type": failure_type,
        "message": message or "Review marked as failed or invalid.",
        "raw_output_preview": preview,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_blocking_issue(run_dir: Path, reason: str, command: str, worked: list[str], next_actions: list[str]) -> None:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Blocking Issue",
        "",
        "## Status",
        "failed",
        "",
        "## Reason",
    ]
    for item in reason.splitlines():
        lines.append(f"- {item.lstrip('- ')}")
    lines.extend(["", "## Last Command", "```bash", command, "```", "", "## What Worked"])
    for item in worked:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Action"])
    for item in next_actions:
        lines.append(f"- {item}")
    lines.append("")
    (reports_dir / "blocking_issue.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(run_dir: Path, metrics: dict[str, Any], config: dict[str, Any]) -> None:
    experiment = config.get("experiment", {})
    benchmark = config.get("benchmark", {})
    model = config.get("model", {})
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Run Summary",
        "",
        f"- **Run ID**: {run_dir.name}",
        f"- **Experiment**: {experiment.get('name')}",
        f"- **Run type**: {experiment.get('run_type')}",
        f"- **Benchmark**: {benchmark.get('name')}",
        f"- **Subset**: {benchmark.get('subset')}",
        f"- **Limit**: {benchmark.get('limit')}",
        f"- **Model**: {model.get('name')}",
        f"- **Metric**: {benchmark.get('metric', 'accuracy')}",
        "",
        "## Metrics",
        "",
        f"- **Parse status**: {metrics.get('parse_status')}",
        f"- **Accuracy**: {metrics.get('accuracy')}",
        f"- **Total**: {metrics.get('total')}",
        f"- **Success**: {metrics.get('success')}",
        f"- **Failed**: {metrics.get('failed')}",
        "",
        "## Held-out Notice",
        "",
        "LongBench-v2 is held-out evaluation data. Do not use its samples for training.",
        "",
    ]
    if metrics.get("parse_status") != "success":
        lines.extend(
            [
                "## Warning",
                "",
                "Output parsing did not fully succeed. See `reports/blocking_issue.md` for details.",
                "",
            ]
        )
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_run(run_dir: Path) -> dict[str, Any]:
    config = _load_config(run_dir)
    benchmark = config.get("benchmark", {})
    model = config.get("model", {})
    raw_dir = run_dir / "evalscope_raw"

    report_metrics, report_path = _read_report_metrics(raw_dir)
    predictions, failed_cases = _parse_predictions(raw_dir)

    if report_path is None and predictions:
        correct = sum(1 for row in predictions if row.get("is_correct") is True)
        report_metrics["success"] = correct
        report_metrics["total"] = len(predictions)
        if report_metrics["accuracy"] is None and predictions:
            report_metrics["accuracy"] = correct / len(predictions) if predictions else None

    if report_metrics["failed"] is None and failed_cases:
        report_metrics["failed"] = len(failed_cases)

    parse_status = "success" if (report_path or predictions or report_metrics["accuracy"] is not None) else "failed"

    metrics = {
        "benchmark": benchmark.get("name"),
        "subset": benchmark.get("subset"),
        "limit": benchmark.get("limit"),
        "model": model.get("name"),
        "metric": benchmark.get("metric", "accuracy"),
        "accuracy": report_metrics["accuracy"],
        "total": report_metrics["total"],
        "success": report_metrics["success"],
        "failed": report_metrics["failed"],
        "parse_status": parse_status,
        "raw_evalscope_dir": "evalscope_raw",
        "source_report": str(report_path.relative_to(run_dir)) if report_path else None,
    }

    parsed_dir = run_dir / "parsed"
    _write_json(parsed_dir / "metrics.json", metrics)
    _write_jsonl(parsed_dir / "predictions.jsonl", predictions)
    _write_jsonl(parsed_dir / "failed_cases.jsonl", failed_cases)

    latency_rows = [
        {"sample_id": row["sample_id"], "latency_seconds": row.get("latency_seconds")}
        for row in predictions
        if row.get("latency_seconds") is not None
    ]
    _write_jsonl(parsed_dir / "latency.jsonl", latency_rows)

    write_summary_md(run_dir, metrics, config)

    if parse_status == "failed":
        command = (run_dir / "command.txt").read_text(encoding="utf-8").strip() if (run_dir / "command.txt").exists() else ""
        write_blocking_issue(
            run_dir,
            reason="Could not locate EvalScope metrics JSON or prediction files in evalscope_raw/.",
            command=command,
            worked=["Run directory exists", "Parser executed"],
            next_actions=[
                "Verify EvalScope completed successfully",
                "Check evalscope_raw/ for reports/ and predictions/",
                "Confirm EvalScope version matches documented output layout",
            ],
        )

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse EvalScope outputs for a run directory.")
    parser.add_argument("--run-dir", required=True, help="Path to a single run directory.")
    args = parser.parse_args()
    parse_run(Path(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
