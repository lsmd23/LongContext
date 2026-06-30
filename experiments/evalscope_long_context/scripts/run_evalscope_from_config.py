#!/usr/bin/env python3
"""Main EvalScope experiment runner driven by YAML config."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = MODULE_ROOT.parents[1]

sys.path.insert(0, str(SCRIPT_DIR))
from collect_run_metadata import HELD_OUT_NOTICE, collect_metadata  # noqa: E402
from parse_evalscope_outputs import parse_run, write_blocking_issue  # noqa: E402


REQUIRED_SECTIONS = ("experiment", "benchmark", "model", "generation", "runtime", "logging")


def repo_root() -> Path:
    return REPO_ROOT


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root() / path


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    for section in REQUIRED_SECTIONS:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    return config


def validate_config(config: dict[str, Any]) -> None:
    model_cfg = config["model"]
    for key in ("name", "api_url_env", "api_key_env"):
        if not model_cfg.get(key):
            raise ValueError(f"model.{key} is required")
    if not config["benchmark"].get("name"):
        raise ValueError("benchmark.name is required")


def make_run_id(config: dict[str, Any], started_at: datetime) -> str:
    benchmark = config["benchmark"]["name"]
    run_type = config["experiment"].get("run_type", "run")
    return started_at.strftime(f"%Y%m%d_%H%M%S_{benchmark}_{run_type}")


def resolve_env(config: dict[str, Any]) -> tuple[str | None, str | None, list[str]]:
    model_cfg = config["model"]
    api_url_env = model_cfg.get("api_url_env", "MODEL_API_URL")
    api_key_env = model_cfg.get("api_key_env", "MODEL_API_KEY")
    errors: list[str] = []

    api_url = os.environ.get(api_url_env)
    api_key = os.environ.get(api_key_env)
    if not api_url:
        errors.append(f"Missing environment variable: {api_url_env}")
    if not api_key:
        errors.append(f"Missing environment variable: {api_key_env}")
    return api_url, api_key, errors


def find_evalscope_executable() -> str | None:
    found = shutil.which("evalscope")
    if found:
        return found
    return None


def redact_command(command: list[str]) -> list[str]:
    redacted = list(command)
    for idx, token in enumerate(redacted):
        if token == "--api-key" and idx + 1 < len(redacted):
            redacted[idx + 1] = "***REDACTED***"
    return redacted


def command_to_text(command: list[str]) -> str:
    return " ".join(redact_command(command))


def build_evalscope_command(
    config: dict[str, Any],
    *,
    api_url: str,
    api_key: str,
    work_dir: Path,
) -> list[str]:
    benchmark = config["benchmark"]
    model_cfg = config["model"]
    generation = config["generation"]
    runtime = config["runtime"]

    cmd = [
        "evalscope",
        "eval",
        "--model",
        str(model_cfg["name"]),
        "--api-url",
        api_url,
        "--api-key",
        api_key,
        "--eval-type",
        str(runtime.get("eval_type", "openai_api")),
        "--datasets",
        str(benchmark["name"]),
        "--work-dir",
        str(work_dir),
        "--no-timestamp",
    ]

    limit = benchmark.get("limit")
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    subset = benchmark.get("subset")
    if subset:
        dataset_args = {benchmark["name"]: {"subset_list": [subset] if isinstance(subset, str) else subset}}
        cmd.extend(["--dataset-args", json.dumps(dataset_args)])

    generation_config = {
        "temperature": generation.get("temperature", 0.0),
        "top_p": generation.get("top_p", 1.0),
        "max_tokens": generation.get("max_output_tokens", 16),
    }
    timeout = generation.get("timeout")
    if timeout is not None:
        generation_config["timeout"] = timeout
    cmd.extend(["--generation-config", json.dumps(generation_config)])

    return cmd


def write_resolved_config(run_dir: Path, config: dict[str, Any], started_at: datetime) -> None:
    resolved = yaml.safe_load(yaml.safe_dump(config))
    resolved.setdefault("resolved", {})
    resolved["resolved"]["started_at"] = started_at.isoformat()
    resolved["resolved"]["api_url_env"] = config["model"].get("api_url_env")
    resolved["resolved"]["api_key_env"] = config["model"].get("api_key_env")
    resolved["resolved"]["held_out_notice"] = HELD_OUT_NOTICE.strip()
    with (run_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False, allow_unicode=True)


def write_env_snapshot(run_dir: Path, config: dict[str, Any]) -> None:
    model_cfg = config["model"]
    api_url_env = model_cfg.get("api_url_env", "MODEL_API_URL")
    api_key_env = model_cfg.get("api_key_env", "MODEL_API_KEY")
    lines = [
        f"platform={platform.platform()}",
        f"{api_url_env}_present={bool(os.environ.get(api_url_env))}",
        f"{api_key_env}_present={bool(os.environ.get(api_key_env))}",
    ]
    (run_dir / "env.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    run_dir: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    *,
    run_id: str,
    command: list[str],
    started_at: datetime,
    finished_at: datetime | None,
    status: str,
) -> None:
    manifest = {
        "run_id": run_id,
        "experiment": config.get("experiment", {}),
        "held_out_notice": HELD_OUT_NOTICE.strip(),
        "benchmark": config.get("benchmark", {}),
        "model": {
            "name": config.get("model", {}).get("name"),
            "api_url_env": config.get("model", {}).get("api_url_env"),
            "api_key_env": config.get("model", {}).get("api_key_env"),
        },
        "git": metadata.get("git", {}),
        "python": metadata.get("python", {}),
        "packages": metadata.get("packages", {}),
        "machine": metadata.get("machine", {}),
        "environment": metadata.get("environment", {}),
        "command": command_to_text(command),
        "evalscope_cli_args": redact_command(command[2:] if len(command) > 2 else []),
        "timing": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat() if finished_at else None,
            "duration_seconds": (finished_at - started_at).total_seconds() if finished_at else None,
        },
        "status": status,
    }
    with (run_dir / "run_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)


def run_evalscope(config_path: Path, *, dry_run: bool = False) -> Path:
    config = load_config(config_path)
    validate_config(config)

    started_at = datetime.now(timezone.utc)
    run_id = make_run_id(config, started_at)
    output_root = resolve_path(config["logging"]["output_root"])
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    evalscope_raw = run_dir / "evalscope_raw"
    evalscope_raw.mkdir(parents=True, exist_ok=True)

    write_resolved_config(run_dir, config, started_at)
    write_env_snapshot(run_dir, config)

    api_url, api_key, env_errors = resolve_env(config)
    metadata = collect_metadata(
        config,
        api_url_present=bool(api_url),
        api_key_present=bool(api_key),
    )

    evalscope_exe = find_evalscope_executable()
    worked: list[str] = ["Config loaded", "Run directory created", "Metadata collected"]

    if dry_run:
        placeholder_cmd = ["evalscope", "eval", "# dry-run: command not executed"]
        write_manifest(
            run_dir,
            config,
            metadata,
            run_id=run_id,
            command=placeholder_cmd,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            status="dry_run",
        )
        (run_dir / "command.txt").write_text("# dry-run: command not executed\n", encoding="utf-8")
        print(run_dir)
        return run_dir

    if env_errors:
        command_repr = "# not executed"
        write_manifest(
            run_dir,
            config,
            metadata,
            run_id=run_id,
            command=[command_repr],
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            status="failed",
        )
        (run_dir / "command.txt").write_text(command_repr + "\n", encoding="utf-8")
        write_blocking_issue(
            run_dir,
            reason="\n".join(env_errors),
            command=command_repr,
            worked=worked,
            next_actions=[
                "Set required environment variables and rerun",
                "See experiments/evalscope_long_context/README.md for examples",
            ],
        )
        print(f"Run directory: {run_dir}", file=sys.stderr)
        raise SystemExit(2)

    if not evalscope_exe:
        command_repr = "# evalscope not found"
        write_manifest(
            run_dir,
            config,
            metadata,
            run_id=run_id,
            command=[command_repr],
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            status="failed",
        )
        (run_dir / "command.txt").write_text(command_repr + "\n", encoding="utf-8")
        write_blocking_issue(
            run_dir,
            reason="EvalScope is not installed or not on PATH.",
            command=command_repr,
            worked=worked,
            next_actions=[
                "Install EvalScope: pip install evalscope",
                "Activate the project conda env (e.g. longcontext) before running",
            ],
        )
        print(f"Run directory: {run_dir}", file=sys.stderr)
        raise SystemExit(3)

    command = build_evalscope_command(config, api_url=api_url, api_key=api_key, work_dir=evalscope_raw)
    if command[0] == "evalscope":
        command[0] = evalscope_exe

    command_text = command_to_text(command)
    (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"
    proc = subprocess.run(
        command,
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    finished_at = datetime.now(timezone.utc)
    status = "success" if proc.returncode == 0 else "failed"
    write_manifest(
        run_dir,
        config,
        metadata,
        run_id=run_id,
        command=command,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
    )

    metrics = parse_run(run_dir)

    if proc.returncode != 0:
        write_blocking_issue(
            run_dir,
            reason=f"EvalScope exited with code {proc.returncode}. See logs/stderr.log.",
            command=command_text,
            worked=worked + ["EvalScope command executed", "Parser invoked"],
            next_actions=[
                "Inspect logs/stderr.log and evalscope_raw/",
                "Verify API endpoint compatibility",
                "Confirm model name and generation settings",
            ],
        )
        print(f"Run directory: {run_dir}", file=sys.stderr)
        raise SystemExit(proc.returncode)

    if metrics.get("parse_status") == "failed":
        print(f"Run directory: {run_dir}", file=sys.stderr)
        raise SystemExit(4)

    print(run_dir)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EvalScope from YAML config.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create run directory and manifest without executing EvalScope.",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    run_evalscope(config_path, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
