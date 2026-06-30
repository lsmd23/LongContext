#!/usr/bin/env python3
"""Collect reproducibility metadata for an EvalScope run."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

HELD_OUT_NOTICE = (
    "LongBench-v2 is held-out evaluation data.\n"
    "Do not use its context, question, choices, answer, rewritten samples, translated samples,\n"
    "or teacher-labeled variants for training."
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except OSError:
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_metadata(
    config: dict[str, Any],
    *,
    api_url_present: bool | None = None,
    api_key_present: bool | None = None,
) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    runtime_cfg = config.get("runtime", {})
    api_url_env = model_cfg.get("api_url_env", "MODEL_API_URL")
    api_key_env = model_cfg.get("api_key_env", "MODEL_API_KEY")

    if api_url_present is None:
        import os

        api_url_present = bool(os.environ.get(api_url_env))
    if api_key_present is None:
        import os

        api_key_present = bool(os.environ.get(api_key_env))

    dirty_output = _run_git(["status", "--porcelain"])
    return {
        "held_out_notice": HELD_OUT_NOTICE.strip(),
        "git": {
            "commit": _run_git(["rev-parse", "HEAD"]),
            "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "is_dirty": bool(dirty_output),
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "packages": {
            "evalscope": _package_version("evalscope"),
            "openai": _package_version("openai"),
            "pyyaml": _package_version("pyyaml") or _package_version("PyYAML"),
        },
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "client_machine": runtime_cfg.get("client_machine", "CPU dev machine"),
        },
        "environment": {
            "model_api_url_env": api_url_env,
            "model_api_url_present": api_url_present,
            "model_api_key_env": api_key_env,
            "model_api_key_present": api_key_present,
        },
    }


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect run metadata.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--output", required=True, help="Path to write metadata YAML.")
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    metadata = collect_metadata(config)
    write_metadata(Path(args.output), metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
