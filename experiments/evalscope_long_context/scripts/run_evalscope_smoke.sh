#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MODULE_ROOT}/../.." && pwd)"
CONFIG="${MODULE_ROOT}/configs/smoke_longbench_v2.yaml"

cd "${REPO_ROOT}"

if [[ -z "${MODEL_API_URL:-}" ]]; then
  echo "ERROR: MODEL_API_URL is not set." >&2
  echo "Example: export MODEL_API_URL=\"https://your-openai-compatible-endpoint/v1\"" >&2
  exit 1
fi

if [[ -z "${MODEL_API_KEY:-}" ]]; then
  echo "ERROR: MODEL_API_KEY is not set." >&2
  echo "Example: export MODEL_API_KEY=\"your_key\"" >&2
  exit 1
fi

if ! command -v evalscope >/dev/null 2>&1; then
  echo "ERROR: evalscope command not found." >&2
  echo "Install with: pip install evalscope" >&2
  exit 1
fi

RUN_DIR="$(python "${SCRIPT_DIR}/run_evalscope_from_config.py" --config "${CONFIG}")"
echo "Run directory: ${RUN_DIR}"
