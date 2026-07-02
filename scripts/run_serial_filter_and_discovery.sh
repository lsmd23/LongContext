#!/usr/bin/env bash
set -euo pipefail

ROOT="${LONGCONTEXT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

source "${CONDA_SH:-$HOME/lisunmuduo/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-longcontext}"

LOG_DIR="$ROOT/data/_pipeline_logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/serial_filter_discovery_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1

HELDOUT_ARGS=(
  --heldout-benchmark longbench
  --heldout-benchmark longbench_v2
)

DISCOVERY_MAX_DATASETS_PER_TERM="${DISCOVERY_MAX_DATASETS_PER_TERM:-100}"
DISCOVERY_MAX_CONFIGS="${DISCOVERY_MAX_CONFIGS:-20}"
DISCOVERY_SAMPLE_ROWS="${DISCOVERY_SAMPLE_ROWS:-1000}"

echo "=== CONFIG ==="
echo "root=$ROOT"
echo "heldout=longbench,longbench_v2"
echo "discovery: max_datasets_per_term=$DISCOVERY_MAX_DATASETS_PER_TERM max_configs=$DISCOVERY_MAX_CONFIGS sample_rows=$DISCOVERY_SAMPLE_ROWS"
echo "log=$RUN_LOG"
echo

echo "=== START FILTER $(date -Is) ==="
python scripts/filter_all_lcqa.py \
  --normalized-root data/normalized \
  --filtered-root data/filtered \
  "${HELDOUT_ARGS[@]}" \
  --progress
echo "=== FILTER DONE $(date -Is) ==="
echo

echo "=== START SUMMARY $(date -Is) ==="
python scripts/summarize_filtered_buckets.py --progress
echo "=== SUMMARY DONE $(date -Is) ==="
echo

echo "=== START DISCOVERY FULL $(date -Is) ==="
python scripts/discover_hf_candidates.py \
  "${HELDOUT_ARGS[@]}" \
  --max-datasets-per-term "$DISCOVERY_MAX_DATASETS_PER_TERM" \
  --max-configs-per-dataset "$DISCOVERY_MAX_CONFIGS" \
  --sample-rows "$DISCOVERY_SAMPLE_ROWS" \
  --output-root data
echo "=== DISCOVERY DONE $(date -Is) ==="
echo

echo "=== ALL DONE $(date -Is) ==="
