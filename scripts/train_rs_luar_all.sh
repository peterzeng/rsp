#!/bin/bash

# Simple driver script to train the Residualized Similarity (RS) model
# with LUAR as the neural base encoder on Reddit, Amazon, and Fanfiction.
#
# Usage (from repo root):
#   bash scripts/train_rs_luar_all.sh
#
# This script is intentionally minimal and just forwards to
# src/train_attention_residual.py with the right arguments.

set -euo pipefail

MODEL_TYPE="luar"
DATASETS=("reddit" "amazon" "fanfiction")
LOG_DIR="training_logs"

mkdir -p "${LOG_DIR}"

run_training() {
  local dataset="$1"
  local timestamp
  timestamp="$(date +"%Y%m%d_%H%M%S")"
  local log_file="${LOG_DIR}/${MODEL_TYPE}_${dataset}_${timestamp}.log"

  echo "=========================================="
  echo "Training RS (${MODEL_TYPE}) on dataset: ${dataset}"
  echo "Logs: ${log_file}"
  echo "=========================================="

  python src/train_attention_residual.py \
    -m "${MODEL_TYPE}" \
    -d "${dataset}" \
    2>&1 | tee "${log_file}"
}

for ds in "${DATASETS[@]}"; do
  run_training "${ds}"
  # Short pause between runs to let CUDA clean up if needed
  sleep 5
done

echo "All LUAR-based RS runs (reddit, amazon, fanfiction) completed."


