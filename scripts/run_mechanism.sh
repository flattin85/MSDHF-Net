#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_PATH="${1:-data/Weather.csv}"
PRED_LEN="${2:-720}"
FEATURES="${3:-M}"
TARGET="${4:-OT}"
CHECKPOINT="${5:-checkpoints/Weather_h720_none.pt}"
python -m msdhf.mechanism --data "$DATA_PATH" --features "$FEATURES" --target "$TARGET" --pred-len "$PRED_LEN" --checkpoint "$CHECKPOINT"
