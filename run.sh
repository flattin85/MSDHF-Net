#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
DATA_PATH="${1:-data/ETTh1.csv}"
PRED_LEN="${2:-96}"
ABLATION="${3:-none}"
FEATURES="${4:-M}"
TARGET="${5:-OT}"
python -m msdhf.train --data "$DATA_PATH" --pred-len "$PRED_LEN" --seq-len 96 --batch-size 32 --lr 1e-4 --lambda-con 1e-3 --ablation "$ABLATION" --features "$FEATURES" --target "$TARGET"
