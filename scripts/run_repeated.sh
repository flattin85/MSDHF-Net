#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_PATH="${1:-data/ETTh1.csv}"
PRED_LEN="${2:-96}"
ABLATION="${3:-none}"
FEATURES="${4:-M}"
TARGET="${5:-OT}"
mkdir -p logs
seeds=(2024 2025 2026 2027 2028)
dataset="$(basename "$DATA_PATH" .csv)"
for seed in "${seeds[@]}"; do
  python -m msdhf.train --data "$DATA_PATH" --features "$FEATURES" --target "$TARGET" --seq-len 96 --pred-len "$PRED_LEN" --batch-size 32 --lr 1e-4 --lambda-con 1e-3 --ablation "$ABLATION" --seed "$seed" --save-dir "checkpoints/seed_${seed}" 2>&1 | tee "logs/${dataset}_h${PRED_LEN}_${ABLATION}_seed${seed}.log"
done
