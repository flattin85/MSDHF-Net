#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT="${1:-data}"
FEATURES="${2:-M}"
TARGET="${3:-OT}"
mkdir -p logs
datasets=(ETTh1 ETTh2 ETTm1 Weather Electricity Traffic)
horizons=(96 192 336 720)
for dataset in "${datasets[@]}"; do
  for horizon in "${horizons[@]}"; do
    python -m msdhf.train --data "$DATA_ROOT/$dataset.csv" --features "$FEATURES" --target "$TARGET" --seq-len 96 --pred-len "$horizon" --batch-size 32 --lr 1e-4 --lambda-con 1e-3 2>&1 | tee "logs/${dataset}_h${horizon}_none.log"
  done
done
