#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT="${1:-data}"
FEATURES="${2:-M}"
TARGET="${3:-OT}"
mkdir -p logs
datasets=(ETTh2 Weather Traffic)
horizons=(96 336)
ablations=(none without_ncd without_local without_global without_igf without_ccf without_lcon)
for dataset in "${datasets[@]}"; do
  for horizon in "${horizons[@]}"; do
    for ablation in "${ablations[@]}"; do
      python -m msdhf.train --data "$DATA_ROOT/$dataset.csv" --features "$FEATURES" --target "$TARGET" --seq-len 96 --pred-len "$horizon" --batch-size 32 --lr 1e-4 --lambda-con 1e-3 --ablation "$ablation" 2>&1 | tee "logs/${dataset}_h${horizon}_${ablation}.log"
    done
  done
done
