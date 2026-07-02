# MSDHF-Net

This is the official repository for the paper **MSDHF-Net: A Multi-Scale Decomposition and Heterogeneous Fusion Network for Long-Term Forecasting**.

The repository provides the PyTorch implementation used for multivariate long-term time series forecasting experiments in the paper. The code follows the proposed method: instance normalization, multi-scale causal average construction, normalized causal decomposition, local dynamic encoding for fluctuations, global structural encoding for trends, intra-scale gated fusion, content-adaptive cross-scale fusion, and a variable-shared lightweight direct prediction head.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Data Format

Place public benchmark CSV files under `data/`. The loader follows the common LTSF format used by Informer, Autoformer, and Time-Series-Library style datasets.

```text
data/
  ETTh1.csv
  ETTh2.csv
  ETTm1.csv
  Weather.csv
  Electricity.csv
  Traffic.csv
```

The time column can be named `date`, `datetime`, `time`, or `timestamp`. It is excluded from model inputs. For ETT datasets, keep the public field order:

```text
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
```

Use `--features M --target OT` for multivariate forecasting. Use `--features S --target OT` for single-variable forecasting.

## Run

Single experiment:

```bash
bash run.sh data/ETTh1.csv 96 none M OT
```

Equivalent Python command:

```bash
python -m msdhf.train --data data/ETTh1.csv --features M --target OT --seq-len 96 --pred-len 192 --batch-size 32 --lr 1e-4 --lambda-con 1e-3
```

Main experiments:

```bash
bash scripts/run_main.sh data M OT
```

Ablation experiments:

```bash
bash scripts/run_ablations.sh data M OT
```

Repeated seeds:

```bash
bash scripts/run_repeated.sh data/ETTh1.csv 96 none M OT
```

Mechanism statistics:

```bash
python -m msdhf.mechanism --data data/Weather.csv --features M --target OT --pred-len 720 --checkpoint checkpoints/Weather_h720_none.pt
```

Profiling:

```bash
python -m msdhf.profile --channels 7 --seq-len 96 --pred-len 96
```
