# FD004 Paper 1: engine-level reliability of conformal RUL uncertainty under operating-condition shift

Code and results for *Engine-Level Reliability of Conformal RUL Uncertainty Under Operating-Condition
Distribution Shift in C-MAPSS FD004 (Part I)*.

The paper asks whether conformal prediction intervals for remaining useful life (RUL) stay reliable
for each engine, not only on average, when operating conditions change. It compares Asymmetric CQR
with an engine-balanced variant (Balanced Engine CQR) under a standard split (IID) and two shifts:
a held-out operating regime (DS-A) and a restricted operating envelope (DS-B).

## Layout

| Path | What it is |
|---|---|
| `notebooks/01_…`, `02_…` | Preprocessing, IID windows, point-model development (Table 1), original IID uncertainty analysis |
| `notebooks/03_…`, `04_…`, `05_…` | Original DS-A, DS-B and master-comparison notebooks (archived; see "History") |
| `notebooks/06_…DS_A_Rerun…` | DS-A, five seeds, pre-registered convergence rule (reported results) |
| `notebooks/07_…IID_Rerun…` | IID, same rule (reported results) |
| `notebooks/08_…DS_B_Rerun…` | DS-B, same rule (reported results) |
| `FD004_IID_Rerun_Results/`, `FD004_DS_A_Rerun_Results/`, `FD004_DS_B_Rerun_Results/` | Everything the reported numbers come from: `protocol.json`, trained models (`models/`), training histories, convergence check, row-level calibration and test predictions, per-engine, life-stage and bootstrap tables, SHA-256 manifest |
| `artifacts/FD004_baseline_tensors.npz` | IID input windows produced by notebook 01 |
| `FD004_DS_B_Results/` | DS-B engine split and envelope thresholds (inputs checked by notebook 08) |
| `models/b6_variants/b6_cal_eval_split_ids.npz` | IID calibration/evaluation engine split (checked by notebook 07) |
| `audit/` | Verification of archived results: frozen DS-B predictions, the regime-conditional baseline fix, the original hash manifest |
| `Supplementary_Data_S1.xlsx` | Engine splits, thresholds, protocol, convergence results |

## Reproduce

1. Get the data (see `DATA.md`) so that `CMAPSSData/train_FD004.txt` exists in this folder.
2. Create an environment with `requirements.txt` (Python 3.12, TensorFlow 2.20).
3. From this folder:

```bash
export FD004_ROOT="$PWD"
jupyter nbconvert --to notebook --execute notebooks/06_FD004_DS_A_Rerun_Converged_FiveSeed.ipynb
```

Repeat for notebooks 07 and 08. Each writes to its own `FD004_*_Rerun_Results` folder. Set
`FD004_SMOKE=1` first for a one-seed, three-epoch pipeline check that writes to a separate
`*_SMOKE` folder. On a laptop CPU a full five-seed run takes about an hour; a GPU is much faster.

Each notebook writes `protocol.json` before training, trains and saves every seed without touching
the evaluation windows, and only then evaluates. If `protocol.json` already exists, the notebook
refuses to run with a different protocol.

Training on a different machine or TensorFlow build can give slightly different weights. The saved
models and row-level predictions in the results folders are the reference for every reported number.

## History (why there are reruns)

The original runs (notebooks 03–05) trained for a fixed 25 epochs. For several seeds that budget
left the model on a plateau where it predicts constant quantiles, and the original DS-A model was
never saved, so its results could not be reproduced. The reported results therefore come from
notebooks 06–08, which use one training rule for all three protocols, fixed before any evaluation:
20% of training engines held out for internal validation, up to 150 epochs, early stopping
(patience 15, from epoch 30), and a convergence check against the constant-quantile loss. Every
seed is reported.

## Citation

See `CITATION.cff`. Archive DOI: [add after the Zenodo release].

## License

Code: MIT (`LICENSE`). The C-MAPSS data are NASA's and are not included.
