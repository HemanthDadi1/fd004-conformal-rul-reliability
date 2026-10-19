
FD004 Paper 1 — Fixed-Spec Five-Seed UQ Confirmation
====================================================

Seeds: 42, 123, 456, 789, 1000

Files
-----
1. FD004_Paper1_FiveSeed_IID_UQ_Confirmatory.ipynb
   Standalone IID confirmatory run using FD004_baseline_tensors.npz.

2. 03_FD004_DS_A_Heldout_Regime_FiveSeed_UQ_Confirmatory.ipynb
   Copy of the frozen DS-A notebook with a final five-seed confirmatory section appended.

3. 04_FD004_DS_B_Restricted_Operating_Envelope_FiveSeed_UQ_Confirmatory.ipynb
   Copy of the frozen DS-B notebook with a final five-seed confirmatory section appended.

4. FD004_Paper1_FiveSeed_UQ_Aggregator.ipynb
   Combines the per-seed CSVs after all three runs.

Protocol safeguards
-------------------
- No tuning or model selection.
- Seeds fixed at 42, 123, 456, 789, 1000.
- 25 epochs, batch size 64, Adam 1e-3.
- Final 20→18 projection + BiLSTM(64) + temporal attention + q10/q50/q90.
- Asymmetric CQR uses finite-sample method='higher' with alpha/2 per side,
  matching the frozen shift notebooks.
- Balanced Engine CQR uses the frozen two-stage implementation:
  within-engine 0.80 np.quantile default interpolation, then across-engine
  finite-sample method='higher'.
- Each notebook writes a seed-42 reproduction gate. If that gate fails,
  do not use the five-seed summary until the runtime/data mismatch is resolved.

Recommended run order
---------------------
IID -> DS-A -> DS-B -> Aggregator.

For DS-A/DS-B, run the frozen preprocessing/window-construction portion first.
The appended confirmatory section trains five fresh models from the frozen arrays.
