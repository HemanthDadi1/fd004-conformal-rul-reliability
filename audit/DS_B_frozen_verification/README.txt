DS-B frozen verification (2026-10-19)

DS_B_frozen_model_rowlevel_predictions.csv was produced by verify_dsb.py from the frozen
FD004_DS_B_Model.keras that was in FD004_DS_B_Results at the time. It reproduced the frozen
DS-B summaries exactly (RMSE 23.999, Asymmetric 75.97%, Balanced 90.63%, r = -0.916).

Later the same day FD004_DS_B_Results was replaced by the Sep 5 Drive copy, which holds a
re-run model (RMSE 25.952). The frozen model (sha256 c2062592..., see
Paper1_artifact_hash_manifest.csv) is no longer on this machine. These CSVs are now the
record of the frozen DS-B predictions.

DS_B_frozen_model_rowlevel_predictions_with_cycle_regime.csv adds endpoint cycle and regime,
joined by row position from a re-export with the same deterministic window order
(engine_id, true_RUL and envelope_score checked identical on every row).
