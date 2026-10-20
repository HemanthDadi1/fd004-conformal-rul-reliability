"""Width- and coverage-matched baselines for Balanced Engine CQR (Paper 1), from saved calibration and test predictions."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
A = 0.20
SRC = {"IID": "FD004_IID_Rerun_Results", "DS-A": "FD004_DS_A_Rerun_Results", "DS-B": "FD004_DS_B_Rerun_Results"}
if len(sys.argv) > 3 and sys.argv[3] == "FD002":
    SRC = {"IID": "FD004_FD002_IID_W30_Results", "DS-A": "FD004_FD002_DS_A_W30_Results", "DS-B": "FD004_FD002_DS_B_W30_Results"}


def cq(s, level):
    n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * level) / n), method="higher"))


def intervals(l, u, dl, du):
    return np.maximum(0, l - dl), np.maximum(0, u + du)


def width_match(lc, uc, ql, qu, target):
    """Extra margin m >= 0 added to both asymmetric tails so the mean calibration width equals target (label-free)."""
    lo_m, hi_m = 0.0, 200.0
    for _ in range(60):
        m = (lo_m + hi_m) / 2
        a, b = intervals(lc, uc, ql + m, qu + m)
        lo_m, hi_m = (m, hi_m) if (b - a).mean() < target else (lo_m, m)
    return (lo_m + hi_m) / 2


rows, eng_rows = [], []
for cond, d in SRC.items():
    test = pd.read_csv(ROOT / d / "test_predictions_rowlevel.csv").rename(columns={"engine_id": "engine"})
    for seed in sorted(test.seed.unique()):
        cal = pd.read_csv(ROOT / d / f"calibration_predictions_seed{seed}.csv").rename(columns={"engine_id": "engine"})
        t = test[test.seed == seed]
        lc, uc, yc, ec = cal.q10.values, cal.q90.values, cal.true_RUL.values, cal.engine.values
        lt, ut, yt, et, st = t.q10.values, t.q90.values, t.true_RUL.values, t.engine.values, t.stage.values
        ql, qu = cq(lc - yc, 1 - A / 2), cq(yc - uc, 1 - A / 2)
        s2 = np.maximum(lc - yc, yc - uc)
        bq = np.array([np.quantile(s2[ec == e], 0.80) for e in np.unique(ec)]); M = len(bq)
        qb = float(np.quantile(bq, min(1.0, np.ceil((M + 1) * (1 - A)) / M), method="higher"))
        bal_c = intervals(lc, uc, qb, qb)
        implied = float(((yc >= bal_c[0]) & (yc <= bal_c[1])).mean())   # pooled calibration coverage Balanced chooses
        m = width_match(lc, uc, ql, qu, float((bal_c[1] - bal_c[0]).mean()))
        methods = {"Asymmetric CQR (80%)": (ql, qu), "Balanced Engine CQR": (qb, qb), "Asymmetric, width-matched": (ql + m, qu + m)}
        for lvl in (0.80, 0.85, 0.90, 0.95):
            d_ = cq(s2, lvl); methods[f"Window CQR {int(lvl * 100)}%"] = (d_, d_)
        dw = cq(s2, implied); methods["Window CQR at Balanced's implied level"] = (dw, dw)
        for name, (dl, du) in methods.items():
            lo, hi = intervals(lt, ut, dl, du)
            cov = (yt >= lo) & (yt <= hi)
            e = pd.Series(cov).groupby(et).mean() * 100
            rows.append({"condition": cond, "seed": seed, "method": name, "PICP": 100 * cov.mean(), "MPIW": float((hi - lo).mean()),
                         "below80": int((e < 80).sum()), "eng_p10": float(np.percentile(e, 10)), "eng_worst": float(e.min()),
                         "mid_PICP": 100 * cov[st == "Mid"].mean(), "implied_level": 100 * implied, "width_margin_m": m})
rows = pd.DataFrame(rows); rows.to_csv(OUT / "width_matched_baselines_by_seed.csv", index=False)
summ = rows.groupby(["condition", "method"], sort=False)[["PICP", "MPIW", "below80", "eng_p10", "eng_worst", "mid_PICP", "implied_level"]].agg(["mean", "std"])
summ.to_csv(OUT / "width_matched_baselines_five_seed.csv")
pd.set_option("display.width", 250)
print(summ.round(2).to_string())
# paired: Balanced vs width-matched Asymmetric, per seed
p = rows.pivot_table(index=["condition", "seed"], columns="method", values=["below80", "eng_p10", "eng_worst", "MPIW", "PICP"])
for c in ["below80", "eng_p10", "eng_worst", "MPIW", "PICP"]:
    diff = p[(c, "Balanced Engine CQR")] - p[(c, "Asymmetric, width-matched")]
    print(c, "Balanced − width-matched Asym:", diff.groupby(level=0).agg(["mean", "std", "min", "max"]).round(2).to_dict("index"))
