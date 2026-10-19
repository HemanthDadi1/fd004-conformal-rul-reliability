"""Rebuild DS-B windows exactly as notebook 04 does, predict with the frozen
FD004_DS_B_Model.keras (no training), and compare with the frozen CSVs."""
import sys, os
import numpy as np, pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ROOT = sys.argv[1]
OUT = sys.argv[2]
RES = os.path.join(ROOT, "FD004_DS_B_Results")
SEED, RUL_CLIP, W, NREG, ENV_Q, ALPHA = 42, 125, 30, 6, 0.80, 0.20
SET = ["setting_1", "setting_2", "setting_3"]
SENS = ["s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"]
COLS = ["unit_id", "time_cycles"] + SET + [f"s_{i}" for i in range(1, 22)]

df = pd.read_csv(os.path.join(ROOT, "CMAPSSData/train_FD004.txt"), sep=r"\s+", header=None).iloc[:, : len(COLS)]
df.columns = COLS
df = df.merge(df.groupby("unit_id")["time_cycles"].max().rename("max_cycle"), on="unit_id")
df["RUL"] = (df["max_cycle"] - df["time_cycles"]).clip(upper=RUL_CLIP)
df["regime"] = KMeans(n_clusters=NREG, random_state=SEED, n_init=10).fit_predict(StandardScaler().fit_transform(df[SET]))

eng = np.array(sorted(df["unit_id"].unique()))
tr, tmp = train_test_split(eng, test_size=0.30, random_state=SEED)
ca, te = train_test_split(tmp, test_size=0.50, random_state=SEED)

# split check against frozen file
split = pd.read_csv(os.path.join(RES, "DS_B_Engine_Split.csv"))
frozen_split = {s: set(split.loc[split.split == s, "unit_id"]) for s in split.split.unique()}
print("split sizes", len(tr), len(ca), len(te), "| frozen file:", {k: len(v) for k, v in frozen_split.items()})
print("split identical:", frozen_split.get("train") == set(tr), frozen_split.get("calibration", frozen_split.get("cal")) == set(ca), frozen_split.get("test") == set(te))

fit = df[df.unit_id.isin(tr)]
stats = fit.groupby("regime")[SET].agg(["mean", "std"])


def score(d):
    s = np.full(len(d), np.nan)
    for r in sorted(d.regime.unique()):
        m = (d.regime == r).to_numpy()
        mu = np.array([stats.loc[r, (c, "mean")] for c in SET], float)
        sd = np.array([stats.loc[r, (c, "std")] for c in SET], float)
        sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
        z = (d.loc[m, SET].to_numpy(float) - mu) / sd
        s[m] = np.sqrt(np.mean(z**2, axis=1))
    return s


fit = fit.reset_index(drop=True)
fit["envelope_score"] = score(fit)
thr = fit.groupby("regime")["envelope_score"].quantile(ENV_Q).to_dict()
frozen_thr = pd.read_csv(os.path.join(RES, "DS_B_Regime_Envelope_Thresholds.csv")).set_index("regime")["envelope_threshold"]
print("max threshold diff vs frozen:", max(abs(thr[r] - frozen_thr[r]) for r in thr))

traj = df.reset_index(drop=True)
traj["envelope_score"] = score(traj)
traj["thr"] = traj.regime.map(thr)
traj["inside"] = traj.envelope_score <= traj.thr
traj = traj.sort_values(["unit_id", "time_cycles"]).reset_index(drop=True)
sc = StandardScaler().fit(traj[traj.unit_id.isin(tr) & traj.inside][SENS])
traj[SENS] = sc.transform(traj[SENS])


def windows(ids, mode):
    X, y, eid, sco, th = [], [], [], [], []
    for e in sorted(ids):
        g = traj[traj.unit_id == e].sort_values("time_cycles").reset_index(drop=True)
        sens, reg = g[SENS].to_numpy(np.float32), g.regime.to_numpy()
        for end in range(W - 1, len(g)):
            if bool(g.inside[end]) != (mode == "inside"):
                continue
            oh = np.eye(NREG, dtype=np.float32)[reg[end - W + 1 : end + 1]]
            X.append(np.concatenate([sens[end - W + 1 : end + 1], oh], -1))
            y.append(float(g.RUL[end])); eid.append(e)
            sco.append(g.envelope_score[end]); th.append(g.thr[end])
    return np.array(X), np.array(y, np.float32), np.array(eid), np.array(sco), np.array(th)


Xc, yc, idc, _, _ = windows(ca, "inside")
Xt, yt, idt, st, tht = windows(te, "outside")
print("cal windows", len(yc), "| test windows", len(yt), "test engines", len(np.unique(idt)))


class FastTemporalAttention(tf.keras.layers.Layer):
    def build(self, s):
        d = s[-1]
        self.W = self.add_weight(name="W", shape=(d, d), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(d,), initializer="zeros")
        self.v = self.add_weight(name="v", shape=(d, 1), initializer="glorot_uniform")

    def call(self, x):
        w = tf.nn.softmax(tf.tensordot(tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b), self.v, axes=1), axis=1)
        return tf.reduce_sum(x * w, axis=1)


model = tf.keras.models.load_model(
    os.path.join(RES, "FD004_DS_B_Model.keras"),
    custom_objects={"FastTemporalAttention": FastTemporalAttention},
    compile=False,
)
pc, pt = model.predict(Xc, batch_size=512, verbose=0), model.predict(Xt, batch_size=512, verbose=0)


def cq(s, a):
    n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - a)) / n), method="higher"))


ql, qu = cq(pc[:, 0] - yc, ALPHA / 2), cq(yc - pc[:, 2], ALPHA / 2)
err = np.maximum(pc[:, 0] - yc, yc - pc[:, 2])
eq = np.array([np.quantile(err[idc == e], 0.80) for e in np.unique(idc)])
qb = cq(eq, ALPHA)

p50 = pt[:, 1]
al, au = np.maximum(0, pt[:, 0] - ql), np.maximum(0, pt[:, 2] + qu)
bl, bu = np.maximum(0, pt[:, 0] - qb), np.maximum(0, pt[:, 2] + qb)
ac, bc = (yt >= al) & (yt <= au), (yt >= bl) & (yt <= bu)
rmse = float(np.sqrt(np.mean((yt - p50) ** 2)))
print(f"Q_lower {ql:.3f} Q_upper {qu:.3f} Q_balanced {qb:.3f}")

rows = []
for e in np.unique(idt):
    m = idt == e
    rows.append(dict(engine_id=int(e), n_windows=int(m.sum()), RMSE=np.sqrt(np.mean((yt[m] - p50[m]) ** 2)),
                     MAE=np.mean(np.abs(yt[m] - p50[m])), Asym_Coverage=100 * ac[m].mean(), Asym_Width=np.mean(au[m] - al[m]),
                     Balanced_Coverage=100 * bc[m].mean(), Balanced_Width=np.mean(bu[m] - bl[m])))
per = pd.DataFrame(rows)

glob = pd.read_csv(os.path.join(RES, "DS_B_Global_Summary.csv")).set_index("Method")
cmp = pd.DataFrame({
    "frozen": [glob.loc["Asymmetric CQR", "RMSE"], glob.loc["Asymmetric CQR", "MAE"], glob.loc["Asymmetric CQR", "Coverage"],
               glob.loc["Asymmetric CQR", "Mean_Width"], glob.loc["Balanced Engine CQR", "Coverage"], glob.loc["Balanced Engine CQR", "Mean_Width"],
               -0.9161808460063612, 27, 50.0],
    "reproduced": [rmse, float(np.mean(np.abs(yt - p50))), 100 * ac.mean(), float(np.mean(au - al)), 100 * bc.mean(), float(np.mean(bu - bl)),
                   float(np.corrcoef(per.RMSE, per.Balanced_Coverage)[0, 1]), int((per.Balanced_Coverage >= 80).sum()), float(per.Balanced_Coverage.min())],
}, index=["Point RMSE", "Point MAE", "Asym PICP %", "Asym MPIW", "Balanced PICP %", "Balanced MPIW",
          "RMSE-coverage r (Pearson)", "Balanced engines >=80%", "Worst engine Balanced %"])
cmp["diff"] = cmp.reproduced - cmp.frozen
print(cmp.round(4).to_string())

fr = pd.read_csv(os.path.join(RES, "DS_B_Engine_Reliability.csv")).set_index("engine_id")
j = per.set_index("engine_id").join(fr, rsuffix="_frozen")
print("per-engine max |diff| RMSE:", float((j.RMSE - j.RMSE_frozen).abs().max()),
      " Balanced cov:", float((j.Balanced_Coverage - j.Balanced_Coverage_frozen).abs().max()))

stage = np.where(yt > 80, "Early", np.where(yt > 40, "Mid", "Late"))
fs = pd.read_csv(os.path.join(RES, "DS_B_Life_Stage_Results.csv")).set_index("Life_Stage")
for s in ["Early", "Mid", "Late"]:
    m = stage == s
    print(f"{s}: n={m.sum()} RMSE {np.sqrt(np.mean((yt[m]-p50[m])**2)):.3f} (frozen {fs.loc[s,'RMSE']:.3f})  "
          f"Balanced {100*bc[m].mean():.2f} (frozen {fs.loc[s,'Balanced_Coverage']:.2f})  bias {np.mean(p50[m]-yt[m]):+.3f}")

os.makedirs(OUT, exist_ok=True)
pd.DataFrame(dict(engine_id=idt, true_RUL=yt, q10=pt[:, 0], q50=p50, q90=pt[:, 2], stage=stage,
                  asym_lower=al, asym_upper=au, bal_lower=bl, bal_upper=bu, envelope_score=st, threshold=tht)
             ).to_csv(os.path.join(OUT, "DS_B_frozen_model_rowlevel_predictions.csv"), index=False)
per.to_csv(os.path.join(OUT, "DS_B_frozen_model_per_engine.csv"), index=False)
cmp.to_csv(os.path.join(OUT, "DS_B_frozen_reproduction_check.csv"))
print("wrote", OUT)
