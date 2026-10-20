"""Figure 4: width vs engines below 80%, Window CQR at fixed levels, Balanced Engine CQR and width-matched Asymmetric CQR."""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
INK, MUTED, GRID, GREY, BLUE, ORANGE = "#1A1A1A", "#5F5F5F", "#E3E3E3", "#9C9C8C", "#2878D7", "#D9822B"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 11})
r = pd.read_csv(src)
m = r.groupby(["condition", "method"])[["MPIW", "below80"]].mean().reset_index()
fig, axes = plt.subplots(1, 3, figsize=(8.96, 3.9), dpi=150, sharey=False)
for ax, c in zip(axes, ["IID", "DS-A", "DS-B"]):
    d = m[m.condition == c].set_index("method")
    lv = [f"Window CQR {k}%" for k in (80, 85, 90, 95)]
    ax.plot(d.loc[lv, "MPIW"], d.loc[lv, "below80"], color=GREY, lw=1.8, marker="o", ms=5, zorder=1)
    for k, name in zip((80, 85, 90, 95), lv):
        ax.annotate(f"{k}%", (d.loc[name, "MPIW"], d.loc[name, "below80"]), textcoords="offset points", xytext=(6, 4), color=MUTED, fontsize=9)
    ax.scatter(d.loc["Asymmetric, width-matched", "MPIW"], d.loc["Asymmetric, width-matched", "below80"], marker="^", s=90, color=ORANGE, zorder=3,
               label="Asymmetric, width-matched")
    ax.scatter(d.loc["Balanced Engine CQR", "MPIW"], d.loc["Balanced Engine CQR", "below80"], marker="s", s=70, color=BLUE, zorder=4,
               label="Balanced Engine CQR")
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    ax.grid(True, color=GRID); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, length=0)
    ax.set_title(c, color=INK, fontsize=12); ax.set_xlabel("Mean interval width (cycles)", color=MUTED)
axes[0].set_ylabel("Engines below 80% coverage", color=MUTED)
axes[2].legend(frameon=False, loc="lower left", fontsize=9)
fig.suptitle("At equal width, the engine-balanced interval is no better: it picks a point on the same curve",
             x=0.012, ha="left", fontsize=13, color=INK)
fig.text(0.012, 0.875, "Five-seed means; grey line = Window CQR at fixed nominal levels 80–95%", ha="left", color=MUTED, fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(out, dpi=150); plt.close(fig)
print("ok")
