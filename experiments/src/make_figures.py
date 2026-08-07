"""Generate paper figures from experiment results."""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito colorblind-safe palette
C = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
})

r1 = json.load(open(os.path.join(RES, "exp1_results.json")))
r23 = json.load(open(os.path.join(RES, "exp2_exp3_results.json")))
ROLES = sorted(r1.keys())

# --- Fig: ROC-AUC grouped bars -------------------------------------------
methods = ["individual role KL", "individual task KL", "allowlist rule",
           "isolation forest", "one-class SVM", "population (LOF)", "combined (best w)"]
labels = ["Role KL", "Task KL", "Allowlist", "iForest", "OC-SVM", "Population", "Combined"]
fig, ax = plt.subplots(figsize=(4.8, 2.2))
x = np.arange(len(ROLES))
w = 0.11
for i, (m, lab) in enumerate(zip(methods, labels)):
    vals = [r1[s]["auc"][m]["roc"] for s in ROLES]
    ax.bar(x + (i - 3) * w, vals, w, label=lab, color=C[i % len(C)])
ax.axhline(0.5, color="gray", lw=0.6, ls="--")
ax.set_xticks(x, [s.capitalize() for s in ROLES])
ax.set_ylabel("ROC-AUC")
ax.set_ylim(0.3, 0.9)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
fig.savefig(os.path.join(FIG, "auc_by_method.pdf"), bbox_inches="tight")
plt.close(fig)

# --- Fig: ROC curves (combined score) ------------------------------------
fig, ax = plt.subplots(figsize=(2.6, 2.4))
for i, s in enumerate(ROLES):
    a = r1[s]["arrays"]
    y = np.array(a["y"]); si = np.array(a["s_ind"]); sp = np.array(a["s_pop"])
    wb = r1[s]["w_best"]
    sc = wb * si + (1 - wb) * sp
    ok = ~np.isnan(sc)
    fpr, tpr, _ = roc_curve(y[ok], sc[ok])
    ax.plot(fpr, tpr, color=C[i], lw=1.2,
            label=f"{s.capitalize()} ({r1[s]['auc']['combined (best w)']['roc']:.2f})")
ax.plot([0, 1], [0, 1], color="gray", lw=0.6, ls="--")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend(frameon=False, loc="lower right")
fig.savefig(os.path.join(FIG, "roc_combined.pdf"), bbox_inches="tight")
plt.close(fig)

# --- Fig: fleet detection curves -----------------------------------------
fig, ax = plt.subplots(figsize=(2.8, 2.2))
cps = r23["config"]["checkpoints"]
for i, (n, r) in enumerate(sorted(r23["exp2"].items(), key=lambda kv: int(kv[0]))):
    tpr = [r["curve"][str(k)]["tpr"] for k in cps]
    ax.plot(cps, tpr, "-o", ms=2.5, lw=1.1, color=C[i], label=f"{int(n):,} agents")
ax.set_xscale("log")
ax.set_xticks(cps, [str(k) for k in cps])
ax.set_xlabel("Episodes observed per agent (k)")
ax.set_ylabel("Rogue-agent detection rate")
ax.set_ylim(0, 1.03)
ax.legend(frameon=False, loc="lower right")
fig.savefig(os.path.join(FIG, "fleet_detection.pdf"), bbox_inches="tight")
plt.close(fig)

# --- Fig: low-and-slow ----------------------------------------------------
fig, ax = plt.subplots(figsize=(2.8, 2.2))
ps = sorted(float(p) for p in r23["exp3_lowslow"])
for i, k in enumerate([20, 50, 100]):
    tpr = [r23["exp3_lowslow"][str(p)]["curve"][str(k)]["tpr"] for p in ps]
    ax.plot(ps, tpr, "-o", ms=2.5, lw=1.1, color=C[i], label=f"k = {k}")
ax.set_xlabel("Rogue activity rate $p$")
ax.set_ylabel("Rogue-agent detection rate")
ax.set_ylim(0, 1.03)
ax.legend(frameon=False, loc="upper left")
fig.savefig(os.path.join(FIG, "low_and_slow.pdf"), bbox_inches="tight")
plt.close(fig)

# --- Fig: mimicry ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.8, 2.2))
names = ["individual", "population", "combined"]
x = np.arange(len(names))
normal = [r23["exp3_mimicry"][n]["normal"]["curve"]["20"]["tpr"] for n in names]
mim = [r23["exp3_mimicry"][n]["mimicry"]["curve"]["20"]["tpr"] for n in names]
ax.bar(x - 0.18, normal, 0.34, color=C[0], label="Unconstrained attacker")
ax.bar(x + 0.18, mim, 0.34, color=C[3], label="Mimicry attacker")
ax.set_xticks(x, ["Individual", "Population", "Combined"])
ax.set_ylabel("Detection rate (k = 20)")
ax.set_ylim(0, 1.0)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=1)
fig.savefig(os.path.join(FIG, "mimicry.pdf"), bbox_inches="tight")
plt.close(fig)

# --- Fig: weight sweep ----------------------------------------------------
fig, ax = plt.subplots(figsize=(2.8, 2.2))
for i, s in enumerate(ROLES):
    sw = r1[s]["weight_sweep"]
    ws = sorted(float(k) for k in sw)
    ax.plot(ws, [sw[f"{w:.1f}" if w != 0 else "0.0"] for w in ws], "-o", ms=2.5,
            lw=1.1, color=C[i], label=s.capitalize())
ax.set_xlabel("Weight $w$ on individual score")
ax.set_ylabel("ROC-AUC of combined score")
ax.legend(frameon=False, loc="lower left", ncol=2)
fig.savefig(os.path.join(FIG, "weight_sweep.pdf"), bbox_inches="tight")
plt.close(fig)

# --- Fig: throughput ------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.8, 2.2))
ns = sorted(int(n) for n in r23["exp2_throughput"])
ms = [r23["exp2_throughput"][str(n)]["ms_per_episode"] for n in ns]
ax.plot(ns, ms, "-o", ms=3, lw=1.1, color=C[0])
ax.set_xscale("log")
ax.set_xlabel("Fleet snapshot size (episodes)")
ax.set_ylabel("Population scoring (ms/episode)")
ax.set_ylim(0, max(ms) * 1.3)
fig.savefig(os.path.join(FIG, "throughput.pdf"), bbox_inches="tight")
plt.close(fig)

print("figures written to", FIG)
