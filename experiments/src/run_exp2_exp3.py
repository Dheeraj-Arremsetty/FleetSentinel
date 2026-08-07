"""Experiments 2 & 3: fleet-scale sequential detection and evasion robustness.

Both are grounded in Experiment 1's *real* per-episode scores: benign agents
draw episode scores from the empirical negative distribution of their role;
rogue agents mix in draws from the empirical compromised (overt) distribution.

Exp 2 — fleet simulation:
  N agents (1k..10k), 1% rogue, horizon T episodes. Per-agent evidence =
  running mean of episode scores. Per-role, per-k thresholds calibrated by
  bootstrap on benign scores for a 1% agent-level false-alarm budget.
  Metrics: agent-level detection rate & FPR vs episodes observed; median
  episodes-to-detection; scoring throughput vs fleet size (real LOF fits).

Exp 3 — evasion:
  (a) low-and-slow: rogue activity rate p in {0.05..1.0}
  (b) mimicry: rogue episodes restricted to compromised episodes whose
      *individual* score evades the individual 95th-percentile threshold;
      compare individual-only vs population-only vs combined evidence.
"""

import json
import os
import time

import numpy as np

from detect import s_pop_snapshot

RNG = np.random.default_rng(11)
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results")

ROGUE_FRAC = 0.01
FLEET_FPR = 0.01          # agent-level false-alarm budget
T = 100                    # episodes per agent over the horizon
CHECKPOINTS = [1, 2, 5, 10, 20, 50, 100]
N_BOOT = 4000


def load_scores():
    r = json.load(open(os.path.join(OUT, "exp1_results.json")))
    roles = {}
    for suite, v in r.items():
        a = v["arrays"]
        y = np.array(a["y"])
        covert = np.array(a["covert"]).astype(bool)
        w = v["w_best"]
        s_ind = np.array(a["s_ind"])
        s_pop = np.array(a["s_pop"])
        ok = ~np.isnan(s_pop)
        scores = {
            "individual": s_ind,
            "population": s_pop,
            "combined": w * s_ind + (1 - w) * s_pop,
        }
        roles[suite] = {
            name: dict(
                neg=sc[ok & (y == 0)],
                pos=sc[ok & (y == 1) & ~covert],  # overt compromised only
            )
            for name, sc in scores.items()
        }
        roles[suite]["_s_ind_neg_q95"] = float(np.quantile(s_ind[ok & (y == 0)], 0.95))
        roles[suite]["_mimicry_mask"] = (
            s_ind[ok & (y == 1) & ~covert] < roles[suite]["_s_ind_neg_q95"]
        )
    return roles


def calibrate_thresholds(neg, checkpoints, fpr, n_boot=N_BOOT):
    """Bootstrap the distribution of running means of benign episode scores."""
    thr = {}
    for k in checkpoints:
        means = neg[RNG.integers(0, len(neg), size=(n_boot, k))].mean(axis=1)
        thr[k] = float(np.quantile(means, 1 - fpr))
    return thr


def simulate(roles, score_name, p_active, mimicry=False, n_agents=2000,
             rogue_frac=ROGUE_FRAC, reps=10):
    """Returns detection curves over checkpoints for one attacker setting."""
    role_names = sorted(k for k in roles if not k.startswith("_"))
    det = {k: [0, 0] for k in CHECKPOINTS}   # k -> [detected rogues, rogues]
    fa = {k: [0, 0] for k in CHECKPOINTS}    # k -> [false alarms, benign]
    latencies = []
    for _, role in [(rep, role) for rep in range(reps) for role in role_names]:
        d = roles[role][score_name]
        neg, pos = d["neg"], d["pos"]
        if mimicry:
            pos = pos[roles[role]["_mimicry_mask"]]
            if len(pos) == 0:
                continue
        thr = calibrate_thresholds(neg, CHECKPOINTS, FLEET_FPR)
        n_role = n_agents // len(role_names)
        n_rogue = max(1, int(round(rogue_frac * n_role)))
        n_benign = n_role - n_rogue
        # benign agents
        b_scores = neg[RNG.integers(0, len(neg), size=(n_benign, T))]
        # rogue agents: each episode active w.p. p_active
        active = RNG.random((n_rogue, T)) < p_active
        r_scores = neg[RNG.integers(0, len(neg), size=(n_rogue, T))]
        r_scores[active] = pos[RNG.integers(0, len(pos), size=int(active.sum()))]
        for k in CHECKPOINTS:
            fa[k][0] += int((b_scores[:, :k].mean(axis=1) > thr[k]).sum())
            fa[k][1] += n_benign
            det_mask = r_scores[:, :k].mean(axis=1) > thr[k]
            det[k][0] += int(det_mask.sum())
            det[k][1] += n_rogue
        # episodes-to-detection: first checkpoint whose test fires
        for i in range(n_rogue):
            fired = [k for k in CHECKPOINTS if r_scores[i, :k].mean() > thr[k]]
            latencies.append(min(fired) if fired else np.inf)
    curve = {
        k: dict(tpr=det[k][0] / max(det[k][1], 1), fpr=fa[k][0] / max(fa[k][1], 1))
        for k in CHECKPOINTS
    }
    finite = [x for x in latencies if np.isfinite(x)]
    med_lat = float(np.median(finite)) if finite else None
    return dict(curve=curve, median_latency=med_lat,
                frac_detected_by_T=float(np.mean([np.isfinite(x) for x in latencies])))


def throughput():
    """Wall-clock cost of population scoring vs fleet snapshot size, using
    real Experiment-1 feature vectors resampled to the target size."""
    X = np.load(os.path.join(OUT, "exp1_X_workspace.npz"))["X"]
    res = {}
    for n in [1000, 2000, 5000, 10000]:
        Xn = X[RNG.integers(0, len(X), size=n)]
        Xn = Xn + RNG.normal(0, 1e-6, Xn.shape)  # break exact duplicates
        t0 = time.perf_counter()
        s_pop_snapshot(Xn)
        dt = time.perf_counter() - t0
        res[n] = dict(seconds=dt, ms_per_episode=dt / n * 1e3)
    return res


def main():
    roles = load_scores()

    exp2 = {}
    for n_agents in [1000, 2000, 5000, 10000]:
        exp2[n_agents] = simulate(roles, "combined", p_active=1.0, n_agents=n_agents)
    exp2_thru = throughput()

    exp3_lowslow = {
        p: simulate(roles, "combined", p_active=p)
        for p in [0.05, 0.1, 0.2, 0.5, 1.0]
    }
    exp3_mimicry = {
        name: dict(
            normal=simulate(roles, name, p_active=1.0, mimicry=False),
            mimicry=simulate(roles, name, p_active=1.0, mimicry=True),
        )
        for name in ["individual", "population", "combined"]
    }
    # fraction of compromised episodes that can evade the individual threshold
    mim_frac = {
        role: float(np.mean(roles[role]["_mimicry_mask"]))
        for role in roles if not role.startswith("_")
    }

    out = dict(exp2=exp2, exp2_throughput=exp2_thru,
               exp3_lowslow=exp3_lowslow, exp3_mimicry=exp3_mimicry,
               mimicry_pool_fraction=mim_frac,
               config=dict(rogue_frac=ROGUE_FRAC, fleet_fpr=FLEET_FPR, T=T,
                           checkpoints=CHECKPOINTS))
    with open(os.path.join(OUT, "exp2_exp3_results.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("=== Exp 2: fleet detection (combined, persistent rogue) ===")
    for n, r in exp2.items():
        c = r["curve"]
        print(f" fleet={n:5d}  det@k10={c[10]['tpr']:.2f} det@k20={c[20]['tpr']:.2f} "
              f"det@k50={c[50]['tpr']:.2f} fpr@k50={c[50]['fpr']:.3f} "
              f"median latency={r['median_latency']}")
    print("=== Exp 2: throughput ===")
    for n, r in exp2_thru.items():
        print(f" n={n:5d}  {r['seconds']:.3f}s  {r['ms_per_episode']:.4f} ms/episode")
    print("=== Exp 3a: low-and-slow (combined) ===")
    for p, r in exp3_lowslow.items():
        c = r["curve"]
        print(f" p={p:.2f}  det@k20={c[20]['tpr']:.2f} det@k100={c[100]['tpr']:.2f} "
              f"median latency={r['median_latency']}")
    print("=== Exp 3b: mimicry (evading individual threshold) ===")
    print(" evadable episode fraction per role:", {k: round(v, 2) for k, v in mim_frac.items()})
    for name, r in exp3_mimicry.items():
        print(f" {name:11s} normal det@k20={r['normal']['curve'][20]['tpr']:.2f}  "
              f"mimicry det@k20={r['mimicry']['curve'][20]['tpr']:.2f}")


if __name__ == "__main__":
    main()
