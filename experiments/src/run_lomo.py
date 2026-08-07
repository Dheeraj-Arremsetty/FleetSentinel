"""Leave-one-model-out (LOMO) generalization experiment.

Question: does FleetSentinel detect compromises in an agent whose backbone
model contributed NO profile data at all?

Protocol, per held-out model M and role:
  - Profile runs = benign episodes of the other five models (fingerprint fit).
  - Test set     = all episodes of M (benign + resisted = negatives,
                   compromised = positives).
  - S_ind: task-conditioned KL vs the 5-model fingerprint.
  - S_pop: LOF over snapshots of [M's negatives + 5-model benign fleet
           context + M's compromised subsampled to 2% contamination].
  - Combined with the per-role weight w* from Experiment 1.
Metrics: ROC-AUC (all compromised, and overt-only), skipping cells with < 5
positives.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from detect import RoleFingerprint, TaskFingerprint, s_pop_snapshot, split_tools

RNG = np.random.default_rng(23)
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results")

CONTAMINATION = 0.02
N_SNAPSHOTS = 30
COVERT = {("travel", "injection_task_6")}


def features_for(frame, fp, fp_task):
    rows = []
    for r in frame.itertuples():
        calls = json.loads(r.calls_json)
        base = fp.features(split_tools(r.tools), r.n_tool_errors)
        novel_rate, n_novel = fp_task.novel_action_stats(r.user_task, calls)
        rows.append(np.concatenate([base, [novel_rate, np.log1p(n_novel)]]))
    return np.stack(rows)


def run_cell(df_role, held_out, w_best):
    profile = df_role[(df_role.model != held_out) & (~df_role.attacked)]
    test = df_role[df_role.model == held_out]
    if test.compromised.sum() < 5:
        return None
    fleet_ctx = profile  # 5-model benign episodes as concurrent fleet traffic

    fp = RoleFingerprint().fit([split_tools(s) for s in profile.tools])
    fp_task = TaskFingerprint().fit(
        (r.user_task, json.loads(r.calls_json)) for r in profile.itertuples()
    )

    y = test.compromised.to_numpy().astype(int)
    s_ind = np.array([
        fp_task.s_ind(r.user_task, json.loads(r.calls_json)) for r in test.itertuples()
    ])

    X_test = features_for(test, fp, fp_task)
    X_ctx = features_for(fleet_ctx, fp, fp_task)

    neg_pos = np.where(y == 0)[0]
    pos_pos = np.where(y == 1)[0]
    n_base = len(neg_pos) + len(X_ctx)
    n_rogue = max(1, int(np.ceil(CONTAMINATION * n_base / (1 - CONTAMINATION))))
    pop_scores = {i: [] for i in range(len(test))}
    for _ in range(N_SNAPSHOTS):
        sampled = RNG.choice(pos_pos, size=min(n_rogue, len(pos_pos)), replace=False)
        members = np.concatenate([neg_pos, sampled])
        X_snap = np.vstack([X_test[members], X_ctx])
        scores = s_pop_snapshot(X_snap)[: len(members)]
        for i, s in zip(members, scores):
            pop_scores[i].append(s)
    s_pop = np.array([np.mean(v) if v else np.nan for v in pop_scores.values()])
    ok = ~np.isnan(s_pop)

    s_comb = w_best * s_ind + (1 - w_best) * s_pop
    suite = df_role.suite.iloc[0]
    covert = np.array([
        bool(c) and (suite, it) in COVERT
        for c, it in zip(test.compromised, test.injection_task)
    ])
    res = dict(
        n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
        auc_all=float(roc_auc_score(y[ok], s_comb[ok])),
    )
    overt = (y == 1) & ~covert
    m = ok & ((y == 0) | overt)
    if overt.sum() >= 5:
        res["auc_overt"] = float(roc_auc_score(y[m], s_comb[m]))
    return res


def main():
    df = pd.read_parquet(os.path.join(HERE, "..", "data", "traces.parquet"))
    r1 = json.load(open(os.path.join(OUT, "exp1_results.json")))
    out = {}
    for model in sorted(df.model.unique()):
        out[model] = {}
        for suite, g in df.groupby("suite"):
            cell = run_cell(g, model, r1[suite]["w_best"])
            if cell:
                out[model][suite] = cell
    with open(os.path.join(OUT, "lomo_results.json"), "w") as f:
        json.dump(out, f, indent=1)

    print(f"{'held-out model':34s} {'role':10s} {'n_pos':>5s} {'AUC':>6s} {'in-fleet':>8s}")
    for model, cells in out.items():
        for suite, c in cells.items():
            base = r1[suite]["auc"]["combined (best w)"]["roc"]
            print(f"{model:34s} {suite:10s} {c['n_pos']:5d} {c['auc_all']:6.3f} {base:8.3f}")
    aucs = [c["auc_all"] for cells in out.values() for c in cells.values()]
    print(f"\nmean LOMO AUC over {len(aucs)} model-role cells: {np.mean(aucs):.3f}")


if __name__ == "__main__":
    main()
