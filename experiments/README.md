# FleetSentinel experiments

Reproduction pipeline for the paper *FleetSentinel: Population-Level Behavioral
Monitoring for Detecting Rogue LLM Agents in Large-Scale Deployments*
(`../paper/main.tex`).

## Data

All traces come from the published runs of the AgentDojo benchmark
(https://github.com/ethz-spylab/agentdojo, `runs/` directory), sparse-cloned
into `agentdojo_repo/`. Six models, four suites (roles), 13,360 episodes.
Label semantics (verified against `agentdojo/task_suite/task_suite.py`):
`security: true` in a run file means the injected task was actually executed
(the episode is *compromised*).

## Pipeline

```
.venv/bin/python src/extract_traces.py    # runs/ JSONs -> data/traces.parquet
.venv/bin/python src/run_exp1.py          # episode-level detection -> results/exp1_*
.venv/bin/python src/run_exp2_exp3.py     # fleet sim + evasion -> results/exp2_exp3_results.json
.venv/bin/python src/make_figures.py      # -> ../paper/figures/*.pdf
```

Environment: Python 3.14 venv with numpy, scikit-learn, pandas, matplotlib,
pyarrow (`python3 -m venv .venv && .venv/bin/pip install numpy scikit-learn
pandas matplotlib pyarrow`).

Compile the paper with `tectonic main.tex` in `../paper`.

## Key design points (mirrors paper Sec. IV)

- `src/detect.py` — `TaskFingerprint`: task-conditioned fingerprints over an
  argument-aware action alphabet (calls whose identifier-like argument values
  were never seen in profile runs map to `tool#novel` symbols);
  `S_ind = 1 - exp(-KL(Q||P))`. `s_pop_snapshot`: LOF over a per-role fleet
  snapshot, `S_pop = 1 - exp(-max(LOF-1,0))`.
- `src/run_exp1.py` — 50/50 benign split into profile/test per role; negatives
  = held-out benign + attacked-but-resisted episodes; positives = compromised.
  Baselines: allowlist, isolation forest, one-class SVM; weight sweep for the
  combined score; overt/covert stratification (`COVERT` map; travel
  injection_task_6 is the only text-only goal).
- `src/run_exp2_exp3.py` — fleet simulation seeded by real per-episode score
  distributions; running-mean evidence with bootstrap-calibrated thresholds at
  1% agent-level false-alarm budget; low-and-slow (activity rate p) and
  mimicry (positives restricted to episodes evading the individual 95th-pct
  threshold) adversaries.

- `run_lomo.py` — leave-one-model-out generalization: fingerprints + fleet
  context from five models, detection on the held-out sixth (mean AUC 0.78
  over 19 model-role cells).
