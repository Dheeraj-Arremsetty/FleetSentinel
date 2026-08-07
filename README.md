# FleetSentinel

**Population-level behavioral monitoring for detecting rogue LLM agents in large-scale deployments.**

LLM agents can be hijacked by indirect prompt injection and made to execute attacker-chosen actions. FleetSentinel detects compromised agents by combining two cheap, purely statistical signals — no LLM in the detection loop:

1. **Individual score** — does this run match its task's behavioral fingerprint? (task-conditioned KL divergence over an argument-aware action alphabet)
2. **Population score** — does this run look like what peer agents in the same role are doing? (local-outlier-factor over fleet snapshots)

Per-agent evidence accumulates across runs, with alarm thresholds calibrated to an explicit fleet-level false-alarm budget.

📄 Paper: [`paper/main.pdf`](paper/main.pdf) (LaTeX source in [`paper/main.tex`](paper/main.tex))

## Key results

Evaluated on 13,360 real execution traces (49,046 tool calls) from six LLMs
(GPT-4o, GPT-4o-mini, Claude 3.5/3.7 Sonnet, Gemini 2.0 Flash, Llama 3.3 70B)
on the [AgentDojo](https://github.com/ethz-spylab/agentdojo) benchmark:

| Result | Number |
|---|---|
| Episode-level detection (ROC-AUC, combined score) | 0.76–0.82 per role |
| Persistent rogue agents flagged within 20 / 50 episodes | 96% / 100% |
| Agent-level false-alarm rate | ≈1% |
| Detection compute per episode | ~0.01 ms (single core) |
| Generalization to unseen backbone (leave-one-model-out) | mean AUC 0.78 |
| Mimicry attacker: individual vs. population evidence | 0.60→0.40 vs. 0.92→0.90 |
| Covert text-only attacks (stated scope limit) | AUC 0.56 (≈chance) |

## Repository layout

```
paper/            LaTeX source, figures, compiled PDF
experiments/
  src/            pipeline (see below)
  data/           traces.parquet (extracted dataset) + agentdojo_runs.zip (raw traces)
  results/        all experiment outputs (JSON / npz)
  README.md       detailed methodology notes
```

## Running the experiments

Requirements: Python 3.11+ (developed on 3.14).

```bash
cd experiments
python3 -m venv .venv
.venv/bin/pip install numpy scikit-learn pandas matplotlib pyarrow
```

The extracted dataset (`data/traces.parquet`) is committed, so experiments run
immediately:

```bash
.venv/bin/python src/run_exp1.py        # episode-level detection + baselines + ablations
.venv/bin/python src/run_exp2_exp3.py   # fleet-scale simulation + evasion (low-and-slow, mimicry)
.venv/bin/python src/run_lomo.py        # leave-one-model-out generalization
.venv/bin/python src/make_figures.py    # regenerate paper figures -> ../paper/figures/
```

Each script prints a summary and writes results to `experiments/results/`.
Run `run_exp1.py` before the other two (they consume its outputs).

### Rebuilding the dataset from raw traces (optional)

```bash
cd experiments
unzip data/agentdojo_runs.zip           # -> agentdojo_repo/runs/... (36MB -> 155MB)
.venv/bin/python src/extract_traces.py  # -> data/traces.parquet
```

Or fetch upstream directly (full provenance): see [`experiments/README.md`](experiments/README.md).

### Compiling the paper

```bash
cd paper && tectonic main.tex
```

## Data provenance

All agent traces are the published runs of the
[AgentDojo](https://github.com/ethz-spylab/agentdojo) benchmark (Debenedetti
et al., NeurIPS 2024) — real executions by six models, with machine-checked
labels for whether each injected task was actually executed. No new model
inference was performed for this paper; the detection pipeline itself uses no
LLM.

## Citation

```bibtex
@misc{arremsetty2026fleetsentinel,
  title  = {FleetSentinel: Population-Level Behavioral Monitoring for
            Detecting Rogue LLM Agents in Large-Scale Deployments},
  author = {Arremsetty, Dheeraj},
  year   = {2026},
  note   = {Preprint},
}
```
