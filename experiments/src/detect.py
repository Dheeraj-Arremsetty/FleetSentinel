"""Behavioral Sentinel detection pipeline.

Terminology matches the paper:
  role        = AgentDojo suite (banking/slack/travel/workspace)
  fingerprint = smoothed tool-usage distribution P estimated from benign
                episodes of a role, pooled across the model fleet
  S_ind       = 1 - exp(-KL(Q || P))  in [0, 1)
  S_pop       = population outlier score from LOF fit on a fleet snapshot
  S           = w * S_ind + (1 - w) * S_pop
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

EPS = 1e-9


def split_tools(s):
    return s.split() if isinstance(s, str) and s else []


def bigrams(seq):
    return set(zip(seq[:-1], seq[1:]))


class RoleFingerprint:
    """Fingerprint + feature extractor for one role, fit on benign episodes."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha  # Laplace smoothing pseudo-count

    def fit(self, benign_tool_seqs):
        vocab = sorted({t for seq in benign_tool_seqs for t in seq})
        self.vocab = vocab
        self.index = {t: i for i, t in enumerate(vocab)}
        counts = np.zeros(len(vocab) + 1)  # last slot = out-of-vocabulary bucket
        for seq in benign_tool_seqs:
            for t in seq:
                counts[self.index[t]] += 1
        self.P = (counts + self.alpha) / (counts + self.alpha).sum()
        self.known_bigrams = set()
        for seq in benign_tool_seqs:
            self.known_bigrams |= bigrams(seq)
        return self

    def q_dist(self, seq):
        counts = np.zeros(len(self.vocab) + 1)
        for t in seq:
            counts[self.index.get(t, len(self.vocab))] += 1
        if counts.sum() == 0:
            # An episode with zero tool calls: undefined distribution; treat as
            # uniform so KL reflects deviation from the role profile.
            counts[:] = 1.0
        return (counts + EPS) / (counts + EPS).sum()

    def s_ind(self, seq):
        q = self.q_dist(seq)
        kl = float(np.sum(q * np.log(q / self.P)))
        return 1.0 - np.exp(-max(kl, 0.0))

    def features(self, seq, n_errors=0):
        """Population-space feature vector for one episode."""
        q = self.q_dist(seq)
        oov = sum(1 for t in seq if t not in self.index)
        bg = bigrams(seq)
        novel_bg = len(bg - self.known_bigrams) / len(bg) if bg else 0.0
        extras = np.array([
            np.log1p(len(seq)),
            np.log1p(len(set(seq))),
            np.log1p(oov),
            np.log1p(n_errors),
            novel_bg,
        ])
        return np.concatenate([q, extras])


def s_pop_snapshot(X, n_neighbors=35):
    """LOF outlier score for every member of one fleet snapshot.

    Fit on the snapshot itself (unsupervised): rogue behavior is detectable
    because it is *rare in the population*, not because labels exist.
    Returns scores in [0, 1] via 1 - exp(-(max(LOF - 1, 0))).
    """
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Xs = (X - mu) / np.where(sd < EPS, 1.0, sd)
    k = min(n_neighbors, len(X) - 1)
    lof = LocalOutlierFactor(n_neighbors=k)
    lof.fit(Xs)
    raw = -lof.negative_outlier_factor_  # ~1 for inliers, >1 for outliers
    return 1.0 - np.exp(-np.maximum(raw - 1.0, 0.0))


def allowlist_score(seq, index):
    """Static-rule baseline: number of tools outside the role allowlist."""
    return float(sum(1 for t in seq if t not in index))


def _norm_val(v):
    return " ".join(str(v).lower().split())[:200]


class TaskFingerprint:
    """Task-conditioned fingerprint over an argument-aware action alphabet.

    An action symbol is the tool name, suffixed with "#novel" when any of its
    argument values was never observed in the benign profile runs for the same
    (role, assigned task). This keeps S_ind a KL divergence between action
    distributions (as in the paper) while letting the alphabet expose
    argument-level deviations: an injected call either uses a tool the task
    never needs, or a known tool with attacker-supplied argument values --
    both map to symbols with near-zero fingerprint probability.
    """

    def __init__(self, alpha=0.5, id_diversity_max=0.5, id_min_count=3):
        self.alpha = alpha
        self.id_diversity_max = id_diversity_max
        self.id_min_count = id_min_count
        self.task_tools = {}    # task -> set of tool names
        self.task_vals = {}     # task -> set of (tool, key, normalized value)
        self.task_counts = {}   # task -> {symbol: count}
        self.role_tools = set()
        self.role_vals = set()
        self._key_occ = {}      # (tool, key) -> occurrence count in benign runs
        self._key_distinct = {} # (tool, key) -> set of distinct values
        self.id_keys = set()    # identifier-like (tool, key) pairs

    @staticmethod
    def _iter_calls(calls):
        for c in calls:
            yield c["f"], c.get("a", {})

    def fit(self, benign_frames):
        """benign_frames: iterable of (task_id, calls list) benign profile runs."""
        for task, calls in benign_frames:
            tt = self.task_tools.setdefault(task, set())
            tv = self.task_vals.setdefault(task, set())
            tc = self.task_counts.setdefault(task, {})
            for f, a in self._iter_calls(calls):
                tt.add(f)
                self.role_tools.add(f)
                for k, v in a.items():
                    nv = _norm_val(v)
                    tv.add((f, k, nv))
                    self.role_vals.add((f, k, nv))
                    self._key_occ[(f, k)] = self._key_occ.get((f, k), 0) + 1
                    self._key_distinct.setdefault((f, k), set()).add(nv)
                tc[f] = tc.get(f, 0) + 1
        # Identifier-like fields: recur across benign runs with few distinct
        # values (recipients, ids, cities...). Free-text fields (subjects,
        # message bodies) vary per run and are excluded from novelty checks.
        for key, n in self._key_occ.items():
            if n >= self.id_min_count and len(self._key_distinct[key]) / n <= self.id_diversity_max:
                self.id_keys.add(key)
        return self

    def symbolize(self, task, calls):
        """Map an episode to action symbols using the task profile (falling
        back to the role profile for tasks with no benign profile runs)."""
        tv = self.task_vals.get(task)
        tt = self.task_tools.get(task)
        use_role = tv is None or len(tv) == 0
        vals = self.role_vals if use_role else tv
        tools = self.role_tools if use_role else tt
        syms = []
        for f, a in self._iter_calls(calls):
            novel_args = any(
                (f, k) in self.id_keys
                and (f, k, _norm_val(v)) not in vals
                and (f, k, _norm_val(v)) not in self.role_vals
                for k, v in a.items()
            )
            unknown_tool = f not in tools
            if unknown_tool or novel_args:
                syms.append(f + "#novel")
            else:
                syms.append(f)
        return syms

    def s_ind(self, task, calls):
        """1 - exp(-KL(Q || P_task)) over the argument-aware alphabet."""
        syms = self.symbolize(task, calls)
        counts = self.task_counts.get(task) or {}
        if not counts:
            # no profile for this task: pool role-level counts
            counts = {}
            for tc in self.task_counts.values():
                for f, n in tc.items():
                    counts[f] = counts.get(f, 0) + n
        alphabet = sorted(set(counts) | set(syms))
        if not alphabet:
            return 1.0
        p = np.array([counts.get(s, 0.0) for s in alphabet]) + self.alpha
        # "#novel" symbols never occur in benign profiles; smoothing gives them
        # small but nonzero mass, so KL stays finite and large.
        p = p / p.sum()
        if not syms:
            q = np.ones(len(alphabet))
        else:
            q = np.array([float(sum(1 for x in syms if x == s)) for s in alphabet])
        q = (q + EPS) / (q + EPS).sum()
        kl = float(np.sum(q * np.log(q / p)))
        return 1.0 - np.exp(-max(kl, 0.0))

    def novel_action_stats(self, task, calls):
        syms = self.symbolize(task, calls)
        if not syms:
            return 0.0, 0
        n_novel = sum(1 for s in syms if s.endswith("#novel"))
        return n_novel / len(syms), n_novel
