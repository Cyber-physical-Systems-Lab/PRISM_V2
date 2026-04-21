"""
Post-experiment metrics for C3 evaluation.

PRIMARY METRICS:
  TSI  — Team Specialization Index: how distinct are agent roles?
  RSI  — Role Stability Index: how consistent are roles across episodes?
  Mutualism fraction: fraction of AGV-picker interactions classified as (+/+).
  Task completion rate: deliveries per episode.
  Convergence episode: first episode where perf ≥ threshold × final perf.

COMPARISON PREDICTION (from x.md):
  Heterogeneous: symbiotic > all baselines on TSI, mutualism, task completion.
  Homogeneous  : symbiotic ≈ team ≈ unclassified (no advantage from classification).
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Sequence


# ──────────────────────────────────────────────────────────────────────────────
# Team Specialization Index
# ──────────────────────────────────────────────────────────────────────────────

def compute_tsi(role_counts: np.ndarray, eps: float = 1e-8) -> float:
    """
    Team Specialization Index ∈ [0, 1].

    role_counts : (n_agents, n_roles) array of role-action counts per agent.
    Returns the mean max-fraction across agents:
      TSI = (1/N) Σ_i  max_r(count_{i,r}) / Σ_r count_{i,r}

    TSI = 1 → every agent always does exactly one role (full specialization).
    TSI = 1/n_roles → every agent distributes uniformly (no specialization).
    """
    if role_counts.ndim != 2:
        raise ValueError("role_counts must be (n_agents, n_roles)")
    totals = role_counts.sum(axis=1, keepdims=True) + eps
    fractions = role_counts / totals
    return float(fractions.max(axis=1).mean())


# ──────────────────────────────────────────────────────────────────────────────
# Role Stability Index
# ──────────────────────────────────────────────────────────────────────────────

def compute_rsi(dominant_role_history: List[List[int]]) -> float:
    """
    Role Stability Index ∈ [0, 1].

    dominant_role_history : list of length n_episodes, each element is
                            a list of length n_agents with the dominant role
                            (argmax of role_counts) for that episode.

    RSI = fraction of consecutive-episode pairs where all agents' dominant
          roles remain the same.

    RSI = 1 → roles never change across episodes.
    RSI = 0 → roles change every episode.
    """
    if len(dominant_role_history) < 2:
        return 1.0
    stable = 0
    for ep in range(1, len(dominant_role_history)):
        if dominant_role_history[ep] == dominant_role_history[ep - 1]:
            stable += 1
    return stable / (len(dominant_role_history) - 1)


# ──────────────────────────────────────────────────────────────────────────────
# Symbiosis emergence tracker
# ──────────────────────────────────────────────────────────────────────────────

class SymbiosisEmergenceTracker:
    """
    Accumulates per-episode relationship distributions and computes
    emergence metrics over a sliding window of episodes.

    Used for the C3 mutualism-fraction curve.
    """

    REL_NAMES = ["mutualism", "commensalism", "competition", "parasitism", "neutral"]
    N_TYPES   = 5

    def __init__(self, window: int = 50):
        self.window = window
        self._episode_dists: List[Dict[str, float]] = []

    def log_episode(self, dist: Dict[str, float]) -> None:
        """Log relationship distribution for one completed episode."""
        self._episode_dists.append(dist)

    def compute_emergence_metrics(self) -> Dict[str, float]:
        """
        Compute windowed mean of relationship fractions.
        Returns dict with keys: mutualism, commensalism, competition,
                                parasitism, neutral, n_episodes_logged.
        """
        recent = self._episode_dists[-self.window :]
        if not recent:
            return {n: 0.0 for n in self.REL_NAMES} | {"n_episodes_logged": 0}

        out: Dict[str, float] = {}
        for name in self.REL_NAMES:
            out[name] = float(np.mean([d.get(name, 0.0) for d in recent]))
        out["n_episodes_logged"] = len(self._episode_dists)
        return out

    @property
    def mutualism_fraction(self) -> float:
        m = self.compute_emergence_metrics()
        return m.get("mutualism", 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Convergence helper
# ──────────────────────────────────────────────────────────────────────────────

def find_convergence_episode(
    reward_curve: Sequence[float],
    threshold: float = 0.95,
    tail: int = 100,
) -> int:
    """
    Return the first episode where performance reaches
    threshold × (mean of last `tail` episodes).

    Measures convergence speed for the C2 claim:
      "same O(1/√T) convergence rate as standard PG".
    """
    arr   = np.asarray(reward_curve, dtype=float)
    final = float(arr[-tail:].mean()) if len(arr) >= tail else float(arr.mean())
    target = threshold * final
    for i, r in enumerate(arr):
        if r >= target:
            return i
    return len(arr)
