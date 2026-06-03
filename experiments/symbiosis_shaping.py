"""Shared PBRS helper for PRISM symbiotic shaping.

Single source of truth for:
1. The shuffled phi-table used by the `prism_shuffled` ablation
   (per-episode permutation of PHI_FULL, reproducible from a seed).
2. The event-grounded per-agent potential function `Phi_i(s)` that turns
   PRISM's typed bonus into proper potential-based reward shaping
   `gamma * Phi_i(s') - Phi_i(s)` (Devlin & Kudenko 2011 NE-preservation).

The raw and pbrs shaping forms are dispatched at the runner level via the
`--shaping_form {raw, pbrs}` flag; this module only provides the building
blocks. The `raw` code path in each runner is left untouched so existing
results stay byte-identical.
"""

from __future__ import annotations

import numpy as np

# Relationship constants. Mirror the values in run_heterogeneous.py;
# importing from there would create a circular dependency, so the
# canonical definitions live here.
REL_MUTUALISM = 0
REL_COMMENSALISM = 1
REL_COMPETITION = 2
REL_PARASITISM = 3
REL_NEUTRAL = 4
N_REL = 5

REL_NAMES = ["mutualism", "commensalism", "competition", "parasitism", "neutral"]

# Default phi tables.
PHI_FULL = [2.0, 1.0, -1.5, -0.5, 0.0]
PHI_POS = [2.0, 1.0, 0.0, 0.0, 0.0]
PHI_NEG = [0.0, 0.0, -1.5, -0.5, 0.0]


def make_phi_table(form: str, seed: int | None = None) -> list[float]:
    """Return a phi-table for the given shaping form.

    form ∈ {"full", "positive", "negative", "shuffled"}.

    "shuffled" returns a deterministic permutation of PHI_FULL keyed by
    `seed`. The same seed always returns the same permutation, which lets
    callers refresh once per episode (seed = run_seed + episode_idx) and
    keep the assignment stable across the episode.
    """
    if form == "full":
        return list(PHI_FULL)
    if form == "positive":
        return list(PHI_POS)
    if form == "negative":
        return list(PHI_NEG)
    if form == "shuffled":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N_REL)
        return [float(PHI_FULL[i]) for i in perm]
    raise ValueError(f"unknown phi-table form: {form!r}")


def event_rel_type(event: dict) -> int:
    """Map a delivery_events / return_events dict to a REL_* constant.

    The classification is based on participant composition:
      - inter-type cooperation (>=1 carrier AND >=1 picker) -> MUTUALISM
      - single-type completion (carriers only or pickers only) -> NEUTRAL
    Package type (SOLO / STANDARD / LARGE) is informational; the typing
    signal that matters for symbiosis is whether both agent classes
    participated, not how many were involved.
    """
    carriers = event.get("carrier_ids", []) or []
    pickers = event.get("picker_ids", []) or []
    if carriers and pickers:
        return REL_MUTUALISM
    return REL_NEUTRAL


class PerAgentPotential:
    """Cumulative typed-cooperation counter rendered as a scalar potential.

    Phi_i(s) = sum over events t<=time(s) of phi(rel(event_t)) / n_pairs,
    summed over events in which agent i participated.

    The PBRS theorem requires Phi to be a function of state. We approximate
    "state" with the cumulative event sequence -- two state-trajectories
    that produce the same `info["delivery_events"]` and
    `info["return_events"]` history will have identical Phi. Since these
    events are themselves deterministic functions of the env state in
    TARWARE (see warehouse.py:923-983, 1064-1127), Phi is a valid
    potential under the standard PBRS framework.
    """

    def __init__(self, n_agents: int):
        self.n_agents = int(n_agents)
        self.phi = np.zeros(self.n_agents, dtype=np.float32)

    def reset(self) -> None:
        self.phi.fill(0.0)

    def update_from_events(
        self,
        info: dict,
        phi_table: list[float],
        n_pairs: int,
    ) -> np.ndarray:
        """Apply the events in `info` to the running potential.

        For every delivery or return event, the participating carriers and
        pickers each receive `phi_table[rel_type] / n_pairs` added to
        their cumulative potential. Returns a snapshot of `self.phi` AFTER
        the update so callers can compute `gamma * new - prev`.
        """
        n_pairs = max(1, int(n_pairs))
        events = []
        events.extend(info.get("delivery_events") or [])
        events.extend(info.get("return_events") or [])
        for ev in events:
            rel = event_rel_type(ev)
            increment = float(phi_table[rel]) / n_pairs
            for aid in ev.get("carrier_ids", []) or []:
                if 0 <= int(aid) < self.n_agents:
                    self.phi[int(aid)] += increment
            for pid in ev.get("picker_ids", []) or []:
                if 0 <= int(pid) < self.n_agents:
                    self.phi[int(pid)] += increment
        return self.phi.copy()


def pbrs_term(
    prev_phi: np.ndarray,
    new_phi: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Standard PBRS shaping term F(s, a, s') = gamma * Phi(s') - Phi(s).

    Returns a per-agent vector that callers add to the per-agent task
    reward. Under Devlin & Kudenko 2011 Theorem 2 this preserves the set
    of Nash-optimal joint policies in the underlying Markov game.
    """
    return float(gamma) * np.asarray(new_phi, dtype=np.float32) - np.asarray(
        prev_phi, dtype=np.float32
    )


__all__ = [
    "REL_MUTUALISM",
    "REL_COMMENSALISM",
    "REL_COMPETITION",
    "REL_PARASITISM",
    "REL_NEUTRAL",
    "N_REL",
    "REL_NAMES",
    "PHI_FULL",
    "PHI_POS",
    "PHI_NEG",
    "make_phi_table",
    "event_rel_type",
    "PerAgentPotential",
    "pbrs_term",
]
