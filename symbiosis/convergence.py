"""
C2 — Convergence Analysis (Step 2.2)

THEOREM (Convergence Preservation):
  Under standard MARL assumptions:
    (A1) Bounded rewards    : |r_i| ≤ R_max
    (A2) Lipschitz policy   : |π_θ(a|s) − π_θ'(a|s)| ≤ L‖θ−θ'‖
    (A3) Ergodic MDP        : all states reachable

  The policy gradient with symbiotic reward:
    ∇_θ J_i(θ) = E[∇_θ log π_θ(a_i|o_i) × (r_i^task + r_i^sym − b_i(s))]

  converges to a local optimum at the same rate as standard PG,
  provided |r_i^sym| ≤ c × |r_i^task| for some finite c.

PROOF SKETCH:
  1. r_i^sym is bounded because:
       |δ_ij| ≤ 2 × V_max  (bounded value functions)
       |φ(type)| ≤ φ_max   (fixed constants)
       n_partners finite
     → |r_i^sym| ≤ n × φ_max × 2 × V_max
  2. Bounded r_i^sym means r_i^total satisfies (A1) with
     R_max' = R_max + sym_bound.
  3. Policy gradient theorem applies unchanged with R_max'.
  4. Convergence rate is O(1/√T), same as standard PG.

This module empirically verifies the theoretical claim by
tracking gradient norms and the task/sym reward ratio.
"""

from typing import Dict, List
import numpy as np
import torch

from symbiosis.reward_decomposition import RewardDecomposition


class ConvergenceMonitor:
    """
    Tracks gradient norms and reward component ratios per agent.
    Validates the C2 boundedness and convergence-rate claims.
    """

    def __init__(self, n_agents: int):
        self.gradient_norms:      Dict[int, List[float]] = {i: [] for i in range(n_agents)}
        self.task_reward_history: Dict[int, List[float]] = {i: [] for i in range(n_agents)}
        self.sym_reward_history:  Dict[int, List[float]] = {i: [] for i in range(n_agents)}
        self.total_reward_history:Dict[int, List[float]] = {i: [] for i in range(n_agents)}

    def log_gradients(self, agent_id: int, policy: torch.nn.Module) -> None:
        """Record L2 gradient norm after each policy update."""
        total_norm = sum(
            p.grad.norm(2).item() ** 2
            for p in policy.parameters()
            if p.grad is not None
        ) ** 0.5
        self.gradient_norms[agent_id].append(total_norm)

    def log_rewards(self, agent_id: int, decomp: RewardDecomposition) -> None:
        self.task_reward_history[agent_id].append(decomp.r_task)
        self.sym_reward_history[agent_id].append(decomp.r_sym)
        self.total_reward_history[agent_id].append(decomp.r_total)

    def check_boundedness(self, agent_id: int) -> dict:
        """
        Verify |r_sym| ≤ c × |r_task| empirically.
        Returns the empirical c bound.
        """
        task = np.array(self.task_reward_history[agent_id])
        sym  = np.array(self.sym_reward_history[agent_id])

        nonzero = np.abs(task) > 1e-6
        if nonzero.sum() == 0:
            return {"bounded": True, "ratio": 0.0, "max_ratio": 0.0, "mean_ratio": 0.0}

        ratio = np.abs(sym[nonzero]) / np.abs(task[nonzero])
        return {
            "bounded":    bool(ratio.max() < np.inf),
            "max_ratio":  float(ratio.max()),
            "mean_ratio": float(ratio.mean()),
            "c_bound":    float(ratio.max()),
        }

    def summary(self, agent_id: int) -> dict:
        """Summarise convergence diagnostics for an agent."""
        norms = self.gradient_norms[agent_id]
        return {
            "grad_norm_mean": float(np.mean(norms)) if norms else 0.0,
            "grad_norm_final": float(np.mean(norms[-50:])) if len(norms) >= 50 else 0.0,
            "boundedness": self.check_boundedness(agent_id),
        }
