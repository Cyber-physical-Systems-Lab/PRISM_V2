"""
C1 — Agent Fitness (Step 1.1)

Biological fitness = long-run reproductive success.
Robotic fitness   = long-run average reward.

This is the only biologically grounded mapping we need.
Everything else follows from it.
"""

import numpy as np
from collections import deque


class AgentFitness:
    """
    Tracks long-run average reward for each agent.

    Fitness  W_i = lim_{T→∞} (1/T) Σ_{t=0}^{T} r_i(t)

    In practice: exponential moving average over a window
    long enough to capture relationship effects.
    """

    def __init__(
        self,
        agent_id: int,
        window: int = 1000,
        gamma: float = 0.99,
    ):
        self.agent_id = agent_id
        self.window = window
        self.gamma = gamma

        self.reward_history: deque = deque(maxlen=window)
        self.fitness: float = 0.0
        self.n_updates: int = 0

    def update(self, reward: float) -> None:
        """Update fitness estimate with a new per-step reward."""
        self.reward_history.append(reward)
        self.n_updates += 1

        # Unbiased after window / 10 samples warm-up
        if len(self.reward_history) >= self.window // 10:
            self.fitness = float(np.mean(self.reward_history))

    def get_fitness(self) -> float:
        return self.fitness

    def reset_window(self) -> None:
        """
        Reset for counterfactual measurement.
        Used to measure fitness WITHOUT a given relationship.
        """
        self.reward_history.clear()
        self.fitness = 0.0
        self.n_updates = 0
