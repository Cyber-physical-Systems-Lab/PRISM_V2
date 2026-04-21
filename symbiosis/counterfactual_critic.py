"""
C1 — Counterfactual Value Functions (Step 1.3)

The key technical requirement for C1: we need BOTH
  V_i(s ; π_joint)      — standard joint critic
  V_i(s ; π_without_j)  — counterfactual (partner zeroed)

This is related to but distinct from COMA (Foerster et al. 2018):
  COMA removes agent ACTIONS in the advantage baseline.
  We remove agent PRESENCE from observations to operationalise
  the biological "without relationship" counterfactual.
"""

import torch
import torch.nn as nn


class JointCritic(nn.Module):
    """
    Centralised critic.  Takes ALL agent observations.

    V_i(s) = estimated value to agent i given full state.
    Output shape: [batch, n_agents]
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_agents  = n_agents
        self.obs_dim   = obs_dim

        self.net = nn.Sequential(
            nn.Linear(n_agents * obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),
        )

    def forward(self, all_obs: torch.Tensor) -> torch.Tensor:
        """all_obs: [batch, n_agents, obs_dim] → [batch, n_agents]"""
        return self.net(all_obs.flatten(start_dim=1))

    def get_value(self, all_obs: torch.Tensor, agent_id: int) -> float:
        return self.forward(all_obs)[:, agent_id].mean().item()


class MarginalCritic(nn.Module):
    """
    Counterfactual critic: value of agent i when agent j is REMOVED
    from the state by zeroing their observation slot.

    Operationalises  W_i^∅  in the C1 definition.

    Training procedure: random agent dropout so the network learns
    marginal values for all possible agent subsets, ensuring reliable
    counterfactual estimates at evaluation time.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.n_agents     = n_agents
        self.obs_dim      = obs_dim
        self.dropout_rate = dropout_rate

        self.net = nn.Sequential(
            nn.Linear(n_agents * obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),
        )

    def forward(
        self,
        all_obs: torch.Tensor,
        exclude_agent: int | None = None,
    ) -> torch.Tensor:
        """
        If exclude_agent is set, zeros out that agent's observation
        before computing values — the counterfactual operation.
        """
        obs = all_obs.clone()
        if exclude_agent is not None:
            obs[:, exclude_agent, :] = 0.0
        return self.net(obs.flatten(start_dim=1))

    def get_value(
        self,
        all_obs: torch.Tensor,
        agent_id: int,
        exclude_agent: int | None = None,
    ) -> float:
        return self.forward(all_obs, exclude_agent)[:, agent_id].mean().item()

    def train_with_dropout(
        self,
        all_obs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training step with random agent dropout.
        Teaches the network marginal values for all agent subsets.
        """
        obs = all_obs.clone()
        for idx in range(self.n_agents):
            if torch.rand(1).item() < self.dropout_rate:
                obs[:, idx, :] = 0.0
        return nn.MSELoss()(self.forward(obs), targets)
