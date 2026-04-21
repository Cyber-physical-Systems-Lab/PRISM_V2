"""
Symbiotic Environment Wrapper for Battery-TA-RWARE.

Wraps the heterogeneous TARWARE environment with four additions:

  1. Relationship detection  — classifies each AGV-picker pair every step
                               using observable step-level signals only.
  2. Observation augmentation — appends a (n_partners × N_REL_TYPES) EMA
                               vector + energy margin scalar to each obs.
  3. Symbiotic reward shaping — adds relationship bonus/penalty to base reward.
  4. Relationship logging     — per-episode log for emergence analysis.

See symbiosis/definitions.py for the formal relationship taxonomy (C1).
See symbiosis/reward_decomposition.py for the C2 formal decomposition.

This wrapper implements a practical, online approximation:
  - Relationship classification from observable step signals only
    (no privileged env state, no value-function counterfactuals).
  - The full C1 counterfactual measurement is implemented in
    symbiosis/counterfactual_critic.py and can be layered on top.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Tuple as GymTuple

# ──────────────────────────────────────────────────────────────────────────────
# Relationship type constants  (mirrors symbiosis/definitions.py for speed)
# ──────────────────────────────────────────────────────────────────────────────

REL_MUTUALISM    = 0   # (+/+) joint delivery, both active
REL_COMMENSALISM = 1   # (+/0) delivery while partner was charging
REL_COMPETITION  = 2   # (-/-) both charging simultaneously
REL_PARASITISM   = 3   # (+/-) asymmetric energy drain
REL_NEUTRAL      = 4   # (0/0) no significant interaction
N_REL_TYPES      = 5

REL_NAMES = ["mutualism", "commensalism", "competition", "parasitism", "neutral"]

NEG_INF = -1e9

# Package type name constants (match tarware.definitions.PackageType names).
_PKG_STANDARD    = "STANDARD"
_PKG_HEAVY       = "HEAVY"
_PKG_PICKER_SOLO = "PICKER_SOLO"


# ──────────────────────────────────────────────────────────────────────────────
# Relationship classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_relationship(
    agv_reward: float,
    picker_reward: float,
    agv_bat_delta: float,
    picker_bat_delta: float,
    agv_pkg_type: Optional[str],
    picker_in_agv_group: bool,
    charger_contested: bool,
) -> int:
    """
    Classify the biological relationship between one AGV-picker pair.

    Uses task-type and reward signals rather than raw battery proxy, enabling
    correct classification across all five relationship types:

      STANDARD delivery, both earned           → MUTUALISM     (+/+)
      HEAVY delivery, picker was nearby         → COMMENSALISM  (+/0)
      One earned while other drained no reward  → PARASITISM    (+/−)
      Both contesting same charger              → COMPETITION   (−/−)
      No significant interaction                → NEUTRAL       (0/0)
    """
    agv_earned    = agv_reward    > 0.05
    picker_earned = picker_reward > 0.05
    agv_charging    = agv_bat_delta    > 0.5
    picker_charging = picker_bat_delta > 0.5

    # Competition: both at same charger simultaneously (resource contention).
    if charger_contested:
        return REL_COMPETITION

    # Mutualism: STANDARD task — both required each other, both earned.
    if agv_pkg_type == _PKG_STANDARD and agv_earned and picker_earned and picker_in_agv_group:
        return REL_MUTUALISM

    # Commensalism: HEAVY task — AGV earned, picker helped passively (small cost, no task reward).
    if agv_pkg_type == _PKG_HEAVY and agv_earned and not picker_earned and picker_bat_delta < -0.2:
        return REL_COMMENSALISM

    # Parasitism: AGV earned while picker expended energy with no return
    # (wasted journey from task interception or charger blocking).
    if agv_earned and not picker_earned and picker_bat_delta < -1.0 and not picker_charging:
        return REL_PARASITISM

    # Parasitism (reversed): picker earned on PICKER_SOLO while AGV was delayed by
    # picker occupying a position the AGV needed (e.g. charger or aisle).
    if picker_earned and not agv_earned and agv_bat_delta < -1.0 and not agv_charging:
        return REL_PARASITISM

    # Competition: both draining energy with no reward (e.g. both chasing same task).
    if not agv_earned and not picker_earned and agv_bat_delta < -1.5 and picker_bat_delta < -0.8:
        return REL_COMPETITION

    return REL_NEUTRAL


# ──────────────────────────────────────────────────────────────────────────────
# Per-agent relationship state (EMA running mean)
# ──────────────────────────────────────────────────────────────────────────────

class RelationshipState:
    """
    Running mean of relationship-type frequencies with each partner,
    maintained as an exponential moving average.

    Shape: (n_partners, N_REL_TYPES).
    Flattened and appended to each agent's observation so the policy
    can condition on relationship history.
    """

    def __init__(self, n_partners: int, alpha: float = 0.05):
        self.n_partners = n_partners
        self.alpha = alpha
        self.ema = np.full(
            (max(n_partners, 1), N_REL_TYPES), 1.0 / N_REL_TYPES, dtype=np.float32
        )

    def update(self, partner_idx: int, rel_type: int) -> None:
        one_hot = np.zeros(N_REL_TYPES, dtype=np.float32)
        one_hot[rel_type] = 1.0
        self.ema[partner_idx] = (
            (1.0 - self.alpha) * self.ema[partner_idx] + self.alpha * one_hot
        )

    def as_feature(self) -> np.ndarray:
        if self.n_partners == 0:
            return np.zeros(0, dtype=np.float32)
        return self.ema[: self.n_partners].flatten()

    def reset(self) -> None:
        self.ema[:] = 1.0 / N_REL_TYPES

    @property
    def feature_dim(self) -> int:
        return self.n_partners * N_REL_TYPES


# ──────────────────────────────────────────────────────────────────────────────
# Symbiotic reward shaper
# ──────────────────────────────────────────────────────────────────────────────

class SymbioticRewardShaper:
    """
    Adds a relationship-type bonus/penalty on top of the base env reward.
    Agents learn which type to seek through experience, not hard-coded rules.
    """

    def __init__(
        self,
        w_mutualism:    float =  2.0,
        w_commensalism: float =  1.0,
        w_competition:  float = -1.5,
        w_parasitism:   float = -0.5,
    ):
        self._w = {
            REL_MUTUALISM:    w_mutualism,
            REL_COMMENSALISM: w_commensalism,
            REL_COMPETITION:  w_competition,
            REL_PARASITISM:   w_parasitism,
            REL_NEUTRAL:      0.0,
        }

    def shape_agv(self, base: float, rel_type: int) -> float:
        return base + self._w[rel_type]

    def shape_picker(self, base: float, rel_type: int) -> float:
        if rel_type == REL_COMMENSALISM:
            return base  # picker was absent — no bonus
        return base + self._w[rel_type]


# ──────────────────────────────────────────────────────────────────────────────
# Main wrapper
# ──────────────────────────────────────────────────────────────────────────────

class SymbioticWrapper(gym.Wrapper):
    """
    Battery-aware symbiotic wrapper for the TARWARE heterogeneous env.

    Parameters
    ----------
    env                   : unwrapped gymnasium env
    low_battery_threshold : battery level [0–100] below which only charging
                            actions are permitted (hard safety mask)
    depletion_penalty     : reward penalty when any agent battery hits 0
    reward_shaper         : SymbioticRewardShaper instance
    rel_alpha             : EMA smoothing factor for relationship history
    activity_bonus        : per-step bonus for busy agents (set 0 to disable)
    """

    def __init__(
        self,
        env: gym.Env,
        low_battery_threshold: float,
        depletion_penalty: float,
        reward_shaper: SymbioticRewardShaper,
        rel_alpha: float = 0.05,
        activity_bonus: float = 0.0,
    ):
        super().__init__(env)
        self.threshold         = low_battery_threshold
        self.depletion_penalty = depletion_penalty
        self.shaper            = reward_shaper
        self.rel_alpha         = rel_alpha
        self.activity_bonus    = activity_bonus

        self._agv_indices:    List[int] = []
        self._picker_indices: List[int] = []
        self._rel_states:     Dict[int, RelationshipState] = {}
        self._charging_ids:   List[int] = []
        self._prev_batteries: Optional[np.ndarray] = None
        self._episode_del:    int = 0

        self.orig_obs_dim_agv:    int = 0
        self.orig_obs_dim_picker: int = 0
        self._charger_dist_map: Optional[np.ndarray] = None
        self._rel_log: List[Dict] = []
        self._obs_augmented = False

    # ── Lazy initialisation ──────────────────────────────────────────────────

    def _init_agents(self) -> None:
        """Build agent metadata and augment observation spaces. Called once."""
        from tarware.warehouse import AgentType

        raw    = self.env.unwrapped
        agents = raw.agents

        self._agv_indices    = [i for i, a in enumerate(agents) if a.type == AgentType.AGV]
        self._picker_indices = [i for i, a in enumerate(agents) if a.type == AgentType.PICKER]
        n_agvs    = len(self._agv_indices)
        n_pickers = len(self._picker_indices)

        orig_spaces = list(self.env.observation_space)
        if self._agv_indices:
            self.orig_obs_dim_agv    = int(np.prod(orig_spaces[self._agv_indices[0]].shape))
        if self._picker_indices:
            self.orig_obs_dim_picker = int(np.prod(orig_spaces[self._picker_indices[0]].shape))

        for i in self._agv_indices:
            self._rel_states[i] = RelationshipState(n_pickers, self.rel_alpha)
        for i in self._picker_indices:
            self._rel_states[i] = RelationshipState(n_agvs, self.rel_alpha)

        if hasattr(raw, "charging_stations") and hasattr(raw, "action_id_to_coords_map"):
            c2id = {v: k for k, v in raw.action_id_to_coords_map.items()}
            self._charging_ids = [
                c2id[(s.y, s.x)]
                for s in raw.charging_stations
                if (s.y, s.x) in c2id
            ]

        self._charger_dist_map = self._compute_charger_dist_map()

        new_spaces = []
        for i, sp in enumerate(orig_spaces):
            rs = self._rel_states.get(i)
            if rs is not None:
                rel_ext = rs.feature_dim
                new_spaces.append(Box(
                    low  = np.concatenate([sp.low,  np.zeros(rel_ext, dtype=np.float32), [-1.0]]),
                    high = np.concatenate([sp.high, np.ones(rel_ext,  dtype=np.float32), [ 1.0]]),
                    dtype=np.float32,
                ))
            else:
                new_spaces.append(sp)
        self.observation_space = GymTuple(new_spaces)
        self._obs_augmented = True

    def _compute_charger_dist_map(self) -> np.ndarray:
        """BFS from all charger positions over the static grid."""
        raw = self.env.unwrapped
        grid_h, grid_w = raw.grid_size
        obstacles = (raw.grid[2] != 0)
        dist = np.full((grid_h, grid_w), np.inf, dtype=np.float32)
        queue: deque = deque()
        for cs in raw.charging_stations:
            if 0 <= cs.y < grid_h and 0 <= cs.x < grid_w:
                dist[cs.y, cs.x] = 0.0
                queue.append((cs.y, cs.x))
        while queue:
            cy, cx = queue.popleft()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < grid_h and 0 <= nx < grid_w:
                    if not obstacles[ny, nx] and dist[ny, nx] == np.inf:
                        dist[ny, nx] = dist[cy, cx] + 1.0
                        queue.append((ny, nx))
        dist[dist == np.inf] = float(grid_h * grid_w)
        return dist

    def _energy_margin(self, agent_idx: int) -> float:
        """Scalar ∈ [-1, 1] indicating how safely the agent can continue."""
        raw = self.env.unwrapped
        agent = raw.agents[agent_idx]
        battery = float(agent.battery)
        dist = float(self._charger_dist_map[int(agent.y), int(agent.x)])
        margin = (battery - dist) / 100.0
        return float(np.clip(margin, -1.0, 1.0))

    # ── Public gym interface ─────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not self._obs_augmented:
            self._init_agents()
        for rs in self._rel_states.values():
            rs.reset()
        self._prev_batteries = self._get_batteries()
        self._episode_del = 0
        return self._augment_obs(obs), info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        curr_bat = self._get_batteries()
        bat_delta = curr_bat - self._prev_batteries
        self._prev_batteries = curr_bat

        rewards = list(rewards)
        step_del = sum(rewards)

        # Build a set of (agv_idx, picker_idx) pairs that share a contested charger.
        contested_pairs: set = set()
        for ai, aj in info.get("contested_chargers", []):
            # Map absolute agent indices to (agv, picker) pair if applicable.
            for agv_i in self._agv_indices:
                for picker_i in self._picker_indices:
                    if set([ai, aj]) == set([agv_i, picker_i]):
                        contested_pairs.add((agv_i, picker_i))

        # Per-agent package type currently being carried.
        carrying = info.get("carrying_pkg_type", [None] * len(rewards))

        for agv_local, agv_idx in enumerate(self._agv_indices):
            for picker_local, picker_idx in enumerate(self._picker_indices):
                agv_pkg = carrying[agv_idx]

                # Determine whether this picker was part of the AGV's carrier group
                # for a delivery this step (direct participation signal).
                picker_in_group = any(
                    picker_idx in grp and agv_idx in grp
                    for grp in info.get("delivery_carrier_groups", [])
                )

                rel = classify_relationship(
                    agv_reward=float(rewards[agv_idx]),
                    picker_reward=float(rewards[picker_idx]),
                    agv_bat_delta=float(bat_delta[agv_idx]),
                    picker_bat_delta=float(bat_delta[picker_idx]),
                    agv_pkg_type=agv_pkg,
                    picker_in_agv_group=picker_in_group,
                    charger_contested=(agv_idx, picker_idx) in contested_pairs,
                )
                self._rel_states[agv_idx].update(picker_local, rel)
                self._rel_states[picker_idx].update(agv_local, rel)
                self._rel_log.append({
                    "agv": agv_idx, "picker": picker_idx, "rel": rel,
                })

                rewards[agv_idx]    = self.shaper.shape_agv(rewards[agv_idx], rel)
                rewards[picker_idx] = self.shaper.shape_picker(rewards[picker_idx], rel)

        if self.activity_bonus > 0:
            for i, busy in enumerate(info.get("vehicles_busy", [])):
                if busy and i < len(rewards):
                    rewards[i] += self.activity_bonus

        # Safety: depletion penalty
        for i, bat in enumerate(curr_bat):
            if bat <= 0.0:
                rewards[i] -= self.depletion_penalty

        self._episode_del += int(round(step_del))
        return self._augment_obs(obs), tuple(rewards), terminated, truncated, info

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_batteries(self) -> np.ndarray:
        agents = self.env.unwrapped.agents
        return np.array([a.battery for a in agents], dtype=np.float32)

    def _augment_obs(self, obs) -> tuple:
        augmented = []
        for i, o in enumerate(obs):
            rs = self._rel_states.get(i)
            if rs is not None:
                margin = self._energy_margin(i)
                augmented.append(
                    np.concatenate([o.flatten(), rs.as_feature(), [margin]])
                )
            else:
                augmented.append(o)
        return tuple(augmented)

    def episode_rel_distribution(self, window_start: int = 0) -> Dict[str, float]:
        """Fraction of each relationship type in the current episode."""
        log = self._rel_log[window_start:]
        if not log:
            return {n: 0.0 for n in REL_NAMES}
        counts = np.zeros(N_REL_TYPES, dtype=int)
        for e in log:
            counts[e["rel"]] += 1
        total = counts.sum()
        return {REL_NAMES[i]: counts[i] / total for i in range(N_REL_TYPES)}
