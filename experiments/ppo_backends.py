"""
Shared PPO training backends for all experiment runners.

Three algorithm variants:
  ippo   — per-agent independent PPO: one separate policy per agent ID,
            no parameter sharing at all (not even type-level sharing)
  hetppo — type-shared decentralised PPO: one policy per agent TYPE
            (π_AGV, π_Picker) with local critics
  mappo  — type-shared centralised PPO: same actors as HetPPO but critics
            see the full global state (concatenation of all agent observations)

Usage
-----
    from experiments.ppo_backends import run_training

    result = run_training(env_id, method, backend, seed, args)
    # result keys: deliveries_curve, mutualism_curve, tsi, rsi,
    #              grad_norms, r_task_history, r_sym_history, boundedness_ratio
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import tarware  # noqa: F401

from analysis.metrics import SymbiosisEmergenceTracker, compute_rsi, compute_tsi
from tarware.definitions import AgentType

# ── Relationship constants ────────────────────────────────────────────────────
REL_MUTUALISM = 0
REL_COMMENSALISM = 1
REL_COMPETITION = 2
REL_PARASITISM = 3
REL_NEUTRAL = 4
REL_NAMES = ["mutualism", "commensalism", "competition", "parasitism", "neutral"]

# ── Per-pair relationship confidence tracker ──────────────────────────────────

class RelTracker:
    """
    Per-(AGV, Picker) pair EMA of relationship-type probability.

    Updated with the shaping label at each step so that the confidence
    value gates the magnitude of φ.  For the symbiotic method, correct
    labels concentrate the EMA on the dominant relationship type
    (conf ≈ 0.7–0.9), keeping φ near full strength.  Any method that
    uses wrong or random labels would dilute the EMA toward uniform
    (conf ≈ 0.2 per type), slashing effective shaping ~5×.
    """

    def __init__(self, alpha: float = 0.05):
        self.ema   = np.full(len(REL_NAMES), 1.0 / len(REL_NAMES), dtype=np.float32)
        self.alpha = alpha

    def update(self, rel: int) -> None:
        one_hot      = np.zeros(len(REL_NAMES), dtype=np.float32)
        one_hot[rel] = 1.0
        self.ema     = (1.0 - self.alpha) * self.ema + self.alpha * one_hot

    def confidence(self, rel: int) -> float:
        return float(self.ema[rel])


ROLE_IDLE = 0
ROLE_CHARGING = 1
ROLE_TASKING = 2
ROLE_DELIVERING = 3
NUM_ROLE_BUCKETS = 4


# ── Neural network ────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── PPO agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    PPO actor-critic with GAE-λ advantages.

    The critic input dimension (critic_dim) can differ from the actor
    observation dimension (obs_dim):
    - local critic  → critic_dim == obs_dim
    - central critic → critic_dim == global state dim

    Gradient norms are recorded after clip_grad_norm_ in every mini-batch
    update and accumulated in self.grad_norms for convergence diagnostics.
    """

    def __init__(
        self,
        obs_dim: int,
        critic_dim: int,
        act_dim: int,
        hidden_dim: int,
        lr: float,
        device: torch.device,
    ):
        self.device = device
        self.actor = MLP(obs_dim, act_dim, hidden_dim).to(device)
        self.critic = MLP(critic_dim, 1, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )
        self.grad_norms: list[float] = []
        self.clear()

    def clear(self) -> None:
        self.obs: list = []
        self.critic_obs: list = []
        self.next_critic_obs: list = []
        self.actions: list = []
        self.logps: list = []
        self.values: list = []
        self.rewards: list = []
        self.dones: list = []
        self.masks: list = []

    @torch.no_grad()
    def act(self, obs: np.ndarray, critic_obs: np.ndarray,
            mask: Optional[np.ndarray] = None) -> tuple[int, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        crit_t = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(obs_t)
        if mask is not None:
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            logits = logits.masked_fill(~mask_t, float('-inf'))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        value = self.critic(crit_t).squeeze(-1)
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def store(
        self,
        obs: np.ndarray,
        critic_obs: np.ndarray,
        next_critic_obs: np.ndarray,
        action: int,
        logp: float,
        value: float,
        reward: float,
        done: float,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.critic_obs.append(np.asarray(critic_obs, dtype=np.float32))
        self.next_critic_obs.append(np.asarray(next_critic_obs, dtype=np.float32))
        self.actions.append(int(action))
        self.logps.append(float(logp))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.masks.append(np.asarray(mask, dtype=np.float32) if mask is not None else None)

    def update(
        self,
        gamma: float,
        lam: float,
        clip: float,
        entropy_coef: float,
        epochs: int,
        mini_batches: int,
    ) -> None:
        n = len(self.actions)
        if not n:
            return

        obs_t = torch.as_tensor(np.asarray(self.obs), dtype=torch.float32, device=self.device)
        crit_t = torch.as_tensor(np.asarray(self.critic_obs), dtype=torch.float32, device=self.device)
        next_crit_t = torch.as_tensor(np.asarray(self.next_critic_obs), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.asarray(self.actions), dtype=torch.int64, device=self.device)
        old_logp_t = torch.as_tensor(np.asarray(self.logps), dtype=torch.float32, device=self.device)

        values = np.asarray(self.values, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        with torch.no_grad():
            next_values = self.critic(next_crit_t).squeeze(-1).cpu().numpy()

        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + gamma * (1.0 - dones[t]) * next_values[t] - values[t]
            gae = delta + gamma * lam * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        batch_size = n
        mini_batch_size = max(32, batch_size // max(1, mini_batches))
        params = list(self.actor.parameters()) + list(self.critic.parameters())

        use_masks = self.masks[0] is not None
        masks_t = (
            torch.as_tensor(np.asarray(self.masks), dtype=torch.bool, device=self.device)
            if use_masks else None
        )

        for _ in range(epochs):
            perm = np.random.permutation(batch_size)
            for start in range(0, batch_size, mini_batch_size):
                idx = perm[start : start + mini_batch_size]
                idx_t = torch.as_tensor(idx, dtype=torch.int64, device=self.device)

                raw_logits = self.actor(obs_t[idx_t])
                if use_masks:
                    raw_logits = raw_logits.masked_fill(~masks_t[idx_t], float('-inf'))
                dist = torch.distributions.Categorical(logits=raw_logits)
                new_logp = dist.log_prob(act_t[idx_t])
                ratio = (new_logp - old_logp_t[idx_t]).exp()
                s1 = ratio * adv_t[idx_t]
                s2 = ratio.clamp(1.0 - clip, 1.0 + clip) * adv_t[idx_t]
                actor_loss = -torch.min(s1, s2).mean()

                vp = self.critic(crit_t[idx_t]).squeeze(-1)
                value_loss = ((vp - ret_t[idx_t]) ** 2).mean()

                loss = actor_loss + 0.5 * value_loss - entropy_coef * dist.entropy().mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)

                # Record post-clip gradient norm for convergence diagnostics
                total_norm = float(
                    sum(p.grad.detach().norm() ** 2 for p in params if p.grad is not None) ** 0.5
                )
                self.grad_norms.append(total_norm)

                self.optimizer.step()

        self.clear()


# ── Relationship classification & reward shaping ──────────────────────────────

def classify_rel(
    agv_delivery: float,
    picker_lift: float,
    agv_bat_delta: float,
    picker_bat_delta: float,
) -> int:
    """Return one of REL_* constants for a single (AGV, picker) pair.

    agv_delivery : raw reward to the AGV  (> 0.5  → delivery occurred)
    picker_lift  : raw reward to the Picker (>= 0.05 → lift-assist reward fired)

    Cooperation happens at TOGGLE_LOAD time (picker gets +0.1 lift-assist),
    not at goal-station delivery time. Classifying purely on delivery assigns
    false credit to all pickers regardless of which one helped.
    """
    agv_charging = agv_bat_delta > 0.5
    picker_charging = picker_bat_delta > 0.5
    delivered = agv_delivery > 0.5
    # >= 0.05: STANDARD load signal is exactly 0.05 * reward_scale (= 0.05 for STANDARD)
    picker_just_lifted = picker_lift >= 0.05

    if delivered or picker_just_lifted:
        return REL_MUTUALISM
    if agv_charging and picker_charging:
        return REL_NEUTRAL
    if agv_charging or picker_charging:
        return REL_COMMENSALISM
    return REL_NEUTRAL


def shape_rewards(
    raw_rewards: list,
    agv_idx: list,
    pick_idx: list,
    bat_d: np.ndarray,
    method: str,
    args,
    trackers: Optional[dict] = None,
) -> tuple[list, list]:
    """
    Apply method-specific reward shaping.

    Returns (shaped_rewards, rels) where rels is the list of relationship
    classifications for all (AGV, picker) pairs this step.

    trackers : optional dict mapping (ai, pi) → RelTracker, used by the
               symbiotic method to apply confidence-weighted φ.  When
               provided, the tracker is updated with the classified label
               and the effective weight is φ × conf.  Pass None (or omit)
               to use flat φ (backward-compatible behaviour).

    When pick_idx is empty (AGV-only team):
    - symbiotic  → identical to individual (no pairs → no shaping)
    - unclassified → identical to individual (no pairs → no shaping)
    - team        → mean reward shared across all AGVs
    """
    rewards = list(raw_rewards)
    n_pairs = max(1, len(agv_idx) * len(pick_idx))
    rels: list[int] = []

    for ai in agv_idx:
        for pi in pick_idx:
            rel = classify_rel(raw_rewards[ai], raw_rewards[pi], bat_d[ai], bat_d[pi])
            rels.append(rel)
            if method == "symbiotic":
                if trackers is not None:
                    trackers[(ai, pi)].update(rel)
                    conf = trackers[(ai, pi)].confidence(rel)
                else:
                    conf = 1.0
                w_table = {
                    REL_MUTUALISM: args.w_mutualism,
                    REL_COMMENSALISM: args.w_commensalism,
                    REL_COMPETITION: args.w_competition,
                    REL_PARASITISM: args.w_parasitism,
                    REL_NEUTRAL: 0.0,
                }
                # Normalise by n_pairs so total shaping stays ~O(w) regardless of fleet size
                w = w_table[rel] * conf / n_pairs
                rewards[ai] += w
                if rel == REL_MUTUALISM:
                    # Picker bonus when it actively lifted this step.
                    # >= 0.05 because the STANDARD load signal is exactly 0.05.
                    if raw_rewards[pi] >= 0.05:
                        rewards[pi] += w
                elif rel != REL_COMMENSALISM:
                    rewards[pi] += w
            elif method == "unclassified":
                rewards[ai] += args.unclassified_bonus / n_pairs
                rewards[pi] += args.unclassified_bonus / n_pairs

    if method == "team":
        mean_r = float(np.mean(rewards))
        rewards = [mean_r] * len(rewards)

    return rewards, rels


# ── Utility helpers ───────────────────────────────────────────────────────────

def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def reset_observations(env: gym.Env, seed: Optional[int] = None) -> list:
    out = env.reset(seed=seed)
    if (
        isinstance(out, tuple)
        and len(out) == 2
        and isinstance(out[1], dict)
        and isinstance(out[0], (list, tuple))
    ):
        return list(out[0])
    return list(out)


def global_state(obs_list: list) -> np.ndarray:
    return np.concatenate(
        [np.asarray(o, dtype=np.float32).reshape(-1) for o in obs_list], axis=0
    )


def infer_role(step_reward: float, bat_delta: float, busy: bool, charging: bool) -> int:
    if step_reward > 0.5:
        return ROLE_DELIVERING
    if charging or bat_delta > 0.5:
        return ROLE_CHARGING
    if busy:
        return ROLE_TASKING
    return ROLE_IDLE


# ── Agent builders ────────────────────────────────────────────────────────────

def build_ippo(
    n_agents: int,
    obs_dims: list[int],
    act_dim: int,
    hidden_dim: int,
    lr: float,
    device: torch.device,
) -> list[PPOAgent]:
    """
    True IPPO: one fully independent policy per agent ID.
    No parameter sharing — not even across agents of the same type.
    Each agent's critic uses only its own local observation.
    Returns a list of N PPOAgent instances indexed by agent ID.
    """
    return [
        PPOAgent(obs_dims[i], obs_dims[i], act_dim, hidden_dim, lr, device)
        for i in range(n_agents)
    ]


def build_hetppo(
    agv_obs_dim: int,
    pick_obs_dim: int,
    act_dim: int,
    hidden_dim: int,
    lr: float,
    device: torch.device,
) -> tuple[PPOAgent, PPOAgent]:
    """
    HetPPO: type-shared decentralised — one policy per agent TYPE, local critics.
    All AGVs share agv_agent; all pickers share pick_agent.
    Equivalent to the "ippo" backend in run_symbiotic.py / run_flat_cooperative.py.
    """
    agv_agent = PPOAgent(agv_obs_dim, agv_obs_dim, act_dim, hidden_dim, lr, device)
    pick_agent = PPOAgent(pick_obs_dim, pick_obs_dim, act_dim, hidden_dim, lr, device)
    return agv_agent, pick_agent


def build_mappo(
    agv_obs_dim: int,
    pick_obs_dim: int,
    state_dim: int,
    act_dim: int,
    hidden_dim: int,
    lr: float,
    device: torch.device,
) -> tuple[PPOAgent, PPOAgent]:
    """
    MAPPO: type-shared centralised — one policy per agent TYPE, centralised critics.
    Critics see the full global state (concatenation of all agent observations).
    Equivalent to the "mappo" backend in run_symbiotic.py / run_flat_cooperative.py.
    """
    agv_agent = PPOAgent(agv_obs_dim, state_dim, act_dim, hidden_dim, lr, device)
    pick_agent = PPOAgent(pick_obs_dim, state_dim, act_dim, hidden_dim, lr, device)
    return agv_agent, pick_agent


# ── Core training function ────────────────────────────────────────────────────

def run_training(
    env_id: str,
    method: str,
    backend: str,
    seed: int,
    args,
) -> dict:
    """
    Run one (env, method, backend, seed) training trial.

    Parameters
    ----------
    env_id   : gymnasium environment ID
    method   : "individual" | "team" | "unclassified" | "symbiotic"
    backend  : "ippo" | "hetppo" | "mappo"
    seed     : integer random seed
    args     : namespace with training hyper-parameters (see CLI defaults below)

    Required args attributes
    ------------------------
    timesteps, lr, hidden_dim, device
    Optional (with defaults): rollout/rollout_steps (256), ppo_epochs/epochs (3),
    mini_batches (4), gamma (0.99), lam (0.95), clip (0.2), entropy_coef (0.01),
    max_ep_steps (500), max_inactivity_steps (None), log_interval (10),
    unclassified_bonus (0.5), w_mutualism (2.0), w_commensalism (1.0),
    w_competition (-1.5), w_parasitism (-0.5)

    Returns
    -------
    dict with keys: backend, method, seed, env, deliveries_curve, mutualism_curve,
                    battery_mean_curve_per_episode, battery_end_curve_per_episode,
                    tsi, rsi, grad_norms, r_task_history, r_sym_history,
                    boundedness_ratio
    """
    device = select_device(getattr(args, "device", "auto"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(
        env_id,
        max_inactivity_steps=getattr(args, "max_inactivity_steps", None),
        max_steps=getattr(args, "max_ep_steps", 500),
    )

    try:
        raw = env.unwrapped
        obs = reset_observations(env, seed=seed)
        agents_meta = raw.agents
        n_agents = len(obs)
        agv_idx = [i for i, a in enumerate(agents_meta) if a.type == AgentType.AGV]
        pick_idx = [i for i, a in enumerate(agents_meta) if a.type == AgentType.PICKER]

        act_dim = raw.action_space.spaces[0].n
        obs_dims = [int(np.asarray(obs[i]).shape[0]) for i in range(n_agents)]
        agv_obs_dim = obs_dims[agv_idx[0]] if agv_idx else obs_dims[0]
        pick_obs_dim = obs_dims[pick_idx[0]] if pick_idx else agv_obs_dim
        state_dim = sum(obs_dims)

        hidden_dim = getattr(args, "hidden_dim", 128)
        lr = args.lr

        # Build policies
        if backend == "ippo":
            policies = build_ippo(n_agents, obs_dims, act_dim, hidden_dim, lr, device)
            agv_agent = pick_agent = None
            is_ippo = True
            centralized = False
        elif backend == "hetppo":
            agv_agent, pick_agent = build_hetppo(agv_obs_dim, pick_obs_dim, act_dim, hidden_dim, lr, device)
            policies = None
            is_ippo = False
            centralized = False
        elif backend == "mappo":
            agv_agent, pick_agent = build_mappo(agv_obs_dim, pick_obs_dim, state_dim, act_dim, hidden_dim, lr, device)
            policies = None
            is_ippo = False
            centralized = True
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Choose from: ippo, hetppo, mappo")

        # Per-pair confidence tracker for symbiotic method.
        # Trackers persist across episodes (no reset) so the EMA concentrates
        # on the true dominant relationship type over the full training run.
        pair_trackers = (
            {(ai, pi): RelTracker() for ai in agv_idx for pi in pick_idx}
            if method == "symbiotic" else None
        )

        # Training hyper-parameters (with sensible defaults)
        rollout = getattr(args, "rollout", getattr(args, "rollout_steps", 256))
        mini_batches = getattr(args, "mini_batches", 4)
        gamma = getattr(args, "gamma", 0.99)
        lam = getattr(args, "lam", 0.95)
        clip = getattr(args, "clip", 0.2)
        entropy_coef = getattr(args, "entropy_coef", 0.01)
        epochs = getattr(args, "ppo_epochs", getattr(args, "epochs", 3))
        log_interval = getattr(args, "log_interval", 10)

        prev_bat = np.array([a.battery for a in raw.agents], dtype=np.float32)
        rel_tracker = SymbiosisEmergenceTracker(window=max(1, log_interval))
        mutualism_curve: list[float] = []
        deliveries_curve: list[int] = []
        battery_mean_curve_per_episode: list[list[float]] = []
        battery_end_curve_per_episode: list[list[float]] = []
        role_counts = np.zeros((n_agents, NUM_ROLE_BUCKETS), dtype=np.float32)
        ep_role_counts = np.zeros((n_agents, NUM_ROLE_BUCKETS), dtype=np.float32)
        dominant_role_history: list[list[int]] = []

        r_task_history: list[float] = []
        r_sym_history: list[float] = []

        ep_deliveries = 0
        ep_rel_counts: dict[int, int] = defaultdict(int)
        ep_battery_sum = np.zeros(n_agents, dtype=np.float64)
        ep_battery_count = 0
        total_steps = 0
        rollout_steps = 0

        while total_steps < args.timesteps:
            state = global_state(obs)
            actions, logps, values = [], [], []

            valid_masks = raw.compute_valid_action_masks(pickers_to_agvs=True)

            for i in range(n_agents):
                local_obs = np.asarray(obs[i], dtype=np.float32)
                agent_mask = valid_masks[i]
                if is_ippo:
                    agent = policies[i]
                    critic_obs = local_obs
                else:
                    agent = agv_agent if i in agv_idx else pick_agent
                    critic_obs = state if centralized else local_obs
                action, logp, value = agent.act(local_obs, critic_obs, mask=agent_mask)
                actions.append(action)
                logps.append(logp)
                values.append(value)

            next_obs_raw, raw_rewards, dones, truncs, info = env.step(actions)
            next_obs = list(next_obs_raw)

            curr_bat = np.array([a.battery for a in raw.agents], dtype=np.float32)
            bat_d = curr_bat - prev_bat
            prev_bat = curr_bat

            ep_battery_sum += curr_bat
            ep_battery_count += 1

            shaped_rewards, rels = shape_rewards(
                list(raw_rewards), agv_idx, pick_idx, bat_d, method, args,
                trackers=pair_trackers,
            )

            # Track reward decomposition
            r_task_step = float(np.mean(list(raw_rewards)))
            r_sym_step = float(np.mean([s - r for s, r in zip(shaped_rewards, raw_rewards)]))
            r_task_history.append(r_task_step)
            r_sym_history.append(r_sym_step)

            next_state = global_state(next_obs)
            done_any = float(any(dones) or any(truncs))

            for i in range(n_agents):
                local_obs = np.asarray(obs[i], dtype=np.float32)
                next_local_obs = np.asarray(next_obs[i], dtype=np.float32)
                if is_ippo:
                    agent = policies[i]
                    critic_obs = local_obs
                    next_critic_obs = next_local_obs
                else:
                    agent = agv_agent if i in agv_idx else pick_agent
                    critic_obs = state if centralized else local_obs
                    next_critic_obs = next_state if centralized else next_local_obs
                agent.store(
                    obs=local_obs,
                    critic_obs=critic_obs,
                    next_critic_obs=next_critic_obs,
                    action=actions[i],
                    logp=logps[i],
                    value=values[i],
                    reward=shaped_rewards[i],
                    done=done_any,
                    mask=valid_masks[i],
                )

            busy_flags = info.get("vehicles_busy", [False] * n_agents)
            charging_flags = info.get("charging_states", [False] * n_agents)
            for i in range(n_agents):
                role = infer_role(raw_rewards[i], bat_d[i], bool(busy_flags[i]), bool(charging_flags[i]))
                role_counts[i, role] += 1.0
                ep_role_counts[i, role] += 1.0

            ep_deliveries += int(info.get("shelf_deliveries", 0))
            for rel in rels:
                ep_rel_counts[rel] += 1

            obs = next_obs
            total_steps += 1
            rollout_steps += 1

            if rollout_steps >= rollout:
                _do_update(is_ippo, policies, agv_agent, pick_agent, pick_idx,
                           gamma, lam, clip, entropy_coef, epochs, mini_batches)
                rollout_steps = 0

            if done_any:
                deliveries_curve.append(ep_deliveries)
                if ep_battery_count > 0:
                    battery_mean_curve_per_episode.append(
                        (ep_battery_sum / ep_battery_count).astype(float).tolist()
                    )
                else:
                    battery_mean_curve_per_episode.append(curr_bat.astype(float).tolist())
                battery_end_curve_per_episode.append(curr_bat.astype(float).tolist())
                total_rel = max(1, sum(ep_rel_counts.values()))
                rel_dist = {REL_NAMES[k]: ep_rel_counts[k] / total_rel for k in range(len(REL_NAMES))}
                rel_tracker.log_episode(rel_dist)
                mutualism_curve.append(rel_tracker.mutualism_fraction)
                dominant_role_history.append(np.argmax(ep_role_counts, axis=1).tolist())
                ep_deliveries = 0
                ep_rel_counts = defaultdict(int)
                ep_role_counts.fill(0.0)
                ep_battery_sum.fill(0.0)
                ep_battery_count = 0
                obs = reset_observations(env, seed=None)
                prev_bat = np.array([a.battery for a in raw.agents], dtype=np.float32)

        if rollout_steps:
            _do_update(is_ippo, policies, agv_agent, pick_agent, pick_idx,
                       gamma, lam, clip, entropy_coef, epochs, mini_batches)

        # Collect gradient norms from all policies
        if is_ippo:
            all_grad_norms: list[float] = []
            for p in policies:
                all_grad_norms.extend(p.grad_norms)
        else:
            all_grad_norms = list(agv_agent.grad_norms)
            if pick_idx:
                all_grad_norms.extend(pick_agent.grad_norms)

        tsi = float(compute_tsi(role_counts)) if np.any(role_counts) else 0.0
        rsi = float(compute_rsi(dominant_role_history)) if dominant_role_history else 0.0

        max_r_sym = max((abs(x) for x in r_sym_history), default=0.0)
        max_r_task = max((abs(x) for x in r_task_history), default=1.0)
        boundedness_ratio = float(max_r_sym / (max_r_task + 1e-8))

        # Confidence diagnostic: avg peak EMA value per pair.
        # symbiotic → ~0.70–0.90; methods without trackers → ~0.20 (uniform prior).
        if pair_trackers:
            all_ema = np.stack([t.ema for t in pair_trackers.values()])
            mean_confidence = float(all_ema.max(axis=1).mean())
        else:
            mean_confidence = 1.0 / len(REL_NAMES)

        return {
            "backend": backend,
            "method": method,
            "seed": seed,
            "env": env_id,
            "deliveries_curve": deliveries_curve,
            "mutualism_curve": mutualism_curve,
            "battery_mean_curve_per_episode": battery_mean_curve_per_episode,
            "battery_end_curve_per_episode": battery_end_curve_per_episode,
            "tsi": tsi,
            "rsi": rsi,
            "grad_norms": all_grad_norms,
            "r_task_history": r_task_history,
            "r_sym_history": r_sym_history,
            "boundedness_ratio": boundedness_ratio,
            "mean_confidence": mean_confidence,
        }

    finally:
        env.close()


def _do_update(
    is_ippo: bool,
    policies,
    agv_agent,
    pick_agent,
    pick_idx: list,
    gamma, lam, clip, entropy_coef, epochs, mini_batches,
) -> None:
    """Call update() on whichever set of agents is active."""
    if is_ippo:
        for p in policies:
            p.update(gamma, lam, clip, entropy_coef, epochs, mini_batches)
    else:
        agv_agent.update(gamma, lam, clip, entropy_coef, epochs, mini_batches)
        if pick_idx:
            pick_agent.update(gamma, lam, clip, entropy_coef, epochs, mini_batches)


# ── Result aggregation ────────────────────────────────────────────────────────

def aggregate_seeds(seed_results: list[dict], convergence_threshold: float = 0.95) -> dict:
    """
    Aggregate per-seed training results into per-method statistics.

    convergence_threshold : fraction of final performance at which an episode
                            is counted as "converged"
    """
    curves = [r["deliveries_curve"] for r in seed_results if r.get("deliveries_curve")]
    min_len = min(len(c) for c in curves) if curves else 0
    trimmed = [c[:min_len] for c in curves]
    arr = np.array(trimmed, dtype=float) if trimmed else np.zeros((1, 1))

    # Final performance = mean over last 10% of episodes
    tail = max(1, arr.shape[1] // 10)
    final_means = arr[:, -tail:].mean(axis=1)  # per-seed final perf
    grand_mean = float(final_means.mean())
    grand_std = float(final_means.std())

    # Convergence episode: first ep where delivery ≥ threshold × final mean
    convergence_eps = []
    for row, final in zip(trimmed, final_means):
        thresh = convergence_threshold * max(final, 1e-6)
        ep = next((i for i, v in enumerate(row) if v >= thresh), len(row))
        convergence_eps.append(ep)

    # Gradient norm stats (aggregate across all seeds)
    all_gn = []
    for r in seed_results:
        all_gn.extend(r.get("grad_norms", []))

    # Boundedness
    boundedness_list = [r.get("boundedness_ratio", 0.0) for r in seed_results]

    # Confidence (symbiotic only; other methods return uniform prior ~0.20)
    confidence_list = [r.get("mean_confidence", 1.0 / len(REL_NAMES)) for r in seed_results]

    battery_mean_eps = [r.get("battery_mean_curve_per_episode", []) for r in seed_results]
    battery_end_eps = [r.get("battery_end_curve_per_episode", []) for r in seed_results]
    all_battery_means = [ep for seed_curves in battery_mean_eps for ep in seed_curves if ep]
    all_battery_ends = [ep for seed_curves in battery_end_eps for ep in seed_curves if ep]
    if all_battery_means:
        battery_mean_per_agent = np.asarray(all_battery_means, dtype=float).mean(axis=0).tolist()
    else:
        battery_mean_per_agent = []
    if all_battery_ends:
        battery_end_mean_per_agent = np.asarray(all_battery_ends, dtype=float).mean(axis=0).tolist()
    else:
        battery_end_mean_per_agent = []

    return {
        "backend": seed_results[0].get("backend", "unknown") if seed_results else "unknown",
        "mean_completion": grand_mean,
        "std_completion": grand_std,
        "deliveries_curves": [r["deliveries_curve"] for r in seed_results],
        "mutualism_curves": [r.get("mutualism_curve", []) for r in seed_results],
        "battery_mean_curves_per_seed": battery_mean_eps,
        "battery_end_curves_per_seed": battery_end_eps,
        "battery_mean_per_agent": battery_mean_per_agent,
        "battery_end_mean_per_agent": battery_end_mean_per_agent,
        "tsi": float(np.mean([r.get("tsi", 0.0) for r in seed_results])),
        "rsi": float(np.mean([r.get("rsi", 0.0) for r in seed_results])),
        "convergence_episodes": int(np.mean(convergence_eps)),
        "grad_norm_mean": float(np.mean(all_gn)) if all_gn else 0.0,
        "grad_norm_final": float(np.mean(all_gn[-max(1, len(all_gn) // 10):])) if all_gn else 0.0,
        "grad_norm_history": all_gn,
        "boundedness_ratio_mean": float(np.mean(boundedness_list)),
        "boundedness_bounded": all(np.isfinite(b) for b in boundedness_list),
        "mean_confidence": float(np.mean(confidence_list)),
        "n_seeds": len(seed_results),
    }
