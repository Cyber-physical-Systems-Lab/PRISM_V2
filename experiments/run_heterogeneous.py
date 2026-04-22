"""
C3 Experiment 1 — heterogeneous training runner.

Heterogeneous environment: picker agents retrieve packages, AGV agents deliver.
Task completion requires both types -> structural capability complementarity.

This runner prefers a local MAPPO implementation (centralized critics shared by
agent type). If that path fails, it falls back to local IPPO. HARL backends are
accepted as fallback targets but require an external installation that is not
part of this repository.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency at runtime
    SummaryWriter = None

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import tarware  # noqa: F401

from analysis.metrics import SymbiosisEmergenceTracker, compute_rsi, compute_tsi
from tarware.definitions import AgentType


REL_MUTUALISM = 0
REL_COMMENSALISM = 1
REL_COMPETITION = 2
REL_PARASITISM = 3
REL_NEUTRAL = 4
REL_NAMES = ["mutualism", "commensalism", "competition", "parasitism", "neutral"]

ROLE_IDLE = 0
ROLE_CHARGING = 1
ROLE_TASKING = 2
ROLE_DELIVERING = 3
NUM_ROLE_BUCKETS = 4


parser = argparse.ArgumentParser(
    description="C3 Experiment 1: Heterogeneous env, all methods",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
METHODS = ["individual", "team", "unclassified", "symbiotic"]
BACKENDS_ALL = ["ippo", "hetppo", "mappo"]
BACKENDS_DEFAULT = ["ippo", "mappo"]

parser.add_argument("--config", default=None, help="Optional YAML config file.")
parser.add_argument("--env", default="tarware-small-4agvs-2pickers-partialobs-chg-v1")
parser.add_argument("--backends", nargs="+", default=None, choices=BACKENDS_ALL,
                    help="Backends to run (homogeneous-style multi-backend mode).")
parser.add_argument("--backend", default="auto", choices=["auto", "mappo", "ippo", "hetppo"],
                    help="Deprecated single-backend selector; kept for compatibility.")
parser.add_argument("--timesteps", default=1_000_000, type=int)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Explicit seeds to run (e.g., --seeds 0 1 2)")
parser.add_argument("--rollout_steps", default=1024, type=int)
parser.add_argument("--epochs", default=4, type=int)
parser.add_argument("--mini_batches", default=8, type=int)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--lam", default=0.95, type=float)
parser.add_argument("--clip", default=0.2, type=float)
parser.add_argument("--entropy_coef", default=0.01, type=float)
parser.add_argument("--hidden_dim", default=128, type=int)
parser.add_argument("--output", default="results/hetero_results.json")
parser.add_argument("--checkpoint_dir", default="runs/c3_hetero")
parser.add_argument("--tb_logdir", default="runs/tensorboard/c3_hetero")
parser.add_argument("--checkpoint_every_episodes", default=100, type=int)
parser.add_argument("--log_interval", default=10, type=int)
parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
parser.add_argument("--max_ep_steps", default=5000, type=int)
parser.add_argument("--max_inactivity_steps", default=None, type=int)
parser.add_argument(
    "--package_distribution",
    default=None,
    type=str,
    help='Optional JSON dict overriding env package_distribution, e.g. \'{"SOLO":0.3,"STANDARD":0.4,"LARGE":0.3}\'',
)
parser.add_argument("--unclassified_bonus", default=0.5, type=float)
parser.add_argument("--w_mutualism", default=2.0, type=float)
parser.add_argument("--w_commensalism", default=1.0, type=float)
parser.add_argument("--w_competition", default=-1.5, type=float)
parser.add_argument("--w_parasitism", default=-0.5, type=float)
parser.add_argument("--low_battery_threshold", default=10.0, type=float)
parser.add_argument("--depletion_penalty", default=5.0, type=float)


def _load_config_defaults(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    defaults = {}

    experiment = config.get("experiment", {})
    env = config.get("env", {})
    training = config.get("training", {})
    safety = config.get("safety", {})
    symbiotic = config.get("symbiotic", {})
    logging = config.get("logging", {})

    if "n_seeds" in experiment:
        defaults["seeds"] = experiment["n_seeds"]
    if "id" in env:
        defaults["env"] = env["id"]
    if "max_steps" in env:
        defaults["max_ep_steps"] = env["max_steps"]
    if "max_inactivity_steps" in env:
        defaults["max_inactivity_steps"] = env["max_inactivity_steps"]
    if "package_distribution" in env:
        # Serialise as a JSON string so argparse can pass it through (string-typed).
        defaults["package_distribution"] = json.dumps(env["package_distribution"])

    defaults.update({
        key: training[key]
        for key in ("timesteps", "rollout_steps", "epochs", "mini_batches", "lr", "entropy_coef")
        if key in training
    })
    defaults.update({
        key: safety[key]
        for key in ("low_battery_threshold", "depletion_penalty")
        if key in safety
    })
    defaults.update({
        key: symbiotic[key]
        for key in ("w_mutualism", "w_commensalism", "w_competition", "w_parasitism")
        if key in symbiotic
    })
    defaults.update({
        key: logging[key]
        for key in ("output", "checkpoint_dir", "tb_logdir", "checkpoint_every_episodes", "log_interval")
        if key in logging
    })
    return defaults


def classify_rel(
    agv_delivery: float,
    picker_lift: float,
    agv_bat_delta: float,
    picker_bat_delta: float,
) -> int:
    """Classify the (AGV, Picker) relationship for this timestep.

    agv_delivery : raw reward to the AGV  (> 0.5  → delivery occurred)
    picker_lift  : raw reward to the Picker (> 0.05 → lift-assist reward fired)

    The key insight: actual cooperation happens at TOGGLE_LOAD time (when the
    picker gets its +0.1 lift-assist reward), not at goal-station delivery time.
    Classifying at delivery time assigns false credit to all pickers regardless
    of which one helped.
    """
    agv_charging = agv_bat_delta > 0.5
    picker_charging = picker_bat_delta > 0.5
    delivered = agv_delivery > 0.5
    # Picker got its +0.1 load-assist reward this step → direct cooperation event
    picker_just_lifted = picker_lift > 0.05

    if delivered or picker_just_lifted:
        return REL_MUTUALISM

    if agv_charging and picker_charging:
        return REL_NEUTRAL

    if agv_charging or picker_charging:
        return REL_COMMENSALISM

    return REL_NEUTRAL


def shape_rewards(raw_rewards: list, agv_idx: list, pick_idx: list, bat_d: np.ndarray, method: str, args) -> tuple[list, list]:
    rewards = list(raw_rewards)
    n_pairs = max(1, len(agv_idx) * len(pick_idx))
    rels = []
    for ai in agv_idx:
        for pi in pick_idx:
            rel = classify_rel(raw_rewards[ai], raw_rewards[pi], bat_d[ai], bat_d[pi])
            rels.append(rel)
            if method == "symbiotic":
                w_table = {
                    REL_MUTUALISM: args.w_mutualism,
                    REL_COMMENSALISM: args.w_commensalism,
                    REL_COMPETITION: args.w_competition,
                    REL_PARASITISM: args.w_parasitism,
                    REL_NEUTRAL: 0.0,
                }
                # Normalise by number of pairs so total shaping stays ~O(w)
                # regardless of fleet size.
                w_val = w_table[rel] / n_pairs
                rewards[ai] += w_val
                if rel == REL_MUTUALISM:
                    # Picker bonus only when it actively lifted this step.
                    # Prevents free-rider effect: pickers must help to share credit.
                    if raw_rewards[pi] > 0.05:
                        rewards[pi] += w_val
                elif rel != REL_COMMENSALISM:
                    rewards[pi] += w_val
            elif method == "unclassified":
                rewards[ai] += args.unclassified_bonus / n_pairs
                rewards[pi] += args.unclassified_bonus / n_pairs

    if method == "team":
        team_reward = float(np.mean(rewards))
        rewards = [team_reward] * len(rewards)

    return rewards, rels


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _reset_observations(env: gym.Env, seed: Optional[int] = None) -> list:
    reset_output = env.reset(seed=seed)
    if (
        isinstance(reset_output, tuple)
        and len(reset_output) == 2
        and isinstance(reset_output[1], dict)
        and isinstance(reset_output[0], (list, tuple))
    ):
        return list(reset_output[0])
    return list(reset_output)


def _global_state(obs_list: list) -> np.ndarray:
    return np.concatenate([np.asarray(obs, dtype=np.float32).reshape(-1) for obs in obs_list], axis=0)


def _infer_role(step_reward: float, bat_delta: float, busy: bool, charging: bool) -> int:
    if step_reward > 0.5:
        return ROLE_DELIVERING
    if charging or bat_delta > 0.5:
        return ROLE_CHARGING
    if busy:
        return ROLE_TASKING
    return ROLE_IDLE


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


class TypePPOAgent:
    def __init__(self, obs_dim: int, critic_dim: int, act_dim: int, hidden_dim: int, lr: float, device: torch.device):
        self.device = device
        self.actor = MLP(obs_dim, act_dim, hidden_dim).to(device)
        self.critic = MLP(critic_dim, 1, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )
        self.clear()

    def clear(self) -> None:
        self.obs = []
        self.critic_obs = []
        self.next_critic_obs = []
        self.actions = []
        self.logps = []
        self.values = []
        self.rewards = []
        self.dones = []

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        critic_obs: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> tuple[int, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        critic_t = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(obs_t)
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=np.float32).reshape(-1)
            if mask_arr.shape[0] == logits.shape[-1] and np.any(mask_arr > 0.0):
                mask_t = torch.as_tensor(mask_arr, dtype=torch.bool, device=self.device).unsqueeze(0)
                logits = logits.masked_fill(~mask_t, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        value = self.critic(critic_t).squeeze(-1)
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

    def update(self, gamma: float, lam: float, clip: float, entropy_coef: float, epochs: int, mini_batches: int) -> Dict[str, float]:
        n = len(self.actions)
        if not n:
            return {
                "actor_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
            }

        obs_t = torch.as_tensor(np.asarray(self.obs), dtype=torch.float32, device=self.device)
        critic_t = torch.as_tensor(np.asarray(self.critic_obs), dtype=torch.float32, device=self.device)
        next_critic_t = torch.as_tensor(np.asarray(self.next_critic_obs), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.asarray(self.actions), dtype=torch.int64, device=self.device)
        old_logp_t = torch.as_tensor(np.asarray(self.logps), dtype=torch.float32, device=self.device)

        values = np.asarray(self.values, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        with torch.no_grad():
            next_values = self.critic(next_critic_t).squeeze(-1).cpu().numpy()

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
        actor_losses = []
        value_losses = []
        entropies = []
        approx_kls = []

        for _ in range(epochs):
            permutation = np.random.permutation(batch_size)
            for start in range(0, batch_size, mini_batch_size):
                idx = permutation[start:start + mini_batch_size]
                idx_t = torch.as_tensor(idx, dtype=torch.int64, device=self.device)

                dist = torch.distributions.Categorical(logits=self.actor(obs_t[idx_t]))
                new_logp = dist.log_prob(act_t[idx_t])
                ratio = (new_logp - old_logp_t[idx_t]).exp()
                surrogate_1 = ratio * adv_t[idx_t]
                surrogate_2 = ratio.clamp(1.0 - clip, 1.0 + clip) * adv_t[idx_t]
                actor_loss = -torch.min(surrogate_1, surrogate_2).mean()

                value_pred = self.critic(critic_t[idx_t]).squeeze(-1)
                value_loss = ((value_pred - ret_t[idx_t]) ** 2).mean()
                entropy = dist.entropy().mean()
                approx_kl = (old_logp_t[idx_t] - new_logp).mean().item()

                loss = actor_loss + 0.5 * value_loss - entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    0.5,
                )
                self.optimizer.step()

                actor_losses.append(float(actor_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                approx_kls.append(float(approx_kl))

        self.clear()
        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        }


def _train_local_ppo_backend(method: str, seed: int, args, backend: str) -> dict:
    centralized = backend == "mappo"
    device = _select_device(args.device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_kwargs = {
        "max_inactivity_steps": args.max_inactivity_steps,
        "max_steps": args.max_ep_steps,
    }
    if args.package_distribution:
        env_kwargs["package_distribution"] = json.loads(args.package_distribution)
    env = gym.make(args.env, **env_kwargs)

    try:
        raw = env.unwrapped
        obs = _reset_observations(env, seed=seed)
        agents = raw.agents
        n_agents = len(obs)
        agv_idx = [i for i, a in enumerate(agents) if a.type == AgentType.AGV]
        pick_idx = [i for i, a in enumerate(agents) if a.type == AgentType.PICKER]

        act_dim = raw.action_space.spaces[0].n
        agv_obs_dim = int(np.asarray(obs[agv_idx[0]]).shape[0])
        pick_obs_dim = int(np.asarray(obs[pick_idx[0]]).shape[0])
        state_dim = int(sum(np.asarray(o).shape[0] for o in obs))

        critic_dim_agv = state_dim if centralized else agv_obs_dim
        critic_dim_picker = state_dim if centralized else pick_obs_dim

        agv_agent = TypePPOAgent(agv_obs_dim, critic_dim_agv, act_dim, args.hidden_dim, args.lr, device)
        pick_agent = TypePPOAgent(pick_obs_dim, critic_dim_picker, act_dim, args.hidden_dim, args.lr, device)

        run_dir = Path(args.checkpoint_dir) / method / f"{backend}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        tb_writer = None
        if SummaryWriter is not None and args.tb_logdir:
            tb_dir = Path(args.tb_logdir) / method / f"{backend}_seed{seed}"
            tb_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_dir))

        prev_bat = np.array([a.battery for a in raw.agents], dtype=np.float32)
        rel_tracker = SymbiosisEmergenceTracker(window=max(1, args.log_interval))
        mutualism_curve = []
        deliveries_curve = []
        role_counts = np.zeros((n_agents, NUM_ROLE_BUCKETS), dtype=np.float32)
        episode_role_counts = np.zeros((n_agents, NUM_ROLE_BUCKETS), dtype=np.float32)
        dominant_role_history = []

        ep_deliveries = 0
        ep_pkg_deliveries: Dict[str, int] = {}
        ep_rel_counts = defaultdict(int)
        ep_battery_sum = np.zeros(n_agents, dtype=np.float64)
        ep_battery_count = 0
        total_steps = 0
        rollout_steps = 0
        episode_rows = []
        update_rows = []
        best_deliveries = float("-inf")
        best_episode = -1

        def _save_checkpoint(name: str, episode_idx: int, is_best: bool) -> None:
            payload: Dict[str, Any] = {
                "method": method,
                "seed": seed,
                "backend": backend,
                "env": args.env,
                "total_steps": total_steps,
                "episode": episode_idx,
                "is_best": is_best,
                "best_deliveries": best_deliveries,
                "agv_actor_state_dict": agv_agent.actor.state_dict(),
                "agv_critic_state_dict": agv_agent.critic.state_dict(),
                "pick_actor_state_dict": pick_agent.actor.state_dict(),
                "pick_critic_state_dict": pick_agent.critic.state_dict(),
                "optimizer_agv_pick_state_dict": agv_agent.optimizer.state_dict(),
                "optimizer_pick_state_dict": pick_agent.optimizer.state_dict(),
            }
            torch.save(payload, run_dir / name)

        while total_steps < args.timesteps:
            state = _global_state(obs)
            actions = []
            logps = []
            values = []

            valid_masks = raw.compute_valid_action_masks(pickers_to_agvs=True)

            for i in range(n_agents):
                local_obs = np.asarray(obs[i], dtype=np.float32)
                critic_obs = state if centralized else local_obs
                agent = agv_agent if i in agv_idx else pick_agent
                action, logp, value = agent.act(local_obs, critic_obs, mask=valid_masks[i])
                actions.append(action)
                logps.append(logp)
                values.append(value)

            next_obs, raw_rewards, dones, truncs, info = env.step(actions)
            next_obs = list(next_obs)

            curr_bat = np.array([a.battery for a in raw.agents], dtype=np.float32)
            bat_d = curr_bat - prev_bat
            prev_bat = curr_bat
            ep_battery_sum += curr_bat
            ep_battery_count += 1

            shaped_rewards, rels = shape_rewards(list(raw_rewards), agv_idx, pick_idx, bat_d, method, args)

            next_state = _global_state(next_obs)
            done_any = float(any(dones) or any(truncs))

            for i in range(n_agents):
                local_obs = np.asarray(obs[i], dtype=np.float32)
                next_local_obs = np.asarray(next_obs[i], dtype=np.float32)
                critic_obs = state if centralized else local_obs
                next_critic_obs = next_state if centralized else next_local_obs
                agent = agv_agent if i in agv_idx else pick_agent
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
                role = _infer_role(raw_rewards[i], bat_d[i], bool(busy_flags[i]), bool(charging_flags[i]))
                role_counts[i, role] += 1.0
                episode_role_counts[i, role] += 1.0

            ep_deliveries += int(info.get("shelf_deliveries", 0))
            for pkg_name, cnt in info.get("deliveries_by_pkg_type", {}).items():
                ep_pkg_deliveries[pkg_name] = ep_pkg_deliveries.get(pkg_name, 0) + int(cnt)
            for rel in rels:
                ep_rel_counts[rel] += 1

            obs = next_obs
            total_steps += 1
            rollout_steps += 1

            if rollout_steps >= args.rollout_steps:
                agv_stats = agv_agent.update(args.gamma, args.lam, args.clip, args.entropy_coef, args.epochs, args.mini_batches)
                pick_stats = pick_agent.update(args.gamma, args.lam, args.clip, args.entropy_coef, args.epochs, args.mini_batches)
                merged = {
                    "step": total_steps,
                    "agv_actor_loss": agv_stats["actor_loss"],
                    "agv_value_loss": agv_stats["value_loss"],
                    "agv_entropy": agv_stats["entropy"],
                    "agv_approx_kl": agv_stats["approx_kl"],
                    "pick_actor_loss": pick_stats["actor_loss"],
                    "pick_value_loss": pick_stats["value_loss"],
                    "pick_entropy": pick_stats["entropy"],
                    "pick_approx_kl": pick_stats["approx_kl"],
                }
                update_rows.append(merged)
                if tb_writer is not None:
                    tb_writer.add_scalar("update/agv_actor_loss", agv_stats["actor_loss"], total_steps)
                    tb_writer.add_scalar("update/agv_value_loss", agv_stats["value_loss"], total_steps)
                    tb_writer.add_scalar("update/pick_actor_loss", pick_stats["actor_loss"], total_steps)
                    tb_writer.add_scalar("update/pick_value_loss", pick_stats["value_loss"], total_steps)
                    tb_writer.add_scalar("update/agv_entropy", agv_stats["entropy"], total_steps)
                    tb_writer.add_scalar("update/pick_entropy", pick_stats["entropy"], total_steps)
                rollout_steps = 0

            if done_any:
                deliveries_curve.append(ep_deliveries)
                total_rel = max(1, sum(ep_rel_counts.values()))
                rel_dist = {REL_NAMES[k]: ep_rel_counts[k] / total_rel for k in range(len(REL_NAMES))}
                rel_tracker.log_episode(rel_dist)
                mutualism_curve.append(rel_tracker.mutualism_fraction)
                dominant_role_history.append(np.argmax(episode_role_counts, axis=1).tolist())

                battery_mean = (ep_battery_sum / max(1, ep_battery_count)).tolist()
                battery_end = curr_bat.tolist()

                episode_idx = len(deliveries_curve)
                episode_payload: Dict[str, Any] = {
                    "episode": episode_idx,
                    "step": total_steps,
                    "deliveries": ep_deliveries,
                    "mutualism_fraction": float(rel_tracker.mutualism_fraction),
                    "competition_fraction": float(rel_dist.get("competition", 0.0)),
                    "mean_raw_reward": float(np.mean(raw_rewards)),
                    "mean_shaped_reward": float(np.mean(shaped_rewards)),
                    "battery_mean_all_agents": float(np.mean(battery_mean)),
                    "battery_min_all_agents": float(np.min(battery_end)),
                    "battery_max_all_agents": float(np.max(battery_end)),
                }
                for i in range(n_agents):
                    episode_payload[f"battery_mean_agent_{i}"] = float(battery_mean[i])
                    episode_payload[f"battery_end_agent_{i}"] = float(battery_end[i])
                for pkg_name in ["SOLO", "STANDARD", "LARGE", "HEAVY", "PICKER_SOLO"]:
                    episode_payload[f"deliveries_{pkg_name.lower()}"] = float(ep_pkg_deliveries.get(pkg_name, 0))
                episode_rows.append(episode_payload)

                if tb_writer is not None:
                    tb_writer.add_scalar("episode/deliveries", ep_deliveries, episode_idx)
                    tb_writer.add_scalar("episode/mutualism_fraction", float(rel_tracker.mutualism_fraction), episode_idx)
                    tb_writer.add_scalar("episode/competition_fraction", float(rel_dist.get("competition", 0.0)), episode_idx)
                    tb_writer.add_scalar("episode/mean_raw_reward", float(np.mean(raw_rewards)), episode_idx)
                    tb_writer.add_scalar("episode/mean_shaped_reward", float(np.mean(shaped_rewards)), episode_idx)
                    tb_writer.add_scalar("episode/battery_mean_all_agents", float(np.mean(battery_mean)), episode_idx)
                    tb_writer.add_scalar("episode/battery_min_all_agents", float(np.min(battery_end)), episode_idx)
                    tb_writer.add_scalar("episode/battery_max_all_agents", float(np.max(battery_end)), episode_idx)
                    for i in range(n_agents):
                        tb_writer.add_scalar(f"battery/mean_agent_{i}", float(battery_mean[i]), episode_idx)
                        tb_writer.add_scalar(f"battery/end_agent_{i}", float(battery_end[i]), episode_idx)

                if ep_deliveries > best_deliveries:
                    best_deliveries = float(ep_deliveries)
                    best_episode = episode_idx
                    _save_checkpoint("checkpoint_best.pt", episode_idx, is_best=True)

                if args.checkpoint_every_episodes > 0 and episode_idx % args.checkpoint_every_episodes == 0:
                    _save_checkpoint(f"checkpoint_ep{episode_idx:05d}.pt", episode_idx, is_best=False)

                ep_deliveries = 0
                ep_pkg_deliveries = {}
                ep_rel_counts = defaultdict(int)
                ep_battery_sum.fill(0.0)
                ep_battery_count = 0
                episode_role_counts.fill(0.0)
                obs = _reset_observations(env, seed=None)
                prev_bat = np.array([a.battery for a in raw.agents], dtype=np.float32)

        if rollout_steps:
            agv_agent.update(args.gamma, args.lam, args.clip, args.entropy_coef, args.epochs, args.mini_batches)
            pick_agent.update(args.gamma, args.lam, args.clip, args.entropy_coef, args.epochs, args.mini_batches)

        tsi = compute_tsi(role_counts) if np.any(role_counts) else 0.0
        rsi = compute_rsi(dominant_role_history) if dominant_role_history else 0.0

        _save_checkpoint("checkpoint_final.pt", len(deliveries_curve), is_best=False)

        if episode_rows:
            with open(run_dir / "eval_metrics.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
                writer.writeheader()
                writer.writerows(episode_rows)
        if update_rows:
            with open(run_dir / "update_metrics.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(update_rows[0].keys()))
                writer.writeheader()
                writer.writerows(update_rows)

        with open(run_dir / "seed_summary.json", "w") as f:
            json.dump(
                {
                    "method": method,
                    "seed": seed,
                    "backend": backend,
                    "env": args.env,
                    "best_deliveries": best_deliveries if best_deliveries > float("-inf") else None,
                    "best_episode": best_episode,
                    "n_episodes": len(deliveries_curve),
                    "tsi": float(tsi),
                    "rsi": float(rsi),
                },
                f,
                indent=2,
            )

        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

        return {
            "backend": backend,
            "deliveries_curve": deliveries_curve,
            "mutualism_curve": mutualism_curve,
            "tsi": float(tsi),
            "rsi": float(rsi),
            "checkpoint_dir": str(run_dir),
            "best_deliveries": best_deliveries if best_deliveries > float("-inf") else 0.0,
            "best_episode": best_episode,
        }
    finally:
        env.close()


def _train_harl_backend(method: str, seed: int, args, backend: str) -> dict:
    raise RuntimeError(f"{backend.upper()} requested but HARL is not installed in this environment")


def run_method(method: str, seed: int, args, backend: str) -> dict:
    try:
        if backend in ("mappo", "ippo", "hetppo"):
            return _train_local_ppo_backend(method, seed, args, backend)
        return _train_harl_backend(method, seed, args, backend)
    except Exception as exc:
        print(f"  [WARN] {backend.upper()} failed for {method}/{seed}: {exc}")
        raise


def aggregate_seeds(seed_results: list) -> dict:
    curves = [r["deliveries_curve"] for r in seed_results if r["deliveries_curve"]]
    min_len = min(len(c) for c in curves) if curves else 0
    trimmed = [c[:min_len] for c in curves]
    arr = np.array(trimmed) if trimmed else np.zeros((1, 1))
    return {
        "backend": seed_results[0].get("backend", "unknown") if seed_results else "unknown",
        "mean_completion": float(arr[:, -10:].mean()) if arr.shape[1] > 10 else float(arr.mean()),
        "std_completion": float(arr[:, -10:].std()) if arr.shape[1] > 10 else float(arr.std()),
        "deliveries_curves": [r["deliveries_curve"] for r in seed_results],
        "mutualism_curves": [r["mutualism_curve"] for r in seed_results],
        "tsi": float(np.mean([r["tsi"] for r in seed_results])) if seed_results else 0.0,
        "rsi": float(np.mean([r["rsi"] for r in seed_results])) if seed_results else 0.0,
    }


def main():
    config_arg_parser = argparse.ArgumentParser(add_help=False)
    config_arg_parser.add_argument("--config", default=None)
    config_args, _ = config_arg_parser.parse_known_args()

    if config_args.config:
        parser.set_defaults(**_load_config_defaults(config_args.config))

    args = parser.parse_args()

    if args.backends:
        backends = list(args.backends)
    elif args.backend == "auto":
        backends = list(BACKENDS_DEFAULT)
    else:
        backends = [args.backend]

    print(f"C3 Experiment 1 — Heterogeneous: {args.env}")
    print(
        f"Methods: {METHODS}, seeds: {args.seeds}, timesteps: {args.timesteps:,}, backends: {backends}"
    )

    all_results: dict = {
        "env": args.env,
        "backend": args.backend,
        "backends": backends,
        "methods": {},
        "results": {},
    }

    for backend in backends:
        all_results["results"][backend] = {}
        for method in METHODS:
            print(f"\n--- Backend: {backend} | Method: {method} ---")
            seed_results = []
            for seed in args.seeds:
                print(f"  Seed {seed} ...")
                result = run_method(method, seed, args, backend)
                seed_results.append(result)
            all_results["results"][backend][method] = aggregate_seeds(seed_results)

    # Compatibility: keep flat methods table for tooling that expects old schema.
    canonical_backend = backends[0]
    all_results["methods"] = all_results["results"][canonical_backend]
    all_results["canonical_backend"] = canonical_backend

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
