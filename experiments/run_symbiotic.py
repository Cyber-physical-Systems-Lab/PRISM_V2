"""
PRISM — Symbiotic reward condition.

Trains IPPO/MAPPO on the symbiotic reward decomposition:

    r_i = r_task_i + r_sym_i

where r_sym is shaped by ecological relationship type (mutualism, commensalism,
competition, parasitism) classified from agent fitness trajectories.

Team composition is fixed: 4 AGVs + 2 pickers (mixed, capability-complementary).
The symbiotic reward uses relationship-weighted bonuses/penalties from
symbiosis/reward_decomposition.py.

Compare against run_flat_cooperative.py which applies r_i = r_task_i + r_collab
with the same team — the only difference is the reward structure.

Usage
-----
python experiments/run_symbiotic.py \\
    --config configs/prism_symbiotic.yaml \\
    --timesteps 1000000 --seeds 3 \\
    --output runs/results/prism_symbiotic.json
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
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import tarware  # noqa: F401

from analysis.metrics import SymbiosisEmergenceTracker, compute_rsi, compute_tsi
from tarware.definitions import AgentType


CONDITION = "symbiotic"

REL_MUTUALISM    = 0
REL_COMMENSALISM = 1
REL_COMPETITION  = 2
REL_PARASITISM   = 3
REL_NEUTRAL      = 4
REL_NAMES = ["mutualism", "commensalism", "competition", "parasitism", "neutral"]

ROLE_IDLE       = 0
ROLE_CHARGING   = 1
ROLE_TASKING    = 2
ROLE_DELIVERING = 3
NUM_ROLE_BUCKETS = 4


# ── Argument parser ────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="PRISM — Symbiotic reward condition (r_task + r_sym)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--config", default=None, help="Path to YAML config (prism_symbiotic.yaml).")
parser.add_argument("--env", default="tarware-small-4agvs-2pickers-partialobs-chg-v1")
parser.add_argument("--backend", default="auto", choices=["auto", "mappo", "ippo"],
                    help="PPO backend. auto tries mappo then ippo.")
parser.add_argument("--timesteps", default=1_000_000, type=int)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
parser.add_argument("--rollout_steps", default=1024, type=int)
parser.add_argument("--epochs", default=4, type=int)
parser.add_argument("--mini_batches", default=8, type=int)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--lam", default=0.95, type=float)
parser.add_argument("--clip", default=0.2, type=float)
parser.add_argument("--entropy_coef", default=0.01, type=float)
parser.add_argument("--hidden_dim", default=128, type=int)
parser.add_argument("--output", default="runs/results/prism_symbiotic.json")
parser.add_argument("--checkpoint_dir", default="runs/prism_symbiotic")
parser.add_argument("--tb_logdir", default="runs/tensorboard/prism_symbiotic")
parser.add_argument("--checkpoint_every_episodes", default=100, type=int)
parser.add_argument("--log_interval", default=10, type=int)
parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
parser.add_argument("--max_ep_steps", default=5000, type=int)
parser.add_argument("--max_inactivity_steps", default=None, type=int)
parser.add_argument("--package_distribution", default=None, type=str,
                    help='JSON dict, e.g. \'{"SOLO":0.3,"STANDARD":0.4}\'')
# Symbiotic shaping weights
parser.add_argument("--w_mutualism",    default=2.0,  type=float)
parser.add_argument("--w_commensalism", default=1.5,  type=float)
parser.add_argument("--w_competition",  default=-1.5, type=float)
parser.add_argument("--w_parasitism",   default=-0.5, type=float)
parser.add_argument("--low_battery_threshold", default=10.0, type=float)
parser.add_argument("--depletion_penalty",      default=5.0,  type=float)


# ── Config loader ──────────────────────────────────────────────────────────────

def _load_config_defaults(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    defaults: dict = {}
    experiment = config.get("experiment", {})
    env        = config.get("env", {})
    training   = config.get("training", {})
    safety     = config.get("safety", {})
    symbiotic  = config.get("symbiotic", {})
    logging    = config.get("logging", {})

    if "n_seeds" in experiment:
        defaults["seeds"] = list(range(experiment["n_seeds"]))
    if "id" in env:
        defaults["env"] = env["id"]
    if "max_steps" in env:
        defaults["max_ep_steps"] = env["max_steps"]
    if "max_inactivity_steps" in env:
        defaults["max_inactivity_steps"] = env["max_inactivity_steps"]
    if "package_distribution" in env:
        defaults["package_distribution"] = json.dumps(env["package_distribution"])

    defaults.update({k: training[k] for k in
                     ("timesteps", "rollout_steps", "epochs", "mini_batches", "lr", "entropy_coef")
                     if k in training})
    defaults.update({k: safety[k] for k in
                     ("low_battery_threshold", "depletion_penalty") if k in safety})
    defaults.update({k: symbiotic[k] for k in
                     ("w_mutualism", "w_commensalism", "w_competition", "w_parasitism")
                     if k in symbiotic})
    defaults.update({k: logging[k] for k in
                     ("output", "checkpoint_dir", "tb_logdir",
                      "checkpoint_every_episodes", "log_interval")
                     if k in logging})
    return defaults


# ── Relationship classification ────────────────────────────────────────────────

def classify_rel(agv_reward: float, picker_reward: float,
                 agv_bat_delta: float, picker_bat_delta: float) -> int:
    """Classify the (AGV, Picker) relationship for one timestep."""
    agv_charging    = agv_bat_delta    > 0.5
    picker_charging = picker_bat_delta > 0.5
    delivered           = agv_reward    > 0.5
    picker_just_lifted  = picker_reward > 0.05

    if delivered or picker_just_lifted:
        return REL_MUTUALISM
    if agv_charging and picker_charging:
        return REL_NEUTRAL
    if agv_charging or picker_charging:
        return REL_COMMENSALISM
    return REL_NEUTRAL


# ── Symbiotic reward shaping ───────────────────────────────────────────────────

def shape_rewards_symbiotic(
    raw_rewards: list,
    agv_idx: list,
    pick_idx: list,
    bat_d: np.ndarray,
    args,
) -> tuple[list, list]:
    """r_i = r_task_i + r_sym_i.

    r_sym is a relationship-type-weighted bonus/penalty.  Normalised by the
    number of AGV-picker pairs so total shaping stays O(w) regardless of fleet
    size.
    """
    rewards  = list(raw_rewards)
    n_pairs  = max(1, len(agv_idx) * len(pick_idx))
    rels: list[int] = []

    w_table = {
        REL_MUTUALISM:    args.w_mutualism,
        REL_COMMENSALISM: args.w_commensalism,
        REL_COMPETITION:  args.w_competition,
        REL_PARASITISM:   args.w_parasitism,
        REL_NEUTRAL:      0.0,
    }

    for ai in agv_idx:
        for pi in pick_idx:
            rel   = classify_rel(raw_rewards[ai], raw_rewards[pi], bat_d[ai], bat_d[pi])
            w_val = w_table[rel] / n_pairs
            rels.append(rel)

            rewards[ai] += w_val
            if rel == REL_MUTUALISM:
                # Picker bonus only when it actively lifted this step.
                if raw_rewards[pi] > 0.05:
                    rewards[pi] += w_val
            elif rel != REL_COMMENSALISM:
                rewards[pi] += w_val

    return rewards, rels


# ── PPO building blocks ────────────────────────────────────────────────────────

def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _reset_observations(env: gym.Env, seed: Optional[int] = None) -> list:
    out = env.reset(seed=seed)
    if (isinstance(out, tuple) and len(out) == 2
            and isinstance(out[1], dict) and isinstance(out[0], (list, tuple))):
        return list(out[0])
    return list(out)


def _global_state(obs_list: list) -> np.ndarray:
    return np.concatenate(
        [np.asarray(o, dtype=np.float32).reshape(-1) for o in obs_list], axis=0
    )


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
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TypePPOAgent:
    """Shared-parameter PPO agent for one agent type (AGV or picker)."""

    def __init__(self, obs_dim: int, critic_dim: int, act_dim: int,
                 hidden_dim: int, lr: float, device: torch.device):
        self.device  = device
        self.actor   = MLP(obs_dim,   act_dim, hidden_dim).to(device)
        self.critic  = MLP(critic_dim, 1,      hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.clear()

    def clear(self) -> None:
        self.obs = []; self.critic_obs = []; self.next_critic_obs = []
        self.actions = []; self.logps = []; self.values = []
        self.rewards = []; self.dones = []

    @torch.no_grad()
    def act(self, obs: np.ndarray, critic_obs: np.ndarray,
            mask: Optional[np.ndarray] = None) -> tuple[int, float, float]:
        obs_t    = torch.as_tensor(obs,        dtype=torch.float32, device=self.device).unsqueeze(0)
        critic_t = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits   = self.actor(obs_t)
        if mask is not None:
            m = np.asarray(mask, dtype=np.float32).reshape(-1)
            if m.shape[0] == logits.shape[-1] and np.any(m > 0):
                logits = logits.masked_fill(
                    ~torch.as_tensor(m, dtype=torch.bool, device=self.device).unsqueeze(0), -1e9
                )
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        value  = self.critic(critic_t).squeeze(-1)
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def store(self, obs, critic_obs, next_critic_obs, action, logp, value, reward, done,
              mask=None) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.critic_obs.append(np.asarray(critic_obs, dtype=np.float32))
        self.next_critic_obs.append(np.asarray(next_critic_obs, dtype=np.float32))
        self.actions.append(int(action)); self.logps.append(float(logp))
        self.values.append(float(value)); self.rewards.append(float(reward))
        self.dones.append(float(done))

    def update(self, gamma, lam, clip, entropy_coef, epochs, mini_batches) -> Dict[str, float]:
        n = len(self.actions)
        if not n:
            return {"actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        obs_t        = torch.as_tensor(np.array(self.obs),              dtype=torch.float32, device=self.device)
        critic_t     = torch.as_tensor(np.array(self.critic_obs),       dtype=torch.float32, device=self.device)
        next_critic_t= torch.as_tensor(np.array(self.next_critic_obs),  dtype=torch.float32, device=self.device)
        act_t        = torch.as_tensor(np.array(self.actions),          dtype=torch.int64,   device=self.device)
        old_logp_t   = torch.as_tensor(np.array(self.logps),            dtype=torch.float32, device=self.device)

        values  = np.array(self.values,  dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones   = np.array(self.dones,   dtype=np.float32)
        with torch.no_grad():
            next_values = self.critic(next_critic_t).squeeze(-1).cpu().numpy()

        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + gamma * (1 - dones[t]) * next_values[t] - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns,    dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        mini_batch_size = max(32, n // max(1, mini_batches))
        a_losses, v_losses, entropies, kls = [], [], [], []

        for _ in range(epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, mini_batch_size):
                idx   = perm[start:start + mini_batch_size]
                idx_t = torch.as_tensor(idx, dtype=torch.int64, device=self.device)
                dist  = torch.distributions.Categorical(logits=self.actor(obs_t[idx_t]))
                new_logp = dist.log_prob(act_t[idx_t])
                ratio  = (new_logp - old_logp_t[idx_t]).exp()
                s1     = ratio * adv_t[idx_t]
                s2     = ratio.clamp(1 - clip, 1 + clip) * adv_t[idx_t]
                a_loss = -torch.min(s1, s2).mean()
                v_loss = ((self.critic(critic_t[idx_t]).squeeze(-1) - ret_t[idx_t]) ** 2).mean()
                ent    = dist.entropy().mean()
                loss   = a_loss + 0.5 * v_loss - entropy_coef * ent
                self.optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
                )
                self.optimizer.step()
                a_losses.append(float(a_loss.item()))
                v_losses.append(float(v_loss.item()))
                entropies.append(float(ent.item()))
                kls.append(float((old_logp_t[idx_t] - new_logp).mean().item()))

        self.clear()
        return {
            "actor_loss": float(np.mean(a_losses)) if a_losses else 0.0,
            "value_loss": float(np.mean(v_losses)) if v_losses else 0.0,
            "entropy":    float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl":  float(np.mean(kls)) if kls else 0.0,
        }


# ── Core training loop ─────────────────────────────────────────────────────────

def _train(seed: int, args, backend: str) -> dict:
    """Train one seed under the symbiotic reward condition."""
    centralized = backend == "mappo"
    device = _select_device(args.device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_kwargs: dict = {"max_inactivity_steps": args.max_inactivity_steps,
                        "max_steps": args.max_ep_steps}
    if args.package_distribution:
        env_kwargs["package_distribution"] = json.loads(args.package_distribution)
    env = gym.make(args.env, **env_kwargs)

    try:
        raw     = env.unwrapped
        obs     = _reset_observations(env, seed=seed)
        agents  = raw.agents
        n_agents= len(obs)
        agv_idx = [i for i, a in enumerate(agents) if a.type == AgentType.AGV]
        pick_idx= [i for i, a in enumerate(agents) if a.type == AgentType.PICKER]

        act_dim      = raw.action_space.spaces[0].n
        agv_obs_dim  = int(np.asarray(obs[agv_idx[0]]).shape[0])
        pick_obs_dim = int(np.asarray(obs[pick_idx[0]]).shape[0])
        state_dim    = int(sum(np.asarray(o).shape[0] for o in obs))

        agv_agent  = TypePPOAgent(agv_obs_dim,  state_dim if centralized else agv_obs_dim,
                                  act_dim, args.hidden_dim, args.lr, device)
        pick_agent = TypePPOAgent(pick_obs_dim, state_dim if centralized else pick_obs_dim,
                                  act_dim, args.hidden_dim, args.lr, device)

        run_dir = Path(args.checkpoint_dir) / f"{backend}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        tb_writer = None
        if SummaryWriter is not None and args.tb_logdir:
            tb_dir = Path(args.tb_logdir) / f"{backend}_seed{seed}"
            tb_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_dir))

        prev_bat     = np.array([a.battery for a in raw.agents], dtype=np.float32)
        rel_tracker  = SymbiosisEmergenceTracker(window=max(1, args.log_interval))
        mutualism_curve = []
        deliveries_curve= []
        energy_curve    = []
        role_counts     = np.zeros((n_agents, NUM_ROLE_BUCKETS), dtype=np.float32)
        ep_role_counts  = np.zeros((n_agents, NUM_ROLE_BUCKETS), dtype=np.float32)
        dominant_role_history = []

        ep_deliveries    = 0
        ep_pkg_deliveries: Dict[str, int] = {}
        ep_rel_counts    = defaultdict(int)
        ep_battery_sum   = np.zeros(n_agents, dtype=np.float64)
        ep_battery_count = 0
        total_steps  = 0
        rollout_steps= 0
        episode_rows = []
        update_rows  = []
        best_deliveries = float("-inf")
        best_episode    = -1

        def _save_checkpoint(name: str, episode_idx: int, is_best: bool) -> None:
            payload: Dict[str, Any] = {
                "condition": CONDITION,
                "seed": seed, "backend": backend, "env": args.env,
                "total_steps": total_steps, "episode": episode_idx,
                "is_best": is_best, "best_deliveries": best_deliveries,
                "agv_actor_state_dict":  agv_agent.actor.state_dict(),
                "agv_critic_state_dict": agv_agent.critic.state_dict(),
                "pick_actor_state_dict":  pick_agent.actor.state_dict(),
                "pick_critic_state_dict": pick_agent.critic.state_dict(),
                "optimizer_agv_pick_state_dict": agv_agent.optimizer.state_dict(),
                "optimizer_pick_state_dict":     pick_agent.optimizer.state_dict(),
            }
            torch.save(payload, run_dir / name)

        while total_steps < args.timesteps:
            state  = _global_state(obs)
            actions, logps, values = [], [], []
            valid_masks = raw.compute_valid_action_masks(pickers_to_agvs=True)

            for i in range(n_agents):
                lo     = np.asarray(obs[i], dtype=np.float32)
                co     = state if centralized else lo
                agent  = agv_agent if i in agv_idx else pick_agent
                a, lp, v = agent.act(lo, co, mask=valid_masks[i])
                actions.append(a); logps.append(lp); values.append(v)

            next_obs, raw_rewards, dones, truncs, info = env.step(actions)
            next_obs = list(next_obs)

            curr_bat = np.array([a.battery for a in raw.agents], dtype=np.float32)
            bat_d    = curr_bat - prev_bat
            prev_bat = curr_bat
            ep_battery_sum += curr_bat
            ep_battery_count += 1

            # ── Symbiotic reward shaping ──────────────────────────────────────
            shaped_rewards, rels = shape_rewards_symbiotic(
                list(raw_rewards), agv_idx, pick_idx, bat_d, args
            )

            next_state = _global_state(next_obs)
            done_any   = float(any(dones) or any(truncs))

            for i in range(n_agents):
                lo   = np.asarray(obs[i],      dtype=np.float32)
                nlo  = np.asarray(next_obs[i], dtype=np.float32)
                co   = state      if centralized else lo
                nco  = next_state if centralized else nlo
                agent = agv_agent if i in agv_idx else pick_agent
                agent.store(obs=lo, critic_obs=co, next_critic_obs=nco,
                            action=actions[i], logp=logps[i], value=values[i],
                            reward=shaped_rewards[i], done=done_any, mask=valid_masks[i])

            busy_flags     = info.get("vehicles_busy",   [False] * n_agents)
            charging_flags = info.get("charging_states", [False] * n_agents)
            for i in range(n_agents):
                role = _infer_role(raw_rewards[i], bat_d[i],
                                   bool(busy_flags[i]), bool(charging_flags[i]))
                role_counts[i, role]    += 1.0
                ep_role_counts[i, role] += 1.0

            ep_deliveries += int(info.get("shelf_deliveries", 0))
            for pkg_name, cnt in info.get("deliveries_by_pkg_type", {}).items():
                ep_pkg_deliveries[pkg_name] = ep_pkg_deliveries.get(pkg_name, 0) + int(cnt)
            for rel in rels:
                ep_rel_counts[rel] += 1

            obs          = next_obs
            total_steps  += 1
            rollout_steps += 1

            if rollout_steps >= args.rollout_steps:
                agv_stats  = agv_agent.update( args.gamma, args.lam, args.clip,
                                               args.entropy_coef, args.epochs, args.mini_batches)
                pick_stats = pick_agent.update(args.gamma, args.lam, args.clip,
                                               args.entropy_coef, args.epochs, args.mini_batches)
                row = {
                    "step": total_steps,
                    "agv_actor_loss": agv_stats["actor_loss"],   "agv_value_loss":  agv_stats["value_loss"],
                    "agv_entropy":    agv_stats["entropy"],      "agv_approx_kl":   agv_stats["approx_kl"],
                    "pick_actor_loss":pick_stats["actor_loss"],  "pick_value_loss": pick_stats["value_loss"],
                    "pick_entropy":   pick_stats["entropy"],     "pick_approx_kl":  pick_stats["approx_kl"],
                }
                update_rows.append(row)
                if tb_writer is not None:
                    for k, v_val in row.items():
                        if k != "step":
                            tb_writer.add_scalar(f"update/{k}", v_val, total_steps)
                rollout_steps = 0

            if done_any:
                deliveries_curve.append(ep_deliveries)
                total_rel = max(1, sum(ep_rel_counts.values()))
                rel_dist  = {REL_NAMES[k]: ep_rel_counts[k] / total_rel
                             for k in range(len(REL_NAMES))}
                rel_tracker.log_episode(rel_dist)
                mutualism_curve.append(rel_tracker.mutualism_fraction)
                dominant_role_history.append(np.argmax(ep_role_counts, axis=1).tolist())

                battery_mean = (ep_battery_sum / max(1, ep_battery_count)).tolist()
                battery_end  = curr_bat.tolist()
                energy_consumed = float(100.0 * n_agents - np.sum(curr_bat))
                energy_curve.append(energy_consumed)

                episode_idx = len(deliveries_curve)
                ep_payload: Dict[str, Any] = {
                    "episode": episode_idx, "step": total_steps,
                    "deliveries":          ep_deliveries,
                    "energy_consumed":     energy_consumed,
                    "mutualism_fraction":  float(rel_tracker.mutualism_fraction),
                    "competition_fraction":float(rel_dist.get("competition", 0.0)),
                    "mean_raw_reward":     float(np.mean(raw_rewards)),
                    "mean_shaped_reward":  float(np.mean(shaped_rewards)),
                    "battery_mean_all_agents": float(np.mean(battery_mean)),
                    "battery_min_all_agents":  float(np.min(battery_end)),
                    "battery_max_all_agents":  float(np.max(battery_end)),
                }
                for i in range(n_agents):
                    ep_payload[f"battery_mean_agent_{i}"] = float(battery_mean[i])
                    ep_payload[f"battery_end_agent_{i}"]  = float(battery_end[i])
                for pkg_name in ["SOLO", "STANDARD", "LARGE", "HEAVY", "PICKER_SOLO"]:
                    ep_payload[f"deliveries_{pkg_name.lower()}"] = float(ep_pkg_deliveries.get(pkg_name, 0))
                episode_rows.append(ep_payload)

                if tb_writer is not None:
                    for k, v_val in ep_payload.items():
                        if k not in ("episode",) and isinstance(v_val, (int, float)):
                            tb_writer.add_scalar(f"episode/{k}", v_val, episode_idx)

                if ep_deliveries > best_deliveries:
                    best_deliveries = float(ep_deliveries)
                    best_episode    = episode_idx
                    _save_checkpoint("checkpoint_best.pt", episode_idx, is_best=True)

                if args.checkpoint_every_episodes > 0 and episode_idx % args.checkpoint_every_episodes == 0:
                    _save_checkpoint(f"checkpoint_ep{episode_idx:05d}.pt", episode_idx, is_best=False)

                ep_deliveries = 0; ep_pkg_deliveries = {}
                ep_rel_counts = defaultdict(int)
                ep_battery_sum.fill(0.0); ep_battery_count = 0
                ep_role_counts.fill(0.0)
                obs      = _reset_observations(env, seed=None)
                prev_bat = np.array([a.battery for a in raw.agents], dtype=np.float32)

        if rollout_steps:
            agv_agent.update( args.gamma, args.lam, args.clip, args.entropy_coef, args.epochs, args.mini_batches)
            pick_agent.update(args.gamma, args.lam, args.clip, args.entropy_coef, args.epochs, args.mini_batches)

        tsi = compute_tsi(role_counts)     if np.any(role_counts)       else 0.0
        rsi = compute_rsi(dominant_role_history) if dominant_role_history else 0.0

        _save_checkpoint("checkpoint_final.pt", len(deliveries_curve), is_best=False)

        if episode_rows:
            with open(run_dir / "eval_metrics.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
                w.writeheader(); w.writerows(episode_rows)
        if update_rows:
            with open(run_dir / "update_metrics.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(update_rows[0].keys()))
                w.writeheader(); w.writerows(update_rows)

        with open(run_dir / "seed_summary.json", "w") as f:
            json.dump({
                "condition": CONDITION, "seed": seed, "backend": backend, "env": args.env,
                "best_deliveries": best_deliveries if best_deliveries > float("-inf") else None,
                "best_episode": best_episode, "n_episodes": len(deliveries_curve),
                "tsi": float(tsi), "rsi": float(rsi),
            }, f, indent=2)

        if tb_writer is not None:
            tb_writer.flush(); tb_writer.close()

        return {
            "backend": backend, "deliveries_curve": deliveries_curve,
            "mutualism_curve": mutualism_curve, "energy_curve": energy_curve,
            "tsi": float(tsi), "rsi": float(rsi),
            "checkpoint_dir": str(run_dir),
            "best_deliveries": best_deliveries if best_deliveries > float("-inf") else 0.0,
            "best_episode": best_episode,
        }
    finally:
        env.close()


# ── Seed aggregation ───────────────────────────────────────────────────────────

def aggregate_seeds(seed_results: list) -> dict:
    curves = [r["deliveries_curve"] for r in seed_results if r["deliveries_curve"]]
    min_len = min(len(c) for c in curves) if curves else 0
    arr = np.array([c[:min_len] for c in curves]) if curves else np.zeros((1, 1))
    return {
        "backend":        seed_results[0].get("backend", "unknown") if seed_results else "unknown",
        "mean_completion":float(arr[:, -10:].mean()) if arr.shape[1] > 10 else float(arr.mean()),
        "std_completion": float(arr[:, -10:].std())  if arr.shape[1] > 10 else float(arr.std()),
        "deliveries_curves":  [r["deliveries_curve"]  for r in seed_results],
        "mutualism_curves":   [r["mutualism_curve"]   for r in seed_results],
        "energy_curves":      [r["energy_curve"]      for r in seed_results],
        "tsi": float(np.mean([r["tsi"] for r in seed_results])) if seed_results else 0.0,
        "rsi": float(np.mean([r["rsi"] for r in seed_results])) if seed_results else 0.0,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()
    if config_args.config:
        parser.set_defaults(**_load_config_defaults(config_args.config))

    args = parser.parse_args()

    if args.backend == "auto":
        backends = ["mappo", "ippo"]
    else:
        backends = [args.backend]

    print(f"PRISM — Symbiotic condition | env: {args.env}")
    print(f"Seeds: {args.seeds} | timesteps: {args.timesteps:,} | backends: {backends}")

    all_results: dict = {
        "condition": CONDITION,
        "env": args.env,
        "backends": backends,
        "results": {},
    }

    for backend in backends:
        all_results["results"][backend] = {}
        print(f"\n--- Backend: {backend} ---")
        seed_results = []
        for seed in args.seeds:
            print(f"  Seed {seed} ...")
            try:
                result = _train(seed, args, backend)
            except Exception as exc:
                print(f"  [WARN] {backend.upper()} failed for seed {seed}: {exc}")
                if backend == backends[-1]:
                    raise
                break
            seed_results.append(result)
        if seed_results:
            all_results["results"][backend] = aggregate_seeds(seed_results)

    canonical_backend = next(
        (b for b in backends if all_results["results"].get(b)), backends[0]
    )
    all_results["canonical_backend"] = canonical_backend

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
