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
import signal
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
from experiments.symbiosis_shaping import PerAgentPotential, make_phi_table, pbrs_term
from tarware.definitions import AgentType


REL_MUTUALISM = 0
REL_COMMENSALISM = 1
REL_COMPETITION = 2
REL_PARASITISM = 3
REL_NEUTRAL = 4
REL_NAMES = ["mutualism", "commensalism", "competition", "parasitism", "neutral"]
PKG_TYPE_NAMES = ("SOLO", "STANDARD", "HEAVY", "PICKER_SOLO", "LARGE")

ROLE_IDLE = 0
ROLE_CHARGING = 1
ROLE_TASKING = 2
ROLE_DELIVERING = 3
NUM_ROLE_BUCKETS = 4


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically so interrupted jobs never leave a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _atomic_write_csv(path: Path, rows: list[dict]) -> None:
    """Rewrite a CSV atomically from in-memory rows."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    """Rewrite a JSONL file atomically from in-memory rows."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            json.dump(row, f)
            f.write("\n")
    tmp.replace(path)


def _pad_curves(curves: list[list[float]]) -> np.ndarray:
    """Right-pad variable-length curves with NaN for alignment."""
    valid = [np.asarray(curve, dtype=np.float32).reshape(-1) for curve in curves if len(curve)]
    if not valid:
        return np.zeros((0, 0), dtype=np.float32)
    max_len = max(len(curve) for curve in valid)
    arr = np.full((len(valid), max_len), np.nan, dtype=np.float32)
    for i, curve in enumerate(valid):
        arr[i, : len(curve)] = curve
    return arr


def _tail_mean(curve: list[float], window: int = 10) -> float:
    """Mean over the last ``window`` points of a single seed curve."""
    arr = np.asarray(curve, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr[-min(window, arr.size):]))


parser = argparse.ArgumentParser(
    description="C3 Experiment 1: Heterogeneous env, all methods",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
METHODS = ["symbiotic", "team", "individual"]
METHODS_ALL = ["symbiotic", "team", "individual", "unclassified"]
METHOD_LABELS = {
    "symbiotic": "PRISM",
    "team": "Flat-cooperative",
    "individual": "Task-only",
    "unclassified": "Unclassified ablation",
}
BACKENDS_ALL = ["ippo", "hetppo", "mappo"]
BACKENDS_DEFAULT = ["ippo", "hetppo", "mappo"]

parser.add_argument("--config", default=None, help="Optional YAML config file.")
parser.add_argument("--env", default="tarware-small-4agvs-2pickers-partialobs-chg-pkgmix-v1")
parser.add_argument("--backends", nargs="+", default=None, choices=BACKENDS_ALL,
                    help="Backends to run (homogeneous-style multi-backend mode).")
parser.add_argument("--methods", nargs="+", default=None, choices=METHODS_ALL,
                    help="Reward conditions to run. Defaults to the manuscript conditions.")
parser.add_argument("--backend", default="auto", choices=["auto", "mappo", "ippo", "hetppo"],
                    help="Deprecated single-backend selector; kept for compatibility.")
parser.add_argument("--timesteps", default=3_000_000, type=int)
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
parser.add_argument("--max_ep_steps", default=1000, type=int)
parser.add_argument("--max_inactivity_steps", default=None, type=int)
parser.add_argument("--unclassified_bonus", default=0.5, type=float)
parser.add_argument("--w_mutualism", default=2.0, type=float)
parser.add_argument("--w_commensalism", default=1.0, type=float)
parser.add_argument("--w_competition", default=-1.5, type=float)
parser.add_argument("--w_parasitism", default=-0.5, type=float)
parser.add_argument("--w_large_form", default=0.0, type=float,
                    help="Bonus split across participants on every LARGE convoy formation.")
parser.add_argument("--w_large_deliver", default=0.0, type=float,
                    help="Bonus split across participants on every LARGE delivery (outbound or inbound).")
parser.add_argument("--w_large_proximity", default=0.0, type=float,
                    help="Per-step bonus for each AGV within --large_proximity_radius cells of a pending LARGE shelf.")
parser.add_argument("--large_proximity_radius", default=3, type=int,
                    help="Chebyshev radius (in cells) for LARGE proximity bonus.")
parser.add_argument("--low_battery_threshold", default=60.0, type=float)
parser.add_argument("--depletion_penalty", default=5.0, type=float)
parser.add_argument("--package_distribution", default=None,
                    help="JSON dict of package-type weights, e.g. '{\"SOLO\":0.3,\"STANDARD\":0.4,\"LARGE\":0.3}'. "
                         "If omitted, the env's registered default applies.")
parser.add_argument("--shaping_form", default="raw", choices=["raw", "pbrs"],
                    help="Form of the symbiotic bonus for the 'symbiotic' method. "
                         "'raw' (default) emits the per-step typed bonus phi(rel)*delta/n_pairs as currently. "
                         "'pbrs' emits gamma*Phi_i(s')-Phi_i(s) using the cumulative event-grounded potential, "
                         "preserving the optimal joint-policy set (Devlin & Kudenko 2011).")
parser.add_argument("--shaping_gamma", default=None, type=float,
                    help="Discount used in the PBRS shaping term gamma*Phi(s')-Phi(s). "
                         "Defaults to args.gamma when omitted.")


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
        defaults["seeds"] = list(range(int(experiment["n_seeds"])))
    if "backends" in experiment:
        defaults["backends"] = experiment["backends"]
    if "methods" in experiment:
        defaults["methods"] = experiment["methods"]
    if "id" in env:
        defaults["env"] = env["id"]
    if "max_steps" in env:
        defaults["max_ep_steps"] = env["max_steps"]
    if "max_inactivity_steps" in env:
        defaults["max_inactivity_steps"] = env["max_inactivity_steps"]
    if "package_distribution" in env and env["package_distribution"] is not None:
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
        for key in ("w_mutualism", "w_commensalism", "w_competition", "w_parasitism",
                    "shaping_form", "shaping_gamma")
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
    agv_in_event: bool = False,
    picker_in_event: bool = False,
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
    delivered = agv_in_event or agv_delivery > 0.5
    # Picker got its +0.1 load-assist reward this step → direct cooperation event
    picker_just_lifted = picker_in_event or picker_lift > 0.05

    if delivered and picker_just_lifted:
        return REL_MUTUALISM
    if delivered or picker_just_lifted:
        return REL_MUTUALISM

    if agv_charging and picker_charging:
        return REL_NEUTRAL

    if agv_charging or picker_charging:
        return REL_COMMENSALISM

    return REL_NEUTRAL


def shape_rewards(raw_rewards: list, agv_idx: list, pick_idx: list, bat_d: np.ndarray, method: str, args, event_carriers: Optional[set] = None, event_pickers: Optional[set] = None) -> tuple[list, list]:
    rewards = list(raw_rewards)
    n_pairs = max(1, len(agv_idx) * len(pick_idx))
    event_carriers = event_carriers or set()
    event_pickers = event_pickers or set()
    rels = []
    for ai in agv_idx:
        for pi in pick_idx:
            rel = classify_rel(
                raw_rewards[ai], raw_rewards[pi], bat_d[ai], bat_d[pi],
                agv_in_event=ai in event_carriers,
                picker_in_event=pi in event_pickers,
            )
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


def apply_large_shaping(
    shaped_rewards: list,
    info: dict,
    agv_idx: list,
    pick_idx: list,
    args,
) -> list:
    """Add LARGE-specific sub-goal bonuses to ``shaped_rewards``.

    Three optional terms, each gated by a non-zero weight in ``args``:
    - ``--w_large_form``: split across participants on every LARGE convoy
      formation reported in ``info["large_load_events"]``.
    - ``--w_large_deliver``: split across participants on every LARGE
      delivery reported in ``info["delivery_events"]`` or
      ``info["return_events"]`` with ``package_type == "LARGE"``.
    - ``--w_large_proximity``: per-step bonus for each AGV within
      ``args.large_proximity_radius`` Chebyshev cells of a pending LARGE
      shelf reported in ``info["pending_large_positions"]``. Capped at
      one bonus per AGV per step (nearest pending LARGE only).
    """
    rewards = list(shaped_rewards)
    w_form = float(getattr(args, "w_large_form", 0.0) or 0.0)
    w_deliver = float(getattr(args, "w_large_deliver", 0.0) or 0.0)
    w_prox = float(getattr(args, "w_large_proximity", 0.0) or 0.0)
    if w_form == 0.0 and w_deliver == 0.0 and w_prox == 0.0:
        return rewards
    if not isinstance(info, dict):
        return rewards

    if w_form != 0.0:
        for event in info.get("large_load_events", []) or []:
            participants = list(event.get("carrier_ids", [])) + list(event.get("picker_ids", []))
            if not participants:
                continue
            share = w_form / len(participants)
            for aid in participants:
                if 0 <= aid < len(rewards):
                    rewards[aid] += share

    if w_deliver != 0.0:
        large_events: list = []
        for event in info.get("delivery_events", []) or []:
            if event.get("package_type") == "LARGE":
                large_events.append(event)
        for event in info.get("return_events", []) or []:
            if event.get("package_type") == "LARGE":
                large_events.append(event)
        for event in large_events:
            participants = list(event.get("carrier_ids", [])) + list(event.get("picker_ids", []))
            if not participants:
                continue
            share = w_deliver / len(participants)
            for aid in participants:
                if 0 <= aid < len(rewards):
                    rewards[aid] += share

    if w_prox != 0.0:
        pending = info.get("pending_large_positions", []) or []
        positions = info.get("agent_positions", []) or []
        if pending and positions:
            radius = max(1, int(getattr(args, "large_proximity_radius", 3) or 3))
            for aid in agv_idx:
                if aid >= len(positions) or aid >= len(rewards):
                    continue
                ax, ay = positions[aid][0], positions[aid][1]
                best = None
                for shelf_pos in pending:
                    d = max(abs(ax - shelf_pos[0]), abs(ay - shelf_pos[1]))
                    if best is None or d < best:
                        best = d
                if best is not None and best <= radius:
                    rewards[aid] += w_prox
    return rewards


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


def _infer_role(step_reward: float, bat_delta: float, busy: bool, charging: bool, delivered: bool = False) -> int:
    if delivered or step_reward > 0.5:
        return ROLE_DELIVERING
    if charging or bat_delta > 0.5:
        return ROLE_CHARGING
    if busy:
        return ROLE_TASKING
    return ROLE_IDLE


def _participants_from_events(info: dict) -> tuple[set, set]:
    """Return (carrier_ids, picker_ids) sets aggregated over both delivery and return events."""
    carriers: set = set()
    pickers: set = set()
    for ev in info.get("delivery_events", []) or []:
        carriers.update(int(i) for i in ev.get("carrier_ids", []))
        pickers.update(int(i) for i in ev.get("picker_ids", []))
    for ev in info.get("return_events", []) or []:
        carriers.update(int(i) for i in ev.get("carrier_ids", []))
        pickers.update(int(i) for i in ev.get("picker_ids", []))
    return carriers, pickers


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
        mini_batch_size = max(256, batch_size // max(1, mini_batches))
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

    env_kwargs: Dict[str, Any] = {
        "max_inactivity_steps": args.max_inactivity_steps,
        "max_steps": args.max_ep_steps,
    }
    if getattr(args, "package_distribution", None):
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

        # PBRS bookkeeping. The potential is a state-only function (zero at
        # episode start, accumulates phi-weighted typed cooperation events).
        # `shaping_form == "raw"` leaves the legacy code path byte-identical;
        # only `--shaping_form pbrs` consults `potential` / `pbrs_phi_table`.
        n_pairs_phi = max(1, len(agv_idx) * len(pick_idx))
        potential = PerAgentPotential(n_agents)
        pbrs_phi_table = make_phi_table("full")
        shaping_form = getattr(args, "shaping_form", "raw")
        shaping_gamma = float(args.shaping_gamma) if getattr(args, "shaping_gamma", None) is not None else float(args.gamma)

        run_dir = Path(args.checkpoint_dir) / method / f"{backend}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_alias_dir = run_dir / "checkpoints"
        checkpoint_alias_dir.mkdir(parents=True, exist_ok=True)
        eval_csv_path = run_dir / "eval_metrics.csv"
        eval_jsonl_path = run_dir / "eval_metrics.jsonl"
        update_csv_path = run_dir / "update_metrics.csv"
        progress_path = run_dir / "seed_progress.json"

        tb_writer = None
        if SummaryWriter is not None and args.tb_logdir:
            tb_dir = Path(args.tb_logdir) / method / f"{backend}_seed{seed}"
            tb_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_dir))

        prev_bat = np.array([a.battery for a in raw.agents], dtype=np.float32)
        last_battery = prev_bat.copy()
        rel_tracker = SymbiosisEmergenceTracker(window=max(1, args.log_interval))
        mutualism_curve = []
        deliveries_curve = []
        battery_mean_curve_per_episode = []
        battery_end_curve_per_episode = []
        role_counts = np.zeros((n_agents, NUM_ROLE_BUCKETS), dtype=np.float32)
        episode_role_counts = np.zeros((n_agents, NUM_ROLE_BUCKETS), dtype=np.float32)
        dominant_role_history = []

        ep_deliveries = 0
        ep_returns = 0
        ep_deliveries_by_type = {name: 0 for name in PKG_TYPE_NAMES}
        ep_returns_by_type = {name: 0 for name in PKG_TYPE_NAMES}
        ep_rel_counts = defaultdict(int)
        ep_event_steps = 0
        ep_delivery_steps = 0
        ep_lift_steps = 0
        ep_battery_sum = np.zeros(n_agents, dtype=np.float64)
        ep_battery_count = 0
        ep_W = np.zeros(n_agents, dtype=np.float64)  # W_i: discounted cumulative return per agent
        ep_time_step = 0  # Track timestep within episode for discount factor
        total_steps = 0
        rollout_steps = 0
        episode_rows = []
        update_rows = []
        best_deliveries = float("-inf")
        best_returns = float("-inf")
        best_cycle_total = float("-inf")
        best_episode = -1
        episode_steps = 0
        episode_start_step = 0
        status = "running"
        shutdown_reason = None
        shutdown_requested = {"flag": False, "reason": None}
        previous_sigterm = None
        previous_sigint = None

        def _handle_shutdown(signum, _frame) -> None:
            shutdown_requested["flag"] = True
            try:
                shutdown_requested["reason"] = signal.Signals(signum).name
            except Exception:
                shutdown_requested["reason"] = f"signal_{signum}"

        def _current_tsi_rsi() -> tuple[float, float]:
            tsi_now = compute_tsi(role_counts) if np.any(role_counts) else 0.0
            rsi_now = compute_rsi(dominant_role_history) if dominant_role_history else 0.0
            return float(tsi_now), float(rsi_now)

        def _build_progress_payload(run_status: str, reason: Optional[str], last_episode: Optional[dict]) -> dict:
            current_battery_mean = (ep_battery_sum / max(1, ep_battery_count)).tolist() if ep_battery_count else []
            current_role_counts = {REL_NAMES[i]: int(ep_rel_counts.get(i, 0)) for i in range(len(REL_NAMES))}
            current_episode = {
                "episode_steps": int(episode_steps),
                "episode_start_step": int(episode_start_step),
                "total_steps": int(total_steps),
                "deliveries": int(ep_deliveries),
                "deliveries_by_pkg_type": dict(ep_deliveries_by_type),
                "returns_by_pkg_type": dict(ep_returns_by_type),
                "battery_mean_all_agents": float(np.mean(current_battery_mean)) if current_battery_mean else 0.0,
                "battery_mean_per_agent": current_battery_mean,
                "battery_end_per_agent": last_battery.tolist(),
                "relation_counts": current_role_counts,
                "role_counts": episode_role_counts.tolist(),
            }
            tsi_now, rsi_now = _current_tsi_rsi()
            return {
                "method": method,
                "seed": seed,
                "backend": backend,
                "env": args.env,
                "status": run_status,
                "reason": reason,
                "total_steps": int(total_steps),
                "completed_episodes": int(len(deliveries_curve)),
                "best_deliveries": float(best_deliveries) if best_deliveries > float("-inf") else None,
                "best_returns": float(best_returns) if best_returns > float("-inf") else None,
                "best_cycle_total": float(best_cycle_total) if best_cycle_total > float("-inf") else None,
                "best_episode": int(best_episode),
                "tsi": tsi_now,
                "rsi": rsi_now,
                "role_counts": role_counts.tolist(),
                "episode_role_counts": episode_role_counts.tolist(),
                "episode_rows_written": int(len(episode_rows)),
                "update_rows_written": int(len(update_rows)),
                "checkpoint_dir": str(run_dir),
                "current_episode": current_episode,
                "last_episode": last_episode,
                "eval_csv_path": str(eval_csv_path),
                "eval_jsonl_path": str(eval_jsonl_path),
                "update_csv_path": str(update_csv_path),
            }

        def _flush_progress(
            run_status: str = "running",
            reason: Optional[str] = None,
            write_eval: bool = False,
            write_update: bool = False,
            last_episode: Optional[dict] = None,
        ) -> None:
            if write_eval:
                _atomic_write_csv(eval_csv_path, episode_rows)
                _atomic_write_jsonl(eval_jsonl_path, episode_rows)
            if write_update:
                _atomic_write_csv(update_csv_path, update_rows)
            _atomic_write_json(progress_path, _build_progress_payload(run_status, reason, last_episode))

        if hasattr(signal, "SIGTERM"):
            previous_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, _handle_shutdown)
        if hasattr(signal, "SIGINT"):
            previous_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _handle_shutdown)

        def _agent_keyed_state() -> dict:
            payload = {}
            for agent_id in range(n_agents):
                if agent_id in agv_idx:
                    payload[f"agent_{agent_id}"] = {
                        "policy": agv_agent.actor.state_dict(),
                        "value": agv_agent.critic.state_dict(),
                        "optimizer": agv_agent.optimizer.state_dict(),
                    }
                else:
                    payload[f"agent_{agent_id}"] = {
                        "policy": pick_agent.actor.state_dict(),
                        "value": pick_agent.critic.state_dict(),
                        "optimizer": pick_agent.optimizer.state_dict(),
                    }
            return payload

        def _agent_keyed_policy_state() -> dict:
            return {
                f"agent_{agent_id}": (
                    agv_agent.actor.state_dict() if agent_id in agv_idx else pick_agent.actor.state_dict()
                )
                for agent_id in range(n_agents)
            }

        def _agent_keyed_value_state() -> dict:
            return {
                f"agent_{agent_id}": (
                    agv_agent.critic.state_dict() if agent_id in agv_idx else pick_agent.critic.state_dict()
                )
                for agent_id in range(n_agents)
            }

        def _save_legacy_aliases(is_best: bool) -> None:
            agent_payload = _agent_keyed_state()
            if is_best:
                torch.save(_agent_keyed_policy_state(), run_dir / "best_policies.pt")
                torch.save(_agent_keyed_value_state(), run_dir / "best_values.pt")
                torch.save(agent_payload, checkpoint_alias_dir / "best_agent.pt")
            else:
                torch.save(agent_payload, checkpoint_alias_dir / f"agent_{int(total_steps)}.pt")

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
                "optimizer_agv_state_dict": agv_agent.optimizer.state_dict(),
                "optimizer_pick_state_dict": pick_agent.optimizer.state_dict(),
            }
            torch.save(payload, run_dir / name)
            _save_legacy_aliases(is_best=is_best)

        try:
            while total_steps < args.timesteps:
                if shutdown_requested["flag"]:
                    status = "interrupted"
                    shutdown_reason = shutdown_requested["reason"]
                    break

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
                episode_steps += 1

                curr_bat = np.array([a.battery for a in raw.agents], dtype=np.float32)
                last_battery = curr_bat.copy()
                bat_d = curr_bat - prev_bat
                prev_bat = curr_bat
                ep_battery_sum += curr_bat
                ep_battery_count += 1

                event_carriers, event_pickers = _participants_from_events(info)
                if method == "symbiotic" and shaping_form == "pbrs":
                    # PBRS form: r_sym_i = gamma*Phi_i(s') - Phi_i(s).
                    # Devlin & Kudenko (2011) Theorem 2 then preserves the
                    # set of Nash-optimal joint policies of the original
                    # Markov game. Rels are still classified for emergence
                    # tracking; we route through shape_rewards with method
                    # "individual" to avoid any task/team rewriting.
                    prev_phi = potential.phi.copy()
                    new_phi = potential.update_from_events(info, pbrs_phi_table, n_pairs_phi)
                    # Finite-horizon terminal correction: Phi_i(s_T) := 0 on
                    # done/truncated transitions. Combined with potential.reset()
                    # at episode start (Phi(s_0)=0), this makes the per-episode
                    # telescoping exact (G^shaped = G^task) and the proof of
                    # Theorem 1 goes through without the infinite-horizon limit.
                    if any(dones) or any(truncs):
                        new_phi = np.zeros_like(new_phi)
                        potential.phi[:] = 0.0
                    pbrs_inc = pbrs_term(prev_phi, new_phi, shaping_gamma)
                    shaped_rewards = [
                        float(raw_rewards[i]) + float(pbrs_inc[i]) for i in range(n_agents)
                    ]
                    _, rels = shape_rewards(
                        list(raw_rewards), agv_idx, pick_idx, bat_d, "individual", args,
                        event_carriers=event_carriers, event_pickers=event_pickers,
                    )
                else:
                    shaped_rewards, rels = shape_rewards(
                        list(raw_rewards), agv_idx, pick_idx, bat_d, method, args,
                        event_carriers=event_carriers, event_pickers=event_pickers,
                    )
                if method == "symbiotic":
                    shaped_rewards = apply_large_shaping(
                        shaped_rewards, info, agv_idx, pick_idx, args,
                    )

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
                delivered_set = event_carriers | event_pickers
                for i in range(n_agents):
                    role = _infer_role(
                        raw_rewards[i], bat_d[i],
                        bool(busy_flags[i]), bool(charging_flags[i]),
                        delivered=i in delivered_set,
                    )
                    role_counts[i, role] += 1.0
                    episode_role_counts[i, role] += 1.0

                ep_deliveries += int(info.get("shelf_deliveries", 0))
                ep_returns += int(info.get("shelf_returns", 0))
                for pkg_name, count in dict(info.get("deliveries_by_pkg_type", {})).items():
                    key = str(pkg_name).upper()
                    if key in ep_deliveries_by_type:
                        ep_deliveries_by_type[key] += int(count)
                for pkg_name, count in dict(info.get("returns_by_pkg_type", {})).items():
                    key = str(pkg_name).upper()
                    if key in ep_returns_by_type:
                        ep_returns_by_type[key] += int(count)
                for rel in rels:
                    ep_rel_counts[rel] += 1
                step_had_delivery = bool(event_carriers)
                step_had_lift = bool(event_pickers)
                if step_had_delivery:
                    ep_delivery_steps += 1
                if step_had_lift:
                    ep_lift_steps += 1
                if step_had_delivery or step_had_lift:
                    ep_event_steps += 1

                # Accumulate W_i (long-run discounted return) per agent
                for i in range(n_agents):
                    ep_W[i] += (args.gamma ** ep_time_step) * shaped_rewards[i]
                ep_time_step += 1

                obs = next_obs
                total_steps += 1
                rollout_steps += 1

                if rollout_steps >= args.rollout_steps:
                    agv_stats = agv_agent.update(args.gamma, args.lam, args.clip, args.entropy_coef, args.epochs, args.mini_batches)
                    pick_stats = pick_agent.update(args.gamma, args.lam, args.clip, args.entropy_coef, args.epochs, args.mini_batches)
                    merged = {
                        "step": total_steps,
                        "total_steps": total_steps,
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
                    _atomic_write_csv(update_csv_path, update_rows)
                    _atomic_write_json(progress_path, _build_progress_payload(status, shutdown_reason, episode_rows[-1] if episode_rows else None))
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
                    mutualism_pair_count = int(ep_rel_counts.get(REL_MUTUALISM, 0))
                    n_agv = max(1, len(agv_idx))
                    n_pick = max(1, len(pick_idx))
                    n_pairs = n_agv * n_pick
                    event_pair_steps = n_pairs * ep_event_steps
                    mutualism_fraction_event = (
                        mutualism_pair_count / event_pair_steps if event_pair_steps > 0 else 0.0
                    )
                    ep_steps_for_rate = max(1, episode_steps)
                    mutualism_events_per_step = ep_event_steps / ep_steps_for_rate
                    mutualism_events_per_delivery = (
                        ep_event_steps / ep_deliveries if ep_deliveries > 0 else 0.0
                    )

                    battery_mean = (ep_battery_sum / max(1, ep_battery_count)).tolist()
                    battery_end = curr_bat.tolist()
                    battery_mean_curve_per_episode.append(battery_mean)
                    battery_end_curve_per_episode.append(battery_end)

                    episode_idx = len(deliveries_curve)
                    ep_cycle_total = ep_deliveries + ep_returns
                    episode_payload: Dict[str, Any] = {
                        "episode": episode_idx,
                        "step": total_steps,
                        "total_steps": total_steps,
                        "episode_steps": episode_steps,
                        "episode_start_step": episode_start_step,
                        "deliveries": ep_deliveries,
                        "returns": ep_returns,
                        "cycle_total": ep_cycle_total,
                        "deliveries_by_pkg_type": dict(ep_deliveries_by_type),
                        "returns_by_pkg_type": dict(ep_returns_by_type),
                        "mutualism_event_steps": int(ep_event_steps),
                        "delivery_steps": int(ep_delivery_steps),
                        "lift_steps": int(ep_lift_steps),
                        "mutualism_fraction_event": float(mutualism_fraction_event),
                        "mutualism_events_per_step": float(mutualism_events_per_step),
                        "mutualism_events_per_delivery": float(mutualism_events_per_delivery),
                        "mean_raw_reward": float(np.mean(raw_rewards)),
                        "mean_shaped_reward": float(np.mean(shaped_rewards)),
                        "battery_mean_all_agents": float(np.mean(battery_mean)),
                        "battery_min_all_agents": float(np.min(battery_end)),
                        "battery_max_all_agents": float(np.max(battery_end)),
                    }
                    for i in range(n_agents):
                        episode_payload[f"battery_mean_agent_{i}"] = float(battery_mean[i])
                        episode_payload[f"battery_end_agent_{i}"] = float(battery_end[i])
                        episode_payload[f"W_agent_{i}"] = float(ep_W[i])  # Fitness (long-run return)
                    for rel_name, rel_value in rel_dist.items():
                        episode_payload[f"relationship_fraction_{rel_name}"] = float(rel_value)
                    episode_payload["mutualism_fraction"] = float(rel_tracker.mutualism_fraction)
                    episode_payload["competition_fraction"] = float(rel_dist.get("competition", 0.0))
                    for pkg_name in PKG_TYPE_NAMES:
                        episode_payload[f"deliveries_pkg_{pkg_name}"] = int(ep_deliveries_by_type[pkg_name])
                        episode_payload[f"returns_pkg_{pkg_name}"] = int(ep_returns_by_type[pkg_name])
                    episode_rows.append(episode_payload)
                    _atomic_write_csv(eval_csv_path, episode_rows)
                    _atomic_write_jsonl(eval_jsonl_path, episode_rows)

                    if tb_writer is not None:
                        tb_writer.add_scalar("episode/deliveries", ep_deliveries, episode_idx)
                        tb_writer.add_scalar("episode/returns", ep_returns, episode_idx)
                        tb_writer.add_scalar("episode/cycle_total", ep_cycle_total, episode_idx)
                        for rel_name, rel_value in rel_dist.items():
                            tb_writer.add_scalar(f"episode/relationship_fraction/{rel_name}", float(rel_value), episode_idx)
                        tb_writer.add_scalar("episode/mutualism_fraction", float(rel_tracker.mutualism_fraction), episode_idx)
                        tb_writer.add_scalar("episode/competition_fraction", float(rel_dist.get("competition", 0.0)), episode_idx)
                        tb_writer.add_scalar("episode/mutualism_fraction_event", float(mutualism_fraction_event), episode_idx)
                        tb_writer.add_scalar("episode/mutualism_event_steps", int(ep_event_steps), episode_idx)
                        tb_writer.add_scalar("episode/mutualism_events_per_step", float(mutualism_events_per_step), episode_idx)
                        tb_writer.add_scalar("episode/mutualism_events_per_delivery", float(mutualism_events_per_delivery), episode_idx)
                        tb_writer.add_scalar("episode/mean_raw_reward", float(np.mean(raw_rewards)), episode_idx)
                        tb_writer.add_scalar("episode/mean_shaped_reward", float(np.mean(shaped_rewards)), episode_idx)
                        tb_writer.add_scalar("episode/battery_mean_all_agents", float(np.mean(battery_mean)), episode_idx)
                        tb_writer.add_scalar("episode/battery_min_all_agents", float(np.min(battery_end)), episode_idx)
                        tb_writer.add_scalar("episode/battery_max_all_agents", float(np.max(battery_end)), episode_idx)
                        for i in range(n_agents):
                            tb_writer.add_scalar(f"battery/mean_agent_{i}", float(battery_mean[i]), episode_idx)
                            tb_writer.add_scalar(f"battery/end_agent_{i}", float(battery_end[i]), episode_idx)

                    if ep_cycle_total > best_cycle_total:
                        best_cycle_total = float(ep_cycle_total)
                        best_deliveries = float(ep_deliveries)
                        best_returns = float(ep_returns)
                        best_episode = episode_idx
                        _save_checkpoint("checkpoint_best.pt", episode_idx, is_best=True)

                    if args.checkpoint_every_episodes > 0 and episode_idx % args.checkpoint_every_episodes == 0:
                        _save_checkpoint(f"checkpoint_ep{episode_idx:05d}.pt", episode_idx, is_best=False)

                    ep_deliveries = 0
                    ep_returns = 0
                    ep_deliveries_by_type = {name: 0 for name in PKG_TYPE_NAMES}
                    ep_returns_by_type = {name: 0 for name in PKG_TYPE_NAMES}
                    ep_rel_counts = defaultdict(int)
                    ep_event_steps = 0
                    ep_delivery_steps = 0
                    ep_lift_steps = 0
                    ep_battery_sum.fill(0.0)
                    ep_battery_count = 0
                    ep_W.fill(0.0)
                    ep_time_step = 0
                    episode_role_counts.fill(0.0)
                    episode_steps = 0
                    episode_start_step = total_steps
                    obs = _reset_observations(env, seed=None)
                    prev_bat = np.array([a.battery for a in raw.agents], dtype=np.float32)
                    last_battery = prev_bat.copy()
                    potential.reset()
                    _flush_progress("running", None, write_eval=True, last_episode=episode_payload)

            if shutdown_requested["flag"]:
                status = "interrupted"
                shutdown_reason = shutdown_requested["reason"]
            else:
                status = "completed"

            if status == "completed" and rollout_steps:
                agv_stats = agv_agent.update(args.gamma, args.lam, args.clip, args.entropy_coef, args.epochs, args.mini_batches)
                pick_stats = pick_agent.update(args.gamma, args.lam, args.clip, args.entropy_coef, args.epochs, args.mini_batches)
                merged = {
                    "step": total_steps,
                    "total_steps": total_steps,
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
                _atomic_write_csv(update_csv_path, update_rows)
                if tb_writer is not None:
                    tb_writer.add_scalar("update/agv_actor_loss", agv_stats["actor_loss"], total_steps)
                    tb_writer.add_scalar("update/agv_value_loss", agv_stats["value_loss"], total_steps)
                    tb_writer.add_scalar("update/pick_actor_loss", pick_stats["actor_loss"], total_steps)
                    tb_writer.add_scalar("update/pick_value_loss", pick_stats["value_loss"], total_steps)
                    tb_writer.add_scalar("update/agv_entropy", agv_stats["entropy"], total_steps)
                    tb_writer.add_scalar("update/pick_entropy", pick_stats["entropy"], total_steps)
                rollout_steps = 0

            tsi = compute_tsi(role_counts) if np.any(role_counts) else 0.0
            rsi = compute_rsi(dominant_role_history) if dominant_role_history else 0.0

            _flush_progress(status, shutdown_reason, write_eval=True, write_update=True, last_episode=episode_rows[-1] if episode_rows else None)

            if status == "completed":
                _save_checkpoint("checkpoint_final.pt", len(deliveries_curve), is_best=False)

                _atomic_write_json(
                    run_dir / "seed_summary.json",
                    {
                        "method": method,
                        "seed": seed,
                        "backend": backend,
                        "env": args.env,
                        "status": status,
                        "total_steps": total_steps,
                        "completed_episodes": len(deliveries_curve),
                        "best_deliveries": best_deliveries if best_deliveries > float("-inf") else None,
                        "best_returns": best_returns if best_returns > float("-inf") else None,
                        "best_cycle_total": best_cycle_total if best_cycle_total > float("-inf") else None,
                        "best_episode": best_episode,
                        "n_episodes": len(deliveries_curve),
                        "tsi": float(tsi),
                        "rsi": float(rsi),
                    },
                )

            return {
                "backend": backend,
                "status": status,
                "reason": shutdown_reason,
                "deliveries_curve": deliveries_curve,
                "mutualism_curve": mutualism_curve,
                "battery_mean_curve_per_episode": battery_mean_curve_per_episode,
                "battery_end_curve_per_episode": battery_end_curve_per_episode,
                "completed_episodes": len(deliveries_curve),
                "total_steps": total_steps,
                "episode_steps": episode_steps,
                "tsi": float(tsi),
                "rsi": float(rsi),
                "checkpoint_dir": str(run_dir),
                "best_deliveries": best_deliveries if best_deliveries > float("-inf") else 0.0,
                "best_episode": best_episode,
                "progress_path": str(progress_path),
            }
        finally:
            if tb_writer is not None:
                try:
                    tb_writer.flush()
                    tb_writer.close()
                except Exception:
                    pass
            if hasattr(signal, "SIGTERM") and previous_sigterm is not None:
                signal.signal(signal.SIGTERM, previous_sigterm)
            if hasattr(signal, "SIGINT") and previous_sigint is not None:
                signal.signal(signal.SIGINT, previous_sigint)
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
    tail_values = [_tail_mean(curve) for curve in curves]
    tail_arr = np.asarray([v for v in tail_values if np.isfinite(v)], dtype=np.float32)
    battery_mean_curves_per_seed = [r.get("battery_mean_curve_per_episode", []) for r in seed_results]
    battery_end_curves_per_seed = [r.get("battery_end_curve_per_episode", []) for r in seed_results]
    return {
        "backend": seed_results[0].get("backend", "unknown") if seed_results else "unknown",
        "mean_completion": float(tail_arr.mean()) if tail_arr.size else 0.0,
        "std_completion": float(tail_arr.std()) if tail_arr.size else 0.0,
        "deliveries_curves": [r["deliveries_curve"] for r in seed_results],
        "mutualism_curves": [r["mutualism_curve"] for r in seed_results],
        "battery_mean_curves_per_seed": battery_mean_curves_per_seed,
        "battery_end_curves_per_seed": battery_end_curves_per_seed,
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
    methods = list(args.methods) if args.methods else list(METHODS)

    print(f"C3 Experiment 1 — Heterogeneous: {args.env}", flush=True)
    print(
        "Methods: "
        f"{[METHOD_LABELS.get(method, method) for method in methods]} "
        f"({methods}), seeds: {args.seeds}, timesteps: {args.timesteps:,}, backends: {backends}",
        flush=True,
    )

    all_results: dict = {
        "env": args.env,
        "backend": args.backend,
        "backends": backends,
        "methods": {},
        "results": {},
    }

    interrupted = False
    interrupt_reason: Optional[str] = None

    for backend in backends:
        all_results["results"][backend] = {}
        for method in methods:
            print(
                f"\n--- Backend: {backend} | Method: {METHOD_LABELS.get(method, method)} ({method}) ---",
                flush=True,
            )
            seed_results = []
            for seed in args.seeds:
                print(f"  Seed {seed} ...", flush=True)
                result = run_method(method, seed, args, backend)
                seed_results.append(result)
                if result.get("status") != "completed":
                    interrupted = True
                    interrupt_reason = result.get("reason") or result.get("status") or "interrupted"
                    break
            all_results["results"][backend][method] = aggregate_seeds(seed_results)
            if interrupted:
                break
        if interrupted:
            break

    # Compatibility: keep flat methods table for tooling that expects old schema.
    canonical_backend = next((b for b in backends if all_results["results"].get(b)), backends[0])
    all_results["methods"] = all_results["results"].get(canonical_backend, {})
    all_results["canonical_backend"] = canonical_backend

    if interrupted:
        out = Path(args.output)
        incomplete_out = out.with_name(f"{out.stem}_incomplete{out.suffix}")
        snapshot = dict(all_results)
        snapshot["status"] = "incomplete"
        snapshot["reason"] = interrupt_reason
        snapshot["completed_backends"] = list(all_results["results"].keys())
        _atomic_write_json(incomplete_out, snapshot)
        print(f"\nRun interrupted ({interrupt_reason}); per-seed progress was flushed to each seed directory.")
        print(f"Incomplete snapshot saved to {incomplete_out}")
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
