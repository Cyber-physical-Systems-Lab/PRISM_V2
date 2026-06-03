"""
Heuristic baseline runner for the current tarware simulation.

This script wraps ``tarware.heuristic.heuristic_episode`` and emits two
parallel outputs:

1. An aggregated ``results/heuristic_baseline.json`` (legacy schema kept for
   backward compatibility with the older C3 tooling).
2. A per-seed run directory tree that mirrors the layout produced by
   ``experiments/run_heterogeneous.py``, so the same retained figure and
   per-package analysis tooling can consume heuristic data alongside
   trained-condition data.

Layout of the per-seed tree (matching the hetero training discovery
convention ``<root>/<method>/<backend>_seed<seed>``):

    <checkpoint_dir>/<env_id>/<method_name>/<backend_name>_seed<seed>/
        eval_metrics.csv
        eval_metrics.jsonl
        behavior_events.jsonl
        seed_summary.json

No model checkpoints, ``update_metrics.csv`` or ``seed_progress.json`` are
written because the heuristic is rule-based and never trains.

Usage:
  python experiments/run_heuristic_baseline.py
  python experiments/run_heuristic_baseline.py --env tarware-small-4agvs-2pickers-partialobs-chg-v1
  python experiments/run_heuristic_baseline.py --num_episodes 20 --seeds 0 1 2
  python experiments/run_heuristic_baseline.py --config configs/heuristic_baseline.yaml
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
import sys
from typing import Optional

import gymnasium as gym
import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import tarware  # noqa: F401
from tarware.definitions import AgentType
from tarware.heuristic import heuristic_episode

from analysis.metrics import compute_rsi, compute_tsi
from experiments.run_heterogeneous import (
    PKG_TYPE_NAMES,
    REL_COMMENSALISM,
    REL_COMPETITION,
    REL_MUTUALISM,
    REL_NAMES,
    REL_PARASITISM,
    REL_NEUTRAL,
    ROLE_CHARGING,
    ROLE_DELIVERING,
    ROLE_IDLE,
    ROLE_TASKING,
    NUM_ROLE_BUCKETS,
    classify_rel,
    _infer_role,
    _participants_from_events,
)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _load_config_defaults(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    defaults = {}
    exp = cfg.get("experiment", {})
    if "num_episodes" in exp:
        defaults["num_episodes"] = int(exp["num_episodes"])
    if "seed" in exp:
        if isinstance(exp["seed"], (list, tuple)):
            defaults["seeds"] = [int(s) for s in exp["seed"]]
        else:
            defaults["seed"] = int(exp["seed"])
    if "seeds" in exp:
        defaults["seeds"] = [int(s) for s in exp["seeds"]]
    logging_cfg = cfg.get("logging", {})
    if "output" in logging_cfg:
        defaults["output"] = logging_cfg["output"]
    if "flush_interval" in logging_cfg:
        defaults["flush_interval"] = int(logging_cfg["flush_interval"])
    if "checkpoint_dir" in logging_cfg:
        defaults["checkpoint_dir"] = logging_cfg["checkpoint_dir"]
    if "method_name" in logging_cfg:
        defaults["method_name"] = logging_cfg["method_name"]
    if "backend_name" in logging_cfg:
        defaults["backend_name"] = logging_cfg["backend_name"]
    heuristic_cfg = cfg.get("heuristic", {})
    if "low_battery_threshold" in heuristic_cfg:
        defaults["low_battery_threshold"] = float(heuristic_cfg["low_battery_threshold"])
    if "max_steps" in heuristic_cfg:
        defaults["max_steps"] = int(heuristic_cfg["max_steps"])
    if "max_inactivity_steps" in heuristic_cfg:
        max_inactivity_steps = heuristic_cfg["max_inactivity_steps"]
        defaults["max_inactivity_steps"] = None if max_inactivity_steps in (None, 0) else int(max_inactivity_steps)
    return defaults


parser = argparse.ArgumentParser(
    description="Heuristic baseline runner with hetero-compatible per-seed outputs",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--config",         default=None, help="Path to YAML config file.")
parser.add_argument("--env",            default=None, help="Single env id. If unset, runs full sweep.")
parser.add_argument("--num_episodes",   default=10,   type=int)
parser.add_argument("--seed",           default=42,   type=int,
                    help="Base seed (used as the only seed when --seeds is empty).")
parser.add_argument("--seeds",          nargs="+", type=int, default=None,
                    help="Explicit list of seeds (e.g. --seeds 0 1 2). Overrides --seed.")
parser.add_argument("--num-seeds",      default=None, type=int,
                    help="Convenience seed count. Uses --seed, --seed+1, ... when --seeds/config seeds are unset.")
parser.add_argument("--output",         default="results/heuristic_baseline.json",
                    help="Aggregated JSON output (legacy schema).")
parser.add_argument("--checkpoint_dir", default="runs/c3_hetero_heuristic",
                    help="Root for per-seed run directories mirroring the hetero layout.")
parser.add_argument("--method_name",    default="heuristic",
                    help="Method label used in the per-seed run path.")
parser.add_argument("--backend_name",   default="rule",
                    help="Backend label used in the per-seed run path.")
parser.add_argument("--flush_interval", default=1, type=int,
                    help="Write to disk every N episodes (1 = after each episode).")
parser.add_argument("--low_battery_threshold", default=10.0, type=float,
                    help="Battery threshold below which idle agents go charge instead of taking new work.")
parser.add_argument("--max_steps", default=None, type=int,
                    help="Force an episode horizon in environment steps (default: env registration).")
parser.add_argument(
    "--return-shelves",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="If False, the heuristic delivers and immediately picks the next outbound "
         "shelf, matching MARL-trained policy behaviour (which has no restock cycle).",
)
parser.add_argument("--max_inactivity_steps", default=0, type=int,
                    help="Inactivity cutoff in environment steps; 0 disables early inactivity termination.")


# ──────────────────────────────────────────────────────────────────────────────
# Environment sweep configuration
# ──────────────────────────────────────────────────────────────────────────────

SWEEP_ENVS = [
    "tarware-tiny-2agvs-1pickers-partialobs-chg-v1",
    "tarware-small-4agvs-2pickers-partialobs-chg-v1",
    "tarware-medium-6agvs-3pickers-partialobs-chg-v1",
    "tarware-large-8agvs-4pickers-partialobs-chg-v1",
    "tarware-extralarge-16agvs-8pickers-partialobs-chg-v1",
]

_LEGACY_PKG_TYPES = list(PKG_TYPE_NAMES)


# ──────────────────────────────────────────────────────────────────────────────
# Atomic I/O (mirrors run_heterogeneous.py)
# ──────────────────────────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _atomic_write_csv(path: Path, rows: list[dict]) -> None:
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
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            json.dump(row, f)
            f.write("\n")
    tmp.replace(path)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Per-episode metric reconstruction from heuristic_episode outputs
# ──────────────────────────────────────────────────────────────────────────────

def _index_agents_by_type(env) -> tuple[list[int], list[int], int]:
    """Return (agv_idx, pick_idx, n_agents) for the unwrapped warehouse env."""
    raw = env.unwrapped if hasattr(env, "unwrapped") else env
    agents = raw.agents
    agv_idx = [i for i, a in enumerate(agents) if a.type == AgentType.AGV]
    pick_idx = [i for i, a in enumerate(agents) if a.type == AgentType.PICKER]
    return agv_idx, pick_idx, len(agents)


def _episode_payload(
    env_id: str,
    episode_idx: int,
    all_infos: list,
    episode_returns: np.ndarray,
    agv_idx: list[int],
    pick_idx: list[int],
    n_agents: int,
    total_steps_so_far: int,
    episode_start_step: int,
) -> tuple[dict, np.ndarray, list[int]]:
    """Build one ``eval_metrics.csv`` row from the per-step ``all_infos``.

    Returns ``(payload, role_counts_episode, dominant_role)`` so the caller
    can accumulate TSI/RSI without re-walking the per-step data.
    """
    episode_steps = len(all_infos)

    # Per-agent battery trace -> per-step deltas & per-episode aggregates
    battery_trace = np.asarray(
        [info.get("battery_levels", [0.0] * n_agents) for info in all_infos],
        dtype=np.float32,
    )
    if battery_trace.size == 0 or battery_trace.shape[1] != n_agents:
        # Defensive fallback: env did not expose per-agent battery levels.
        battery_trace = np.zeros((max(1, episode_steps), n_agents), dtype=np.float32)

    battery_mean_per_agent = battery_trace.mean(axis=0).tolist()
    battery_end_per_agent = battery_trace[-1].tolist()

    # Build per-step bat_d (current - previous), starting from t=0 with no prior
    # frame. Use the very first frame as the t=-1 reference, so the first delta
    # is zero -- matching the convention used at the start of a hetero episode.
    bat_d_per_step = np.zeros_like(battery_trace)
    if battery_trace.shape[0] >= 2:
        bat_d_per_step[1:] = battery_trace[1:] - battery_trace[:-1]

    # Per-step relationship classification (same rules as run_heterogeneous.py)
    ep_rel_counts: dict[int, int] = defaultdict(int)
    ep_event_steps = 0
    ep_delivery_steps = 0
    ep_lift_steps = 0
    role_counts_episode = np.zeros((n_agents, NUM_ROLE_BUCKETS), dtype=np.float32)

    ep_deliveries = 0
    ep_returns = 0
    ep_deliveries_by_type = {name: 0 for name in PKG_TYPE_NAMES}
    ep_returns_by_type = {name: 0 for name in PKG_TYPE_NAMES}

    for t, info in enumerate(all_infos):
        # Cooperation event sets for this step
        event_carriers, event_pickers = _participants_from_events(info)
        step_had_delivery = bool(event_carriers)
        step_had_lift = bool(event_pickers)
        if step_had_delivery:
            ep_delivery_steps += 1
        if step_had_lift:
            ep_lift_steps += 1
        if step_had_delivery or step_had_lift:
            ep_event_steps += 1

        # Classify each (agv, picker) pair this step. We pass agv_delivery=0
        # and picker_lift=0 because the agv_in_event / picker_in_event flags
        # (derived from delivery/return events) carry the same information.
        for ai in agv_idx:
            for pi in pick_idx:
                rel = classify_rel(
                    0.0, 0.0,
                    float(bat_d_per_step[t, ai]),
                    float(bat_d_per_step[t, pi]),
                    agv_in_event=ai in event_carriers,
                    picker_in_event=pi in event_pickers,
                )
                ep_rel_counts[rel] += 1

        # Per-agent role inference from busy / charging flags
        busy_flags = info.get("vehicles_busy", [False] * n_agents)
        charging_flags = info.get("charging_states", [False] * n_agents)
        delivered_set = event_carriers | event_pickers
        # The heuristic episode does not expose per-step reward, but
        # _infer_role only uses it as a proxy for delivery; we set 0 and
        # rely on the explicit ``delivered`` flag (carriers/pickers) and on
        # the bat_d charging proxy.
        for i in range(n_agents):
            role = _infer_role(
                step_reward=0.0,
                bat_delta=float(bat_d_per_step[t, i]),
                busy=bool(busy_flags[i]) if i < len(busy_flags) else False,
                charging=bool(charging_flags[i]) if i < len(charging_flags) else False,
                delivered=i in delivered_set,
            )
            role_counts_episode[i, role] += 1.0

        # Per-step delivery/return tallies
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

    # Aggregate relationship distribution
    total_rel = max(1, sum(ep_rel_counts.values()))
    rel_dist = {REL_NAMES[k]: ep_rel_counts.get(k, 0) / total_rel for k in range(len(REL_NAMES))}

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

    # Per-episode reward statistics. The heuristic does not produce per-step
    # rewards, only the cumulative episode return; we use per-agent-per-step
    # means as a stable proxy for the hetero ``mean_raw_reward`` column.
    mean_raw_reward = float(np.mean(episode_returns)) / max(1, episode_steps)
    mean_shaped_reward = mean_raw_reward  # no shaping for the rule baseline

    cycle_total = ep_deliveries + ep_returns

    payload: dict = {
        "episode": episode_idx,
        "step": total_steps_so_far + episode_steps,
        "total_steps": total_steps_so_far + episode_steps,
        "episode_steps": int(episode_steps),
        "episode_start_step": int(episode_start_step),
        "deliveries": int(ep_deliveries),
        "returns": int(ep_returns),
        "cycle_total": int(cycle_total),
        "deliveries_by_pkg_type": dict(ep_deliveries_by_type),
        "returns_by_pkg_type": dict(ep_returns_by_type),
        "mutualism_fraction": float(rel_dist.get("mutualism", 0.0)),
        "competition_fraction": float(rel_dist.get("competition", 0.0)),
        "commensalism_fraction": float(rel_dist.get("commensalism", 0.0)),
        "neutral_fraction": float(rel_dist.get("neutral", 0.0)),
        "parasitism_fraction": float(rel_dist.get("parasitism", 0.0)),
        "mutualism_event_steps": int(ep_event_steps),
        "delivery_steps": int(ep_delivery_steps),
        "lift_steps": int(ep_lift_steps),
        "mutualism_fraction_event": float(mutualism_fraction_event),
        "mutualism_events_per_step": float(mutualism_events_per_step),
        "mutualism_events_per_delivery": float(mutualism_events_per_delivery),
        "mean_raw_reward": float(mean_raw_reward),
        "mean_shaped_reward": float(mean_shaped_reward),
        "battery_mean_all_agents": float(np.mean(battery_mean_per_agent)),
        "battery_min_all_agents": float(np.min(battery_end_per_agent)),
        "battery_max_all_agents": float(np.max(battery_end_per_agent)),
    }
    for i in range(n_agents):
        payload[f"battery_mean_agent_{i}"] = float(battery_mean_per_agent[i])
        payload[f"battery_end_agent_{i}"] = float(battery_end_per_agent[i])
        # W_agent_i in the hetero schema is the discounted cumulative return;
        # the heuristic only exposes the undiscounted total per agent, so we
        # log that here (clearly documented in the script docstring).
        payload[f"W_agent_{i}"] = float(episode_returns[i])
    for pkg_name in PKG_TYPE_NAMES:
        payload[f"deliveries_pkg_{pkg_name}"] = int(ep_deliveries_by_type[pkg_name])
        payload[f"returns_pkg_{pkg_name}"] = int(ep_returns_by_type[pkg_name])

    dominant_role = np.argmax(role_counts_episode, axis=1).tolist()
    return payload, role_counts_episode, dominant_role


def _episode_package_rows(
    env_id: str,
    episode_idx: int,
    all_infos: list,
    episode_behavior: list,
    method_name: str,
    backend_name: str,
    trained_seed: int,
    eval_seed: int,
) -> list[dict]:
    """Reconstruct per-package metrics for one heuristic evaluation episode."""
    if not all_infos:
        return []

    n_agents = len(all_infos[0].get("agent_positions", []) or [])
    if n_agents == 0:
        return []

    first_assignment_by_shelf: dict[int, int] = {}
    for event in episode_behavior:
        if event.get("event_type") != "planner_assign":
            continue
        shelf_id = int(event.get("shelf_id", -1))
        if shelf_id < 0 or shelf_id in first_assignment_by_shelf:
            continue
        first_assignment_by_shelf[shelf_id] = int(event.get("episode_step", 0))

    active_prev: set[int] = set()
    in_flight: dict[int, dict[str, int]] = {}
    per_agent_step_distance: list[dict[int, int]] = [dict() for _ in range(n_agents)]
    prev_positions: list[tuple[int, int]] | None = None
    rows: list[dict] = []

    for step_idx, info in enumerate(all_infos):
        active_now = {int(shelf_id) for shelf_id in (info.get("request_queue_ids", []) or [])}
        for shelf_id in active_now - active_prev:
            in_flight.setdefault(
                shelf_id,
                {
                    "created_step": int(step_idx),
                    "first_assignment_step": None,
                },
            )
        active_prev = active_now

        positions = info.get("agent_positions", []) or []
        if prev_positions is None:
            prev_positions = [tuple(int(coord) for coord in pos) for pos in positions]
        else:
            for agent_i, pos in enumerate(positions):
                if agent_i >= n_agents or agent_i >= len(prev_positions):
                    continue
                current = (int(pos[0]), int(pos[1]))
                prev = prev_positions[agent_i]
                per_agent_step_distance[agent_i][step_idx] = abs(current[0] - prev[0]) + abs(current[1] - prev[1])
            prev_positions = [tuple(int(coord) for coord in pos) for pos in positions]

        for ev in info.get("delivery_events", []) or []:
            shelf_id = int(ev.get("shelf_id", -1))
            pkg = in_flight.pop(shelf_id, None)
            if pkg is None:
                pkg = {"created_step": 0, "first_assignment_step": None}

            created = int(pkg["created_step"])
            assignment = first_assignment_by_shelf.get(shelf_id)
            if assignment is None:
                assignment = created

            delivered = int(step_idx)
            carrier_ids = [int(x) for x in ev.get("carrier_ids", []) or []]
            picker_ids = [int(x) for x in ev.get("picker_ids", []) or []]
            attributed = set(carrier_ids) | set(picker_ids)

            team_dist_full = 0
            team_dist_assigned = 0
            for agent_i in attributed:
                traces = per_agent_step_distance[agent_i]
                for step, distance in traces.items():
                    if created <= step <= delivered:
                        team_dist_full += int(distance)
                    if assignment <= step <= delivered:
                        team_dist_assigned += int(distance)

            rows.append(
                {
                    "method": method_name,
                    "backend": backend_name,
                    "trained_seed": int(trained_seed),
                    "eval_episode": int(episode_idx),
                    "eval_seed": int(eval_seed),
                    "env": env_id,
                    "shelf_id": shelf_id,
                    "package_type": str(ev.get("package_type", "UNKNOWN")),
                    "created_step": created,
                    "first_assignment_step": int(assignment),
                    "delivered_step": delivered,
                    "steps_used_assignment": int(delivered - assignment),
                    "steps_used_lifecycle": int(delivered - created),
                    "team_distance_full_lifecycle": int(team_dist_full),
                    "team_distance_assignment_only": int(team_dist_assigned),
                    "carrier_ids": carrier_ids,
                    "picker_ids": picker_ids,
                    "n_carriers": len(carrier_ids),
                    "n_pickers": len(picker_ids),
                }
            )

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Legacy aggregated-JSON builder (kept for backward compatibility)
# ──────────────────────────────────────────────────────────────────────────────

def _build_env_result(
    env_id: str,
    deliveries_all: list,
    deliveries_by_type_all: list,
    battery_mean_per_episode: list,
    battery_end_per_episode: list,
    tsi: float,
    rsi: float,
    elapsed: float,
) -> dict:
    """Produce the legacy ``results/heuristic_baseline.json`` per-env entry."""
    battery_mean_overall = (
        np.asarray(battery_mean_per_episode, dtype=float).mean(axis=0).tolist()
        if battery_mean_per_episode else []
    )
    battery_end_mean = (
        np.asarray(battery_end_per_episode, dtype=float).mean(axis=0).tolist()
        if battery_end_per_episode else []
    )

    mean_by_type = {
        k: float(np.mean([e.get(k, 0) for e in deliveries_by_type_all]))
        for k in _LEGACY_PKG_TYPES
    } if deliveries_by_type_all else {k: 0.0 for k in _LEGACY_PKG_TYPES}
    std_by_type = {
        k: float(np.std([e.get(k, 0) for e in deliveries_by_type_all]))
        for k in _LEGACY_PKG_TYPES
    } if deliveries_by_type_all else {k: 0.0 for k in _LEGACY_PKG_TYPES}

    return {
        "env":                            env_id,
        "mean_deliveries":                float(np.mean(deliveries_all)) if deliveries_all else 0.0,
        "std_deliveries":                 float(np.std(deliveries_all))  if deliveries_all else 0.0,
        "deliveries_curve":               deliveries_all,
        "deliveries_by_type_curve":       deliveries_by_type_all,
        "mean_deliveries_by_type":        mean_by_type,
        "std_deliveries_by_type":         std_by_type,
        "battery_mean_per_agent":         battery_mean_overall,
        "battery_end_mean_per_agent":     battery_end_mean,
        "battery_mean_curve_per_episode": battery_mean_per_episode,
        "battery_end_curve_per_episode":  battery_end_per_episode,
        "tsi":                            tsi,
        "rsi":                            rsi,
        "n_episodes":                     len(deliveries_all),
        "elapsed_sec":                    round(elapsed, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-seed run dir writer
# ──────────────────────────────────────────────────────────────────────────────

def _run_seed_for_env(
    env_id: str,
    seed_base: int,
    num_episodes: int,
    method_name: str,
    backend_name: str,
    checkpoint_dir: Path,
    flush_interval: int,
    low_battery_threshold: float,
    return_shelves: bool,
    max_steps: Optional[int],
    max_inactivity_steps: Optional[int],
) -> dict:
    """Run ``num_episodes`` heuristic episodes for one (env, seed) pair.

    Writes the hetero-compatible per-seed run directory and returns a dict
    summarising the seed (also used by the legacy aggregated JSON output).
    """
    print(f"  {env_id} | seed {seed_base} x {num_episodes} eps ...", end=" ", flush=True)
    t0 = time.time()

    try:
        env = gym.make(
            env_id,
            max_inactivity_steps=max_inactivity_steps,
            max_steps=max_steps,
        )
    except Exception as e:
        print(f"SKIP ({e})")
        return {"error": str(e), "env": env_id, "seed": seed_base}

    try:
        env.reset(seed=seed_base)
    except Exception:
        pass

    agv_idx, pick_idx, n_agents = _index_agents_by_type(env)

    run_dir = checkpoint_dir / env_id / method_name / f"{backend_name}_seed{seed_base}"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_csv_path = run_dir / "eval_metrics.csv"
    eval_jsonl_path = run_dir / "eval_metrics.jsonl"
    behavior_jsonl_path = run_dir / "behavior_events.jsonl"
    per_package_jsonl_path = run_dir / "per_package.jsonl"
    seed_summary_path = run_dir / "seed_summary.json"

    episode_rows: list[dict] = []
    behavior_rows: list[dict] = []
    deliveries_all: list[float] = []
    deliveries_by_type_all: list[dict] = []
    battery_mean_per_episode: list[list[float]] = []
    battery_end_per_episode: list[list[float]] = []
    per_package_rows: list[dict] = []

    role_counts_total = np.zeros((n_agents, NUM_ROLE_BUCKETS), dtype=np.float32)
    dominant_role_history: list[list[int]] = []
    best_deliveries = float("-inf")
    best_returns = float("-inf")
    best_cycle_total = float("-inf")
    best_episode = -1
    total_steps = 0
    np.random.seed(seed_base)

    for ep in range(num_episodes):
        ep_seed = seed_base + ep
        try:
            result = heuristic_episode(
                env,
                seed=ep_seed,
                low_battery_threshold=low_battery_threshold,
                return_shelves=return_shelves,
            )
        except Exception as e:
            print(f"\n    Episode {ep} failed: {e}")
            continue

        if not (isinstance(result, (list, tuple)) and len(result) >= 3):
            continue

        all_infos = result[0] if isinstance(result[0], list) else []
        episode_returns = np.asarray(result[2], dtype=np.float64)
        episode_behavior = result[6] if len(result) >= 7 and isinstance(result[6], list) else []
        if episode_returns.shape[0] != n_agents:
            # Episode returns size may differ if env reports for a different
            # agent ordering; pad/truncate defensively.
            tmp = np.zeros(n_agents, dtype=np.float64)
            tmp[: min(n_agents, episode_returns.shape[0])] = episode_returns[
                : min(n_agents, episode_returns.shape[0])
            ]
            episode_returns = tmp

        episode_idx = len(episode_rows) + 1
        episode_start_step = total_steps
        payload, ep_role_counts, ep_dom_role = _episode_payload(
            env_id=env_id,
            episode_idx=episode_idx,
            all_infos=all_infos,
            episode_returns=episode_returns,
            agv_idx=agv_idx,
            pick_idx=pick_idx,
            n_agents=n_agents,
            total_steps_so_far=total_steps,
            episode_start_step=episode_start_step,
        )

        episode_rows.append(payload)
        for event in episode_behavior:
            behavior_rows.append(
                _json_safe(
                    {
                        "episode": int(episode_idx),
                        "step": int(total_steps + int(event.get("episode_step", 0))),
                        "total_steps": int(total_steps + int(event.get("episode_step", 0))),
                        **event,
                    }
                )
            )
        per_package_rows.extend(
            _episode_package_rows(
                env_id=env_id,
                episode_idx=episode_idx,
                all_infos=all_infos,
                episode_behavior=episode_behavior,
                method_name=method_name,
                backend_name=backend_name,
                trained_seed=seed_base,
                eval_seed=ep_seed,
            )
        )
        deliveries_all.append(float(payload["deliveries"]))
        deliveries_by_type_all.append(dict(payload["deliveries_by_pkg_type"]))
        battery_mean_per_episode.append(
            [payload[f"battery_mean_agent_{i}"] for i in range(n_agents)]
        )
        battery_end_per_episode.append(
            [payload[f"battery_end_agent_{i}"] for i in range(n_agents)]
        )

        role_counts_total += ep_role_counts
        dominant_role_history.append(ep_dom_role)

        ep_cycle_total = int(payload["cycle_total"])
        if ep_cycle_total > best_cycle_total:
            best_cycle_total = float(ep_cycle_total)
            best_deliveries = float(payload["deliveries"])
            best_returns = float(payload["returns"])
            best_episode = episode_idx

        total_steps += int(payload["episode_steps"])

        if (ep + 1) % flush_interval == 0:
            _atomic_write_csv(eval_csv_path, episode_rows)
            _atomic_write_jsonl(eval_jsonl_path, episode_rows)
            _atomic_write_jsonl(behavior_jsonl_path, behavior_rows)
            _atomic_write_jsonl(per_package_jsonl_path, per_package_rows)

    # Always flush at the end (covers num_episodes % flush_interval != 0)
    _atomic_write_csv(eval_csv_path, episode_rows)
    _atomic_write_jsonl(eval_jsonl_path, episode_rows)
    _atomic_write_jsonl(behavior_jsonl_path, behavior_rows)
    _atomic_write_jsonl(per_package_jsonl_path, per_package_rows)

    tsi = float(compute_tsi(role_counts_total)) if np.any(role_counts_total) else 0.0
    rsi = float(compute_rsi(dominant_role_history)) if dominant_role_history else 0.0

    summary = {
        "method": method_name,
        "seed": int(seed_base),
        "backend": backend_name,
        "env": env_id,
        "status": "completed",
        "total_steps": int(total_steps),
        "completed_episodes": len(deliveries_all),
        "best_deliveries": best_deliveries if best_deliveries > float("-inf") else None,
        "best_returns": best_returns if best_returns > float("-inf") else None,
        "best_cycle_total": best_cycle_total if best_cycle_total > float("-inf") else None,
        "best_episode": int(best_episode),
        "n_episodes": len(deliveries_all),
        "behavior_events": len(behavior_rows),
        "tsi": tsi,
        "rsi": rsi,
    }
    _atomic_write_json(seed_summary_path, summary)

    elapsed = time.time() - t0
    try:
        rel_dir = run_dir.relative_to(ROOT)
    except ValueError:
        rel_dir = run_dir
    print(f"{len(deliveries_all)} eps, {elapsed:.1f}s -> {rel_dir}")

    env.close()

    return {
        "env": env_id,
        "seed": int(seed_base),
        "run_dir": str(run_dir),
        "deliveries_all": deliveries_all,
        "deliveries_by_type_all": deliveries_by_type_all,
        "battery_mean_per_episode": battery_mean_per_episode,
        "battery_end_per_episode": battery_end_per_episode,
        "tsi": tsi,
        "rsi": rsi,
        "elapsed_sec": round(elapsed, 1),
        "summary": summary,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _config_sweep_envs(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("env", {}).get("envs", None)


def main():
    config_pre = argparse.ArgumentParser(add_help=False)
    config_pre.add_argument("--config", default=None)
    config_args, _ = config_pre.parse_known_args()
    if config_args.config:
        parser.set_defaults(**_load_config_defaults(config_args.config))
    args = parser.parse_args()

    if args.env:
        envs = [args.env]
    elif config_args.config:
        envs = _config_sweep_envs(config_args.config) or SWEEP_ENVS
    else:
        envs = SWEEP_ENVS

    if args.seeds:
        seeds = list(args.seeds)
    elif args.num_seeds:
        seeds = [int(args.seed) + i for i in range(int(args.num_seeds))]
    else:
        seeds = [int(args.seed)]
    checkpoint_dir = Path(args.checkpoint_dir)
    max_inactivity_steps = None if args.max_inactivity_steps in (None, 0) else int(args.max_inactivity_steps)

    print(
        f"Heuristic baseline -- {len(envs)} env(s) x {len(seeds)} seed(s) x "
        f"{args.num_episodes} ep(s) -> {checkpoint_dir}"
    )

    out = Path(args.output)
    legacy_results: dict = {}

    for env_id in envs:
        per_seed_results = []
        for seed in seeds:
            seed_out = _run_seed_for_env(
                env_id=env_id,
                seed_base=int(seed),
                num_episodes=args.num_episodes,
                method_name=args.method_name,
                backend_name=args.backend_name,
                checkpoint_dir=checkpoint_dir,
                flush_interval=args.flush_interval,
                low_battery_threshold=args.low_battery_threshold,
                return_shelves=args.return_shelves,
                max_steps=args.max_steps,
                max_inactivity_steps=max_inactivity_steps,
            )
            per_seed_results.append(seed_out)

        # Aggregate legacy stats across seeds for this env (matches the old
        # single-seed schema; multi-seed means/std are over the pooled
        # per-episode deliveries from all seeds).
        deliveries_all: list[float] = []
        deliveries_by_type_all: list[dict] = []
        battery_mean_per_episode: list[list[float]] = []
        battery_end_per_episode: list[list[float]] = []
        tsi_per_seed: list[float] = []
        rsi_per_seed: list[float] = []
        elapsed_total = 0.0
        skipped_reason = None

        for r in per_seed_results:
            if "error" in r:
                skipped_reason = r["error"]
                continue
            deliveries_all.extend(r["deliveries_all"])
            deliveries_by_type_all.extend(r["deliveries_by_type_all"])
            battery_mean_per_episode.extend(r["battery_mean_per_episode"])
            battery_end_per_episode.extend(r["battery_end_per_episode"])
            tsi_per_seed.append(r["tsi"])
            rsi_per_seed.append(r["rsi"])
            elapsed_total += r["elapsed_sec"]

        if skipped_reason is not None and not deliveries_all:
            legacy_results[env_id] = {"error": skipped_reason}
            continue

        legacy_results[env_id] = _build_env_result(
            env_id=env_id,
            deliveries_all=deliveries_all,
            deliveries_by_type_all=deliveries_by_type_all,
            battery_mean_per_episode=battery_mean_per_episode,
            battery_end_per_episode=battery_end_per_episode,
            tsi=float(np.mean(tsi_per_seed)) if tsi_per_seed else 0.0,
            rsi=float(np.mean(rsi_per_seed)) if rsi_per_seed else 0.0,
            elapsed=elapsed_total,
        )
        legacy_results[env_id]["per_seed_run_dirs"] = [r.get("run_dir") for r in per_seed_results if "run_dir" in r]
        legacy_results[env_id]["seeds"] = [r["seed"] for r in per_seed_results if "seed" in r]

        _atomic_write_json(out, legacy_results)

    _atomic_write_json(out, legacy_results)
    print(f"\nLegacy aggregated results saved to {out}")
    print(f"Per-seed run directories under {checkpoint_dir}")

    # Summary table
    print("\n--- Heuristic Baseline Summary ---")
    print(f"{'Env':<55}  {'Del/ep':>8}  {'+/-':>6}  {'TSI':>6}  {'RSI':>6}")
    for env_id, r in legacy_results.items():
        if "error" in r:
            print(f"{env_id:<55}  ERROR: {r['error']}")
            continue
        print(
            f"{env_id:<55}  {r['mean_deliveries']:>8.2f}  "
            f"{r['std_deliveries']:>6.2f}  {r['tsi']:>6.3f}  {r['rsi']:>6.3f}"
        )


if __name__ == "__main__":
    main()
