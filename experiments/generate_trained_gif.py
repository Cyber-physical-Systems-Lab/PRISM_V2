"""
Generate a GIF from a trained IPPO/HetPPO/MAPPO checkpoint.

Loads the actor networks saved by run_heterogeneous.py and runs a
deterministic evaluation rollout, then renders it with the same
matplotlib renderer used by generate_symbiosis_gifs.py.

Usage
-----
python experiments/generate_trained_gif.py \
    --checkpoint local_runs/checkpoints_200k/symbiotic/ippo_seed1/checkpoint_best.pt \
    --method symbiotic \
    --steps 600 \
    --fps 6 \
    --output_dir assets

The script can also compare two checkpoints side-by-side (baseline vs symbiotic)
if --checkpoint_baseline is supplied; in that case it produces a third GIF
with the two episodes stitched vertically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import tarware  # noqa: F401
import gymnasium as gym

from training.symbiotic_wrapper import (
    SymbioticWrapper,
    SymbioticRewardShaper,
    REL_MUTUALISM,
    REL_COMMENSALISM,
    REL_COMPETITION,
    REL_PARASITISM,
    REL_NEUTRAL,
    REL_NAMES,
    N_REL_TYPES,
)
from tarware.definitions import AgentType, PackageType
from experiments.generate_symbiosis_gifs import render_frame, _title_card, _resize_frames

import matplotlib
matplotlib.use("Agg")
import imageio


# ── Re-create the same MLP used during training ───────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
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


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_actors(checkpoint_path: str, obs_dims: List[int], act_dim: int,
                agv_indices: List[int], picker_indices: List[int],
                hidden_dim: int = 128, device: str = "cpu"):
    """
    Reconstruct actor networks from a checkpoint saved by run_heterogeneous.py.

    Returns a list of actors (one per agent) in agent-index order.
    Handles both HetPPO (shared AGV actor, shared picker actor) and
    IPPO (per-agent actors stored under 'policies' key).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    actors: List[Optional[nn.Module]] = [None] * (max(agv_indices + picker_indices) + 1)

    if "agv_actor_state_dict" in ckpt:
        # HetPPO / MAPPO — shared actor per type
        agv_obs_dim = obs_dims[agv_indices[0]] if agv_indices else obs_dims[0]
        pick_obs_dim = obs_dims[picker_indices[0]] if picker_indices else agv_obs_dim

        agv_actor = MLP(agv_obs_dim, act_dim, hidden_dim).to(device)
        agv_actor.load_state_dict(ckpt["agv_actor_state_dict"])
        agv_actor.eval()

        pick_actor = MLP(pick_obs_dim, act_dim, hidden_dim).to(device)
        pick_actor.load_state_dict(ckpt["pick_actor_state_dict"])
        pick_actor.eval()

        for i in agv_indices:
            actors[i] = agv_actor
        for i in picker_indices:
            actors[i] = pick_actor

    elif "policies" in ckpt:
        # IPPO — per-agent actors
        for i, state_dict in enumerate(ckpt["policies"]):
            actor = MLP(obs_dims[i], act_dim, hidden_dim).to(device)
            actor.load_state_dict(state_dict["actor"])
            actor.eval()
            actors[i] = actor
    else:
        raise ValueError(f"Unrecognised checkpoint format. Keys: {list(ckpt.keys())}")

    print(f"  Loaded checkpoint: {Path(checkpoint_path).name} "
          f"(step {ckpt.get('total_steps', '?')}, "
          f"best_deliveries={ckpt.get('best_deliveries', '?')})")
    return actors, ckpt


@torch.no_grad()
def policy_actions(actors: List[nn.Module], obs: tuple, masks: np.ndarray,
                   deterministic: bool = True) -> List[int]:
    """
    Select actions using the trained actor networks.
    Uses argmax (greedy/deterministic) for evaluation rollouts.
    Falls back to random valid action if actor is None.
    """
    actions = []
    for i, (actor, ob) in enumerate(zip(actors, obs)):
        valid = np.where(masks[i] > 0)[0]
        if not valid.size:
            actions.append(0)
            continue
        if actor is None:
            actions.append(int(np.random.choice(valid)))
            continue
        x = torch.tensor(ob, dtype=torch.float32).unsqueeze(0)
        logits = actor(x).squeeze(0).numpy()
        # Mask invalid actions with large negative value then argmax
        masked = np.full(logits.shape, -1e9)
        masked[valid] = logits[valid]
        if deterministic:
            action = int(np.argmax(masked))
        else:
            probs = np.exp(masked - masked.max())
            probs /= probs.sum()
            action = int(np.random.choice(len(probs), p=probs))
        actions.append(action)
    return actions


# ── Episode runner ────────────────────────────────────────────────────────────

def run_trained_episode(wrapper: SymbioticWrapper, raw_env, actors: List[nn.Module],
                        seed: int, max_steps: int, deterministic: bool = True,
                        raw_obs_dims: Optional[List[int]] = None):
    """Run one episode with the trained policy; return frames + rel_events.

    raw_obs_dims: if set, slice the first N features from each agent's (possibly
    augmented) observation before passing to the actor network. This handles
    checkpoints trained on raw env obs before SymbioticWrapper augmentation.
    """
    obs, _ = wrapper.reset(seed=seed)
    # Full battery at episode start for a clean visual
    for agent in raw_env.agents:
        agent.battery = 100.0
    wrapper._prev_batteries = np.array([a.battery for a in raw_env.agents], dtype=np.float32)

    cumulative = {t.name: 0 for t in PackageType}
    frames = []
    rel_events = []
    n_pairs = len(wrapper._agv_indices) * len(wrapper._picker_indices)

    for step in range(max_steps):
        masks = raw_env.compute_valid_action_masks()
        actor_obs = obs
        if raw_obs_dims is not None:
            actor_obs = tuple(np.asarray(o, dtype=np.float32)[:d]
                              for o, d in zip(obs, raw_obs_dims))
        actions = policy_actions(actors, actor_obs, masks, deterministic=deterministic)
        obs, rewards, term, trunc, info = wrapper.step(actions)

        for k, v in info.get("deliveries_by_pkg_type", {}).items():
            cumulative[k] = cumulative.get(k, 0) + v

        # Detect relationship events for overlay
        log_tail = wrapper._rel_log[-n_pairs:] if wrapper._rel_log else []
        for entry in log_tail:
            if entry["rel"] != REL_NEUTRAL:
                rel_events.append((step, entry["rel"]))

        event_label = None
        if log_tail:
            unique = {e["rel"] for e in log_tail if e["rel"] != REL_NEUTRAL}
            if unique:
                dom = min(unique)
                _LABEL_MAP = {
                    REL_MUTUALISM:    "Mutualism    (+/+)",
                    REL_COMMENSALISM: "Commensalism (+/0)",
                    REL_COMPETITION:  "Competition  (-/-)",
                    REL_PARASITISM:   "Parasitism   (+/-)",
                }
                event_label = _LABEL_MAP.get(dom)

        frame = render_frame(raw_env, wrapper, step, dict(cumulative), event_label)
        frames.append(frame)

        if any(term) or any(trunc):
            break

    return frames, rel_events, cumulative


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate GIF from trained IPPO/HetPPO checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint_best.pt or checkpoint_final.pt")
    parser.add_argument("--checkpoint_baseline", default=None,
                        help="Optional second checkpoint for comparison GIF")
    parser.add_argument("--env",
                        default="tarware-small-4agvs-2pickers-partialobs-chg-symbiosis-v1")
    parser.add_argument("--method", default="symbiotic",
                        help="Method name (used in output filenames and titles)")
    parser.add_argument("--seed",    default=0,   type=int)
    parser.add_argument("--steps",   default=600, type=int)
    parser.add_argument("--fps",     default=6,   type=int)
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample from policy distribution instead of argmax")
    parser.add_argument("--output_dir", default="assets")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building env: {args.env}")
    base_env = gym.make(args.env, max_inactivity_steps=None)
    env = SymbioticWrapper(
        base_env,
        low_battery_threshold=10.0,
        depletion_penalty=5.0,
        reward_shaper=SymbioticRewardShaper(
            w_mutualism=2.0, w_commensalism=1.5,
            w_competition=-1.5, w_parasitism=-0.5,
        ),
        rel_alpha=0.05,
    )
    raw = base_env.unwrapped

    # Initialise wrapper so obs spaces are augmented
    obs, _ = env.reset(seed=args.seed)
    agv_idx    = env._agv_indices
    picker_idx = env._picker_indices
    act_dim    = raw.action_size
    hidden_dim = 128

    # Compute obs dims from the RAW env (without symbiotic augmentation).
    # Checkpoints are trained on raw env obs; the wrapper adds extra features
    # that must be stripped before passing to the actor networks.
    raw_obs, _ = base_env.reset(seed=args.seed)
    raw_obs_dims = [int(np.prod(o.shape)) for o in raw_obs]
    obs_dims = raw_obs_dims  # used to reconstruct actor network input size

    print(f"Loading checkpoint: {args.checkpoint}")
    actors, ckpt = load_actors(
        args.checkpoint, obs_dims, act_dim, agv_idx, picker_idx, hidden_dim
    )

    np.random.seed(args.seed)

    # ── Primary rollout ───────────────────────────────────────────────────────
    print(f"Running trained episode (max {args.steps} steps, "
          f"{'stochastic' if args.stochastic else 'deterministic'}) …")
    frames, rel_events, final_counts = run_trained_episode(
        env, raw, actors, args.seed, args.steps,
        deterministic=not args.stochastic, raw_obs_dims=raw_obs_dims
    )
    n_nonneutral = len(rel_events)
    print(f"  {len(frames)} frames, {n_nonneutral} non-neutral relationship events")
    print(f"  Deliveries: {dict(final_counts)}")

    # Overview GIF
    overview_path = out_dir / f"trained_{args.method}_overview.gif"
    print(f"Saving overview GIF → {overview_path}")
    imageio.mimsave(str(overview_path), frames, fps=args.fps, loop=0)

    # Scenarios GIF (one clip per detected relationship type)
    _REL_META = {
        REL_MUTUALISM:    ("Mutualism  (+/+)",    "AGV + Picker cooperate on STANDARD task"),
        REL_COMMENSALISM: ("Commensalism  (+/0)", "Picker assists HEAVY task; AGV gains energy discount"),
        REL_COMPETITION:  ("Competition  (-/-)",  "Agents contest the same charging station"),
        REL_PARASITISM:   ("Parasitism  (+/-)",   "One agent benefits while another drains with no return"),
    }
    scenario_clips = []
    found_types: set = set()
    before_frames, after_frames = 12, 20

    for step_idx, rel_type in rel_events:
        if rel_type in found_types or rel_type == REL_NEUTRAL:
            continue
        if rel_type not in _REL_META:
            continue
        title, subtitle = _REL_META[rel_type]
        card = _title_card(title, subtitle, rel_type,
                           size=(frames[0].shape[1], frames[0].shape[0]))
        start = max(0, step_idx - before_frames)
        end   = min(len(frames), step_idx + after_frames + 1)
        clip  = frames[start:end]
        if not clip:
            continue
        scenario_clips.extend([card] * max(1, args.fps))
        scenario_clips.extend(clip)
        scenario_clips.append(clip[-1])
        found_types.add(rel_type)

    # Always add a neutralism clip from early episode
    neutral_step = min(args.steps // 5, len(frames) - 1)
    neutral_clip = frames[max(0, neutral_step - 8): neutral_step + 15]
    if neutral_clip:
        card = _title_card(
            "Neutralism  (0/0)",
            "Agents operate independently — SOLO and PICKER_SOLO tasks",
            REL_NEUTRAL,
            size=(frames[0].shape[1], frames[0].shape[0]),
        )
        scenario_clips.extend([card] * max(1, args.fps))
        scenario_clips.extend(neutral_clip)

    if scenario_clips:
        target_shape = frames[0].shape
        scenario_clips = _resize_frames(scenario_clips, target_shape)
        scenarios_path = out_dir / f"trained_{args.method}_scenarios.gif"
        print(f"Saving scenarios GIF → {scenarios_path}  ({len(scenario_clips)} frames)")
        imageio.mimsave(str(scenarios_path), scenario_clips, fps=args.fps, loop=0)
        print(f"  Relationship types captured: {[REL_NAMES[r] for r in sorted(found_types)]}")

    # ── Optional: comparison with baseline checkpoint ─────────────────────────
    if args.checkpoint_baseline:
        print(f"\nLoading baseline checkpoint: {args.checkpoint_baseline}")
        actors_base, ckpt_base = load_actors(
            args.checkpoint_baseline, obs_dims, act_dim, agv_idx, picker_idx, hidden_dim
        )
        env2_base = gym.make(args.env, max_inactivity_steps=None)
        env_base = SymbioticWrapper(
            env2_base,
            low_battery_threshold=10.0, depletion_penalty=5.0,
            reward_shaper=SymbioticRewardShaper(
                w_mutualism=2.0, w_commensalism=1.5,
                w_competition=-1.5, w_parasitism=-0.5,
            ),
            rel_alpha=0.05,
        )
        raw_base = env2_base.unwrapped
        env_base.reset(seed=args.seed)

        print(f"Running baseline episode …")
        frames_base, _, counts_base = run_trained_episode(
            env_base, raw_base, actors_base, args.seed, args.steps,
            deterministic=not args.stochastic, raw_obs_dims=raw_obs_dims,
        )
        print(f"  Deliveries (baseline): {dict(counts_base)}")

        # Stitch baseline (left) and symbiotic (right) side-by-side
        n = min(len(frames), len(frames_base))
        comparison = []
        for f_sym, f_base in zip(frames[:n], frames_base[:n]):
            h = max(f_sym.shape[0], f_base.shape[0])
            w = max(f_sym.shape[1], f_base.shape[1])
            row = np.concatenate([
                np.pad(f_base, ((0, h - f_base.shape[0]), (0, w - f_base.shape[1]), (0, 0))),
                np.pad(f_sym,  ((0, h - f_sym.shape[0]),  (0, w - f_sym.shape[1]),  (0, 0))),
            ], axis=1)
            comparison.append(row.astype(np.uint8))
        comp_path = out_dir / f"comparison_{args.method}_vs_baseline.gif"
        print(f"Saving comparison GIF → {comp_path}")
        imageio.mimsave(str(comp_path), comparison, fps=args.fps, loop=0)

    print("Done.")


if __name__ == "__main__":
    main()
