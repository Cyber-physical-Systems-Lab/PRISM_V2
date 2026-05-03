"""
PRISM — Evaluation and GIF generation.

Loads a checkpoint saved by run_symbiotic.py or run_flat_cooperative.py,
runs deterministic evaluation episodes on the raw environment (no wrapper,
matching training conditions exactly), and produces:

  1. eval_summary.json  — deliveries, energy, relationship fractions per episode
  2. <condition>_overview.gif    — full episode animation with relationship overlays
  3. <condition>_scenarios.gif   — one clip per detected relationship type
  4. comparison.gif              — flat-coop (left) vs symbiotic (right) side-by-side
                                   (only when --checkpoint_baseline is supplied)

Usage
-----
# Single condition
python experiments/evaluate_and_gif.py \\
    --checkpoint local_runs/checkpoints/prism_symbiotic/.../checkpoint_best.pt \\
    --condition symbiotic \\
    --episodes 5 --steps 1000 --fps 6 \\
    --output_dir local_runs/eval

# Side-by-side comparison
python experiments/evaluate_and_gif.py \\
    --checkpoint      local_runs/checkpoints/prism_symbiotic/.../checkpoint_best.pt \\
    --checkpoint_baseline local_runs/checkpoints/prism_flat_coop/.../checkpoint_best.pt \\
    --condition symbiotic --steps 1000 --fps 6 \\
    --output_dir local_runs/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tarware  # noqa: F401
from tarware.definitions import AgentType, PackageType

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio


# ── Relationship constants ─────────────────────────────────────────────────────

REL_MUTUALISM    = 0
REL_COMMENSALISM = 1
REL_COMPETITION  = 2
REL_PARASITISM   = 3
REL_NEUTRAL      = 4
REL_NAMES        = ["mutualism", "commensalism", "competition", "parasitism", "neutral"]
REL_COLORS       = {
    REL_MUTUALISM:    "#27AE60",
    REL_COMMENSALISM: "#00BCD4",
    REL_COMPETITION:  "#E74C3C",
    REL_PARASITISM:   "#8E44AD",
    REL_NEUTRAL:      "#95A5A6",
}
REL_LABELS = {
    REL_MUTUALISM:    "Mutualism    (+/+)",
    REL_COMMENSALISM: "Commensalism (+/0)",
    REL_NEUTRAL:      "Neutral       (0/0)",
}
# Only show observed relationship types in the GIF sidebar
REL_DISPLAY = [REL_MUTUALISM, REL_COMMENSALISM, REL_NEUTRAL]


# ── MLP (identical to training) ───────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_actors(ckpt_path: str, agv_obs_dim: int, pick_obs_dim: int,
                act_dim: int, agv_idx: List[int], pick_idx: List[int],
                hidden: int = 128) -> Tuple[nn.Module, nn.Module, dict]:
    """Load AGV and picker actor networks from a PRISM checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    agv_actor  = MLP(agv_obs_dim,  act_dim, hidden)
    pick_actor = MLP(pick_obs_dim, act_dim, hidden)
    agv_actor.load_state_dict(ckpt["agv_actor_state_dict"])
    pick_actor.load_state_dict(ckpt["pick_actor_state_dict"])
    agv_actor.eval(); pick_actor.eval()
    print(f"  Loaded: {Path(ckpt_path).name}  "
          f"step={ckpt.get('total_steps','?')}  "
          f"best_deliveries={ckpt.get('best_deliveries','?')}")
    return agv_actor, pick_actor, ckpt


# ── Relationship classifier ───────────────────────────────────────────────────

def classify_rel(agv_r: float, pick_r: float, agv_bd: float, pick_bd: float) -> int:
    delivered       = agv_r    >= 0.5   # inclusive: STANDARD gives exactly 0.5
    picker_lifted   = pick_r   >= 0.05
    agv_charging    = agv_bd   > 0.5
    pick_charging   = pick_bd  > 0.5
    if delivered and picker_lifted:     # joint AGV+picker delivery → mutualism
        return REL_MUTUALISM
    if picker_lifted and not delivered: # picker-only delivery → commensalism
        return REL_COMMENSALISM
    if agv_charging and pick_charging:
        return REL_NEUTRAL
    if agv_charging or pick_charging:
        return REL_COMMENSALISM
    return REL_NEUTRAL


# ── Frame renderer ────────────────────────────────────────────────────────────

AGENT_COLORS = {
    "agv":    "#E67E22",   # orange
    "picker": "#3498DB",   # blue
}
PKG_COLORS = {
    "SOLO":        "#3498DB",
    "STANDARD":    "#27AE60",
    "LARGE":       "#F1C40F",
    "HEAVY":       "#E67E22",
    "PICKER_SOLO": "#9B59B6",
}

def render_frame(raw_env, step: int, deliveries: Dict[str, int],
                 rel_matrix: Dict[Tuple[int,int], int],
                 active_rel: Optional[int] = None) -> np.ndarray:
    """Render one warehouse frame as an RGB numpy array."""
    grid_h, grid_w = raw_env.grid_size
    cell = 40
    sidebar = 320
    fig_w = grid_w * cell + sidebar
    fig_h = max(grid_h * cell, 600)

    fig = plt.figure(figsize=(fig_w / 100, fig_h / 100), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")

    ax = fig.add_axes([0, 0, grid_w * cell / fig_w, 1.0])
    ax.set_facecolor("#16213e")
    ax.set_xlim(0, grid_w); ax.set_ylim(0, grid_h)
    ax.set_aspect("equal"); ax.axis("off")

    # Shelves — colour by package type if in request queue
    pkg_type_map = {}
    for shelf in raw_env.request_queue:
        pkg_type = getattr(shelf, "package_type", None)
        if pkg_type is not None:
            pkg_name = pkg_type.name if hasattr(pkg_type, "name") else str(pkg_type)
            pkg_type_map[shelf.id] = pkg_name

    requested_ids = {s.id for s in raw_env.request_queue}
    for shelf in raw_env.shelfs:
        r, c = shelf.y, shelf.x
        if shelf.id in requested_ids:
            pkg_name = pkg_type_map.get(shelf.id, "STANDARD")
            color = PKG_COLORS.get(pkg_name, "#2d5a27")
            edge  = "#ffffff"
            label = pkg_name[0]  # first letter: S, L, H, P
        else:
            color, edge, label = "#1e3d1a", "#3a7a32", "·"
        rect = plt.Rectangle((c, grid_h - r - 1), 1, 1, color=color, linewidth=0.4,
                              edgecolor=edge, alpha=0.85)
        ax.add_patch(rect)
        ax.text(c + 0.5, grid_h - r - 0.5, label, ha="center", va="center",
                color="white", fontsize=5, fontweight="bold")

    # Charging stations
    for cs in getattr(raw_env, "charging_stations", []):
        r, c = cs.y, cs.x
        rect = plt.Rectangle((c, grid_h - r - 1), 1, 1, color="#1a3a5c",
                              linewidth=0.5, edgecolor="#2980b9")
        ax.add_patch(rect)
        ax.text(c + 0.5, grid_h - r - 0.5, "⚡", ha="center", va="center", fontsize=7)

    # Agents
    for agent in raw_env.agents:
        r, c = agent.y, agent.x
        is_agv = agent.type == AgentType.AGV
        color  = AGENT_COLORS["agv"] if is_agv else AGENT_COLORS["picker"]
        marker = "h" if is_agv else "D"
        ax.plot(c + 0.5, grid_h - r - 0.5, marker, color=color,
                markersize=14, markeredgecolor="white", markeredgewidth=0.5)
        label = f"A{agent.id}" if is_agv else f"P{agent.id}"
        ax.text(c + 0.5, grid_h - r - 0.5, label, ha="center", va="center",
                color="white", fontsize=5, fontweight="bold")
        # Battery bar
        bat_frac = getattr(agent, "battery", 100) / 100.0
        bat_color = "#27AE60" if bat_frac > 0.3 else "#E74C3C"
        ax.plot([c + 0.1, c + 0.1 + 0.8 * bat_frac], [grid_h - r - 0.08] * 2,
                color=bat_color, lw=3, solid_capstyle="round")

    # Active relationship overlay
    if active_rel is not None and active_rel != REL_NEUTRAL:
        fig.patch.set_facecolor(REL_COLORS.get(active_rel, "#1a1a2e") + "22")
        ax.set_title(REL_LABELS.get(active_rel, ""), color=REL_COLORS.get(active_rel, "white"),
                     fontsize=10, fontweight="bold", pad=4)

    # Sidebar
    ax2 = fig.add_axes([grid_w * cell / fig_w, 0, sidebar / fig_w, 1.0])
    ax2.set_facecolor("#0f0f23"); ax2.axis("off")
    y = 0.97

    # Step counter
    ax2.text(0.1, y, f"Step {step}", color="#aaaaaa", fontsize=8,
             transform=ax2.transAxes); y -= 0.06

    # Total deliveries — prominent
    total_del = sum(deliveries.values())
    ax2.text(0.5, y, f"{total_del}", color="#27AE60", fontsize=26,
             fontweight="bold", ha="center", transform=ax2.transAxes); y -= 0.07
    ax2.text(0.5, y, "deliveries", color="#aaaaaa", fontsize=8,
             ha="center", transform=ax2.transAxes); y -= 0.07

    # Per-package breakdown (only non-zero or requested types)
    ax2.text(0.1, y, "By package type", color="#aaaaaa", fontsize=7,
             transform=ax2.transAxes); y -= 0.05
    for pkg, cnt in deliveries.items():
        color = PKG_COLORS.get(pkg, "gray")
        ax2.plot([0.08], [y], "s", color=color, markersize=7, transform=ax2.transAxes)
        ax2.text(0.20, y, f"{pkg[:4]}", color="#cccccc", fontsize=7,
                 transform=ax2.transAxes, va="center")
        ax2.text(0.72, y, str(cnt), color="white", fontsize=8,
                 fontweight="bold", transform=ax2.transAxes, va="center")
        y -= 0.048

    # Package type legend
    y -= 0.01
    ax2.text(0.1, y, "Queue legend", color="#aaaaaa", fontsize=7,
             transform=ax2.transAxes); y -= 0.045
    for pkg_name, pkg_color in PKG_COLORS.items():
        ax2.plot([0.08], [y], "s", color=pkg_color, markersize=6, transform=ax2.transAxes)
        ax2.text(0.20, y, pkg_name, color="#bbbbbb", fontsize=6,
                 transform=ax2.transAxes, va="center")
        y -= 0.040

    # Relationship types (observed only)
    y -= 0.01
    ax2.text(0.1, y, "Relationships", color="#aaaaaa", fontsize=7,
             transform=ax2.transAxes); y -= 0.045
    for rel_id in REL_DISPLAY:
        c = REL_COLORS[rel_id]
        ax2.plot([0.08], [y], "s", color=c, markersize=6, transform=ax2.transAxes)
        ax2.text(0.20, y, REL_LABELS[rel_id], color="white", fontsize=6,
                 transform=ax2.transAxes, va="center")
        y -= 0.040

    # Current pair states
    if rel_matrix:
        y -= 0.01
        ax2.text(0.1, y, "Active pairs", color="#aaaaaa", fontsize=7,
                 transform=ax2.transAxes); y -= 0.04
        for (ai, pi), rel in sorted(rel_matrix.items()):
            c = REL_COLORS[rel]
            label = f"A{ai}↔P{pi}: {REL_NAMES[rel][:3].upper()}"
            ax2.text(0.10, y, label, color=c, fontsize=6,
                     transform=ax2.transAxes); y -= 0.035

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env, raw_env, agv_actor, pick_actor,
                agv_idx, pick_idx, seed: int, max_steps: int,
                render: bool = True):
    obs_list, _ = env.reset(seed=seed)
    obs_list = list(obs_list)

    # Start all agents at full battery for GIF clarity.
    # Trained policies handle full-battery starts (within training distribution).
    for agent in raw_env.agents:
        agent.battery = 100.0

    prev_bat = np.array([a.battery for a in raw_env.agents], dtype=np.float32)
    deliveries = {p.name: 0 for p in PackageType}
    ep_rel_counts = defaultdict(int)
    frames = []
    rel_events = []

    for step in range(max_steps):
        valid_masks = raw_env.compute_valid_action_masks(pickers_to_agvs=True)
        actions = []
        for i, obs in enumerate(obs_list):
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            actor = agv_actor if i in agv_idx else pick_actor
            with torch.no_grad():
                logits = actor(obs_t).squeeze(0).numpy()
            mask = valid_masks[i]
            valid = np.where(np.asarray(mask) > 0)[0]
            if not valid.size:
                actions.append(0)
                continue
            logits_t = torch.as_tensor(logits)
            logits_t_masked = logits_t.clone().fill_(-1e9)
            logits_t_masked[valid] = logits_t[valid]
            dist = torch.distributions.Categorical(logits=logits_t_masked)
            actions.append(int(dist.sample().item()))

        next_obs, raw_rewards, dones, truncs, info = env.step(actions)
        next_obs = list(next_obs)

        curr_bat = np.array([a.battery for a in raw_env.agents], dtype=np.float32)
        bat_d    = curr_bat - prev_bat
        prev_bat = curr_bat

        # Classify relationships
        rel_matrix = {}
        for ai in agv_idx:
            for pi in pick_idx:
                rel = classify_rel(raw_rewards[ai], raw_rewards[pi], bat_d[ai], bat_d[pi])
                rel_matrix[(ai, pi)] = rel
                ep_rel_counts[rel] += 1
                if rel != REL_NEUTRAL:
                    rel_events.append((step, rel))

        # Track deliveries
        for pkg, cnt in info.get("deliveries_by_pkg_type", {}).items():
            deliveries[pkg] = deliveries.get(pkg, 0) + int(cnt)

        active_rel = None
        if rel_events and rel_events[-1][0] == step:
            active_rel = rel_events[-1][1]

        if render:
            frame = render_frame(raw_env, step, deliveries, rel_matrix, active_rel)
            frames.append(frame)

        obs_list = next_obs
        if any(dones) or any(truncs):
            break

    total_rel = max(1, sum(ep_rel_counts.values()))
    rel_dist = {REL_NAMES[k]: ep_rel_counts[k] / total_rel for k in range(len(REL_NAMES))}

    return frames, rel_events, deliveries, rel_dist


# ── GIF builder ───────────────────────────────────────────────────────────────

def save_scenarios_gif(frames, rel_events, out_path: Path, fps: int,
                       before: int = 10, after: int = 20) -> None:
    clips = []
    found = set()
    for step_idx, rel in rel_events:
        if rel in found or rel == REL_NEUTRAL:
            continue
        start = max(0, step_idx - before)
        end   = min(len(frames), step_idx + after + 1)
        clip  = frames[start:end]
        if clip:
            clips.extend(clip)
            found.add(rel)
    if clips:
        imageio.mimsave(str(out_path), clips, fps=fps, loop=0)
        print(f"  Scenarios GIF → {out_path}  ({len(clips)} frames, "
              f"types: {[REL_NAMES[r] for r in sorted(found)]})")
    else:
        print("  No non-neutral events — skipping scenarios GIF")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PRISM — Evaluate trained agents and generate GIFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",          required=True)
    parser.add_argument("--checkpoint_baseline", default=None,
                        help="Flat-coop checkpoint for comparison GIF")
    parser.add_argument("--env", default="tarware-small-4agvs-2pickers-partialobs-chg-v1")
    parser.add_argument("--condition", default="symbiotic")
    parser.add_argument("--episodes",  default=3,    type=int)
    parser.add_argument("--steps",     default=1000, type=int)
    parser.add_argument("--fps",       default=6,    type=int)
    parser.add_argument("--seed",      default=0,    type=int)
    parser.add_argument("--output_dir", default="local_runs/eval")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Building env: {args.env}")
    env     = gym.make(args.env, max_inactivity_steps=None, max_steps=args.steps, package_distribution={"SOLO":0.20,"STANDARD":0.30,"LARGE":0.10,"HEAVY":0.25,"PICKER_SOLO":0.15})
    raw_env = env.unwrapped
    obs0, _ = env.reset(seed=args.seed)

    agents   = raw_env.agents
    agv_idx  = [i for i, a in enumerate(agents) if a.type == AgentType.AGV]
    pick_idx = [i for i, a in enumerate(agents) if a.type == AgentType.PICKER]
    act_dim  = raw_env.action_space.spaces[0].n

    agv_obs_dim  = int(np.asarray(obs0[agv_idx[0]]).shape[0])
    pick_obs_dim = int(np.asarray(obs0[pick_idx[0]]).shape[0])

    print(f"Loading checkpoint: {args.checkpoint}")
    agv_actor, pick_actor, ckpt = load_actors(
        args.checkpoint, agv_obs_dim, pick_obs_dim, act_dim, agv_idx, pick_idx
    )

    # ── Multi-episode evaluation ───────────────────────────────────────────────
    all_deliveries = []
    all_rel_dists  = []
    all_frames     = []
    all_rel_events = []

    for ep in range(args.episodes):
        print(f"  Episode {ep+1}/{args.episodes} …", flush=True)
        render = (ep == 0)   # only render first episode for GIF
        frames, rel_events, deliveries, rel_dist = run_episode(
            env, raw_env, agv_actor, pick_actor,
            agv_idx, pick_idx, seed=args.seed + ep,
            max_steps=args.steps, render=render
        )
        total_del = sum(deliveries.values())
        print(f"    deliveries={total_del}  {dict(deliveries)}")
        print(f"    relationships: { {k: f'{v:.3f}' for k,v in rel_dist.items()} }")
        all_deliveries.append(deliveries)
        all_rel_dists.append(rel_dist)
        if render:
            all_frames  = frames
            all_rel_events = rel_events

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "condition":     args.condition,
        "checkpoint":    args.checkpoint,
        "env":           args.env,
        "n_episodes":    args.episodes,
        "mean_deliveries": float(np.mean([sum(d.values()) for d in all_deliveries])),
        "deliveries_by_episode": [dict(d) for d in all_deliveries],
        "mean_rel_dist": {
            k: float(np.mean([rd[k] for rd in all_rel_dists]))
            for k in REL_NAMES
        },
    }
    summary_path = out / f"{args.condition}_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMean deliveries: {summary['mean_deliveries']:.1f}")
    rel_str = {k: f"{v:.3f}" for k, v in summary['mean_rel_dist'].items()}
    print(f"Relationship distribution: {rel_str}")
    print(f"Summary → {summary_path}")

    # ── GIFs ──────────────────────────────────────────────────────────────────
    if all_frames:
        overview_path = out / f"{args.condition}_overview.gif"
        imageio.mimsave(str(overview_path), all_frames, fps=args.fps, loop=0)
        print(f"Overview GIF → {overview_path}  ({len(all_frames)} frames)")

        scenarios_path = out / f"{args.condition}_scenarios.gif"
        save_scenarios_gif(all_frames, all_rel_events, scenarios_path, args.fps)

    # ── Comparison GIF ────────────────────────────────────────────────────────
    if args.checkpoint_baseline and all_frames:
        print(f"\nLoading baseline: {args.checkpoint_baseline}")
        env2     = gym.make(args.env, max_inactivity_steps=None, max_steps=args.steps, package_distribution={"SOLO":0.20,"STANDARD":0.30,"LARGE":0.10,"HEAVY":0.25,"PICKER_SOLO":0.15})
        raw_env2 = env2.unwrapped
        obs2, _  = env2.reset(seed=args.seed)
        agv_obs2  = int(np.asarray(obs2[agv_idx[0]]).shape[0])
        pick_obs2 = int(np.asarray(obs2[pick_idx[0]]).shape[0])
        agv_base, pick_base, _ = load_actors(
            args.checkpoint_baseline, agv_obs2, pick_obs2, act_dim, agv_idx, pick_idx
        )
        frames_base, _, deliveries_base, _ = run_episode(
            env2, raw_env2, agv_base, pick_base,
            agv_idx, pick_idx, seed=args.seed,
            max_steps=args.steps, render=True
        )
        print(f"  Baseline deliveries: {dict(deliveries_base)}")
        env2.close()

        n = min(len(all_frames), len(frames_base))
        bar_h = 36  # header bar height in pixels
        flat_del  = sum(deliveries_base.values())
        sym_del   = int(np.mean([sum(d.values()) for d in all_deliveries]))
        comparison = []
        for step_i, (f_sym, f_flat) in enumerate(zip(all_frames[:n], frames_base[:n])):
            h = max(f_sym.shape[0], f_flat.shape[0])
            left  = np.pad(f_flat, ((0, h-f_flat.shape[0]), (0,0), (0,0)))
            right = np.pad(f_sym,  ((0, h-f_sym.shape[0]),  (0,0), (0,0)))
            w_each = left.shape[1]
            # Build labelled header bar
            fig_bar, ax_bar = plt.subplots(figsize=(w_each * 2 / 100, bar_h / 100), dpi=100)
            fig_bar.patch.set_facecolor("#1a1a2e")
            ax_bar.set_facecolor("#1a1a2e")
            ax_bar.axis("off")
            ax_bar.text(0.25, 0.5, "Flat-cooperative", color="#95A5A6",
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        transform=ax_bar.transAxes)
            ax_bar.text(0.75, 0.5, "PRISM (Symbiotic)", color="#27AE60",
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        transform=ax_bar.transAxes)
            fig_bar.canvas.draw()
            bar_buf = np.frombuffer(fig_bar.canvas.buffer_rgba(), dtype=np.uint8)
            bw, bh = fig_bar.canvas.get_width_height()
            bar_rgba = bar_buf.reshape(bh, bw, 4)[:, :, :3]
            plt.close(fig_bar)
            # Resize bar to match frame width exactly
            if bar_rgba.shape[1] != w_each * 2:
                from PIL import Image as _PIL
                bar_rgba = np.array(_PIL.fromarray(bar_rgba).resize(
                    (w_each * 2, bar_h), _PIL.LANCZOS))
            row = np.concatenate([left, right], axis=1)
            row_labelled = np.concatenate([bar_rgba, row], axis=0)
            comparison.append(row_labelled.astype(np.uint8))

        comp_path = out / "comparison_flat_vs_symbiotic.gif"
        imageio.mimsave(str(comp_path), comparison, fps=args.fps, loop=0)
        print(f"Comparison GIF → {comp_path}  ({len(comparison)} frames)")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
