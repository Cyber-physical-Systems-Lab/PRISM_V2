"""
Generate symbiosis scenario animations.

Produces two GIF files in assets/:

  symbiosis_overview.gif   — one full episode; grid + live relationship matrix + stats panel
  symbiosis_scenarios.gif  — stitched clips (title card → event clip) for each of the five
                             relationship types detected during the run

Usage
-----
python experiments/generate_symbiosis_gifs.py               # defaults
python experiments/generate_symbiosis_gifs.py \\
    --env  tarware-small-4agvs-2pickers-partialobs-chg-symbiosis-v1 \\
    --seed 42 --steps 600 --fps 8 --output_dir assets

The script is headless (no display required) and uses matplotlib + imageio only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import tarware  # noqa: F401
import gymnasium as gym
from tarware.definitions import AgentType, Direction, PackageType
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

# ── Colour palette ────────────────────────────────────────────────────────────

# Shelf colours by PackageType (requested / not-requested variants)
_SHELF_COLORS = {
    PackageType.SOLO:        ("#5B9BD5", "#2E75B6"),   # cornflower  / steel blue
    PackageType.STANDARD:    ("#70AD47", "#375623"),   # lime green  / forest green
    PackageType.LARGE:       ("#FFC000", "#C09000"),   # amber       / dark amber
    PackageType.HEAVY:       ("#ED7D31", "#A85100"),   # orange      / burnt orange
    PackageType.PICKER_SOLO: ("#B96FDB", "#7B2FA8"),   # violet      / deep violet
}
_SHELF_DEFAULT = ("#8496A9", "#5A6A78")  # fallback for unknown type

_GOAL_COLOR   = "#2C2C2C"
_CHARGE_COLOR = "#B0E0E6"   # powder blue
_HIGHWAY_COLOR = "#F0EDE4"  # warm off-white
_GRID_COLOR    = "#D0CCC0"

# Agent colours indexed by agent position in agents list
_AGV_COLORS = [
    "#E67E22",  # carrot orange
    "#CA6F1E",  # dark orange
    "#F39C12",  # sunflower
    "#A04000",  # brown-orange
    "#D35400",  # pumpkin
]
_PICKER_COLORS = [
    "#2E86C1",  # cobalt blue
    "#1B4F72",  # dark navy
    "#76D7EA",  # sky blue
    "#154360",  # midnight
]

# Relationship ring colours
_REL_COLORS = {
    REL_MUTUALISM:    "#27AE60",  # emerald
    REL_COMMENSALISM: "#00BCD4",  # cyan
    REL_COMPETITION:  "#E74C3C",  # red
    REL_PARASITISM:   "#8E44AD",  # purple
    REL_NEUTRAL:      None,       # no ring
}
_REL_LABELS = {
    REL_MUTUALISM:    "Mutualism    (+/+)",
    REL_COMMENSALISM: "Commensalism (+/0)",
    REL_COMPETITION:  "Competition  (-/-)",
    REL_PARASITISM:   "Parasitism   (+/-)",
    REL_NEUTRAL:      "Neutral       (0/0)",
}

# ── Rendering helpers ─────────────────────────────────────────────────────────

def _cell_xy(col: int, row: int, rows: int, cell: float = 1.0) -> Tuple[float, float]:
    """Bottom-left corner of a grid cell in matplotlib coords (y-flipped)."""
    return col * cell, (rows - row - 1) * cell


def _dominant_rel(rel_state_ema: np.ndarray) -> int:
    """Return the dominant relationship type from a (n_partners, N_REL_TYPES) EMA."""
    if rel_state_ema.size == 0:
        return REL_NEUTRAL
    totals = rel_state_ema.sum(axis=0)
    dom = int(np.argmax(totals))
    # Only colour ring if mutualism/commensalism/competition/parasitism exceeds threshold
    if dom == REL_NEUTRAL or totals[dom] < 0.15:
        return REL_NEUTRAL
    return dom


def render_frame(
    raw_env,
    wrapper: SymbioticWrapper,
    step: int,
    deliveries_by_type: Dict[str, int],
    event_label: Optional[str] = None,
) -> np.ndarray:
    """Render one frame as a numpy uint8 RGB array (H×W×3).

    Layout: warehouse grid (left 70%) | info panel (right 30%).
    """
    rows, cols = raw_env.grid_size
    CELL = 1.0  # one matplotlib unit per grid cell

    fig_w = 16.0
    fig_h = max(6.0, rows * fig_w * 0.7 / cols)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=90)

    # ── Grid axis (left 70%) ─────────────────────────────────────────────────
    ax = fig.add_axes([0.0, 0.0, 0.70, 1.0])
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(_HIGHWAY_COLOR)

    # Shelf aisles background (non-highway cells)
    for y in range(rows):
        for x in range(cols):
            if not raw_env.highways[y, x]:
                bx, by = _cell_xy(x, y, rows)
                ax.add_patch(mpatches.Rectangle(
                    (bx, by), CELL, CELL,
                    fc="#E8E4D8", ec=_GRID_COLOR, lw=0.3, zorder=0
                ))

    # Grid lines
    for r in range(rows + 1):
        ax.axhline(r, color=_GRID_COLOR, lw=0.3, zorder=0)
    for c in range(cols + 1):
        ax.axvline(c, color=_GRID_COLOR, lw=0.3, zorder=0)

    # Goals
    for gx, gy in raw_env.goals:
        bx, by = _cell_xy(gx, gy, rows)
        ax.add_patch(mpatches.Rectangle(
            (bx, by), CELL, CELL, fc=_GOAL_COLOR, ec="none", zorder=1
        ))

    # Charging stations (highlight occupied ones)
    occupied_charger_positions = {
        (a.x, a.y)
        for a in raw_env.agents
        if (a.x, a.y) in raw_env._charging_station_positions
    }
    for station in raw_env.charging_stations:
        bx, by = _cell_xy(station.x, station.y, rows)
        occupied = (station.x, station.y) in occupied_charger_positions
        fc = "#FFD700" if occupied else _CHARGE_COLOR   # gold when in use
        ax.add_patch(mpatches.Rectangle(
            (bx + 0.05, by + 0.05), CELL - 0.1, CELL - 0.1,
            fc=fc, ec="#87CEEB", lw=1.0, zorder=1
        ))
        ax.text(
            bx + CELL * 0.5, by + CELL * 0.5, "⚡",
            ha="center", va="center", fontsize=6, zorder=2
        )

    # Shelves (coloured by package type and requested status)
    requested_ids = {s.id for s in raw_env.request_queue}
    for shelf in raw_env.shelfs:
        if any(a.carrying_shelf is shelf for a in raw_env.agents):
            continue  # carried shelves drawn with agent
        requested = shelf.id in requested_ids
        colors = _SHELF_COLORS.get(shelf.package_type, _SHELF_DEFAULT)
        fc = colors[0] if requested else colors[1]
        pad = 0.08
        bx, by = _cell_xy(shelf.x, shelf.y, rows)
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx + pad, by + pad), CELL - 2 * pad, CELL - 2 * pad,
            boxstyle="round,pad=0.02", fc=fc, ec="white" if requested else "none",
            lw=1.0 if requested else 0, zorder=2
        ))
        # Package type initial
        label = shelf.package_type.name[0]  # S / T / L / H / P
        ax.text(
            bx + CELL * 0.5, by + CELL * 0.5, label,
            ha="center", va="center", fontsize=4.5,
            color="white", fontweight="bold", zorder=3
        )

    # ── Agents ───────────────────────────────────────────────────────────────
    agv_indices    = [i for i, a in enumerate(raw_env.agents) if a.type == AgentType.AGV]
    picker_indices = [i for i, a in enumerate(raw_env.agents) if a.type == AgentType.PICKER]

    for i, agent in enumerate(raw_env.agents):
        is_agv   = agent.type == AgentType.AGV
        is_picker = agent.type == AgentType.PICKER

        if is_agv:
            local_idx = agv_indices.index(i)
            color = _AGV_COLORS[local_idx % len(_AGV_COLORS)]
        elif is_picker:
            local_idx = picker_indices.index(i)
            color = _PICKER_COLORS[local_idx % len(_PICKER_COLORS)]
        else:
            color = "#95A5A6"

        cx = agent.x + CELL * 0.5
        cy = (rows - agent.y - 1) + CELL * 0.5

        # Relationship ring (drawn first, behind agent)
        if i in wrapper._rel_states:
            dom = _dominant_rel(wrapper._rel_states[i].ema)
            ring_color = _REL_COLORS.get(dom)
            if ring_color:
                ring = plt.Circle(
                    (cx, cy), CELL * 0.44,
                    fc="none", ec=ring_color, lw=2.5, zorder=3
                )
                ax.add_patch(ring)

        # Agent body
        if is_agv:
            # Hexagon for AGVs
            theta = np.linspace(0, 2 * np.pi, 7)[:-1]
            r = CELL * 0.35
            xs = cx + r * np.cos(theta + np.pi / 6)
            ys = cy + r * np.sin(theta + np.pi / 6)
            poly = plt.Polygon(list(zip(xs, ys)), fc=color, ec="white", lw=1.0, zorder=4)
            ax.add_patch(poly)
        else:
            # Diamond for pickers
            diamond = plt.Polygon(
                [(cx, cy + CELL * 0.35), (cx + CELL * 0.35, cy),
                 (cx, cy - CELL * 0.35), (cx - CELL * 0.35, cy)],
                fc=color, ec="white", lw=1.0, zorder=4
            )
            ax.add_patch(diamond)

        # Direction indicator
        dir_offsets = {
            Direction.UP:    (0,  0.18),
            Direction.DOWN:  (0, -0.18),
            Direction.LEFT:  (-0.18, 0),
            Direction.RIGHT: (0.18,  0),
        }
        dx, dy = dir_offsets.get(agent.dir, (0, 0))
        ax.plot(cx + dx, cy + dy, "o", color="white", ms=2.5, zorder=5)

        # Carried shelf indicator
        if agent.carrying_shelf:
            shelf = agent.carrying_shelf
            colors = _SHELF_COLORS.get(shelf.package_type, _SHELF_DEFAULT)
            fc = colors[0]
            rect = mpatches.Rectangle(
                (agent.x + 0.05, (rows - agent.y - 1) + 0.05),
                CELL - 0.1, CELL - 0.1,
                fc=fc, alpha=0.35, ec=colors[0], lw=1.5, zorder=3
            )
            ax.add_patch(rect)
            if getattr(agent.carrying_shelf, "picker_assisted", False):
                ax.text(
                    cx, cy + CELL * 0.42, "✓",
                    ha="center", va="center", fontsize=5,
                    color=_REL_COLORS[REL_COMMENSALISM], fontweight="bold", zorder=6
                )

        # Agent ID label
        label = f"A{local_idx + 1}" if is_agv else f"P{local_idx + 1}"
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=5.5, color="white", fontweight="bold", zorder=6)

        # Battery bar (thin strip below agent)
        bat = agent.battery / 100.0
        bar_w = CELL * 0.7
        bar_h = CELL * 0.08
        bx_bar = agent.x + (CELL - bar_w) / 2
        by_bar = (rows - agent.y - 1) + 0.02
        # Background
        ax.add_patch(mpatches.Rectangle(
            (bx_bar, by_bar), bar_w, bar_h, fc="#555", zorder=5
        ))
        # Fill
        bat_color = "#27AE60" if bat > 0.6 else ("#F1C40F" if bat > 0.3 else "#E74C3C")
        ax.add_patch(mpatches.Rectangle(
            (bx_bar, by_bar), bar_w * bat, bar_h, fc=bat_color, zorder=6
        ))

    # Event highlight
    if event_label:
        ax.text(
            cols * 0.5, rows - 0.3, event_label,
            ha="center", va="top", fontsize=10, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1A1A2E", alpha=0.85),
            zorder=10
        )

    # ── Info panel (right 30%) ────────────────────────────────────────────────
    ax2 = fig.add_axes([0.71, 0.0, 0.29, 1.0])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_facecolor("#1A1A2E")
    fig.patch.set_facecolor("#1A1A2E")

    def ptext(x, y, txt, **kw):
        defaults = dict(transform=ax2.transAxes, fontsize=8,
                        color="white", va="top")
        defaults.update(kw)
        ax2.text(x, y, txt, **defaults)

    ptext(0.05, 0.97, f"Step {step}", fontsize=11, fontweight="bold", color="#ECF0F1")
    ptext(0.05, 0.91, "Deliveries", fontsize=9, fontweight="bold", color="#BDC3C7")

    y_del = 0.87
    pkg_colors_hex = {
        "SOLO":        _SHELF_COLORS[PackageType.SOLO][0],
        "STANDARD":    _SHELF_COLORS[PackageType.STANDARD][0],
        "LARGE":       _SHELF_COLORS[PackageType.LARGE][0],
        "HEAVY":       _SHELF_COLORS[PackageType.HEAVY][0],
        "PICKER_SOLO": _SHELF_COLORS[PackageType.PICKER_SOLO][0],
    }
    for pkg, count in deliveries_by_type.items():
        dot_color = pkg_colors_hex.get(pkg, "#aaa")
        ax2.add_patch(mpatches.Circle((0.06, y_del - 0.005), 0.018,
                                       fc=dot_color, transform=ax2.transAxes, zorder=3))
        ptext(0.12, y_del, f"{pkg[:4]:<4}  {count:>3}", fontsize=7.5, color="#ECF0F1")
        y_del -= 0.055

    # Relationship matrix
    n_agvs   = len(agv_indices)
    n_pickers = len(picker_indices)

    y_mat = y_del - 0.03
    ptext(0.05, y_mat, "Relationships", fontsize=9, fontweight="bold", color="#BDC3C7")
    y_mat -= 0.05

    cell_w = min(0.18, 0.85 / max(n_agvs, 1))
    cell_h = 0.065

    # Column headers (AGVs)
    for ai, agv_i in enumerate(agv_indices):
        ax2.text(
            0.12 + ai * cell_w + cell_w * 0.5, y_mat,
            f"A{ai + 1}", ha="center", va="top",
            transform=ax2.transAxes, fontsize=6.5, color="#BDC3C7"
        )
    y_mat -= 0.045

    # Rows (pickers × AGVs)
    for pi, picker_i in enumerate(picker_indices):
        ax2.text(
            0.03, y_mat - cell_h * 0.5, f"P{pi + 1}",
            ha="left", va="center",
            transform=ax2.transAxes, fontsize=6.5, color="#BDC3C7"
        )
        for ai, agv_i in enumerate(agv_indices):
            # Get relationship from picker's perspective toward this AGV
            rs = wrapper._rel_states.get(picker_i)
            if rs is not None and rs.n_partners > ai:
                rel = int(np.argmax(rs.ema[ai]))
                threshold = rs.ema[ai].max()
            else:
                rel = REL_NEUTRAL
                threshold = 0.0

            fc = _REL_COLORS.get(rel) or "#2C3E50"
            ax2.add_patch(mpatches.FancyBboxPatch(
                (0.12 + ai * cell_w, y_mat - cell_h),
                cell_w * 0.88, cell_h * 0.85,
                boxstyle="round,pad=0.01",
                fc=fc, ec="#ECF0F1", lw=0.3,
                transform=ax2.transAxes, zorder=3
            ))
            ax2.text(
                0.12 + ai * cell_w + cell_w * 0.44,
                y_mat - cell_h * 0.5,
                REL_NAMES[rel][:3].upper(),
                ha="center", va="center",
                transform=ax2.transAxes, fontsize=5.0,
                color="white", fontweight="bold"
            )
        y_mat -= cell_h + 0.01

    # Legend
    y_leg = max(0.12, y_mat - 0.04)
    ptext(0.05, y_leg, "Legend", fontsize=8, fontweight="bold", color="#BDC3C7")
    y_leg -= 0.045
    for rel_id, label in _REL_LABELS.items():
        fc = _REL_COLORS.get(rel_id) or "#2C3E50"
        ax2.add_patch(mpatches.Rectangle(
            (0.05, y_leg - 0.018), 0.06, 0.022,
            fc=fc, transform=ax2.transAxes, zorder=3
        ))
        ptext(0.14, y_leg, label, fontsize=6.0, color="#BDC3C7")
        y_leg -= 0.04

    # Agent type legend
    ax2.add_patch(plt.Polygon(
        [(0.065, y_leg), (0.10, y_leg - 0.018), (0.065, y_leg - 0.035),
         (0.03, y_leg - 0.018)],
        fc=_PICKER_COLORS[0], transform=ax2.transAxes, zorder=3
    ))
    ptext(0.14, y_leg - 0.005, "Picker (diamond)", fontsize=6.0, color="#BDC3C7")
    y_leg -= 0.04

    theta = np.linspace(0, 2 * np.pi, 7)[:-1]
    hex_xs = [0.065 + 0.03 * np.cos(t + np.pi / 6) for t in theta]
    hex_ys = [y_leg - 0.018 + 0.02 * np.sin(t + np.pi / 6) for t in theta]
    ax2.add_patch(plt.Polygon(
        list(zip(hex_xs, hex_ys)),
        fc=_AGV_COLORS[0], transform=ax2.transAxes, zorder=3
    ))
    ptext(0.14, y_leg - 0.005, "AGV (hexagon)", fontsize=6.0, color="#BDC3C7")

    # Convert figure to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3]   # drop alpha
    plt.close(fig)
    return arr.astype(np.uint8)


def _title_card(title: str, subtitle: str, rel_id: int, size=(1440, 810)) -> np.ndarray:
    """Solid colour title card frame."""
    fc = _REL_COLORS.get(rel_id) or "#2C3E50"
    fig = plt.figure(figsize=(size[0] / 90, size[1] / 90), dpi=90)
    fig.patch.set_facecolor(fc if fc else "#2C3E50")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_facecolor(fc if fc else "#2C3E50")
    ax.text(0.5, 0.58, title, ha="center", va="center",
            fontsize=28, fontweight="bold", color="white",
            transform=ax.transAxes)
    ax.text(0.5, 0.42, subtitle, ha="center", va="center",
            fontsize=16, color="white", alpha=0.9,
            transform=ax.transAxes)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    arr = np.asarray(canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return arr.astype(np.uint8)


# ── Main runner ───────────────────────────────────────────────────────────────

def _smart_actions(raw_env) -> List[int]:
    """Heuristic-biased action selection for richer event coverage in animations.

    Strategy:
    - AGVs prefer STANDARD shelves (highest cooperative reward) when pickers are free.
    - Pickers follow AGV targets when valid, otherwise target PICKER_SOLO shelves.
    - Agents with battery < 25 go to charger.
    - Falls back to random valid action when no heuristic applies.
    """
    from tarware.definitions import AgentType, PackageType
    masks = raw_env.compute_valid_action_masks()
    actions = []

    agv_targets = {}   # agv_idx → chosen shelf action_id

    for i, agent in enumerate(raw_env.agents):
        valid = list(np.where(masks[i] > 0)[0])
        if not valid:
            actions.append(0)
            continue

        # Low battery → charge
        if agent.battery < 25:
            charger_actions = []
            for v in valid:
                if v > 0:
                    coords = raw_env.action_id_to_coords_map.get(v)
                    if coords and (coords[1], coords[0]) in raw_env._charging_station_positions:
                        charger_actions.append(v)
            if charger_actions:
                actions.append(int(np.random.choice(charger_actions)))
                continue

        if agent.type == AgentType.AGV:
            # Prefer STANDARD shelves when picker is available; else HEAVY
            shelf_actions = []
            for v in valid:
                if v > len(raw_env.goals):  # shelf action
                    coords = raw_env.action_id_to_coords_map.get(v)
                    if coords:
                        shelf_id = raw_env.grid[2, coords[0], coords[1]]  # SHELVES layer
                        if shelf_id:
                            shelf = raw_env.shelfs[shelf_id - 1]
                            if shelf in raw_env.request_queue:
                                shelf_actions.append((v, shelf.package_type))
            # Prioritise: STANDARD > HEAVY > SOLO > others
            priority_map = {PackageType.STANDARD: 0, PackageType.HEAVY: 1,
                            PackageType.SOLO: 2, PackageType.LARGE: 3}
            if shelf_actions:
                shelf_actions.sort(key=lambda x: priority_map.get(x[1], 9))
                chosen = shelf_actions[0][0]
                agv_targets[i] = chosen
                actions.append(chosen)
                continue

        elif agent.type == AgentType.PICKER:
            # Categorise available picker targets.
            standard_targets = []   # STANDARD shelves an AGV is heading for
            picker_solo = []        # PICKER_SOLO-exclusive shelves
            for v in valid:
                if v <= len(raw_env.goals):
                    continue
                coords = raw_env.action_id_to_coords_map.get(v)
                if not coords:
                    continue
                shelf_id = raw_env.grid[2, coords[0], coords[1]]
                if shelf_id:
                    shelf = raw_env.shelfs[shelf_id - 1]
                    if shelf not in raw_env.request_queue:
                        continue
                    if shelf.package_type == PackageType.PICKER_SOLO:
                        picker_solo.append(v)
                    elif shelf.package_type == PackageType.STANDARD:
                        # Only go if an AGV is already targeting this shelf.
                        for agv in raw_env.agents[:raw_env.num_agvs]:
                            agv_dest = raw_env.action_id_to_coords_map.get(agv.target)
                            if agv_dest == coords:
                                standard_targets.append(v)
                                break

            # Preference: STANDARD (follow AGV) > PICKER_SOLO (30% chance) > random
            if standard_targets:
                actions.append(int(np.random.choice(standard_targets)))
                continue
            # 30% chance picker goes for PICKER_SOLO to create neutralism events
            if picker_solo and np.random.random() < 0.3:
                actions.append(int(np.random.choice(picker_solo)))
                continue

        actions.append(int(np.random.choice(valid)))

    return actions


def run_episode(wrapper: SymbioticWrapper, raw_env, seed: int, max_steps: int):
    """Run one episode, returning (frames, rel_events, cumulative_deliveries_per_step)."""
    obs, _ = wrapper.reset(seed=seed)
    # Start agents at full battery so they can complete deliveries from the first frame.
    for agent in raw_env.agents:
        agent.battery = 100.0
    wrapper._prev_batteries = np.array([a.battery for a in raw_env.agents], dtype=np.float32)
    cumulative = {t.name: 0 for t in PackageType}

    frames: List[np.ndarray] = []
    rel_events: List[Tuple[int, int]] = []
    step_deliveries: List[Dict] = []

    n_pairs = len(wrapper._agv_indices) * len(wrapper._picker_indices)

    for step in range(max_steps):
        actions = _smart_actions(raw_env)
        obs, rewards, term, trunc, info = wrapper.step(actions)

        for k, v in info.get("deliveries_by_pkg_type", {}).items():
            cumulative[k] = cumulative.get(k, 0) + v
        step_deliveries.append(dict(cumulative))

        # Record all non-neutral relationship events this step
        log_tail = wrapper._rel_log[-n_pairs:] if wrapper._rel_log else []
        for entry in log_tail:
            if entry["rel"] != REL_NEUTRAL:
                rel_events.append((step, entry["rel"]))

        # Detect event label for overlay
        event_label = None
        if log_tail:
            unique_rels = {e["rel"] for e in log_tail if e["rel"] != REL_NEUTRAL}
            if unique_rels:
                dom = min(unique_rels)  # lowest enum = most interesting
                event_label = _REL_LABELS.get(dom)

        frame = render_frame(raw_env, wrapper, step, dict(cumulative), event_label)
        frames.append(frame)

        if any(term) or any(trunc):
            break

    return frames, rel_events, step_deliveries


def extract_clip(frames: List[np.ndarray], event_step: int,
                 before: int = 15, after: int = 25) -> List[np.ndarray]:
    start = max(0, event_step - before)
    end   = min(len(frames), event_step + after + 1)
    return frames[start:end]


def _resize_frames(frames: List[np.ndarray], target_shape) -> List[np.ndarray]:
    """Resize all frames to the same HxW so imageio can stack them."""
    from PIL import Image
    h, w = target_shape[:2]
    out = []
    for f in frames:
        if f.shape[:2] != (h, w):
            img = Image.fromarray(f).resize((w, h), Image.LANCZOS)
            f = np.array(img)
        out.append(f)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate symbiosis GIF animations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        default="tarware-small-4agvs-2pickers-partialobs-chg-symbiosis-v1",
    )
    parser.add_argument("--seed",       default=42,    type=int)
    parser.add_argument("--steps",      default=600,   type=int,
                        help="Max steps per episode")
    parser.add_argument("--fps",        default=6,     type=int,
                        help="Frames per second for GIFs")
    parser.add_argument("--output_dir", default="assets")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building env: {args.env}")
    # Override max_inactivity_steps so the episode doesn't terminate before
    # deliveries can happen (registered default is 100, far too short for heuristic agents).
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

    np.random.seed(args.seed)

    # ── Run episode ───────────────────────────────────────────────────────────
    print(f"Running episode (max {args.steps} steps) …")
    frames, rel_events, _ = run_episode(env, raw, args.seed, args.steps)
    print(f"  {len(frames)} frames captured, {len(rel_events)} non-neutral relationship events")

    # ── 1. Overview GIF ───────────────────────────────────────────────────────
    overview_path = out_dir / "symbiosis_overview.gif"
    print(f"Saving overview GIF → {overview_path}")
    imageio.mimsave(str(overview_path), frames, fps=args.fps, loop=0)

    # ── 2. Scenario GIF ───────────────────────────────────────────────────────
    # Collect one representative clip per relationship type
    scenario_clips: List[np.ndarray] = []
    found_types = set()

    rel_type_meta = {
        REL_MUTUALISM:    ("Mutualism  (+/+)",    "AGV + Picker cooperate on STANDARD task — both earn delivery credit"),
        REL_COMMENSALISM: ("Commensalism  (+/0)", "Picker assists HEAVY task — AGV gains energy discount, picker unchanged"),
        REL_COMPETITION:  ("Competition  (-/-)",  "Multiple agents contest the same charging station simultaneously"),
        REL_PARASITISM:   ("Parasitism  (+/-)",   "One agent benefits while another expends energy with no return"),
    }

    for step_idx, rel_type in rel_events:
        if rel_type in found_types or rel_type == REL_NEUTRAL:
            continue
        if rel_type not in rel_type_meta:
            continue

        title, subtitle = rel_type_meta[rel_type]
        card = _title_card(title, subtitle, rel_type, size=(frames[0].shape[1], frames[0].shape[0]))
        card_frames = [card] * max(1, args.fps)   # 1-second title card

        clip = extract_clip(frames, step_idx, before=12, after=20)
        if not clip:
            continue

        scenario_clips.extend(card_frames)
        scenario_clips.extend(clip)
        scenario_clips.append(clip[-1])            # 1-frame pause after clip
        found_types.add(rel_type)

    # Add neutralism: pick a step where all agents are just doing solo work
    neutral_step = args.steps // 4  # early episode — agents scattered, no interaction
    neutral_clip = extract_clip(frames, min(neutral_step, len(frames) - 1), before=8, after=15)
    if neutral_clip:
        card = _title_card(
            "Neutralism  (0/0)",
            "Agents operate independently — SOLO and PICKER_SOLO tasks; no interaction",
            REL_NEUTRAL,
            size=(frames[0].shape[1], frames[0].shape[0]),
        )
        scenario_clips.extend([card] * max(1, args.fps))
        scenario_clips.extend(neutral_clip)

    if scenario_clips:
        # Normalise all frames to same shape
        target_shape = frames[0].shape
        scenario_clips = _resize_frames(scenario_clips, target_shape)

        scenario_path = out_dir / "symbiosis_scenarios.gif"
        print(f"Saving scenarios GIF → {scenario_path}  ({len(scenario_clips)} frames)")
        imageio.mimsave(str(scenario_path), scenario_clips, fps=args.fps, loop=0)
        print(f"  Relationship types captured: {[REL_NAMES[r] for r in sorted(found_types)]}")
    else:
        print("  No non-neutral events detected — try a longer --steps or different --seed")

    print("Done.")


if __name__ == "__main__":
    main()
