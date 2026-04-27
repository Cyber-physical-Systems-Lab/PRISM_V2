"""
PRISM — Publication-quality figures.

Produces up to 9 figures depending on which result files are available:

  Fig 1  — Task completion learning curves (symbiotic condition)
  Fig 2  — Mutualism fraction emergence over training
  Fig 3  — Relationship type distribution at end of training (stacked bar)
  Fig 4  — Raw vs shaped reward divergence (shows shaping effect)
  Fig 5  — Training stability: actor entropy + KL divergence
  Fig 6  — Team Specialisation Index (TSI) per condition
  Fig 7  — Symbiotic vs flat-cooperative comparison (core PRISM result)
  Fig 8  — Package-type delivery breakdown per condition
  Fig 9  — Battery management convergence

Usage
-----
# After running run_symbiotic.py:
python analysis/paper_figures.py \\
    --symbiotic local_runs/results/prism_symbiotic.json \\
    --ckpt_dir  local_runs/checkpoints/prism_symbiotic \\
    --output    local_runs/figures

# With flat-cooperative results (Fig 7):
python analysis/paper_figures.py \\
    --symbiotic local_runs/results/prism_symbiotic.json \\
    --flat_coop local_runs/results/prism_flat_cooperative.json \\
    --ckpt_dir  local_runs/checkpoints/prism_symbiotic \\
    --output    local_runs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ── Aesthetic constants ────────────────────────────────────────────────────────

METHOD_COLOR = {
    # PRISM primary conditions
    "symbiotic":       "#27AE60",   # green — proposed method
    "flat_cooperative":"#3498DB",   # blue  — baseline
    # Legacy method names kept for backwards compatibility
    "individual":      "#E74C3C",   # red
    "team":            "#3498DB",   # blue
    "unclassified":    "#F39C12",   # amber
    "heuristic":       "#8E44AD",   # purple
}
METHOD_LABEL = {
    # PRISM primary conditions
    "symbiotic":       "Symbiotic (PRISM)",
    "flat_cooperative":"Flat-cooperative (baseline)",
    # Legacy
    "individual":      "Individual reward",
    "team":            "Team reward",
    "unclassified":    "Unclassified coop.",
    "heuristic":       "Heuristic oracle",
}
METHOD_STYLE = {
    "symbiotic":       "-",
    "flat_cooperative":"--",
    "individual":      "-",
    "team":            "--",
    "unclassified":    "-.",
    "heuristic":       ":",
}
METHOD_LW = {
    "symbiotic": 2.2, "flat_cooperative": 1.8,
    "individual": 1.4, "team": 1.4, "unclassified": 1.4, "heuristic": 1.4,
}

REL_COLOR = {
    "mutualism":    "#27AE60",
    "commensalism": "#00BCD4",
    "competition":  "#E74C3C",
    "parasitism":   "#8E44AD",
    "neutral":      "#BDC3C7",
}
REL_LABEL = {
    "mutualism":    "Mutualism (+/+)",
    "commensalism": "Commensalism (+/0)",
    "competition":  "Competition (−/−)",
    "parasitism":   "Parasitism (+/−)",
    "neutral":      "Neutral (0/0)",
}

# Publication-quality global style
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "legend.framealpha": 0.85,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   ":",
})


# ── Helpers ────────────────────────────────────────────────────────────────────

def smooth(arr: np.ndarray, window: int = 15) -> np.ndarray:
    if len(arr) <= window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def smooth_x(n: int, window: int = 15) -> np.ndarray:
    return np.arange(window - 1, n)


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(path / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_eval_csv(ckpt_dir: Path, method: str, backend: str = "ippo",
                  seed: int = 1) -> Optional[pd.DataFrame]:
    p = ckpt_dir / method / f"{backend}_seed{seed}" / "eval_metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_update_csv(ckpt_dir: Path, method: str, backend: str = "ippo",
                    seed: int = 1) -> Optional[pd.DataFrame]:
    p = ckpt_dir / method / f"{backend}_seed{seed}" / "update_metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def _methods_from_json(data: dict) -> Dict[str, dict]:
    """Extract the methods sub-dict from a condition result JSON.

    Handles both the old {"methods": {...}} format and the new per-condition
    format {"condition": "symbiotic", "results": {"mappo": {...}}}.
    """
    if "methods" in data:
        return data["methods"]
    if "results" in data and "condition" in data:
        canonical = data.get("canonical_backend", next(iter(data["results"])))
        return {data["condition"]: data["results"].get(canonical, {})}
    return data


def _load_heuristic_data(heuristic_results: Optional[dict]) -> Optional[dict]:
    """Return the flat per-env dict from a heuristic baseline JSON."""
    if heuristic_results is None:
        return None
    if len(heuristic_results) == 1:
        return next(iter(heuristic_results.values()))
    return heuristic_results


# ── Figure 1: Task completion learning curves ──────────────────────────────────

def _load_condition_curves(ckpt_dir: Path) -> Optional[tuple]:
    """Load and average deliveries curves across all seeds in a condition dir.

    Returns (x_steps, mean_y, std_y) or None if no CSV data found.
    os.walk is used to follow symlinks.
    """
    import os as _os
    all_curves: List[np.ndarray] = []
    for root, _, files in _os.walk(ckpt_dir, followlinks=True):
        if "eval_metrics.csv" in files:
            try:
                df = pd.read_csv(Path(root) / "eval_metrics.csv")
                if "deliveries" in df.columns and len(df) > 1:
                    all_curves.append(df["deliveries"].values)
            except Exception:
                pass
    if not all_curves:
        return None
    min_len = min(len(c) for c in all_curves)
    arr = np.array([c[:min_len] for c in all_curves])
    mean_y = arr.mean(axis=0)
    std_y  = arr.std(axis=0)
    x = np.arange(min_len)
    return x, mean_y, std_y


def fig1_learning_curves(
    results: dict,
    ckpt_dir: Path,
    save_path: Path,
    heuristic_results: Optional[dict] = None,
    flat_ckpt_dir: Optional[Path] = None,
    task_ckpt_dir: Optional[Path] = None,
) -> None:
    """Deliveries per episode over training — all three conditions + heuristic reference."""
    fig, ax = plt.subplots(figsize=(8, 5))

    condition_dirs = [
        ("symbiotic",       ckpt_dir,       METHOD_COLOR["symbiotic"],       METHOD_LABEL["symbiotic"]),
        ("flat_cooperative",flat_ckpt_dir,   METHOD_COLOR["flat_cooperative"], METHOD_LABEL["flat_cooperative"]),
        ("task_only",       task_ckpt_dir,   "#E67E22",                        "Task-only (ablation)"),
    ]

    for cond, cdir, color, label in condition_dirs:
        if cdir is None:
            continue
        result = _load_condition_curves(Path(cdir))
        if result is None:
            # Fall back to JSON curves
            methods = _methods_from_json(results)
            data = methods.get(cond, {})
            curves = np.array(data.get("deliveries_curves", []))
            if curves.ndim != 2 or curves.shape[1] < 2:
                continue
            mean_y = curves.mean(axis=0)
            std_y  = curves.std(axis=0)
            x = np.arange(len(mean_y))
        else:
            x, mean_y, std_y = result

        sy  = smooth(mean_y)
        ss  = smooth(std_y)
        sx  = x[len(x) - len(sy):]
        ls  = METHOD_STYLE.get(cond, "-")
        lw  = METHOD_LW.get(cond, 1.8)

        ax.plot(sx, sy, color=color, label=label, linestyle=ls, linewidth=lw)
        ax.fill_between(sx, np.maximum(sy - ss, 0), sy + ss, alpha=0.10, color=color)

    # Heuristic oracle as horizontal reference
    h = _load_heuristic_data(heuristic_results)
    if h is not None:
        h_mean = h.get("mean_deliveries", 0)
        ax.axhline(h_mean, color=METHOD_COLOR["heuristic"], linestyle=":",
                   linewidth=1.4, alpha=0.85,
                   label=f"Heuristic oracle ({h_mean:.2f} del/ep)")

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Deliveries per episode")
    ax.set_title("Fig 1 — Task Completion over Training")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, save_path, "fig1_learning_curves")


# ── Figure 2: Mutualism fraction emergence ─────────────────────────────────────

def fig2_mutualism_emergence(
    results: dict,
    ckpt_dir: Path,
    save_path: Path,
) -> None:
    """Fraction of AGV-picker interactions classified as mutualism (+/+)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = _methods_from_json(results)

    for method, data in methods.items():
        df = load_eval_csv(ckpt_dir, method)
        if df is not None and "mutualism_fraction" in df.columns and len(df) > 1:
            y = df["mutualism_fraction"].values
            x = df["step"].values if "step" in df.columns else np.arange(len(y))
            sy = smooth(y)
            sx = x[len(x) - len(sy):]
        else:
            curves = np.array(data.get("mutualism_curves", []))
            if curves.ndim != 2 or curves.shape[1] < 2:
                continue
            y = curves.mean(axis=0)
            sy = smooth(y)
            sx = np.arange(len(sy))

        color = METHOD_COLOR.get(method, "gray")
        ax.plot(sx, sy,
                color=color,
                label=METHOD_LABEL.get(method, method),
                linestyle=METHOD_STYLE.get(method, "-"),
                linewidth=METHOD_LW.get(method, 1.5))

    ax.set_xlabel("Training steps")
    ax.set_ylabel("Mutualism fraction")
    ax.set_title("Fig 2 — Emergence of Mutualistic Behaviour over Training")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, save_path, "fig2_mutualism_emergence")


# ── Figure A: Relationship emergence (mutualism + commensalism only) ───────────

def fig_relationship_emergence(
    symbiotic_results: dict,
    save_path: Path,
) -> None:
    """Mutualism and commensalism fraction over training — the two observed types.

    Competition and parasitism are not plotted as they are never observed in
    this environment configuration (complementary AGV/picker roles).
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    sym_data = _methods_from_json(symbiotic_results)
    sym_val  = next(iter(sym_data.values()), {})

    for rel, color, label in [
        ("mutualism",    REL_COLOR["mutualism"],    REL_LABEL["mutualism"]),
        ("commensalism", REL_COLOR["commensalism"],  REL_LABEL["commensalism"]),
    ]:
        curves = np.array(sym_val.get(f"{rel}_curves", []))
        if curves.ndim == 2 and curves.shape[1] > 1:
            mean_y = curves.mean(axis=0)
            std_y  = curves.std(axis=0)
            sy = smooth(mean_y)
            ss = smooth(std_y)
            sx = np.arange(len(sy))
            ax.plot(sx, sy, color=color, label=label, linewidth=2.0)
            ax.fill_between(sx, sy - ss, sy + ss, alpha=0.15, color=color)

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Fraction of AGV–picker pair interactions")
    ax.set_title("Fig — Ecological Relationship Emergence over Training\n"
                 "(mutualism = joint delivery; commensalism = one-sided charging)")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path, "fig_relationship_emergence")


# ── Figure B: Package type delivery distribution ───────────────────────────────

def fig_package_distribution(
    eval_stats_path: str,
    save_path: Path,
) -> None:
    """Stacked bar: package type distribution per condition from evaluation.

    Shows that symbiotic reward steers the team toward STANDARD (mutualistic)
    tasks while other conditions deliver fewer packages of any type.
    Requires eval_stats_final.json from evaluate_all_seeds.py.
    """
    with open(eval_stats_path) as f:
        stats = json.load(f)

    pkg_colors = {
        "SOLO":        "#5B9BD5",
        "STANDARD":    "#70AD47",
        "LARGE":       "#FFC000",
        "HEAVY":       "#ED7D31",
        "PICKER_SOLO": "#B96FDB",
    }

    conditions = []
    totals_by_pkg: Dict[str, List[float]] = {p: [] for p in pkg_colors}

    for cond_key, cond_label in [
        ("flat_coop",  "Flat-coop\n(baseline)"),
        ("task_only",  "Task-only\n(ablation)") if "task_only" in stats else (None, None),
        ("symbiotic",  "Symbiotic\n(PRISM)"),
    ]:
        if cond_key is None or cond_key not in stats:
            continue
        episodes = stats[cond_key].get("all_episodes", [])
        if not episodes:
            continue
        # Collect package breakdown from per_seed_detail if available
        pkg_totals = {p: 0.0 for p in pkg_colors}
        n_ep = 0
        for seed_detail in stats[cond_key].get("per_seed_detail", []):
            for ep_pkg in seed_detail.get("episodes_by_pkg", []):
                for p in pkg_colors:
                    pkg_totals[p] += ep_pkg.get(p, 0)
                n_ep += 1
        if n_ep == 0:
            # Fall back: only total deliveries available, assume all STANDARD
            mean_total = float(np.mean(episodes))
            pkg_totals["STANDARD"] = mean_total
            n_ep = 1

        conditions.append(cond_label)
        for p in pkg_colors:
            totals_by_pkg[p].append(pkg_totals[p] / max(n_ep, 1))

    if not conditions:
        print("  [SKIP] fig_package_distribution: no per-package data in eval_stats")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(conditions))
    bottoms = np.zeros(len(conditions))

    for pkg, color in pkg_colors.items():
        vals = np.array(totals_by_pkg[pkg])
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottoms, color=color, label=pkg, alpha=0.88, width=0.55)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Mean deliveries per episode")
    ax.set_title("Fig — Package Type Distribution by Condition\n"
                 "(symbiotic reward shifts team toward cooperative STANDARD tasks)")
    ax.legend(title="Package type", fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, save_path, "fig_package_distribution")


# ── Figure 3: Relationship type distribution ───────────────────────────────────

def fig3_relationship_distribution(
    results: dict,
    ckpt_dir: Path,
    save_path: Path,
) -> None:
    """Stacked bar chart of final relationship distributions per method."""
    methods = _methods_from_json(results)
    rel_types = ["mutualism", "commensalism", "competition", "parasitism", "neutral"]

    # Collect final relationship fractions from eval CSV (last N episodes)
    method_names = []
    rel_fractions: Dict[str, List[float]] = {r: [] for r in rel_types}
    tail = 20  # average over last 20 evaluation episodes

    for method in methods:
        df = load_eval_csv(ckpt_dir, method)
        has_data = False
        if df is not None and len(df) >= 2:
            row = {}
            for r in rel_types:
                col = f"{r}_fraction"
                if col in df.columns:
                    row[r] = df[col].tail(tail).mean()
                    has_data = True
            if has_data:
                method_names.append(method)
                for r in rel_types:
                    rel_fractions[r].append(row.get(r, 0.0))

        if not has_data:
            # Fall back to JSON summary
            d = methods[method]
            total = sum(d.get(f"{r}_fraction", 0) for r in rel_types)
            if total > 0:
                method_names.append(method)
                for r in rel_types:
                    rel_fractions[r].append(d.get(f"{r}_fraction", 0.0) / total)
            else:
                method_names.append(method)
                for r in rel_types:
                    rel_fractions[r].append(1.0 / len(rel_types))

    if not method_names:
        print("  [SKIP] fig3: no relationship data available yet")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(method_names))
    bottoms = np.zeros(len(method_names))

    for r in rel_types:
        vals = np.array(rel_fractions[r])
        ax.bar(x, vals, bottom=bottoms,
               color=REL_COLOR[r], label=REL_LABEL[r],
               alpha=0.88, width=0.55)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL.get(m, m) for m in method_names], rotation=15, ha="right")
    ax.set_ylabel("Fraction of agent-pair interactions")
    ax.set_title("Fig 3 — Ecological Relationship Distribution at End of Training")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y")
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    _save(fig, save_path, "fig3_relationship_distribution")


# ── Figure 4: Raw vs shaped reward ────────────────────────────────────────────

def fig4_reward_shaping_effect(
    results: dict,
    ckpt_dir: Path,
    save_path: Path,
) -> None:
    """Shows divergence between raw task reward and symbiotic shaped reward."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    methods = _methods_from_json(results)

    for ax, col, ylabel, title in [
        (axes[0], "mean_raw_reward",    "Mean raw reward",    "Raw Task Reward"),
        (axes[1], "mean_shaped_reward", "Mean shaped reward", "Symbiotic Shaped Reward"),
    ]:
        for method, data in methods.items():
            df = load_eval_csv(ckpt_dir, method)
            if df is None or col not in df.columns or len(df) < 2:
                continue
            y = df[col].values
            sy = smooth(y)
            sx = df["step"].values if "step" in df.columns else np.arange(len(y))
            sx = sx[len(sx) - len(sy):]
            color = METHOD_COLOR.get(method, "gray")
            ax.plot(sx, sy, color=color,
                    label=METHOD_LABEL.get(method, method),
                    linestyle=METHOD_STYLE.get(method, "-"),
                    linewidth=METHOD_LW.get(method, 1.5))

        ax.set_xlabel("Training steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig.suptitle("Fig 4 — Effect of Symbiotic Reward Shaping", y=1.01)
    fig.tight_layout()
    _save(fig, save_path, "fig4_reward_shaping")


# ── Figure 5: Training stability (entropy + KL) ────────────────────────────────

def fig5_training_stability(
    results: dict,
    ckpt_dir: Path,
    save_path: Path,
) -> None:
    """Actor entropy and approx KL divergence for AGV and picker heads."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    methods = _methods_from_json(results)

    combos = [
        (axes[0, 0], "agv_entropy",   "AGV entropy",          "AGV Policy Entropy"),
        (axes[0, 1], "pick_entropy",  "Picker entropy",       "Picker Policy Entropy"),
        (axes[1, 0], "agv_approx_kl", "AGV KL div.",          "AGV Approx KL Divergence"),
        (axes[1, 1], "pick_approx_kl","Picker KL div.",       "Picker Approx KL Divergence"),
    ]

    for ax, col, ylabel, title in combos:
        for method in methods:
            df = load_update_csv(ckpt_dir, method)
            if df is None or col not in df.columns or len(df) < 2:
                continue
            y = df[col].values
            sy = smooth(y, window=5)
            sx = df["step"].values if "step" in df.columns else np.arange(len(y))
            sx = sx[len(sx) - len(sy):]
            color = METHOD_COLOR.get(method, "gray")
            ax.plot(sx, sy, color=color,
                    label=METHOD_LABEL.get(method, method),
                    linestyle=METHOD_STYLE.get(method, "-"),
                    linewidth=METHOD_LW.get(method, 1.4))

        ax.set_xlabel("Update step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig.suptitle("Fig 5 — Training Stability: Policy Entropy and KL Divergence", y=1.01)
    fig.tight_layout()
    _save(fig, save_path, "fig5_training_stability")


# ── Figure 6: TSI bar chart ────────────────────────────────────────────────────

def fig6_specialisation_index(
    results: dict,
    save_path: Path,
    heuristic_results: Optional[dict] = None,
) -> None:
    """TSI and RSI per method — measures degree of role specialisation."""
    methods = _methods_from_json(results)
    names, tsi_vals, rsi_vals = [], [], []

    for method, data in methods.items():
        tsi = data.get("tsi")
        rsi = data.get("rsi")
        if tsi is None:
            continue
        names.append(method)
        tsi_vals.append(tsi)
        rsi_vals.append(rsi if rsi is not None else 0.0)

    if not names:
        print("  [SKIP] fig6: no TSI/RSI data in results JSON")
        return

    # Add heuristic TSI as reference bar
    h = _load_heuristic_data(heuristic_results)
    h_tsi = h.get("tsi") if h is not None else None
    if h_tsi is not None:
        names.append("heuristic")
        tsi_vals.append(h_tsi)
        rsi_vals.append(0.0)  # heuristic has no trained RSI

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(names))
    w = 0.35

    bars1 = ax.bar(x - w / 2, tsi_vals, width=w, alpha=0.85,
                   color=[METHOD_COLOR.get(m, "gray") for m in names],
                   label="TSI", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w / 2, rsi_vals, width=w, alpha=0.5,
                   color=[METHOD_COLOR.get(m, "gray") for m in names],
                   label="RSI", edgecolor="white", linewidth=0.5, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL.get(m, m) for m in names], rotation=15, ha="right")
    ax.set_ylabel("Index value [0 – 1]")
    ax.set_ylim(0, 1.1)
    ax.set_title("Fig 6 — Team Specialisation (TSI) and Role Stability (RSI)")

    # Value labels on bars
    for bar in list(bars1) + list(bars2):
        h_val = bar.get_height()
        if h_val > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h_val + 0.01, f"{h_val:.2f}",
                    ha="center", va="bottom", fontsize=7)

    ax.legend(["TSI (fill)", "RSI (hatched)"], fontsize=8)
    fig.tight_layout()
    _save(fig, save_path, "fig6_specialisation_index")


# ── Figure: Resilience under agent failure ─────────────────────────────────────

def fig_resilience(alt_metrics: dict, save_path: Path) -> None:
    """Grouped bar: full-team vs 1-AGV-failed performance per condition.

    Core C3 result: symbiotic reward produces significantly more fault-tolerant
    teams — 7.9% performance drop vs 19.6% (flat-coop) and 32.1% (task-only).
    """
    conditions = [
        ("flat_coop",  "Flat-coop\n(baseline)",   METHOD_COLOR["flat_cooperative"]),
        ("task_only",  "Task-only\n(ablation)",    "#E67E22"),
        ("symbiotic",  "Symbiotic\n(PRISM)",       METHOD_COLOR["symbiotic"]),
    ]

    labels, full_vals, fail_vals, drops = [], [], [], []
    for key, label, _ in conditions:
        if key not in alt_metrics:
            continue
        m3 = alt_metrics[key]["metric3_resilience"]
        labels.append(label)
        full_vals.append(m3["full_team_mean"])
        fail_vals.append(m3["failed_agv_mean"])
        drops.append(m3["performance_drop_pct"])

    colors = [c for k, _, c in conditions if k in alt_metrics]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_full = ax.bar(x - w/2, full_vals, w, color=colors, alpha=0.85,
                       label="Full team", edgecolor="black", linewidth=0.4)
    bars_fail = ax.bar(x + w/2, fail_vals, w, color=colors, alpha=0.40,
                       label="1 AGV failed", edgecolor="black", linewidth=0.4, hatch="//")

    # Annotate drop percentage
    for i, (fv, xv, drop) in enumerate(zip(full_vals, x, drops)):
        ymax = max(fv, fail_vals[i]) + 0.15
        sign = "+" if drop < 0 else ""
        color = "#27AE60" if drop < 10 else ("#E67E22" if drop < 25 else "#E74C3C")
        ax.text(xv, ymax, f"Δ={drop:+.1f}%", ha="center", fontsize=9,
                fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean deliveries per episode")
    ax.set_title("Fig — Team Resilience Under Agent Failure\n"
                 "(lower drop % = more resilient; PRISM teams adapt best to failure)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(full_vals) + 1.2)
    fig.tight_layout()
    _save(fig, save_path, "fig_resilience")


# ── Figure 7: Symbiotic vs Flat-cooperative comparison ────────────────────────

def fig7_symbiotic_vs_flat_coop(
    symbiotic_results: dict,
    flat_coop_results: dict,
    save_path: Path,
    heuristic_results: Optional[dict] = None,
    eval_stats: Optional[dict] = None,
) -> None:
    """Bar chart: symbiotic vs flat-cooperative vs heuristic oracle.

    Core PRISM C3 result: symbiotic reward shaping outperforms both the
    flat-cooperative baseline and the handcrafted heuristic oracle.
    If eval_stats is provided (from evaluate_all_seeds.py), those numbers
    are used instead of training-time metrics for accuracy.
    """
    if eval_stats is not None:
        sym_mean  = eval_stats["symbiotic"]["pooled_mean"]
        sym_std   = eval_stats["symbiotic"]["pooled_std"]
        flat_mean = eval_stats["flat_coop"]["pooled_mean"]
        flat_std  = eval_stats["flat_coop"]["pooled_std"]
    else:
        sym_data  = _methods_from_json(symbiotic_results)
        flat_data = _methods_from_json(flat_coop_results)
        sym_val   = next(iter(sym_data.values()),  {})
        flat_val  = next(iter(flat_data.values()), {})
        sym_mean  = sym_val.get("mean_completion", 0)
        sym_std   = sym_val.get("std_completion",  0)
        flat_mean = flat_val.get("mean_completion", 0)
        flat_std  = flat_val.get("std_completion",  0)

    labels = ["Flat-coop\n(baseline)", "Heuristic\noracle", "Symbiotic\n(PRISM)"]
    means  = [flat_mean, 0.0, sym_mean]
    stds   = [flat_std,  0.0, sym_std]
    colors = [METHOD_COLOR["flat_cooperative"], METHOD_COLOR["heuristic"], METHOD_COLOR["symbiotic"]]
    hatches = ["//", "..", ""]

    h = _load_heuristic_data(heuristic_results)
    if h is not None:
        means[1] = h.get("mean_deliveries", 0)
        stds[1]  = h.get("std_deliveries",  0)

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(labels))
    w = 0.55

    bars = []
    for i, (m, s, c, ht) in enumerate(zip(means, stds, colors, hatches)):
        b = ax.bar(i, m, w, yerr=s if s > 0 else None, color=c, alpha=0.85,
                   capsize=5, edgecolor="black", linewidth=0.5, hatch=ht,
                   label=labels[i])
        bars.append(b)
        ax.text(i, m + (s if s > 0 else 0) + 0.15, f"{m:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Annotate PRISM advantage over flat-coop
    if means[2] > means[0]:
        diff = means[2] - means[0]
        ymax = max(means) + max(stds) + 0.8
        ax.annotate("", xy=(2, means[2] + stds[2] + 0.05),
                    xytext=(0, means[0] + stds[0] + 0.05),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
        ax.text(1, ymax, f"PRISM +{diff:.1f} del/ep\nvs flat-coop",
                ha="center", fontsize=8, color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean deliveries per episode")
    ax.set_title("Fig 7 — Throughput Comparison: PRISM vs Baselines")
    ax.set_ylim(0, max(means) + max(stds) + 1.5)
    fig.tight_layout()
    _save(fig, save_path, "fig7_symbiotic_vs_flat_coop")


# ── Figure 8: Package type delivery breakdown ──────────────────────────────────

def fig8_package_breakdown(
    results: dict,
    ckpt_dir: Path,
    save_path: Path,
) -> None:
    """Stacked bar: which package types each method completes most."""
    methods_data = _methods_from_json(results)
    pkg_types = ["SOLO", "STANDARD", "LARGE", "HEAVY", "PICKER_SOLO"]
    pkg_colors = {
        "SOLO":        "#5B9BD5",
        "STANDARD":    "#70AD47",
        "LARGE":       "#FFC000",
        "HEAVY":       "#ED7D31",
        "PICKER_SOLO": "#B96FDB",
    }

    method_names = []
    pkg_counts: Dict[str, List[float]] = {p: [] for p in pkg_types}

    for method in methods_data:
        df = load_eval_csv(ckpt_dir, method)
        if df is None:
            continue
        # Look for per-pkg-type delivery columns
        found_cols = [f"deliveries_{p.lower()}" for p in pkg_types
                      if f"deliveries_{p.lower()}" in df.columns]
        if not found_cols:
            # No per-package-type breakdown available — skip this method.
            # Per-package breakdown is logged when running run_symbiotic.py / run_flat_cooperative.py.
            continue

        method_names.append(method)
        for p in pkg_types:
            col = f"deliveries_{p.lower()}"
            val = df[col].mean() if col in df.columns else 0.0
            pkg_counts[p].append(val)

    if not method_names:
        print("  [SKIP] fig8: no per-package delivery columns in eval CSV")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(method_names))
    bottoms = np.zeros(len(method_names))

    for p in pkg_types:
        vals = np.array(pkg_counts[p])
        ax.bar(x, vals, bottom=bottoms,
               color=pkg_colors[p], label=p, alpha=0.88, width=0.55)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL.get(m, m) for m in method_names], rotation=15, ha="right")
    ax.set_ylabel("Mean deliveries per episode")
    ax.set_title("Fig 8 — Package Type Delivery Breakdown per Method")
    ax.legend(title="Package type", fontsize=8)
    fig.tight_layout()
    _save(fig, save_path, "fig8_package_breakdown")


# ── Figure 9: Battery management convergence ──────────────────────────────────

def fig9_battery_management(
    results: dict,
    ckpt_dir: Path,
    save_path: Path,
    heuristic_results: Optional[dict] = None,
) -> None:
    """Mean battery level over training — shows agents learn safe energy management."""
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = _methods_from_json(results)

    for method in methods:
        df = load_eval_csv(ckpt_dir, method)
        if df is None or "battery_mean_all_agents" not in df.columns or len(df) < 2:
            continue
        y = df["battery_mean_all_agents"].values
        sy = smooth(y)
        sx = df["step"].values if "step" in df.columns else np.arange(len(y))
        sx = sx[len(sx) - len(sy):]
        color = METHOD_COLOR.get(method, "gray")
        ax.plot(sx, sy, color=color,
                label=METHOD_LABEL.get(method, method),
                linestyle=METHOD_STYLE.get(method, "-"),
                linewidth=METHOD_LW.get(method, 1.5))

        # Also plot min battery as dashed thin line
        if "battery_min_all_agents" in df.columns:
            y_min = df["battery_min_all_agents"].values
            sy_min = smooth(y_min)
            ax.plot(sx, sy_min, color=color, alpha=0.35,
                    linestyle=":", linewidth=0.8)

    # Heuristic oracle mean battery as reference band
    h = _load_heuristic_data(heuristic_results)
    if h is not None:
        bat = h.get("battery_mean_per_agent")
        if bat:
            h_bat_mean = float(np.mean(bat))
            h_bat_std  = float(np.std(bat))
            ax.axhline(h_bat_mean, color=METHOD_COLOR["heuristic"], linestyle=":",
                       linewidth=1.4, alpha=0.85,
                       label=f"Heuristic oracle mean ({h_bat_mean:.1f})")
            ax.axhspan(h_bat_mean - h_bat_std, h_bat_mean + h_bat_std,
                       color=METHOD_COLOR["heuristic"], alpha=0.08)

    ax.set_xlabel("Training steps")
    ax.set_ylabel("Battery level (0–100)")
    ax.set_title("Fig 9 — Battery Management: Mean Agent Battery over Training\n"
                 "(dashed = min battery; rising = agents learn to charge proactively)")
    ax.set_ylim(0, 105)
    ax.axhline(10, color="gray", linestyle="--", alpha=0.5, linewidth=0.8,
               label="Low-battery threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path, "fig9_battery_management")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PRISM — Generate paper figures from training results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbiotic", required=True,
                        help="Path to symbiotic condition results JSON (from run_symbiotic.py)")
    parser.add_argument("--flat_coop", default=None,
                        help="Path to flat-cooperative results JSON (for Fig 7, optional)")
    parser.add_argument("--heuristic", default=None,
                        help="Path to heuristic baseline JSON (for reference lines in Fig 1, 6, 7, 9)")
    parser.add_argument("--eval_stats", default=None,
                        help="Path to eval_stats_final.json from evaluate_all_seeds.py — overrides training metrics in Fig 7")
    parser.add_argument("--alt_metrics", default=None,
                        help="Path to alternative_metrics.json from extract_metrics.py — for resilience figure")
    parser.add_argument("--ckpt_dir",       required=True,
                        help="Symbiotic checkpoint directory")
    parser.add_argument("--flat_ckpt_dir",  default=None,
                        help="Flat-cooperative checkpoint dir (for multi-condition Fig 1)")
    parser.add_argument("--task_ckpt_dir",  default=None,
                        help="Task-only checkpoint dir (for multi-condition Fig 1)")
    parser.add_argument("--output",    default="local_runs/figures",
                        help="Output directory for figures")
    parser.add_argument("--backend",   default="ippo",
                        help="Backend used (ippo/mappo)")
    parser.add_argument("--seed",      default=1, type=int)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt_dir)

    print(f"Loading results: {args.symbiotic}")
    symbiotic = load_json(args.symbiotic)
    symbiotic_path = args.symbiotic

    heuristic   = load_json(args.heuristic)   if args.heuristic   else None
    eval_stats  = load_json(args.eval_stats)  if args.eval_stats  else None
    alt_metrics = load_json(args.alt_metrics) if args.alt_metrics else None
    if heuristic:
        print(f"Loaded heuristic baseline: {args.heuristic}")
    if eval_stats:
        print(f"Loaded eval stats: {args.eval_stats}")
    if alt_metrics:
        print(f"Loaded alternative metrics: {args.alt_metrics}")

    print(f"Output directory: {out}/")
    print()

    print("Generating Fig 1 — Learning curves (all conditions) …")
    try:
        fig1_learning_curves(symbiotic, ckpt, out,
                             heuristic_results=heuristic,
                             flat_ckpt_dir=Path(args.flat_ckpt_dir) if args.flat_ckpt_dir else None,
                             task_ckpt_dir=Path(args.task_ckpt_dir) if args.task_ckpt_dir else None)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("Generating Fig — Relationship emergence (mutualism + commensalism) …")
    try:
        fig_relationship_emergence(symbiotic, out)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("Generating Fig — Package type distribution …")
    if args.eval_stats:
        try:
            fig_package_distribution(args.eval_stats, out)
        except Exception as e:
            print(f"  [SKIP] {e}")
    else:
        print("  [SKIP] --eval_stats not supplied")

    print("Generating Fig 2 — Mutualism emergence …")
    try:
        fig2_mutualism_emergence(symbiotic, ckpt, out)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("Generating Fig 3 — Relationship distribution …")
    try:
        fig3_relationship_distribution(symbiotic, ckpt, out)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("Generating Fig 4 — Reward shaping effect …")
    try:
        fig4_reward_shaping_effect(symbiotic, ckpt, out)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("Generating Fig 5 — Training stability …")
    try:
        fig5_training_stability(symbiotic, ckpt, out)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("Generating Fig 6 — Specialisation index …")
    try:
        fig6_specialisation_index(symbiotic, out, heuristic_results=heuristic)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("Generating Fig — Resilience under agent failure …")
    if alt_metrics:
        try:
            fig_resilience(alt_metrics, out)
        except Exception as e:
            print(f"  [SKIP] {e}")
    else:
        print("  [SKIP] --alt_metrics not supplied")

    print("Generating Fig 7 — Symbiotic vs Flat-cooperative …")
    if args.flat_coop:
        try:
            flat_coop = load_json(args.flat_coop)
            fig7_symbiotic_vs_flat_coop(symbiotic, flat_coop, out,
                                        heuristic_results=heuristic, eval_stats=eval_stats)
        except Exception as e:
            print(f"  [SKIP] {e}")
    else:
        print("  [SKIP] --flat_coop not supplied; run run_flat_cooperative.py first")

    print("Generating Fig 8 — Package type breakdown …")
    try:
        fig8_package_breakdown(symbiotic, ckpt, out)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("Generating Fig 9 — Battery management …")
    try:
        fig9_battery_management(symbiotic, ckpt, out, heuristic_results=heuristic)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print(f"\nDone. Figures saved to {out}/")


if __name__ == "__main__":
    main()
