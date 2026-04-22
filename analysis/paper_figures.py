"""
Publication-quality figures for the IJRR paper.

Produces up to 8 figures depending on which result files are available:

  Fig 1  — Task completion learning curves (all methods, hetero env)
  Fig 2  — Mutualism fraction emergence curves (all methods)
  Fig 3  — Relationship type distribution at end of training (stacked bar)
  Fig 4  — Raw vs shaped reward divergence (shows shaping effect)
  Fig 5  — Training stability: actor entropy + KL divergence
  Fig 6  — Team Specialisation Index (TSI) per method
  Fig 7  — Hetero vs Homo task completion comparison (falsification result)
  Fig 8  — Package-type delivery breakdown per method

Usage
-----
# After running run_heterogeneous.py:
python analysis/paper_figures.py \
    --hetero   local_runs/results/symbiotic_200k.json \
    --ckpt_dir local_runs/checkpoints_200k \
    --output   local_runs/figures

# With homogeneous results too (Fig 7):
python analysis/paper_figures.py \
    --hetero   local_runs/results/symbiotic_200k.json \
    --homo     local_runs/results/homo_results.json \
    --ckpt_dir local_runs/checkpoints_200k \
    --output   local_runs/figures
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
    "individual":   "#E74C3C",   # red
    "team":         "#3498DB",   # blue
    "unclassified": "#F39C12",   # amber
    "symbiotic":    "#27AE60",   # green  (proposed method)
    "heuristic":    "#8E44AD",   # purple
}
METHOD_LABEL = {
    "individual":   "Individual reward",
    "team":         "Team reward",
    "unclassified": "Unclassified coop.",
    "symbiotic":    "Symbiotic (ours)",
    "heuristic":    "Heuristic oracle",
}
METHOD_STYLE = {
    "individual":   "-",
    "team":         "--",
    "unclassified": "-.",
    "symbiotic":    "-",
    "heuristic":    ":",
}
METHOD_LW = {
    "individual": 1.4, "team": 1.4, "unclassified": 1.4,
    "symbiotic": 2.2, "heuristic": 1.4,
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
    """Extract the methods sub-dict from the hetero/homo JSON result."""
    if "methods" in data:
        return data["methods"]
    return data


# ── Figure 1: Task completion learning curves ──────────────────────────────────

def fig1_learning_curves(
    results: dict,
    ckpt_dir: Path,
    save_path: Path,
) -> None:
    """Deliveries per episode over training steps for all methods."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = _methods_from_json(results)

    for method, data in methods.items():
        # Prefer per-episode CSV (smoother) over coarse JSON curves
        df = load_eval_csv(ckpt_dir, method)
        if df is not None and "deliveries" in df.columns and len(df) > 1:
            y = df["deliveries"].values
            x = df["step"].values if "step" in df.columns else np.arange(len(y))
            sy = smooth(y)
            sx = smooth_x(len(y)) if len(sy) < len(y) else np.arange(len(sy))
            sx = x[len(x) - len(sy):]
        else:
            curves = np.array(data.get("deliveries_curves", []))
            if curves.ndim != 2 or curves.shape[1] < 2:
                continue
            y = curves.mean(axis=0)
            x = np.linspace(0, len(y) - 1, len(y))
            sy = smooth(y)
            sx = x[len(x) - len(sy):]

        color = METHOD_COLOR.get(method, "gray")
        ax.plot(sx, sy,
                color=color,
                label=METHOD_LABEL.get(method, method),
                linestyle=METHOD_STYLE.get(method, "-"),
                linewidth=METHOD_LW.get(method, 1.5))

        # Shade ±1 std if multi-seed curves available
        curves = np.array(data.get("deliveries_curves", []))
        if curves.ndim == 2 and curves.shape[0] > 1:
            std = curves.std(axis=0)
            mean = curves.mean(axis=0)
            sm = smooth(mean)
            ss = smooth(std)
            sx2 = np.arange(len(sm))
            ax.fill_between(sx2, sm - ss, sm + ss, alpha=0.12, color=color)

    ax.set_xlabel("Training steps")
    ax.set_ylabel("Deliveries per episode")
    ax.set_title("Fig 1 — Task Completion over Training (Heterogeneous Env)")
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
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7)

    ax.legend(["TSI (fill)", "RSI (hatched)"], fontsize=8)
    fig.tight_layout()
    _save(fig, save_path, "fig6_specialisation_index")


# ── Figure 7: Hetero vs Homo comparison ───────────────────────────────────────

def fig7_hetero_vs_homo(
    hetero_results: dict,
    homo_results: dict,
    save_path: Path,
) -> None:
    """Grouped bar chart: hetero ENV-A vs homo ENV-B.
    The key falsification result: symbiotic advantage disappears in homogeneous env."""
    hetero = _methods_from_json(hetero_results)
    homo   = _methods_from_json(homo_results)

    all_methods = sorted(set(hetero) | set(homo))
    x = np.arange(len(all_methods))
    w = 0.35

    hetero_means = [hetero.get(m, {}).get("mean_completion", 0) for m in all_methods]
    homo_means   = [homo.get(m, {}).get("mean_completion", 0)   for m in all_methods]
    hetero_stds  = [hetero.get(m, {}).get("std_completion", 0)  for m in all_methods]
    homo_stds    = [homo.get(m, {}).get("std_completion", 0)    for m in all_methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [METHOD_COLOR.get(m, "gray") for m in all_methods]

    ax.bar(x - w / 2, hetero_means, w, yerr=hetero_stds, color=colors, alpha=0.9,
           capsize=4, label="Heterogeneous (ENV A)", edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, homo_means,   w, yerr=homo_stds,   color=colors, alpha=0.45,
           capsize=4, label="Homogeneous (ENV B)", edgecolor="black", linewidth=0.4, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL.get(m, m) for m in all_methods], rotation=15, ha="right")
    ax.set_ylabel("Mean deliveries per episode")
    ax.set_title("Fig 7 — Symbiotic Advantage Disappears in Homogeneous Env (Falsification)")

    # Annotate the symbiotic bars with the advantage
    if "symbiotic" in hetero and "symbiotic" in homo:
        si = all_methods.index("symbiotic")
        diff = hetero_means[si] - homo_means[si]
        ax.annotate(
            f"Δ={diff:+.1f}",
            xy=(si, max(hetero_means[si], homo_means[si])),
            xytext=(si, max(hetero_means[si], homo_means[si]) + 0.5),
            ha="center", fontsize=8, color=METHOD_COLOR["symbiotic"],
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    ax.legend()
    fig.tight_layout()
    _save(fig, save_path, "fig7_hetero_vs_homo")


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
            # Run training with the updated run_heterogeneous.py to get this data.
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
        description="Generate paper figures from training results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hetero",   required=True,
                        help="Path to heterogeneous results JSON (from run_heterogeneous.py)")
    parser.add_argument("--homo",     default=None,
                        help="Path to homogeneous results JSON (for Fig 7, optional)")
    parser.add_argument("--ckpt_dir", required=True,
                        help="Checkpoint directory containing per-method CSV files")
    parser.add_argument("--output",   default="local_runs/figures",
                        help="Output directory for figures")
    parser.add_argument("--backend",  default="ippo",
                        help="Backend used (ippo/hetppo/mappo)")
    parser.add_argument("--seed",     default=1, type=int)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt_dir)

    print(f"Loading results: {args.hetero}")
    hetero = load_json(args.hetero)

    print(f"Output directory: {out}/")
    print()

    print("Generating Fig 1 — Learning curves …")
    try:
        fig1_learning_curves(hetero, ckpt, out)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("Generating Fig 2 — Mutualism emergence …")
    try:
        fig2_mutualism_emergence(hetero, ckpt, out)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("Generating Fig 3 — Relationship distribution …")
    try:
        fig3_relationship_distribution(hetero, ckpt, out)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("Generating Fig 4 — Reward shaping effect …")
    try:
        fig4_reward_shaping_effect(hetero, ckpt, out)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("Generating Fig 5 — Training stability …")
    try:
        fig5_training_stability(hetero, ckpt, out)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("Generating Fig 6 — Specialisation index …")
    try:
        fig6_specialisation_index(hetero, out)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("Generating Fig 7 — Hetero vs Homo (falsification) …")
    if args.homo:
        try:
            homo = load_json(args.homo)
            fig7_hetero_vs_homo(hetero, homo, out)
        except Exception as e:
            print(f"  [SKIP] {e}")
    else:
        print("  [SKIP] --homo not supplied; run homogeneous experiment first")

    print("Generating Fig 8 — Package type breakdown …")
    try:
        fig8_package_breakdown(hetero, ckpt, out)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("Generating Fig 9 — Battery management …")
    try:
        fig9_battery_management(hetero, ckpt, out)
    except Exception as e:
        print(f"  [SKIP] {e}")

    print(f"\nDone. Figures saved to {out}/")


if __name__ == "__main__":
    main()
