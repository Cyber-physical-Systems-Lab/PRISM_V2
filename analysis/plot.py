"""
Plotting utilities for C3 experimental results.

Generates the four key figures for the IJRR paper:
  Fig 1 — Mutualism fraction curves (heterogeneous vs homogeneous)
  Fig 2 — Task completion comparison (all methods, both env types)
  Fig 3 — Convergence speed comparison (validates C2 claim)
  Fig 4 — Heterogeneity gradient: advantage vs h ∈ [0, 1]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {
    "individual":   "#e74c3c",
    "team":         "#3498db",
    "unclassified": "#f39c12",
    "symbiotic":    "#2ecc71",
    "heuristic":    "#9b59b6",
}

METHOD_LABELS = {
    "individual":   "Individual (baseline)",
    "team":         "Team reward (baseline)",
    "unclassified": "Unclassified coop (baseline)",
    "symbiotic":    "Symbiotic (proposed)",
    "heuristic":    "Heuristic (oracle)",
}


def smooth(arr: np.ndarray, window: int = 20) -> np.ndarray:
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_mutualism_curves(
    results: Dict[str, Dict],
    save_path: Optional[Path] = None,
    title: str = "Mutualism Fraction over Training",
) -> plt.Figure:
    """Fig 1: mutualism fraction per method, mean ± std across seeds."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_methods: List[str] = []

    for method, data in results.items():
        curves = np.array(data.get("mutualism_curves", []))
        if curves.ndim != 2 or curves.shape[0] == 0:
            continue
        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)
        x    = np.arange(len(mean))
        color = METHOD_COLORS.get(method, "gray")
        ax.plot(x, smooth(mean), label=METHOD_LABELS.get(method, method), color=color)
        ax.fill_between(x,
                        smooth(np.maximum(0, mean - std)),
                        smooth(np.minimum(1, mean + std)),
                        alpha=0.15, color=color)
        plotted_methods.append(method)

    if not plotted_methods:
        plt.close(fig)
        raise ValueError(
            "No mutualism data found: expected non-empty 'mutualism_curves' per method "
            "in hetero_results.json (e.g., from experiments/run_heterogeneous.py)."
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mutualism fraction")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_task_completion(
    hetero_results: Dict[str, Dict],
    homo_results: Dict[str, Dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Fig 2: bar chart of final mean task completion (both env types)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

    for ax, results, title in [
        (axes[0], hetero_results, "Heterogeneous (ENV A)"),
        (axes[1], homo_results,   "Homogeneous (ENV B)"),
    ]:
        methods = list(results.keys())
        means   = [results[m].get("mean_completion", 0) for m in methods]
        stds    = [results[m].get("std_completion",  0) for m in methods]
        colors  = [METHOD_COLORS.get(m, "gray") for m in methods]
        labels  = [METHOD_LABELS.get(m, m) for m in methods]

        bars = ax.bar(labels, means, color=colors, alpha=0.85, yerr=stds, capsize=4)
        ax.set_title(title)
        ax.set_ylabel("Mean deliveries / episode")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_heterogeneity_gradient(
    gradient_results: Dict[float, Dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Fig 4: symbiotic advantage vs heterogeneity parameter h."""
    fig, ax = plt.subplots(figsize=(7, 5))

    h_vals = sorted(gradient_results.keys())
    sym_means = []
    unc_means = []

    for h in h_vals:
        r = gradient_results[h]
        sym_means.append(r.get("symbiotic", {}).get("mean_completion", 0))
        unc_means.append(r.get("unclassified", {}).get("mean_completion", 0))

    sym_means = np.array(sym_means)
    unc_means = np.array(unc_means)
    advantage = sym_means - unc_means

    ax.plot(h_vals, advantage, "o-", color=METHOD_COLORS["symbiotic"], linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Heterogeneity parameter h")
    ax.set_ylabel("Symbiotic − Unclassified  (deliveries/ep)")
    ax.set_title("Advantage of Symbiotic Reward vs Heterogeneity Degree")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    hetero = load_results(results_dir / "hetero_results.json")
    homo   = load_results(results_dir / "homo_results.json")

    try:
        plot_mutualism_curves(
            hetero["methods"],
            save_path=figures_dir / "fig1_mutualism_hetero.pdf",
            title="Mutualism Fraction — Heterogeneous Env (ENV A)",
        )
    except ValueError as exc:
        print(f"[WARN] Skipping fig1_mutualism_hetero.pdf: {exc}")
    plot_task_completion(
        {m: hetero["methods"][m] for m in hetero["methods"]},
        {m: homo["methods"][m]   for m in homo["methods"]},
        save_path=figures_dir / "fig2_task_completion.pdf")
    print(f"Figures saved to {figures_dir}/")
