"""
Plotting utilities for PRISM C3 experimental results.

Generates the key figures comparing the two reward conditions:
  Fig 1 — Mutualism fraction emergence (symbiotic condition)
  Fig 2 — Task completion comparison (symbiotic vs flat-cooperative)
  Fig 3 — Convergence speed comparison (validates C2 claim)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {
    "symbiotic":       "#2ecc71",
    "flat_cooperative":"#3498db",
    "heuristic":       "#9b59b6",
}

METHOD_LABELS = {
    "symbiotic":        "Symbiotic (PRISM)",
    "flat_cooperative": "Flat-cooperative (baseline)",
    "heuristic":        "Heuristic (oracle)",
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
    """Fig 1: mutualism fraction per condition, mean ± std across seeds."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted: List[str] = []

    for method, data in results.items():
        curves = np.array(data.get("mutualism_curves", []))
        if curves.ndim != 2 or curves.shape[0] == 0:
            continue
        mean  = curves.mean(axis=0)
        std   = curves.std(axis=0)
        x     = np.arange(len(mean))
        color = METHOD_COLORS.get(method, "gray")
        ax.plot(x, smooth(mean), label=METHOD_LABELS.get(method, method), color=color)
        ax.fill_between(x,
                        smooth(np.maximum(0, mean - std)),
                        smooth(np.minimum(1, mean + std)),
                        alpha=0.15, color=color)
        plotted.append(method)

    if not plotted:
        plt.close(fig)
        raise ValueError(
            "No mutualism data found: expected non-empty 'mutualism_curves' per condition "
            "in the results JSON (from run_symbiotic.py / run_flat_cooperative.py)."
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
    symbiotic_results: Dict[str, Dict],
    flat_coop_results: Dict[str, Dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Fig 2: bar chart of final mean task completion — symbiotic vs flat-cooperative."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

    for ax, results, title in [
        (axes[0], symbiotic_results,  "Symbiotic (PRISM)"),
        (axes[1], flat_coop_results,  "Flat-cooperative (baseline)"),
    ]:
        methods = list(results.keys())
        means   = [results[m].get("mean_completion", 0) for m in methods]
        stds    = [results[m].get("std_completion",  0) for m in methods]
        colors  = [METHOD_COLORS.get(m, "gray") for m in methods]
        labels  = [METHOD_LABELS.get(m, m) for m in methods]

        ax.bar(labels, means, color=colors, alpha=0.85, yerr=stds, capsize=4)
        ax.set_title(title)
        ax.set_ylabel("Mean deliveries / episode")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    symbiotic = load_results(results_dir / "prism_symbiotic.json")
    flat_coop  = load_results(results_dir / "prism_flat_cooperative.json")

    sym_methods  = symbiotic.get("results",  symbiotic)
    flat_methods = flat_coop.get("results",  flat_coop)

    try:
        plot_mutualism_curves(
            sym_methods,
            save_path=figures_dir / "fig1_mutualism_symbiotic.pdf",
            title="Mutualism Fraction — Symbiotic Condition",
        )
    except ValueError as exc:
        print(f"[WARN] Skipping fig1: {exc}")

    plot_task_completion(
        sym_methods,
        flat_methods,
        save_path=figures_dir / "fig2_task_completion.pdf",
    )
    print(f"Figures saved to {figures_dir}/")
