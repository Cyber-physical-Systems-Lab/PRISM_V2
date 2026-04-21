"""
analyze_results.py — Plot all findings from an experiment_results_*.json file.

Usage:
    python experiments/analyze_results.py                           # auto-pick latest
    python experiments/analyze_results.py runs/results/experiment_results_4866070.json
    python experiments/analyze_results.py --out runs/results/figs
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linewidth":    0.6,
    "figure.dpi":        150,
})

METHOD_COLORS = {
    "individual":  "#4878CF",
    "team":        "#6ACC65",
    "unclassified":"#D65F5F",
    "symbiotic":   "#B47CC7",
}
METHOD_LABELS = {
    "individual":  "Individual",
    "team":        "Team reward",
    "unclassified":"Unclassified (no shaping)",
    "symbiotic":   "Symbiotic (ours)",
}
METHODS = ["individual", "team", "unclassified", "symbiotic"]

BACKEND_STYLES = {
    "ippo":   {"ls": ":",  "marker": "o", "label": "IPPO"},
    "hetppo": {"ls": "--", "marker": "s", "label": "HetPPO"},
    "mappo":  {"ls": "-",  "marker": "^", "label": "MAPPO"},
}

# ── Ablation-specific style ────────────────────────────────────────────────────
ABLATION_COLORS = {
    **METHOD_COLORS,
    "random_typed":   "#E8874A",
    "positive_only":  "#56B4E9",
    "negative_only":  "#CC79A7",
    "obs_only":       "#009E73",
    "delta_weighted": "#F0E442",
}
ABLATION_LABELS = {
    **METHOD_LABELS,
    "random_typed":   "Random typed (ctrl)",
    "positive_only":  "Positive φ only",
    "negative_only":  "Negative φ only",
    "obs_only":       "Obs aug only",
    "delta_weighted": "Delta-weighted φ",
}
ABLATION_ORDER = [
    "individual", "symbiotic", "random_typed",
    "positive_only", "negative_only", "obs_only", "delta_weighted",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def smooth(x, w=5):
    """Uniform moving average."""
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def mean_std(curves):
    """curves: list of lists (one per seed). Returns mean, std arrays aligned to shortest."""
    min_len = min(len(c) for c in curves)
    arr = np.array([c[:min_len] for c in curves], dtype=float)
    return arr.mean(0), arr.std(0)


def plot_learning_curves(ax, methods_data, smooth_w=5, ylabel="Deliveries / episode",
                         title=None, heuristic_line=None, show_legend=True):
    """Plot mean ± std learning curves for each method."""
    for m in METHODS:
        if m not in methods_data:
            continue
        curves = methods_data[m].get("deliveries_curves", [])
        if not curves:
            continue
        mu, sd = mean_std(curves)
        if smooth_w > 1:
            mu = smooth(mu, smooth_w)
            sd = smooth(sd, smooth_w)
        x = np.arange(len(mu))
        c = METHOD_COLORS[m]
        ax.plot(x, mu, color=c, lw=1.8, label=METHOD_LABELS[m])
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.18)

    if heuristic_line is not None:
        ax.axhline(heuristic_line, color="gray", ls="--", lw=1.4,
                   label=f"Heuristic ({heuristic_line:.1f})")

    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left", framealpha=0.7)


def _battery_seed_curves(method_entry):
    """
    Return list of per-seed battery curves (one scalar per episode).

    Expected preferred format:
      method_entry["battery_mean_curves_per_seed"] =
        [seed][episode][agent_battery]

    Fallback format:
      method_entry["battery_mean_curve_per_episode"] =
        [episode][agent_battery]
    """
    curves = method_entry.get("battery_mean_curves_per_seed", [])
    out = []
    for seed_curves in curves:
        arr = np.asarray(seed_curves, dtype=float)
        if arr.ndim == 2 and arr.shape[0] > 0:
            out.append(arr.mean(axis=1).tolist())

    if out:
        return out

    single = method_entry.get("battery_mean_curve_per_episode", [])
    arr = np.asarray(single, dtype=float)
    if arr.ndim == 2 and arr.shape[0] > 0:
        return [arr.mean(axis=1).tolist()]
    return []


def plot_battery_curves(ax, methods_data, smooth_w=5, title=None, show_legend=True):
    """Plot mean episode battery (averaged over agents) with mean ± std across seeds."""
    plotted = False
    for m in METHODS:
        if m not in methods_data:
            continue
        curves = _battery_seed_curves(methods_data[m])
        if not curves:
            continue
        mu, sd = mean_std(curves)
        if smooth_w > 1:
            mu = smooth(mu, smooth_w)
            sd = smooth(sd, smooth_w)
        x = np.arange(len(mu))
        c = METHOD_COLORS[m]
        ax.plot(x, mu, color=c, lw=1.8, label=METHOD_LABELS[m])
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.18)
        plotted = True

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Battery level (mean over agents)")
    if title:
        ax.set_title(title)
    if show_legend and plotted:
        ax.legend(loc="upper left", framealpha=0.7)
    return plotted


# ── Figure 1: Heterogeneous training curves ────────────────────────────────────

def fig_hetero_curves(data, out_dir, heuristic_mean=None):
    h = data["results"]["heterogeneous"]
    env = h["env"]
    methods = h["methods"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    fig.suptitle(f"Heterogeneous env — {env}", fontsize=12, y=1.01)

    # Left: deliveries
    plot_learning_curves(
        axes[0], methods, smooth_w=5,
        ylabel="Deliveries / episode",
        title="Delivery rate over training",
        heuristic_line=heuristic_mean,
    )

    # Right: mutualism fraction
    ax = axes[1]
    for m in METHODS:
        if m not in methods:
            continue
        mc = methods[m].get("mutualism_curves")
        if not mc:
            continue
        mu, sd = mean_std(mc)
        mu = smooth(mu * 100, 5)   # → percent
        sd = smooth(sd * 100, 5)
        x = np.arange(len(mu))
        c = METHOD_COLORS[m]
        ax.plot(x, mu, color=c, lw=1.8, label=METHOD_LABELS[m])
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.18)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mutualism fraction (%)")
    ax.set_title("Mutualism emergence over training")
    ax.legend(loc="upper left", framealpha=0.7)

    fig.tight_layout()
    path = out_dir / "fig1_hetero_curves.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 2: Heuristic baseline ───────────────────────────────────────────────

def fig_heuristic(data, out_dir):
    hdata = data["results"]["heuristic"]
    envs  = list(hdata.keys())
    n     = len(envs)
    env_short = [e.replace("tarware-", "").replace("-partialobs-chg-v1", "").replace("-partialobs-v1","") for e in envs]

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.8), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("Heuristic baseline — delivery curve per environment", fontsize=12)

    for ax, env, label in zip(axes, envs, env_short):
        curve = hdata[env]["deliveries_curve"]
        mean  = hdata[env]["mean_deliveries"]
        std   = hdata[env]["std_deliveries"]
        ax.plot(curve, color="#2B7BBA", lw=1.5, alpha=0.8)
        ax.axhline(mean, color="#E05C00", ls="--", lw=1.4,
                   label=f"Mean={mean:.1f} ± {std:.1f}")
        ax.set_title(label)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Deliveries")
        ax.legend(fontsize=8, framealpha=0.7)

    fig.tight_layout()
    path = out_dir / "fig2_heuristic_baseline.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 3: Gradient sensitivity (h sweep) ───────────────────────────────────

def fig_gradient(data, out_dir):
    """
    Works with both the old flat format and the new backend-aware format.

    Old: data["checks"]["C3"]["gradient_advantage_by_h"] = {h_str: float}
    New: data["advantage"] = {h_str: {backend: float}}
    """
    # New standalone format (gradient_results_*.json)
    if "advantage" in data and "h_values" in data:
        _fig_gradient_standalone(data, out_dir)
        return

    # Old combined format
    checks = data.get("checks", {}).get("C3", {})
    adv_by_h = checks.get("gradient_advantage_by_h", {})
    if not adv_by_h:
        gdata = data["results"].get("gradient", {})
        if not gdata:
            print("  [skip] No gradient data found.")
            return
        # New backend-nested gradient section inside combined JSON
        first_val = next(iter(gdata.values()), {})
        if isinstance(first_val, dict) and "advantage" in gdata:
            _fig_gradient_standalone(gdata, out_dir)
            return
        for h_str, methods in gdata.items():
            sym = methods.get("symbiotic", {}).get("mean_completion", 0)
            unc = methods.get("unclassified", {}).get("mean_completion", 0)
            adv_by_h[h_str] = sym - unc

    hs  = sorted(float(k) for k in adv_by_h)
    adv = [adv_by_h[str(h)] for h in hs]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    colors = ["#B47CC7" if a >= 0 else "#D65F5F" for a in adv]
    ax.bar(hs, adv, width=0.08, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Heterogeneity parameter h")
    ax.set_ylabel("Advantage: symbiotic − baseline (deliveries/ep)")
    ax.set_title("Symbiotic advantage vs. heterogeneity (gradient sweep)")
    ax.set_xticks(hs)
    fig.tight_layout()
    path = out_dir / "fig3_gradient_sensitivity.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _fig_gradient_standalone(gdata, out_dir):
    """
    gradient_results_*.json: advantage keyed by h then backend.
    Plots one line per backend showing symbiotic−unclassified advantage vs h.
    """
    advantage = gdata.get("advantage", {})
    h_values = sorted(float(k) for k in advantage)
    backends = gdata.get("backends", list(BACKEND_STYLES.keys()))

    fig, ax = plt.subplots(figsize=(7, 4))
    for backend in backends:
        sty = BACKEND_STYLES.get(backend, {"ls": "-", "marker": "o", "label": backend})
        adv = [advantage.get(str(h), {}).get(backend, float("nan")) for h in h_values]
        ax.plot(h_values, adv, ls=sty["ls"], marker=sty["marker"], lw=1.8,
                color="#B47CC7", alpha=0.5 + 0.5 * (backends.index(backend) / max(1, len(backends) - 1)),
                label=sty["label"])

    ax.axhline(0, color="black", lw=0.8, ls="-")
    ax.set_xlabel("Heterogeneity level h")
    ax.set_ylabel("Symbiotic − Unclassified (deliveries/ep)")
    ax.set_title("Symbiotic advantage grows with heterogeneity")
    ax.set_xticks(h_values)
    ax.set_xticklabels([f"h={h:.1f}" for h in h_values])
    ax.legend(framealpha=0.7)
    fig.tight_layout()
    path = out_dir / "fig3_gradient_sensitivity.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 4: Convergence — learning & gradient norms ─────────────────────────

def fig_convergence(data, out_dir):
    """
    Works with the new backend-aware convergence JSON and the old flat format.

    New: data["results"] = {backend: {method: {deliveries_curves, grad_norm_history, ...}}}
    Old: data["results"]["convergence"]["methods"] = {method: {deliveries_curves, ...}}
    """
    # New standalone convergence JSON (convergence_results_*.json)
    if "backends" in data and "results" in data and isinstance(
        next(iter(data["results"].values()), None), dict
    ):
        first = next(iter(data["results"].values()))
        if isinstance(first, dict) and isinstance(next(iter(first.values()), None), dict):
            _fig_convergence_standalone(data, out_dir)
            return

    # Old combined JSON path
    conv = data["results"].get("convergence", {})
    if not conv:
        print("  [skip] No convergence data found.")
        return

    # New backend-nested inside combined JSON
    if "backends" in conv:
        _fig_convergence_standalone(conv, out_dir)
        return

    if "methods" not in conv:
        print("  [skip] No convergence methods found.")
        return
    methods = conv["methods"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(f"Convergence analysis — {conv.get('env', '')}", fontsize=12)

    ax = axes[0]
    for m in METHODS:
        if m not in methods:
            continue
        curves = methods[m].get("deliveries_curves", [])
        if not curves:
            continue
        mu, sd = mean_std(curves)
        x = np.arange(len(mu))
        c = METHOD_COLORS[m]
        ep = methods[m].get("mean_convergence_episode")
        label = METHOD_LABELS[m]
        if ep:
            label += f" (conv@{ep:.0f})"
        ax.plot(x, mu, color=c, lw=1.8, label=label)
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.18)
        if ep and ep <= len(mu):
            ax.axvline(ep, color=c, ls=":", lw=1.2, alpha=0.7)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Deliveries / episode")
    ax.set_title("Learning curves")
    ax.legend(loc="upper left", framealpha=0.7, fontsize=8)

    ax = axes[1]
    for m in METHODS:
        if m not in methods:
            continue
        gnc = methods[m].get("gradient_norm_curves", [])
        if not gnc:
            continue
        mu, _ = mean_std(gnc)
        mu = smooth(mu, 3)
        ax.plot(np.arange(len(mu)), mu, color=METHOD_COLORS[m], lw=1.8, label=METHOD_LABELS[m])
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Gradient norm")
    ax.set_title("Gradient norm (stability)")
    ax.legend(loc="upper right", framealpha=0.7, fontsize=8)

    fig.tight_layout()
    path = out_dir / "fig4_convergence.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _fig_convergence_standalone(conv, out_dir):
    """
    convergence_results_*.json: one row per backend, two columns (curves + grad norms).
    """
    backends = conv.get("backends", list(conv["results"].keys()))
    n_backends = len(backends)
    env = conv.get("env", "")

    fig, axes = plt.subplots(n_backends, 2, figsize=(12, 3.5 * n_backends), squeeze=False)
    fig.suptitle(f"Convergence analysis — {env}\n(all backends × methods)", fontsize=12)

    for row, backend in enumerate(backends):
        b_data = conv["results"].get(backend, {})
        sty = BACKEND_STYLES.get(backend, {"label": backend})

        # Left: delivery curves
        ax = axes[row, 0]
        ax.set_title(f"{sty['label']} — delivery curves", fontsize=10)
        for m in METHODS:
            if m not in b_data:
                continue
            curves = b_data[m].get("deliveries_curves", [])
            if not curves:
                continue
            mu, sd = mean_std(curves)
            x = np.arange(len(mu))
            c = METHOD_COLORS[m]
            conv_ep = b_data[m].get("convergence_episodes")
            label = METHOD_LABELS[m]
            if conv_ep:
                label += f" (conv@{conv_ep})"
            ax.plot(x, smooth(mu, 5), color=c, lw=1.8, label=label)
            ax.fill_between(x, smooth(mu - sd, 5), smooth(mu + sd, 5), color=c, alpha=0.18)
            if conv_ep and conv_ep < len(mu):
                ax.axvline(conv_ep, color=c, ls=":", lw=1.0, alpha=0.6)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Deliveries / ep")
        ax.legend(loc="upper left", framealpha=0.7, fontsize=7)

        # Right: gradient norm history (flat per-minibatch list → smooth & plot)
        ax = axes[row, 1]
        ax.set_title(f"{sty['label']} — gradient norms", fontsize=10)
        for m in METHODS:
            if m not in b_data:
                continue
            gn = b_data[m].get("grad_norm_history", [])
            if not gn:
                continue
            gn_arr = smooth(np.array(gn, dtype=float), w=max(5, len(gn) // 50))
            x = np.linspace(0, 1, len(gn_arr))
            ax.plot(x, gn_arr, color=METHOD_COLORS[m], lw=1.4, label=METHOD_LABELS[m])
        ax.set_xlabel("Training progress (normalised)")
        ax.set_ylabel("Gradient norm")
        ax.legend(loc="upper right", framealpha=0.7, fontsize=7)

    fig.tight_layout()
    path = out_dir / "fig4_convergence.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 5: TSI / RSI bar chart ─────────────────────────────────────────────

def fig_tsi_rsi(data, out_dir):
    """Grouped bar: TSI and RSI per method for hetero & homo."""
    sections = {
        "Heterogeneous": data["results"].get("heterogeneous", {}).get("methods", {}),
        "Homogeneous":   data["results"].get("homogeneous",   {}).get("methods", {}),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    metrics = [("tsi", "TSI (task specialisation)"), ("rsi", "RSI (role stability)")]

    for ax, (key, ylabel) in zip(axes, metrics):
        x      = np.arange(2)       # two environments
        width  = 0.18
        offsets= np.linspace(-0.28, 0.28, len(METHODS))
        for i, m in enumerate(METHODS):
            vals = []
            for sec_name, sec_data in sections.items():
                v = sec_data.get(m, {}).get(key)
                vals.append(v if v is not None else 0.0)
            ax.bar(x + offsets[i], vals, width=width,
                   color=METHOD_COLORS[m], label=METHOD_LABELS[m],
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(list(sections.keys()))
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_ylim(0, 1.08)
        ax.axhline(1.0, color="gray", ls="--", lw=0.7, alpha=0.5)
        if key == "tsi":
            ax.legend(fontsize=8, framealpha=0.7)

    fig.suptitle("Specialisation & role-stability metrics", fontsize=12)
    fig.tight_layout()
    path = out_dir / "fig5_tsi_rsi.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 6: Hetero vs Homo final performance ────────────────────────────────

def fig_hetero_vs_homo(data, out_dir):
    """Bar chart: final mean_completion per method × environment."""
    sections = {
        "Heterogeneous": data["results"].get("heterogeneous", {}).get("methods", {}),
        "Homogeneous":   data["results"].get("homogeneous",   {}).get("methods", {}),
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    x       = np.arange(2)
    width   = 0.18
    offsets = np.linspace(-0.28, 0.28, len(METHODS))

    for i, m in enumerate(METHODS):
        means = []
        errs  = []
        for sec_data in sections.values():
            md = sec_data.get(m, {})
            means.append(md.get("mean_completion", 0.0))
            errs.append(md.get("std_completion",   0.0))
        ax.bar(x + offsets[i], means, width=width, yerr=errs,
               color=METHOD_COLORS[m], label=METHOD_LABELS[m],
               capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(list(sections.keys()))
    ax.set_ylabel("Mean deliveries / episode (final)")
    ax.set_title("Final delivery performance: heterogeneous vs. homogeneous")
    ax.legend(fontsize=9, framealpha=0.7)
    fig.tight_layout()
    path = out_dir / "fig6_hetero_vs_homo.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 7: Claim-check summary table ───────────────────────────────────────

def fig_claims_summary(data, out_dir):
    checks = data["checks"]

    rows = []
    # C1
    c1 = checks["C1"]
    rows += [
        ("C1", "Mutualism data present",
         "Yes" if c1.get("heterogeneous_mutualism_data_present") else "No",
         c1.get("heterogeneous_mutualism_data_present", False)),
        ("C1", "TSI (symbiotic, hetero)",
         f"{c1.get('heterogeneous_tsi_symbiotic',0):.3f}", True),
        ("C1", "RSI (symbiotic, hetero)",
         f"{c1.get('heterogeneous_rsi_symbiotic',0):.3f}", True),
    ]
    # C2
    c2 = checks["C2"]
    rows += [
        ("C2", "Sym. rewards bounded",
         "Yes" if c2.get("symbiotic_all_bounded") else "No",
         c2.get("symbiotic_all_bounded", False)),
        ("C2", "Max |r_sym| / r_task",
         f"{c2.get('symbiotic_max_ratio',0):.3f}", True),
        ("C2", "Conv. episode (symbiotic)",
         f"{c2.get('symbiotic_mean_convergence_episode',0):.1f}", True),
        ("C2", "Conv. episode (individual)",
         f"{c2.get('individual_mean_convergence_episode',0):.1f}", True),
    ]
    # C3
    c3 = checks["C3"]
    rows += [
        ("C3", "Hetero symbiotic > unclassified",
         f"{c3.get('heterogeneous_symbiotic_minus_unclassified',0):+.3f}",
         c3.get("heterogeneous_symbiotic_minus_unclassified", -1) >= 0),
        ("C3", "Homo symbiotic > unclassified",
         f"{c3.get('homogeneous_symbiotic_minus_unclassified',0):+.3f}",
         c3.get("homogeneous_symbiotic_minus_unclassified", -1) >= 0),
        ("C3", "Hetero advantage > homo",
         "Yes" if c3.get("hetero_advantage_exceeds_homo") else "No",
         c3.get("hetero_advantage_exceeds_homo", False)),
    ]

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.5))
    ax.axis("off")
    col_labels = ["Claim", "Check", "Value", "Pass"]
    table_data = [[r[0], r[1], r[2], "✓" if r[3] else "✗"] for r in rows]
    colors_col = [["white"] * 4 for _ in rows]
    for i, r in enumerate(rows):
        colors_col[i][3] = "#c8f7c5" if r[3] else "#f7c5c5"

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellColours=colors_col,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width([0, 1, 2, 3])
    ax.set_title("Claim verification summary", fontsize=12, pad=14)

    fig.tight_layout()
    path = out_dir / "fig7_claims_summary.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 8: Homogeneous training curves ─────────────────────────────────────

def fig_homo_curves(data, out_dir):
    """
    Works with new backend-nested format and old flat format.
    Old: ho["methods"] = {method: {deliveries_curves}}
    New: ho["results"] = {backend: {method: {deliveries_curves}}}
    """
    ho = data.get("results", {})
    # If we received the top-level homo JSON directly
    if "methods" in data and "env" in data:
        ho = data
    elif "falsification" in data:
        ho = data
    elif "homogeneous" in ho:
        ho = ho["homogeneous"]

    if not ho:
        print("  [skip] No homogeneous data found.")
        return

    env = ho.get("env", "")

    # Detect format
    results_block = ho.get("results", ho.get("methods", {}))
    first_val = next(iter(results_block.values()), {})

    if isinstance(first_val, dict) and "deliveries_curves" not in first_val:
        # New format: results_block = {backend: {method: {...}}}
        backends = ho.get("backends", list(results_block.keys()))
        n = len(backends)
        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4), sharey=True, squeeze=False)
        fig.suptitle(f"Homogeneous env (falsification) — {env}", fontsize=12)
        for col, backend in enumerate(backends):
            ax = axes[0, col]
            sty = BACKEND_STYLES.get(backend, {"label": backend})
            ax.set_title(sty["label"], fontsize=10)
            b_methods = results_block.get(backend, {})
            plot_learning_curves(ax, b_methods, smooth_w=5,
                                 ylabel="Deliveries / episode" if col == 0 else "",
                                 show_legend=(col == 0))
        fig.tight_layout()
    else:
        # Old flat format: results_block = {method: {deliveries_curves}}
        fig, ax = plt.subplots(figsize=(6.5, 4))
        plot_learning_curves(ax, results_block, smooth_w=5,
                             ylabel="Deliveries / episode",
                             title=f"Homogeneous env — {env}")
        fig.tight_layout()

    path = out_dir / "fig8_homo_curves.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig_battery_checkups(data, out_dir):
    """
    Plot battery checkup curves similarly to delivery curves when battery logs are present.

    Supports:
      - standalone homo results JSON (backend-nested)
      - combined experiment_results JSON
      - heuristic battery traces if available
    """
    sections = []

    if "falsification" in data and "results" in data:
        # Standalone homogeneous results: one panel per backend.
        backends = data.get("backends", list(data.get("results", {}).keys()))
        for backend in backends:
            b_methods = data.get("results", {}).get(backend, {})
            sections.append((f"Homogeneous - {backend.upper()}", b_methods))
    else:
        # Combined experiment JSON
        hetero_methods = data.get("results", {}).get("heterogeneous", {}).get("methods", {})
        if hetero_methods:
            sections.append(("Heterogeneous", hetero_methods))

        homo = data.get("results", {}).get("homogeneous", {})
        if "results" in homo:
            for backend in homo.get("backends", list(homo["results"].keys())):
                sections.append((f"Homogeneous - {backend.upper()}", homo["results"].get(backend, {})))
        elif "methods" in homo:
            sections.append(("Homogeneous", homo.get("methods", {})))

    plotted_sections = []
    for title, methods_data in sections:
        has_any = any(_battery_seed_curves(methods_data.get(m, {})) for m in METHODS)
        if has_any:
            plotted_sections.append((title, methods_data))

    heuristic = data.get("results", {}).get("heuristic", {}) if isinstance(data.get("results", {}), dict) else {}
    heuristic_envs = []
    for env_name, env_data in heuristic.items():
        curve = env_data.get("battery_mean_curve_per_episode", [])
        arr = np.asarray(curve, dtype=float)
        if arr.ndim == 2 and arr.shape[0] > 0:
            heuristic_envs.append((env_name, arr.mean(axis=1)))

    total_panels = len(plotted_sections) + (1 if heuristic_envs else 0)
    if total_panels == 0:
        print("  [skip] No battery telemetry found for plotting.")
        return

    fig, axes = plt.subplots(1, total_panels, figsize=(5.6 * total_panels, 4), squeeze=False)
    axes = axes[0]
    panel = 0

    for title, methods_data in plotted_sections:
        plot_battery_curves(
            axes[panel],
            methods_data,
            smooth_w=5,
            title=f"Battery checkup - {title}",
            show_legend=True,
        )
        panel += 1

    if heuristic_envs:
        ax = axes[panel]
        for env_name, curve in heuristic_envs:
            short = env_name.replace("tarware-", "").replace("-partialobs-chg-v1", "")
            y = smooth(np.asarray(curve, dtype=float), 5)
            x = np.arange(len(y))
            ax.plot(x, y, lw=1.6, label=short)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Battery level (mean over agents)")
        ax.set_title("Battery checkup - Heuristic")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

    fig.tight_layout()
    path = out_dir / "fig9_battery_checkups.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 10: Ablation study — 2×2 analysis panel ───────────────────────────

def fig_ablation(data: dict, out_dir: Path) -> None:
    """
    Generate a 2×2 figure from ablation_results_*.json.

    Panel layout:
      [0,0] Learning curves  — deliveries/ep over episodes (all 7 methods)
      [0,1] Task completion  — mean_deliveries ± std bar chart
      [1,0] Mutualism frac   — mutualism_fraction bar chart per method
      [1,1] Confidence       — mean_confidence bar chart with reference lines

    Accepts the top-level ablation dict or the full JSON (auto-detects both).
    """
    import csv

    abl = data.get("ablation", data)
    if not abl:
        print("  [skip] No ablation data found.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [m for m in ABLATION_ORDER if m in abl]
    colors  = [ABLATION_COLORS[m] for m in methods]
    labels  = [ABLATION_LABELS[m] for m in methods]
    x       = np.arange(len(methods))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    cfg = data.get("config", {})
    env_name = cfg.get("env", "")
    fig.suptitle(
        f"Ablation Study — {env_name}\n"
        f"({cfg.get('timesteps', '?'):,} steps × {cfg.get('seeds', '?')} seeds)",
        fontsize=12, y=1.01,
    )

    # ── Panel [0,0]: Learning curves ─────────────────────────────────────────
    ax = axes[0, 0]
    for m, c in zip(methods, colors):
        curves = abl[m].get("deliveries_curves", [])
        if not curves:
            continue
        mu, sd = mean_std(curves)
        if len(mu) > 1:
            mu = smooth(mu, w=max(3, len(mu) // 20))
            sd = smooth(sd, w=max(3, len(sd) // 20))
        xp = np.arange(len(mu))
        ax.plot(xp, mu, color=c, lw=1.6, label=ABLATION_LABELS[m])
        ax.fill_between(xp, mu - sd, mu + sd, color=c, alpha=0.15)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Deliveries / episode")
    ax.set_title("Learning curves")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.75,
              ncol=1 if len(methods) <= 4 else 2)

    # ── Panel [0,1]: Task completion bar chart ────────────────────────────────
    ax = axes[0, 1]
    means = [abl[m]["mean_deliveries"] for m in methods]
    stds  = [abl[m]["std_deliveries"]  for m in methods]
    bars  = ax.bar(x, means, color=colors, yerr=stds, capsize=4,
                   edgecolor="white", linewidth=0.5)
    # Annotate each bar with the value
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean deliveries / episode (tail)")
    ax.set_title("Task completion")

    # ── Panel [1,0]: Mutualism fraction bar chart ─────────────────────────────
    ax = axes[1, 0]
    mut_vals = [abl[m]["mutualism_fraction"] for m in methods]
    comp_vals = [abl[m].get("competition_fraction", 0.0) for m in methods]
    width = 0.35
    ax.bar(x - width / 2, mut_vals,  width, color=colors, alpha=0.85,
           label="Mutualism", edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, comp_vals, width, color=colors, alpha=0.45,
           hatch="//", label="Competition", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of interactions (tail)")
    ax.set_title("Relationship fractions (mutualism solid, competition hatched)")
    ax.legend(fontsize=8, framealpha=0.7)

    # ── Panel [1,1]: Confidence bar chart ────────────────────────────────────
    ax = axes[1, 1]
    conf_vals = [abl[m]["mean_confidence"] for m in methods]
    cbars = ax.bar(x, conf_vals, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(cbars, conf_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    # Reference lines: expected random baseline (1/N_REL ≈ 0.20) and symbiotic target
    ax.axhline(0.20, color="gray",   ls="--", lw=1.2, alpha=0.7,
               label="Random expected (0.20)")
    ax.axhline(0.75, color="#B47CC7", ls=":",  lw=1.2, alpha=0.7,
               label="Symbiotic target (0.75)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean peak confidence (EMA)")
    ax.set_title("Relationship confidence\n(low → shaping diluted; high → full shaping)")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, framealpha=0.7)

    fig.tight_layout()
    path = out_dir / "fig_ablation_analysis.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def table_ablation(data: dict, out_dir: Path) -> None:
    """
    Write ablation_summary.csv summarising all 7 ablation methods.

    Columns: method, mean_deliveries, std_deliveries, vs_baseline,
             mutualism_fraction, competition_fraction, mean_confidence
    """
    import csv

    abl = data.get("ablation", data)
    if not abl:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = abl.get("individual", {}).get("mean_deliveries", 0.0)
    methods  = [m for m in ABLATION_ORDER if m in abl]

    path = out_dir / "ablation_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "mean_deliveries", "std_deliveries", "vs_baseline",
            "mutualism_fraction", "competition_fraction", "mean_confidence",
        ])
        for m in methods:
            r = abl[m]
            writer.writerow([
                m,
                f"{r['mean_deliveries']:.4f}",
                f"{r['std_deliveries']:.4f}",
                f"{r['mean_deliveries'] - baseline:+.4f}",
                f"{r['mutualism_fraction']:.4f}",
                f"{r.get('competition_fraction', 0.0):.4f}",
                f"{r['mean_confidence']:.4f}",
            ])
    print(f"  Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def find_latest_result(results_dir: Path) -> Path:
    """Find the most recently modified experiment JSON in results_dir."""
    patterns = [
        "experiment_results_*.json",
        "convergence_results_*.json",
        "gradient_results_*.json",
        "homo_results_*.json",
        "ablation_results_*.json",
    ]
    files = []
    for pat in patterns:
        files.extend(results_dir.glob(pat))
    if not files:
        raise FileNotFoundError(f"No experiment JSON files found in {results_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def _detect_json_type(data: dict) -> str:
    """
    Return one of:
      'ablation'    — ablation_results_*.json (has 'ablation' key)
      'combined'    — experiment_results_*.json (has 'checks')
      'convergence' — convergence_results_*.json (has 'backends', no 'h_values'/'falsification')
      'gradient'    — gradient_results_*.json (has 'h_values')
      'homo'        — homo_results_*.json (has 'falsification')
    """
    if "ablation" in data:
        return "ablation"
    if "checks" in data:
        return "combined"
    if "methods" in data and "env" in data:
        return "homo"
    if "h_values" in data:
        return "gradient"
    if "falsification" in data:
        return "homo"
    if "backends" in data and "results" in data:
        return "convergence"
    return "combined"  # fallback: try combined handlers


def main():
    parser = argparse.ArgumentParser(
        description="Plot experiment findings from any result JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "json_path", nargs="?", default=None,
        help="Path to *_results_*.json (default: latest in runs/results/)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory for figures (default: <json_dir>/figures/)",
    )
    args = parser.parse_args()

    # Resolve input file
    if args.json_path:
        json_path = Path(args.json_path)
    else:
        results_dir = Path("runs/results")
        json_path   = find_latest_result(results_dir)

    print(f"Reading: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    json_type = _detect_json_type(data)
    print(f"JSON type detected: {json_type}")

    # Resolve output directory
    out_dir = Path(args.out) if args.out else json_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures → {out_dir}/\n")

    if json_type == "ablation":
        # Standalone ablation_results_*.json
        fig_ablation(data, out_dir)
        table_ablation(data, out_dir)

    elif json_type == "convergence":
        # Standalone convergence_results_*.json
        fig_convergence(data, out_dir)

    elif json_type == "gradient":
        # Standalone gradient_results_*.json
        fig_gradient(data, out_dir)

    elif json_type == "homo":
        # Standalone homo_results_*.json
        fig_homo_curves(data, out_dir)
        fig_battery_checkups(data, out_dir)

    else:
        # Full combined experiment_results_*.json — generate all figures
        heuristic_mean = None
        try:
            hdata = data["results"]["heuristic"]
            hetero_env = data["results"]["heterogeneous"]["env"]
            if hetero_env in hdata:
                heuristic_mean = hdata[hetero_env]["mean_deliveries"]
            else:
                heuristic_mean = next(iter(hdata.values()))["mean_deliveries"]
        except (KeyError, StopIteration):
            pass

        fig_hetero_curves(data, out_dir, heuristic_mean)
        fig_heuristic(data, out_dir)
        fig_gradient(data, out_dir)
        fig_convergence(data, out_dir)
        fig_tsi_rsi(data, out_dir)
        fig_hetero_vs_homo(data, out_dir)
        fig_claims_summary(data, out_dir)
        fig_homo_curves(data, out_dir)
        fig_battery_checkups(data, out_dir)

    n_png = len(list(out_dir.glob("*.png")))
    n_pdf = len(list(out_dir.glob("*.pdf")))
    print(f"\nDone. {n_png} PNG / {n_pdf} PDF files in {out_dir}")


if __name__ == "__main__":
    main()
