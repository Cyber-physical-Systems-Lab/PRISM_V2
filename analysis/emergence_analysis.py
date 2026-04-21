"""
Emergent Behavior Analysis for TARWARE MARL Runs
=================================================
Detects four coordination behaviors that may have emerged spontaneously in
the trained agent populations, using per-episode CSV logs as the primary
data source (not the aggregated result JSONs which lose per-episode detail).

Behaviors detected:
  1. Energy-aware cooperation       — agents learned to conserve and share energy
  2. Spontaneous task redistribution — agents self-organized task assignments
  3. Reduced conflict               — agents learned to avoid resource depletion conflicts
  4. Role differentiation           — agents settled into stable, distinct functional roles

Primary data:  runs/c3_hetero/**/eval_metrics.csv
               runs/c3_hetero/**/update_metrics.csv
Secondary data: runs/results/hetero_results.json  (TSI / RSI scalars)

Usage:
    python -m analysis.emergence_analysis
    python -m analysis.emergence_analysis --runs-dir /path/to/runs
    python -m analysis.emergence_analysis --no-plot
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from analysis.metrics import find_convergence_episode


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

METHODS = ["individual", "team", "unclassified", "symbiotic"]

_METHOD_COLORS = {
    "individual":   "#1f77b4",
    "team":         "#ff7f0e",
    "unclassified": "#2ca02c",
    "symbiotic":    "#d62728",
}
_METHOD_LABELS = {
    "individual":   "Individual",
    "team":         "Team",
    "unclassified": "Unclassified",
    "symbiotic":    "Symbiotic",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunRecord:
    method: str
    seed: str
    run_path: Path
    eval_df: pd.DataFrame
    update_df: Optional[pd.DataFrame]
    peak_deliveries: float


@dataclass
class EmergenceResult:
    behavior: str
    detected: bool
    confidence: float          # 0–1
    evidence: Dict
    stats: Dict
    description: str


# ─────────────────────────────────────────────────────────────────────────────
# CSV discovery and loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        return df if len(df) > 0 else None
    except Exception:
        return None


def _rolling_peak(series: pd.Series, window: int = 10) -> float:
    if len(series) < window:
        return float(series.max()) if len(series) > 0 else 0.0
    return float(series.rolling(window).mean().max())


def collect_runs(runs_dir: Path) -> List[RunRecord]:
    """Discover all eval_metrics.csv under c3_hetero and build RunRecord list."""
    c3_dir = runs_dir / "c3_hetero"
    if not c3_dir.exists():
        return []

    records: List[RunRecord] = []
    for csv_path in sorted(c3_dir.rglob("eval_metrics.csv")):
        seed_dir    = csv_path.parent
        method_dir  = seed_dir.parent
        method_name = method_dir.name
        if method_name not in METHODS:
            continue

        eval_df = _load_csv(csv_path)
        if eval_df is None or "deliveries" not in eval_df.columns:
            continue

        update_path = seed_dir / "update_metrics.csv"
        update_df   = _load_csv(update_path)

        seed_id = seed_dir.name
        peak    = _rolling_peak(eval_df["deliveries"])

        records.append(RunRecord(
            method=method_name,
            seed=seed_id,
            run_path=seed_dir,
            eval_df=eval_df,
            update_df=update_df,
            peak_deliveries=peak,
        ))

    return records


def filter_outstanding(records: List[RunRecord], top_fraction: float = 0.5) -> List[RunRecord]:
    """Return runs in the top fraction by peak deliveries."""
    if not records:
        return []
    peaks = np.array([r.peak_deliveries for r in records])
    threshold = np.percentile(peaks, (1 - top_fraction) * 100)
    outstanding = [r for r in records if r.peak_deliveries >= threshold]
    # always keep at least one record per method
    methods_covered = {r.method for r in outstanding}
    for r in records:
        if r.method not in methods_covered:
            outstanding.append(r)
            methods_covered.add(r.method)
    return outstanding


def records_by_method(records: List[RunRecord]) -> Dict[str, List[RunRecord]]:
    out: Dict[str, List[RunRecord]] = {m: [] for m in METHODS}
    for r in records:
        out[r.method].append(r)
    return out


def load_tsi_rsi(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load TSI/RSI scalars from hetero_results.json."""
    path = results_dir / "hetero_results.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        out: Dict[str, Dict[str, float]] = {}
        for method, md in data.get("methods", {}).items():
            out[method] = {"tsi": float(md.get("tsi", 0.0)),
                           "rsi": float(md.get("rsi", 0.0))}
        return out
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _window_idx(n: int, start_frac: float, end_frac: float) -> slice:
    return slice(int(n * start_frac), max(int(n * end_frac), 1))


def battery_conservation_score(df: pd.DataFrame) -> float:
    """
    Ratio of late-window battery mean to early-window battery mean.
    >1 means agents learned to keep batteries healthier over time.
    """
    col = "battery_mean_all_agents"
    if col not in df.columns or len(df) < 4:
        return 1.0
    n = len(df)
    early = df[col].iloc[_window_idx(n, 0.0, 0.25)].mean()
    late  = df[col].iloc[_window_idx(n, 0.75, 1.0)].mean()
    return float(late / early) if early > 0 else 1.0


def depletion_rate(df: pd.DataFrame, start_frac: float = 0.0,
                   end_frac: float = 1.0) -> float:
    """Fraction of episodes in window where battery_min == 0 (any agent depleted)."""
    col = "battery_min_all_agents"
    if col not in df.columns:
        return 0.0
    n = len(df)
    window = df[col].iloc[_window_idx(n, start_frac, end_frac)]
    return float((window == 0).mean())


def per_agent_battery_divergence(df: pd.DataFrame) -> float:
    """
    Mean episode-level std across per-agent battery columns.
    High value = different agents consistently have different energy profiles.
    """
    bat_cols = [c for c in df.columns if c.startswith("battery_mean_agent_")]
    if len(bat_cols) < 2:
        return 0.0
    return float(df[bat_cols].std(axis=1).mean())


def entropy_trajectory(df: pd.DataFrame, col: str) -> Tuple[float, float, float]:
    """(initial_mean, final_mean, reduction_fraction) for an entropy column."""
    if col not in df.columns or len(df) < 4:
        return (0.0, 0.0, 0.0)
    n = len(df)
    init  = float(df[col].iloc[_window_idx(n, 0.0, 0.1)].mean())
    final = float(df[col].iloc[_window_idx(n, 0.9, 1.0)].mean())
    reduction = (init - final) / init if init > 0 else 0.0
    return (init, final, reduction)


def delivery_stats(df: pd.DataFrame) -> Dict[str, float]:
    col = "deliveries"
    if col not in df.columns or len(df) == 0:
        return {"peak": 0.0, "final_mean": 0.0, "improvement": 0.0, "convergence_ep": -1}
    n = len(df)
    early_mean = float(df[col].iloc[_window_idx(n, 0.0, 0.25)].mean())
    final_mean = float(df[col].iloc[_window_idx(n, 0.75, 1.0)].mean())
    peak       = _rolling_peak(df[col])
    conv_ep    = find_convergence_episode(df[col].values)
    improvement = (final_mean - early_mean) / max(early_mean, 0.01)
    return {
        "peak": peak,
        "final_mean": final_mean,
        "early_mean": early_mean,
        "improvement": improvement,
        "convergence_ep": conv_ep,
    }


def battery_min_trend_pvalue(df: pd.DataFrame) -> float:
    """p-value for increasing (improving) trend in battery_min over episodes.
    Low p = battery_min is significantly rising = fewer depletions over time.
    Returns 1.0 (no evidence) when the series is constant or too short."""
    col = "battery_min_all_agents"
    if col not in df.columns or len(df) < 10:
        return 1.0
    y = df[col].values
    if np.std(y) < 1e-9:          # constant series → correlation undefined
        return 1.0
    x = np.arange(len(df))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = stats.spearmanr(x, y)
    return 1.0 if np.isnan(p) else float(p)


def _aggregate(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "n": len(values)}


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class EmergenceDetector:

    # Thresholds
    BATTERY_CONSERVATION_MIN = 1.10   # 10% improvement in battery mean
    DEPLETION_REDUCTION_MIN  = 0.15   # 15 percentage-point reduction
    ENTROPY_REDUCTION_MIN    = 0.70   # 70% reduction in policy entropy
    DELIVERY_IMPROVEMENT_MIN = 0.50   # 50% relative improvement
    ENTROPY_ASYMMETRY_MIN    = 0.50   # AGV vs picker entropy ratio
    BATTERY_DIVERGENCE_MIN   = 5.0    # std across agents (0-100 scale)
    TSI_THRESHOLD            = 0.80
    RSI_THRESHOLD            = 0.80
    P_THRESHOLD              = 0.10

    def __init__(
        self,
        by_method: Dict[str, List[RunRecord]],
        tsi_rsi:   Dict[str, Dict[str, float]],
    ):
        self.by_method = by_method
        self.tsi_rsi   = tsi_rsi

    def _eval_dfs(self, method: str) -> List[pd.DataFrame]:
        return [r.eval_df for r in self.by_method.get(method, [])]

    def _update_dfs(self, method: str) -> List[pd.DataFrame]:
        return [r.update_df for r in self.by_method.get(method, [])
                if r.update_df is not None]

    # ── 1. Energy-aware cooperation ─────────────────────────────────────────

    def detect_energy_cooperation(self) -> EmergenceResult:
        """
        Agents that observe all battery levels should learn to avoid depletion,
        especially in runs with symbiotic reward shaping.  We look for:
          (a) battery_conservation_score > 1.10  (healthier batteries over time)
          (b) depletion rate drops from early to late training
        Across all outstanding runs, not just one method vs another.
        """
        all_cons_scores:  List[float] = []
        all_depl_drop:    List[float] = []
        best_method = None
        best_score  = -np.inf

        method_summary: Dict[str, Dict] = {}
        for method, dfs in [(m, self._eval_dfs(m)) for m in METHODS]:
            if not dfs:
                continue
            cons  = [battery_conservation_score(df) for df in dfs]
            early = [depletion_rate(df, 0.0, 0.25)  for df in dfs]
            late  = [depletion_rate(df, 0.75, 1.0)  for df in dfs]
            drops = [e - l for e, l in zip(early, late)]
            method_summary[method] = {
                "conservation_score": _aggregate(cons),
                "depletion_drop":     _aggregate(drops),
                "early_depletion":    _aggregate(early),
                "late_depletion":     _aggregate(late),
            }
            all_cons_scores.extend(cons)
            all_depl_drop.extend(drops)
            score = np.mean(cons) + np.mean(drops)
            if score > best_score:
                best_score  = score
                best_method = method

        # Is there a method where both signals are clear?
        detected = False
        conf_scores: List[float] = []
        for method, s in method_summary.items():
            cons_ok = s["conservation_score"]["mean"] > self.BATTERY_CONSERVATION_MIN
            drop_ok = s["depletion_drop"]["mean"]     > self.DEPLETION_REDUCTION_MIN
            if cons_ok or drop_ok:
                detected = True
            conf_scores.append(
                0.5 * min(s["conservation_score"]["mean"] / self.BATTERY_CONSERVATION_MIN, 1.5)
                + 0.5 * min(max(s["depletion_drop"]["mean"], 0) / max(self.DEPLETION_REDUCTION_MIN, 1e-6), 1.5)
            )

        confidence = round(float(np.mean(conf_scores)) / 1.5, 3) if conf_scores else 0.0
        evidence = {
            "best_method": best_method,
            "method_summary": {
                m: {
                    "conservation_score": round(v["conservation_score"]["mean"], 3),
                    "depletion_drop":     round(v["depletion_drop"]["mean"],     3),
                    "early_depletion":    round(v["early_depletion"]["mean"],    3),
                    "late_depletion":     round(v["late_depletion"]["mean"],     3),
                }
                for m, v in method_summary.items()
            },
        }

        best_s = method_summary.get(best_method, {}) if best_method else {}
        desc = (
            f"Best: {best_method} | "
            f"conservation={best_s.get('conservation_score', {}).get('mean', 0):.2f}x "
            f"(thresh {self.BATTERY_CONSERVATION_MIN:.2f}), "
            f"depl.drop={best_s.get('depletion_drop', {}).get('mean', 0):.2f} "
            f"(thresh {self.DEPLETION_REDUCTION_MIN:.2f})"
        )
        return EmergenceResult(
            behavior="energy_aware_cooperation",
            detected=detected, confidence=confidence,
            evidence=evidence, stats={}, description=desc,
        )

    # ── 2. Spontaneous task redistribution ──────────────────────────────────

    def detect_task_redistribution(self) -> EmergenceResult:
        """
        Self-organized task assignment manifests as:
          (a) AGV policy entropy drops to near-zero in at least some seeds
              (agents commit to specific task roles without being told to)
          (b) Peak delivery count substantially exceeds early-training baseline

        We report the fraction of seeds where each signal is clearly present,
        since emergence is stochastic: not every random seed converges to the
        same specialised solution.
        """
        method_summary: Dict[str, Dict] = {}
        best_method = None
        best_score  = -np.inf

        for method in METHODS:
            eval_dfs   = self._eval_dfs(method)
            update_dfs = self._update_dfs(method)
            if not eval_dfs:
                continue

            # Per-seed: peak-to-baseline ratio (more robust than final vs early)
            del_stats        = [delivery_stats(df) for df in eval_dfs]
            early_means      = [d["early_mean"] for d in del_stats]
            peaks            = [d["peak"]        for d in del_stats]
            peak_ratios      = [pk / max(em, 0.01) for pk, em in zip(peaks, early_means)]
            frac_peak_ok     = float(np.mean([r > 2.0 for r in peak_ratios]))

            # Per-seed: AGV entropy reduction
            agv_reductions: List[float] = []
            agv_finals:      List[float] = []
            for udf in update_dfs:
                _, af, red = entropy_trajectory(udf, "agv_entropy")
                agv_reductions.append(red)
                agv_finals.append(af)

            frac_ent_ok = float(np.mean([r > self.ENTROPY_REDUCTION_MIN
                                          for r in agv_reductions])) if agv_reductions else 0.0
            best_ent_red = max(agv_reductions) if agv_reductions else 0.0

            method_summary[method] = {
                "peak_deliveries":           _aggregate(peaks),
                "peak_to_baseline_ratio":    _aggregate(peak_ratios),
                "frac_seeds_peak_ok":        frac_peak_ok,
                "agv_entropy_reduction":     _aggregate(agv_reductions),
                "frac_seeds_entropy_ok":     frac_ent_ok,
                "best_seed_entropy_red":     best_ent_red,
                "n_seeds":                   len(eval_dfs),
            }
            # Score: fraction of seeds showing either signal
            score = 0.5 * frac_peak_ok + 0.5 * frac_ent_ok
            if score > best_score:
                best_score  = score
                best_method = method

        # Detected if ANY method has >= 25% of seeds showing spontaneous specialisation
        FRAC_THRESHOLD = 0.25
        detected   = False
        conf_vals: List[float] = []
        for m, s in method_summary.items():
            frac = max(s["frac_seeds_peak_ok"], s["frac_seeds_entropy_ok"])
            if frac >= FRAC_THRESHOLD:
                detected = True
            conf_vals.append(min(frac / FRAC_THRESHOLD, 1.0))

        confidence = round(float(np.mean(conf_vals)), 3) if conf_vals else 0.0
        evidence = {
            "best_method":       best_method,
            "frac_threshold":    FRAC_THRESHOLD,
            "method_summary": {
                m: {
                    "peak_deliveries":          round(v["peak_deliveries"]["mean"], 1),
                    "peak_to_baseline_ratio":   round(v["peak_to_baseline_ratio"]["mean"], 2),
                    "frac_seeds_peak_ok":       round(v["frac_seeds_peak_ok"], 2),
                    "frac_seeds_entropy_ok":    round(v["frac_seeds_entropy_ok"], 2),
                    "best_seed_entropy_red_pct": round(v["best_seed_entropy_red"] * 100, 1),
                    "n_seeds":                  v["n_seeds"],
                }
                for m, v in method_summary.items()
            },
        }
        best_s = method_summary.get(best_method, {}) if best_method else {}
        desc = (
            f"Best: {best_method} | "
            f"frac_specialized={best_s.get('frac_seeds_entropy_ok', 0):.0%}, "
            f"best_ent_red={best_s.get('best_seed_entropy_red', 0)*100:.0f}%, "
            f"peak_del={best_s.get('peak_deliveries', {}).get('mean', 0):.1f}"
        )
        return EmergenceResult(
            behavior="spontaneous_task_redistribution",
            detected=detected, confidence=confidence,
            evidence=evidence, stats={}, description=desc,
        )

    # ── 3. Reduced conflict ──────────────────────────────────────────────────

    def detect_reduced_conflict(self) -> EmergenceResult:
        """
        Battery depletion (battery_min = 0) is a proxy for resource conflict —
        when agents deplete, they block task completion and disrupt other agents.
        If agents learned to coordinate implicitly, late-training depletion rate
        should be significantly lower than early-training depletion rate.
        We also check for a statistically significant rising trend in battery_min.
        """
        method_summary: Dict[str, Dict] = {}
        best_method = None
        best_score  = -np.inf

        for method in METHODS:
            dfs = self._eval_dfs(method)
            if not dfs:
                continue
            early_dep  = [depletion_rate(df, 0.0,  0.25) for df in dfs]
            late_dep   = [depletion_rate(df, 0.75, 1.0)  for df in dfs]
            drops      = [e - l for e, l in zip(early_dep, late_dep)]
            trend_pvals = [battery_min_trend_pvalue(df) for df in dfs]

            safe_pvals  = [p for p in trend_pvals if not np.isnan(p)]
            sig_trend   = float(np.mean([p < self.P_THRESHOLD for p in safe_pvals])) if safe_pvals else 0.0
            mean_pval   = float(np.mean(safe_pvals)) if safe_pvals else 1.0

            # Late-training battery_min mean (> 0 means agents avoided full depletion)
            bat_min_late = [
                float(df["battery_min_all_agents"].iloc[_window_idx(len(df), 0.75, 1.0)].mean())
                for df in dfs if "battery_min_all_agents" in df.columns
            ]

            method_summary[method] = {
                "early_depletion":     _aggregate(early_dep),
                "late_depletion":      _aggregate(late_dep),
                "depletion_drop":      _aggregate(drops),
                "trend_pvalue_mean":   mean_pval,
                "significant_trend":   sig_trend,
                "battery_min_late":    _aggregate(bat_min_late),
            }
            score = float(np.nan_to_num(np.mean(drops) + sig_trend, nan=0.0))
            if score > best_score:
                best_score  = score
                best_method = method

        DEPL_THRESH = 0.05   # 5 pp drop in depletion rate is meaningful
        detected = False
        conf_vals: List[float] = []
        for m, s in method_summary.items():
            drop_ok      = s["depletion_drop"]["mean"]    > DEPL_THRESH
            trend_ok     = s["significant_trend"]         > 0.3
            bat_min_ok   = s["battery_min_late"]["mean"]  > 2.0  # any non-zero late battery
            if drop_ok or trend_ok or bat_min_ok:
                detected = True
            conf_vals.append(
                0.4 * min(max(s["depletion_drop"]["mean"], 0) / max(DEPL_THRESH, 1e-6), 1.0)
                + 0.3 * s["significant_trend"]
                + 0.3 * min(s["battery_min_late"]["mean"] / 10.0, 1.0)
            )

        confidence = round(float(np.mean(conf_vals)), 3) if conf_vals else 0.0
        evidence = {
            "best_method": best_method,
            "method_summary": {
                m: {
                    "early_depletion_pct":  round(v["early_depletion"]["mean"]   * 100, 1),
                    "late_depletion_pct":   round(v["late_depletion"]["mean"]    * 100, 1),
                    "depletion_drop_pct":   round(v["depletion_drop"]["mean"]    * 100, 1),
                    "battery_min_late":     round(v["battery_min_late"]["mean"],  2),
                    "trend_p_mean":         round(v["trend_pvalue_mean"],         4),
                    "sig_trend_fraction":   round(v["significant_trend"],         2),
                }
                for m, v in method_summary.items()
            },
        }
        best_s = method_summary.get(best_method, {}) if best_method else {}
        desc = (
            f"Best: {best_method} | "
            f"depletion {best_s.get('early_depletion', {}).get('mean', 0)*100:.0f}%"
            f"→{best_s.get('late_depletion', {}).get('mean', 0)*100:.0f}%, "
            f"bat_min_late={best_s.get('battery_min_late', {}).get('mean', 0):.1f}, "
            f"sig_trend={best_s.get('significant_trend', 0):.0%}"
        )
        return EmergenceResult(
            behavior="reduced_conflict",
            detected=detected, confidence=confidence,
            evidence=evidence, stats={}, description=desc,
        )

    # ── 4. Role differentiation ──────────────────────────────────────────────

    def detect_role_differentiation(self) -> EmergenceResult:
        """
        Role differentiation shows up as:
          (a) TSI > 0.80 and RSI > 0.80  (from pre-computed JSON scalars)
          (b) Entropy asymmetry: AGV entropy drops much more than picker entropy
              (AGVs converge to delivery role; pickers remain adaptive)
          (c) Per-agent battery divergence: different agents show different energy
              profiles across episodes (energy signatures of distinct roles)
        """
        method_summary: Dict[str, Dict] = {}
        best_method = None
        best_score  = -np.inf

        for method in METHODS:
            eval_dfs   = self._eval_dfs(method)
            update_dfs = self._update_dfs(method)

            # TSI / RSI from JSON
            tsi = self.tsi_rsi.get(method, {}).get("tsi", 0.0)
            rsi = self.tsi_rsi.get(method, {}).get("rsi", 0.0)

            # Battery divergence
            bat_divs: List[float] = []
            if eval_dfs:
                bat_divs = [per_agent_battery_divergence(df) for df in eval_dfs]

            # Entropy asymmetry (AGV vs picker)
            agv_finals:  List[float] = []
            pick_finals: List[float] = []
            if update_dfs:
                for udf in update_dfs:
                    _, af, _ = entropy_trajectory(udf, "agv_entropy")
                    agv_finals.append(af)
                    if "pick_entropy" in udf.columns:
                        _, pf, _ = entropy_trajectory(udf, "pick_entropy")
                        pick_finals.append(pf)

            entropy_asymmetry = None
            if agv_finals and pick_finals and len(agv_finals) == len(pick_finals):
                agv_mean  = float(np.mean(agv_finals))
                pick_mean = float(np.mean(pick_finals))
                # asymmetry = how much more picker entropy remains vs AGV
                entropy_asymmetry = pick_mean - agv_mean  # positive = AGV more specialized

            method_summary[method] = {
                "tsi":               tsi,
                "rsi":               rsi,
                "battery_divergence": _aggregate(bat_divs),
                "agv_final_entropy":  _aggregate(agv_finals),
                "pick_final_entropy": _aggregate(pick_finals),
                "entropy_asymmetry":  entropy_asymmetry,
            }
            score = tsi + rsi + (entropy_asymmetry or 0.0) / 5.0
            if score > best_score:
                best_score  = score
                best_method = method

        detected = False
        conf_vals: List[float] = []
        for m, s in method_summary.items():
            tsi_ok  = s["tsi"] > self.TSI_THRESHOLD
            rsi_ok  = s["rsi"] > self.RSI_THRESHOLD
            asym_ok = (s["entropy_asymmetry"] is not None
                       and s["entropy_asymmetry"] > self.ENTROPY_ASYMMETRY_MIN)
            div_ok  = s["battery_divergence"]["mean"] > self.BATTERY_DIVERGENCE_MIN
            if (tsi_ok and rsi_ok) or (asym_ok and div_ok):
                detected = True
            tsi_score  = min(s["tsi"]  / self.TSI_THRESHOLD, 1.0)
            rsi_score  = min(s["rsi"]  / self.RSI_THRESHOLD, 1.0)
            asym_score = min(max(s["entropy_asymmetry"] or 0.0, 0.0) / self.ENTROPY_ASYMMETRY_MIN, 1.0)
            conf_vals.append((tsi_score + rsi_score + asym_score) / 3)

        confidence = round(float(np.mean(conf_vals)), 3) if conf_vals else 0.0
        evidence = {
            "best_method": best_method,
            "tsi_threshold": self.TSI_THRESHOLD,
            "rsi_threshold": self.RSI_THRESHOLD,
            "method_summary": {
                m: {
                    "tsi":                round(v["tsi"], 4),
                    "rsi":                round(v["rsi"], 4),
                    "battery_divergence": round(v["battery_divergence"]["mean"], 2),
                    "agv_final_entropy":  round(v["agv_final_entropy"]["mean"],  3),
                    "pick_final_entropy": round(v["pick_final_entropy"]["mean"], 3),
                    "entropy_asymmetry":  round(v["entropy_asymmetry"], 3)
                                          if v["entropy_asymmetry"] is not None else None,
                }
                for m, v in method_summary.items()
            },
        }
        best_s = method_summary.get(best_method, {}) if best_method else {}
        desc = (
            f"Best: {best_method} | "
            f"TSI={best_s.get('tsi', 0):.3f}, RSI={best_s.get('rsi', 0):.3f}, "
            f"entropy asym={best_s.get('entropy_asymmetry') or 'N/A'}, "
            f"bat-div={best_s.get('battery_divergence', {}).get('mean', 0):.1f}"
        )
        return EmergenceResult(
            behavior="role_differentiation",
            detected=detected, confidence=confidence,
            evidence=evidence, stats={}, description=desc,
        )

    def run_all(self) -> Dict[str, EmergenceResult]:
        return {
            "energy_aware_cooperation":        self.detect_energy_cooperation(),
            "spontaneous_task_redistribution": self.detect_task_redistribution(),
            "reduced_conflict":                self.detect_reduced_conflict(),
            "role_differentiation":            self.detect_role_differentiation(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="same")


def _label_detected(ax: plt.Axes, result: EmergenceResult) -> None:
    color = "#2ca02c" if result.detected else "#d62728"
    label = f"{'DETECTED' if result.detected else 'NOT DETECTED'}  conf={result.confidence:.2f}"
    ax.text(0.02, 0.97, label, transform=ax.transAxes,
            fontsize=7, color=color, va="top", fontweight="bold")


def _plot_curve_per_method(ax, by_method, col, ylabel, smooth=5):
    for method in METHODS:
        dfs = [r.eval_df for r in by_method.get(method, []) if col in r.eval_df.columns]
        if not dfs:
            continue
        min_len = min(len(df) for df in dfs)
        arr = np.array([df[col].values[:min_len] for df in dfs])
        mean = _smooth(arr.mean(axis=0), smooth)
        std  = arr.std(axis=0)
        eps  = np.arange(min_len)
        c    = _METHOD_COLORS.get(method, "gray")
        ax.plot(eps, mean, color=c, label=_METHOD_LABELS.get(method, method), linewidth=1.5)
        ax.fill_between(eps, mean - std, mean + std, color=c, alpha=0.15)
    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7)


def plot_dashboard(
    results:   Dict[str, EmergenceResult],
    by_method: Dict[str, List[RunRecord]],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Emergent Behavior Analysis — TARWARE MARL Outstanding Runs",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Panel 1 — Energy-aware cooperation: battery mean over training
    ax1.set_title("1. Energy-Aware Cooperation\n(battery mean over training)", fontsize=9)
    _plot_curve_per_method(ax1, by_method, "battery_mean_all_agents",
                           "Battery mean (0-100)", smooth=10)
    _label_detected(ax1, results["energy_aware_cooperation"])

    # Panel 2 — Spontaneous task redistribution: delivery learning curves
    ax2.set_title("2. Spontaneous Task Redistribution\n(deliveries per episode)", fontsize=9)
    _plot_curve_per_method(ax2, by_method, "deliveries", "Deliveries / episode")
    # Mark peak for each method
    for method in METHODS:
        dfs = [r.eval_df for r in by_method.get(method, [])]
        if not dfs:
            continue
        peaks = [_rolling_peak(df["deliveries"]) for df in dfs if "deliveries" in df.columns]
        if peaks:
            ax2.axhline(np.mean(peaks), color=_METHOD_COLORS.get(method, "gray"),
                        linestyle="--", alpha=0.5, linewidth=0.8)
    _label_detected(ax2, results["spontaneous_task_redistribution"])

    # Panel 3 — Reduced conflict: battery_min over training
    ax3.set_title("3. Reduced Conflict\n(battery min — higher = fewer depletions)", fontsize=9)
    _plot_curve_per_method(ax3, by_method, "battery_min_all_agents",
                           "Battery min (0-100)", smooth=10)
    _label_detected(ax3, results["reduced_conflict"])

    # Panel 4 — Role differentiation: per-method bar (TSI, RSI, entropy asym)
    ax4.set_title("4. Role Differentiation\n(TSI · RSI · AGV–Picker entropy asymmetry)", fontsize=9)
    ev = results["role_differentiation"].evidence.get("method_summary", {})
    methods_with_data = [m for m in METHODS if m in ev]
    x  = np.arange(len(methods_with_data))
    w  = 0.28
    tsi_vals  = [ev[m]["tsi"]  for m in methods_with_data]
    rsi_vals  = [ev[m]["rsi"]  for m in methods_with_data]
    asym_vals = [
        min(max(ev[m].get("entropy_asymmetry") or 0.0, 0.0) / 5.0, 1.0)
        for m in methods_with_data
    ]
    ax4.bar(x - w, tsi_vals,  w, label="TSI",      color="#5e81ac", alpha=0.85, edgecolor="black", linewidth=0.6)
    ax4.bar(x,     rsi_vals,  w, label="RSI",      color="#bf616a", alpha=0.85, edgecolor="black", linewidth=0.6)
    ax4.bar(x + w, asym_vals, w, label="Ent.Asym/5", color="#a3be8c", alpha=0.85, edgecolor="black", linewidth=0.6)
    ax4.axhline(0.80, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="0.80 threshold")
    ax4.set_xticks(x)
    ax4.set_xticklabels([_METHOD_LABELS.get(m, m) for m in methods_with_data], fontsize=7)
    ax4.set_ylim(0, 1.1)
    ax4.set_ylabel("Index / normalized value", fontsize=8)
    ax4.legend(fontsize=7, loc="lower right")
    _label_detected(ax4, results["role_differentiation"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

_BEHAVIOR_LABELS = {
    "energy_aware_cooperation":        "Energy cooperation    ",
    "spontaneous_task_redistribution": "Task redistribution   ",
    "reduced_conflict":                "Reduced conflict      ",
    "role_differentiation":            "Role differentiation  ",
}


def print_report(results: Dict[str, EmergenceResult],
                 all_records: List[RunRecord],
                 outstanding: List[RunRecord]) -> None:
    W = 90
    print()
    print("╔" + "═" * W + "╗")
    print(f"║{'EMERGENT BEHAVIOR ANALYSIS — TARWARE MARL RUNS':^{W}}║")
    print(f"║{f'Runs analysed: {len(all_records)} total / {len(outstanding)} outstanding':^{W}}║")
    print("╠" + "═" * 24 + "╦" + "═" * 10 + "╦" + "═" * 8 + "╦" + "═" * (W - 44) + "╣")
    print(f"║{'Behavior':<24}║{'Detected':^10}║{'Conf.':^8}║{'Key evidence':<{W - 44}}║")
    print("╠" + "═" * 24 + "╬" + "═" * 10 + "╬" + "═" * 8 + "╬" + "═" * (W - 44) + "╣")
    for key, res in results.items():
        tag  = "  YES  " if res.detected else "   NO  "
        name = _BEHAVIOR_LABELS.get(key, key[:24])
        desc = res.description[: W - 45]
        print(f"║{name:<24}║{tag:^10}║{res.confidence:^8.2f}║{desc:<{W - 44}}║")
    print("╚" + "═" * 24 + "╩" + "═" * 10 + "╩" + "═" * 8 + "╩" + "═" * (W - 44) + "╝")
    print()

    detected = [k for k, r in results.items() if r.detected]
    print(f"Summary: {len(detected)}/{len(results)} behaviors detected.")
    if detected:
        print("Detected:", ", ".join(_BEHAVIOR_LABELS.get(k, k).strip() for k in detected))

    print()
    print("Outstanding run breakdown (top 50% by peak deliveries):")
    by_m: Dict[str, List[RunRecord]] = {}
    for r in outstanding:
        by_m.setdefault(r.method, []).append(r)
    for method in METHODS:
        runs = by_m.get(method, [])
        if runs:
            peaks = [r.peak_deliveries for r in runs]
            print(f"  {method:15s}: {len(runs)} run(s), peak_del={np.mean(peaks):.1f}±{np.std(peaks):.1f}")
    print()


def save_report(results: Dict[str, EmergenceResult],
                output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"),
                        help="Base runs directory containing c3_hetero/ and results/ (default: runs)")
    parser.add_argument("--top-fraction", type=float, default=0.5,
                        help="Fraction of runs to consider outstanding (default: 0.5)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating the dashboard figure")
    args = parser.parse_args(argv)

    print(f"Scanning runs directory: {args.runs_dir.resolve()}")
    all_records = collect_runs(args.runs_dir)
    if not all_records:
        print("ERROR: No eval_metrics.csv files found under "
              f"{args.runs_dir / 'c3_hetero'}.", file=sys.stderr)
        return 1

    print(f"Found {len(all_records)} run(s) across "
          f"{len({r.method for r in all_records})} method(s).")

    outstanding = filter_outstanding(all_records, args.top_fraction)
    by_method   = records_by_method(outstanding)

    results_dir = args.runs_dir / "results"
    tsi_rsi     = load_tsi_rsi(results_dir)
    if not tsi_rsi:
        print("Note: hetero_results.json not found — TSI/RSI will be 0.")

    detector = EmergenceDetector(by_method=by_method, tsi_rsi=tsi_rsi)
    results  = detector.run_all()

    print_report(results, all_records, outstanding)

    report_path = results_dir / "emergence_report.json"
    save_report(results, report_path)
    print(f"Report saved to: {report_path}")

    if not args.no_plot:
        fig_path = results_dir / "figures" / "emergence_report.png"
        plot_dashboard(results, by_method, fig_path)
        print(f"Figure saved to: {fig_path}")
        fig_pdf = results_dir / "figures" / "emergence_report.pdf"
        plot_dashboard(results, by_method, fig_pdf)
        print(f"Figure (PDF) saved to: {fig_pdf}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
