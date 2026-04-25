"""
PRISM — C2 Convergence Validation.

Validates that symbiotic reward shaping preserves convergence class:
  - Gradient norms stay bounded (no explosion)
  - r_sym stays bounded relative to r_task (boundedness ratio)
  - Convergence speed comparison: symbiotic vs flat-cooperative

Runs both conditions at shorter budget (500k steps) and analyses the
update_metrics.csv files written by each seed for gradient norm history.

Usage
-----
python experiments/run_convergence.py \\
    --symbiotic_json  runs/results/prism_symbiotic_*.json \\
    --flat_coop_json  runs/results/prism_flat_coop_*.json \\
    --symbiotic_ckpt  runs/prism_symbiotic \\
    --flat_coop_ckpt  runs/prism_flat_cooperative \\
    --output          runs/results/convergence_results.json

If the result JSONs don't exist yet, pass --run to train from scratch:
    python experiments/run_convergence.py --run --timesteps 500000 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load_csv(path: Path) -> Optional[Dict[str, List[float]]]:
    """Load a CSV into a dict of column → list of floats."""
    if not path.exists():
        return None
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None
    cols = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            try:
                cols[k].append(float(v))
            except (ValueError, TypeError):
                cols[k].append(0.0)
    return cols


def _convergence_episode(curve: List[float], threshold: float = 0.90) -> int:
    """First episode where the smoothed curve exceeds threshold × final mean."""
    arr = np.array(curve, dtype=float)
    if len(arr) < 10:
        return -1
    target = threshold * arr[-20:].mean()
    w = min(10, len(arr) // 5)
    smoothed = np.convolve(arr, np.ones(w) / w, mode="valid")
    above = np.where(smoothed >= target)[0]
    return int(above[0]) if len(above) else -1


def _boundedness_ratio(eval_csv: Dict[str, List[float]]) -> Dict[str, float]:
    """Compute |mean_shaped - mean_raw| / (|mean_raw| + 1e-8) as boundedness proxy."""
    shaped = np.array(eval_csv.get("mean_shaped_reward", [0.0]))
    raw    = np.array(eval_csv.get("mean_raw_reward",    [0.0]))
    diff   = np.abs(shaped - raw)
    denom  = np.abs(raw) + 1e-8
    ratio  = diff / denom
    return {
        "mean":    float(ratio.mean()),
        "max":     float(ratio.max()),
        "bounded": bool(ratio.max() < 10.0),   # bounded if shaping < 10× task reward
    }


def _analyse_condition(result_json: Path, ckpt_dir: Path,
                       backend: str, n_seeds: int) -> Dict:
    data = json.load(open(result_json))
    canonical = data.get("canonical_backend", backend)
    r = data.get("results", {}).get(canonical, {})

    delivery_curves = r.get("deliveries_curves", [])
    grad_norm_history: List[List[float]] = []
    boundedness_per_seed: List[Dict] = []

    for seed_idx in range(n_seeds):
        seed_dir = ckpt_dir / f"{canonical}_seed{seed_idx}"
        update_csv = _load_csv(seed_dir / "update_metrics.csv")
        eval_csv   = _load_csv(seed_dir / "eval_metrics.csv")

        if update_csv and "agv_grad_norm" in update_csv:
            gn = [a + p for a, p in zip(update_csv["agv_grad_norm"],
                                         update_csv.get("pick_grad_norm", [0.0]*len(update_csv["agv_grad_norm"])))]
            grad_norm_history.append(gn)

        if eval_csv:
            boundedness_per_seed.append(_boundedness_ratio(eval_csv))

    convergence_eps = [_convergence_episode(c) for c in delivery_curves if c]

    return {
        "mean_completion":      r.get("mean_completion", 0.0),
        "std_completion":       r.get("std_completion",  0.0),
        "tsi":                  r.get("tsi", 0.0),
        "rsi":                  r.get("rsi", 0.0),
        "convergence_episodes": int(np.mean(convergence_eps)) if convergence_eps else -1,
        "grad_norm_mean":       float(np.mean([np.mean(g) for g in grad_norm_history])) if grad_norm_history else None,
        "grad_norm_max":        float(np.max([np.max(g)  for g in grad_norm_history])) if grad_norm_history else None,
        "grad_norm_bounded":    all(np.max(g) < 1.0 for g in grad_norm_history) if grad_norm_history else None,
        "boundedness_ratio_mean": float(np.mean([b["mean"] for b in boundedness_per_seed])) if boundedness_per_seed else None,
        "boundedness_ratio_max":  float(np.max( [b["max"]  for b in boundedness_per_seed])) if boundedness_per_seed else None,
        "boundedness_bounded":    all(b["bounded"] for b in boundedness_per_seed) if boundedness_per_seed else None,
        "deliveries_curves":    delivery_curves,
        "mutualism_curves":     r.get("mutualism_curves", []),
        "grad_norm_histories":  grad_norm_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM C2 — Convergence validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbiotic_json",  default=None)
    parser.add_argument("--flat_coop_json",  default=None)
    parser.add_argument("--symbiotic_ckpt",  default="runs/prism_symbiotic")
    parser.add_argument("--flat_coop_ckpt",  default="runs/prism_flat_cooperative")
    parser.add_argument("--backend",         default="mappo")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output",          default="runs/results/convergence_results.json")
    # If --run is passed, train from scratch first
    parser.add_argument("--run",             action="store_true",
                        help="Run training before analysing (uses configs/prism_*.yaml)")
    parser.add_argument("--timesteps",       default=500_000, type=int)
    parser.add_argument("--symbiotic_output", default="runs/results/convergence_symbiotic.json")
    parser.add_argument("--flat_coop_output", default="runs/results/convergence_flat_coop.json")
    args = parser.parse_args()

    py = sys.executable

    if args.run:
        print("=== Training symbiotic condition (C2) ===")
        subprocess.run([
            py, "experiments/run_symbiotic.py",
            "--config",       "configs/prism_symbiotic.yaml",
            "--timesteps",    str(args.timesteps),
            "--seeds",        *map(str, args.seeds),
            "--checkpoint_dir", args.symbiotic_ckpt,
            "--output",       args.symbiotic_output,
        ], cwd=ROOT, check=True)

        print("=== Training flat-cooperative condition (C2) ===")
        subprocess.run([
            py, "experiments/run_flat_cooperative.py",
            "--config",       "configs/prism_flat_cooperative.yaml",
            "--timesteps",    str(args.timesteps),
            "--seeds",        *map(str, args.seeds),
            "--checkpoint_dir", args.flat_coop_ckpt,
            "--output",       args.flat_coop_output,
        ], cwd=ROOT, check=True)

        args.symbiotic_json  = args.symbiotic_output
        args.flat_coop_json  = args.flat_coop_output

    if not args.symbiotic_json or not args.flat_coop_json:
        parser.error("Provide --symbiotic_json and --flat_coop_json, or use --run")

    print("Analysing convergence metrics …")
    sym  = _analyse_condition(Path(args.symbiotic_json),  Path(args.symbiotic_ckpt),
                               args.backend, len(args.seeds))
    flat = _analyse_condition(Path(args.flat_coop_json),  Path(args.flat_coop_ckpt),
                               args.backend, len(args.seeds))

    combined = {
        "conditions": ["symbiotic", "flat_cooperative"],
        "results": {
            "symbiotic":       sym,
            "flat_cooperative": flat,
        },
        "C2_checks": {
            "symbiotic_grad_norms_bounded":     sym["grad_norm_bounded"],
            "flat_coop_grad_norms_bounded":     flat["grad_norm_bounded"],
            "symbiotic_r_sym_bounded":          sym["boundedness_bounded"],
            "symbiotic_faster_convergence":     (
                sym["convergence_episodes"] < flat["convergence_episodes"]
                if sym["convergence_episodes"] > 0 and flat["convergence_episodes"] > 0
                else None
            ),
            "symbiotic_convergence_ep":         sym["convergence_episodes"],
            "flat_coop_convergence_ep":         flat["convergence_episodes"],
            "symbiotic_grad_norm_mean":         sym["grad_norm_mean"],
            "flat_coop_grad_norm_mean":         flat["grad_norm_mean"],
            "symbiotic_boundedness_ratio_mean": sym["boundedness_ratio_mean"],
        },
    }

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nC2 Convergence Results:")
    print(f"{'Metric':<40} {'Symbiotic':>12} {'Flat-coop':>12}")
    print("-" * 66)
    print(f"{'Mean completion':<40} {sym['mean_completion']:>12.3f} {flat['mean_completion']:>12.3f}")
    print(f"{'Convergence episode':<40} {sym['convergence_episodes']:>12} {flat['convergence_episodes']:>12}")
    if sym["grad_norm_mean"] is not None:
        print(f"{'Grad norm mean':<40} {sym['grad_norm_mean']:>12.4f} {flat['grad_norm_mean']:>12.4f}")
    if sym["boundedness_ratio_mean"] is not None:
        print(f"{'Boundedness ratio (r_sym/r_task)':<40} {sym['boundedness_ratio_mean']:>12.4f} {'N/A':>12}")
    print(f"{'Grad norms bounded (<1.0)':<40} {str(sym['grad_norm_bounded']):>12} {str(flat['grad_norm_bounded']):>12}")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
