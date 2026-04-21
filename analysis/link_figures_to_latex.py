"""Copy experiment figures to the LaTeX figure directory."""
from pathlib import Path
import shutil

RUNS_FIG  = Path("/crex/proj/symmarl_ijrr2025/xuezhi/runs/results/figures")
LATEX_FIG = Path(__file__).parent.parent / "latex/IJRR_SymCoord/figure"

NEEDED = [
    "fig1_hetero_curves.pdf",
    "fig5_tsi_rsi.pdf",
    "fig6_hetero_vs_homo.pdf",
    "emergence_report.pdf",
]


def main() -> None:
    LATEX_FIG.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for name in NEEDED:
        src = RUNS_FIG / name
        dst = LATEX_FIG / name
        if not src.exists():
            print(f"  MISSING  : {src}")
            all_ok = False
        elif dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            print(f"  UP-TO-DATE: {name}")
        else:
            shutil.copy2(src, dst)
            print(f"  COPIED   : {name}  →  {dst}")
    if all_ok:
        print(f"\nAll figures are in {LATEX_FIG}")
    else:
        print("\nRe-run emergence_analysis.py to generate missing PDFs, then re-run this script.")


if __name__ == "__main__":
    main()
