"""Reproduce every figure and number of the repository, phase by phase.

Usage, from the repository root:

    python scripts/run_all.py                    # phases 1 to 5 (about 17 min on a laptop CPU)
    python scripts/run_all.py --with-mass-matrix # also the NUTS mass matrix comparison (+7 min)
    python scripts/run_all.py --from-phase 4     # start at phase 4 (needs earlier outputs)

Each phase script writes its figures to figures/ and its numbers to results/.
All random seeds are fixed, so a rerun reproduces the same numbers on the same
software versions.
"""

from __future__ import annotations

import argparse
import runpy
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

PHASES: list[tuple[int, str]] = [
    (1, "phase1_forward.py"),
    (2, "phase2_synthetic_data.py"),
    (3, "phase3_calibration.py"),
    (4, "phase4_identifiability.py"),
    (5, "phase5_counterfactual.py"),
]


def main() -> None:
    """Run the selected phase scripts in order and report their wall times."""
    parser = argparse.ArgumentParser(description="Reproduce all results of Metamorph.")
    parser.add_argument("--from-phase", type=int, default=1, help="first phase to run (1 to 5)")
    parser.add_argument("--with-mass-matrix", action="store_true",
                        help="also run scripts/phase3_mass_matrix.py")
    args = parser.parse_args()

    to_run = [(p, s) for p, s in PHASES if p >= args.from_phase]
    if args.with_mass_matrix and args.from_phase <= 3:
        to_run.insert([p for p, _ in to_run].index(3) + 1, (3, "phase3_mass_matrix.py"))

    timings = []
    for phase, script in to_run:
        print(f"\n=== Phase {phase}: {script} ===", flush=True)
        t0 = time.perf_counter()
        sys.argv = [script]  # phase scripts parse their own (empty) command line
        runpy.run_path(str(SCRIPTS_DIR / script), run_name="__main__")
        timings.append((script, time.perf_counter() - t0))

    print("\n=== Done ===")
    for script, seconds in timings:
        print(f"  {script:30s} {seconds / 60:6.1f} min")


if __name__ == "__main__":
    main()
