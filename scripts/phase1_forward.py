"""Phase 1: forward model at the true parameters.

Produces:
    figures/phase1_trajectories.png  trajectories per compartment, bare vs coated
    figures/phase1_solver_check.png  JAX vs SciPy vs exact solution
    results/phase1_summary.txt       numbers quoted in the README and docs

Run from the repository root:  python scripts/phase1_forward.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import metamorph  # noqa: F401  (enables float64 in JAX)
from metamorph.model import (
    T_END,
    TISSUES,
    TRUE_PARAMS,
    compartment_totals,
    simulate,
    simulate_exact,
    simulate_scipy,
    system_matrix,
)
from metamorph.plotting import COLOR_BARE, COLOR_COAT, apply_style, plot_trajectories

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"


def main() -> None:
    """Simulate, compare solvers, save figures and a text summary."""
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)

    ts = np.linspace(0.0, T_END, 481)
    ys_jax = np.asarray(simulate(TRUE_PARAMS, ts))  # first call includes JIT compilation
    t0 = time.perf_counter()
    ys_jax = np.asarray(simulate(TRUE_PARAMS, ts))
    t_jax = time.perf_counter() - t0
    t0 = time.perf_counter()
    ys_scipy = simulate_scipy(TRUE_PARAMS, ts)
    t_scipy = time.perf_counter() - t0
    ys_exact = simulate_exact(TRUE_PARAMS, ts)

    # Figure 1: trajectories.
    fig = plot_trajectories(ts, ys_jax, title="Forward model at the true parameters (IV bolus of bare particles, 100 %ID)")
    fig.savefig(FIG_DIR / "phase1_trajectories.png")
    plt.close(fig)

    # Figure 2: solver agreement (absolute error vs exact solution).
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    err_jax = np.abs(ys_jax - ys_exact).max(axis=1)
    err_scipy = np.abs(ys_scipy - ys_exact).max(axis=1)
    ax.semilogy(ts[1:], err_jax[1:], color=COLOR_BARE, label="diffrax Tsit5")
    ax.semilogy(ts[1:], err_scipy[1:], color=COLOR_COAT, label="SciPy LSODA")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("max abs. error vs expm (%ID)")
    ax.set_title("Solver accuracy against the exact solution", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase1_solver_check.png")
    plt.close(fig)

    # Text summary.
    eig = np.linalg.eigvals(system_matrix(TRUE_PARAMS)).real
    nonzero = np.sort(-eig[np.abs(eig) > 1e-12])
    report_ts = np.array([5 / 60, 0.25, 0.5, 1.0, 4.0, 8.0, 24.0])
    totals = compartment_totals(np.asarray(simulate(TRUE_PARAMS, report_ts)))
    columns = ["blood", *TISSUES, "A_renal", "A_hepatic"]

    lines = ["Phase 1 summary (true parameters, dose 100 %ID)", ""]
    lines.append("Amounts in %ID (bare + coated):")
    lines.append("t (h)    " + " ".join(f"{c:>9}" for c in columns) + "      sum")
    for k, t in enumerate(report_ts):
        row = [totals[c][k] for c in columns]
        lines.append(f"{t:7.3f}  " + " ".join(f"{v:9.3f}" for v in row) + f"  {sum(row):8.4f}")
    i_end = -1
    coated_fraction_blood = ys_jax[:, 1] / (ys_jax[:, 0] + ys_jax[:, 1])
    lines += [
        "",
        f"Max |sum - 100| over 481 times, diffrax: {np.abs(ys_jax.sum(axis=1) - 100).max():.2e}",
        f"Max |sum - 100| over 481 times, SciPy:   {np.abs(ys_scipy.sum(axis=1) - 100).max():.2e}",
        f"Max abs error vs exact, diffrax: {np.abs(ys_jax - ys_exact).max():.2e} %ID",
        f"Max abs error vs exact, SciPy:   {np.abs(ys_scipy - ys_exact).max():.2e} %ID",
        f"Max abs difference diffrax vs SciPy: {np.abs(ys_jax - ys_scipy).max():.2e} %ID",
        f"Wall time (481 outputs): diffrax {t_jax * 1e3:.1f} ms (after JIT), SciPy {t_scipy * 1e3:.1f} ms",
        "",
        "Decay rates of the system (-eigenvalues of A, 1/h), excluding the two zero",
        "eigenvalues of the cumulative states:",
        "  " + ", ".join(f"{v:.4f}" for v in nonzero),
        f"  stiffness ratio (max / min) = {nonzero.max() / nonzero.min():.0f}",
        "",
        f"Coated fraction of blood content at t = 1 h: {np.interp(1.0, ts, coated_fraction_blood):.3f}",
        f"Brain content at t_end = {T_END:.0f} h: {ys_jax[i_end, 6] + ys_jax[i_end, 11]:.4f} %ID "
        f"(bare {ys_jax[i_end, 6]:.4f}, coated {ys_jax[i_end, 11]:.4f})",
    ]
    text = "\n".join(lines)
    (RES_DIR / "phase1_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
