"""Phase 3 (supporting): diagonal vs dense mass matrix for NUTS.

Runs one chain (300 warmup + 300 draws, same seed) with each mass matrix and
reports leapfrog steps per iteration and sampling efficiency.

Produces:
    results/phase3_mass_matrix.txt

Run from the repository root:  python scripts/phase3_mass_matrix.py
"""

from __future__ import annotations

import time
from pathlib import Path

import arviz as az

import metamorph  # noqa: F401  (enables float64 in JAX)
from metamorph.data import load_dataset
from metamorph.inference import run_nuts, to_inference_data

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Compare the two mass matrix choices and write a small report."""
    dataset = load_dataset(ROOT / "data")
    lines = [
        "NUTS mass matrix comparison (1 chain, 300 warmup + 300 draws, seed 0)",
        "",
        f"{'mass matrix':12s} {'wall (s)':>9s} {'leapfrog/iter':>14s} {'min ESS bulk':>13s} "
        f"{'ESS per 1000 grad':>18s} {'divergences':>12s}",
    ]
    for dense in (False, True):
        t0 = time.perf_counter()
        mcmc = run_nuts(dataset, num_warmup=300, num_samples=300, num_chains=1, seed=0, dense_mass=dense)
        wall = time.perf_counter() - t0
        idata = to_inference_data(mcmc)
        n_steps = idata.sample_stats["n_steps"].values
        ess = az.summary(idata, kind="diagnostics")["ess_bulk"].min()
        n_div = int(idata.sample_stats["diverging"].values.sum())
        lines.append(
            f"{'dense' if dense else 'diagonal':12s} {wall:9.0f} {n_steps.mean():14.1f} {ess:13.0f} "
            f"{ess / n_steps.sum() * 1000:18.2f} {n_div:12d}"
        )
    lines += ["", "Wall times include JIT compilation and depend on the machine."]
    text = "\n".join(lines)
    (ROOT / "results" / "phase3_mass_matrix.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
