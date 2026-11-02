"""Phase 5: counterfactual particle design under posterior uncertainty.

Question: the reference particle (d = 5 nm, h = 1) accumulates more than
0.2 %ID in the brain at 24 h. What is the smallest change in (d, h) that
brings it below 0.2 %ID?

Steps:
    1. Map of brain(24 h) over a 41 x 41 grid of designs, for 1000 posterior
       draws (optimization set) and at the posterior mean.
    2. Point-estimate counterfactual (at the posterior mean), plus single-lever
       variants (change only d, or only h) and a "safety margin" variant.
    3. Probability of success of each candidate, on 3000 held-out draws.
    4. Robust counterfactual: smallest change with P(success) >= 0.9, optimized
       on the 1000 draws and checked on the 3000 held-out draws.

Requires results/phase3_posterior.nc.

Produces:
    figures/phase5_success_map.png
    figures/phase5_brain_distributions.png
    results/phase5_summary.txt

Run from the repository root:  python scripts/phase5_counterfactual.py
"""

from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import metamorph  # noqa: F401  (enables float64 in JAX)
from metamorph.counterfactual import (
    ROBUST_LEVEL,
    THRESHOLD,
    best_grid_point,
    brain_over_draws,
    evaluate_grid,
    point_counterfactual,
    robust_counterfactual,
)
from metamorph.model import PARAM_NAMES
from metamorph.properties import D_REF, H_REF, design_distance, from_normalized, to_normalized
from metamorph.plotting import COLOR_BARE, COLOR_COAT, COLOR_MUTED, COLOR_TOTAL, apply_style

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

N_OPT = 1000
U_GRID = np.linspace(-1.0, 5.0, 41)  # d from 3 to 15 nm
V_GRID = np.linspace(-3.0, 3.0, 41)  # h from 1/8 to 8
SAFETY_FACTOR = 0.8  # "add a margin" heuristic: aim for 0.8 * threshold at the point estimate


def single_lever(point: dict, lever: str) -> tuple[float, float] | None:
    """Smallest change of one property (the other fixed) with brain(point) = threshold.

    Returns None if the threshold cannot be reached within the design bounds.
    """
    one = {n: np.array([point[n]]) for n in PARAM_NAMES}

    def gap_u(u: float) -> float:
        d, h = from_normalized(u, 0.0)
        return float(np.log(brain_over_draws(one, d, h)[0]) - np.log(THRESHOLD))

    def gap_v(v: float) -> float:
        d, h = from_normalized(0.0, v)
        return float(np.log(brain_over_draws(one, d, h)[0]) - np.log(THRESHOLD))

    if lever == "d":
        if gap_u(5.0) > 0:
            return None
        u = brentq(gap_u, 0.0, 5.0)
        return tuple(map(float, from_normalized(u, 0.0)))
    if gap_v(3.0) > 0:
        return None
    v = brentq(gap_v, 0.0, 3.0)
    return tuple(map(float, from_normalized(0.0, v)))


def main() -> None:
    """Run all phase 5 steps."""
    idata = az.from_netcdf(RES_DIR / "phase3_posterior.nc")
    truth = json.loads((ROOT / "data" / "true_parameters.json").read_text(encoding="utf-8"))["parameters"]
    all_draws = {n: idata.posterior[n].values.ravel() for n in PARAM_NAMES}
    n_total = len(all_draws[PARAM_NAMES[0]])
    perm = np.random.default_rng(0).permutation(n_total)
    opt_draws = {n: v[perm[:N_OPT]] for n, v in all_draws.items()}
    val_draws = {n: v[perm[N_OPT:]] for n, v in all_draws.items()}
    point = {n: float(v.mean()) for n, v in all_draws.items()}  # posterior mean of each rate
    point_1 = {n: np.array([point[n]]) for n in PARAM_NAMES}
    truth_1 = {n: np.array([truth[n]]) for n in PARAM_NAMES}

    # ------------------------------------------------------------ 1. grid
    grid = evaluate_grid(opt_draws, U_GRID, V_GRID)  # (nv, nu, N_OPT)
    grid_point = evaluate_grid(point_1, U_GRID, V_GRID)[..., 0]
    p_map = (grid <= THRESHOLD).mean(axis=-1)
    q_map = np.quantile(grid, ROBUST_LEVEL, axis=-1)

    # ------------------------------------------------------------ 2. and 4. counterfactuals
    cf_point = point_counterfactual(point, best_grid_point(grid_point <= THRESHOLD, U_GRID, V_GRID))
    cf_margin = point_counterfactual(
        point, best_grid_point(grid_point <= SAFETY_FACTOR * THRESHOLD, U_GRID, V_GRID),
        threshold=SAFETY_FACTOR * THRESHOLD,
    )
    cf_robust = robust_counterfactual(opt_draws, best_grid_point(q_map <= THRESHOLD, U_GRID, V_GRID))
    d_only = single_lever(point, "d")
    h_only = single_lever(point, "h")

    candidates = {
        "original particle": (D_REF, H_REF),
        "point estimate (optimal)": (cf_point.d, cf_point.h),
        "point estimate, d only": d_only,
        "point estimate, h only": h_only,
        f"point estimate, margin x{SAFETY_FACTOR}": (cf_margin.d, cf_margin.h),
        "robust, P >= 0.9": (cf_robust.d, cf_robust.h),
        "maximal h (h = 8), d unchanged": (D_REF, 8.0),
    }
    candidates = {k: v for k, v in candidates.items() if v is not None}

    # ------------------------------------------------------------ 3. evaluation on held-out draws
    lines = [
        "Phase 5 summary",
        "",
        f"Threshold: brain(24 h) <= {THRESHOLD} %ID. Reference particle d = {D_REF} nm, h = {H_REF}.",
        f"Normalized distance: sqrt(u^2 + v^2), u = (d - {D_REF}) / 2 nm, v = log2(h / {H_REF}).",
        f"Posterior draws: {N_OPT} for optimization, {n_total - N_OPT} held out for evaluation.",
        f"Optimizer status: point {cf_point.success}, margin {cf_margin.success}, robust {cf_robust.success}",
        "",
        f"{'candidate':34s} {'d (nm)':>7s} {'h':>6s} {'dist':>6s} {'brain@mean':>11s} "
        f"{'P(success)':>11s} {'90% interval of brain':>24s} {'truth':>7s}",
    ]
    results = {}
    for name, (d, h) in candidates.items():
        b_val = brain_over_draws(val_draws, d, h)
        b_point = float(brain_over_draws(point_1, d, h)[0])
        b_true = float(brain_over_draws(truth_1, d, h)[0])
        p = float(np.mean(b_val <= THRESHOLD))
        lo, hi = np.quantile(b_val, [0.05, 0.95])
        dist = float(design_distance(d, h))
        results[name] = dict(d=d, h=h, dist=dist, p=p, b_val=b_val, b_point=b_point, b_true=b_true)
        lines.append(
            f"{name:34s} {d:7.2f} {h:6.2f} {dist:6.2f} {b_point:11.3f} {p:11.3f} "
            f"{f'[{lo:.3f}, {hi:.3f}]':>24s} {b_true:7.3f}"
        )
    p_opt_robust = float(np.mean(brain_over_draws(opt_draws, cf_robust.d, cf_robust.h) <= THRESHOLD))
    lines += [
        "",
        "brain@mean = brain amount at the posterior-mean rates; truth = brain amount at the true",
        "rates (unknown in practice, shown because the data are synthetic).",
        f"Robust counterfactual: P(success) = {p_opt_robust:.3f} on the optimization draws, "
        f"{results['robust, P >= 0.9']['p']:.3f} on the held-out draws.",
        "",
        "Changing h alone reaches the threshold at the posterior mean: "
        + ("yes" if h_only is not None else "no, not even at the upper bound h = 8"),
        "Changing d alone reaches the threshold at the posterior mean: "
        + ("yes" if d_only is not None else "no"),
        "",
        "Grid search (41 x 41, optimization draws) best feasible points, for comparison with SLSQP:",
        f"  point estimate: u, v = {best_grid_point(grid_point <= THRESHOLD, U_GRID, V_GRID)}",
        f"  robust:         u, v = {best_grid_point(q_map <= THRESHOLD, U_GRID, V_GRID)}",
    ]

    # ------------------------------------------------------------ figures
    apply_style()
    D, Hh = np.meshgrid(*from_normalized(U_GRID, V_GRID))
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    mesh = ax.pcolormesh(D, Hh, p_map, cmap="Blues", vmin=0, vmax=1, shading="nearest", rasterized=True)
    cs = ax.contour(D, Hh, p_map, levels=[0.5, ROBUST_LEVEL], colors=[COLOR_MUTED, COLOR_TOTAL], linewidths=1.2)
    ax.clabel(cs, fmt={0.5: "P = 0.5", ROBUST_LEVEL: f"P = {ROBUST_LEVEL}"}, fontsize=7)
    ax.contour(D, Hh, grid_point, levels=[THRESHOLD], colors=[COLOR_COAT], linewidths=1.6, linestyles="--")
    ax.plot([], [], color=COLOR_COAT, ls="--", label="threshold reached at posterior mean")
    markers = {
        "original particle": ("o", COLOR_TOTAL),
        "point estimate (optimal)": ("s", COLOR_COAT),
        "maximal h (h = 8), d unchanged": ("^", COLOR_COAT),
        "robust, P >= 0.9": ("*", "#1baf7a"),
    }
    for name, (mk, col) in markers.items():
        r = results[name]
        ax.plot(r["d"], r["h"], mk, color=col, ms=11 if mk == "*" else 8, mec="white", mew=1.0,
                label=f"{name} (P = {r['p']:.2f})", zorder=5)
    theta = np.linspace(0, np.pi / 2, 50)
    for name in ("point estimate (optimal)", "robust, P >= 0.9"):
        rr = results[name]["dist"]
        dd, hh = from_normalized(rr * np.cos(theta), rr * np.sin(theta))
        ax.plot(dd, hh, color=COLOR_MUTED, lw=0.6, ls=":")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("hydrodynamic diameter d (nm)")
    ax.set_ylabel("coating propensity h (log2 scale)")
    ax.set_title(f"P(brain at 24 h <= {THRESHOLD} %ID) over the posterior", loc="left")
    fig.colorbar(mesh, ax=ax, label="probability of success")
    ax.legend(loc="upper right", fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase5_success_map.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    bins = np.geomspace(0.05, 0.5, 60)
    for name, col in [("original particle", COLOR_MUTED), ("maximal h (h = 8), d unchanged", COLOR_COAT),
                      ("point estimate (optimal)", COLOR_BARE), ("robust, P >= 0.9", "#1baf7a")]:
        r = results[name]
        ax.hist(r["b_val"], bins=bins, histtype="step", lw=2, color=col, label=f"{name} (P = {r['p']:.2f})")
    ax.axvline(THRESHOLD, color=COLOR_TOTAL, lw=1.2)
    ax.text(THRESHOLD, ax.get_ylim()[1] * 0.95, " threshold", va="top", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("brain amount at 24 h (%ID), held-out posterior draws")
    ax.set_ylabel("count")
    ax.legend(fontsize=7)
    ax.set_title("Predicted brain accumulation for each candidate design", loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase5_brain_distributions.png")
    plt.close(fig)

    text = "\n".join(lines)
    (RES_DIR / "phase5_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
