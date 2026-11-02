"""Phase 3: Bayesian calibration with NumPyro NUTS.

Steps:
    1. Prior predictive check (before looking at the fit).
    2. NUTS, 4 chains, diagnostics with ArviZ (R-hat, ESS, divergences).
    3. Posterior predictive check with 90 % bands.
    4. Table: true value vs posterior mean and 90 % credible interval.

Produces:
    figures/phase3_prior_predictive.png
    figures/phase3_trace.png
    figures/phase3_posterior_predictive.png
    figures/phase3_marginals.png
    results/phase3_posterior.nc           posterior draws (ArviZ / netCDF), reused later
    results/phase3_parameter_table.md
    results/phase3_summary.txt

Run from the repository root:  python scripts/phase3_calibration.py [--quick]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

import metamorph  # noqa: F401  (enables float64 in JAX)
from metamorph.data import OBSERVABLES, load_dataset, load_true_parameters
from metamorph.inference import (
    PRIORS,
    SIGMA_PRIOR_SCALE,
    add_noise,
    parameter_table,
    predict_observables,
    run_nuts,
    sample_prior,
    to_inference_data,
)
from metamorph.model import PARAM_NAMES, T_END
from metamorph.plotting import COLOR_BARE, COLOR_COAT, COLOR_TOTAL, apply_style

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

TS_PLOT = np.geomspace(1 / 60, T_END, 120)  # log-spaced so the first hour is resolved


def band_figure(
    title: str,
    dataset,
    model_curves: np.ndarray,
    noisy_curves: np.ndarray,
    color: str,
    band_label: str,
) -> plt.Figure:
    """Seven panels: 90 % band of noisy predictions, 90 % band of the model, data.

    Args:
        title: Figure title.
        dataset: Observed data.
        model_curves: Noiseless predictions, shape ``(n, len(TS_PLOT), 7)``.
        noisy_curves: Predictions with observation noise, same shape.
        color: Band color.
        band_label: Legend label prefix ("prior" or "posterior").
    """
    apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4), sharex=True)
    axes = axes.ravel()
    lo_n, hi_n = np.nanquantile(noisy_curves, [0.05, 0.95], axis=0)
    lo_m, med_m, hi_m = np.nanquantile(model_curves, [0.05, 0.5, 0.95], axis=0)
    for k, name in enumerate(OBSERVABLES):
        ax = axes[k]
        ax.fill_between(TS_PLOT, lo_n[:, k], hi_n[:, k], color=color, alpha=0.18, lw=0,
                        label=f"90 % {band_label} predictive (with noise)")
        ax.fill_between(TS_PLOT, lo_m[:, k], hi_m[:, k], color=color, alpha=0.45, lw=0,
                        label=f"90 % {band_label} credible band (model)")
        ax.plot(TS_PLOT, med_m[:, k], color=color, lw=1.5, label="median")
        rows = dataset.observable == name
        ax.scatter(dataset.time[rows], dataset.value[rows], s=18, color=COLOR_TOTAL,
                   edgecolor="white", linewidth=0.7, zorder=3, label="data")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(name.capitalize(), loc="left")
    axes[3].tick_params(labelbottom=True)
    for ax in axes[3:7]:
        ax.set_xlabel("time (h, log scale)")
    for ax in axes[[0, 4]]:
        ax.set_ylabel("amount (%ID)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].axis("off")
    axes[-1].legend(handles, labels, loc="center left", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_marginals(post: dict, prior: dict, truth: dict) -> plt.Figure:
    """Prior vs posterior histograms on a log axis, with the true value.

    Bins are equally spaced in log10 and each bar shows the probability of its
    bin (not a density in x), so shapes are comparable on the log axis.
    """
    apply_style()
    fig, axes = plt.subplots(3, 5, figsize=(12, 7))
    axes = axes.ravel()
    for j, name in enumerate((*PARAM_NAMES, "sigma")):
        ax = axes[j]
        x, p = post[name], prior[name]
        lo = min(x.min(), np.quantile(p, 0.005))
        hi = max(x.max(), np.quantile(p, 0.995))
        bins = np.geomspace(lo, hi, 60)
        ax.hist(p, bins=bins, weights=np.full(len(p), 1 / len(p)), color=COLOR_COAT, alpha=0.35, label="prior")
        ax.hist(x, bins=bins, weights=np.full(len(x), 1 / len(x)), color=COLOR_BARE, alpha=0.8, label="posterior")
        ax.axvline(truth[name], color=COLOR_TOTAL, lw=1.5, label="true")
        ax.set_xscale("log")
        ax.set_yticks([])
        ax.set_title(name, loc="left")
    axes[0].legend(fontsize=7)
    for ax in axes[13:]:
        ax.axis("off")
    fig.suptitle("Prior vs posterior marginals (log axis, probability per log-bin), true value in black",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def coverage(dataset, ts_grid_curves: np.ndarray) -> float:
    """Fraction of data points inside the 90 % band of noisy predictions at their times."""
    unique_t = np.unique(dataset.time)
    lo, hi = np.nanquantile(ts_grid_curves, [0.05, 0.95], axis=0)  # (len(unique_t), 7)
    t_idx = np.searchsorted(unique_t, dataset.time)
    inside = (dataset.value >= lo[t_idx, dataset.obs_index]) & (dataset.value <= hi[t_idx, dataset.obs_index])
    return float(inside.mean())


def main(quick: bool) -> None:
    """Run all phase 3 steps."""
    for d in (FIG_DIR, RES_DIR):
        d.mkdir(exist_ok=True)
    dataset = load_dataset(DATA_DIR)
    true_params, true_sigma = load_true_parameters(DATA_DIR)
    unique_t = np.unique(dataset.time)
    lines = ["Phase 3 summary", ""]

    # ---------------------------------------------------------------- 1. prior predictive
    n_prior = 1000
    prior = sample_prior(n_prior, seed=1)
    prior_model = predict_observables(prior, TS_PLOT)
    prior_noisy = add_noise(prior_model, prior["sigma"], seed=2)
    failed = ~np.all(np.isfinite(prior_model), axis=(1, 2))
    prior_at_data = add_noise(predict_observables(prior, unique_t), prior["sigma"], seed=3)
    fig = band_figure("Prior predictive check (1000 prior draws)", dataset,
                      prior_model, prior_noisy, COLOR_COAT, "prior")
    fig.savefig(FIG_DIR / "phase3_prior_predictive.png")
    plt.close(fig)
    brain24 = prior_model[:, -1, OBSERVABLES.index("brain")]
    urine24 = prior_model[:, -1, OBSERVABLES.index("urine")]
    lines += [
        "1. Prior predictive check",
        f"   draws: {n_prior}, solver failures: {failed.sum()}",
        f"   data points inside the 90 % prior predictive band: {coverage(dataset, prior_at_data):.3f}",
        f"   prior 90 % interval of brain at 24 h: [{np.nanquantile(brain24, 0.05):.3g}, "
        f"{np.nanquantile(brain24, 0.95):.3g}] %ID",
        f"   prior 90 % interval of urine at 24 h: [{np.nanquantile(urine24, 0.05):.3g}, "
        f"{np.nanquantile(urine24, 0.95):.3g}] %ID",
        "",
    ]

    # ---------------------------------------------------------------- 2. NUTS
    num_warmup, num_samples = (200, 200) if quick else (1000, 1000)
    t0 = time.perf_counter()
    mcmc = run_nuts(dataset, num_warmup=num_warmup, num_samples=num_samples, num_chains=4, seed=0)
    wall = time.perf_counter() - t0
    idata = to_inference_data(mcmc)
    idata.to_netcdf(RES_DIR / "phase3_posterior.nc")

    n_div = int(idata.sample_stats["diverging"].values.sum())
    summ = az.summary(idata, kind="diagnostics")
    lines += [
        "2. NUTS",
        f"   4 chains x ({num_warmup} warmup + {num_samples} draws), dense mass matrix, target_accept 0.8, "
        f"wall time {wall:.0f} s",
        f"   divergent transitions after warmup: {n_div} / {4 * num_samples}",
        f"   mean leapfrog steps per iteration: {idata.sample_stats['n_steps'].values.mean():.1f}",
        f"   max R-hat: {summ['r_hat'].max():.4f}",
        f"   min ESS bulk: {summ['ess_bulk'].min():.0f}, min ESS tail: {summ['ess_tail'].min():.0f}",
        "",
        "   per-parameter diagnostics:",
    ]
    lines += ["   " + l for l in summ.to_string().splitlines()]
    lines.append("")

    apply_style()
    axes = az.plot_trace(idata, var_names=[*PARAM_NAMES, "sigma"], compact=False,
                         figsize=(10, 22), backend_kwargs={"layout": "constrained"})
    plt.gcf().savefig(FIG_DIR / "phase3_trace.png", dpi=110)
    plt.close("all")

    # ---------------------------------------------------------------- 3. posterior predictive
    post = {name: idata.posterior[name].values.ravel() for name in (*PARAM_NAMES, "sigma")}
    rng = np.random.default_rng(4)
    pick = rng.choice(len(post["sigma"]), size=min(800, len(post["sigma"])), replace=False)
    draws = {k: v[pick] for k, v in post.items()}
    post_model = predict_observables(draws, TS_PLOT)
    post_noisy = add_noise(post_model, draws["sigma"], seed=5)
    post_at_data = add_noise(predict_observables(draws, unique_t), draws["sigma"], seed=6)
    fig = band_figure("Posterior predictive check (800 posterior draws)", dataset,
                      post_model, post_noisy, COLOR_BARE, "posterior")
    fig.savefig(FIG_DIR / "phase3_posterior_predictive.png")
    plt.close(fig)
    lines += [
        "3. Posterior predictive check",
        f"   data points inside the 90 % posterior predictive band: {coverage(dataset, post_at_data):.3f}",
        "",
    ]

    # ---------------------------------------------------------------- 4. table
    truth = dict(true_params, sigma=true_sigma)
    rows = parameter_table(idata, truth)
    md = [
        "| Parameter | True | Prior median | Posterior mean | Posterior median | 90 % CI | True in CI | R-hat | ESS bulk |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        prior_med = PRIORS[r["name"]].median if r["name"] in PRIORS else float("nan")
        prior_txt = f"{prior_med:.3g}" if r["name"] in PRIORS else f"HalfN({SIGMA_PRIOR_SCALE})"
        md.append(
            f"| {r['name']} | {r['true']:.3g} | {prior_txt} | {r['mean']:.3g} | {r['median']:.3g} | "
            f"[{r['lower']:.3g}, {r['upper']:.3g}] | {'yes' if r['covered'] else 'no'} | "
            f"{r['r_hat']:.3f} | {r['ess_bulk']:.0f} |"
        )
    (RES_DIR / "phase3_parameter_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    n_cov = sum(r["covered"] for r in rows)
    width = {r["name"]: r["upper"] / r["lower"] for r in rows}
    lines += ["4. Parameter table (also in results/phase3_parameter_table.md)", *md, "",
              f"   true value inside 90 % CI: {n_cov} / {len(rows)}",
              "   width of 90 % CI as a ratio upper / lower (1 = perfectly determined):"]
    lines += [f"     {k:12s} {v:8.2f}" for k, v in sorted(width.items(), key=lambda kv: kv[1])]

    fig = plot_marginals(post, prior, truth)
    fig.savefig(FIG_DIR / "phase3_marginals.png")
    plt.close(fig)

    text = "\n".join(lines)
    (RES_DIR / "phase3_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="short chains, for testing only")
    main(parser.parse_args().quick)
