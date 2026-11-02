"""Phase 4: identifiability and global sensitivity.

Steps:
    1. Posterior correlations in log space (heatmap) and pair plots of key parameters.
    2. Contraction: how much the data narrowed each parameter relative to the prior.
    3. Which data would fix poorly determined parameters: Laplace (Fisher
       information) approximation of the posterior for alternative designs,
       validated against NUTS on the actual design.
    4. Global sensitivity (Sobol indices over the prior) of the brain and liver
       amounts at 24 h, compared with local one-at-a-time elasticities.

Requires results/phase3_posterior.nc (run scripts/phase3_calibration.py first).

Produces:
    figures/phase4_correlation.png
    figures/phase4_pairs.png
    figures/phase4_designs.png
    figures/phase4_sobol.png
    results/phase4_summary.txt

Run from the repository root:  python scripts/phase4_identifiability.py
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

import metamorph  # noqa: F401  (enables float64 in JAX)
from metamorph.inference import PRIORS
from metamorph.model import PARAM_NAMES
from metamorph.plotting import COLOR_BARE, COLOR_COAT, COLOR_MUTED, COLOR_TOTAL, apply_style
from metamorph.sensitivity import (
    DESIGNS,
    final_amounts,
    laplace_log_correlation,
    laplace_log_sd,
    local_elasticities,
    sobol_indices,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

PAIR_PARAMS = ("k_coat", "CL_renal", "k_in_liver", "k_in_spleen", "m_RES", "k_in_brain", "m_brain", "k_out_slow", "CL_liver")
DESIGN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")  # validated categorical slots 1-5
Z90 = 1.645


def plot_correlation(corr: np.ndarray, labels: list[str]) -> plt.Figure:
    """Annotated heatmap of a correlation matrix (diverging, gray-free white midpoint)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and abs(corr[i, j]) >= 0.3:
                ax.text(j, i, f"{corr[i, j]:+.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(corr[i, j]) > 0.65 else COLOR_TOTAL)
    fig.colorbar(im, ax=ax, shrink=0.8, label="posterior correlation of log parameters")
    ax.set_title("Posterior correlations in log space (|r| >= 0.3 annotated)", loc="left")
    fig.tight_layout()
    return fig


def plot_pairs(log_samples: dict[str, np.ndarray], truth: dict[str, float]) -> plt.Figure:
    """Lower-triangle pair plot of log10 parameters, true values marked."""
    apply_style()
    names = PAIR_PARAMS
    n = len(names)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(log_samples[names[0]]), size=1500, replace=False)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    for i, yi in enumerate(names):
        for j, xj in enumerate(names):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            x = log_samples[xj] / np.log(10)
            if i == j:
                ax.hist(x, bins=40, color=COLOR_BARE, alpha=0.8)
                ax.axvline(np.log10(truth[xj]), color=COLOR_TOTAL, lw=1.2)
                ax.set_yticks([])
            else:
                y = log_samples[yi] / np.log(10)
                ax.scatter(x[pick], y[pick], s=2, color=COLOR_BARE, alpha=0.35, lw=0)
                ax.plot(np.log10(truth[xj]), np.log10(truth[yi]), "o", ms=5,
                        color=COLOR_TOTAL, mec="white", mew=0.8)
                r = np.corrcoef(log_samples[xj], log_samples[yi])[0, 1]
                ax.text(0.04, 0.9, f"r={r:+.2f}", transform=ax.transAxes, fontsize=7, color=COLOR_MUTED)
            ax.grid(False)
            ax.tick_params(labelsize=6)
            if i == n - 1:
                ax.set_xlabel(f"log10 {xj}", fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(f"log10 {yi}", fontsize=7)
            elif j > 0:
                ax.set_yticklabels([])
    fig.suptitle("Posterior pair plot (log10 scale); black = true value", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_designs(widths: dict[str, np.ndarray], nuts_width: np.ndarray) -> plt.Figure:
    """Grouped bars of the predicted 90 % CI width factor per parameter and design."""
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(PARAM_NAMES))
    k = len(widths)
    bw = 0.8 / k
    for m, (name, w) in enumerate(widths.items()):
        ax.bar(x - 0.4 + bw * (m + 0.5), w, width=bw * 0.92, color=DESIGN_COLORS[m], label=name)
    ax.scatter(x - 0.4 + bw * 0.5, nuts_width, marker="_", s=120, color=COLOR_TOTAL, zorder=4,
               label="NUTS, current design (check)")
    ax.set_yscale("log")
    ax.set_xticks(x, PARAM_NAMES, rotation=40, ha="right")
    ax.set_ylabel("90 % CI width factor (upper / lower)")
    ax.axhline(1, color=COLOR_MUTED, lw=0.8)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.5)  # room for the legend
    ax.set_title("Which data would fix it? Laplace prediction of posterior uncertainty per design", loc="left")
    fig.tight_layout()
    return fig


def plot_sobol(res, elasticities: np.ndarray) -> plt.Figure:
    """Sobol S1 and ST (with bootstrap 90 % intervals) for brain and liver, plus local elasticities."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), gridspec_kw={"width_ratios": [2, 1]})
    x = np.arange(len(PARAM_NAMES))
    for r, name in enumerate(["Brain at 24 h", "Liver at 24 h"]):
        ax = axes[r, 0]
        for k, (S, ci, col, lab) in enumerate(
            [(res.S1[r], res.S1_ci[r], COLOR_BARE, "first order S1"), (res.ST[r], res.ST_ci[r], COLOR_COAT, "total ST")]
        ):
            pos = x - 0.2 + 0.4 * k
            ax.bar(pos, S, width=0.38, color=col, label=lab)
            ax.errorbar(pos, S, yerr=[S - ci[:, 0], ci[:, 1] - S], fmt="none", ecolor=COLOR_TOTAL, lw=0.8, capsize=1.5)
        ax.set_xticks(x, PARAM_NAMES if r == 1 else [""] * len(x), rotation=40, ha="right")
        ax.set_ylabel("Sobol index")
        ax.set_title(f"{name}: variance-based (prior ranges)", loc="left")
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        if r == 0:
            ax.legend(fontsize=8)
        ax2 = axes[r, 1]
        order = np.argsort(np.abs(elasticities[r]))
        ax2.barh(np.arange(len(x)), elasticities[r][order], color=COLOR_MUTED)
        ax2.set_yticks(np.arange(len(x)), [PARAM_NAMES[i] for i in order], fontsize=7)
        ax2.axvline(0, color=COLOR_TOTAL, lw=0.8)
        ax2.set_title("Local elasticity at posterior mean", loc="left", fontsize=9)
        if r == 1:
            ax2.set_xlabel("d log(amount) / d log(theta)")
    fig.tight_layout()
    return fig


def main() -> None:
    """Run all phase 4 analyses."""
    idata = az.from_netcdf(RES_DIR / "phase3_posterior.nc")
    import json

    meta = json.loads((ROOT / "data" / "true_parameters.json").read_text(encoding="utf-8"))
    truth = meta["parameters"]
    log_s = {n: np.log(idata.posterior[n].values.ravel()) for n in PARAM_NAMES}
    sigma_post = idata.posterior["sigma"].values.ravel()
    prior_sd = np.array([PRIORS[n].sd_log for n in PARAM_NAMES])
    lines = ["Phase 4 summary", ""]

    # ------------------------------------------------------------ 1. correlations
    L = np.stack([log_s[n] for n in PARAM_NAMES], axis=1)
    corr = np.corrcoef(L, rowvar=False)
    fig = plot_correlation(corr, list(PARAM_NAMES))
    fig.savefig(FIG_DIR / "phase4_correlation.png")
    plt.close(fig)
    fig = plot_pairs(log_s, truth)
    fig.savefig(FIG_DIR / "phase4_pairs.png", dpi=150)
    plt.close(fig)
    pairs = [(abs(corr[i, j]), PARAM_NAMES[i], PARAM_NAMES[j], corr[i, j])
             for i in range(12) for j in range(i + 1, 12)]
    pairs.sort(reverse=True)
    lines.append("1. Strongest posterior correlations (log parameters):")
    lines += [f"   {a:12s} - {b:12s} r = {r:+.3f}" for _, a, b, r in pairs[:8]]
    i_s, i_c = PARAM_NAMES.index("k_out_slow"), PARAM_NAMES.index("CL_liver")
    lines += [f"   (k_out_slow - CL_liver: r = {corr[i_s, i_c]:+.3f})", ""]

    # ------------------------------------------------------------ 2. contraction
    post_sd = L.std(axis=0)
    contraction = 1 - post_sd**2 / prior_sd**2
    lines.append("2. Posterior contraction 1 - var_post / var_prior (log scale); 1 = fully learned:")
    for j in np.argsort(contraction):
        lines.append(f"   {PARAM_NAMES[j]:12s} prior sd {prior_sd[j]:.2f}  posterior sd {post_sd[j]:.3f}  "
                     f"contraction {contraction[j]:.3f}")
    lines.append("")

    # ------------------------------------------------------------ 3. designs
    mu = np.array([np.log(PRIORS[n].median) for n in PARAM_NAMES])
    z_hat = (L.mean(axis=0) - mu) / prior_sd
    sigma_hat = float(sigma_post.mean())
    sds = {name: laplace_log_sd(z_hat, sigma_hat, arms, prior_sd) for name, arms in DESIGNS.items()}
    widths = {name: np.exp(2 * Z90 * sd) for name, sd in sds.items()}
    nuts_width = np.exp(2 * Z90 * post_sd)
    fig = plot_designs(widths, nuts_width)
    fig.savefig(FIG_DIR / "phase4_designs.png")
    plt.close(fig)
    header = f"   {'parameter':12s} {'NUTS':>7s} " + " ".join(f"{d:>26s}" for d in DESIGNS)
    lines += [
        "3. Laplace (Fisher) approximation at the posterior mean, sigma = "
        f"{sigma_hat:.3f}. Values: 90 % CI width factor exp(2 * 1.645 * sd).",
        "   NUTS column = actual posterior of the current design (validation of the approximation).",
        header,
    ]
    for j, n in enumerate(PARAM_NAMES):
        lines.append(f"   {n:12s} {nuts_width[j]:7.2f} " + " ".join(f"{widths[d][j]:26.2f}" for d in DESIGNS))
    rel_err = np.abs(sds["current"] - post_sd) / post_sd
    lines.append(f"   Laplace vs NUTS sd on current design: median relative difference "
                 f"{np.median(rel_err):.2f}, max {rel_err.max():.2f} ({PARAM_NAMES[int(np.argmax(rel_err))]})")
    il, im_ = PARAM_NAMES.index("k_in_liver"), PARAM_NAMES.index("m_RES")
    ib, imb = PARAM_NAMES.index("k_in_brain"), PARAM_NAMES.index("m_brain")
    lines.append("   Laplace correlations per design (k_in_liver, m_RES) / (k_in_brain, m_brain) / (k_out_slow, CL_liver):")
    for name, arms in DESIGNS.items():
        c = laplace_log_correlation(z_hat, sigma_hat, arms)
        lines.append(f"     {name:26s} {c[il, im_]:+.2f} / {c[ib, imb]:+.2f} / {c[i_s, i_c]:+.2f}")
    lines.append("")

    # ------------------------------------------------------------ 4. Sobol
    res = sobol_indices(final_amounts, n_base=4096, seed=0)
    elast = local_elasticities(z_hat)
    fig = plot_sobol(res, elast)
    fig.savefig(FIG_DIR / "phase4_sobol.png")
    plt.close(fig)
    lines += [
        f"4. Sobol indices of log10 amount at 24 h, inputs = prior (independent log-normal), "
        f"N = {res.n_base}, {res.n_evaluations} model runs, bootstrap 90 % intervals.",
    ]
    for r, name in enumerate(["brain", "liver"]):
        lines.append(f"   {name}: sum S1 = {res.S1[r].sum():.3f} (1 - sum S1 = share of variance from interactions)")
        for j in np.argsort(-res.ST[r]):
            lines.append(
                f"     {PARAM_NAMES[j]:12s} S1 {res.S1[r, j]:+.3f} [{res.S1_ci[r, j, 0]:+.3f}, {res.S1_ci[r, j, 1]:+.3f}]"
                f"   ST {res.ST[r, j]:.3f} [{res.ST_ci[r, j, 0]:.3f}, {res.ST_ci[r, j, 1]:.3f}]"
                f"   local elasticity {elast[r, j]:+.3f}"
            )
    text = "\n".join(lines)
    (RES_DIR / "phase4_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
