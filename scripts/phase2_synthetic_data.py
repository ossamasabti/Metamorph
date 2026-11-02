"""Phase 2: generate the synthetic biodistribution dataset.

Produces:
    data/synthetic_data.csv              noisy measurements (long format)
    data/true_parameters.json            true parameters, noise scale, seed, design
    figures/phase2_dataset.png           data vs noiseless truth, log scale
    figures/phase2_early_sensitivity.png local sensitivities of blood and urine
                                         to k_coat and CL_renal over time
    results/phase2_summary.txt           numbers quoted in the docs

Run from the repository root:  python scripts/phase2_synthetic_data.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import metamorph  # noqa: F401  (enables float64 in JAX)
from metamorph.data import (
    N_REPLICATES,
    OBSERVABLES,
    OBSERVATION_TIMES,
    SEED,
    TRUE_SIGMA,
    generate_dataset,
    load_dataset,
    noiseless_observations,
    observation_matrix,
    save_dataset,
)
from metamorph.model import T_END, TRUE_PARAMS, simulate, simulate_exact
from metamorph.plotting import COLOR_BARE, COLOR_COAT, COLOR_MUTED, COLOR_TOTAL, apply_style

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"


def plot_dataset(dataset, ts: np.ndarray, truth: np.ndarray) -> plt.Figure:
    """Data points and noiseless truth per observable, on a log amount axis."""
    apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True)
    axes = axes.ravel()
    for k, name in enumerate(OBSERVABLES):
        ax = axes[k]
        ax.plot(ts[1:], truth[1:, k], color=COLOR_TOTAL, lw=1.5, label="truth (noiseless)")
        rows = dataset.observable == name
        ax.scatter(
            dataset.time[rows], dataset.value[rows], s=22, color=COLOR_BARE,
            edgecolor="white", linewidth=0.8, zorder=3, label="measurement",
        )
        ax.set_yscale("log")
        ax.set_title(f"{name.capitalize()} ({len(OBSERVATION_TIMES[name])} times)", loc="left")
    axes[0].legend(loc="upper right")
    axes[3].tick_params(labelbottom=True)
    axes[-1].axis("off")
    axes[-1].text(
        0.0, 0.95,
        f"{len(dataset)} measurements\n"
        f"{N_REPLICATES} replicates per time\n"
        f"log-normal noise, sigma = {TRUE_SIGMA}\n"
        f"seed = {SEED}\n\n"
        "Observed: totals (bare + coated)\n"
        "Not observed: A_hepatic",
        va="top", color=COLOR_MUTED, transform=axes[-1].transAxes,
    )
    for ax in axes[3:7]:
        ax.set_xlabel("time (h)")
    for ax in axes[[0, 4]]:
        ax.set_ylabel("amount (%ID)")
    fig.suptitle("Synthetic dataset: measurements vs true model", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def log_sensitivities(ts: np.ndarray) -> np.ndarray:
    """Local sensitivities d log(observable) / d log(theta) for theta in (k_coat, CL_renal).

    Computed by reverse-mode automatic differentiation through the diffrax solver
    (the default diffrax adjoint supports reverse mode only, which is also what
    NUTS uses for gradients of the log density).

    Returns:
        Array of shape ``(len(ts), 2 observables [blood, urine], 2 parameters)``.
    """
    H = jnp.asarray(observation_matrix()[[0, -1]])  # blood, urine

    def log_obs(log_theta: jnp.ndarray) -> jnp.ndarray:
        params = dict(TRUE_PARAMS, k_coat=jnp.exp(log_theta[0]), CL_renal=jnp.exp(log_theta[1]))
        ys = simulate(params, ts)
        return jnp.log(ys @ H.T)

    log_theta0 = jnp.log(jnp.array([TRUE_PARAMS["k_coat"], TRUE_PARAMS["CL_renal"]]))
    return np.asarray(jax.jacrev(log_obs)(log_theta0))


def plot_sensitivities(ts: np.ndarray, sens: np.ndarray) -> plt.Figure:
    """Two panels (blood, urine), one line per parameter, dense sampling times marked."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for k, (ax, name) in enumerate(zip(axes, ["Blood (total)", "Urine (cumulative)"])):
        ax.axhline(0, color=COLOR_MUTED, lw=0.8)
        ax.plot(ts, sens[:, k, 0], color=COLOR_COAT, label="k_coat")
        ax.plot(ts, sens[:, k, 1], color=COLOR_BARE, label="CL_renal")
        for t in OBSERVATION_TIMES["blood"]:
            ax.axvline(t, color=COLOR_MUTED, lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("time (h, log scale); dotted = sampling times")
        ax.set_title(name, loc="left")
    axes[0].set_ylabel("d log(amount) / d log(theta)")
    axes[0].legend()
    fig.suptitle("Local log-sensitivities at the true parameters", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def main() -> None:
    """Generate, save, reload, plot and summarize the synthetic dataset."""
    for d in (DATA_DIR, FIG_DIR, RES_DIR):
        d.mkdir(exist_ok=True)

    dataset = generate_dataset(TRUE_PARAMS, sigma=TRUE_SIGMA, n_replicates=N_REPLICATES, seed=SEED)
    save_dataset(dataset, DATA_DIR, TRUE_PARAMS, TRUE_SIGMA, SEED)
    dataset = load_dataset(DATA_DIR)  # use exactly what is on disk

    ts = np.linspace(0.0, T_END, 961)
    truth = simulate_exact(TRUE_PARAMS, ts) @ observation_matrix().T
    fig = plot_dataset(dataset, ts, truth)
    fig.savefig(FIG_DIR / "phase2_dataset.png")
    plt.close(fig)

    ts_sens = np.geomspace(1 / 60, T_END, 300)
    sens = log_sensitivities(ts_sens)
    fig = plot_sensitivities(ts_sens, sens)
    fig.savefig(FIG_DIR / "phase2_early_sensitivity.png")
    plt.close(fig)

    # Summary numbers.
    residuals = np.log(dataset.value) - np.log(noiseless_observations(TRUE_PARAMS, dataset))
    lines = [
        "Phase 2 summary",
        "",
        f"Rows: {len(dataset)}  (seed {SEED}, sigma {TRUE_SIGMA}, {N_REPLICATES} replicates)",
    ]
    for name in OBSERVABLES:
        times = ", ".join(f"{t:g}" for t in np.round(OBSERVATION_TIMES[name], 4))
        lines.append(f"  {name:7s}: {np.sum(dataset.observable == name):3d} rows at t = {times} h")
    lines += [
        "",
        f"Empirical std of log residuals: {residuals.std(ddof=1):.4f} (true sigma {TRUE_SIGMA})",
        f"Empirical mean of log residuals: {residuals.mean():+.4f}",
        "",
        "Local log-sensitivities d log(y) / d log(theta) at sampling times:",
        "  t (h)   blood:k_coat  blood:CL_renal  urine:k_coat  urine:CL_renal",
    ]
    for t in OBSERVATION_TIMES["blood"]:
        j = np.argmin(np.abs(ts_sens - t))
        s = sens[j]
        lines.append(
            f"  {t:6.3f}  {s[0, 0]:+12.3f}  {s[0, 1]:+14.3f}  {s[1, 0]:+12.3f}  {s[1, 1]:+14.3f}"
        )
    j_end = -1
    lines += [
        "",
        "Correlation over sampling times between the k_coat and CL_renal sensitivity",
        "columns (values near -1 or +1 mean the two effects look alike):",
    ]
    idx = [int(np.argmin(np.abs(ts_sens - t))) for t in OBSERVATION_TIMES["blood"]]
    for label, rows in [("all 7 times", idx), ("only 1, 4, 8, 24 h", idx[3:])]:
        cols = sens[rows].transpose(1, 0, 2).reshape(-1, 2)  # stack blood and urine rows
        r = np.corrcoef(cols[:, 0], cols[:, 1])[0, 1]
        lines.append(f"  {label:20s}: r = {r:+.3f}")
    lines.append("")
    lines.append(f"Urine at t_end: d log / d log k_coat = {sens[j_end, 1, 0]:+.3f}, "
                 f"d log / d log CL_renal = {sens[j_end, 1, 1]:+.3f}")
    text = "\n".join(lines)
    (RES_DIR / "phase2_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
