"""Identifiability and sensitivity tools (phase 4).

Three tools:

1. ``laplace_log_sd``: a local (Laplace / Fisher information) approximation of
   the posterior standard deviation of each log-parameter, for any experimental
   design. It answers "which additional data would determine a parameter"
   without re-running NUTS for every candidate design. It is validated against
   the NUTS posterior of the actual design.

2. ``sobol_indices``: variance-based global sensitivity (Sobol first-order and
   total indices) of a scalar output, with parameters varied jointly over the
   prior. Estimators: Saltelli (2010) for first order, Jansen (1999) for total.

3. ``local_elasticities``: one-at-a-time local sensitivities
   d log(output) / d log(theta) at a single parameter point, for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm, qmc

from metamorph.data import T_DENSE, T_SPARSE
from metamorph.inference import (
    INFERENCE_ATOL,
    INFERENCE_MAX_STEPS,
    INFERENCE_RTOL,
    params_from_z,
)
from metamorph.model import (
    IDX_A_HEPATIC,
    IDX_A_RENAL,
    IDX_B_BARE,
    IDX_B_COAT,
    N_STATES,
    PARAM_NAMES,
    SLICE_T_BARE,
    SLICE_T_COAT,
    TISSUES,
    simulate,
)

# --------------------------------------------------------------------------
# Experimental designs
# --------------------------------------------------------------------------


def _row(indices: list[int]) -> np.ndarray:
    h = np.zeros(N_STATES)
    h[indices] = 1.0
    return h


OBSERVABLE_ROWS: dict[str, np.ndarray] = {
    "blood": _row([IDX_B_BARE, IDX_B_COAT]),
    "blood_bare": _row([IDX_B_BARE]),
    "blood_coated": _row([IDX_B_COAT]),
    "urine": _row([IDX_A_RENAL]),
    "feces": _row([IDX_A_HEPATIC]),
    **{t: _row([SLICE_T_BARE.start + i, SLICE_T_COAT.start + i]) for i, t in enumerate(TISSUES)},
}


@dataclass(frozen=True)
class Arm:
    """One experimental arm: an injection and what is measured, and when.

    Attributes:
        coated_fraction: Fraction of the dose injected already coated.
        schedule: Observable name (key of ``OBSERVABLE_ROWS``) to sampling times.
        n_replicates: Independent replicates per (observable, time).
    """

    coated_fraction: float
    schedule: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    n_replicates: int = 3


BASE_SCHEDULE: dict[str, tuple[float, ...]] = {
    "blood": T_DENSE,
    "urine": T_DENSE,
    **{t: T_SPARSE for t in TISSUES},
}

DESIGNS: dict[str, list[Arm]] = {
    # The actual phase 2 experiment.
    "current": [Arm(0.0, BASE_SCHEDULE)],
    # Add fecal (biliary) excretion, cumulative, at the dense times.
    "+ feces": [Arm(0.0, {**BASE_SCHEDULE, "feces": T_DENSE})],
    # Measure bare and coated particles in blood separately (e.g. after
    # separating corona-covered particles), instead of the total.
    "+ blood bare/coated split": [
        Arm(0.0, {k: v for k, v in BASE_SCHEDULE.items() if k != "blood"}
            | {"blood_bare": T_DENSE, "blood_coated": T_DENSE})
    ],
    # Harvest tissues also at the early times (5, 15, 30 min). Costs three more
    # groups of animals. Included to test whether the gain from "+ feces" comes
    # from early liver information rather than from the feces themselves.
    "+ early tissue times": [Arm(0.0, {**BASE_SCHEDULE, **{t: T_DENSE for t in TISSUES}})],
    # A second group of animals receives particles pre-incubated in plasma
    # (fully coated), with the same blood and tissue sampling as the first
    # group. Urine is not sampled in that arm: coated particles are not
    # filtered, so urine would contain nothing (and carry no information).
    "+ pre-coated arm": [
        Arm(0.0, BASE_SCHEDULE),
        Arm(1.0, {k: v for k, v in BASE_SCHEDULE.items() if k != "urine"}),
    ],
}


def _arm_log_predictions(z: jnp.ndarray, arm: Arm) -> jnp.ndarray:
    """Log of the noiseless prediction for every (observable, time) of an arm."""
    times = np.unique(np.concatenate([np.asarray(t) for t in arm.schedule.values()]))
    ys = simulate(
        params_from_z(z), times, rtol=INFERENCE_RTOL, atol=INFERENCE_ATOL,
        max_steps=INFERENCE_MAX_STEPS, coated_fraction=arm.coated_fraction,
    )
    preds = []
    for name, obs_times in arm.schedule.items():
        idx = np.searchsorted(times, np.asarray(obs_times))
        preds.append(ys[idx] @ jnp.asarray(OBSERVABLE_ROWS[name]))
    return jnp.log(jnp.concatenate(preds))


def fisher_information(z: np.ndarray, sigma: float, arms: list[Arm]) -> np.ndarray:
    """Fisher information of the log-normal likelihood with respect to z.

    For log y ~ Normal(log f(z), sigma^2), the Fisher information is
    J^T J / sigma^2 per replicate, where J = d log f / d z.

    Args:
        z: Standardized log-parameters, shape ``(12,)``.
        sigma: Log-scale noise standard deviation.
        arms: Experimental design.

    Returns:
        Array of shape ``(12, 12)``.
    """
    info = np.zeros((len(PARAM_NAMES), len(PARAM_NAMES)))
    for arm in arms:
        J = np.asarray(jax.jacrev(_arm_log_predictions)(jnp.asarray(z), arm))
        info += arm.n_replicates * J.T @ J / sigma**2
    return info


def laplace_log_sd(z: np.ndarray, sigma: float, arms: list[Arm], prior_sd_log: np.ndarray) -> np.ndarray:
    """Approximate posterior standard deviation of log(theta_j) for a design.

    The posterior precision in z is approximated by Fisher information plus the
    prior precision (identity, since z ~ Normal(0, I)). The standard deviation
    of log theta_j is s_j times the standard deviation of z_j.

    Args:
        z: Point at which the approximation is made (e.g. posterior mean of z).
        sigma: Noise scale.
        arms: Experimental design.
        prior_sd_log: Prior log-scale standard deviations s_j, shape ``(12,)``.

    Returns:
        Array of shape ``(12,)``.
    """
    precision = fisher_information(z, sigma, arms) + np.eye(len(PARAM_NAMES))
    cov_z = np.linalg.inv(precision)
    return np.sqrt(np.diag(cov_z)) * prior_sd_log


def laplace_log_correlation(z: np.ndarray, sigma: float, arms: list[Arm]) -> np.ndarray:
    """Approximate posterior correlation matrix of the log-parameters."""
    cov = np.linalg.inv(fisher_information(z, sigma, arms) + np.eye(len(PARAM_NAMES)))
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


# --------------------------------------------------------------------------
# Global sensitivity (Sobol)
# --------------------------------------------------------------------------


def final_amounts(z: jnp.ndarray, t_end: float = 24.0) -> jnp.ndarray:
    """log10 of the brain and liver amounts (%ID) at ``t_end`` for one z.

    Returns:
        Array ``[log10 brain(t_end), log10 liver(t_end)]``.
    """
    ys = simulate(
        params_from_z(z), jnp.array([t_end]), rtol=INFERENCE_RTOL, atol=INFERENCE_ATOL,
        max_steps=INFERENCE_MAX_STEPS, throw=False,
    )[0]
    brain = ys[SLICE_T_BARE.start + TISSUES.index("brain")] + ys[SLICE_T_COAT.start + TISSUES.index("brain")]
    liver = ys[SLICE_T_BARE.start + TISSUES.index("liver")] + ys[SLICE_T_COAT.start + TISSUES.index("liver")]
    return jnp.log10(jnp.stack([brain, liver]))


def evaluate_batched(fn: Callable[[jnp.ndarray], jnp.ndarray], z: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Apply a scalar-or-vector function of z to many rows, in vmapped batches."""
    f = jax.jit(jax.vmap(fn))
    out = [np.asarray(f(jnp.asarray(z[i:i + batch]))) for i in range(0, len(z), batch)]
    return np.concatenate(out)


@dataclass
class SobolResult:
    """First-order (S1) and total (ST) indices with bootstrap 90 % intervals.

    Arrays have shape ``(n_outputs, 12)``; intervals ``(n_outputs, 12, 2)``.
    """

    S1: np.ndarray
    ST: np.ndarray
    S1_ci: np.ndarray
    ST_ci: np.ndarray
    n_base: int
    n_evaluations: int


def sobol_indices(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    n_base: int = 4096,
    seed: int = 0,
    n_bootstrap: int = 500,
) -> SobolResult:
    """Sobol indices of a vector output with independent standard normal inputs z.

    Inputs follow the prior (z ~ Normal(0, I), i.e. independent log-normal rates).
    Sampling uses a scrambled Sobol sequence of dimension 2d split into matrices
    A and B; AB_i is A with column i taken from B. Cost: n_base * (d + 2)
    model evaluations.

    Estimators (V = variance of the output over A and B):
        S1_i = mean(f(B) * (f(AB_i) - f(A))) / V          (Saltelli et al. 2010)
        ST_i = mean((f(A) - f(AB_i))^2) / (2 V)             (Jansen 1999)

    Args:
        fn: Function from z (shape ``(12,)``) to outputs (shape ``(k,)``).
        n_base: Number of base samples N (a power of 2 for the Sobol sequence).
        seed: Seed of the scrambled Sobol sequence and the bootstrap.
        n_bootstrap: Bootstrap resamples for the intervals.

    Returns:
        A ``SobolResult``.
    """
    d = len(PARAM_NAMES)
    u = qmc.Sobol(d=2 * d, scramble=True, seed=seed).random(n_base)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    A, B = norm.ppf(u[:, :d]), norm.ppf(u[:, d:])
    AB = np.stack([np.where(np.arange(d) == i, B, A) for i in range(d)])  # (d, N, d)

    fA = evaluate_batched(fn, A)
    fB = evaluate_batched(fn, B)
    fAB = evaluate_batched(fn, AB.reshape(-1, d)).reshape(d, n_base, -1)  # (d, N, k)

    def estimate(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b, ab = fA[rows], fB[rows], fAB[:, rows]
        V = np.var(np.concatenate([a, b]), axis=0)
        s1 = np.mean(b[None] * (ab - a[None]), axis=1) / V
        st = 0.5 * np.mean((a[None] - ab) ** 2, axis=1) / V
        return s1.T, st.T  # (k, d)

    S1, ST = estimate(np.arange(n_base))
    rng = np.random.default_rng(seed)
    boot = [estimate(rng.integers(0, n_base, n_base)) for _ in range(n_bootstrap)]
    s1_b = np.stack([b[0] for b in boot])
    st_b = np.stack([b[1] for b in boot])
    S1_ci = np.stack(np.quantile(s1_b, [0.05, 0.95], axis=0), axis=-1)
    ST_ci = np.stack(np.quantile(st_b, [0.05, 0.95], axis=0), axis=-1)
    return SobolResult(S1, ST, S1_ci, ST_ci, n_base, n_base * (d + 2))


def local_elasticities(z: np.ndarray) -> np.ndarray:
    """One-at-a-time local sensitivities d log(output) / d log(theta_j) at z.

    Because log theta_j = mu_j + s_j z_j, the derivative with respect to
    log theta_j is the derivative with respect to z_j divided by s_j.

    Args:
        z: Point of evaluation, shape ``(12,)``.

    Returns:
        Array of shape ``(2, 12)``: rows brain, liver (natural-log elasticities).
    """
    from metamorph.inference import PRIORS

    s = np.array([PRIORS[n].sd_log for n in PARAM_NAMES])
    J = np.asarray(jax.jacrev(final_amounts)(jnp.asarray(z)))  # d log10 / dz
    return J * np.log(10.0) / s[None, :]
