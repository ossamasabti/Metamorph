"""Bayesian calibration with NumPyro (NUTS) and ArviZ diagnostics.

Parametrization. Every rate is positive and the rates span three orders of
magnitude (k_in_brain about 0.01/h, k_coat about 1/h). Each parameter is
therefore given a log-normal prior and sampled in standardized log space:

    z_j ~ Normal(0, 1),      theta_j = exp(mu_j + s_j * z_j),

so that theta_j ~ LogNormal(mu_j, s_j). The sampler only sees the z_j, which
are unconstrained and of order one. This is the non-centered form of a
log-normal prior: it removes the positivity constraint and puts all
parameters on the same scale, which helps NUTS adapt one step size.

Likelihood. Multiplicative log-normal noise (see ``metamorph.data``):

    log y_obs ~ Normal(log y_model(theta), sigma^2).
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from metamorph.data import Dataset, observation_matrix
from metamorph.model import PARAM_NAMES, simulate


class LogNormalPrior(NamedTuple):
    """Log-normal prior given by its median and log-scale standard deviation."""

    median: float
    sd_log: float


# Prior medians come from order-of-magnitude reasoning, not from the true
# values (which a modeller would not know). sd_log = 1 means one prior standard
# deviation is a factor e = 2.7, and the central 90 % prior interval spans a
# factor exp(2 * 1.645) = 27, i.e. more than one order of magnitude.
PRIORS: dict[str, LogNormalPrior] = {
    # Hard-corona maturation over minutes to hours: median 1/h (time scale 1 h).
    "k_coat": LogNormalPrior(1.0, 1.0),
    # Renal filtration of small particles: blood half-life of tens of minutes to
    # a few hours for ultrasmall particles. Median 0.5/h.
    "CL_renal": LogNormalPrior(0.5, 1.0),
    # Hepatobiliary elimination is slow (hours to days). Median 0.1/h.
    "CL_liver": LogNormalPrior(0.1, 1.0),
    # Uptake into well perfused organs: time scale of a few hours. Median 0.2/h.
    "k_in_lung": LogNormalPrior(0.2, 1.0),
    "k_in_liver": LogNormalPrior(0.2, 1.0),
    "k_in_spleen": LogNormalPrior(0.2, 1.0),
    "k_in_kidney": LogNormalPrior(0.2, 1.0),
    # Blood-brain barrier: brain typically holds well below 1 %ID, one to two
    # orders of magnitude less than other organs. Median 0.01/h.
    "k_in_brain": LogNormalPrior(0.01, 1.0),
    # Opsonized particles are expected to be taken up more by macrophages
    # (median 2), but a decrease is not excluded (5 % prior quantile about 0.6).
    "m_RES": LogNormalPrior(2.0, 0.7),
    # The corona is expected to hinder barrier crossing (median 0.5), but an
    # increase is not excluded (95 % prior quantile about 1.6).
    "m_brain": LogNormalPrior(0.5, 0.7),
    # Return from exchange tissues: time scale of hours. Median 0.3/h.
    "k_out_fast": LogNormalPrior(0.3, 1.0),
    # Return from sequestering tissues: time scale of a day. Median 0.05/h.
    "k_out_slow": LogNormalPrior(0.05, 1.0),
}

# Noise scale on the log scale: HalfNormal(0.5) puts 95 % of its mass below
# about 1, i.e. a coefficient of variation below about 100 %. Weakly informative.
SIGMA_PRIOR_SCALE: float = 0.5

# The data have about 20 % noise, so solver errors of 1e-6 (relative) are
# negligible. Looser tolerances than in phase 1 make sampling faster.
INFERENCE_RTOL: float = 1e-6
INFERENCE_ATOL: float = 1e-8
INFERENCE_MAX_STEPS: int = 20_000


class ModelData(NamedTuple):
    """Arrays the NumPyro model needs, precomputed from a ``Dataset``."""

    unique_times: jnp.ndarray
    time_index: jnp.ndarray
    obs_index: jnp.ndarray
    log_values: jnp.ndarray | None


def prepare(dataset: Dataset, with_values: bool = True) -> ModelData:
    """Convert a dataset to the arrays used by ``numpyro_model``.

    Args:
        dataset: Long-format dataset.
        with_values: If False, ``log_values`` is None (prior predictive mode).

    Returns:
        The model inputs.
    """
    return ModelData(
        unique_times=jnp.asarray(dataset.unique_times),
        time_index=jnp.asarray(dataset.time_index),
        obs_index=jnp.asarray(dataset.obs_index),
        log_values=jnp.log(jnp.asarray(dataset.value)) if with_values else None,
    )


def params_from_z(z: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Map standardized log-space coordinates to model parameters.

    Args:
        z: Array whose last axis has length 12, ordered as ``PARAM_NAMES``.

    Returns:
        Dict from parameter name to ``exp(mu + s * z)``.
    """
    return {
        name: jnp.exp(jnp.log(PRIORS[name].median) + PRIORS[name].sd_log * z[..., j])
        for j, name in enumerate(PARAM_NAMES)
    }


def predict_log_observations(params: Mapping[str, jnp.ndarray], data: ModelData) -> jnp.ndarray:
    """Log of the noiseless model prediction for each dataset row.

    Args:
        params: Model parameters (JAX scalars).
        data: Output of ``prepare``.

    Returns:
        Array with one entry per row. Non-finite if the solver failed.
    """
    ys = simulate(
        params,
        data.unique_times,
        rtol=INFERENCE_RTOL,
        atol=INFERENCE_ATOL,
        max_steps=INFERENCE_MAX_STEPS,
        throw=False,
    )
    obs = ys @ jnp.asarray(observation_matrix()).T
    pred = obs[data.time_index, data.obs_index]
    # Amounts are positive in exact arithmetic; the floor only protects the log
    # against round-off at very early times.
    return jnp.log(jnp.maximum(pred, 1e-12))


def numpyro_model(data: ModelData) -> None:
    """NumPyro model: log-normal priors, ODE, log-normal likelihood.

    Args:
        data: Output of ``prepare``. If ``data.log_values`` is None the model
            samples synthetic observations (prior predictive).
    """
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([len(PARAM_NAMES)]).to_event(1))
    params = params_from_z(z)
    for name, value in params.items():
        numpyro.deterministic(name, value)
    sigma = numpyro.sample("sigma", dist.HalfNormal(SIGMA_PRIOR_SCALE))
    log_pred = predict_log_observations(params, data)
    numpyro.sample("log_obs", dist.Normal(log_pred, sigma), obs=data.log_values)


def run_nuts(
    dataset: Dataset,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    target_accept_prob: float = 0.8,
    chain_method: str = "sequential",
    dense_mass: bool = True,
) -> MCMC:
    """Run NUTS on the dataset.

    Args:
        dataset: Observed data.
        num_warmup: Warmup (adaptation) iterations per chain.
        num_samples: Retained draws per chain.
        num_chains: Number of independent chains.
        seed: Seed of the JAX random key.
        target_accept_prob: Target acceptance probability for step-size adaptation.
        chain_method: "sequential" (default) or "parallel". With jax 0.10 and
            diffrax 0.7, "parallel" (pmap over chains) fails inside diffrax with a
            closure-conversion error, so chains are run one after the other.
            Results are identical in distribution; only wall time differs.
        dense_mass: If True, adapt a full (12 + 1)-dimensional mass matrix during
            warmup instead of a diagonal one. The posterior has strong linear
            correlations in log space (for example k_coat with CL_renal, k_in_liver
            with m_RES); a dense mass matrix removes them and lets NUTS take
            fewer, longer leapfrog steps.

    Returns:
        The fitted ``numpyro.infer.MCMC`` object.
    """
    kernel = NUTS(numpyro_model, target_accept_prob=target_accept_prob, dense_mass=dense_mass)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), prepare(dataset), extra_fields=("diverging", "num_steps"))
    return mcmc


def to_inference_data(mcmc: MCMC) -> az.InferenceData:
    """Convert to ArviZ, keeping the 12 named parameters and sigma as variables."""
    samples = mcmc.get_samples(group_by_chain=True)
    posterior = {name: np.asarray(samples[name]) for name in PARAM_NAMES}
    posterior["sigma"] = np.asarray(samples["sigma"])
    extra = mcmc.get_extra_fields(group_by_chain=True)
    sample_stats = {
        "diverging": np.asarray(extra["diverging"]),
        "n_steps": np.asarray(extra["num_steps"]),  # leapfrog steps per iteration
    }
    return az.from_dict(posterior=posterior, sample_stats=sample_stats)


def sample_prior(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Draw parameters and sigma from the prior.

    Args:
        n: Number of draws.
        seed: NumPy random seed.

    Returns:
        Dict from name (12 parameters and "sigma") to an array of shape ``(n,)``.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, len(PARAM_NAMES)))
    draws = {k: np.asarray(v) for k, v in params_from_z(jnp.asarray(z)).items()}
    draws["sigma"] = np.abs(rng.normal(0.0, SIGMA_PRIOR_SCALE, size=n))
    return draws


def predict_observables(draws: Mapping[str, np.ndarray], ts: np.ndarray) -> np.ndarray:
    """Noiseless observables for many parameter draws, on a time grid.

    Args:
        draws: Dict from parameter name to an array of shape ``(n,)``.
        ts: Times in hours, shape ``(m,)``, increasing, first entry > 0 allowed.

    Returns:
        Array of shape ``(n, m, 7)`` ordered as ``metamorph.data.OBSERVABLES``.
        Rows where the solver failed contain non-finite values.
    """
    H = jnp.asarray(observation_matrix())
    batch = {name: jnp.asarray(draws[name]) for name in PARAM_NAMES}

    def one(params: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        ys = simulate(
            params, ts, rtol=INFERENCE_RTOL, atol=INFERENCE_ATOL,
            max_steps=INFERENCE_MAX_STEPS, throw=False,
        )
        return ys @ H.T

    return np.asarray(jax.jit(jax.vmap(one))(batch))


def add_noise(pred: np.ndarray, sigma: np.ndarray, seed: int = 0) -> np.ndarray:
    """Apply multiplicative log-normal noise draw by draw.

    Args:
        pred: Array of shape ``(n, ...)``.
        sigma: Array of shape ``(n,)``, one noise scale per draw.
        seed: NumPy random seed.

    Returns:
        Noisy array, same shape as ``pred``.
    """
    rng = np.random.default_rng(seed)
    shape = (-1,) + (1,) * (pred.ndim - 1)
    return pred * np.exp(sigma.reshape(shape) * rng.standard_normal(pred.shape))


def parameter_table(
    idata: az.InferenceData, true_params: Mapping[str, float], prob: float = 0.9
) -> list[dict[str, float | str | bool]]:
    """True value vs posterior mean, median and central credible interval.

    Args:
        idata: Posterior.
        true_params: True values (may include "sigma").
        prob: Mass of the central (equal-tailed) interval.

    Returns:
        One dict per parameter with keys name, true, mean, median, lower, upper,
        covered, r_hat, ess_bulk.
    """
    lo_q, hi_q = (1 - prob) / 2, 1 - (1 - prob) / 2
    summary = az.summary(idata, kind="diagnostics")
    rows = []
    for name in (*PARAM_NAMES, "sigma"):
        x = idata.posterior[name].values.ravel()
        lower, upper = np.quantile(x, [lo_q, hi_q])
        true = float(true_params[name])
        rows.append(
            {
                "name": name,
                "true": true,
                "mean": float(x.mean()),
                "median": float(np.median(x)),
                "lower": float(lower),
                "upper": float(upper),
                "covered": bool(lower <= true <= upper),
                "r_hat": float(summary.loc[name, "r_hat"]),
                "ess_bulk": float(summary.loc[name, "ess_bulk"]),
            }
        )
    return rows
