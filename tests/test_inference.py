"""Tests of the inference module (fast checks; no full NUTS run)."""

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_density

import metamorph  # noqa: F401
from metamorph.data import generate_dataset, noiseless_observations
from metamorph.inference import (
    PRIORS,
    add_noise,
    numpyro_model,
    params_from_z,
    predict_log_observations,
    predict_observables,
    prepare,
    sample_prior,
)
from metamorph.model import PARAM_NAMES, TRUE_PARAMS


def _z_of(params: dict) -> jnp.ndarray:
    return jnp.array(
        [(np.log(params[n]) - np.log(PRIORS[n].median)) / PRIORS[n].sd_log for n in PARAM_NAMES]
    )


def test_every_parameter_has_a_prior() -> None:
    assert set(PRIORS) == set(PARAM_NAMES)


def test_params_from_z_inverts_standardization() -> None:
    params = params_from_z(_z_of(TRUE_PARAMS))
    for name in PARAM_NAMES:
        np.testing.assert_allclose(float(params[name]), TRUE_PARAMS[name], rtol=1e-12)


def test_jax_prediction_matches_scipy_reference() -> None:
    ds = generate_dataset(TRUE_PARAMS, seed=0)
    log_pred = predict_log_observations(params_from_z(_z_of(TRUE_PARAMS)), prepare(ds))
    np.testing.assert_allclose(np.exp(log_pred), noiseless_observations(TRUE_PARAMS, ds), rtol=1e-5)


def test_log_density_is_higher_at_truth_and_has_finite_gradient() -> None:
    ds = generate_dataset(TRUE_PARAMS, seed=0)
    data = prepare(ds)

    def logp(z: jnp.ndarray) -> jnp.ndarray:
        return log_density(numpyro_model, (data,), {}, {"z": z, "sigma": 0.2})[0]

    z_true = _z_of(TRUE_PARAMS)
    z_far = z_true + 1.0
    assert logp(z_true) > logp(z_far)
    grad = jax.grad(logp)(z_true)
    assert np.all(np.isfinite(np.asarray(grad)))


def test_prior_draws_and_noise_shapes() -> None:
    draws = sample_prior(8, seed=0)
    assert set(draws) == set(PARAM_NAMES) | {"sigma"}
    assert all(np.all(v > 0) for v in draws.values())
    pred = predict_observables(draws, np.array([0.5, 4.0, 24.0]))
    assert pred.shape == (8, 3, 7)
    noisy = add_noise(pred, draws["sigma"], seed=0)
    assert noisy.shape == pred.shape
