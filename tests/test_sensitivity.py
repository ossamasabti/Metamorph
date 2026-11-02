"""Tests of the sensitivity tools on cases with known answers."""

import jax.numpy as jnp
import numpy as np

import metamorph  # noqa: F401
from metamorph.inference import PRIORS
from metamorph.model import PARAM_NAMES
from metamorph.sensitivity import DESIGNS, fisher_information, laplace_log_sd, sobol_indices


def _linear(z: jnp.ndarray) -> jnp.ndarray:
    """y = z0 + 2 z1, with z ~ N(0, I): S1 = ST = (1/5, 4/5, 0, ...)."""
    return jnp.stack([z[0] + 2.0 * z[1]])


def _product(z: jnp.ndarray) -> jnp.ndarray:
    """y = z0 * z1: pure interaction, S1 = 0 and ST = 1 for both inputs."""
    return jnp.stack([z[0] * z[1]])


def test_sobol_additive_model() -> None:
    res = sobol_indices(_linear, n_base=2048, seed=0, n_bootstrap=50)
    expected = np.zeros(len(PARAM_NAMES))
    expected[:2] = [0.2, 0.8]
    np.testing.assert_allclose(res.S1[0], expected, atol=0.03)
    np.testing.assert_allclose(res.ST[0], expected, atol=0.03)


def test_sobol_detects_pure_interaction() -> None:
    # Heavy-tailed output (product of normals): use a larger sample.
    res = sobol_indices(_product, n_base=8192, seed=0, n_bootstrap=50)
    np.testing.assert_allclose(res.S1[0, :2], 0.0, atol=0.05)
    np.testing.assert_allclose(res.ST[0, :2], 1.0, atol=0.08)


def test_more_data_never_increases_laplace_uncertainty() -> None:
    """Adding an experimental arm adds positive semi-definite information."""
    s = np.array([PRIORS[n].sd_log for n in PARAM_NAMES])
    z = np.zeros(len(PARAM_NAMES))
    base = laplace_log_sd(z, 0.2, DESIGNS["current"], s)
    more = laplace_log_sd(z, 0.2, DESIGNS["+ pre-coated arm"], s)
    assert np.all(more <= base + 1e-12)
    info = fisher_information(z, 0.2, DESIGNS["current"])
    np.testing.assert_allclose(info, info.T, atol=1e-8 * np.abs(info).max())
    assert np.linalg.eigvalsh(info).min() > -1e-8 * np.abs(info).max()
