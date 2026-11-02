"""Tests of the property maps and the counterfactual tools."""

import jax.numpy as jnp
import numpy as np

import metamorph  # noqa: F401
from metamorph.counterfactual import brain_over_draws, point_counterfactual, probability_of_success
from metamorph.model import (
    SLICE_T_BARE,
    SLICE_T_COAT,
    TISSUES,
    TRUE_PARAMS,
    simulate,
    system_matrix,
    system_matrix_jax,
)
from metamorph.properties import (
    D_REF,
    H_REF,
    apply_properties,
    brain_factor,
    coat_factor,
    design_distance,
    from_normalized,
    renal_factor,
    to_normalized,
)

BRAIN = TISSUES.index("brain")


def test_maps_equal_one_at_reference_and_have_stated_monotonicity() -> None:
    np.testing.assert_allclose(float(renal_factor(D_REF)), 1.0)
    np.testing.assert_allclose(float(brain_factor(D_REF)), 1.0)
    np.testing.assert_allclose(float(coat_factor(H_REF)), 1.0)
    d = jnp.linspace(3.0, 15.0, 50)
    assert np.all(np.diff(np.asarray(renal_factor(d))) < 0)
    assert np.all(np.diff(np.asarray(brain_factor(d))) < 0)
    np.testing.assert_allclose(float(coat_factor(2 * H_REF)), 2.0)


def test_normalization_roundtrip() -> None:
    d, h = from_normalized(1.5, -2.0)
    u, v = to_normalized(d, h)
    np.testing.assert_allclose([float(u), float(v)], [1.5, -2.0])
    np.testing.assert_allclose(float(design_distance(D_REF + 6.0, H_REF * 16.0)), 5.0)


def test_jax_system_matrix_matches_numpy() -> None:
    params = apply_properties(TRUE_PARAMS, 7.3, 2.5)
    A_np = system_matrix({k: float(v) for k, v in params.items()})
    A_jx = np.asarray(system_matrix_jax({k: jnp.asarray(v) for k, v in params.items()}))
    np.testing.assert_allclose(A_jx, A_np, atol=1e-15)


def test_expm_brain_matches_diffrax() -> None:
    for d, h in [(5.0, 1.0), (8.0, 0.5), (4.0, 6.0)]:
        ys = np.asarray(simulate(apply_properties(TRUE_PARAMS, d, h), np.array([24.0])))
        diffrax_brain = ys[0, SLICE_T_BARE.start + BRAIN] + ys[0, SLICE_T_COAT.start + BRAIN]
        draws = {k: np.array([v]) for k, v in TRUE_PARAMS.items()}
        np.testing.assert_allclose(brain_over_draws(draws, d, h)[0], diffrax_brain, rtol=1e-6)


def test_probability_of_success_extremes() -> None:
    draws = {k: np.full(5, v) for k, v in TRUE_PARAMS.items()}
    assert probability_of_success(draws, D_REF, H_REF, threshold=10.0) == 1.0
    assert probability_of_success(draws, D_REF, H_REF, threshold=0.01) == 0.0


def test_point_counterfactual_lands_on_threshold() -> None:
    cf = point_counterfactual(TRUE_PARAMS, np.array([1.0, 1.0]), threshold=0.2)
    assert cf.success
    draws = {k: np.array([v]) for k, v in TRUE_PARAMS.items()}
    np.testing.assert_allclose(brain_over_draws(draws, cf.d, cf.h)[0], 0.2, rtol=1e-4)
