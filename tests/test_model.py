"""Tests of the forward model: parameter mapping, solver agreement, mass balance."""

import numpy as np
import pytest

import metamorph  # noqa: F401  (enables float64 in JAX)
from metamorph.model import (
    DEFAULT_DOSE,
    IDX_A_HEPATIC,
    IDX_A_RENAL,
    N_STATES,
    T_END,
    TRUE_PARAMS,
    expand_rates,
    rhs,
    rhs_numpy,
    simulate,
    simulate_exact,
    simulate_scipy,
    system_matrix,
)

TS = np.array([0.0, 5 / 60, 0.25, 0.5, 1.0, 4.0, 8.0, 24.0])


@pytest.fixture(scope="module")
def solutions() -> dict[str, np.ndarray]:
    """Solve once with the three methods at the true parameters."""
    return {
        "jax": np.asarray(simulate(TRUE_PARAMS, TS)),
        "scipy": simulate_scipy(TRUE_PARAMS, TS),
        "exact": simulate_exact(TRUE_PARAMS, TS),
    }


def test_expand_rates_matches_stated_grouping() -> None:
    k_in_bare, k_in_coat, k_out = (np.asarray(a) for a in expand_rates(TRUE_PARAMS))
    for jax_arr, np_arr in zip((k_in_bare, k_in_coat, k_out), expand_rates(TRUE_PARAMS, xp=np)):
        np.testing.assert_allclose(jax_arr, np_arr)
    np.testing.assert_allclose(k_in_bare, [0.3, 0.4, 0.1, 0.2, 0.01])
    np.testing.assert_allclose(k_in_coat, [0.3, 1.2, 0.3, 0.2, 0.003])
    np.testing.assert_allclose(k_out, [0.5, 0.03, 0.03, 0.5, 0.03])


def test_rhs_implementations_agree_on_random_states() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.uniform(0.0, 50.0, size=N_STATES)
        np.testing.assert_allclose(
            np.asarray(rhs(0.0, y, TRUE_PARAMS)), rhs_numpy(0.0, y, TRUE_PARAMS), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(system_matrix(TRUE_PARAMS) @ y, rhs_numpy(0.0, y, TRUE_PARAMS), atol=1e-12)


def test_system_matrix_columns_sum_to_zero() -> None:
    """Mass conservation in matrix form: every outflow is someone's inflow."""
    np.testing.assert_allclose(system_matrix(TRUE_PARAMS).sum(axis=0), 0.0, atol=1e-14)


def test_jax_matches_scipy(solutions: dict[str, np.ndarray]) -> None:
    np.testing.assert_allclose(solutions["jax"], solutions["scipy"], rtol=1e-6, atol=1e-8)


def test_both_solvers_match_exact_solution(solutions: dict[str, np.ndarray]) -> None:
    np.testing.assert_allclose(solutions["jax"], solutions["exact"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(solutions["scipy"], solutions["exact"], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("method", ["jax", "scipy", "exact"])
def test_total_mass_is_conserved(solutions: dict[str, np.ndarray], method: str) -> None:
    np.testing.assert_allclose(solutions[method].sum(axis=1), DEFAULT_DOSE, rtol=1e-8)


def test_mass_balance_with_random_parameters() -> None:
    """Conservation must hold for any positive parameters, not only the true ones."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        params = {k: v * float(np.exp(rng.normal(0.0, 0.7))) for k, v in TRUE_PARAMS.items()}
        ys = np.asarray(simulate(params, TS))
        np.testing.assert_allclose(ys.sum(axis=1), DEFAULT_DOSE, rtol=1e-8)
        assert np.all(ys > -1e-8), "amounts must stay non-negative"


def test_cleared_amounts_are_nondecreasing(solutions: dict[str, np.ndarray]) -> None:
    ys = solutions["jax"]
    assert np.all(np.diff(ys[:, IDX_A_RENAL]) >= 0)
    assert np.all(np.diff(ys[:, IDX_A_HEPATIC]) >= 0)


def test_no_coated_particles_without_coating() -> None:
    params = dict(TRUE_PARAMS, k_coat=0.0)
    ys = np.asarray(simulate(params, np.array([1.0, T_END])))
    coated = ys[:, [1] + list(range(7, 12))]
    np.testing.assert_allclose(coated, 0.0, atol=1e-12)
