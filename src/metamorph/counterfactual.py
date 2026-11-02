"""Counterfactual design: smallest change of particle properties that brings brain
accumulation below a threshold, at a point estimate and under posterior uncertainty.

Question: the reference particle (d = 5 nm, h = 1) accumulates more than
``THRESHOLD`` %ID in the brain at t_end = 24 h. What is the smallest change
(Delta d, Delta h), in the normalized distance of ``metamorph.properties``,
that brings it below the threshold?

* Point-estimate counterfactual: minimize u^2 + v^2 subject to
  brain(theta_bar; d, h) <= THRESHOLD, with theta_bar the posterior mean.
* Probability of success of any candidate: fraction of posterior draws theta_s
  for which brain(theta_s; d, h) <= THRESHOLD.
* Robust counterfactual: minimize u^2 + v^2 subject to
  P(success) >= ``ROBUST_LEVEL``, written as: the ``ROBUST_LEVEL``-quantile of
  brain(theta_s; d, h) over the draws is <= THRESHOLD.

Both problems are non-convex (the effect of d on the brain is not monotone),
so each is solved by (1) an exhaustive grid search over the normalized design
plane, then (2) local refinement with SLSQP started from the best grid point.

Model evaluation uses the exact solution y(t) = expm(A t) y0, which is valid
because the system is linear and is about 40 times cheaper than the adaptive
ODE solver; agreement with diffrax is tested in tests/test_counterfactual.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm
from scipy.optimize import minimize

from metamorph.model import (
    PARAM_NAMES,
    SLICE_T_BARE,
    SLICE_T_COAT,
    T_END,
    TISSUES,
    initial_state,
    system_matrix_jax,
)
from metamorph.properties import (
    D_BOUNDS,
    H_BOUNDS,
    apply_properties,
    from_normalized,
    to_normalized,
)

THRESHOLD: float = 0.2  # %ID in the brain at T_END
ROBUST_LEVEL: float = 0.9  # required probability of success

_BRAIN = TISSUES.index("brain")
_Y0 = jnp.asarray(initial_state())


def brain_amount(params: Mapping[str, jnp.ndarray], d: jnp.ndarray, h: jnp.ndarray, t: float = T_END) -> jnp.ndarray:
    """Brain content (%ID, bare + coated) at time ``t`` for a particle (d, h).

    Args:
        params: Reference-particle rates (one draw; JAX scalars).
        d: Diameter (nm).
        h: Coating propensity.
        t: Time (h).

    Returns:
        Scalar amount in %ID.
    """
    y = expm(system_matrix_jax(apply_properties(params, d, h)) * t) @ _Y0
    return y[SLICE_T_BARE.start + _BRAIN] + y[SLICE_T_COAT.start + _BRAIN]


# Vectorized over posterior draws (axis 0 of every parameter array).
_brain_over_draws = jax.jit(jax.vmap(brain_amount, in_axes=(0, None, None)))


def brain_over_draws(draws: Mapping[str, np.ndarray], d: float, h: float) -> np.ndarray:
    """Brain content at T_END for every posterior draw, for one design (d, h)."""
    batch = {n: jnp.asarray(draws[n]) for n in PARAM_NAMES}
    return np.asarray(_brain_over_draws(batch, jnp.asarray(d), jnp.asarray(h)))


def probability_of_success(draws: Mapping[str, np.ndarray], d: float, h: float, threshold: float = THRESHOLD) -> float:
    """Fraction of posterior draws for which brain(d, h) <= threshold."""
    return float(np.mean(brain_over_draws(draws, d, h) <= threshold))


def evaluate_grid(
    draws: Mapping[str, np.ndarray], u_grid: np.ndarray, v_grid: np.ndarray, batch: int = 64
) -> np.ndarray:
    """Brain content at T_END for every grid design and every draw.

    Args:
        draws: Posterior draws, arrays of shape ``(S,)``.
        u_grid: Normalized diameter changes, shape ``(nu,)``.
        v_grid: Normalized propensity changes (log2), shape ``(nv,)``.
        batch: Number of designs evaluated per compiled call.

    Returns:
        Array of shape ``(nv, nu, S)``.
    """
    U, V = np.meshgrid(u_grid, v_grid)
    d_all, h_all = from_normalized(U.ravel(), V.ravel())
    params = {n: jnp.asarray(draws[n]) for n in PARAM_NAMES}
    over_designs = jax.jit(jax.vmap(lambda d, h: jax.vmap(brain_amount, in_axes=(0, None, None))(params, d, h)))
    out = []
    for i in range(0, len(d_all), batch):
        out.append(np.asarray(over_designs(jnp.asarray(d_all[i:i + batch]), jnp.asarray(h_all[i:i + batch]))))
    return np.concatenate(out).reshape(len(v_grid), len(u_grid), -1)


@dataclass
class Counterfactual:
    """A proposed design change.

    Attributes:
        d: Diameter (nm).
        h: Coating propensity.
        u: Normalized diameter change.
        v: Normalized propensity change (log2).
        distance: sqrt(u^2 + v^2).
        success: Whether the optimizer reported success.
    """

    d: float
    h: float
    u: float
    v: float
    distance: float
    success: bool


def _normalized_bounds() -> list[tuple[float, float]]:
    u_lo, v_lo = to_normalized(D_BOUNDS[0], H_BOUNDS[0])
    u_hi, v_hi = to_normalized(D_BOUNDS[1], H_BOUNDS[1])
    return [(float(u_lo), float(u_hi)), (float(v_lo), float(v_hi))]


def _refine(constraint_fn, x0: np.ndarray) -> Counterfactual:
    """SLSQP refinement of min u^2 + v^2 s.t. constraint_fn(x) >= 0, from x0."""
    value_and_grad = jax.jit(jax.value_and_grad(constraint_fn))

    def con(x: np.ndarray) -> float:
        return float(value_and_grad(jnp.asarray(x))[0])

    def con_jac(x: np.ndarray) -> np.ndarray:
        return np.asarray(value_and_grad(jnp.asarray(x))[1])

    res = minimize(
        lambda x: float(x @ x),
        x0,
        jac=lambda x: 2 * x,
        method="SLSQP",
        bounds=_normalized_bounds(),
        constraints=[{"type": "ineq", "fun": con, "jac": con_jac}],
        options={"ftol": 1e-10, "maxiter": 200},
    )
    u, v = res.x
    d, h = from_normalized(u, v)
    feasible = con(res.x) >= -1e-6
    return Counterfactual(float(d), float(h), float(u), float(v), float(np.hypot(u, v)), bool(res.success and feasible))


def best_grid_point(feasible: np.ndarray, u_grid: np.ndarray, v_grid: np.ndarray) -> np.ndarray:
    """Feasible grid point closest to the origin; raises if none is feasible."""
    U, V = np.meshgrid(u_grid, v_grid)
    dist = np.where(feasible, np.hypot(U, V), np.inf)
    if not np.isfinite(dist.min()):
        raise ValueError("no feasible design on the grid")
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    return np.array([U[i, j], V[i, j]])


def point_counterfactual(
    params_point: Mapping[str, float], x0: np.ndarray, threshold: float = THRESHOLD
) -> Counterfactual:
    """Smallest change such that brain(theta_point; d, h) <= threshold.

    The constraint is written in log space, log(threshold) - log(brain) >= 0,
    which is better scaled than the raw amounts.

    Args:
        params_point: Point estimate of the reference-particle rates.
        x0: Starting point (u, v), e.g. from ``best_grid_point``.
        threshold: Brain threshold (%ID).
    """
    p = {n: jnp.asarray(params_point[n]) for n in PARAM_NAMES}

    def constraint(x: jnp.ndarray) -> jnp.ndarray:
        d, h = from_normalized(x[0], x[1])
        return jnp.log(threshold) - jnp.log(brain_amount(p, d, h))

    return _refine(constraint, x0)


def robust_counterfactual(
    draws: Mapping[str, np.ndarray],
    x0: np.ndarray,
    level: float = ROBUST_LEVEL,
    threshold: float = THRESHOLD,
) -> Counterfactual:
    """Smallest change such that the ``level``-quantile of brain over draws <= threshold.

    The empirical quantile is piecewise smooth in (u, v), so SLSQP is started
    close to the solution (from the grid search).

    Args:
        draws: Posterior draws used for optimization.
        x0: Starting point (u, v).
        level: Required probability of success.
        threshold: Brain threshold (%ID).
    """
    batch = {n: jnp.asarray(draws[n]) for n in PARAM_NAMES}
    over_draws = jax.vmap(brain_amount, in_axes=(0, None, None))

    def constraint(x: jnp.ndarray) -> jnp.ndarray:
        d, h = from_normalized(x[0], x[1])
        return jnp.log(threshold) - jnp.log(jnp.quantile(over_draws(batch, d, h), level))

    return _refine(constraint, x0)
