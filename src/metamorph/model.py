"""Forward model: a blood + five-tissue compartmental model with a bare/coated state.

Units: amounts in percent of injected dose (%ID), time in hours, rates in 1/h.

State vector layout (14 entries):

    index 0        B_b        bare particles in blood
    index 1        B_c        coated particles in blood
    index 2..6     T_{i,b}    bare particles in tissue i (lung, liver, spleen, kidney, brain)
    index 7..11    T_{i,c}    coated particles in tissue i (same order)
    index 12       A_renal    cumulative amount cleared by the kidney (urine)
    index 13       A_hepatic  cumulative amount cleared by the liver (bile / feces)

The two cumulative states make the system closed: the sum of all 14 entries
equals the dose at every time, which is the basis of the mass balance test.

Equations (s in {b, c}, i over tissues):

    dB_b/dt       = -k_coat B_b - sum_i k_in[i,b] B_b + sum_i k_out[i] T_{i,b} - CL_renal B_b
    dB_c/dt       =  k_coat B_b - sum_i k_in[i,c] B_c + sum_i k_out[i] T_{i,c}
    dT_{i,s}/dt   =  k_in[i,s] B_s - k_out[i] T_{i,s} - [i = liver] CL_liver T_{i,s}
    dA_renal/dt   =  CL_renal B_b
    dA_hepatic/dt =  CL_liver (T_{liver,b} + T_{liver,c})

Parameter grouping (12 free parameters, see ``expand_rates``):

    k_in[i,c] = k_in[i,b] * m_i, with m_liver = m_spleen = m_RES, m_brain free,
                m_lung = m_kidney = 1 (fixed).
    k_out[lung] = k_out[kidney] = k_out_fast
    k_out[liver] = k_out[spleen] = k_out[brain] = k_out_slow

Two implementations are provided on purpose:

* ``simulate``: JAX + diffrax. Differentiable, used later for NUTS.
* ``simulate_scipy``: a plain NumPy right-hand side solved with SciPy. It is the
  independent reference against which the JAX version is tested.

Because the system is linear (dy/dt = A y), ``simulate_exact`` also gives the
exact solution y(t) = expm(A t) y0, used as a third check in the tests.
"""

from __future__ import annotations

from types import ModuleType
from typing import Mapping

import diffrax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

TISSUES: tuple[str, ...] = ("lung", "liver", "spleen", "kidney", "brain")
N_TISSUES: int = len(TISSUES)
LIVER: int = TISSUES.index("liver")

PARAM_NAMES: tuple[str, ...] = (
    "k_coat",
    "CL_renal",
    "CL_liver",
    "k_in_lung",
    "k_in_liver",
    "k_in_spleen",
    "k_in_kidney",
    "k_in_brain",
    "m_RES",
    "m_brain",
    "k_out_fast",
    "k_out_slow",
)

# "True" parameter values used to generate the synthetic data (rates in 1/h).
# Justifications are in docs/phase1_parameters.md and in the README.
TRUE_PARAMS: dict[str, float] = {
    "k_coat": 1.5,  # effective hard-corona maturation, half-life about 28 min
    "CL_renal": 0.8,  # glomerular filtration of bare particles in blood
    "CL_liver": 0.05,  # slow hepatobiliary elimination, half-life about 14 h
    "k_in_lung": 0.3,
    "k_in_liver": 0.4,  # Kupffer cell uptake
    "k_in_spleen": 0.1,
    "k_in_kidney": 0.2,  # tissue uptake, distinct from filtration
    "k_in_brain": 0.01,  # blood-brain barrier: very low uptake
    "m_RES": 3.0,  # coated particles are recognized by liver and spleen macrophages
    "m_brain": 0.3,  # the corona reduces passage across the blood-brain barrier
    "k_out_fast": 0.5,  # return from exchange tissues (lung, kidney)
    "k_out_slow": 0.03,  # return from sequestering tissues (liver, spleen, brain)
}

STATE_NAMES: tuple[str, ...] = (
    ("B_b", "B_c")
    + tuple(f"T_{t}_b" for t in TISSUES)
    + tuple(f"T_{t}_c" for t in TISSUES)
    + ("A_renal", "A_hepatic")
)
N_STATES: int = len(STATE_NAMES)

# Index helpers for the state vector.
IDX_B_BARE: int = 0
IDX_B_COAT: int = 1
SLICE_T_BARE = slice(2, 2 + N_TISSUES)
SLICE_T_COAT = slice(2 + N_TISSUES, 2 + 2 * N_TISSUES)
IDX_A_RENAL: int = 2 + 2 * N_TISSUES
IDX_A_HEPATIC: int = IDX_A_RENAL + 1

DEFAULT_DOSE: float = 100.0  # %ID
T_END: float = 24.0  # h


def expand_rates(
    params: Mapping[str, float], xp: ModuleType = jnp
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Map the 12 free parameters to per-tissue rate vectors.

    Args:
        params: Mapping from each name in ``PARAM_NAMES`` to a positive value.
        xp: Array module, ``jax.numpy`` (default, differentiable) or ``numpy``
            (used by the SciPy reference to avoid JAX dispatch overhead).

    Returns:
        Tuple ``(k_in_bare, k_in_coat, k_out)``, each of shape ``(5,)`` and
        ordered as ``TISSUES``. Units: 1/h.
    """
    k_in_bare = xp.stack(
        [
            params["k_in_lung"],
            params["k_in_liver"],
            params["k_in_spleen"],
            params["k_in_kidney"],
            params["k_in_brain"],
        ]
    )
    one = xp.ones_like(params["m_RES"])
    multipliers = xp.stack([one, params["m_RES"], params["m_RES"], one, params["m_brain"]])
    k_in_coat = k_in_bare * multipliers
    k_out = xp.stack(
        [
            params["k_out_fast"],
            params["k_out_slow"],
            params["k_out_slow"],
            params["k_out_fast"],
            params["k_out_slow"],
        ]
    )
    return k_in_bare, k_in_coat, k_out


def initial_state(dose: float = DEFAULT_DOSE, coated_fraction: float = 0.0) -> np.ndarray:
    """Return the initial state: an IV bolus in blood.

    Args:
        dose: Injected amount in %ID.
        coated_fraction: Fraction of the dose injected already coated (0 = bare
            particles, the default; 1 = particles pre-incubated in plasma). Used
            in phase 4 to evaluate a hypothetical second experimental arm.

    Returns:
        Array of shape ``(14,)``, zero except for ``B_b`` and ``B_c``.
    """
    y0 = np.zeros(N_STATES)
    y0[IDX_B_BARE] = dose * (1.0 - coated_fraction)
    y0[IDX_B_COAT] = dose * coated_fraction
    return y0


# --------------------------------------------------------------------------
# JAX / diffrax implementation
# --------------------------------------------------------------------------


def rhs(t: float, y: jnp.ndarray, params: Mapping[str, float]) -> jnp.ndarray:
    """Right-hand side dy/dt of the ODE system (JAX version).

    Args:
        t: Time in hours (unused: the system is autonomous; kept for the solver API).
        y: State vector of shape ``(14,)``.
        params: Mapping from parameter names to values.

    Returns:
        Time derivative of ``y``, shape ``(14,)``.
    """
    del t
    k_in_bare, k_in_coat, k_out = expand_rates(params)
    liver_mask = jnp.zeros(N_TISSUES).at[LIVER].set(1.0)

    blood_bare = y[IDX_B_BARE]
    blood_coat = y[IDX_B_COAT]
    tissue_bare = y[SLICE_T_BARE]
    tissue_coat = y[SLICE_T_COAT]

    d_blood_bare = (
        -params["k_coat"] * blood_bare
        - jnp.sum(k_in_bare) * blood_bare
        + jnp.sum(k_out * tissue_bare)
        - params["CL_renal"] * blood_bare
    )
    d_blood_coat = (
        params["k_coat"] * blood_bare
        - jnp.sum(k_in_coat) * blood_coat
        + jnp.sum(k_out * tissue_coat)
    )
    d_tissue_bare = (
        k_in_bare * blood_bare - k_out * tissue_bare - params["CL_liver"] * liver_mask * tissue_bare
    )
    d_tissue_coat = (
        k_in_coat * blood_coat - k_out * tissue_coat - params["CL_liver"] * liver_mask * tissue_coat
    )
    d_renal = params["CL_renal"] * blood_bare
    d_hepatic = params["CL_liver"] * (tissue_bare[LIVER] + tissue_coat[LIVER])

    return jnp.concatenate(
        [
            jnp.stack([d_blood_bare, d_blood_coat]),
            d_tissue_bare,
            d_tissue_coat,
            jnp.stack([d_renal, d_hepatic]),
        ]
    )


def simulate(
    params: Mapping[str, float],
    ts: np.ndarray | jnp.ndarray,
    dose: float = DEFAULT_DOSE,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 100_000,
    throw: bool = True,
    coated_fraction: float = 0.0,
) -> jnp.ndarray:
    """Solve the ODE with diffrax and return the states at the requested times.

    The solver is Tsit5 (explicit Runge-Kutta 5(4)) with adaptive steps. With
    the true parameters the fastest time scale is about 0.3 h and the slowest
    about 50 h (stiffness ratio about 170, see results/phase1_summary.txt), so
    the problem is not stiff enough to require an implicit method.

    Parameters are converted to float64 JAX arrays so that a new parameter
    value does not trigger a recompilation (Python floats would be treated as
    static by the JIT-compiled solver).

    Args:
        params: Mapping from parameter names to values (floats or JAX scalars).
        ts: Increasing output times in hours, all >= 0.
        dose: Injected amount in %ID.
        rtol: Relative tolerance of the step-size controller.
        atol: Absolute tolerance of the step-size controller.
        max_steps: Maximum number of solver steps.
        throw: If True, raise on solver failure. If False, failed solves return
            non-finite values instead; used during MCMC, where an extreme
            parameter proposal must be rejected rather than crash the run.
        coated_fraction: Fraction of the dose injected already coated.

    Returns:
        Array of shape ``(len(ts), 14)``.
    """
    ts = jnp.asarray(ts, dtype=jnp.float64)
    params = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()}
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Tsit5(),
        t0=0.0,
        t1=ts[-1],
        dt0=0.01,
        y0=jnp.asarray(initial_state(dose, coated_fraction)),
        args=params,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
        throw=throw,
    )
    return solution.ys


# --------------------------------------------------------------------------
# NumPy / SciPy reference implementation
# --------------------------------------------------------------------------


def rhs_numpy(t: float, y: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Right-hand side written as an explicit loop over tissues (reference version).

    This is a deliberately literal transcription of the equations, independent
    of the vectorized JAX code, so that the two can be checked against each other.

    Args:
        t: Time in hours (unused).
        y: State vector of shape ``(14,)``.
        params: Mapping from parameter names to values.

    Returns:
        Time derivative of ``y``, shape ``(14,)``.
    """
    del t
    k_in_bare, k_in_coat, k_out = expand_rates(params, xp=np)
    dy = np.zeros(N_STATES)

    blood_bare = y[IDX_B_BARE]
    blood_coat = y[IDX_B_COAT]

    dy[IDX_B_BARE] = -params["k_coat"] * blood_bare - params["CL_renal"] * blood_bare
    dy[IDX_B_COAT] = params["k_coat"] * blood_bare

    for i in range(N_TISSUES):
        ib = SLICE_T_BARE.start + i
        ic = SLICE_T_COAT.start + i
        uptake_bare = k_in_bare[i] * blood_bare
        uptake_coat = k_in_coat[i] * blood_coat
        return_bare = k_out[i] * y[ib]
        return_coat = k_out[i] * y[ic]

        dy[IDX_B_BARE] += -uptake_bare + return_bare
        dy[IDX_B_COAT] += -uptake_coat + return_coat
        dy[ib] = uptake_bare - return_bare
        dy[ic] = uptake_coat - return_coat

        if i == LIVER:
            dy[ib] -= params["CL_liver"] * y[ib]
            dy[ic] -= params["CL_liver"] * y[ic]
            dy[IDX_A_HEPATIC] += params["CL_liver"] * (y[ib] + y[ic])

    dy[IDX_A_RENAL] = params["CL_renal"] * blood_bare
    return dy


def simulate_scipy(
    params: Mapping[str, float],
    ts: np.ndarray,
    dose: float = DEFAULT_DOSE,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> np.ndarray:
    """Solve the ODE with ``scipy.integrate.solve_ivp`` (LSODA), reference version.

    Args:
        params: Mapping from parameter names to float values.
        ts: Increasing output times in hours, all >= 0.
        dose: Injected amount in %ID.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        Array of shape ``(len(ts), 14)``.
    """
    params = {k: float(v) for k, v in params.items()}
    ts = np.asarray(ts, dtype=float)
    solution = solve_ivp(
        rhs_numpy,
        t_span=(0.0, float(ts[-1])),
        y0=initial_state(dose),
        t_eval=ts,
        args=(params,),
        method="LSODA",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"solve_ivp failed: {solution.message}")
    return solution.y.T


# --------------------------------------------------------------------------
# Exact solution (the system is linear)
# --------------------------------------------------------------------------


def system_matrix(params: Mapping[str, float]) -> np.ndarray:
    """Build the 14 x 14 matrix A such that dy/dt = A y.

    Each column of A sums to zero: whatever leaves one state enters another
    (cleared amounts enter A_renal or A_hepatic). This is the matrix form of
    mass conservation.

    Args:
        params: Mapping from parameter names to float values.

    Returns:
        Array of shape ``(14, 14)``.
    """
    k_in_bare, k_in_coat, k_out = expand_rates(params, xp=np)
    A = np.zeros((N_STATES, N_STATES))

    A[IDX_B_BARE, IDX_B_BARE] = -(params["k_coat"] + k_in_bare.sum() + params["CL_renal"])
    A[IDX_B_COAT, IDX_B_BARE] = params["k_coat"]
    A[IDX_A_RENAL, IDX_B_BARE] = params["CL_renal"]
    A[IDX_B_COAT, IDX_B_COAT] = -k_in_coat.sum()

    for i in range(N_TISSUES):
        ib = SLICE_T_BARE.start + i
        ic = SLICE_T_COAT.start + i
        A[ib, IDX_B_BARE] = k_in_bare[i]
        A[ic, IDX_B_COAT] = k_in_coat[i]
        A[IDX_B_BARE, ib] = k_out[i]
        A[IDX_B_COAT, ic] = k_out[i]
        A[ib, ib] = -k_out[i]
        A[ic, ic] = -k_out[i]
        if i == LIVER:
            for idx in (ib, ic):
                A[idx, idx] -= params["CL_liver"]
                A[IDX_A_HEPATIC, idx] = params["CL_liver"]
    return A


def system_matrix_jax(params: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """JAX version of ``system_matrix`` (differentiable, vmappable).

    Args:
        params: Mapping from parameter names to JAX scalars.

    Returns:
        Array of shape ``(14, 14)`` such that dy/dt = A y.
    """
    k_in_bare, k_in_coat, k_out = expand_rates(params)
    liver = jnp.zeros(N_TISSUES).at[LIVER].set(1.0)
    tb = jnp.arange(SLICE_T_BARE.start, SLICE_T_BARE.stop)
    tc = jnp.arange(SLICE_T_COAT.start, SLICE_T_COAT.stop)

    A = jnp.zeros((N_STATES, N_STATES))
    A = A.at[IDX_B_BARE, IDX_B_BARE].set(-(params["k_coat"] + k_in_bare.sum() + params["CL_renal"]))
    A = A.at[IDX_B_COAT, IDX_B_BARE].set(params["k_coat"])
    A = A.at[IDX_A_RENAL, IDX_B_BARE].set(params["CL_renal"])
    A = A.at[IDX_B_COAT, IDX_B_COAT].set(-k_in_coat.sum())
    A = A.at[tb, IDX_B_BARE].set(k_in_bare)
    A = A.at[tc, IDX_B_COAT].set(k_in_coat)
    A = A.at[IDX_B_BARE, tb].set(k_out)
    A = A.at[IDX_B_COAT, tc].set(k_out)
    A = A.at[tb, tb].set(-k_out - params["CL_liver"] * liver)
    A = A.at[tc, tc].set(-k_out - params["CL_liver"] * liver)
    A = A.at[IDX_A_HEPATIC, tb].set(params["CL_liver"] * liver)
    A = A.at[IDX_A_HEPATIC, tc].set(params["CL_liver"] * liver)
    return A


def simulate_exact(
    params: Mapping[str, float], ts: np.ndarray, dose: float = DEFAULT_DOSE
) -> np.ndarray:
    """Exact solution y(t) = expm(A t) y0 of the linear system.

    Args:
        params: Mapping from parameter names to float values.
        ts: Output times in hours.
        dose: Injected amount in %ID.

    Returns:
        Array of shape ``(len(ts), 14)``.
    """
    params = {k: float(v) for k, v in params.items()}
    A = system_matrix(params)
    y0 = initial_state(dose)
    return np.stack([expm(A * t) @ y0 for t in np.asarray(ts, dtype=float)])


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------


def compartment_totals(ys: np.ndarray) -> dict[str, np.ndarray]:
    """Sum bare and coated amounts per compartment.

    Args:
        ys: Array of shape ``(n_times, 14)`` returned by a ``simulate*`` function.

    Returns:
        Dict with keys ``"blood"``, each tissue name, ``"A_renal"`` and
        ``"A_hepatic"``; each value has shape ``(n_times,)``.
    """
    ys = np.asarray(ys)
    totals = {"blood": ys[:, IDX_B_BARE] + ys[:, IDX_B_COAT]}
    for i, tissue in enumerate(TISSUES):
        totals[tissue] = ys[:, SLICE_T_BARE.start + i] + ys[:, SLICE_T_COAT.start + i]
    totals["A_renal"] = ys[:, IDX_A_RENAL]
    totals["A_hepatic"] = ys[:, IDX_A_HEPATIC]
    return totals
