"""Map interpretable particle properties to rate constants (stated assumptions).

Two design properties:

* ``d``: hydrodynamic diameter of the bare particle (nm).
* ``h``: coating propensity (dimensionless), a relative measure of how fast
  plasma proteins form a hard corona on the surface (for example, set by
  surface chemistry: PEG density, charge).

The calibrated posterior (phase 3) describes a reference particle with
``d = D_REF`` and ``h = H_REF``. A different particle is described by
multiplying the affected rates of each posterior draw by a factor that equals
1 at the reference:

    CL_renal(d)   = CL_renal   * sigmoid((D_CUTOFF - d) / D_WIDTH) / sigmoid((D_CUTOFF - D_REF) / D_WIDTH)
    k_in_brain(d) = k_in_brain * exp(-(d - D_REF) / BBB_LENGTH)
    k_coat(h)     = k_coat     * h / H_REF

Assumptions (also listed in the README):

1. Renal filtration has a soft size cutoff near 6 nm (glomerular filtration
   of ultrasmall particles), modelled as a logistic function of d.
2. Uptake across the blood-brain barrier decreases exponentially with size,
   with a decay length of 4 nm.
3. The coating rate is proportional to the coating propensity.
4. Nothing else depends on d or h. In reality, size also affects liver and
   spleen uptake, and surface chemistry affects the multipliers m_RES and
   m_brain; these effects are deliberately left out to keep the maps simple.
5. The maps are exact (no uncertainty in the constants); only the calibrated
   parameters are uncertain.
"""

from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp

D_REF: float = 5.0  # nm, diameter of the calibrated reference particle
H_REF: float = 1.0  # coating propensity of the reference particle
D_CUTOFF: float = 6.0  # nm, renal filtration cutoff (midpoint of the logistic)
D_WIDTH: float = 0.7  # nm, width of the renal cutoff
BBB_LENGTH: float = 4.0  # nm, decay length of brain uptake with size

D_BOUNDS: tuple[float, float] = (3.0, 15.0)  # nm, feasible design range
H_BOUNDS: tuple[float, float] = (1 / 8, 8.0)  # feasible design range

# Normalization of the design change: one unit = 2 nm in diameter, or a
# doubling (or halving) of the coating propensity.
D_SCALE: float = 2.0  # nm


def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))


def renal_factor(d: jnp.ndarray) -> jnp.ndarray:
    """Multiplier of CL_renal for diameter ``d`` (nm); equals 1 at ``D_REF``."""
    return _sigmoid((D_CUTOFF - d) / D_WIDTH) / _sigmoid((D_CUTOFF - D_REF) / D_WIDTH)


def brain_factor(d: jnp.ndarray) -> jnp.ndarray:
    """Multiplier of k_in_brain for diameter ``d`` (nm); equals 1 at ``D_REF``."""
    return jnp.exp(-(d - D_REF) / BBB_LENGTH)


def coat_factor(h: jnp.ndarray) -> jnp.ndarray:
    """Multiplier of k_coat for coating propensity ``h``; equals 1 at ``H_REF``."""
    return h / H_REF


def apply_properties(params: Mapping[str, jnp.ndarray], d: jnp.ndarray, h: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Rates of a particle with properties (d, h), given reference-particle rates.

    Args:
        params: Rates of the reference particle (floats or arrays, e.g. one
            posterior draw, or a batch of draws).
        d: Hydrodynamic diameter (nm).
        h: Coating propensity.

    Returns:
        A new parameter dict with CL_renal, k_in_brain and k_coat modified.
    """
    out = dict(params)
    out["CL_renal"] = params["CL_renal"] * renal_factor(d)
    out["k_in_brain"] = params["k_in_brain"] * brain_factor(d)
    out["k_coat"] = params["k_coat"] * coat_factor(h)
    return out


def to_normalized(d: jnp.ndarray, h: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Normalized design change (u, v) = ((d - D_REF) / D_SCALE, log2(h / H_REF))."""
    return (d - D_REF) / D_SCALE, jnp.log2(h / H_REF)


def from_normalized(u: jnp.ndarray, v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Inverse of ``to_normalized``: returns (d, h)."""
    return D_REF + D_SCALE * u, H_REF * 2.0**v


def design_distance(d: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """Normalized distance sqrt(u^2 + v^2) from the reference particle."""
    u, v = to_normalized(d, h)
    return jnp.sqrt(u**2 + v**2)
