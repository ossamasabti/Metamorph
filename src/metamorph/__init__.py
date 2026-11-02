"""Metamorph: a compartmental model of nanoparticle transport with a state change.

The package enables 64-bit floats in JAX at import time. Amounts span several
orders of magnitude (brain content is about 0.3 %ID while liver content is
about 40 %ID), and the mass balance test needs better than single precision.
"""

import jax

jax.config.update("jax_enable_x64", True)

from metamorph.model import (  # noqa: E402
    PARAM_NAMES,
    STATE_NAMES,
    TISSUES,
    TRUE_PARAMS,
    simulate,
    simulate_scipy,
)

__all__ = [
    "PARAM_NAMES",
    "STATE_NAMES",
    "TISSUES",
    "TRUE_PARAMS",
    "simulate",
    "simulate_scipy",
]
