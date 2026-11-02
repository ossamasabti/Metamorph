"""Synthetic experiment: observation design, noise model, dataset I/O.

The design mimics a biodistribution study after an IV bolus:

* Blood (bare + coated, not separable) and cumulative urinary excretion
  (A_renal) are sampled densely, including early times (5, 15, 30 min),
  because serial blood sampling and urine collection are non-terminal.
* Tissue totals (bare + coated) are sampled sparsely (1, 4, 24 h), because
  organ harvesting is terminal: each time point costs a group of animals
  (or chips).
* A_hepatic (biliary / fecal excretion) is not observed.

Noise model: multiplicative log-normal with a single scale ``sigma``,

    y_obs = y_true * exp(eps),  eps ~ Normal(0, sigma^2),

equivalently log y_obs ~ Normal(log y_true, sigma^2). Replicates are
independent draws.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from metamorph.model import (
    IDX_A_RENAL,
    IDX_B_BARE,
    IDX_B_COAT,
    N_STATES,
    SLICE_T_BARE,
    SLICE_T_COAT,
    TISSUES,
    simulate_scipy,
)

OBSERVABLES: tuple[str, ...] = ("blood", *TISSUES, "urine")

T_DENSE: tuple[float, ...] = (5 / 60, 15 / 60, 30 / 60, 1.0, 4.0, 8.0, 24.0)  # h
T_SPARSE: tuple[float, ...] = (1.0, 4.0, 24.0)  # h

OBSERVATION_TIMES: dict[str, tuple[float, ...]] = {
    "blood": T_DENSE,
    "urine": T_DENSE,
    **{tissue: T_SPARSE for tissue in TISSUES},
}

TRUE_SIGMA: float = 0.2  # log-scale noise, about 20 % coefficient of variation
N_REPLICATES: int = 3
SEED: int = 20261102


def observation_matrix() -> np.ndarray:
    """Matrix H such that ``H @ y`` gives the observable amounts from a state ``y``.

    Each observable is a sum of states: blood = B_b + B_c, tissue i =
    T_{i,b} + T_{i,c}, urine = A_renal. The bare/coated split is not observed.

    Returns:
        Array of shape ``(len(OBSERVABLES), 14)`` with entries 0 or 1.
    """
    H = np.zeros((len(OBSERVABLES), N_STATES))
    H[0, [IDX_B_BARE, IDX_B_COAT]] = 1.0
    for i in range(len(TISSUES)):
        H[1 + i, SLICE_T_BARE.start + i] = 1.0
        H[1 + i, SLICE_T_COAT.start + i] = 1.0
    H[-1, IDX_A_RENAL] = 1.0
    return H


@dataclass(frozen=True)
class Dataset:
    """Long-format dataset: one row per measurement.

    Attributes:
        observable: Observable name per row, values in ``OBSERVABLES``.
        time: Sampling time per row (h).
        replicate: Replicate index per row (0-based).
        value: Measured amount per row (%ID), strictly positive.
    """

    observable: np.ndarray
    time: np.ndarray
    replicate: np.ndarray
    value: np.ndarray

    def __len__(self) -> int:
        return len(self.value)

    @property
    def obs_index(self) -> np.ndarray:
        """Row index into ``OBSERVABLES`` (and into the rows of ``observation_matrix``)."""
        return np.array([OBSERVABLES.index(o) for o in self.observable])

    @property
    def unique_times(self) -> np.ndarray:
        """Sorted unique sampling times; the ODE only needs to be solved at these."""
        return np.unique(self.time)

    @property
    def time_index(self) -> np.ndarray:
        """Row index into ``unique_times``."""
        return np.searchsorted(self.unique_times, self.time)


def design_rows(n_replicates: int = N_REPLICATES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate (observable, time, replicate) for every planned measurement.

    Args:
        n_replicates: Number of independent replicates per (observable, time).

    Returns:
        Three aligned arrays: observable names, times (h), replicate indices.
    """
    names, times, reps = [], [], []
    for name in OBSERVABLES:
        for t in OBSERVATION_TIMES[name]:
            for r in range(n_replicates):
                names.append(name)
                times.append(t)
                reps.append(r)
    return np.array(names), np.array(times, dtype=float), np.array(reps, dtype=int)


def noiseless_observations(params: Mapping[str, float], dataset: Dataset) -> np.ndarray:
    """Model prediction for each row of a dataset (no noise).

    Args:
        params: Model parameters.
        dataset: Dataset whose rows define (observable, time).

    Returns:
        Array of shape ``(len(dataset),)`` in %ID.
    """
    ys = simulate_scipy(params, dataset.unique_times)  # (n_times, 14)
    obs = ys @ observation_matrix().T  # (n_times, n_observables)
    return obs[dataset.time_index, dataset.obs_index]


def generate_dataset(
    params: Mapping[str, float],
    sigma: float = TRUE_SIGMA,
    n_replicates: int = N_REPLICATES,
    seed: int = SEED,
) -> Dataset:
    """Simulate the experiment and add multiplicative log-normal noise.

    The reference SciPy solver is used here on purpose: data are generated
    with a different solver than the one used for inference (diffrax), so the
    calibration does not benefit from sharing the exact same numerical errors.

    Args:
        params: "True" model parameters.
        sigma: Standard deviation of the log-scale noise.
        n_replicates: Independent replicates per (observable, time).
        seed: Seed of the NumPy random generator.

    Returns:
        The noisy dataset.
    """
    names, times, reps = design_rows(n_replicates)
    template = Dataset(names, times, reps, np.ones_like(times))
    mean = noiseless_observations(params, template)
    rng = np.random.default_rng(seed)
    noisy = mean * np.exp(sigma * rng.standard_normal(mean.shape))
    return Dataset(names, times, reps, noisy)


def save_dataset(
    dataset: Dataset,
    directory: Path,
    params: Mapping[str, float],
    sigma: float,
    seed: int,
) -> None:
    """Write ``synthetic_data.csv`` and ``true_parameters.json`` to ``directory``.

    Args:
        dataset: Dataset to save.
        directory: Output folder (created if missing).
        params: True parameters used to generate the data.
        sigma: True noise scale.
        seed: Random seed used.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "synthetic_data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["observable", "time_h", "replicate", "amount_pct_id"])
        for row in zip(dataset.observable, dataset.time, dataset.replicate, dataset.value):
            writer.writerow([row[0], f"{row[1]:.6f}", int(row[2]), f"{row[3]:.8g}"])
    meta = {
        "description": "True values used to generate data/synthetic_data.csv",
        "units": {"rates": "1/h", "amounts": "%ID", "time": "h"},
        "parameters": dict(params),
        "sigma_log": sigma,
        "seed": seed,
        "observation_times_h": {k: list(v) for k, v in OBSERVATION_TIMES.items()},
        "n_replicates": int(dataset.replicate.max() + 1),
    }
    (directory / "true_parameters.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_dataset(directory: Path) -> Dataset:
    """Read ``synthetic_data.csv`` written by ``save_dataset``.

    Args:
        directory: Folder containing ``synthetic_data.csv``.

    Returns:
        The dataset.
    """
    with open(directory / "synthetic_data.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return Dataset(
        observable=np.array([r["observable"] for r in rows]),
        time=np.array([float(r["time_h"]) for r in rows]),
        replicate=np.array([int(r["replicate"]) for r in rows]),
        value=np.array([float(r["amount_pct_id"]) for r in rows]),
    )


def load_true_parameters(directory: Path) -> tuple[dict[str, float], float]:
    """Read the true parameters and noise scale from ``true_parameters.json``.

    Args:
        directory: Folder containing ``true_parameters.json``.

    Returns:
        Tuple ``(params, sigma)``.
    """
    meta = json.loads((directory / "true_parameters.json").read_text(encoding="utf-8"))
    return meta["parameters"], meta["sigma_log"]
