"""Tests of the synthetic data generation."""

from pathlib import Path

import numpy as np

import metamorph  # noqa: F401
from metamorph.data import (
    OBSERVABLES,
    OBSERVATION_TIMES,
    Dataset,
    design_rows,
    generate_dataset,
    load_dataset,
    noiseless_observations,
    observation_matrix,
    save_dataset,
)
from metamorph.model import TRUE_PARAMS, compartment_totals, simulate_scipy


def test_observation_matrix_matches_compartment_totals() -> None:
    ts = np.array([0.5, 4.0])
    ys = simulate_scipy(TRUE_PARAMS, ts)
    obs = ys @ observation_matrix().T
    totals = compartment_totals(ys)
    for k, name in enumerate(OBSERVABLES):
        key = "A_renal" if name == "urine" else name
        np.testing.assert_allclose(obs[:, k], totals[key])


def test_design_size() -> None:
    names, _, _ = design_rows(n_replicates=3)
    expected = 3 * sum(len(OBSERVATION_TIMES[o]) for o in OBSERVABLES)
    assert len(names) == expected == 87


def test_generation_is_reproducible_and_positive() -> None:
    a = generate_dataset(TRUE_PARAMS, seed=1)
    b = generate_dataset(TRUE_PARAMS, seed=1)
    c = generate_dataset(TRUE_PARAMS, seed=2)
    np.testing.assert_array_equal(a.value, b.value)
    assert not np.allclose(a.value, c.value)
    assert np.all(a.value > 0)


def test_noise_scale_is_recovered_with_many_replicates() -> None:
    ds = generate_dataset(TRUE_PARAMS, sigma=0.2, n_replicates=200, seed=0)
    resid = np.log(ds.value) - np.log(noiseless_observations(TRUE_PARAMS, ds))
    assert abs(resid.std() - 0.2) < 0.01
    assert abs(resid.mean()) < 0.01


def test_zero_noise_gives_model_output() -> None:
    ds = generate_dataset(TRUE_PARAMS, sigma=0.0)
    np.testing.assert_allclose(ds.value, noiseless_observations(TRUE_PARAMS, ds))


def test_save_load_roundtrip(tmp_path: Path) -> None:
    ds = generate_dataset(TRUE_PARAMS, seed=3)
    save_dataset(ds, tmp_path, TRUE_PARAMS, 0.2, 3)
    loaded = load_dataset(tmp_path)
    assert isinstance(loaded, Dataset)
    np.testing.assert_array_equal(loaded.observable, ds.observable)
    np.testing.assert_allclose(loaded.time, ds.time, atol=1e-6)
    np.testing.assert_allclose(loaded.value, ds.value, rtol=1e-7)
