"""Kinematic consistency of the plane-strain element tests (P82W2-N02).

Covers the two driver fixes that the rheology calibration depends on:

* ``bulk_deformation_rate`` recovers the equivalent shear rate of an imposed
  linear velocity field exactly (pure shear -> ``gamma``, isochoric plane-strain
  compression -> ``2 |eps|``), including under periodic wrapping;
* ``run_biaxial_compression`` drives the piston with its real velocity and
  reports the *measured* bulk rate next to the nominal one.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensordem.rubble_scenarios import (
    ScenarioError,
    bulk_deformation_rate,
    run_biaxial_compression,
)


def _cloud(n=400, width=0.4, height=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, 2)) * np.array([width, height])


def test_pure_shear_velocity_field_gives_the_shear_rate():
    pos = _cloud()
    gamma = 7.5
    vel = np.column_stack([gamma * pos[:, 1], np.zeros(len(pos))])
    out = bulk_deformation_rate(pos, vel, length_x=0.4)
    assert out["gamma_dot"] == pytest.approx(gamma, rel=1e-10)
    assert out["eps_dot_x"] == pytest.approx(0.0, abs=1e-10)
    assert out["eps_dot_y"] == pytest.approx(0.0, abs=1e-10)


def test_isochoric_plane_strain_compression_gives_twice_the_axial_rate():
    pos = _cloud()
    eps = 0.8
    # no periodic centring here: on a periodic axis a linear vx(x) is not a
    # single-valued function of the wrapped coordinate (that seam term is a
    # declared limitation of the position-only isochoric remap)
    vel = np.column_stack([eps * pos[:, 0], -eps * pos[:, 1]])
    out = bulk_deformation_rate(pos, vel)
    assert out["gamma_dot"] == pytest.approx(2.0 * eps, rel=1e-10)
    assert out["eps_dot_x"] == pytest.approx(eps, rel=1e-10)
    assert out["eps_dot_y"] == pytest.approx(-eps, rel=1e-10)


def test_periodic_centering_does_not_bias_the_rate():
    """Two particles wrapped to opposite ends are neighbours, not outliers."""
    width = 0.4
    ys = np.linspace(0.02, 0.28, 200)
    xs = np.concatenate([np.full(100, 0.001), np.full(100, width - 0.001)])
    pos = np.column_stack([xs, ys])
    gamma = 3.0
    vel = np.column_stack([gamma * ys, np.zeros(len(ys))])
    raw = bulk_deformation_rate(pos, vel)
    wrapped = bulk_deformation_rate(pos, vel, length_x=width)
    assert raw["gamma_dot"] == pytest.approx(gamma, rel=1e-10)
    assert wrapped["gamma_dot"] == pytest.approx(gamma, rel=1e-10)


def test_too_few_particles_raise():
    pos = _cloud(n=4)
    with pytest.raises(ScenarioError):
        bulk_deformation_rate(pos, np.zeros_like(pos), length_x=0.4)


def test_compression_piston_carries_its_velocity_and_rate_is_measured():
    run = run_biaxial_compression(
        strain_rate=4.0, seed=0, n_particles=160, mean_radius=0.012,
        width=0.24, height=0.24, phi=0.80, total_strain=0.06, snapshots=3,
        steady_fraction=0.3)
    wall = np.asarray(run["wall_velocity"], dtype=float)
    assert wall.shape == (2, 2)
    assert wall[1, 1] < 0.0                       # piston moves down
    assert wall[0, 1] == 0.0                      # bottom wall is static
    assert wall[1, 1] == pytest.approx(-4.0 * 0.24, rel=1e-12)
    assert run["bulk_margin"] > 0.0
    # the realised bulk rate is reported and is not the nominal one verbatim
    assert 0.2 < run["measured_over_nominal_rate"] < 1.5
    for snap in run["snapshots"]:
        prof = snap["profile"]
        assert prof["gamma_dot_nominal"] > 0.0
        assert np.isfinite(prof["slope"]) and prof["slope"] > 0.0