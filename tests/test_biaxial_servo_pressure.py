"""Regression test: the lateral servo must hold sigma_3 at finite axial strain.

Defect (m20304, CASE-DEM-BIAXIAL-FRICTION): ``run_biaxial_shear_band`` built the
lateral servo planes with ``servo_width`` fixed at the *initial* specimen height
and never updated it.  Because the servo measures its pressure as
``force / (servo_width * thickness)``, the regulated quantity was the
initial-height-referenced stress, and the true confining pressure drifted as
``sigma_3 / (1 - eps_axial)`` -- measured 174 kPa against a 100 kPa target at
``eps_axial = 0.33``.  A Mohr-Coulomb angle read off such a test is
mis-attributed, so this test pins the invariant "sigma_3 holds at finite strain".

The specimen is deliberately small (6 x 10, ~60 discs) so the test stays a
short local run.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.biaxial_shear_band import run_biaxial_shear_band


@pytest.fixture(scope="module")
def small_run():
    return run_biaxial_shear_band(
        confining_stress=100.0e3, columns=6, rows=10, radius=0.75,
        strain_rate=0.05, strain_target=0.12, consolidation_steps=400,
        snapshot_strain=1.0e9, wall_friction=0.0,
        servo_gain=3.0e-4, servo_velocity_limit=2.0,
        material_overrides={"freeze_bond_fraction": 0.0},
    )


def test_lateral_stress_does_not_drift_with_strain(small_run):
    hist = small_run["stress_path"]
    eps = np.asarray(hist["axial_strain"])
    sh = np.asarray(hist["sigma_h"])
    target = 100.0e3
    # compare the late branch (eps > 0.06) against the early branch
    early = sh[(eps > 0.02) & (eps < 0.04)]
    late = sh[eps > 0.08]
    assert early.size and late.size
    # stale servo would give late/early ~ (1-0.03)/(1-0.10) = 1.08 growth and a
    # tail value of sigma_3/(1-0.10) = 111 % of target; allow a servo lag band
    assert late.mean() < 1.10 * early.mean()
    assert 0.85 < late.mean() / target < 1.12


def test_axial_strain_actually_advances(small_run):
    hist = small_run["stress_path"]
    eps = np.asarray(hist["axial_strain"])
    assert eps[-1] > 0.08          # the platen真的是位移控制
    assert np.all(np.diff(eps) >= -1e-12)


def test_specimen_is_unbonded_when_bonds_are_disabled(small_run):
    assert small_run["specimen"]["bonds"] == 0