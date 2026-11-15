"""``strain_target`` is a logarithmic (true) strain budget and is actually met.

Defect found in m22106: the shear loop broke on the *engineering* strain while
the step budget ``strain_target / (strain_rate * dt)`` is a budget on the
logarithmic strain (the platen velocity is proportional to the current height,
so ``d(log H)/dt = strain_rate``).  A run asked for ``eps_a = 0.60`` therefore
stopped at engineering strain 0.451 (= log 0.60) when the budget ran out, and
reported nothing about the shortfall -- which is why earlier rounds of
CASE-DEM-BIAXIAL-FRICTION documented "eps_a -> 0.45" for a declared 0.60.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.biaxial_shear_band import run_biaxial_shear_band

TARGET = 0.30
RATE = 0.20          # fast: this test is about strain accounting, not physics


@pytest.fixture(scope="module")
def run() -> dict:
    return run_biaxial_shear_band(
        confining_stress=100.0e3, columns=5, rows=10, radius=0.75,
        strain_rate=RATE, strain_target=TARGET, consolidation_steps=50,
        snapshot_strain=1.0e9, wall_friction=0.0, lateral_control="isochoric",
        material_overrides={"freeze_bond_fraction": 0.0})


def test_declared_target_is_reached_in_log_strain(run: dict) -> None:
    eps_log = float(np.asarray(run["stress_path"]["axial_strain_log"])[-1])
    dt = float(run["specimen"]["dt"])
    one_step = RATE * dt
    assert eps_log >= TARGET - 1e-12, (
        f"declared log-strain target {TARGET} not reached: got {eps_log}")
    assert eps_log <= TARGET + 2.0 * one_step, (
        f"run overshot the declared target by more than two steps: {eps_log}")


def test_engineering_strain_lags_log_strain_as_expected(run: dict) -> None:
    sp = run["stress_path"]
    eps = np.asarray(sp["axial_strain"], dtype=float)
    eps_log = np.asarray(sp["axial_strain_log"], dtype=float)
    # uniaxial compression of an isochoric cell: log = -ln(1 - engineering)
    expected = -np.log1p(-eps)
    assert np.allclose(eps_log, expected, atol=2e-3 * max(1.0, eps_log[-1] / TARGET))
    # and the two measures differ substantially at the declared target, which is
    # exactly why the old break condition ended the run early
    assert eps[-1] < 0.95 * eps_log[-1]


def test_run_is_volume_preserving(run: dict) -> None:
    ev = np.asarray(run["stress_path"]["volumetric_strain_log"], dtype=float)
    # an exactly volume-preserving cell reports log eps_v ~ 0; the residual is
    # the wall-discretisation drift, not a boundary leak
    assert np.all(np.abs(ev) < 5e-3)
    assert abs(float(ev[-1]) - float(ev[0])) < 5e-3
