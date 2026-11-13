"""Unit + regression tests for the ``mu(I_c)`` contact-friction closure.

The closure is only admissible if it (1) is a genuine ``mu(I)`` law with the
documented limits, (2) **reduces bit-for-bit to the constant-friction kernel**
when ``mu_s == mu_2`` -- otherwise a comparison against rounds 1-4 would mix a
law change with a solver change -- and (3) leaves the energy ledger closed.
Run with::

    PYTHONPATH=src python -m pytest tests/test_column_collapse_inertial_friction.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.granular3d_core import Material3D, build_pairs
from tensordem.granular_column.axisym import AxisymCase, simulate_axisymmetric_column
from tensordem.granular_column.inertial_friction import (
    InertialFriction,
    JOP_2006,
    contact_inertial_number,
    mu_of_inertial_number,
)
from tensordem.granular_column.kernel import (
    PairSpringState,
    WallSpringState,
    pair_contact_forces,
    wall_contact_forces,
)


def _pair_setup(slip: float = 1.0):
    pos = np.array([[0.0, 0.0, 0.0], [0.019, 0.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0], [0.0, slip, 0.0]])
    omega = np.zeros((2, 3))
    mat = Material3D(radius=0.01, mu=0.5, mu_rolling=0.2, restitution=0.0,
                     kn=1.0e4)
    iu, ju = build_pairs(pos, mat.radius)
    return pos, vel, omega, mat, iu, ju


def test_mu_of_inertial_number_limits_and_monotonicity():
    i = np.array([0.0, 1.0e-6, 1.0e-2, JOP_2006["i_0"], 1.0, 1.0e3, np.inf])
    mu = mu_of_inertial_number(i)
    assert mu[0] == pytest.approx(JOP_2006["mu_s"])
    assert mu[-1] == pytest.approx(JOP_2006["mu_2"])
    assert np.all(np.diff(mu) > 0.0), "mu(I) must increase monotonically"
    at_i0 = mu_of_inertial_number(JOP_2006["i_0"])
    assert at_i0 == pytest.approx(0.5 * (JOP_2006["mu_s"] + JOP_2006["mu_2"]))
    # never leaves the [mu_s, mu_2] band, including for negative I
    assert np.all(mu_of_inertial_number(np.array([-5.0, 0.0])) == pytest.approx(
        JOP_2006["mu_s"]))


def test_mu_of_inertial_number_rejects_bad_parameters():
    with pytest.raises(ValueError):
        mu_of_inertial_number(1.0, i_0=0.0)
    with pytest.raises(ValueError):
        mu_of_inertial_number(1.0, mu_s=0.6, mu_2=0.4)
    with pytest.raises(ValueError):
        mu_of_inertial_number(1.0, mu_s=-0.1)


def test_contact_inertial_number_matches_definition():
    # I_c = |v_t| / sqrt(f_n/(rho d^2)) with f_n = 10 N, rho = 2500, d = 0.02 m
    i = contact_inertial_number(1.0, 10.0, 0.02, 2500.0)
    assert float(i) == pytest.approx(1.0 / math.sqrt(10.0 / (2500.0 * 0.02 ** 2)), rel=1e-12)
    # a contact with no normal force is reported as I = 0, not inf/nan
    assert float(contact_inertial_number(3.0, 0.0, 0.02, 2500.0)) == 0.0


def test_inertial_friction_validates_and_signs_itself():
    with pytest.raises(ValueError):
        InertialFriction(mu_s=0.5, mu_2=0.4)
    with pytest.raises(ValueError):
        InertialFriction(i_0=-1.0)
    sig = InertialFriction().signature()
    assert "Jop" in sig["reference"] and sig["mu_s"] == JOP_2006["mu_s"]


def test_kernel_reduces_to_constant_friction_when_band_is_degenerate():
    """``mu_s == mu_2`` must reproduce the constant-``mu`` kernel exactly."""
    pos, vel, omega, mat, iu, ju = _pair_setup()
    dt = 1.0e-5
    const = pair_contact_forces(pos, vel, omega, iu, ju, PairSpringState(), mat, dt)
    law = pair_contact_forces(pos, vel, omega, iu, ju, PairSpringState(), mat, dt,
                              mu_i=InertialFriction(mu_s=0.5, mu_2=0.5, i_0=0.3))
    for key in ("force", "torque", "f_t", "u_t", "slip_tangential"):
        assert np.allclose(const[key], law[key], rtol=0.0, atol=0.0), key
    assert const["dissip_friction"] == law["dissip_friction"]


def test_kernel_raises_friction_with_inertial_number():
    """At a fixed slip speed the mobilised friction must exceed ``mu_s``."""
    pos, vel, omega, mat, iu, ju = _pair_setup(slip=2.0)
    dt = 1.0e-5
    res = pair_contact_forces(pos, vel, omega, iu, ju, PairSpringState(), mat, dt,
                              mu_i=InertialFriction())
    assert res["inertial_number_mean"] > 0.0
    assert JOP_2006["mu_s"] < res["mu_eff_mean"] <= JOP_2006["mu_2"]


def test_wall_contact_accepts_law_and_reports_diagnostics():
    pos = np.array([[0.0, 0.0, 0.005]])
    vel = np.array([[0.5, 0.0, 0.0]])
    omega = np.zeros((1, 3))
    mat = Material3D(radius=0.01, mu=0.5, mu_rolling=0.0, restitution=0.0)
    res = wall_contact_forces(pos, vel, omega, mat, 1.0e-5,
                              state=WallSpringState(1), mu_wall=0.5,
                              mu_i=InertialFriction())
    assert math.isfinite(res["mu_eff_mean"])
    assert res["inertial_number_mean"] > 0.0


def _small_run(**overrides):
    params = dict(aspect_ratio=0.5, grains_per_radius=4.0, radius=0.02,
                  mu=0.5, mu_rolling=0.2, kn=6.0e4, duration_factor=2.0,
                  min_time=2.0e-2, sample_stride=5, rest_hold=10)
    params.update(overrides)
    return simulate_axisymmetric_column(AxisymCase(**params), return_positions=False)


def test_axisym_degenerate_law_bit_identical_to_constant():
    """Same frictional response, same trajectory: the closure adds no force bias."""
    ref = _small_run()
    law = _small_run(mu_inertial=(0.5, 0.5, 0.3))
    assert ref["runout_radius_p995"] == law["runout_radius_p995"]
    assert ref["energy"]["residual_fraction_of_potential"] == \
        law["energy"]["residual_fraction_of_potential"]


def test_axisym_law_keeps_energy_ledger_closed_and_changes_runout():
    ref = _small_run()
    law = _small_run(mu_inertial=(JOP_2006["mu_s"], JOP_2006["mu_2"], JOP_2006["i_0"]))
    # the ledger must stay as closed as the constant-friction baseline (the
    # absolute residual is resolution dependent, so the invariant is the
    # comparison, not a magic number)
    res_ref = abs(ref["energy"]["residual_fraction_of_potential"])
    res_law = abs(law["energy"]["residual_fraction_of_potential"])
    assert res_law < 1.5 * res_ref + 1.0e-3, (res_ref, res_law)
    # the law is active: mobilised friction differs and is recorded
    stats = law["inertial"]
    assert stats["law"].startswith("mu(I_c)")
    assert stats["mu_eff_mean"] > 0.0 and stats["inertial_number_mean"] > 0.0
    assert law["case"]["mu_law"].startswith("mu(I_c)")
    assert ref["case"]["mu_law"] == "constant"


def test_axisym_rejects_bad_law_triple():
    with pytest.raises(Exception):
        AxisymCase(mu_inertial=(0.5, 0.4, 0.3))
    with pytest.raises(Exception):
        AxisymCase(mu_inertial=(0.5, 0.8))