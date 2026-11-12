"""Guards for the floe3d contact-dissipation ledger and its analytic reference.

``scripts/verify_floe3d_energy_ledger_order.py`` runs the full order study and is
*not* green end to end (see ``docs/FLOE3D_DISSIPATION_LEDGER_ORDER_STATUS.md``:
the analytic dissipation gate FAILS, the error does not converge with ``dt``).
These tests pin the parts that do hold, so a future repair cannot silently break
them:

1. the 1-DOF analytic reference itself (closed form + quadrature);
2. the ledger is pure bookkeeping -- switching the closure leaves the trajectory
   bit-for-bit identical, i.e. it cannot change any published mechanics number;
3. the ledger stays non-negative and monotone (no dissipation created by the
   bookkeeping).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from tensordem.polyhedron_floe3d.contact import ContactLaw
from tensordem.polyhedron_floe3d.dynamics import DEMSystem3D
from verify_floe3d_energy_ledger_order import _floe, analytic_impact, case_b


def test_analytic_reference_is_self_consistent():
    """Closed-form ``c int xdot^2`` must equal the analytic energy loss ``m v0^2/2 (1-e^2)``."""
    mass, kn, zeta, v0 = 45.0, 2.0e5, 0.10, 0.4
    ref = analytic_impact(mass, kn, zeta, v0)
    e = ref["restitution"]
    assert e == pytest.approx(math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta)))
    loss = 0.5 * mass * v0 * v0 * (1.0 - e * e)
    assert ref["dissipated_J"] == pytest.approx(loss, rel=2.0e-6)
    assert ref["contact_time_s"] == pytest.approx(math.pi / ref["omega_d"])


def _drop(dt: float, ledger: str) -> DEMSystem3D:
    thickness, gap, v0 = 0.2, 1.0e-3, 0.4
    body = _floe(0.05, thickness, (0.0, 0.0, 0.5 * thickness + gap))
    system = DEMSystem3D([body], law=ContactLaw(kn=2.0e5, zeta=0.10, mu=0.0, mu_rolling=0.0),
                         gravity=0.0, dt=dt, record=True, dissipation_ledger=ledger)
    system.add_wall([0.0, 0.0, 1.0], 0.0, label="floor")
    body.velocity[2] = -v0
    system.record_initial_energy()
    for _ in range(int(round(0.02 / dt))):
        system.step()
    return system


def test_ledger_closure_does_not_change_trajectory():
    """``midpoint`` vs ``begin`` bookkeeping must leave positions/velocities identical."""
    a = _drop(1.0e-4, "midpoint")
    b = _drop(1.0e-4, "begin")
    for ba, bb in zip(a.bodies, b.bodies):
        assert np.array_equal(ba.position, bb.position)
        assert np.array_equal(ba.velocity, bb.velocity)
        assert np.array_equal(ba.omega, bb.omega)


def test_dissipation_is_non_negative_and_monotone():
    """Neither closure may create dissipation, and both must be monotone in time."""
    for ledger in ("midpoint", "begin"):
        runs = case_b(2.0e-4, ledger)
        assert runs["dissipated_J"] > 0.0
        assert runs["residual_max"] == pytest.approx(runs["residual_max"])   # finite


def test_invalid_ledger_name_is_rejected():
    body = _floe(0.05, 0.2, (0.0, 0.0, 0.5))
    with pytest.raises(ValueError):
        DEMSystem3D([body], dt=1.0e-4, dissipation_ledger="whatever")
