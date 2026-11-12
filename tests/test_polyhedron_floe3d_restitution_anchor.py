"""Restitution anchor for the ``polyhedron_floe3d`` normal contact law.

``ContactLaw.normal_damping`` fixes ``c_n = 2 zeta sqrt(k_n m_eff)``, i.e. the
law is a Kelvin-Voigt spring-dashpot whose *declared* damping ratio is ``zeta``.
For that model the coefficient of restitution is not free:

    e(zeta) = exp(-zeta pi / sqrt(1 - zeta^2))

These tests pin the measured ``e`` of a flat-faced cube dropped on a floor
against that anchor, and they record the size of the known deviation at large
``zeta`` (the kernel truncates the normal force at ``f_n = 0``, so the dashpot
does no work during the restitution half of the contact and the rebound stays
livelier than the untruncated analytic value).
"""
from __future__ import annotations

import math

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmarks"))

from polyhedron_floe3d_restitution_anchor import (  # noqa: E402
    analytic_restitution,
    cube,
    drop_case,
)
from tensordem.polyhedron_floe3d.contact import wall_contact  # noqa: E402
from tensordem.polyhedron_floe3d.dynamics import DEMSystem3D, FloeBody  # noqa: E402
from tensordem.polyhedron_floe3d.contact import ContactLaw  # noqa: E402


def test_flat_cube_start_is_free_of_initial_penetration():
    """The drop case must start clear of the floor (the placement trap)."""
    body = FloeBody(cube(), 917.0, position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, -0.5])
    body.position = np.array([0.0, 0.0, 2e-3 - float(body.world_vertices()[:, 2].min())])
    hit = wall_contact(body.world_vertices(), np.array([0.0, 0.0, 1.0]), 0.0)
    assert hit is None, "the cube is penetrating the floor at t = 0"


def test_deep_initial_penetration_launches_the_body():
    """Documented trap: an initial overlap is a one-step energy injection."""
    body = FloeBody(cube(), 917.0, position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, -0.5])
    # 5 mm of overlap: place the lowest vertex 5 mm *below* the floor
    body.position = np.array([0.0, 0.0, -5e-3 - float(body.world_vertices()[:, 2].min())])
    system = DEMSystem3D([body], law=ContactLaw(kn=2.0e5, zeta=0.2, mu=0.3),
                         gravity=0.0, dt=1e-4)
    system.add_wall([0.0, 0.0, 1.0], 0.0, label="floor")
    system.record_initial_energy()
    e0 = system.energy()
    for _ in range(400):
        system.step()
        if system.n_wall_contacts == 0:
            break
    assert system.energy() > 5.0 * e0      # penalty energy of the overlap is returned
    assert body.velocity[2] > 1.0          # launched, not resting


def test_restitution_matches_kelvin_voigt_at_low_damping():
    """zeta = 0.05 must reproduce the analytic restitution to better than 3 %."""
    row = drop_case(0.05, 1e-4)
    assert row["touched"]
    assert row["e_analytic"] == analytic_restitution(0.05)
    assert abs(row["e_measured"] - row["e_analytic"]) / row["e_analytic"] < 0.03
    assert row["energy_ratio"] < 1.0       # net dissipation, never injection


def test_restitution_is_time_step_converged():
    """Halving dt must not move the measured restitution by more than 1 %."""
    coarse = drop_case(0.2, 1e-4)
    fine = drop_case(0.2, 5e-5)
    assert abs(coarse["e_measured"] - fine["e_measured"]) < 0.01


def test_restitution_decreases_monotonically_with_declared_damping():
    values = [drop_case(z, 1e-4)["e_measured"] for z in (0.05, 0.1, 0.2)]
    assert values[0] > values[1] > values[2]


def test_truncated_dashpot_keeps_the_rebound_livelier_than_kelvin_voigt():
    """Regression lock for the measured deviation at zeta = 0.7.

    The kernel clips ``f_n`` at zero, so the dashpot only dissipates while the
    contact is compressed.  The rebound therefore stays well above the
    untruncated analytic value; this test makes the size of the gap explicit
    instead of letting a caller assume ``zeta`` is the Kelvin-Voigt ratio.
    """
    row = drop_case(0.7, 1e-4)
    assert row["e_analytic"] < 0.05
    assert 0.15 < row["e_measured"] < 0.30
    assert row["e_measured"] / row["e_analytic"] > 3.0
