"""Dynamics, buoyancy and ledger tests for the 3-D polyhedral floe DEM.

The energy bookkeeping of a DEM is the part most easily faked, so the tests here
distinguish the *drift* of the discrete invariant of the integrator from the
bounded oscillation of the continuous energy, and check that the ledger of a
dissipative run never *creates* energy.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.polyhedron_floe3d.contact import ContactLaw
from tensordem.polyhedron_floe3d.dynamics import DEMSystem3D, FloeBody
from tensordem.polyhedron_floe3d.geometry import extrude_polygon, regular_polygon
from tensordem.polyhedron_floe3d.study import conservation_case, peak_force_analytic


def _plate(area: float = 0.25, thickness: float = 0.2, density: float = 900.0,
           position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)) -> FloeBody:
    polygon = regular_polygon(area, 6, phase=math.pi / 6.0)
    shape = extrude_polygon(polygon, thickness, principal=False)
    return FloeBody(shape, density, position=list(position), velocity=list(velocity))


def test_peak_force_closed_form_limits():
    mass, kn, v0 = 45.0, 4.0e6, 0.4
    undamped = peak_force_analytic(mass, kn, 0.0, v0)
    assert abs(undamped - v0 * math.sqrt(kn * mass)) < 1e-9
    # damping always lowers the peak of a spring-dashpot impact
    assert peak_force_analytic(mass, kn, 0.3, v0) < undamped


def test_conservation_case_books():
    """Mass, energy and momentum ledgers of an elastic impact."""
    result = conservation_case()
    assert result["mass_residual"] <= 1e-12
    # the discrete invariant of the kick-then-drift scheme: this is the auditable
    # conservation residual (drift from start to end of the impact)
    assert result["energy_residual_shadow"] <= 1e-10
    assert result["momentum_residual"] <= 1e-10
    # the closed-form peak of a face-on spring-dashpot impact
    assert result["peak_force_rel_error"] < 1e-4
    # the continuous energy oscillates as omega*dt/2 for a first-order symplectic
    # scheme; it must do that and not drift
    assert result["energy_residual_true"] < 0.01


def test_shadow_energy_converges_quadratically():
    result = conservation_case(refinement=(1000, 2000, 4000))
    oscillations = [row["energy_residual_shadow_oscillation"] for row in result["refinement"]]
    assert oscillations[0] > oscillations[1] > oscillations[2]
    ratio = oscillations[0] / oscillations[2]
    assert ratio > 3.0        # ~4 for a second-order bounded oscillation


def test_dissipative_impact_never_gains_energy():
    body = _plate(velocity=(1.0, 0.0, 0.0))
    # start one millimetre behind the wall, so the contact happens inside the run
    body.position[0] = 0.35 - float(body.shape.vertices[:, 0].max()) - 1.0e-3
    system = DEMSystem3D([body], law=ContactLaw(kn=4.0e6, zeta=0.3, mu=0.3),
                         gravity=0.0, dt=2.0e-6)
    system.add_wall([-1.0, 0.0, 0.0], -0.35, label="wall")
    system.record_initial_energy()
    for _ in range(4000):
        system.step()
    assert system.dissipated > 0.0
    assert system.energy() <= system.energy_residual() * abs(system._initial_energy) \
        + system._initial_energy + 1e-9


def test_floating_plate_is_in_vertical_equilibrium():
    """The buoyancy of the field builder must balance the weight exactly."""
    density, water_density = 900.0, 1025.0
    thickness = 0.12
    body = _plate(area=0.2, thickness=thickness, density=density)
    draft = density / water_density * thickness
    body.position[2] = -draft + 0.5 * thickness       # as in field.build_brash_ice_field
    system = DEMSystem3D([body], law=ContactLaw(), gravity=9.81, water_level=0.0,
                         water_density=water_density, dt=1.0e-4)
    before = body.position[2]
    for _ in range(200):
        system.step()
    assert abs(body.position[2] - before) < 1e-6
    assert abs(body.velocity[2]) < 1e-6


def test_plane_constraint_freezes_out_of_plane_motion():
    body = _plate(thickness=0.1, velocity=(0.2, 0.0, 0.3), position=(0.0, 0.0, 0.4))
    body.omega = np.array([0.5, -0.2, 1.0])
    system = DEMSystem3D([body], law=ContactLaw(), gravity=0.0, dt=1.0e-5,
                         constrain_to_plane=True)
    system.step()
    assert body.position[2] == 0.0
    assert body.velocity[2] == 0.0
    assert body.omega[0] == 0.0 and body.omega[1] == 0.0
    assert abs(body.velocity[0] - 0.2) < 1e-12
    rotation = body.rotation
    assert abs(rotation[2, 2] - 1.0) < 1e-12


def test_pair_contact_transfers_momentum_and_keeps_the_ledger_positive():
    left = _plate(area=0.2, thickness=0.1, velocity=(0.5, 0.0, 0.0), position=(-0.25, 0.0, 0.0))
    right = _plate(area=0.2, thickness=0.1, velocity=(-0.5, 0.0, 0.0), position=(0.25, 0.0, 0.0))
    system = DEMSystem3D([left, right], law=ContactLaw(kn=2.0e5, zeta=0.1, mu=0.2),
                         gravity=0.0, dt=1.0e-5)
    momentum_before = system.total_momentum()
    for _ in range(4000):
        system.step()
    assert np.linalg.norm(system.total_momentum() - momentum_before) < 1e-9
    assert system.dissipated > 0.0
    assert left.velocity[0] < 0.5 and right.velocity[0] > -0.5