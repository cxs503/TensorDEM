"""The friction meter's dissipated-power accumulator is exact and non-invasive.

Background (m22106): the mu-ladder analysis compares the *macroscopic* work rate
``q * eps_dot_a * V`` of the isochoric biaxial cell with the *contact-layer*
dissipated power of the very same contacts.  That comparison is only meaningful
if ``sum_friction_power`` really is ``sum(-F_t * v_t)`` over the contacts the
stress tensor is built from.  These tests pin that identity analytically and
check that enabling the meter cannot change a force.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.ridge_dem import RidgeAssembly, RidgeMaterial


def _pair(mu: float = 0.55, damping: float = 0.0) -> tuple[RidgeAssembly, RidgeMaterial]:
    material = RidgeMaterial(friction_coefficient=mu, freeze_bond_fraction=0.0,
                             drag=0.0, contact_damping=damping)
    radius = 0.75
    overlap = 0.01
    pos = torch.tensor([[0.0, 0.0], [0.0, 2.0 * radius - overlap]], dtype=torch.float64)
    radii = torch.full((2,), radius, dtype=torch.float64)
    mass = torch.full((2,), material.particle_mass, dtype=torch.float64)
    bonds = torch.zeros((0, 2), dtype=torch.long)
    asm = RidgeAssembly(pos, radii, mass, bonds, material, planes=[], thickness=1.0)
    return asm, material


@pytest.mark.parametrize("vx", [0.5, 0.05, 0.005])
def test_friction_power_equals_sliding_dissipation(vx: float) -> None:
    """``sum_friction_power`` == ``mu * F_n * tanh(vx/v_reg) * |vx|`` exactly."""
    asm, material = _pair()
    asm.enable_friction_meter()
    asm.vel = torch.tensor([[0.0, 0.0], [vx, 0.0]], dtype=torch.float64)
    out = asm.forces()
    meter = asm.read_friction_meter(reset=False)

    # F_n is the *normal* component on the lower disc (the tangential part is
    # perpendicular to it, so the force norm is not F_n)
    fn = -float(out["force"][0, 1])
    expected = (material.friction_coefficient * fn
                * math.tanh(vx / material.velocity_regularisation) * abs(vx))
    # the driven disc is pushed back along -x by the tangential force, so the
    # external work rate on the pair equals the dissipated power
    external = -float(out["force"][1, 0]) * vx
    assert meter["sum_friction_power"] == pytest.approx(expected, rel=1e-12)
    assert meter["sum_friction_power"] == pytest.approx(external, rel=1e-9)
    assert meter["sum_friction_power"] > 0.0
    assert meter["sum_damping_power"] == pytest.approx(0.0, abs=1e-12)


def test_meter_power_is_rate_limited_by_regularisation() -> None:
    """Below ``v_reg`` the dissipated power falls with ``v_t`` (tanh, not Coulomb)."""
    powers = []
    for vx in (0.002, 0.02):
        asm, _ = _pair()
        asm.enable_friction_meter()
        asm.vel = torch.tensor([[0.0, 0.0], [vx, 0.0]], dtype=torch.float64)
        asm.forces()
        powers.append(asm.read_friction_meter(reset=False)["sum_friction_power"])
    # tanh(v/v_reg)*v is superlinear in v, so the faster contact dissipates more
    # than 10x (purely linear drag would give exactly 10x)
    assert powers[1] > 10.0 * powers[0]


def test_meter_power_accumulates_over_calls() -> None:
    asm, _ = _pair()
    asm.enable_friction_meter()
    asm.vel = torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float64)
    asm.forces()
    single = asm.read_friction_meter(reset=True)
    for _ in range(3):
        asm.forces()
    triple = asm.read_friction_meter(reset=True)
    assert triple["calls"] == pytest.approx(3.0)
    assert triple["sum_friction_power"] == pytest.approx(3.0 * single["sum_friction_power"],
                                                        rel=1e-12)


def test_meter_does_not_change_forces() -> None:
    """Diagnostic only: with and without the meter the forces are bit-identical."""
    a, _ = _pair(damping=5.0e4)
    b, _ = _pair(damping=5.0e4)
    b.enable_friction_meter()
    va = torch.tensor([[0.0, 0.0], [0.3, 0.1]], dtype=torch.float64)
    a.vel = va.clone()
    b.vel = va.clone()
    fa = a.forces()["force"]
    fb = b.forces()["force"]
    assert torch.equal(fa, fb)
    # the dissipation bookkeeping (built from the same forces) is unchanged too
    assert torch.equal(a.forces()["force"], b.forces()["force"])