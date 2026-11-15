"""The rolling-resistance "couple" is algebraically an added sliding friction.

``RidgeAssembly.contact_forces`` realises the rolling-resistance couple as an
extra tangential force ``-mu_r * F_n * tanh(v_t / v_reg)`` on the same direction
as the sliding friction ``-mu * F_n * tanh(v_t / v_reg)``.  Particles in this
model carry no rotational degree of freedom, so no torque exists; the operator is

    F_t = -(mu + mu_r) * F_n * tanh(v_t / v_reg).

That identity is what these tests pin.  It matters for the biaxial critical-state
anchor (``CASE-DEM-BIAXIAL-FRICTION``): a phi_cv(mu_r) "calibration" is not
evidence about rolling resistance physics, it is a re-parameterisation of mu.
The tests are exact (to floating-point) rather than approximate, because the two
code paths that must agree are the same expression evaluated with different
``(mu, mu_r)`` split of the same sum.
"""
from __future__ import annotations

import pytest
import torch

from tensordem.ridge_dem import RidgeAssembly, RidgeMaterial

DT_OVERRIDE = None


def _pair_force(mu: float, mu_r: float, v_t: float = 1.0, overlap: float = 0.1) -> torch.Tensor:
    """Contact force tensor of two overlapping disks slipping tangentially."""
    m = RidgeMaterial(friction_coefficient=mu, rolling_friction_coefficient=mu_r,
                      freeze_bond_fraction=0.0)
    dist = 1.5 - overlap
    pos = torch.tensor([[0.0, 0.0], [dist, 0.0]], dtype=torch.float64)
    radii = torch.tensor([0.75, 0.75], dtype=torch.float64)
    mass = torch.tensor([m.particle_mass, m.particle_mass], dtype=torch.float64)
    bonds = torch.empty((0, 2), dtype=torch.int64)
    vel = torch.tensor([[0.0, 0.0], [0.0, v_t]], dtype=torch.float64)
    asm = RidgeAssembly(pos, radii, mass, bonds, material=m, planes=[])
    asm.vel = vel
    return asm.forces(dt=m.recommended_dt)["force"]


def test_rolling_surrogate_equals_extra_friction_exactly():
    """force(mu=a, mu_r=b) == force(mu=a+b, mu_r=0) for every split of the sum."""
    for a, b in [(0.55, 0.20), (0.55, 0.40), (0.00, 0.30), (0.20, 0.00), (0.10, 0.10)]:
        split = _pair_force(a, b)
        merged = _pair_force(a + b, 0.0)
        assert torch.allclose(split, merged, atol=1e-9, rtol=1e-12), (
            f"mu={a}, mu_r={b} must equal mu={a + b}, mu_r=0; "
            f"delta={ (split - merged).abs().max().item() }")


def test_pure_rolling_resistance_is_pure_friction():
    """A specimen with mu=0 and mu_r>0 slips exactly like one with mu=mu_r, mu_r=0."""
    pure_roll = _pair_force(0.0, 0.35)
    pure_slide = _pair_force(0.35, 0.0)
    assert torch.allclose(pure_roll, pure_slide, atol=1e-9, rtol=1e-12)


def test_no_rotational_state_is_used():
    """The surrogate cannot be a torque: RidgeAssembly carries no angular DOF."""
    m = RidgeMaterial(freeze_bond_fraction=0.0)
    pos = torch.tensor([[0.0, 0.0], [1.4, 0.0]], dtype=torch.float64)
    radii = torch.tensor([0.75, 0.75], dtype=torch.float64)
    mass = torch.tensor([m.particle_mass, m.particle_mass], dtype=torch.float64)
    asm = RidgeAssembly(pos, radii, mass, torch.empty((0, 2), dtype=torch.int64),
                        material=m, planes=[])
    for name in ("omega", "ang_vel", "theta", "orientation", "inertia"):
        assert not hasattr(asm, name), f"unexpected rotational attribute {name!r}"


def test_tangential_saturation_is_mu_plus_mu_r_times_normal_force():
    """At full slip |F_t| -> (mu + mu_r) * F_n."""
    mu, mu_r = 0.55, 0.25
    f = _pair_force(mu, mu_r, v_t=50.0, overlap=0.1)
    fx, fy = float(f[0, 0]), float(f[0, 1])
    fn = abs(fx)
    # k_n * overlap, damping negligible at tanh -> 1 saturation
    analytic_fn = 1.0e7 * 0.1
    assert fn == pytest.approx(analytic_fn, rel=0.05)
    assert abs(fy) == pytest.approx((mu + mu_r) * fn, rel=0.10)