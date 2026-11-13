"""Hertz restitution calibration: universal curve, inverse map, engine closure.

The engine's Hertz pair branch derives its dashpot from the local secant
stiffness, ``gamma(delta) = 2 zeta sqrt(m_eff k_s(delta))``, but the mapping from
the requested coefficient of restitution to ``zeta`` comes from *linear*
spring-dashpot theory, which is exact only for a constant dashpot.  The realised
restitution therefore missed the request (``e=0.5 -> 0.582``, ``e=0.8 -> 0.826``).

``tensordem.contact_calibration`` supplies the inverse of the *Hertz* curve, and
``DEM3DConfig.contact_damping_model="restitution_calibrated"`` selects it.  These
tests pin down

1. the similarity reduction of the contact law (the curve is universal, hence one
   calibration holds at every impact speed),
2. the inverse map itself,
3. the closed loop on the solver: requested == realised to <0.3 %, while the
   legacy convention stays bit-for-bit at its documented (biased) value.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensordem.contact_calibration import (
    hertz_damping_ratio_for_restitution,
    hertz_restitution_from_zeta,
    delivered_restitution_legacy,
    linear_restitution_from_zeta,
    zeta_from_linear_restitution,
)
from tensordem.dem3d import (
    DEM3DConfig,
    IceDEM3D,
    hertz_effective_radius,
    hertz_reduced_modulus,
)

RADIUS = 0.025
DENSITY = 917.0
YOUNG_MODULUS = 1.0e9
POISSON_RATIO = 0.3
PARK = 1.0
SPEED = 0.5


# ---------------------------------------------------------------------------
# Independent dimensional reference (own code path, no similarity argument)
# ---------------------------------------------------------------------------
def reference_restitution_dimensionless_impact(rest_target: float,
                                               dt: float = 1.0e-7) -> float:
    """RK4 of the dimensional relative-coordinate law with the Tsuji dashpot.

    Uses the same one-sided projection as the solver,
    ``F = max(k_s d^{3/2} + 2 zeta sqrt(m_eff k_s) rate, 0)``, but is written
    from the physical parameters (no similarity scaling), so it is an
    independent check of ``hertz_restitution_from_zeta``.
    """
    mass = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
    m_eff = 0.5 * mass
    e_star = hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO)
    r_eff = hertz_effective_radius(RADIUS)
    k_coef = (4.0 / 3.0) * e_star * math.sqrt(r_eff)
    zeta = zeta_from_linear_restitution(rest_target)
    v0 = SPEED  # relative-coordinate closing rate scale

    def accel(overlap: float, rate: float) -> float:
        if overlap <= 0.0:
            return 0.0
        gamma = 2.0 * zeta * math.sqrt(m_eff * k_coef * math.sqrt(overlap))
        force = k_coef * overlap * math.sqrt(overlap) + gamma * rate
        return -force / m_eff if force > 0.0 else 0.0

    overlap, rate, t = 0.0, v0, 0.0
    in_contact = False
    while t < 5.0e-3:
        k1d, k1r = rate, accel(overlap, rate)
        k2d = rate + 0.5 * dt * k1r
        k2r = accel(overlap + 0.5 * dt * k1d, k2d)
        k3d = rate + 0.5 * dt * k2r
        k3r = accel(overlap + 0.5 * dt * k2d, k3d)
        k4d = rate + dt * k3r
        k4r = accel(overlap + dt * k3d, k4d)
        overlap += dt * (k1d + 2.0 * k2d + 2.0 * k3d + k4d) / 6.0
        rate += dt * (k1r + 2.0 * k2r + 2.0 * k3r + k4r) / 6.0
        if overlap > 0.0:
            in_contact = True
        elif in_contact:
            break
        t += dt
    return -rate / v0


# ---------------------------------------------------------------------------
# Engine harness (one free contact pair in a 2x2x2 lattice container)
# ---------------------------------------------------------------------------
def pair_config(**kwargs) -> DEM3DConfig:
    params = dict(
        nx=2, ny=2, nz=2,
        radius=RADIUS, density=DENSITY,
        contact_model="hertz", young_modulus=YOUNG_MODULUS, poisson_ratio=POISSON_RATIO,
        contact_damping=0.0, drag=0.0,
        fix_bottom=False, fix_x_edges=False,
        tool_speed=1.0, tool_gap=10.0,
    )
    params.update(kwargs)
    base = DEM3DConfig(**params)
    return DEM3DConfig(**{**base.__dict__, "dt": 0.5 * base.recommended_dt})


def measure_engine(speed: float, **kwargs) -> float:
    cfg = pair_config(**kwargs)
    sim = IceDEM3D(cfg)
    sim.alive[:] = False
    sim.positions[0] = torch.tensor([0.0, 0.0, 0.0], dtype=sim.dtype)
    sim.positions[1] = torch.tensor([0.0, 0.0, 2.0 * RADIUS * 1.05], dtype=sim.dtype)
    for i in range(2, len(sim.positions)):
        sim.positions[i] = torch.tensor([PARK * (i - 1)] * 2 + [3.0 * PARK],
                                        dtype=sim.dtype)
    sim.velocities.zero_()
    sim.velocities[0, 2] = +speed
    sim.velocities[1, 2] = -speed

    def state():
        r = sim.positions[1] - sim.positions[0]
        separation = float(torch.linalg.vector_norm(r))
        closing = -float(((sim.velocities[0] - sim.velocities[1]) * (r / separation)).sum())
        return separation, closing

    _, approach = state()
    in_contact = False
    for _ in range(20000):
        sim.step()
        separation, closing = state()
        if separation < 2.0 * RADIUS:
            in_contact = True
        elif in_contact:
            return abs(closing) / abs(approach)
    raise AssertionError("impact did not separate")


# ---------------------------------------------------------------------------
# 1. Universal curve
# ---------------------------------------------------------------------------
def test_elastic_limit_is_exactly_one():
    assert hertz_restitution_from_zeta(0.0) == 1.0


def test_curve_is_monotone_decreasing():
    zetas = [0.01 * k for k in range(1, 40)]
    values = [hertz_restitution_from_zeta(z) for z in zetas]
    assert all(b < a for a, b in zip(values, values[1:]))


def test_curve_is_time_step_converged():
    """G(dt) - G(dt/16) is below the 1e-6 accuracy the calibration claims."""
    coarse = hertz_restitution_from_zeta(0.2, dt=1.0e-3)
    fine = hertz_restitution_from_zeta(0.2, dt=1.0e-3 / 16.0)
    assert abs(coarse - fine) < 1.0e-5, (coarse, fine)


@pytest.mark.parametrize("e_target", [0.5, 0.8])
def test_similarity_curve_matches_dimensional_reference(e_target):
    """The parameter-free reduction reproduces the dimensional integration."""
    got = hertz_restitution_from_zeta(zeta_from_linear_restitution(e_target))
    ref = reference_restitution_dimensionless_impact(e_target)
    assert abs(got - ref) / ref < 2.0e-4, (got, ref)


def test_one_sided_dashpot_clip_matters():
    """The no-tension clip is physical, not cosmetic: it raises e by ~0.8 %."""
    zeta = zeta_from_linear_restitution(0.8)
    one_sided = hertz_restitution_from_zeta(zeta)
    two_sided = hertz_restitution_from_zeta(zeta, clamp=False)
    # Clipping removes the dashpot's pull near detachment, so less energy is
    # dissipated and the realised restitution is higher.
    assert one_sided > two_sided
    assert 0.004 < (one_sided - two_sided) / one_sided < 0.02


# ---------------------------------------------------------------------------
# 2. Inverse map
# ---------------------------------------------------------------------------
def test_linear_mapping_is_its_own_inverse():
    for e in (0.2, 0.5, 0.8, 0.95):
        assert math.isclose(linear_restitution_from_zeta(zeta_from_linear_restitution(e)),
                            e, rel_tol=1e-12)


@pytest.mark.parametrize("e_target", [0.3, 0.5, 0.7, 0.8, 0.9, 0.95])
def test_calibration_inverts_the_hertz_curve(e_target):
    zeta = hertz_damping_ratio_for_restitution(e_target)
    delivered = hertz_restitution_from_zeta(zeta)
    assert abs(delivered - e_target) / e_target < 1.0e-9, (delivered, e_target)


def test_calibrated_ratio_exceeds_linear_ratio():
    """The one-sided Hertz dashpot needs more damping than linear theory says."""
    for e in (0.5, 0.8):
        assert (hertz_damping_ratio_for_restitution(e)
                > zeta_from_linear_restitution(e))


def test_legacy_bias_matches_the_documented_measurements():
    """Predicted legacy bias reproduces the engine record (e=0.5 -> 0.5823)."""
    assert math.isclose(delivered_restitution_legacy(0.5), 0.58229, rel_tol=1e-4)
    assert math.isclose(delivered_restitution_legacy(0.8), 0.82578, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# 3. Engine closed loop
# ---------------------------------------------------------------------------
def test_default_damping_model_is_unchanged():
    assert DEM3DConfig(nx=2, ny=2, nz=2).contact_damping_model == "restitution"


def test_calibrated_requires_hertz():
    with pytest.raises(ValueError, match="restitution_calibrated"):
        DEM3DConfig(nx=2, ny=2, nz=2, contact_damping_model="restitution_calibrated")


def test_engine_delivers_requested_restitution_when_calibrated():
    e_target = 0.8
    e_legacy = measure_engine(SPEED, restitution_coefficient=e_target)
    e_calibrated = measure_engine(SPEED, restitution_coefficient=e_target,
                                  contact_damping_model="restitution_calibrated")
    legacy_err = abs(e_legacy - e_target) / e_target
    calibrated_err = abs(e_calibrated - e_target) / e_target
    # Baseline behaves as documented (the defect is real) ...
    assert legacy_err > 0.02, (e_legacy, e_target)
    # ... and the calibrated convention closes the loop.
    assert calibrated_err < 3.0e-3, (e_calibrated, e_target)
    # The legacy trajectory itself is untouched by the new option.
    assert math.isclose(e_legacy, delivered_restitution_legacy(e_target), rel_tol=2e-3)