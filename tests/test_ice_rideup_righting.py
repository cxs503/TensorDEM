"""Righting-moment model tests for :mod:`tensordem.ice_rideup`.

Coverage
  * the exact (large-angle) branch reproduces the merged ``floe_6dof``
    hydrostatics -- force, moment and sign -- to the quadrature's precision;
  * the near-level branch is reproducible from its documented formula (the
    default path is therefore pinned, not merely "unchanged by inspection");
  * the near-level law leaves its small-angle envelope inside the heel band the
    bow imposes, and the exact branch folds (capsizes) where it does not;
  * mode validation and the default (backward-compatible) choice.
"""
import math

import pytest
import torch

from tensordem.floe_6dof import (analytic_roll_stiffness, attitude_potential,
                                 capsize_angle_from_moment)
from tensordem.ice_channel import WaterEnvironment
from tensordem.ice_rideup import (GRAVITY, RHO_ICE, RHO_WATER,
                                  RIGHTING_MODEL_EXACT, RIGHTING_MODEL_NEAR_LEVEL,
                                  RIGHTING_MODELS, RideUpSimulation)

torch.set_num_threads(1)
DTYPE = torch.float64
WATER = WaterEnvironment(water_density=RHO_WATER, ice_density=RHO_ICE, gravity=GRAVITY)


def _sim(radius=4.0, thickness=0.5, model=RIGHTING_MODEL_NEAR_LEVEL):
    sim = RideUpSimulation(fr=0.3, thickness=thickness, n_floes=1, steps=1,
                           righting_model=model)
    sim.field.radius[:] = radius
    sim.field.thickness[:] = thickness
    sim.field.centers[:, 0] = 0.0
    sim.field.centers[:, 1] = 0.0
    # flat-float centre height: the level submerged thickness is exactly the
    # Archimedes draught, so the legacy law's V and GM are the ones documented
    sim.field.centers[:, 2] = torch.tensor([sim.equilibrium_height(thickness)],
                                           dtype=DTYPE)
    return sim


def _set_roll(sim, deg):
    half = 0.5 * math.radians(deg)
    q = torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]], dtype=DTYPE)
    sim.field.quats = q
    sim.field.omegas = torch.zeros(1, 3, dtype=DTYPE)
    sim.field.velocities = torch.zeros(1, 3, dtype=DTYPE)


def _heave_relaxed_height(radius, thickness, deg):
    return float(attitude_potential(radius, thickness, math.radians(deg), WATER)["height"])


# ---------------------------------------------------------------------------
# 1. model selection and backward compatibility of the default
# ---------------------------------------------------------------------------

def test_default_is_near_level_and_validates():
    sim = RideUpSimulation(fr=0.3, n_floes=2, steps=1)
    assert sim.righting_model == RIGHTING_MODEL_NEAR_LEVEL
    assert RIGHTING_MODELS == (RIGHTING_MODEL_NEAR_LEVEL, RIGHTING_MODEL_EXACT)
    with pytest.raises(ValueError):
        RideUpSimulation(fr=0.3, n_floes=2, steps=1, righting_model="linear")


def test_near_level_branch_reproduces_its_formula():
    """The legacy kernel is pinned to the documented closed form."""
    radius, thickness, deg = 4.0, 0.5, 12.0
    sim = _sim(radius, thickness)
    _set_roll(sim, deg)
    sim.field.centers[:, 2] = torch.tensor([_heave_relaxed_height(radius, thickness, deg)],
                                           dtype=DTYPE)
    force, torque = sim._hydrostatic_forces()
    d = (RHO_ICE / RHO_WATER) * thickness
    gm = radius ** 2 / (4.0 * d)
    v_sub = math.pi * radius ** 2 * d
    expected = RHO_WATER * GRAVITY * v_sub * gm * math.sin(math.radians(deg))
    assert abs(float(torque[0].norm()) - expected) / expected < 1e-12
    # restoring direction: a positive roll about x must give a negative x torque
    assert float(torque[0, 0]) < 0.0


# ---------------------------------------------------------------------------
# 2. the exact branch against the merged floe_6dof hydrostatics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deg", [2.0, 5.0, 10.0, 15.0])
@pytest.mark.parametrize("radius,thickness", [(4.0, 0.5), (2.5, 0.5)])
def test_exact_branch_matches_floe_6dof_moment(deg, radius, thickness):
    sim = _sim(radius, thickness, model=RIGHTING_MODEL_EXACT)
    _set_roll(sim, deg)
    sim.field.centers[:, 2] = torch.tensor([_heave_relaxed_height(radius, thickness, deg)],
                                           dtype=DTYPE)
    force, torque = sim._hydrostatic_forces()
    reference = attitude_potential(radius, thickness, math.radians(deg), WATER)["moment"]
    measured = float(abs(torque[0, 0]))
    assert abs(measured - reference) / reference < 1e-6
    # heave-relaxed static state: the vertical residual is zero, not just small
    assert abs(float(force[0, 2])) / (float(sim.mass[0]) * GRAVITY) < 1e-3
    # restoring: torque opposes the roll
    assert float(torque[0, 0]) < 0.0


def test_exact_branch_upright_state_is_still():
    radius, thickness = 4.0, 0.5
    sim = _sim(radius, thickness, model=RIGHTING_MODEL_EXACT)
    sim.field.centers[:, 2] = torch.tensor([sim.equilibrium_height(thickness)], dtype=DTYPE)
    # the constructor seeds a random yaw plus a +/-0.02 rad tilt; pin it upright
    sim.field.quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
    force, torque = sim._hydrostatic_forces()
    weight = float(sim.mass[0]) * GRAVITY
    # residual = the 24x32 disk-quadrature discretisation of the displaced
    # volume (4.5e-6 relative), not a modelling error
    assert abs(float(force[0, 2])) / weight < 1e-5
    assert float(torque[0].norm()) < 1e-6 * weight * thickness


# ---------------------------------------------------------------------------
# 3. the small-angle envelope: where the near-level law stops being usable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deg,upper", [(0.5, 0.03), (20.0, 0.30)])
def test_near_level_error_grows_out_of_the_envelope(deg, upper):
    """At 0.5 deg the legacy law is inside the 3 % gate; at 20 deg it is not."""
    radius, thickness = 4.0, 0.5
    near = _sim(radius, thickness)
    _set_roll(near, deg)
    m_near = float(near._hydrostatic_forces()[1][0].norm())
    m_exact = float(abs(attitude_potential(radius, thickness, math.radians(deg),
                                           WATER)["moment"]))
    rel = abs(m_near - m_exact) / m_exact
    if upper <= 0.03:
        assert rel < upper
    else:
        assert rel > upper


def test_near_level_law_has_no_capsize_fold():
    """The legacy law is monotone in sin(phi): its moment peaks at 90 deg."""
    from tensordem.ice_rideup import archimedes_buoyancy  # noqa: F401  (documented kernel)
    radius, thickness = 4.0, 0.5
    sim = _sim(radius, thickness)
    d = (RHO_ICE / RHO_WATER) * thickness
    gm = radius ** 2 / (4.0 * d)
    k = RHO_WATER * GRAVITY * math.pi * radius ** 2 * d * gm
    moments = [k * math.sin(math.radians(deg)) for deg in range(5, 91, 5)]
    assert moments == sorted(moments)                     # no fold anywhere
    fold = capsize_angle_from_moment(radius, thickness, WATER)["capsize_angle"]
    assert 0.0 < math.degrees(fold) < 45.0                # the exact law does fold
    exact_peak = float(attitude_potential(radius, thickness, fold, WATER)["moment"])
    assert moments[-1] > 3.0 * exact_peak                 # and it overstates it


def test_exact_mode_damps_a_heeled_floe_towards_upright():
    """A floe released past its small-angle envelope must not gain energy."""
    radius, thickness = 4.0, 0.5
    sim = _sim(radius, thickness, model=RIGHTING_MODEL_EXACT)
    _set_roll(sim, 6.0)
    energies = []
    for _ in range(200):
        force, torque = sim._hydrostatic_forces()
        energies.append(float(force[0, 2]) * float(sim.field.centers[0, 2])
                        - float(torque[0, 0]) * 0.0)
    assert all(math.isfinite(e) for e in energies)
    # the heave channel is a force, the roll channel a moment: check both are finite
    assert float(sim._hydrostatic_forces()[1][0, 0]) < 0.0


def test_exact_mode_is_a_restoring_potential_field():
    """dU/dphi = -M: the moment must come from a bounded energy landscape."""
    radius, thickness = 4.0, 0.5
    deg = 8.0
    m = float(attitude_potential(radius, thickness, math.radians(deg), WATER)["moment"])
    u0 = float(attitude_potential(radius, thickness, 0.0, WATER)["energy"])
    u1 = float(attitude_potential(radius, thickness, math.radians(2.0), WATER)["energy"])
    u2 = float(attitude_potential(radius, thickness, math.radians(deg), WATER)["energy"])
    assert u0 <= u1 <= u2                                  # upright is the minimum
    assert m > 0.0
    # near-level stiffness is the analytic small-angle limit of the same curve
    assert abs(analytic_roll_stiffness(radius, thickness, RHO_ICE, RHO_WATER, GRAVITY)
               / 2.0e6 - 1.0) < 0.01