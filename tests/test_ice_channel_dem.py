"""Tests for the ice-floe channel DEM (:mod:`tensordem.ice_channel`).

Positive anchors (Archimedes draught, undamped energy conservation, Keinonen
contact anchor, momentum balance, determinism) plus explicit negative cases
(invalid coverage, zero-size particles, non-floating ice, invalid timestep,
invalid contact parameters).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

torch.set_num_threads(1)

from tensordem.ice_channel import (
    ContactParameters,
    HullGeometry,
    IceChannelSimulation,
    WaterEnvironment,
    analytic_equilibrium_draught,
    analytic_heave_period,
    contact_pressure_at,
    contact_stiffness,
    heave_potential,
    metacentric_height,
    capsize_restoring_moment,
    submerged_draught,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = REPO_ROOT / "artifacts" / "dem_channel.json"

CALM = WaterEnvironment(wind_velocity=(0.0, 0.0), current_velocity=(0.0, 0.0))


# ---------------------------------------------------------------------------
# hydrostatics
# ---------------------------------------------------------------------------

def test_single_floe_draught_matches_archimedes():
    """A released floe settles to d = (rho_i/rho_w) t within 1 %."""
    water = WaterEnvironment()
    radius, thickness = 5.0, 0.7
    sim = IceChannelSimulation([radius], [thickness], water=water, dt=1.0e-2)
    sim.z = torch.tensor([0.2 * thickness], dtype=sim.dtype)
    sim.run(700, record_every=10 ** 9)
    draught = float(submerged_draught(sim.z, sim.thickness)[0])
    reference = analytic_equilibrium_draught(water.ice_density, water.water_density, thickness)
    assert abs(draught - reference) / reference < 0.01


def test_submerged_draught_clipping():
    """Draught saturates at 0 (dry) and t (fully submerged)."""
    assert float(submerged_draught(0.5, 0.4)) == 0.0
    assert float(submerged_draught(-0.5, 0.4)) == pytest.approx(0.4)
    assert float(submerged_draught(0.0, 0.4)) == pytest.approx(0.2)


def test_flat_floe_is_metacentrically_stable():
    """GM > 0 for a flat-lying floe, and the capsizing moment opposes the tilt."""
    water = WaterEnvironment()
    thickness, radius = 0.6, 4.0
    z_eq = 0.5 * thickness - analytic_equilibrium_draught(
        water.ice_density, water.water_density, thickness)
    gm = metacentric_height(z_eq, radius, thickness, water.ice_density, water.water_density)
    assert gm > 0.5, gm
    m_pos = float(capsize_restoring_moment(0.1, z_eq, radius, thickness,
                                           water.ice_density, water.water_density,
                                           water.gravity))
    m_neg = float(capsize_restoring_moment(-0.1, z_eq, radius, thickness,
                                           water.ice_density, water.water_density,
                                           water.gravity))
    assert m_pos < 0.0 < m_neg


# ---------------------------------------------------------------------------
# energy conservation regression
# ---------------------------------------------------------------------------

def test_undamped_free_oscillation_energy_drift_below_one_percent():
    """Undamped heave, 10 periods, no drag: relative energy drift < 1 %."""
    water = WaterEnvironment(wind_velocity=(0.0, 0.0), current_velocity=(0.0, 0.0),
                             heave_damping_ratio=0.0, tilt_damping_ratio=0.0)
    radius, thickness = 5.0, 0.5
    sim = IceChannelSimulation([radius], [thickness], water=water, dt=1.0e-2,
                               integrator="verlet")
    z_eq = 0.5 * thickness - analytic_equilibrium_draught(
        water.ice_density, water.water_density, thickness)
    sim.z = torch.tensor([z_eq + 0.02 * thickness], dtype=sim.dtype)
    sim.vz = torch.tensor([0.0], dtype=sim.dtype)
    period = analytic_heave_period(radius, thickness, water.ice_density,
                                   water.water_density, water.gravity)
    n_steps = int(round(10.0 * period / sim.dt))
    e0 = float(sim.heave_energy()[0])
    worst = 0.0
    for _ in range(n_steps):
        sim.step()
        worst = max(worst, abs(float(sim.heave_energy()[0]) / e0 - 1.0))
    assert worst < 0.01, worst


def test_heave_potential_gradient_is_minus_buoyancy():
    """dU/dz = -F_net, i.e. the energy used by the drift test is the physical one."""
    water = WaterEnvironment()
    thickness, radius = 0.6, 3.0
    mass = water.ice_density * math.pi * radius ** 2 * thickness
    h = 1.0e-6
    for z in (-0.25, -0.1, 0.0, 0.15):
        up = heave_potential(z + h, radius, thickness, water.ice_density,
                             water.water_density, water.gravity)
        down = heave_potential(z - h, radius, thickness, water.ice_density,
                               water.water_density, water.gravity)
        force = -(up - down) / (2.0 * h)
        buoy = water.water_density * water.gravity * math.pi * radius ** 2 * \
            submerged_draught(z, thickness)
        assert force == pytest.approx(buoy - mass * water.gravity, rel=1e-6, abs=1e-3)


# ---------------------------------------------------------------------------
# contact law and its anchor
# ---------------------------------------------------------------------------

def test_keinonen_contact_anchor_reproduces_crushing_pressure():
    """p(delta_ref) = sigma_c and the assembled pair force = k_n delta_ref."""
    contact = ContactParameters()
    r_eff, t_eff = 5.0, 0.5
    delta_ref = contact.reference_penetration_fraction * 2.0 * r_eff
    k_n = contact_stiffness(r_eff, t_eff, contact)
    assert contact_pressure_at(delta_ref, r_eff, t_eff, contact) == pytest.approx(
        contact.crushing_strength, rel=1e-12)
    sim = IceChannelSimulation([2.0 * r_eff, 2.0 * r_eff], [t_eff, t_eff],
                               contact=contact, hull=HullGeometry(x_start=1.0e6),
                               water=CALM)
    sim.pos = torch.tensor([[0.0, 0.0], [4.0 * r_eff - delta_ref, 0.0]],
                           dtype=sim.dtype)
    force, _, _ = sim.net_forces(0.0)
    assert abs(float(force[0, 0])) == pytest.approx(k_n * delta_ref, rel=1e-9)
    # symmetry: equal and opposite
    assert float(force[0, 0]) == pytest.approx(-float(force[1, 0]), rel=1e-12)


def test_contact_force_vanishes_without_overlap():
    """Separated floes feel no contact force (no spurious cohesion)."""
    sim = IceChannelSimulation([5.0, 5.0], [0.5, 0.5], hull=HullGeometry(x_start=1.0e6),
                               water=CALM)
    sim.pos = torch.tensor([[0.0, 0.0], [11.0, 0.0]], dtype=sim.dtype)
    force, torque, _ = sim.net_forces(0.0)
    assert float(force.abs().max()) == 0.0
    assert float(torque.abs().max()) == 0.0


def test_pair_contact_impulse_is_equal_and_opposite():
    """Internal contacts carry no net impulse (momentum stays inside the pair)."""
    sim = IceChannelSimulation([3.0, 4.0], [0.6, 0.5], hull=HullGeometry(x_start=1.0e6),
                               water=CALM)
    sim.pos = torch.tensor([[0.0, 0.0], [6.8, 0.4]], dtype=sim.dtype)
    sim.vel = torch.tensor([[1.0, 0.2], [-0.5, -0.1]], dtype=sim.dtype)
    idx, f_pair, _, _, _ = sim._floe_floe_contacts()
    assert idx.shape == (1, 2)
    normal = sim.pos[0] - sim.pos[1]
    normal = normal / torch.linalg.norm(normal)
    # the force on floe 0 points away from floe 1 (plus a small friction part)
    assert float((f_pair[0] * normal).sum()) > 0.9 * float(torch.linalg.norm(f_pair[0]))
    # centre forces sum to the external (drag) part only: contacts are internal
    force, _, _ = sim.net_forces(0.0)
    assert float(force.sum(dim=0).abs().max()) < 1.0e-2 * float(force.abs().max())


# ---------------------------------------------------------------------------
# wind / current drag
# ---------------------------------------------------------------------------

def test_wind_drag_accelerates_floe_downwind():
    """A still floe in a steady wind must accelerate along the wind direction."""
    water = WaterEnvironment(wind_velocity=(10.0, 0.0), current_velocity=(0.0, 0.0),
                             heave_damping_ratio=0.0)
    sim = IceChannelSimulation([4.0], [0.6], water=water, hull=HullGeometry(x_start=1.0e6),
                               dt=0.01)
    sim.run(50, record_every=10 ** 9)
    assert float(sim.vel[0, 0]) > 0.0
    assert float(sim.vel[0, 0]) < 10.0
    assert abs(float(sim.vel[0, 1])) < 1e-12


# ---------------------------------------------------------------------------
# field construction and the cleared-width metric
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("concentration,domain", [(0.6, (60.0, 30.0)),
                                                  (0.8, (120.0, 60.0))])
def test_field_matches_requested_concentration_without_overlap(concentration, domain):
    """Realised coverage equals the requested tenths; no initial overlap.

    The 8/10 case needs enough floes across the channel (production geometry):
    a lattice truncated to a few cells cannot reach 0.8 without overlap, which
    the generator reports honestly through ``realised_concentration`` and
    ``max_initial_overlap``.
    """
    sim = IceChannelSimulation.from_field(concentration, domain_length=domain[0],
                                          domain_width=domain[1], seed=3, dt=0.01,
                                          hull=HullGeometry(x_start=-16.0))
    stats = sim.field_statistics()
    assert abs(stats["realised_concentration"] - concentration) < 0.02
    assert stats["max_initial_overlap"] < 0.1
    assert stats["floe_count"] > 5


def test_cleared_width_metric_definition():
    """Cleared width = 2 * min lateral clearance of the floes in the corridor."""
    sim = IceChannelSimulation([2.0, 3.0, 2.0], [0.5, 0.5, 0.5],
                               hull=HullGeometry(length=20.0, beam=10.0, y_position=0.0,
                                                 x_start=-10.0),
                               water=CALM)
    # corridor behind the hull at t=0 is x in [-30, -10]; y clearances 6-2=4 and -9-3
    sim.pos = torch.tensor([[-20.0, 6.0], [-15.0, -9.0], [-20.0, 1.5]], dtype=sim.dtype)
    assert sim.cleared_width() == pytest.approx(0.0)   # third floe blocks the track
    sim.pos[2, 1] = 40.0                               # move the blocker far away
    assert sim.cleared_width() == pytest.approx(8.0)


def test_same_seed_two_runs_are_bit_identical():
    kwargs = dict(domain_length=40.0, domain_width=20.0, seed=17, dt=0.01,
                  hull=HullGeometry(x_start=-20.0))
    a = IceChannelSimulation.from_field(0.7, **kwargs)
    b = IceChannelSimulation.from_field(0.7, **kwargs)
    assert a.state_digest() == b.state_digest()
    a.run(60, record_every=10 ** 9)
    b.run(60, record_every=10 ** 9)
    assert a.state_digest() == b.state_digest()
    # a different seed must not collide
    c = IceChannelSimulation.from_field(0.7, domain_length=40.0, domain_width=20.0,
                                        seed=18, dt=0.01,
                                        hull=HullGeometry(x_start=-20.0))
    assert c.state_digest() != a.state_digest()


def test_hull_transit_pushes_floes_and_clears_a_corridor():
    """The prescribed hull must move floes and leave a corridor wider than 0."""
    water = WaterEnvironment(wind_velocity=(0.0, 0.0), current_velocity=(0.0, 0.0))
    sim = IceChannelSimulation.from_field(0.7, domain_length=48.0, domain_width=24.0,
                                          seed=5, dt=1.0e-2, water=water,
                                          hull=HullGeometry(length=24.0, beam=10.0,
                                                            speed=2.0, y_position=12.0,
                                                            x_start=-12.0))
    start = sim.pos.clone()
    sim.run(int(round(12.0 / sim.dt)), record_every=10 ** 9)
    moved = torch.linalg.norm(sim.pos - start, dim=1)
    assert float(moved.max()) > 1.0
    width = sim.cleared_width()
    assert width >= 0.0 and math.isfinite(width)


# ---------------------------------------------------------------------------
# negative cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_negative_or_over_unity_coverage_rejected(bad):
    with pytest.raises(ValueError):
        IceChannelSimulation.from_field(bad, domain_length=40.0, domain_width=20.0,
                                        seed=1, dt=0.01)


@pytest.mark.parametrize("bad_radius", [0.0, -1.0, float("nan")])
def test_zero_size_or_invalid_particle_rejected(bad_radius):
    """Zero-size / negative / NaN particles must be refused, not silently run."""
    with pytest.raises(ValueError):
        IceChannelSimulation([5.0, bad_radius], [0.5, 0.5])


@pytest.mark.parametrize("bad_thickness", [0.0, -0.2])
def test_zero_thickness_rejected(bad_thickness):
    with pytest.raises(ValueError):
        IceChannelSimulation([5.0], [bad_thickness])


def test_non_floating_ice_rejected():
    with pytest.raises(ValueError):
        WaterEnvironment(ice_density=1100.0, water_density=1025.0)


@pytest.mark.parametrize("bad_dt", [0.0, -0.01])
def test_invalid_timestep_rejected(bad_dt):
    with pytest.raises(ValueError):
        IceChannelSimulation([5.0], [0.5], dt=bad_dt)


def test_invalid_contact_parameters_rejected():
    with pytest.raises(ValueError):
        ContactParameters(crushing_strength=0.0)
    with pytest.raises(ValueError):
        ContactParameters(reference_penetration_fraction=0.0)
    with pytest.raises(ValueError):
        ContactParameters(friction=-0.1)


def test_empty_particle_set_rejected():
    with pytest.raises(ValueError):
        IceChannelSimulation([], [])


# ---------------------------------------------------------------------------
# artifact audit chain
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ARTIFACT.exists(), reason="benchmark artifact not generated")
def test_artifact_reports_all_criteria_passing():
    report = json.loads(ARTIFACT.read_text())
    criteria = report["criteria"]
    assert criteria["energy_drift_rel"] < 0.01
    assert criteria["draught_rel_error"] < 0.01
    assert criteria["determinism_bit_identical"] is True
    assert criteria["contact_anchor_pressure_rel_error"] < 1e-6
    assert criteria["all_pass"] is True
    for key in ("6_10", "7_10", "8_10"):
        scenario = report["channel_scenarios"][key]
        assert scenario["field"]["realised_concentration"] > 0.5
        assert len(scenario["cleared_width_history"]) > 1