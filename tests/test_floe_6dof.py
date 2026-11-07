"""Tests for :mod:`tensordem.floe_6dof` (IN22-N02).

The suite covers the hydrostatics of a tilted disk (including the negative
cases the campaign asks for), quaternion algebra and integration, the two
capsize-angle routes and their consistency, the H1 regression lock of the
tilt-locked mode, and the numerical invariants of the 6-DOF integrator.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.floe_6dof import (
    CapsizeParameters,
    SixDofFloeSimulation,
    analytic_metacentric_height,
    analytic_roll_stiffness,
    capsize_angle_from_hessian,
    capsize_angle_from_moment,
    capsize_curve,
    disk_quadrature,
    heave_equilibrium_height,
    integrate_quaternion,
    quat_from_axis_angle,
    quat_from_euler,
    quat_identity,
    quat_multiply,
    quat_normalize,
    quat_norm_error,
    quat_tilt_angle,
    quat_to_matrix,
    submerged_hydrostatics,
)
from tensordem.ice_channel import (
    ContactParameters,
    HullGeometry,
    IceChannelSimulation,
    WaterEnvironment,
)

WATER = WaterEnvironment()
QUAD = disk_quadrature(16, 24)
FAST_QUAD = disk_quadrature(12, 16)


# ---------------------------------------------------------------------------
# Quadrature and hydrostatics
# ---------------------------------------------------------------------------

def test_disk_quadrature_integrates_moments():
    quad = disk_quadrature(16, 24)
    assert float(quad.weights.sum()) == pytest.approx(math.pi, rel=1e-14)
    second = float((quad.weights * quad.points[:, 0] ** 2).sum())
    assert second == pytest.approx(math.pi / 4.0, rel=1e-13)


def test_flat_submerged_volume_matches_archimedes():
    radius, thickness = 4.0, 0.4
    state = submerged_hydrostatics([radius], [thickness], quat_identity(1),
                                   [0.0], WATER, QUAD)
    analytic = 0.5 * math.pi * radius ** 2 * thickness
    assert float(state.volume[0]) == pytest.approx(analytic, rel=1e-13)
    assert abs(float(state.buoyancy[0, 2])
               - WATER.water_density * WATER.gravity * analytic) < 1e-9


def test_flat_centre_of_buoyancy_is_quarter_thickness_below_centre():
    radius, thickness = 4.0, 0.4
    state = submerged_hydrostatics([radius], [thickness], quat_identity(1),
                                   [0.0], WATER, QUAD)
    assert float(state.centroid_body[0, 2]) == pytest.approx(-0.25 * thickness,
                                                             rel=1e-10)
    assert abs(float(state.centroid_body[0, 0])) < 1e-12
    assert abs(float(state.centroid_body[0, 1])) < 1e-12


def test_heave_equilibrium_matches_closed_form_draught():
    radius, thickness = 4.0, 0.4
    z = heave_equilibrium_height(radius, thickness, quat_identity(1), WATER, QUAD)
    expected = 0.5 * thickness - (WATER.ice_density / WATER.water_density) * thickness
    assert z == pytest.approx(expected, abs=1e-12)


def test_equilibrium_buoyancy_gravity_balance_below_1e_10():
    radius, thickness = 4.0, 0.4
    z = heave_equilibrium_height(radius, thickness, quat_identity(1), WATER, QUAD)
    state = submerged_hydrostatics([radius], [thickness], quat_identity(1), [z],
                                  WATER, QUAD)
    residual = abs(float(state.buoyancy[0, 2] + state.weight[0, 2])) / abs(
        float(state.weight[0, 2]))
    assert residual < 1e-10


def test_upside_down_floe_is_energetically_identical():
    radius, thickness = 3.0, 0.35
    flat = submerged_hydrostatics([radius], [thickness], quat_identity(1),
                                 [-0.1], WATER, QUAD)
    flipped = submerged_hydrostatics([radius], [thickness],
                                     quat_from_euler([math.pi], [0.0], [0.0]),
                                     [-0.1], WATER, QUAD)
    assert float(flipped.volume[0]) == pytest.approx(float(flat.volume[0]), rel=1e-6)
    assert float(flipped.energy[0]) == pytest.approx(float(flat.energy[0]), rel=1e-6)


def test_small_angle_roll_stiffness_matches_metacentric_formula():
    radius, thickness = 4.0, 0.4
    phi = 1e-4
    q = quat_from_euler([phi], [0.0], [0.0])
    z = heave_equilibrium_height(radius, thickness, q, WATER, QUAD)
    state = submerged_hydrostatics([radius], [thickness], q, [z], WATER, QUAD)
    numeric = float(-state.moment[0, 0]) / phi
    analytic = analytic_roll_stiffness(radius, thickness, WATER.ice_density,
                                       WATER.water_density, WATER.gravity)
    assert numeric == pytest.approx(analytic, rel=1e-4)


def test_metacentric_height_formula_limits():
    # BM dominates for a wide, thin disk; GM must stay positive
    assert analytic_metacentric_height(4.0, 0.4, 917.0, 1025.0) > 0.0
    # a thicker floe has a smaller BM term
    assert (analytic_metacentric_height(4.0, 0.2, 917.0, 1025.0)
            > analytic_metacentric_height(4.0, 0.8, 917.0, 1025.0))


def test_righting_moment_is_positive_until_the_edge_on_pose():
    radius, thickness = 3.0, 0.4
    curve = capsize_curve(radius, thickness, WATER, tilt_count=91,
                          tilt_max=0.5 * math.pi, quad=FAST_QUAD)
    assert float(curve.moment[5]) > 0.0            # ~5 deg, restoring
    assert abs(float(curve.moment[-1])) < 1e-6 * float(curve.moment.max())


# ---------------------------------------------------------------------------
# Capsize angle: two routes must agree (criterion a)
# ---------------------------------------------------------------------------

def test_capsize_angle_routes_agree_within_two_percent():
    thickness = 0.4
    moment = capsize_angle_from_moment(3.0, thickness, WATER, FAST_QUAD,
                                       scan_points=91)
    hessian = capsize_angle_from_hessian(3.0, thickness, WATER, FAST_QUAD,
                                         scan_points=181)
    rel = abs(moment["capsize_angle"] - hessian["capsize_angle"]) / hessian["capsize_angle"]
    assert rel < 0.02
    assert 0.0 < moment["capsize_angle"] < 0.5 * math.pi


def test_fold_angle_shifts_to_larger_heel_for_thicker_floes():
    # a thin disk leaves the flat-face regime at phi ~ atan(d/R) (a few degrees)
    # and therefore folds early; a thicker disk keeps a wall-sided waterplane
    # over a much wider heel range, so its righting-moment fold moves outwards
    thin = capsize_angle_from_moment(4.0, 0.2, WATER, FAST_QUAD, scan_points=91)
    thick = capsize_angle_from_moment(4.0, 0.8, WATER, FAST_QUAD, scan_points=91)
    assert thick["capsize_angle"] > thin["capsize_angle"]
    assert 0.0 < thin["capsize_angle"] < 0.5 * math.pi
    assert 0.0 < thick["capsize_angle"] < 0.5 * math.pi


def test_capsize_angle_rejects_bad_scan_bounds():
    with pytest.raises(ValueError):
        capsize_angle_from_moment(3.0, 0.4, WATER, FAST_QUAD, upper=0.0)
    with pytest.raises(ValueError):
        capsize_angle_from_hessian(3.0, 0.4, WATER, FAST_QUAD, scan_points=5)


# ---------------------------------------------------------------------------
# Quaternions
# ---------------------------------------------------------------------------

def test_quat_identity_and_matrix():
    q = quat_identity(2)
    assert torch.allclose(quat_to_matrix(q), torch.eye(3, dtype=torch.float64).expand(2, 3, 3))


def test_quat_roll_of_ninety_degrees_maps_axes():
    q = quat_from_euler([0.5 * math.pi], [0.0], [0.0])
    matrix = quat_to_matrix(q)[0]
    assert matrix[2, 2] == pytest.approx(0.0, abs=1e-12)
    assert quat_tilt_angle(q)[0] == pytest.approx(0.5 * math.pi, abs=1e-9)


def test_quat_multiply_matches_matrix_product():
    a = quat_from_euler([0.3], [0.0], [0.0])
    b = quat_from_euler([0.0], [0.0], [0.7])
    product = quat_multiply(a, b)
    assert torch.allclose(quat_to_matrix(product),
                          quat_to_matrix(a) @ quat_to_matrix(b), atol=1e-12)


def test_integrate_quaternion_preserves_norm_over_many_steps():
    q = quat_identity(1)
    omega = torch.tensor([[0.7, -1.3, 2.1]])
    for _ in range(2000):
        q = integrate_quaternion(q, omega, 4e-3)
    assert float(quat_norm_error(q)) < 1e-12


def test_integrate_quaternion_matches_exact_rotation():
    q = quat_identity(1)
    omega = torch.tensor([[0.0, 0.0, 1.0]])
    dt = 0.25
    q = integrate_quaternion(q, omega, dt)
    matrix = quat_to_matrix(q)[0]
    assert matrix[0, 0] == pytest.approx(math.cos(dt), abs=1e-12)
    assert matrix[1, 0] == pytest.approx(math.sin(dt), abs=1e-12)


def test_quat_normalize_rejects_zero_norm():
    with pytest.raises(ValueError):
        quat_normalize(torch.zeros((1, 4), dtype=torch.float64))


def test_quat_normalize_rejects_non_finite():
    bad = torch.tensor([[1.0, float("nan"), 0.0, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError):
        quat_normalize(bad)


def test_quat_from_axis_angle_rejects_zero_axis():
    with pytest.raises(ValueError):
        quat_from_axis_angle(torch.zeros((1, 3), dtype=torch.float64),
                             torch.tensor([0.5], dtype=torch.float64))


def test_integrate_quaternion_rejects_batch_mismatch():
    with pytest.raises(ValueError):
        integrate_quaternion(quat_identity(2), torch.zeros((3, 3), dtype=torch.float64), 1e-3)


def test_integrate_quaternion_rejects_non_finite_dt():
    with pytest.raises(ValueError):
        integrate_quaternion(quat_identity(1), torch.ones((1, 3), dtype=torch.float64),
                             float("inf"))


# ---------------------------------------------------------------------------
# Simulation: regression lock, invariants, negative cases
# ---------------------------------------------------------------------------

def small_field():
    radius = [3.0, 3.0, 3.0]
    thickness = [0.4, 0.4, 0.4]
    pos = torch.tensor([[0.0, 0.0], [5.4, 0.4], [0.2, 5.6]], dtype=torch.float64)
    return radius, thickness, pos


def test_tilt_locked_digest_reproduces_h1_bit_for_bit():
    radius, thickness, pos = small_field()
    reference = IceChannelSimulation(radius, thickness, dt=8e-3)
    locked = SixDofFloeSimulation(radius, thickness, mode="tilt_locked", dt=8e-3)
    reference.pos = pos.clone()
    locked.pos = pos.clone()
    for _ in range(60):
        reference.step()
        locked.step()
    assert locked.state_digest() == reference.state_digest()


def test_tilt_locked_rejects_non_trivial_rotation_state():
    radius, thickness, _ = small_field()
    sim = SixDofFloeSimulation(radius, thickness, mode="tilt_locked")
    sim.omega_body = torch.full((sim.n, 3), 1e-3, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        sim.state_digest()


def test_six_dof_quaternion_norm_stays_below_1e_12():
    radius, thickness, pos = small_field()
    sim = SixDofFloeSimulation(radius, thickness, mode="six_dof", dt=4e-3,
                               capsize_angles=[0.25, 0.25, 0.25])
    sim.pos = pos.clone()
    sim.set_flat_equilibrium()
    sim.omega_body = torch.full((sim.n, 3), 0.4, dtype=torch.float64)
    out = sim.run_transit(0.4, record_every=10)
    assert out["worst_quaternion_norm_error"] < 1e-12


def test_six_dof_buoyancy_balance_below_1e_10_at_equilibrium():
    radius, thickness, pos = small_field()
    sim = SixDofFloeSimulation(radius, thickness, mode="six_dof",
                               capsize_angles=[0.25, 0.25, 0.25])
    sim.pos = pos.clone()
    sim.set_flat_equilibrium()
    assert sim.hydrostatic_balance_error() < 1e-10


def test_six_dof_free_floe_stays_flat_without_moment():
    sim = SixDofFloeSimulation([3.0], [0.4], mode="six_dof", capsize_angles=[0.25])
    sim.pos = torch.zeros((1, 2), dtype=torch.float64)
    sim.hull = HullGeometry(speed=0.0, x_start=-500.0)
    sim.set_flat_equilibrium()
    sim.water = WaterEnvironment(wind_velocity=(0.0, 0.0))
    out = sim.run_transit(0.5, record_every=25)
    assert out["max_tilt_deg"] < 1e-6
    assert out["capsized_fraction"] == 0.0


def test_capsize_flag_triggers_past_the_fold_angle():
    sim = SixDofFloeSimulation([3.0], [0.4], mode="six_dof", dt=4e-3,
                               capsize_angles=[0.1])
    sim.pos = torch.zeros((1, 2), dtype=torch.float64)
    sim.hull = HullGeometry(speed=0.0, x_start=-500.0)
    sim.water = WaterEnvironment(wind_velocity=(0.0, 0.0))
    sim.set_flat_equilibrium()
    sim.quat = quat_from_euler([0.3], [0.0], [0.0])   # 0.3 rad > 0.1 rad fold
    out = sim.run_transit(0.1, record_every=5)
    assert out["capsized_fraction"] == 1.0


def test_capsize_angles_must_be_positive():
    with pytest.raises(ValueError):
        SixDofFloeSimulation([3.0], [0.4], mode="six_dof", capsize_angles=[-0.1])


def test_capsize_angles_length_must_match_population():
    with pytest.raises(ValueError):
        SixDofFloeSimulation([3.0, 3.0], [0.4, 0.4], mode="six_dof",
                             capsize_angles=[0.2])


def test_non_positive_radius_rejected_and_inertia_stays_positive():
    with pytest.raises(ValueError):
        SixDofFloeSimulation([0.0], [0.4], mode="six_dof", capsize_angles=[0.2])
    with pytest.raises(ValueError):
        SixDofFloeSimulation([-1.0], [0.4], mode="six_dof", capsize_angles=[0.2])
    sim = SixDofFloeSimulation([3.0], [0.4], mode="six_dof", capsize_angles=[0.2])
    assert bool((sim.inertia_body > 0.0).all())


def test_illegal_submerged_volume_is_rejected():
    # non-finite centre height
    with pytest.raises(ValueError):
        submerged_hydrostatics([3.0], [0.4], quat_identity(1), [float("nan")],
                               WATER, FAST_QUAD)
    # non-positive thickness has no volume at all
    with pytest.raises(ValueError):
        submerged_hydrostatics([3.0], [0.0], quat_identity(1), [0.0], WATER, FAST_QUAD)


def test_never_floatable_density_is_rejected_by_metacentric_height():
    with pytest.raises(ValueError):
        analytic_metacentric_height(3.0, 0.4, 1025.0, 1025.0)


def test_negative_restitution_and_friction_are_rejected():
    with pytest.raises(ValueError):
        ContactParameters(friction=-0.2)
    with pytest.raises(ValueError):
        ContactParameters(normal_damping_ratio=-0.1)


def test_negative_submodel_parameters_are_rejected():
    with pytest.raises(ValueError):
        CapsizeParameters(shoulder_submergence=-0.1)
    with pytest.raises(ValueError):
        CapsizeParameters(roll_damping_ratio=-0.1)
    with pytest.raises(ValueError):
        CapsizeParameters(hull_pressure_height_fraction=1.5)


def test_projected_footprint_shrinks_when_the_floe_stands_on_edge():
    sim = SixDofFloeSimulation([3.0], [0.4], mode="six_dof", capsize_angles=[0.2])
    sim.pos = torch.tensor([[-10.0, 10.0]], dtype=torch.float64)
    sim.domain = (100.0, 40.0)
    sim.time = 0.0
    flat = sim.cleared_width()
    sim.quat = quat_from_euler([0.5 * math.pi], [0.0], [0.0])
    sim.tilt_angle = quat_tilt_angle(sim.quat)
    edge_on = sim.cleared_width()
    assert flat == pytest.approx(2.0 * (10.0 - 3.0), rel=1e-12)
    assert edge_on == pytest.approx(2.0 * (10.0 - 0.2), rel=1e-12)
    assert edge_on > flat


def test_six_dof_mode_changes_the_state_relative_to_tilt_locked():
    radius, thickness, pos = small_field()
    locked = SixDofFloeSimulation(radius, thickness, mode="tilt_locked", dt=8e-3)
    full = SixDofFloeSimulation(radius, thickness, mode="six_dof", dt=8e-3,
                                capsize_angles=[0.25, 0.25, 0.25])
    for sim in (locked, full):
        sim.pos = pos.clone()
        sim.set_flat_equilibrium()
    locked.run_transit(0.2, record_every=10)
    full.run_transit(0.2, record_every=10)
    assert locked.state_digest() != full.state_digest()
    assert float(quat_norm_error(full.quat)) < 1e-12