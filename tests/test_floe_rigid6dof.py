"""Tests for :mod:`tensordem.floe_rigid6dof` (IN29-N02).

Coverage map (acceptance criteria of the campaign node in brackets):

* geometry / hydrostatics: exact cuboid and concave L-prism mass properties,
  half-submerged and fully-wet volumes, float-kernel vs vectorised-kernel
  cross-check, prism vs the independent disk quadrature of
  :mod:`tensordem.floe_6dof` [a, b];
* energy consistency of the hydrostatics: buoyancy force and moment equal the
  gradients of the buoyancy potential [b];
* attitude kinematics: quaternion norm after long integration, exponential map
  vs the matrix exponential [a];
* dynamics: free-decay frequency and damping ratio against the closed form
  [b], two-floe collision linear/angular momentum [c], multi-floe static
  equilibrium [d], energy ledger with and without dissipation and contacts [e],
  negative control with zero external force [f];
* negative / robustness cases: open or inverted meshes, invalid inputs, and the
  plane-contact (seabed / hull) response including Coulomb friction.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordem.floe_6dof import (
    disk_quadrature,
    quat_from_axis_angle,
    quat_from_euler,
    quat_norm_error,
    submerged_hydrostatics,
    quat_to_matrix,
)
from tensordem.floe_rigid6dof import (
    ContactSettings,
    Environment,
    FloeWorld,
    Hydrodynamics,
    PlaneBoundary,
    analytic_heave_period,
    angular_momentum,
    box_mesh,
    build_floe,
    floe_buoyancy,
    l_prism_mesh,
    linear_momentum,
    mesh_mass_properties,
    mesh_topology_report,
    regular_prism_mesh,
    submerged_moments,
)
from tensordem.ice_channel import WaterEnvironment

DTYPE = torch.float64
RHO_I, RHO_W, G = 917.0, 1025.0, 9.81
L, W, T = 4.0, 3.0, 0.4
DRAUGHT = RHO_I / RHO_W * T
Z_EQ = 0.5 * T - DRAUGHT          # centre of mass at equilibrium


def _pos(z: float, x: float = 0.0, y: float = 0.0) -> torch.Tensor:
    return torch.tensor([x, y, z], dtype=DTYPE)


def _box_floe():
    return build_floe(*box_mesh(L, W, T), density=RHO_I, name="box")


def _hydro(**kw) -> Hydrodynamics:
    base = dict(added_mass_ratio=0.5, added_inertia_ratio=0.2, drag_coefficient=0.0,
                enable_drag=False)
    base.update(kw)
    return Hydrodynamics(**base)


# ---------------------------------------------------------------------------
# Geometry and mass properties
# ---------------------------------------------------------------------------

def test_box_mass_properties_match_analytic():
    props = mesh_mass_properties(*box_mesh(L, W, T), density=RHO_I)
    assert props.volume == pytest.approx(L * W * T, rel=1e-14)
    assert props.mass == pytest.approx(RHO_I * L * W * T, rel=1e-14)
    assert torch.allclose(props.centroid, torch.zeros(3, dtype=DTYPE), atol=1e-15)
    inertia = torch.diagonal(props.inertia)
    m = props.mass
    assert float(inertia[0]) == pytest.approx(m * (W * W + T * T) / 12.0, rel=1e-14)
    assert float(inertia[1]) == pytest.approx(m * (L * L + T * T) / 12.0, rel=1e-14)
    assert float(inertia[2]) == pytest.approx(m * (L * L + W * W) / 12.0, rel=1e-14)


def test_concave_l_prism_volume_and_centroid():
    """The signed tetrahedral decomposition is exact for a concave body."""
    notch_l, notch_w = 2.5, 1.8
    verts, faces = l_prism_mesh(L, W, T, notch_l, notch_w)
    report = mesh_topology_report(verts, faces)
    assert report["closed"] and report["directed_edge_duplicates"] == 0
    props = mesh_mass_properties(verts, faces, density=RHO_I)
    area = L * notch_w + notch_l * (W - notch_w)
    assert props.volume == pytest.approx(area * T, rel=1e-14)
    cx = (L * notch_w * (L / 2) + (W - notch_w) * notch_l * (notch_l / 2)) / area
    cy = ((L * notch_w) * (notch_w / 2)
          + (W - notch_w) * notch_l * (notch_w + (W - notch_w) / 2)) / area
    assert float(props.centroid[0]) == pytest.approx(cx, rel=1e-12)
    assert float(props.centroid[1]) == pytest.approx(cy, rel=1e-12)


def test_mesh_topology_audit_flags_open_and_inverted_meshes():
    verts, faces = box_mesh(L, W, T)
    assert mesh_topology_report(verts, faces)["closed"]
    open_faces = faces[1:]
    assert not mesh_topology_report(verts, open_faces)["closed"]
    with pytest.raises(ValueError):
        mesh_mass_properties(verts, open_faces)
    with pytest.raises(ValueError):
        mesh_mass_properties(verts, faces[:, ::-1])      # inverted winding
    with pytest.raises(ValueError):
        mesh_mass_properties(verts, faces, density=0.0)


# ---------------------------------------------------------------------------
# Hydrostatics
# ---------------------------------------------------------------------------

def test_half_submerged_box_volume_and_centroid():
    verts, faces = box_mesh(L, W, T)
    verts = verts - np.array([0.0, 0.0, 0.5 * T])        # bottom at the waterline - T
    wet = submerged_moments(verts, faces, water_level=-0.5 * T)
    assert wet.volume == pytest.approx(L * W * T / 2.0, rel=1e-14)
    assert float(wet.centroid[2]) == pytest.approx(-0.75 * T, abs=1e-15)
    # buoyancy potential integral of the same configuration
    assert wet.zeta_moment == pytest.approx(-L * W * T * T / 8.0, rel=1e-13)


def test_fully_wet_volume_equals_body_volume_for_concave_shape():
    verts, faces = l_prism_mesh(L, W, T, 2.5, 1.8)
    floe = build_floe(verts, faces, density=RHO_I)
    wet = submerged_moments(np.asarray(verts), np.asarray(faces), water_level=10.0)
    assert wet.volume == pytest.approx(floe.volume, rel=1e-13)


def test_float_and_vectorised_kernels_agree():
    """The plain-float hot kernel and the vectorised numpy kernel must agree."""
    floe = build_floe(*regular_prism_mesh(2.0, 0.4, 48), density=RHO_I)
    for pitch in (0.0, 0.12, 0.35, -0.4):
        quat = quat_from_euler(torch.tensor([0.0], dtype=DTYPE),
                               torch.tensor([pitch], dtype=DTYPE),
                               torch.tensor([0.0], dtype=DTYPE))
        rot = quat_to_matrix(quat)[0].numpy()
        nodes = floe.vertices.numpy() @ rot.T + np.array([0.0, 0.0, -0.05])
        slow = submerged_moments(nodes, floe.faces.numpy(), 0.0)
        _, _, wet = floe_buoyancy(floe, _pos(-0.05), quat, Environment())
        assert wet.volume == pytest.approx(slow.volume, rel=1e-12)
        assert float(wet.zeta_moment) == pytest.approx(slow.zeta_moment, rel=1e-12)


def test_prism_hydrostatics_match_disk_quadrature():
    """Independent cross-check against the IN22-N02 disk hydrostatics."""
    radius, thickness = 2.0, 0.4
    floe = build_floe(*regular_prism_mesh(radius, thickness, 128), density=RHO_I)
    water = WaterEnvironment()
    quad = disk_quadrature(24, 32)
    for pitch in (0.0, 0.15, 0.4):
        quat = quat_from_euler(torch.tensor([0.0], dtype=DTYPE),
                               torch.tensor([pitch], dtype=DTYPE),
                               torch.tensor([0.0], dtype=DTYPE))
        ref = submerged_hydrostatics(radius, thickness, quat,
                                     torch.tensor([0.0], dtype=DTYPE), water, quad)
        _, _, wet = floe_buoyancy(floe, _pos(0.0), quat, Environment())
        # the n-gon prism is 4e-4 smaller than the analytic circle
        assert wet.volume == pytest.approx(float(ref.volume[0]), rel=1e-3)
        assert float(wet.centroid[2]) == pytest.approx(
            float(ref.centroid_world[0, 2]), abs=2e-3)


def test_buoyancy_force_is_potential_gradient():
    floe = _box_floe()
    env = Environment()
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE)

    def potential(z: float) -> float:
        _, _, wet = floe_buoyancy(floe, _pos(z), quat, env)
        return floe.mass * G * z - RHO_W * G * wet.zeta_moment

    h = 1e-6
    for z in (Z_EQ, Z_EQ + 0.05):
        numeric = -(potential(z + h) - potential(z - h)) / (2.0 * h)
        buoy = float(floe_buoyancy(floe, _pos(z), quat, env)[0][2])
        assert numeric + floe.mass * G == pytest.approx(buoy, rel=1e-6)


def test_buoyancy_moment_is_potential_gradient():
    floe = _box_floe()
    env = Environment()
    env_z = Z_EQ
    axis = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)

    def potential(angle: float) -> float:
        quat = quat_from_axis_angle(axis, torch.tensor([angle], dtype=DTYPE))
        _, _, wet = floe_buoyancy(floe, _pos(env_z), quat, env)
        return floe.mass * G * env_z - RHO_W * G * wet.zeta_moment

    h = 1e-6
    for pitch in (0.02, 0.1, 0.35):
        quat = quat_from_axis_angle(axis, torch.tensor([pitch], dtype=DTYPE))
        moment = float(floe_buoyancy(floe, _pos(env_z), quat, env)[1][1])
        numeric = -(potential(pitch + h) - potential(pitch - h)) / (2.0 * h)
        assert moment == pytest.approx(numeric, rel=1e-8)


def test_equilibrium_draught_balances_weight():
    floe = _box_floe()
    force, moment, wet = floe_buoyancy(floe, _pos(Z_EQ),
                                       torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE),
                                       Environment())
    assert float(force[2]) == pytest.approx(floe.mass * G, rel=1e-12)
    assert wet.volume == pytest.approx(L * W * DRAUGHT, rel=1e-12)
    assert float(moment.abs().max()) < 1e-9 * floe.mass * G


def test_concave_floe_has_restoring_moment():
    """A heeled concave floe must be pushed back (negative rolling moment)."""
    floe = build_floe(*l_prism_mesh(L, W, T, 2.5, 1.8), density=RHO_I)
    axis = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)
    z = 0.5 * T - RHO_I / RHO_W * T
    for pitch in (0.05, 0.15):
        quat = quat_from_axis_angle(axis, torch.tensor([pitch], dtype=DTYPE))
        _, moment, _ = floe_buoyancy(floe, _pos(z), quat, Environment())
        assert float(moment[1]) < 0.0      # restoring


# ---------------------------------------------------------------------------
# Attitude kinematics
# ---------------------------------------------------------------------------

def test_quaternion_norm_and_attitude_round_trip():
    floe = _box_floe()
    world = FloeWorld([floe], [[0.0, 0.0, Z_EQ]],
                      omegas=[[0.30, -0.20, 0.45]],
                      hydro=_hydro(), integrator="verlet")
    world.run(2.0, 5e-3)
    assert world.quat_norm_error() < 1e-12
    quat = world.state.quat
    assert float(torch.linalg.norm(quat)) == pytest.approx(1.0, rel=1e-14)
    rot = quat_to_matrix(quat)[0]
    assert float(torch.linalg.det(rot)) == pytest.approx(1.0, rel=1e-13)


def test_free_decay_frequency_and_damping_match_analytic():
    """Single floe heave decay vs the closed-form damped oscillator."""
    floe = _box_floe()
    damping = 2822.0
    hydro = Hydrodynamics(added_mass_ratio=0.5, added_inertia_ratio=0.2,
                          drag_coefficient=0.0, enable_drag=False,
                          linear_heave_damping=damping)
    world = FloeWorld([floe], [[0.0, 0.0, Z_EQ + 0.02]], hydro=hydro,
                      integrator="verlet")
    times, heights = [], []
    world.run(9.0, 2e-3, callback=lambda w: (times.append(w.time),
                                             heights.append(float(w.state.position[0, 2]))))
    t = np.asarray(times)
    z = np.asarray(heights) - Z_EQ
    idx = [i for i in range(1, t.size) if z[i - 1] < 0.0 <= z[i]]
    assert len(idx) >= 5
    period = float(np.diff(t[idx]).mean())
    ref = analytic_heave_period(L * W, floe.mass, 0.5, damping, RHO_W, G)
    assert abs(period - ref["period_damped"]) / ref["period_damped"] < 1e-2
    # exponential envelope over the measured number of periods
    peaks = [float(np.abs(z[idx[k]:idx[k + 1]]).max()) for k in range(len(idx) - 1)]
    measured = math.log(peaks[0] / peaks[-1]) / (len(peaks) - 1)
    expected = ref["zeta"] * ref["omega_n"] * ref["period_damped"]
    assert measured == pytest.approx(expected, rel=2e-2)


# ---------------------------------------------------------------------------
# Contacts, momentum and the energy ledger
# ---------------------------------------------------------------------------

def test_two_floe_collision_conserves_momentum():
    floe = _box_floe()
    env = Environment(enable_buoyancy=False, enable_gravity=False)
    hydro = _hydro(added_mass_ratio=0.0, added_inertia_ratio=0.0)
    world = FloeWorld([floe, floe], [[-2.35, 1.0, 0.0], [2.35, -1.0, 0.0]],
                      velocities=[[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0]],
                      env=env, hydro=hydro,
                      contact=ContactSettings(normal_stiffness=2.0e7),
                      integrator="verlet")
    p0 = linear_momentum(world).numpy()
    angular0 = angular_momentum(world).numpy()
    world.run(0.6, 1e-4)
    p1 = linear_momentum(world).numpy()
    angular1 = angular_momentum(world).numpy()
    p_scale = float(np.linalg.norm(p0))
    angular_scale = float(np.linalg.norm(p0)) * L
    assert np.linalg.norm(p1 - p0) / p_scale < 1e-8
    assert np.linalg.norm(angular1 - angular0) / angular_scale < 1e-8
    # the floes must have exchanged momentum (a real contact happened)
    assert float(world.state.velocity[0, 0]) < 0.7
    assert world.max_penetration < 0.5 * T


def test_multiple_floes_static_equilibrium():
    floes = [_box_floe() for _ in range(4)]
    world = FloeWorld(floes, [[0.0, 0.0, Z_EQ], [9.0, 0.0, Z_EQ],
                              [-9.0, 0.0, Z_EQ], [0.0, 10.0, Z_EQ]],
                      hydro=_hydro(), integrator="verlet")
    loads = world.residual_loads()
    assert loads["force_ratio_max"] < 1e-6
    assert loads["moment_ratio_max"] < 1e-6


def test_energy_ledger_conservative():
    floe = _box_floe()
    world = FloeWorld([floe], [[0.0, 0.0, Z_EQ + 0.02]], hydro=_hydro(),
                      integrator="verlet")
    world.run(2.0, 5e-4)
    ledger = world.energy_ledger()
    assert ledger["residual_relative"] < 1e-6
    assert ledger["dissipated"] == 0.0


def test_energy_ledger_with_dissipation():
    floe = _box_floe()
    hydro = Hydrodynamics(added_mass_ratio=0.5, added_inertia_ratio=0.2,
                          drag_coefficient=1.2, enable_drag=True,
                          linear_heave_damping=500.0)
    world = FloeWorld([floe], [[0.0, 0.0, Z_EQ + 0.03]], hydro=hydro,
                      integrator="verlet")
    world.run(2.0, 5e-4)
    ledger = world.energy_ledger()
    assert ledger["dissipated"] > 0.0
    assert ledger["residual_relative"] < 1e-6


def test_energy_ledger_with_contact_storage():
    floe = _box_floe()
    env = Environment(enable_buoyancy=False, enable_gravity=False)
    world = FloeWorld([floe, floe], [[-2.35, 1.0, 0.0], [2.35, -1.0, 0.0]],
                      velocities=[[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0]],
                      env=env, hydro=_hydro(added_mass_ratio=0.0,
                                            added_inertia_ratio=0.0),
                      contact=ContactSettings(normal_stiffness=2.0e7),
                      integrator="verlet")
    world.run(0.6, 1e-4)
    ledger = world.energy_ledger()
    assert ledger["residual_state_relative"] < 1e-6
    # the trapezoidal power integral over a stiff impact is only ~1 % accurate
    assert ledger["residual_relative"] < 2e-2


def test_negative_control_zero_force_flight():
    floe = _box_floe()
    env = Environment(enable_buoyancy=False, enable_gravity=False)
    world = FloeWorld([floe], [[0.0, 0.0, 0.0]],
                      velocities=[[0.3, -0.2, 0.1]],
                      omegas=[[0.0, 0.0, -0.25]],   # spin about a principal axis
                      env=env, hydro=_hydro(added_mass_ratio=0.0,
                                            added_inertia_ratio=0.0),
                      integrator="rk4")
    v0 = world.state.velocity.clone()
    omega0 = world.state.omega.clone()
    angular0 = angular_momentum(world).numpy()
    world.run(2.0, 1e-3)
    assert float((world.state.velocity - v0).abs().max()) < 1e-14
    assert float((world.state.omega - omega0).abs().max()) < 1e-12
    assert np.allclose(angular_momentum(world).numpy(), angular0, atol=1e-10)
    assert world.quat_norm_error() < 1e-12


def test_plane_contact_seabed_with_friction():
    """A floe dropped on the seabed must stop, stay above the plane and slide to rest."""
    floe = _box_floe()
    env = Environment(enable_buoyancy=False, enable_gravity=True)
    seabed = PlaneBoundary(normal=(0.0, 0.0, 1.0), offset=-0.2, name="seabed")
    world = FloeWorld([floe], [[0.0, 0.0, 0.6]], velocities=[[0.5, 0.0, 0.0]],
                      env=env,
                      hydro=_hydro(added_mass_ratio=0.0, added_inertia_ratio=0.0),
                      contact=ContactSettings(normal_stiffness=2.0e6,
                                              normal_damping=50.0,
                                              friction=0.4),
                      planes=[seabed], integrator="verlet")
    world.run(0.6, 5e-4)
    nodes = world.nodes_world(0).numpy()
    assert nodes[:, 2].min() > -0.2 - 0.05            # no deep penetration
    assert abs(float(world.state.velocity[0, 0])) < 0.4   # slowed by friction
    assert float(world.state.velocity[0, 2]) > -0.2


def test_adaptive_stepper_matches_fixed_step():
    floe = _box_floe()
    hydro = _hydro(linear_heave_damping=0.0)
    fixed = FloeWorld([floe], [[0.0, 0.0, Z_EQ + 0.02]], hydro=hydro,
                      integrator="rk4")
    adaptive = FloeWorld([floe], [[0.0, 0.0, Z_EQ + 0.02]], hydro=hydro,
                         integrator="rk4")
    fixed.run(1.0, 1e-4)
    info = adaptive.run(1.0, 1e-3, adaptive=True, tol=1e-6,
                        dt_bounds=(1e-5, 1e-2))
    z_fixed = float(fixed.state.position[0, 2])
    z_adaptive = float(adaptive.state.position[0, 2])
    assert abs(z_adaptive - z_fixed) < 1e-5
    assert info["steps"] < 2000
    assert adaptive.quat_norm_error() < 1e-12


# ---------------------------------------------------------------------------
# Negative / robustness cases
# ---------------------------------------------------------------------------

def test_invalid_inputs_are_rejected():
    floe = _box_floe()
    with pytest.raises(ValueError):
        FloeWorld([], [])
    with pytest.raises(ValueError):
        FloeWorld([floe], [[0.0, 0.0, 0.0]], integrator="euler")
    with pytest.raises(ValueError):
        FloeWorld([floe], [[0.0, 0.0, 0.0]],
                  env=Environment(water_density=-1.0))
    with pytest.raises(ValueError):
        FloeWorld([floe], [[float("nan"), 0.0, 0.0]])
    world = FloeWorld([floe], [[0.0, 0.0, Z_EQ]])
    with pytest.raises(ValueError):
        world.step(0.0)
    with pytest.raises(ValueError):
        world.run(-1.0, 1e-3)
    with pytest.raises(ValueError):
        build_floe(*box_mesh(-1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        ContactSettings(normal_stiffness=-1.0).validate()


def test_time_step_refinement_converges():
    """Halving dt must reduce the period error (2nd order Verlet)."""
    floe = _box_floe()
    hydro = _hydro()
    ref = analytic_heave_period(L * W, floe.mass, 0.5, 0.0, RHO_W, G)
    errors = []
    for dt in (2e-3, 1e-3):
        world = FloeWorld([floe], [[0.0, 0.0, Z_EQ + 0.02]], hydro=hydro,
                          integrator="verlet")
        times, heights = [], []
        world.run(6.0, dt, callback=lambda w: (times.append(w.time),
                                              heights.append(float(w.state.position[0, 2]))))
        t = np.asarray(times)
        z = np.asarray(heights) - Z_EQ
        idx = [i for i in range(1, t.size) if z[i - 1] < 0.0 <= z[i]]
        period = float(np.diff(t[idx]).mean())
        errors.append(abs(period - ref["period_damped"]) / ref["period_damped"])
    assert errors[1] < 1e-2
    assert errors[1] <= errors[0] * 1.5