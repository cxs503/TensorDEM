"""Multi-sphere (clump) particles: mass properties, contacts, rigid dynamics.

Two classes of claim are pinned here.

*Geometry / mass properties.*  The union inertia must agree with the closed-form
dumbbell value to machine precision (criterion 1 of the campaign node), must
agree with an independent voxel integration of the union geometry, must be
strictly smaller than the naive member sum when members overlap, and must sit in
the physically admissible set (positive, principal moments satisfying the
triangle inequality).

*Rigid dynamics with member-summed contacts.*  Free fall must conserve energy and
leave the orientation untouched; a "rolling-pin" dumbbell must roll down an
inclined plane with the analytically predicted acceleration and a mirror-image
trajectory when the tilt is reversed; static contact forces must balance the
weight; and the clump-sphere contact must satisfy Newton's third law.

*Rejection paths.*  Zero/negative radii, over-limit member overlap, coincident
members, disconnected members, triple overlaps, empty member lists and
non-positive density are all hard errors -- a clump that silently accepts them
produces plausible but wrong inertia later.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.clump import (
    Clump,
    ClumpGeometryError,
    ClumpOverlapError,
    ClumpTopologyError,
    SphereMember,
    double_sphere_clump,
    double_sphere_inertia_analytic,
    double_sphere_mass_analytic,
    lens_moments,
    mass_properties_by_voxel,
    sphere_moments,
    triple_intersections,
    union_moments,
)
from tensordem.clump_dynamics import (
    ClumpBody,
    ContactConfig,
    PlaneContact,
    SphereObstacle,
    effective_mass_along,
    quaternion_from_axis_angle,
    restitution_to_damping_ratio,
)

RADIUS = 0.05
DENSITY = 900.0


# --------------------------------------------------------------------------
# Geometry and mass properties
# --------------------------------------------------------------------------
@pytest.mark.parametrize("separation", [2 * RADIUS, 2.2 * RADIUS])
def test_double_sphere_inertia_matches_closed_form(separation: float) -> None:
    """Criterion 1: union inertia tensor vs closed-form dumbbell value.

    The closed form is the parallel-axis sum of two whole spheres, which equals
    the union only when the members do not overlap (touching or separated), so
    the criterion is stated for ``separation >= 2 radius``.
    """
    clump = double_sphere_clump(RADIUS, separation, density=DENSITY,
                                require_connected=False)
    reference = double_sphere_inertia_analytic(RADIUS, separation, density=DENSITY)
    relative = np.max(np.abs(clump.inertia_about_com - reference)) / np.max(np.abs(reference))
    assert relative < 1e-10, relative
    # mass and centre of mass of the two-sphere aggregate
    assert abs(clump.mass - double_sphere_mass_analytic(RADIUS, separation, DENSITY)) < 1e-14
    assert np.allclose(clump.center_of_mass, 0.0, atol=1e-15)
    # the axial moment is the no-parallel-axis case
    assert abs(clump.inertia_about_axis((1, 0, 0)) - 2 * 0.4 * (0.5 * clump.mass) * RADIUS ** 2) < 1e-15
    # off-diagonal terms vanish by construction
    off_diagonal = clump.inertia_about_com - np.diag(np.diag(clump.inertia_about_com))
    assert np.max(np.abs(off_diagonal)) < 1e-15 * np.max(np.abs(reference))


def test_union_equals_member_sum_when_members_do_not_overlap() -> None:
    clump = double_sphere_clump(RADIUS, 2.2 * RADIUS, density=DENSITY,
                                require_connected=False)
    np.testing.assert_allclose(clump.inertia_union, clump.inertia_member_sum,
                               rtol=1e-13, atol=1e-20)
    assert abs(clump.volume - clump.member_sum_volume) < 1e-15


def test_overlap_removes_volume_and_inertia() -> None:
    """Ignoring the overlap double-counts mass: the union must be smaller."""
    clump = double_sphere_clump(RADIUS, 1.5 * RADIUS, density=DENSITY,
                                max_overlap_fraction=0.6)
    removed = clump.member_sum_volume - clump.volume
    assert removed > 0.0
    # the lens volume of two equal spheres has a closed form
    r, d = RADIUS, 1.5 * RADIUS
    lens_analytic = math.pi * (2 * r - d) ** 2 * (4 * r + d) / 12.0
    assert abs(removed - lens_analytic) / lens_analytic < 1e-12
    for axial, transverse in ((0, 0), (1, 1), (2, 2)):
        assert clump.inertia_union[axial, transverse] < clump.inertia_member_sum[axial, transverse]
    # ... and the union mass must follow the corrected volume
    assert abs(clump.mass - DENSITY * (clump.member_sum_volume - lens_analytic)) < 1e-15


@pytest.mark.parametrize("separation", [2 * RADIUS, 1.5 * RADIUS])
@pytest.mark.parametrize("resolution,tolerance", [(96, 2e-3), (192, 5e-4)])
def test_union_inertia_matches_voxel_integration(separation: float,
                                                 resolution: int,
                                                 tolerance: float) -> None:
    """Independent numerical integration of the union geometry."""
    clump = double_sphere_clump(RADIUS, separation, density=DENSITY,
                                max_overlap_fraction=0.6)
    voxel = mass_properties_by_voxel(clump.centers, clump.radii, DENSITY,
                                     resolution=resolution)
    assert abs(voxel["mass"] - clump.mass) / clump.mass < tolerance
    relative = np.max(np.abs(voxel["inertia"] - clump.inertia_about_com)) / np.max(
        np.abs(clump.inertia_about_com))
    assert relative < tolerance, (resolution, relative)
    np.testing.assert_allclose(voxel["center_of_mass"], clump.center_of_mass,
                               atol=1e-3 * RADIUS)


def test_voxel_error_contracts_with_resolution() -> None:
    clump = double_sphere_clump(RADIUS, 1.5 * RADIUS, density=DENSITY,
                                max_overlap_fraction=0.6)
    errors = []
    for resolution in (48, 96, 192):
        voxel = mass_properties_by_voxel(clump.centers, clump.radii, DENSITY,
                                         resolution=resolution)
        errors.append(abs(voxel["mass"] - clump.mass) / clump.mass)
    # midpoint voxel integration converges towards the exact union value
    assert errors[-1] < errors[0]
    assert errors[-1] < 1e-3


def test_inertia_tensor_is_physical() -> None:
    clump = Clump(
        (SphereMember((0.0, 0.0, 0.0), 0.04),
         SphereMember((0.065, 0.0, 0.0), 0.035),
         SphereMember((0.03, 0.05, 0.0), 0.03)),
        density=DENSITY, max_overlap_fraction=0.5)
    inertia = clump.inertia_about_com
    np.testing.assert_allclose(inertia, inertia.T, atol=1e-18)
    eigenvalues = clump.principal_moments
    assert np.all(eigenvalues > 0.0)
    # principal moments of a 3-D body obey I1 + I2 >= I3 (triangle inequality)
    assert eigenvalues[0] + eigenvalues[1] >= eigenvalues[2] - 1e-15
    # the inertia about any axis lies between the extreme principal moments
    axis = np.array([0.3, -0.7, 0.5])
    moment = clump.inertia_about_axis(axis)
    assert eigenvalues[0] - 1e-15 <= moment <= eigenvalues[2] + 1e-15
    assert abs(clump.mass - DENSITY * clump.volume) < 1e-18


def test_inertia_is_rotation_invariant() -> None:
    """Rotating the body frame rotates I but leaves the spectrum unchanged."""
    members = (SphereMember((0.0, 0.0, 0.0), 0.04),
               SphereMember((0.065, 0.0, 0.0), 0.035))
    base = Clump(members, density=DENSITY, max_overlap_fraction=0.5)
    angle = 0.7
    rotation = np.array([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotated = Clump(tuple(SphereMember(tuple(rotation @ np.asarray(m.center)), m.radius)
                          for m in members), density=DENSITY, max_overlap_fraction=0.5)
    np.testing.assert_allclose(rotated.mass, base.mass, rtol=1e-14)
    np.testing.assert_allclose(rotated.principal_moments, base.principal_moments,
                               rtol=1e-12)
    np.testing.assert_allclose(rotated.inertia_about_com,
                               rotation @ base.inertia_about_com @ rotation.T,
                               rtol=1e-12, atol=1e-20)


def test_lens_moments_match_closed_form_volume() -> None:
    r, d = 0.05, 0.08
    volume, first, second = lens_moments((-0.5 * d, 0, 0), r, (0.5 * d, 0, 0), r)
    analytic = math.pi * (2 * r - d) ** 2 * (4 * r + d) / 12.0
    assert abs(volume - analytic) / analytic < 1e-13
    # symmetric lens: centroid on the line of centres, at the mid-plane
    np.testing.assert_allclose(first / volume, [0.0, 0.0, 0.0], atol=1e-15)
    # displacement scaling: Q_xx - Q_yy = (I_yy - I_xx)/... symmetric trace identity
    inertia = np.trace(second) * np.eye(3) - second
    # the lens is a flattened disc: its largest inertia is about the line of centres
    assert inertia[0, 0] > inertia[1, 1] > 0.0
    # trace identity: I_xx - I_zz == Q_zz - Q_xx for the slice-frame moments
    assert abs((inertia[0, 0] - inertia[2, 2]) - (second[2, 2] - second[0, 0])) < 1e-20


def test_sphere_moments_closed_form() -> None:
    volume, first, second = sphere_moments((0.1, -0.2, 0.3), 0.05)
    assert abs(volume - 4.0 / 3.0 * math.pi * 0.05 ** 3) < 1e-18
    np.testing.assert_allclose(first / volume, (0.1, -0.2, 0.3), atol=1e-16)
    inertia = np.trace(second) * np.eye(3) - second
    # Steiner: I_COM = 2/5 m r^2, I_origin = I_COM + m (|c|^2 - c c^T)
    centre = np.array([0.1, -0.2, 0.3])
    expected = (0.4 * volume * 0.05 ** 2) * np.eye(3) + volume * (
        float(centre @ centre) * np.eye(3) - np.outer(centre, centre))
    np.testing.assert_allclose(inertia, expected, rtol=1e-13, atol=1e-20)


def test_union_moments_are_additive_for_disjoint_balls() -> None:
    centers = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    radii = np.array([0.05, 0.07])
    volume, first, second = union_moments(centers, radii)
    v0, s0, q0 = sphere_moments(centers[0], radii[0])
    v1, s1, q1 = sphere_moments(centers[1], radii[1])
    np.testing.assert_allclose(volume, v0 + v1, rtol=1e-15)
    np.testing.assert_allclose(first, s0 + s1, rtol=1e-15)
    np.testing.assert_allclose(second, q0 + q1, rtol=1e-15)


# --------------------------------------------------------------------------
# Rejection paths (negative controls)
# --------------------------------------------------------------------------
def test_rejects_zero_radius_member() -> None:
    with pytest.raises(ClumpGeometryError, match="radius must be positive"):
        Clump((SphereMember((0.0, 0.0, 0.0), 0.0), SphereMember((0.1, 0.0, 0.0), 0.05)))


@pytest.mark.parametrize("bad_radius", [-0.02, float("nan"), float("inf")])
def test_rejects_invalid_radius(bad_radius: float) -> None:
    with pytest.raises(ClumpGeometryError):
        Clump((SphereMember((0.0, 0.0, 0.0), 0.05),
               SphereMember((0.2, 0.0, 0.0), bad_radius)))


def test_rejects_overlap_above_limit() -> None:
    # two spheres of radius 0.05 whose centres are 0.06 apart: penetration 0.04
    # is 80 % of the smaller radius, far above the default 25 % limit.
    with pytest.raises(ClumpOverlapError, match="exceeding max_overlap_fraction"):
        Clump((SphereMember((0.0, 0.0, 0.0), 0.05),
               SphereMember((0.06, 0.0, 0.0), 0.05)), max_overlap_fraction=0.25)
    # exactly at the limit is accepted, just below it is rejected
    Clump((SphereMember((0.0, 0.0, 0.0), 0.05),
           SphereMember((0.0875, 0.0, 0.0), 0.05)), max_overlap_fraction=0.25)
    with pytest.raises(ClumpOverlapError):
        Clump((SphereMember((0.0, 0.0, 0.0), 0.05),
               SphereMember((0.0874, 0.0, 0.0), 0.05)), max_overlap_fraction=0.25)


def test_rejects_coincident_members() -> None:
    with pytest.raises(ClumpOverlapError):
        Clump((SphereMember((0.0, 0.0, 0.0), 0.05),
               SphereMember((0.0, 0.0, 0.0), 0.05)))


def test_rejects_empty_member_list() -> None:
    with pytest.raises(ClumpGeometryError, match="at least one member"):
        Clump(())
    with pytest.raises(ClumpGeometryError):
        Clump([])


@pytest.mark.parametrize("density", [0.0, -900.0, float("nan")])
def test_rejects_invalid_density(density: float) -> None:
    with pytest.raises(ClumpGeometryError, match="density"):
        Clump((SphereMember((0.0, 0.0, 0.0), 0.05),), density=density)


def test_rejects_disconnected_members() -> None:
    with pytest.raises(ClumpTopologyError, match="disconnected"):
        Clump((SphereMember((0.0, 0.0, 0.0), 0.05),
               SphereMember((1.0, 0.0, 0.0), 0.05)))
    # ... unless the user explicitly asks for a non-rigid set
    loose = Clump((SphereMember((0.0, 0.0, 0.0), 0.05),
                   SphereMember((1.0, 0.0, 0.0), 0.05)), require_connected=False)
    assert len(loose) == 2


def test_rejects_triple_overlap() -> None:
    """Three members sharing a common point cannot be handled by inclusion-exclusion."""
    side = 0.085  # equilateral triangle; centroid distance 0.0491 < r = 0.05
    members = (SphereMember((0.0, 0.0, 0.0), 0.05),
               SphereMember((side, 0.0, 0.0), 0.05),
               SphereMember((0.5 * side, side * math.sqrt(3.0) / 2.0, 0.0), 0.05))
    assert triple_intersections(np.array([m.center for m in members]),
                                np.array([m.radius for m in members])) == [(0, 1, 2)]
    with pytest.raises(ClumpTopologyError, match="common point"):
        Clump(members, max_overlap_fraction=0.4)
    # a chain of three members with exclusive overlaps is fine
    chain = (SphereMember((0.0, 0.0, 0.0), 0.05),
             SphereMember((0.08, 0.0, 0.0), 0.05),
             SphereMember((0.16, 0.0, 0.0), 0.05))
    assert triple_intersections(np.array([m.center for m in chain]),
                                np.array([m.radius for m in chain])) == []
    Clump(chain, max_overlap_fraction=0.5)


def test_rejects_unknown_inertia_model() -> None:
    with pytest.raises(ClumpGeometryError, match="inertia_model"):
        Clump((SphereMember((0.0, 0.0, 0.0), 0.05),), inertia_model="hull")


# --------------------------------------------------------------------------
# Rigid-body dynamics
# --------------------------------------------------------------------------
def test_restitution_to_damping_ratio_limits() -> None:
    assert restitution_to_damping_ratio(1.0) == 0.0
    ratio = restitution_to_damping_ratio(0.5)
    assert 0.2 < ratio < 0.23
    with pytest.raises(ValueError):
        restitution_to_damping_ratio(0.0)
    with pytest.raises(ValueError):
        restitution_to_damping_ratio(1.5)


def test_effective_mass_matches_translation_limit() -> None:
    clump = Clump((SphereMember((0.0, 0.0, 0.0), RADIUS),), density=DENSITY)
    inverse_inertia = np.linalg.inv(clump.inertia_about_com)
    # impulse through the centre of mass: pure translation
    assert abs(effective_mass_along(clump, (0, 0, 0), (0, 0, 1), inverse_inertia)
               - clump.mass) < 1e-15
    # impulse at the bottom of the sphere: less mass responds (rotation is excited)
    tangential = effective_mass_along(clump, (0, 0, -RADIUS), (1, 0, 0), inverse_inertia)
    assert abs(tangential - 1.0 / (1.0 / clump.mass + RADIUS ** 2 / (0.4 * clump.mass * RADIUS ** 2))) < 1e-14


def test_free_fall_conserves_energy_and_orientation() -> None:
    """Criterion 2a: free fall without rotation conserves energy exactly."""
    clump = double_sphere_clump(RADIUS, density=DENSITY)
    body = ClumpBody(clump, position=(0.0, 0.0, 2.0), gravity=(0.0, 0.0, -9.81))
    dt, steps = 1e-3, 2000
    history = body.run(steps, dt)
    energy = history["total_energy"]
    reference = abs(energy[0])
    # round-off accumulates as sqrt(n_steps) * eps; 1e-11 relative is machine level
    assert np.max(np.abs(energy - energy[0])) / reference < 1e-11
    assert np.max(np.abs(history["omega_body"])) == 0.0
    np.testing.assert_allclose(history["quaternion"], np.tile(
        [1.0, 0.0, 0.0, 0.0], (len(history["time"]), 1)), atol=1e-15)
    assert np.max(np.abs(history["angular_momentum"])) == 0.0
    # trajectory against the closed-form parabola
    time = history["time"]
    analytic_z = 2.0 - 0.5 * 9.81 * time ** 2
    assert np.max(np.abs(history["position"][:, 2] - analytic_z)) < 1e-11
    assert abs(body.velocity[2] + 9.81 * steps * dt) < 1e-10


def test_static_clump_on_plane_balances_weight() -> None:
    """Summed member normal forces must carry the weight; penetration is analytic."""
    clump = double_sphere_clump(RADIUS, density=DENSITY)
    stiffness = 1.0e5
    plane = PlaneContact((0.0, 0.0, 1.0), (0.0, 0.0, 0.0),
                         ContactConfig(normal_stiffness=stiffness, restitution=0.5,
                                       friction=0.6))
    expected_penetration = clump.mass * 9.81 / (2.0 * stiffness)
    body = ClumpBody(clump, position=(0.0, 0.0, RADIUS - expected_penetration),
                     gravity=(0.0, 0.0, -9.81),
                     orientation=quaternion_from_axis_angle((0, 0, 1), math.pi / 2))
    body.run(2000, 1e-4, (plane,))
    forces = plane.member_forces()
    assert np.all(forces[:, 2] > 0.0)
    assert abs(forces[:, 2].sum() - clump.mass * 9.81) / (clump.mass * 9.81) < 1e-6
    assert abs(plane.penetrations().max() - expected_penetration) / expected_penetration < 1e-3
    assert abs(body.velocity[2]) < 1e-6
    # two symmetric member contacts carry equal load
    assert abs(forces[0, 2] - forces[1, 2]) / forces[0, 2] < 1e-6


def test_clump_sphere_contact_satisfies_third_law() -> None:
    clump = double_sphere_clump(RADIUS, density=DENSITY)
    # obstacle placed under one member only, so the impact also torques the clump
    obstacle = SphereObstacle((0.0, RADIUS, -0.2), 0.02,
                              config=ContactConfig(normal_stiffness=1e5,
                                                   restitution=0.5, friction=0.3))
    body = ClumpBody(clump, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, -0.7),
                     gravity=(0.0, 0.0, -9.81),
                     orientation=quaternion_from_axis_angle((0, 0, 1), math.pi / 2))
    maximum_contact = 0.0
    approach = 0.0
    obstacle_reaction = 0.0
    was_in_contact = False
    for _ in range(6000):
        body.step(1e-4, (obstacle,))
        member_forces = obstacle.member_forces()
        if member_forces.size:
            assert np.allclose(obstacle.obstacle_force(), -member_forces.sum(axis=0),
                               rtol=1e-15, atol=1e-18)
            maximum_contact = max(maximum_contact, float(obstacle.penetrations().max()))
            obstacle_reaction = min(obstacle_reaction, float(obstacle.obstacle_force()[2]))
        in_contact = bool(obstacle.penetrations().max() > 0.0) if member_forces.size else False
        if in_contact:
            was_in_contact = True
            approach = min(approach, float(body.velocity[2]))
        elif was_in_contact:
            break
    assert maximum_contact > 0.0
    assert obstacle_reaction < 0.0          # the clump pushes the obstacle down
    # the clump is repelled: its downward speed after the contact is smaller
    assert float(body.velocity[2]) > approach + 1e-3, (float(body.velocity[2]), approach)


def test_inclined_plane_rolling_matches_analytic_acceleration() -> None:
    """Criterion 2b: rolling-pin dumbbell rolls with a = g sin(theta)/(1+I/(m r^2))."""
    angle = math.radians(10.0)
    normal = np.array([math.sin(angle), 0.0, math.cos(angle)])
    downslope = np.array([math.cos(angle), 0.0, -math.sin(angle)])
    clump = double_sphere_clump(RADIUS, density=DENSITY)
    stiffness = 1.0e5
    penetration = clump.mass * 9.81 * math.cos(angle) / (2.0 * stiffness)
    plane = PlaneContact(normal, (0.0, 0.0, 0.0),
                         ContactConfig(normal_stiffness=stiffness, restitution=0.5,
                                       friction=0.6))
    # dumbbell axis transverse to the slope: both members stay in contact while
    # the clump rolls about its own axis (a "rolling pin").
    body = ClumpBody(clump, position=normal * (RADIUS - penetration),
                     orientation=quaternion_from_axis_angle((0, 0, 1), math.pi / 2),
                     gravity=(0.0, 0.0, -9.81))
    duration, dt = 0.5, 1e-4
    body.run(int(duration / dt), dt, (plane,))
    inertia_roll = clump.inertia_about_axis((1.0, 0.0, 0.0))
    expected = 9.81 * math.sin(angle) / (1.0 + inertia_roll / (clump.mass * RADIUS ** 2))
    measured = float(body.velocity @ downslope) / duration
    assert abs(measured - expected) / expected < 2e-2, (measured, expected)
    # direction: down-slope translation and the matching rotation sense
    assert float(body.position @ downslope) > 0.0
    omega = body.angular_velocity
    assert omega[0] > 0.0
    # rolling without slip at both member contacts
    slip = float(body.velocity @ downslope) - omega[0] * RADIUS
    assert abs(slip) < 1e-3 * abs(float(body.velocity @ downslope))
    # normal direction: the clump stays on the plane
    assert abs(float(body.position @ normal) - RADIUS) < 0.01 * RADIUS


def test_inclined_plane_direction_reverses_with_tilt() -> None:
    angle = math.radians(10.0)
    clump = double_sphere_clump(RADIUS, density=DENSITY)
    stiffness = 1.0e5
    penetration = clump.mass * 9.81 * math.cos(angle) / (2.0 * stiffness)
    results = []
    for sign in (+1.0, -1.0):
        normal = np.array([sign * math.sin(angle), 0.0, math.cos(angle)])
        downslope = np.array([sign * math.cos(angle), 0.0, -math.sin(angle)])
        plane = PlaneContact(normal, (0.0, 0.0, 0.0),
                             ContactConfig(normal_stiffness=stiffness,
                                           restitution=0.5, friction=0.6))
        body = ClumpBody(clump, position=normal * (RADIUS - penetration),
                         orientation=quaternion_from_axis_angle((0, 0, 1), math.pi / 2),
                         gravity=(0.0, 0.0, -9.81))
        body.run(3000, 1e-4, (plane,))
        results.append((float(body.position @ downslope), float(body.angular_velocity[0])))
    assert results[0][0] > 0.0 and results[1][0] > 0.0
    assert results[0][0] == pytest.approx(results[1][0], rel=1e-6)
    assert results[0][1] == pytest.approx(-results[1][1], rel=1e-6, abs=1e-9)


def test_measured_restitution_is_in_band() -> None:
    """The linear spring-dashpot is calibrated for e; check the achieved ratio."""
    clump = Clump((SphereMember((0.0, 0.0, 0.0), RADIUS),), density=DENSITY)
    ratios = {}
    for target in (0.2, 0.5, 0.8):
        plane = PlaneContact((0.0, 0.0, 1.0), (0.0, 0.0, 0.0),
                             ContactConfig(normal_stiffness=1e5, restitution=target,
                                           friction=0.0))
        body = ClumpBody(clump, position=(0.0, 0.0, RADIUS + 2e-3),
                         gravity=(0.0, 0.0, -9.81))
        incoming = 0.0
        outgoing = None
        was_in_contact = False
        for _ in range(20000):
            body.step(2e-5, (plane,))
            speed = body.velocity[2]
            in_contact = bool(plane.penetrations()[0] > 0.0)
            if in_contact:
                was_in_contact = True
                incoming = min(incoming, speed)
            elif was_in_contact and incoming < -1e-6:
                outgoing = speed
                break
        assert outgoing is not None
        ratios[target] = -outgoing / incoming
    # The linear spring-dashpot reproduces the requested restitution to roughly
    # 0.03-0.06 absolute (it systematically under-shoots, see the module docstring
    # and docs/clump_multisphere_particles.md); the band is 0.1 absolute.
    for target, measured in ratios.items():
        assert abs(measured - target) < 0.1, (target, measured)
    assert ratios[0.2] < ratios[0.5] < ratios[0.8]


def test_contact_forces_are_frame_equivariant() -> None:
    """Rotating the whole configuration rotates the summed contact force."""
    clump = double_sphere_clump(RADIUS, density=DENSITY)
    angle = math.radians(15.0)
    rotation = np.array([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    normal = np.array([0.0, 0.0, 1.0])
    plane = PlaneContact(normal, (0.0, 0.0, 0.0),
                         ContactConfig(normal_stiffness=1e5, restitution=0.5, friction=0.0))
    base = ClumpBody(clump, position=(0.0, 0.0, RADIUS - 1e-4),
                     orientation=quaternion_from_axis_angle((0, 0, 1), math.pi / 2))
    force_base, _ = base.net_force((plane,), dt=1e-4)
    plane.reset()

    tilted_normal = rotation @ normal
    plane2 = PlaneContact(tilted_normal, (0.0, 0.0, 0.0),
                          ContactConfig(normal_stiffness=1e5, restitution=0.5,
                                        friction=0.0))
    tilted = ClumpBody(clump, position=rotation @ np.array([0.0, 0.0, RADIUS - 1e-4]),
                       orientation=quaternion_from_axis_angle(tilted_normal, math.pi / 2))
    force_tilted, _ = tilted.net_force((plane2,), dt=1e-4)
    np.testing.assert_allclose(force_tilted, rotation @ force_base, rtol=1e-10, atol=1e-12)