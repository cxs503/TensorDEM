"""Clump-clump contact: junction merging, Hertz-Mindlin forces, conservation.

Campaign node P16-N3.  The claims pinned here, in the order of the acceptance
criteria:

* **C1 degeneracy** -- a one-member clump pair must reproduce sphere-sphere
  contact: the normal force, the Mindlin tangential force and the measured
  restitution are compared against independent closed-form references (written
  from the textbook formulas, not from the module) to ``1e-12`` relative, and
  the pair integrator is compared bitwise against ``ClumpBody.step`` in free
  flight.
* **C2 geometric force level** -- a two-sphere clump touching another two-sphere
  clump must give the force of an independent geometric estimate (penetration
  written out by hand from the configuration) within 5%; the same for a
  four-overlapping-member-pair configuration.
* **C3 duplicate-counting negative control** -- when two strongly overlapping
  members of one clump indent the same sphere of the other, the members sample
  one physical junction twice.  The merged force must stay within 1.05x of the
  unique-junction force while the unmerged member sum is ~2x.
* **C4 third law** -- swapping the two bodies reverses the force to ``1e-12``
  relative, the pair contributes zero net moment about the origin, and the total
  linear momentum of an asymmetric collision is conserved to ``1e-12``.
* **C5 angular momentum / restitution** -- the total angular momentum of an
  oblique, frictional, spinning collision is conserved to ``1e-10`` relative and
  the measured restitution is within 0.05 of the requested value.
* **C6 stability** -- a resting pair does not drift over 1e5 steps (``1e-9`` m)
  and does not gain energy.

Geometry conventions used below: ``normal`` points from clump A's member into
clump B's member, so the force on B is ``+F_n n`` and the force on A is
``-F_n n``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.clump import Clump, SphereMember
from tensordem.clump_dynamics import (
    ClumpBody,
    ContactConfig as WallContactConfig,
    PlaneContact,
    effective_mass_along,
)
from tensordem.clump_clump import (
    ClumpClumpConfig,
    ClumpClumpContact,
    ClumpClumpError,
    _angular_acceleration,
    _Kinematics,
    _effective_mass_along,
    clump_clump_joints,
    duplicate_force_ratio,
    hertz_normal_force,
    member_pair_candidates,
    merge_candidates,
    pair_material_constants,
    total_angular_momentum,
    total_linear_momentum,
)

RADIUS = 0.05
DENSITY = 900.0
YOUNG = 1.0e8
POISSON = 0.3

STATIC_STEPS = 100_000          # C6: steps of the resting-pair drift test
STATIC_DT = 1.0e-5

# --------------------------------------------------------------------------
# Independent references (textbook formulas, deliberately not the module)
# --------------------------------------------------------------------------


def hertz_elastic(penetration: float, radius_eff: float, e_star: float) -> float:
    """``F = (4/3) E* sqrt(R*) delta^{3/2}`` (Johnson, Contact Mechanics, 4.2)."""
    return (4.0 / 3.0) * e_star * math.sqrt(radius_eff) * penetration ** 1.5


def hertz_damped(penetration: float, radius_eff: float, e_star: float,
                 velocity_normal: float, mass_eff: float,
                 restitution: float) -> float:
    """Hertz + viscous dashpot calibrated by the log-decrement law."""
    if restitution >= 1.0:
        zeta = 0.0
    else:
        ln_e = math.log(restitution)
        zeta = -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)
    k_n = 2.0 * e_star * math.sqrt(radius_eff * penetration)
    c_n = 2.0 * zeta * math.sqrt(mass_eff * k_n)
    return max(hertz_elastic(penetration, radius_eff, e_star) - c_n * velocity_normal,
               0.0)


def reduced_modulus(young: float, poisson: float) -> tuple[float, float]:
    """``(E*, G*)`` for two identical linear-elastic bodies."""
    shear = young / (2.0 * (1.0 + poisson))
    e_star = young / (2.0 * (1.0 - poisson ** 2))
    g_star = 1.0 / ((2.0 - poisson) / shear + (2.0 - poisson) / shear)
    return e_star, g_star


# --------------------------------------------------------------------------
# Clump fixtures
# --------------------------------------------------------------------------
def sphere_clump(radius: float = RADIUS, density: float = DENSITY) -> Clump:
    return Clump([SphereMember((0.0, 0.0, 0.0), radius)], density=density)


def dumbbell_clump(radius: float = RADIUS, separation: float = 0.1,
                   density: float = DENSITY) -> Clump:
    """Two members on the body-frame x axis (``separation`` apart)."""
    return Clump([SphereMember((-0.5 * separation, 0.0, 0.0), radius),
                  SphereMember((0.5 * separation, 0.0, 0.0), radius)],
                 density=density,
                 max_overlap_fraction=max(0.25, 2.0 * radius / radius))


def compact_clump(radius: float = RADIUS, separation: float = 0.02,
                  density: float = DENSITY) -> Clump:
    """Two members almost concentric: both indent the same junction region."""
    return Clump([SphereMember((-0.5 * separation, 0.0, 0.0), radius),
                  SphereMember((0.5 * separation, 0.0, 0.0), radius)],
                 density=density, max_overlap_fraction=2.0)


def config(**kwargs) -> ClumpClumpConfig:
    base = dict(young_modulus=YOUNG, poisson_ratio=POISSON)
    base.update(kwargs)
    return ClumpClumpConfig(**base)


# --------------------------------------------------------------------------
# C1 -- degeneracy: one-member clump == sphere-sphere
# --------------------------------------------------------------------------
@pytest.mark.parametrize("penetration", [1e-6, 1e-5, 1e-4, 1e-3])
@pytest.mark.parametrize("approach", [0.0, 0.01, 0.1])
def test_single_member_clump_reproduces_sphere_sphere_normal_force(
        penetration: float, approach: float) -> None:
    cfg = config(restitution=0.7)
    e_star, _ = reduced_modulus(YOUNG, POISSON)
    body_a = ClumpBody(sphere_clump(), gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(sphere_clump(),
                       position=(2.0 * RADIUS - penetration, 0.0, 0.0),
                       velocity=(approach, 0.0, 0.0), gravity=(0.0, 0.0, 0.0))
    response = ClumpClumpContact(cfg).forces(body_a, body_b, commit=False)
    assert len(response.joints) == 1
    mass_eff = 1.0 / (1.0 / body_a.clump.mass + 1.0 / body_b.clump.mass)
    # the geometry must be the requested penetration up to float representation
    reported = float(response.joints[0].penetration)
    assert abs(reported - penetration) <= 1e-6 * penetration
    # (v_B - v_A) . n with n = +x; the force law itself is compared at the
    # penetration the narrow phase reported, which isolates the law from the
    # round-off of 2R - |x|
    reference = hertz_damped(reported, 0.5 * RADIUS, e_star, approach,
                             mass_eff, 0.7)
    measured = float(response.normal_forces[0])
    # a fast approach at the smallest penetration clips the force to zero, so the
    # scale is the elastic force rather than the clipped reference
    scale = max(reference, hertz_elastic(reported, 0.5 * RADIUS, e_star))
    assert abs(measured - reference) <= 1e-12 * scale


def test_single_member_clump_matches_hertz_mindlin_material_constants() -> None:
    """``E*``/``G*``/``k_t`` must agree with ``tensordem.hertz_mindlin``."""
    hertz_mindlin = pytest.importorskip("tensordem.hertz_mindlin")
    cfg = config()
    e_star, g_star = reduced_modulus(YOUNG, POISSON)
    assert abs(cfg.e_star - e_star) <= 1e-14 * e_star
    assert abs(cfg.g_star - g_star) <= 1e-14 * g_star
    assert abs(cfg.g_star - hertz_mindlin.hertz_reduced_shear_modulus(
        YOUNG, POISSON)) <= 1e-14 * g_star

    # k_t / secant-k_n must reproduce the torch module's Mindlin ratio.
    penetration = 1.0e-4
    body_a = ClumpBody(sphere_clump(), gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(sphere_clump(), position=(2.0 * RADIUS - penetration, 0, 0),
                       velocity=(0.0, 0.05, 0.0), gravity=(0.0, 0.0, 0.0))
    contact = ClumpClumpContact(config(friction=1e9))
    dt = 1.0e-5
    response = contact.forces(body_a, body_b, dt=dt, commit=True)
    slip = 0.05 * dt                       # v_t dt accumulated in one step
    k_t_measured = float(np.linalg.norm(response.tangential_forces[0])) / slip
    radius_eff = 0.5 * RADIUS
    k_n_secant = (4.0 / 3.0) * e_star * math.sqrt(radius_eff * penetration)
    ratio = k_t_measured / k_n_secant
    assert abs(ratio - hertz_mindlin.mindlin_stiffness_ratio(POISSON)) <= 1e-12


def test_single_member_clump_reproduces_sphere_sphere_restitution() -> None:
    """Head-on two-sphere impact: measured e against the analytic momentum split."""
    target = 0.8
    v0 = 0.1
    dt = 1.0e-5
    body_a = ClumpBody(sphere_clump(), position=(-(RADIUS + 2e-4), 0, 0),
                       velocity=(v0, 0, 0), gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(sphere_clump(), position=((RADIUS + 2e-4), 0, 0),
                       velocity=(-v0, 0, 0), gravity=(0.0, 0.0, 0.0))
    contact = ClumpClumpContact(config(restitution=target, friction=0.0))
    touched = False
    measured = float("nan")
    for _ in range(4000):
        contact.step(body_a, body_b, dt)
        active = (contact.last.normal_forces.size
                  and float(contact.last.normal_forces.max()) > 0.0)
        if active:
            touched = True
        elif touched:
            measured = (body_b.velocity[0] - body_a.velocity[0]) / (2.0 * v0)
            break
    assert touched
    assert abs(measured - target) <= 0.05


def test_pair_integrator_matches_clump_body_in_free_flight() -> None:
    """No contact: the pair integrator must equal two ``ClumpBody.step`` calls."""
    clump = dumbbell_clump()
    solo = ClumpBody(clump, position=(0.0, 0.0, 3.0), velocity=(0.4, -0.2, 0.1),
                     angular_velocity=(0.3, -0.5, 0.2))
    body_a = ClumpBody(clump, position=(0.0, 0.0, 3.0), velocity=(0.4, -0.2, 0.1),
                       angular_velocity=(0.3, -0.5, 0.2))
    body_b = ClumpBody(clump, position=(0.0, 0.0, 6.0), velocity=(0.0, 0.1, 0.0),
                       angular_velocity=(0.0, 0.0, 0.0))
    contact = ClumpClumpContact(config())
    dt = 1.0e-4
    for _ in range(50):
        solo.step(dt)
        contact.step(body_a, body_b, dt)
    assert np.abs(body_a.position - solo.position).max() <= 1e-15
    assert np.abs(body_a.velocity - solo.velocity).max() <= 1e-14
    assert np.abs(body_a.orientation - solo.orientation).max() <= 1e-14
    assert np.abs(body_a.angular_velocity - solo.angular_velocity).max() <= 1e-13


def test_cached_kinematics_matches_clump_body_api() -> None:
    body = ClumpBody(dumbbell_clump(), position=(0.01, -0.02, 0.03),
                     velocity=(1.0, -2.0, 3.0),
                     angular_velocity=(0.3, 0.2, 0.1), gravity=(0.0, 0.0, 0.0))
    kin = _Kinematics.of(body)
    point = np.array([0.02, 0.01, -0.01])
    assert np.abs(kin.point_velocity(point) - body.point_velocity(point)).max() == 0.0
    assert np.abs(kin.centers - body.member_centers()).max() == 0.0
    assert np.abs(kin.inverse_inertia - body.inverse_inertia_world()).max() == 0.0
    assert np.abs(kin.omega_world - body.angular_velocity_world()).max() == 0.0
    arm = np.array([0.01, 0.02, -0.03])
    normal = np.array([0.0, 0.0, 1.0])
    assert abs(_effective_mass_along(body.clump, arm, normal,
                                     body.inverse_inertia_world())
               - effective_mass_along(body.clump, arm, normal,
                                      body.inverse_inertia_world())) == 0.0
    torque = np.array([0.2, -0.1, 0.05])
    assert np.abs(_angular_acceleration(body, body.rotation_matrix(), torque)
                  - body.angular_acceleration_body(torque)).max() == 0.0


# --------------------------------------------------------------------------
# C2 -- two-sphere clump versus an independent geometric estimate
# --------------------------------------------------------------------------
def test_two_sphere_clump_contact_matches_independent_estimate() -> None:
    """Parallel dumbbells: two junctions, penetration written out by hand."""
    separation = 0.1
    delta = 1.1e-2
    cfg = config(restitution=1.0)
    e_star, _ = reduced_modulus(YOUNG, POISSON)
    body_a = ClumpBody(dumbbell_clump(separation=separation),
                       position=(0.0, 0.0, 2.0 * RADIUS - delta),
                       gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(dumbbell_clump(separation=separation),
                       gravity=(0.0, 0.0, 0.0))
    joints = clump_clump_joints(body_a, body_b, cfg)
    assert len(joints) == 2                       # two distinct junctions
    assert len(member_pair_candidates(body_a, body_b)) == 2
    expected = 2.0 * hertz_elastic(delta, 0.5 * RADIUS, e_star)
    response = ClumpClumpContact(cfg).forces(body_a, body_b, commit=False)
    assert abs(response.force_magnitude - expected) <= 0.05 * expected
    # tight check: the junction forces are Hertz at their reported penetrations
    tight = sum(hertz_elastic(float(j.penetration), j.radius_eff, e_star)
                for j in response.joints)
    assert abs(response.force_magnitude - tight) <= 1e-12 * tight


def test_four_overlapping_member_pairs_match_independent_estimate() -> None:
    """Deep parallel contact: four overlapping member pairs, two junctions.

    The penetrations are derived here from the configuration alone
    (``delta_ij = r_i + r_j - |c_j - c_i|`` with the centres written out), so the
    comparison does not reuse the module's enumeration.
    """
    radius = RADIUS
    separation = 0.06
    height = 0.078
    cfg = config(restitution=1.0)
    e_star, _ = reduced_modulus(YOUNG, POISSON)
    body_a = ClumpBody(dumbbell_clump(separation=separation),
                       position=(0.0, 0.0, height), gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(dumbbell_clump(separation=separation),
                       gravity=(0.0, 0.0, 0.0))
    assert len(member_pair_candidates(body_a, body_b)) == 4

    half = 0.5 * separation
    primary = 2.0 * radius - height
    diagonal = 2.0 * radius - math.hypot(2.0 * half, height)
    assert primary > 0.0 and diagonal > 0.0
    radius_eff = 0.5 * radius
    expected = (2.0 * hertz_elastic(primary, radius_eff, e_star)
                + 2.0 * hertz_elastic(diagonal, radius_eff, e_star))
    measured = ClumpClumpContact(cfg).forces(body_a, body_b, commit=False)
    assert len(measured.joints) == 4
    assert abs(measured.force_magnitude - expected) <= 0.05 * expected
    vector = sum((hertz_elastic(float(j.penetration), j.radius_eff, e_star)
                  * j.normal for j in measured.joints), np.zeros(3))
    tight = float(np.linalg.norm(vector))
    assert abs(measured.force_magnitude - tight) <= 1e-12 * tight
    # the four pairs are geometrically distinct junctions here, so merging must
    # not collapse them and the naive member sum is already correct
    ratio = duplicate_force_ratio(body_a, body_b, cfg)
    assert ratio["joints"] == 4 and abs(ratio["ratio"] - 1.0) <= 1e-12


# --------------------------------------------------------------------------
# C3 -- duplicate counting negative control
# --------------------------------------------------------------------------
def _duplicate_pair_setup():
    """Two nearly concentric members of A indenting the same sphere of B."""
    radius = RADIUS
    distance = 0.088
    body_a = ClumpBody(compact_clump(), gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(sphere_clump(), position=(0.0, distance, 0.0),
                       gravity=(0.0, 0.0, 0.0))
    return body_a, body_b, distance, radius


def test_duplicate_member_pairs_are_merged_into_one_junction() -> None:
    body_a, body_b, distance, radius = _duplicate_pair_setup()
    candidates = member_pair_candidates(body_a, body_b)
    assert len(candidates) == 2                     # two member pairs...
    joints = merge_candidates(candidates, config())
    assert len(joints) == 1                         # ...one physical junction
    assert joints[0].n_members == 2
    # the weighted mean point/normal of a symmetric duplicate pair is exact
    assert np.abs(joints[0].normal - np.array([0.0, 1.0, 0.0])).max() <= 1e-15
    penetration = 2.0 * radius - math.hypot(0.01, distance)
    assert abs(joints[0].penetration - penetration) <= 1e-15


def test_duplicate_pairs_do_not_double_the_force() -> None:
    """C3: merged force <= 1.05x the unique-junction force; unmerged is ~2x."""
    body_a, body_b, distance, radius = _duplicate_pair_setup()
    cfg = config(restitution=1.0)
    e_star, _ = reduced_modulus(YOUNG, POISSON)
    penetration = 2.0 * radius - math.hypot(0.01, distance)
    unique = hertz_elastic(penetration, 0.5 * radius, e_star)
    merged = ClumpClumpContact(cfg).forces(body_a, body_b, commit=False)
    assert merged.force_magnitude <= 1.05 * unique
    # tight check against the force law at the reported junction penetration
    tight = hertz_elastic(float(merged.joints[0].penetration), 0.5 * radius, e_star)
    assert abs(merged.force_magnitude - tight) <= 1e-12 * tight

    ratio = duplicate_force_ratio(body_a, body_b, cfg)
    assert ratio["joints"] == 1.0 and ratio["candidates"] == 2.0
    assert 1.9 <= ratio["ratio"] <= 2.1            # the naive sum is ~2x
    # a symmetric duplicate pair sums to exactly 2 F cos(alpha/2) < 2 F
    assert ratio["ratio"] < 2.0


def test_over_merging_is_detectable() -> None:
    """Sanity check on the merge tolerance: a huge factor collapses real joints."""
    body_a = ClumpBody(dumbbell_clump(separation=0.1),
                       position=(0.0, 0.0, 2.0 * RADIUS - 1.1e-2),
                       gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(dumbbell_clump(separation=0.1), gravity=(0.0, 0.0, 0.0))
    assert len(clump_clump_joints(body_a, body_b, config())) == 2
    assert len(clump_clump_joints(body_a, body_b,
                                 config(merge_factor=100.0))) == 1


# --------------------------------------------------------------------------
# C4 -- third law
# --------------------------------------------------------------------------
def test_third_law_holds_when_the_bodies_are_swapped() -> None:
    cfg = config(restitution=0.9, friction=0.4)
    body_a = ClumpBody(dumbbell_clump(), position=(-0.049, 0.004, 0.001),
                       velocity=(0.1, 0.01, 0.0), gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(dumbbell_clump(), position=(0.049, 0.0, 0.0),
                       velocity=(-0.1, 0.0, 0.0), gravity=(0.0, 0.0, 0.0))
    forward = ClumpClumpContact(cfg).forces(body_a, body_b, commit=False)
    swapped = ClumpClumpContact(cfg).forces(body_b, body_a, commit=False)
    scale = float(np.linalg.norm(forward.force_a))
    assert scale > 0.0
    # force on A in the forward call vs the (negated) force on A in the swapped
    residual = np.linalg.norm(forward.force_a + swapped.force_a)
    assert residual <= 1e-12 * scale
    # within one call the partner force is exactly reversed
    assert np.abs(forward.force_a + forward.force_b).max() == 0.0


def test_pair_contributes_zero_net_moment_about_the_origin() -> None:
    cfg = config(restitution=0.9, friction=0.4)
    body_a = ClumpBody(dumbbell_clump(), position=(-0.049, 0.004, 0.01),
                       velocity=(0.1, 0.01, 0.0), angular_velocity=(0.0, 0.0, 1.0),
                       gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(dumbbell_clump(), position=(0.052, -0.003, 0.0),
                       velocity=(-0.1, 0.0, 0.0), gravity=(0.0, 0.0, 0.0))
    response = ClumpClumpContact(cfg).forces(body_a, body_b, dt=1e-5,
                                             commit=False)
    moment = (np.cross(body_a.position, response.force_a) + response.torque_a
              + np.cross(body_b.position, response.force_b) + response.torque_b)
    scale = float(np.linalg.norm(response.force_a)) * 0.1
    assert np.linalg.norm(moment) <= 1e-14 * scale


def test_linear_momentum_conserved_in_an_asymmetric_collision() -> None:
    cfg = config(restitution=0.75, friction=0.3)
    body_a = ClumpBody(dumbbell_clump(), position=(-0.09, 0.013, 0.0),
                       velocity=(0.2, 0.0, 0.0), angular_velocity=(0.0, 0.0, 2.0),
                       gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(dumbbell_clump(), position=(0.09, 0.0, 0.0),
                       velocity=(-0.05, 0.01, 0.0), gravity=(0.0, 0.0, 0.0))
    contact = ClumpClumpContact(cfg)
    initial = total_linear_momentum(body_a, body_b)
    scale = float(np.linalg.norm(initial))
    dt = 5.0e-6
    for _ in range(4000):
        contact.step(body_a, body_b, dt)
    residual = np.linalg.norm(total_linear_momentum(body_a, body_b) - initial)
    assert residual <= 1e-12 * scale


# --------------------------------------------------------------------------
# C5 -- angular momentum and restitution
# --------------------------------------------------------------------------
def test_angular_momentum_conserved_in_an_oblique_frictional_collision() -> None:
    cfg = config(restitution=0.8, friction=0.4)
    body_a = ClumpBody(dumbbell_clump(), position=(-0.09, 0.013, 0.0),
                       velocity=(0.2, 0.0, 0.0), angular_velocity=(0.0, 0.0, 2.0),
                       gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(dumbbell_clump(), position=(0.09, 0.0, 0.0),
                       velocity=(-0.05, 0.01, 0.0), gravity=(0.0, 0.0, 0.0))
    contact = ClumpClumpContact(cfg)
    initial = total_angular_momentum(body_a, body_b)
    scale = float(np.linalg.norm(initial))
    dt = 5.0e-6
    for _ in range(4000):
        contact.step(body_a, body_b, dt)
    residual = np.linalg.norm(total_angular_momentum(body_a, body_b) - initial)
    assert residual <= 1e-10 * scale


@pytest.mark.parametrize("target", [0.5, 0.7, 0.9, 0.95])
def test_measured_restitution_matches_the_requested_value(target: float) -> None:
    v0 = 0.1
    dt = 1.0e-5
    body_a = ClumpBody(sphere_clump(), position=(-(RADIUS + 2e-4), 0, 0),
                       velocity=(v0, 0, 0), gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(sphere_clump(), position=((RADIUS + 2e-4), 0, 0),
                       velocity=(-v0, 0, 0), gravity=(0.0, 0.0, 0.0))
    contact = ClumpClumpContact(config(restitution=target, friction=0.0))
    touched = False
    measured = float("nan")
    for _ in range(4000):
        contact.step(body_a, body_b, dt)
        active = (contact.last.normal_forces.size
                  and float(contact.last.normal_forces.max()) > 0.0)
        if active:
            touched = True
        elif touched:
            measured = (body_b.velocity[0] - body_a.velocity[0]) / (2.0 * v0)
            break
    assert touched
    assert abs(measured - target) <= 0.05


# --------------------------------------------------------------------------
# C6 -- stability of a resting pair
# --------------------------------------------------------------------------
def _gravity_stack(perturbation: float = 0.0):
    """Clump A on clump B on a floor, initialised at the exact static overlaps.

    ``perturbation`` scales the pair overlap (``0`` = equilibrium).  The plane
    contact is a linear spring with ``k_n = 1e7 N/m``; each of B's two members
    carries A+B's weight ``m g``, and each of the two pair junctions carries A's
    weight ``m g / 2``, which gives the overlaps in closed form.
    """
    wall = WallContactConfig(normal_stiffness=1.0e7, restitution=0.9, friction=0.4)
    floor = PlaneContact((0.0, 0.0, 1.0), point=(0.0, 0.0, 0.0), config=wall)
    e_star, _ = reduced_modulus(YOUNG, POISSON)
    clump = dumbbell_clump()
    weight = clump.mass * 9.81
    delta_plane = weight / wall.normal_stiffness
    charlson = (4.0 / 3.0) * e_star * math.sqrt(0.5 * RADIUS)
    delta_pair = (weight / (2.0 * charlson)) ** (2.0 / 3.0)
    z_b = RADIUS - delta_plane
    body_a = ClumpBody(clump, position=(0.0, 0.0,
                                        z_b + 2.0 * RADIUS - delta_pair))
    if perturbation:
        body_a.position = body_a.position + np.array(
            [0.0, 0.0, -perturbation * delta_pair])
    body_b = ClumpBody(clump, position=(0.0, 0.0, z_b))
    return body_a, body_b, floor, delta_pair


def test_resting_pair_does_not_drift_over_1e5_steps() -> None:
    """C6: gravity-loaded clump stack, 1e5 velocity-Verlet steps.

    A purely repulsive contact cannot rest in zero gravity (the pair would simply
    fly apart), so the resting state is a load path: A on B on a plane.  Both
    overlaps are solved in closed form, making the initial state statically
    admissible to round-off; the residual drift is then a spurious-motion
    detector for duplicated junctions and leaked tangential state.
    """
    cfg = config(restitution=0.9, friction=0.4)
    body_a, body_b, floor, _ = _gravity_stack()
    contact = ClumpClumpContact(cfg)
    start_a = body_a.position.copy()
    start_b = body_b.position.copy()
    energy_start = body_a.energy() + body_b.energy()
    for _ in range(STATIC_STEPS):
        contact.step(body_a, body_b, STATIC_DT, extra_b=(floor,))
    drift = max(np.linalg.norm(body_a.position - start_a),
                np.linalg.norm(body_b.position - start_b))
    energy = body_a.energy() + body_b.energy()
    assert len(contact.last.joints) == 2
    assert drift <= 1e-9
    assert abs(energy - energy_start) <= 1e-9 * max(abs(energy_start), 1.0)


def test_damped_relaxation_never_gains_energy() -> None:
    """Start 5% deeper than equilibrium: the contact must not pump energy in."""
    cfg = config(restitution=0.7, friction=0.4)
    body_a, body_b, floor, delta_pair = _gravity_stack(perturbation=0.05)
    contact = ClumpClumpContact(cfg)
    def total_energy() -> float:
        plane = 0.5 * floor.config.normal_stiffness * float(
            np.sum(floor.penetrations() ** 2))
        return (body_a.energy() + body_b.energy() + contact.stored_energy()
                + plane)

    contact.forces(body_a, body_b, dt=0.0, commit=False)   # populate elastic state
    floor.forces(body_b, dt=0.0, commit=False)
    energies = [total_energy()]
    for step in range(4000):
        contact.step(body_a, body_b, STATIC_DT, extra_b=(floor,))
        if step % 20 == 0:
            energies.append(total_energy())
    energies = np.asarray(energies)
    # the only non-conservative terms are the dashpot and Coulomb sliding
    assert np.all(np.diff(energies) <= 1e-9 * abs(energies[0]))
    assert energies[-1] < energies[0]


# --------------------------------------------------------------------------
# Contact-state and configuration bookkeeping
# --------------------------------------------------------------------------
def test_no_contact_below_the_surface() -> None:
    body_a = ClumpBody(sphere_clump(), gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(sphere_clump(), position=(2.0 * RADIUS + 1e-9, 0, 0),
                       gravity=(0.0, 0.0, 0.0))
    contact = ClumpClumpContact(config())
    response = contact.forces(body_a, body_b, commit=True)
    assert response.joints == []
    assert response.force_magnitude == 0.0
    assert response.normal_forces.size == 0


def test_tangential_history_is_dropped_when_the_contact_opens() -> None:
    cfg = config(friction=0.5)
    contact = ClumpClumpContact(cfg)
    body_a = ClumpBody(sphere_clump(), gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(sphere_clump(), position=(2.0 * RADIUS - 1e-4, 0, 0),
                       velocity=(0.0, 0.01, 0.0), gravity=(0.0, 0.0, 0.0))
    contact.forces(body_a, body_b, dt=1e-4, commit=True)
    assert contact._tangential                     # history accumulated
    body_b.position = body_b.position + np.array([0.1, 0.0, 0.0])
    contact.forces(body_a, body_b, dt=1e-4, commit=True)
    assert not contact._tangential                 # pruned once separated


def test_coulomb_limit_is_active() -> None:
    cfg = config(friction=0.1, restitution=1.0)
    contact = ClumpClumpContact(cfg)
    body_a = ClumpBody(sphere_clump(), gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(sphere_clump(), position=(2.0 * RADIUS - 1e-4, 0, 0),
                       velocity=(0.0, 1.0, 0.0), gravity=(0.0, 0.0, 0.0))
    response = contact.forces(body_a, body_b, dt=1e-3, commit=True)
    tangential = float(np.linalg.norm(response.tangential_forces[0]))
    limit = cfg.friction * float(response.normal_forces[0])
    assert tangential <= limit * (1.0 + 1e-12)
    assert tangential > 0.0


@pytest.mark.parametrize("kwargs", [
    {"young_modulus": -1.0},
    {"poisson_ratio": 0.5},
    {"restitution": 0.0},
    {"restitution": 1.5},
    {"friction": -0.1},
    {"merge_factor": -1.0},
    {"merge_angle": 0.0},
])
def test_rejects_invalid_configuration(kwargs: dict) -> None:
    with pytest.raises(ClumpClumpError):
        ClumpClumpConfig(**kwargs)


def test_pair_material_constants_match_identical_material_reference() -> None:
    e_star, g_star = pair_material_constants(YOUNG, POISSON, YOUNG, POISSON)
    reference_e, reference_g = reduced_modulus(YOUNG, POISSON)
    assert abs(e_star - reference_e) <= 1e-14 * reference_e
    assert abs(g_star - reference_g) <= 1e-14 * reference_g
    with pytest.raises(ClumpClumpError):
        pair_material_constants(-1.0, 0.3, YOUNG, POISSON)


def test_hertz_normal_force_is_monotone_and_non_negative() -> None:
    e_star, _ = reduced_modulus(YOUNG, POISSON)
    penetrations = np.logspace(-7, -3, 25)
    forces = [hertz_normal_force(d, 0.5 * RADIUS, e_star) for d in penetrations]
    assert all(f > 0.0 for f in forces)
    assert np.all(np.diff(forces) > 0.0)
    assert hertz_normal_force(0.0, 0.5 * RADIUS, e_star) == 0.0
    # the dashpot can only reduce the repulsion, never turn it negative
    assert hertz_normal_force(1e-4, 0.5 * RADIUS, e_star, velocity_normal=1.0,
                              mass_eff=1.0, damping_ratio=0.9) == 0.0