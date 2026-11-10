"""Acceptance tests for the ship-ice collision energy-conservation surface.

Every numerical check here is against an *independent* route or a published
relation - never the module compared with itself:

* section integral ``Z = b*t^2/4``          vs fibre quadrature of the same section
* mechanism energy ``E = M_p*theta``        vs external work ``P_u*d`` (virtual work)
* energy balance residual                   vs ``E_k - (E_struct + E_friction)``
* layer allocation ``E_total = sum E_k``    vs Simpson quadrature of ``dE/dd``
* damaged volume                            vs Minorsky (1959) empirical line

Criteria (a)-(f) of campaign node IN63-N01 map one-to-one onto tests named
``test_criterion_*``.
"""
import math

import pytest

from tensordem.hull_collision import (
    CapacityExhausted,
    DoubleHullSide,
    SideShellLayer,
    collapse_load,
    compare_damage_volume,
    damage_geometry,
    hinge_energy,
    hinge_rotations,
    kinetic_energy,
    minorsky_coefficient,
    minorsky_damage_volume,
    minorsky_energy_J,
    plastic_moment,
    plastic_moment_rect_fiber,
    plastic_section_modulus_rect,
    section_plastic_modulus,
    solve_penetration,
    three_point_bending_anchor,
    verify_layer_decomposition,
)
from tensordem.hull_collision.scenario import (
    CONTACT_HEIGHT,
    CONTACT_LENGTH,
    FRAME_SPAN,
    SCRATCH_SIGMA_Y,
    design_case,
    effective_mass,
)

SWEEP_VELOCITIES = (2.0, 4.0, 6.0, 8.0)
SEVERE_MASS = effective_mass()          # 2.2e6 kg, E(V=8) = 70.4 MJ
OVERLOAD_MASS = 4.5e6                   # E(V=6) = 81 MJ -> inner shell engaged


# --------------------------------------------------------------- section level
def test_rect_plastic_modulus_matches_fibre_quadrature():
    for width, thickness in ((1.0, 0.02), (0.6, 0.022), (2.5, 0.008)):
        z_closed = plastic_section_modulus_rect(width, thickness)
        m_fiber = plastic_moment_rect_fiber(SCRATCH_SIGMA_Y, width, thickness, n_fibers=20001)
        assert m_fiber / SCRATCH_SIGMA_Y == pytest.approx(z_closed, rel=1e-6)


def test_numerical_section_modulus_reproduces_solid_rectangle():
    z_general = section_plastic_modulus([(-0.01, 0.01, 1.0)])
    assert z_general == pytest.approx(plastic_section_modulus_rect(1.0, 0.02), rel=1e-9)


def test_numerical_section_modulus_is_datum_invariant():
    a = section_plastic_modulus([(0.0, 0.02, 0.6), (0.02, 0.42, 0.011), (0.42, 0.437, 0.15)])
    shift = 123.25
    b = section_plastic_modulus(
        [(y0 + shift, y1 + shift, w) for y0, y1, w in ((0.0, 0.02, 0.6), (0.02, 0.42, 0.011), (0.42, 0.437, 0.15))]
    )
    assert a == pytest.approx(b, rel=1e-9)


def test_plastic_moment_is_yield_times_modulus():
    z = plastic_section_modulus_rect(0.6, 0.022)
    assert plastic_moment(SCRATCH_SIGMA_Y, z) == pytest.approx(SCRATCH_SIGMA_Y * z)


def test_hinge_rotations_mechanism_kinematics():
    assert hinge_rotations(0.12, 2.4, bc="clamped") == pytest.approx(8 * 0.12 / 2.4)
    assert hinge_rotations(0.12, 2.4, bc="simply_supported") == pytest.approx(4 * 0.12 / 2.4)
    with pytest.raises(ValueError):
        hinge_rotations(0.1, 2.4, bc="pinned_pinned")
    with pytest.raises(ValueError):
        hinge_rotations(2.0, 2.4)  # beyond small-rotation validity


def test_collapse_load_equals_hinge_energy_over_deflection():
    """Energy identity of the mechanism: ``P_u*d == M_p*theta(d)``."""
    z = plastic_section_modulus_rect(0.6, 0.022)
    mp = plastic_moment(SCRATCH_SIGMA_Y, z)
    for bc in ("clamped", "simply_supported"):
        for d in (0.01, 0.1, 0.3):
            p_u = collapse_load(mp, 2.4, bc=bc)
            assert p_u * d == pytest.approx(hinge_energy(mp, d, 2.4, bc=bc), rel=1e-12)


# ------------------------------------------------------------ criterion (b)
@pytest.mark.parametrize("bc", ["simply_supported", "clamped"])
def test_criterion_b_three_point_bending_anchor_within_one_percent(bc):
    anchor = three_point_bending_anchor(
        sigma_y=SCRATCH_SIGMA_Y,
        width=0.6,
        thickness=0.022,
        span=FRAME_SPAN,
        deflection=0.12,
        bc=bc,
        n_steps=2048,
    )
    assert anchor["max_rel_error"] <= 0.01
    assert anchor["rel_error_incremental_vs_closed"] <= 0.01
    assert anchor["rel_error_virtual_work_vs_closed"] <= 0.01
    # the closed form really is E = M_p*theta_max, not a proxy
    assert anchor["energy_closed_J"] == pytest.approx(
        anchor["plastic_moment_closed_Nm"] * anchor["theta_max_rad"], rel=1e-12
    )


# ------------------------------------------------------------ criterion (c)
def test_criterion_c_zero_impact_velocity_gives_zero_penetration():
    structure = design_case()
    solution = solve_penetration(kinetic_energy(SEVERE_MASS, 0.0), structure)
    assert solution.penetration_depth == 0.0
    assert solution.iterations == 0
    assert solution.structural_energy == 0.0
    assert solution.friction_energy == 0.0
    assert solution.inner_shell_intact is True
    # and a non-zero velocity must *not* give zero depth (negative-control flip)
    assert solve_penetration(kinetic_energy(SEVERE_MASS, 2.0), structure).penetration_depth > 0.0


# ------------------------------------------------------------ criterion (a)
@pytest.mark.parametrize("velocity", SWEEP_VELOCITIES)
def test_criterion_a_energy_conservation_residual(velocity):
    structure = design_case()
    solution = solve_penetration(kinetic_energy(SEVERE_MASS, velocity), structure)
    assert solution.residual_relative <= 0.02
    # the residual is the *reported* difference between the three terms
    assert solution.residual == pytest.approx(
        solution.kinetic_energy - solution.structural_energy - solution.friction_energy,
        rel=0.0,
        abs=1e-6,
    )
    # friction is mu times the structural work (work-average normal force form)
    assert solution.friction_energy == pytest.approx(
        structure.friction_coefficient * solution.structural_energy, rel=1e-12
    )


def test_friction_reduces_penetration():
    from dataclasses import replace

    structure = design_case()
    energy = kinetic_energy(SEVERE_MASS, 6.0)
    with_friction = solve_penetration(energy, structure).penetration_depth
    no_friction = solve_penetration(energy, replace(structure, friction_coefficient=0.0)).penetration_depth
    assert no_friction > with_friction


# ------------------------------------------------------------ criterion (f)
def test_criterion_f_penetration_monotone_in_velocity():
    structure = design_case()
    depths = [solve_penetration(kinetic_energy(SEVERE_MASS, v), structure).penetration_depth for v in SWEEP_VELOCITIES]
    assert all(b > a for a, b in zip(depths, depths[1:])), depths
    energies = [s.kinetic_energy for s in (solve_penetration(kinetic_energy(SEVERE_MASS, v), structure) for v in SWEEP_VELOCITIES)]
    assert all(b > a for a, b in zip(energies, energies[1:]))


# ------------------------------------------------------------ criterion (d)
def test_criterion_d_layer_allocation_sums_to_total():
    structure = design_case()
    solution = solve_penetration(kinetic_energy(OVERLOAD_MASS, 6.0), structure)  # E = 81 MJ
    assert len(solution.layer_energies) == 2
    outer, inner = solution.layer_energies
    assert outer > 0.0 and inner > 0.0, "overload case must engage the inner shell"
    assert sum(solution.layer_energies) == pytest.approx(solution.structural_energy, rel=1e-12)
    # independent route: Simpson quadrature of the force curve
    check = verify_layer_decomposition(structure, solution.penetration_depth, n_intervals=20000)
    assert check["rel_error"] <= 0.02
    assert abs(check["energy_quadrature_J"] - solution.structural_energy) / solution.structural_energy <= 0.02


def test_criterion_d_outer_absorbs_before_inner():
    structure = design_case()
    outer_capacity = structure.layers[0].capacity
    below = solve_penetration(0.9 * outer_capacity, structure)
    assert below.inner_shell_intact is True
    assert below.layer_energies[1] == 0.0
    assert below.layer_energies[0] == pytest.approx(below.structural_energy, rel=1e-12)


def test_criterion_d_inner_shell_intact_flag():
    structure = design_case()
    intact = solve_penetration(kinetic_energy(SEVERE_MASS, 8.0), structure)
    assert intact.inner_shell_intact is True
    assert intact.breached_layers == ("outer_shell",)
    breached = solve_penetration(kinetic_energy(OVERLOAD_MASS, 6.0), structure)
    assert breached.inner_shell_intact is False
    assert set(breached.breached_layers) == {"outer_shell", "inner_shell"}
    assert breached.layer_local_penetrations[1] >= structure.layers[1].thickness


def test_capacity_exhausted_is_reported_not_silently_clipped():
    structure = design_case()
    with pytest.raises(CapacityExhausted):
        solve_penetration(1.2 * structure.total_capacity, structure)


# ------------------------------------------------------------ criterion (e)
def test_criterion_e_minorsky_cross_check_in_calibration_range():
    structure = design_case()
    solution = solve_penetration(kinetic_energy(SEVERE_MASS, 8.0), structure)   # 70.4 MJ, severe
    geometry = damage_geometry(structure, solution, contact_length=CONTACT_LENGTH)
    comparison = compare_damage_volume(
        model_volume_m3=geometry.volume_m3,
        mass=SEVERE_MASS,
        velocity=8.0,
        sigma_y=SCRATCH_SIGMA_Y,
    )
    assert comparison["in_minorsky_range"] is True
    assert comparison["consistent_formulations"] is True
    assert abs(comparison["relative_error"]) <= 0.30


def test_criterion_e_minorsky_cross_check_overload_case():
    structure = design_case()
    solution = solve_penetration(kinetic_energy(OVERLOAD_MASS, 6.0), structure)  # 81 MJ
    geometry = damage_geometry(structure, solution, contact_length=CONTACT_LENGTH)
    comparison = compare_damage_volume(
        model_volume_m3=geometry.volume_m3,
        mass=OVERLOAD_MASS,
        velocity=6.0,
        sigma_y=SCRATCH_SIGMA_Y,
    )
    assert comparison["in_minorsky_range"] is True
    assert abs(comparison["relative_error"]) <= 0.30


def test_minorsky_relation_round_trip_and_coefficient_derivation():
    assert minorsky_energy_J(1.0) == pytest.approx(47.6e6, rel=1e-12)
    assert minorsky_damage_volume(minorsky_energy_J(3.0)) == pytest.approx(3.0, rel=1e-12)
    # C reproduces Minorsky's line exactly at the reference yield stress
    assert minorsky_coefficient() == pytest.approx(235.0e6 / (2 * 47.1e6), rel=1e-12)
    comparison = compare_damage_volume(
        model_volume_m3=1.0, mass=2.35e6 * 0.0 + 2.0e6, velocity=7.0, sigma_y=235.0e6
    )
    assert comparison["volume_minorsky_m3"] == pytest.approx(
        comparison["kinetic_energy_J"] / 47.1e6, rel=1e-12
    )


def test_low_energy_cases_are_flagged_outside_minorsky_range():
    structure = design_case()
    solution = solve_penetration(kinetic_energy(SEVERE_MASS, 2.0), structure)  # 4.4 MJ
    geometry = damage_geometry(structure, solution, contact_length=CONTACT_LENGTH)
    comparison = compare_damage_volume(
        model_volume_m3=geometry.volume_m3, mass=SEVERE_MASS, velocity=2.0, sigma_y=SCRATCH_SIGMA_Y
    )
    assert comparison["in_minorsky_range"] is False


# ------------------------------------------------------------------ geometry
def test_damage_geometry_extents_and_volume():
    structure = design_case()
    solution = solve_penetration(kinetic_energy(SEVERE_MASS, 8.0), structure)
    geometry = damage_geometry(structure, solution, contact_length=CONTACT_LENGTH)
    assert geometry.length_m == pytest.approx(CONTACT_LENGTH + 2 * FRAME_SPAN)
    assert geometry.depth_m == pytest.approx(solution.penetration_depth)
    assert geometry.width_m == pytest.approx(CONTACT_HEIGHT)
    assert geometry.volume_m3 == pytest.approx(sum(geometry.layer_volumes_m3), rel=1e-12)
    assert geometry.inner_shell_intact is True
    geometry_overload = damage_geometry(
        structure,
        solve_penetration(kinetic_energy(OVERLOAD_MASS, 6.0), structure),
        contact_length=CONTACT_LENGTH,
    )
    assert geometry_overload.volume_m3 > geometry.volume_m3


def test_invalid_inputs_are_rejected():
    structure = design_case()
    with pytest.raises(ValueError):
        solve_penetration(-1.0, structure)
    with pytest.raises(TypeError):
        solve_penetration(1.0, object())
    with pytest.raises(ValueError):
        SideShellLayer("x", 0.0, 355e6, 2.4, 3.0, 5, 1e-3, 1e-3, 2.0)
    with pytest.raises(ValueError):
        DoubleHullSide(layers=())
