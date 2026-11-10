"""Acceptance tests for the 3-D contact-to-resistance mapping closure (P316-W1-N09).

Every criterion of the node has at least one test that would fail if the
capability regressed:

* closure (:func:`decompose`), negative control, taxonomy priority, sign
  convention, input validation, time averaging   -> pure layer, no DEM;
* calibration identities (cap, engagement stiffness, characteristic length,
  moment capacity, Lindqvist equivalence)        -> closed-form checks;
* solver wiring (contact records, bond-graph detachment) -> tiny DEM cases.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.threefield.resistance import Environment, HullGeometry, IceProperties, lindqvist_resistance
from tensordem.resistance_closure import (
    BENDING,
    CRUSHING,
    MECHANISMS,
    SUBMERSION,
    ClosureCase,
    HullContactSet,
    LindqvistReference,
    average_decompositions,
    build_reference,
    compare_components,
    decompose,
    flexural_demand_ratio,
    flexural_moment_capacity,
    plate_characteristic_length,
    unit_heading,
)
from tensordem.resistance_closure.dem import (
    CappedBowDEM,
    build_simulation,
    cap_engagement_stiffness,
    run_case,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _contacts(
    forces=((10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (30.0, 0.0, -5.0)),
    *,
    detached=(False, False, True),
    caps=(math.inf, 20.0, math.inf),
    positions=((0.0, 0.0, 0.0), (0.0, 0.0, 0.1), (0.0, 0.0, -0.1)),
) -> HullContactSet:
    """Build a contact set whose normal force is the x-component of ``forces``.

    The third contact carries a downward friction component so that it is a
    *submersion* contact (detached **and** pressing the ice down); its horizontal
    contribution is still the x-component.
    """
    count = len(forces)
    force = torch.tensor(forces, dtype=torch.float64)
    normals = torch.zeros(count, 3, dtype=torch.float64)
    normals[:, 0] = 1.0
    normal_force = force[:, 0].clone()
    friction = force.clone()
    friction[:, 0] = 0.0
    return HullContactSet(
        positions=torch.tensor(positions, dtype=torch.float64),
        normals=normals,
        normal_force=normal_force,
        friction_force=friction,
        penetration=torch.full((count,), 0.01, dtype=torch.float64),
        cap_force=torch.tensor(caps, dtype=torch.float64),
        detached=torch.tensor(detached, dtype=torch.bool),
        submerged_fraction=torch.full((count,), 0.5, dtype=torch.float64),
    )


def _tiny_case(**changes) -> ClosureCase:
    base = dict(
        thickness_m=0.08, particle_radius_m=0.02, n_length=4, n_width=2,
        bow_speed_m_s=0.1, steps=3, contact_mask_every=1, embed_m=0.005, mass_scale=1.0,
    )
    base.update(changes)
    return ClosureCase.physical(**base)


# ---------------------------------------------------------------------------
# criterion (a): closure of the three-component decomposition
# ---------------------------------------------------------------------------
def test_empty_contact_set_is_exactly_zero():
    empty = HullContactSet(
        positions=torch.zeros(0, 3, dtype=torch.float64),
        normals=torch.zeros(0, 3, dtype=torch.float64),
        normal_force=torch.zeros(0, dtype=torch.float64),
        friction_force=torch.zeros(0, 3, dtype=torch.float64),
        penetration=torch.zeros(0, dtype=torch.float64),
        cap_force=torch.zeros(0, dtype=torch.float64),
        detached=torch.zeros(0, dtype=torch.bool),
        submerged_fraction=torch.zeros(0, dtype=torch.float64),
    )
    result = decompose(empty)
    assert result.r_total == 0.0
    assert result.residual == 0.0
    assert result.relative_residual == 0.0
    assert result.components == {name: 0.0 for name in MECHANISMS}
    assert result.shares == {name: 0.0 for name in MECHANISMS}
    assert result.contact_count == 0


def test_closure_residual_is_round_off_for_mixed_contacts():
    result = decompose(_contacts())
    assert result.r_total == pytest.approx(60.0)
    assert result.residual == pytest.approx(0.0, abs=1e-12)
    assert result.relative_residual <= 1e-15
    assert sum(result.components.values()) == pytest.approx(result.r_total)


def test_component_membership_matches_the_taxonomy():
    result = decompose(_contacts())
    # contact 0: uncapped, attached   -> bending
    # contact 1: capped, attached     -> crushing
    # contact 2: detached             -> submersion overrides everything else
    assert result.components["bending"] == pytest.approx(10.0)
    assert result.components["crushing"] == pytest.approx(20.0)
    assert result.components["submersion"] == pytest.approx(30.0)
    assert result.contact_counts == {"crushing": 1, "bending": 1, "submersion": 1}
    assert result.capped_count == 1
    assert result.detached_count == 1


def test_detached_ice_overrides_a_capped_contact():
    contacts = _contacts(
        forces=((5.0, 0.0, -1.0),), detached=(True,), caps=(5.0,),
        positions=((0.0, 0.0, 0.0),),
    )
    assert contacts.saturated().tolist() == [True]
    assert contacts.labels().tolist() == [SUBMERSION]
    result = decompose(contacts)
    assert result.components["crushing"] == 0.0
    assert result.components["submersion"] == pytest.approx(5.0)
    # ... while detached ice squeezed forward (no downward force) is crushing
    squeezed = _contacts(
        forces=((5.0, 0.0, 1.0),), detached=(True,), caps=(5.0,),
        positions=((0.0, 0.0, 0.0),),
    )
    assert squeezed.labels().tolist() == [CRUSHING]


def test_below_cap_is_not_saturated():
    contacts = _contacts(forces=((19.999, 0.0, 0.0),), detached=(False,), caps=(20.0,),
                         positions=((0.0, 0.0, 0.0),))
    assert contacts.saturated().tolist() == [False]
    assert contacts.labels().tolist() == [BENDING]
    at_cap = _contacts(forces=((20.0, 0.0, 0.0),), detached=(False,), caps=(20.0,),
                       positions=((0.0, 0.0, 0.0),))
    assert at_cap.saturated().tolist() == [True]
    assert at_cap.labels().tolist() == [CRUSHING]


def test_heading_sign_convention_and_normalisation():
    contacts = _contacts()
    forward = decompose(contacts, heading=(3.0, 0.0, 0.0))
    assert forward.r_total == pytest.approx(60.0)
    assert forward.reaction_vector[0] == pytest.approx(-60.0)
    backward = decompose(contacts, heading=(-1.0, 0.0, 0.0))
    assert backward.r_total == pytest.approx(-60.0)
    assert sum(backward.components.values()) == pytest.approx(-60.0)
    with pytest.raises(ValueError):
        unit_heading((0.0, 0.0, 0.0))


def test_transverse_forces_do_not_leak_into_the_heading_resistance():
    contacts = _contacts(
        forces=((0.0, 50.0, 0.0), (0.0, 0.0, -25.0)), detached=(False, False),
        caps=(math.inf, math.inf), positions=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    result = decompose(contacts)
    assert result.r_total == pytest.approx(0.0)
    assert result.lateral_reaction == pytest.approx(-50.0)
    assert result.vertical_reaction == pytest.approx(25.0)


def test_invalid_contact_sets_are_rejected():
    bad_normals = _contacts()
    with pytest.raises(ValueError):
        HullContactSet(
            positions=bad_normals.positions,
            normals=bad_normals.normals * 2.0,
            normal_force=bad_normals.normal_force,
            friction_force=bad_normals.friction_force,
            penetration=bad_normals.penetration,
            cap_force=bad_normals.cap_force,
            detached=bad_normals.detached,
            submerged_fraction=bad_normals.submerged_fraction,
        )
    with pytest.raises(ValueError):  # negative normal force
        HullContactSet(
            positions=bad_normals.positions,
            normals=bad_normals.normals,
            normal_force=-bad_normals.normal_force,
            friction_force=bad_normals.friction_force,
            penetration=bad_normals.penetration,
            cap_force=bad_normals.cap_force,
            detached=bad_normals.detached,
            submerged_fraction=bad_normals.submerged_fraction,
        )
    with pytest.raises(ValueError):  # wrong shape
        HullContactSet(
            positions=torch.zeros(2, 2, dtype=torch.float64),
            normals=torch.zeros(2, 3, dtype=torch.float64),
            normal_force=torch.zeros(2, dtype=torch.float64),
            friction_force=torch.zeros(2, 3, dtype=torch.float64),
            penetration=torch.zeros(2, dtype=torch.float64),
            cap_force=torch.zeros(2, dtype=torch.float64),
            detached=torch.zeros(2, dtype=torch.bool),
            submerged_fraction=torch.zeros(2, dtype=torch.float64),
        )


def test_time_average_preserves_closure():
    snapshots = [decompose(_contacts()), decompose(_contacts(forces=((1.0, 0.0, 0.0),
                                                                    (2.0, 0.0, 0.0),
                                                                    (4.0, 0.0, -1.0))))]
    averaged = average_decompositions(snapshots)
    assert averaged.r_total == pytest.approx((60.0 + 7.0) / 2.0)
    assert averaged.residual == pytest.approx(0.0, abs=1e-12)
    # second snapshot: 1 (uncapped) + 2 (uncapped, below its cap) -> bending,
    # 4 (detached) -> submersion, no crushing contact at all
    assert averaged.components["crushing"] == pytest.approx(20.0 / 2.0)
    assert averaged.components["bending"] == pytest.approx((10.0 + 3.0) / 2.0)
    assert averaged.components["submersion"] == pytest.approx((30.0 + 4.0) / 2.0)
    with pytest.raises(ValueError):
        average_decompositions([])


# ---------------------------------------------------------------------------
# calibration identities
# ---------------------------------------------------------------------------
def test_plastic_cap_is_zero_at_no_penetration_and_monotone():
    case = _tiny_case()
    sim = build_simulation(case)
    penetrations = torch.tensor([0.0, 1e-4, 1e-3, 1e-2], dtype=torch.float64)
    caps = sim.plastic_cap(penetrations)
    assert caps[0].item() == 0.0
    assert bool((caps[1:] > caps[:-1]).all())
    # chord(d) = 2 sqrt(2 r d - d^2) -> cap ~ sqrt(d) in the small-penetration limit
    small = sim.plastic_cap(torch.tensor([1e-8, 4e-8], dtype=torch.float64))
    assert small[1] / small[0] == pytest.approx(2.0, rel=1e-3)


def test_cap_engagement_stiffness_satisfies_its_definition():
    pressure, radius, fraction = 7.5e5, 0.04, 0.1
    stiffness = cap_engagement_stiffness(pressure, radius, fraction)
    penetration = fraction * radius
    chord = 2.0 * math.sqrt(2.0 * radius * penetration - penetration ** 2)
    cap = pressure * chord * 2.0 * radius
    assert stiffness * penetration == pytest.approx(cap, rel=1e-12)
    with pytest.raises(ValueError):
        cap_engagement_stiffness(pressure, radius, 1.5)


def test_contact_stiffness_is_cap_controlled_not_penalty_controlled():
    """Above the engagement stiffness the contact force equals the cap."""
    case = _tiny_case()
    sim = build_simulation(case)
    penetration = torch.tensor([0.2 * case.particle_radius_m], dtype=torch.float64)
    cap = sim.plastic_cap(penetration)
    stiff = sim.config.contact_stiffness * penetration
    assert float(stiff) > float(cap)  # penalty would exceed the physical cap


def test_characteristic_length_and_moment_capacity_scaling():
    l1 = plate_characteristic_length(5.0e9, 0.4)
    l2 = plate_characteristic_length(5.0e9 * 16.0, 0.4)
    assert l2 / l1 == pytest.approx(2.0, rel=1e-12)  # l ~ E^(1/4)
    l3 = plate_characteristic_length(5.0e9, 0.8)
    assert l3 / l1 == pytest.approx(2.0 ** 0.75, rel=1e-12)  # l ~ h^(3/4)
    assert flexural_moment_capacity(5e5, 0.4) == pytest.approx(5e5 * 0.16 / 6.0)
    assert flexural_moment_capacity(1e6, 0.4) == pytest.approx(2 * flexural_moment_capacity(5e5, 0.4))


def test_flexural_demand_ratio_scales_with_load_and_thickness():
    base = flexural_demand_ratio(1.0e3, 7.0, 5e5, 0.4)
    assert flexural_demand_ratio(2.0e3, 7.0, 5e5, 0.4) == pytest.approx(2 * base)
    thicker = flexural_demand_ratio(1.0e3, plate_characteristic_length(5e9, 0.8), 5e5, 0.8)
    assert thicker < base  # thicker ice carries the same line load more easily


def test_physical_case_derives_the_bond_law_from_the_material():
    case = _tiny_case()
    radius = case.particle_radius_m
    assert case.bond_stiffness == pytest.approx(case.elastic_modulus_pa * 2 * radius)
    assert case.breaking_strain == pytest.approx(case.flexural_strength_pa / case.elastic_modulus_pa)
    assert case.crushing_pressure_pa == pytest.approx(case.crushing_efficiency * case.compressive_strength_pa)
    assert case.effective_density_kg_m3 == pytest.approx(case.density_kg_m3 * case.mass_scale)
    assert case.effective_water_density_kg_m3 == pytest.approx(case.water_density_kg_m3 * case.mass_scale)
    # floating equilibrium is invariant under mass scaling
    assert case.submerged_fraction_rest == pytest.approx(917.0 / 1025.0)


def test_case_geometry_properties():
    case = _tiny_case()
    assert case.layers == max(2, round(case.thickness_m / (2 * case.particle_radius_m)))
    assert case.realised_thickness_m == pytest.approx(case.layers * 2 * case.particle_radius_m)
    assert case.model_width_m == pytest.approx(case.n_width * 2 * case.particle_radius_m)
    assert case.model_length_m == pytest.approx((case.n_length - 1) * 2 * case.particle_radius_m)
    assert case.water_level_m == pytest.approx(case.realised_thickness_m * case.submerged_fraction_rest)


# ---------------------------------------------------------------------------
# Lindqvist reference reduction
# ---------------------------------------------------------------------------
def test_reference_reproduces_the_threefield_components():
    reference = build_reference(thickness_m=0.6, beam_m=20.0, length_m=120.0,
                                entry_angle_deg=22.0, speed_m_s=2.0, friction_coefficient=0.05)
    direct = lindqvist_resistance(
        IceProperties(thickness_m=0.6), HullGeometry(beam_m=20.0, length_m=120.0,
                                                    waterline_entry_angle_deg=22.0),
        2.0, Environment(),
    )
    assert reference.components_n["crushing"] == pytest.approx(direct["crushing_N"])
    assert reference.components_n["bending"] == pytest.approx(direct["bending_N"])
    assert reference.components_n["submersion"] == pytest.approx(direct["submersion_N"])
    assert reference.total_n == pytest.approx(sum(reference.components_n.values()))
    assert sum(reference.shares.values()) == pytest.approx(1.0)
    assert reference.components_per_width_n_m["bending"] == pytest.approx(
        reference.components_n["bending"] / 20.0)


def test_comparison_reports_component_and_share_errors():
    reference = LindqvistReference(
        beam_m=2.0, length_m=10.0,
        components_n={"crushing": 200.0, "bending": 50.0, "submersion": 50.0},
        speed_dependent_n=0.0,
    )
    exact = compare_components(
        {"crushing": 100.0, "bending": 25.0, "submersion": 25.0}, 150.0, reference
    )
    assert exact["total_error_pct"] == pytest.approx(0.0, abs=1e-12)
    assert exact["total_passed"] is True
    assert exact["shares_passed"] is True
    off = compare_components(
        {"crushing": 50.0, "bending": 25.0, "submersion": 25.0}, 100.0, reference
    )
    assert off["total_signed_error_pct"] < -15.0  # DEM is below Lindqvist
    assert off["total_error_pct"] > 15.0          # reported magnitude is unsigned
    assert off["total_passed"] is False


# ---------------------------------------------------------------------------
# solver wiring (tiny DEM cases)
# ---------------------------------------------------------------------------
def test_bond_graph_detachment_marks_the_cut_off_half():
    case = _tiny_case(n_length=6, n_width=2, thickness_m=0.04, particle_radius_m=0.02)
    sim = build_simulation(case)
    pairs = sim.pairs
    positions = sim.positions
    side_a = positions[pairs[:, 0], 0] < 0.0
    side_b = positions[pairs[:, 1], 0] < 0.0
    sim.alive[side_a != side_b] = False  # cut the lattice on the x = 0 plane
    labels = sim.cluster_labels()
    detached = sim.detached_mask(labels)
    assert int(detached.sum()) > 0
    assert int(detached.sum()) < len(positions)
    assert len(torch.unique(labels)) >= 2


def test_contact_records_are_consistent_and_capped():
    case = _tiny_case()
    sim = build_simulation(case)
    sim._wedge_contact()
    contacts = sim.contact_set()
    assert contacts.count > 0
    assert contacts.count == len(contacts.positions)
    assert bool((contacts.normal_force > 0).all())
    assert bool((contacts.cap_force > 0).all())
    assert bool((contacts.normal_force <= contacts.cap_force * (1 + 1e-9)).all())
    norms = torch.linalg.vector_norm(contacts.normals, dim=1)
    assert float((norms - 1.0).abs().max()) < 1e-12
    reaction = -contacts.force_on_particle.sum(dim=0)
    decomposition = decompose(contacts)
    assert decomposition.reaction_vector[0] == pytest.approx(float(reaction[0]))
    assert abs(decomposition.residual) <= 1e-12 * max(1.0, abs(decomposition.r_total))


def test_run_case_history_is_closed_and_finite():
    case = _tiny_case()
    result = run_case(case)
    assert result["series"]
    for row in result["series"]:
        assert all(math.isfinite(value) for value in row.values())
        assert abs(row["residual_N"]) <= 1e-12 * max(1.0, abs(row["r_total_N"]))
    assert result["max_relative_residual"] <= 1e-12
    assert set(result["steady"]["components"]) == set(MECHANISMS)
    assert result["case"]["particle_count"] > 0


def test_zero_load_case_has_no_resistance():
    case = _tiny_case(steps=2, embed_m=0.0, initial_gap_m=0.20, bow_speed_m_s=0.01)
    result = run_case(case)
    assert all(value == 0.0 for value in result["steady"]["components"].values())
    assert result["steady"]["r_total"] == 0.0
    assert result["case"]["first_contact_step"] is None


def test_capped_bow_dem_rejects_invalid_parameters():
    config = _tiny_case().to_config()
    with pytest.raises(ValueError):
        CappedBowDEM(config, crushing_pressure_pa=0.0)
    with pytest.raises(ValueError):
        CappedBowDEM(config, crushing_pressure_pa=1e6, friction_coefficient=-0.1)
    with pytest.raises(ValueError):
        ClosureCase(mass_scale=-1.0)
    with pytest.raises(ValueError):
        ClosureCase(crushing_efficiency=1.5)

# ---------------------------------------------------------------------------
# verdict logic of the campaign (kept separate from the physics)
# ---------------------------------------------------------------------------
def _summary(total_per_width: float, shares: dict[str, float], error_pct: float,
             residual: float = 1e-16) -> dict:
    return {
        "steady": {"components": {name: total_per_width * value for name, value in shares.items()},
                   "shares": dict(shares), "r_total": total_per_width},
        "steady_per_width": {"r_total_N_m": total_per_width},
        "max_relative_residual": residual,
        "lindqvist": {
            "total_error_pct": abs(error_pct),
            "total_signed_error_pct": error_pct,
            "dem_total_N_m": total_per_width,
            "lindqvist_total_N_m": total_per_width / (1.0 + error_pct / 100.0),
        },
    }


def test_criteria_pass_on_consistent_synthetic_cases():
    from tensordem.resistance_closure.campaign import _evaluate_criteria

    same = {"crushing": 0.6, "bending": 0.2, "submersion": 0.2}
    thickness = {
        "0.20": _summary(100.0, same, 5.0),
        "0.40": _summary(101.0, same, 5.0),
    }
    criteria = _evaluate_criteria(
        base_result=thickness["0.20"],
        thickness_cases=thickness,
        refined_result=_summary(102.0, same, 5.0),
        control={"zero": True, "series_max_abs_total": 0.0, "max_relative_residual": 0.0},
        mass_results={"100": _summary(100.0, same, 5.0), "400": _summary(103.0, same, 5.0)},
    )
    assert criteria["a_closure_residual"]["passed"] is True
    assert criteria["b_lindqvist_total"]["passed"] is True
    assert criteria["c_crushing_trend"]["passed"] is False   # equal shares do not increase
    assert criteria["d_grid_refinement"]["passed"] is True    # 1 % change
    assert criteria["e_negative_control"]["passed"] is True


def test_criteria_detect_each_failure_mode():
    from tensordem.resistance_closure.campaign import _evaluate_criteria

    growing = {"crushing": 0.5, "bending": 0.3, "submersion": 0.2}
    low = {"crushing": 0.4, "bending": 0.4, "submersion": 0.2}
    thickness = {
        "0.20": _summary(100.0, low, 40.0, residual=5e-3),
        "0.40": _summary(300.0, growing, 40.0, residual=5e-3),
    }
    criteria = _evaluate_criteria(
        base_result=thickness["0.20"],
        thickness_cases=thickness,
        refined_result=_summary(150.0, growing, 40.0, residual=5e-3),
        control={"zero": False, "series_max_abs_total": 1.0, "max_relative_residual": 0.0},
        mass_results={"100": _summary(100.0, growing, 40.0),
                      "400": _summary(300.0, growing, 40.0)},
    )
    assert criteria["a_closure_residual"]["passed"] is False   # 5e-3 > 1e-3
    assert criteria["b_lindqvist_total"]["passed"] is False    # 40 % > 15 %
    assert criteria["c_crushing_trend"]["passed"] is True      # 0.4 -> 0.5
    assert criteria["d_grid_refinement"]["passed"] is False    # 50 % change
    assert criteria["e_negative_control"]["passed"] is False
    assert criteria["mass_scale_axis"]["passed"] is False      # 100 % spread
