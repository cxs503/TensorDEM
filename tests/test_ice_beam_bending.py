"""Tests for :mod:`tensordem.ice_beam_bending` (Timoshenko anchor for DEM3D)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem import ice_beam_bending as ib
from tensordem.lattice_elasticity3d import periodic_elastic_constants

COARSE = ib.IceBeamSpec().with_particle_diameter(0.05)   # h/a = 2, 315 particles
MEDIUM = ib.IceBeamSpec().with_particle_diameter(0.025)  # h/a = 4, 1845 particles


# --------------------------------------------------------------------------
# Specification and lattice constants
# --------------------------------------------------------------------------
def test_spec_counts_match_geometry():
    spec = ib.IceBeamSpec()
    assert spec.counts() == (201, 41, 21)
    assert spec.as_dict()["depth_particles"] == 20
    assert spec.lattice_spacing_m == pytest.approx(0.005)


@pytest.mark.parametrize("bad", [
    dict(length_m=1.003),
    dict(thickness_m=0.0),
    dict(particle_diameter_m=0.0),
    dict(bond_strength_Pa=-1.0),
])
def test_spec_validation_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        ib.IceBeamSpec(**bad)


def test_lattice_modulus_inversion_is_exact():
    spec = ib.IceBeamSpec()
    props = ib.lattice_elastic_properties(spec)
    assert props["young_modulus_Pa"] == pytest.approx(9.0e9, rel=1e-12)
    assert props["poisson_ratio"] == pytest.approx(0.35, rel=1e-12)
    assert props["shear_modulus_Pa"] == pytest.approx(props["C44_Pa"])
    assert props["plate_modulus_Pa"] == pytest.approx(
        props["young_modulus_Pa"] / (1.0 - props["poisson_ratio"] ** 2))
    assert props["breaking_strain"] == pytest.approx(
        spec.bond_strength_Pa / props["young_modulus_Pa"], rel=1e-12)
    assert props["shear_breaking_strain"] == pytest.approx(
        props["breaking_strain"] * spec.shear_strength_multiplier)
    # lattice constants must be a factor of the bond stiffness only
    unit = periodic_elastic_constants(1.0, props["lattice_spacing_m"])
    ratio = props["bond_stiffness_N_m"]
    for name in ("C11_Pa", "C12_Pa", "C44_Pa"):
        assert props[name] == pytest.approx(unit[name] * ratio, rel=1e-12)


def test_bond_strength_scale_only_changes_strength():
    base = ib.IceBeamSpec()
    scaled = base.with_bond_strength_scale(1.2)
    p0 = ib.lattice_elastic_properties(base)
    p1 = ib.lattice_elastic_properties(scaled)
    assert p1["bond_stiffness_N_m"] == pytest.approx(p0["bond_stiffness_N_m"])
    assert p1["breaking_strain"] == pytest.approx(1.2 * p0["breaking_strain"])
    assert scaled.bond_strength_Pa == pytest.approx(1.2 * base.bond_strength_Pa)


# --------------------------------------------------------------------------
# Closed-form beam theory
# --------------------------------------------------------------------------
def test_beam_theory_closed_forms():
    spec = ib.IceBeamSpec()
    theory = ib.beam_theory(spec)
    E, b, h, L = 9.0e9, 0.2, 0.1, 1.0
    inertia = b * h ** 3 / 12.0
    assert theory.second_moment_m4 == pytest.approx(inertia)
    assert theory.flexural_rigidity_Nm2 == pytest.approx(E * inertia)

    load = 1000.0
    assert theory.deflection_midspan(load, "euler_bernoulli") == pytest.approx(
        load * L ** 3 / (48 * E * inertia), rel=1e-12)
    assert theory.deflection_midspan(load, "timoshenko") == pytest.approx(
        load * L ** 3 / (48 * E * inertia) + load * L / (4 * theory.shear_rigidity_N),
        rel=1e-12)
    assert theory.sigma_max(load) == pytest.approx(3 * load * L / (2 * b * h ** 2))
    # the brief's PL/(4bh^2) omits the 1/6 section-modulus factor
    assert theory.sigma_max_brief_formula(load) == pytest.approx(
        theory.sigma_max(load) / 6.0)
    assert theory.failure_load_brief_formula() == pytest.approx(
        6.0 * theory.failure_load())
    assert theory.failure_load() == pytest.approx(
        2 * spec.bond_strength_Pa * b * h ** 2 / (3 * L))
    # symmetry and end conditions of the written formula
    x = np.array([0.0, 0.13, 0.25, 0.37, 0.5])
    w = theory.deflection(x, load, "timoshenko")
    assert w[0] == pytest.approx(0.0, abs=1e-18)
    assert theory.deflection(np.array([L]), load)[0] == pytest.approx(0.0, abs=1e-14)
    assert theory.deflection(L - x, load) == pytest.approx(w)
    # monotone increase towards midspan
    assert np.all(np.diff(w) > 0)
    # the shear term is a small but non-zero fraction at L/h = 10
    frac = theory.as_dict()["shear_fraction_of_midspan_deflection"]
    assert 0.005 < frac < 0.05
    # stress profile: linear, zero at the neutral axis, extreme fibre = sigma_max
    z = np.array([0.0, theory.neutral_axis_z_m, h])
    sigma = theory.sigma_at_z(z, load)
    assert sigma[1] == pytest.approx(0.0, abs=1e-9)
    assert sigma[0] == pytest.approx(theory.sigma_max(load), rel=1e-12)
    assert sigma[2] == pytest.approx(-theory.sigma_max(load), rel=1e-12)


def test_plane_strain_theory_uses_plate_modulus():
    spec = ib.IceBeamSpec()
    props = ib.lattice_elastic_properties(spec)
    ps = ib.beam_theory(spec, props)
    pn = ib.beam_theory(spec, props, plane_strain=True)
    assert pn.young_modulus_Pa == pytest.approx(props["plate_modulus_Pa"])
    assert pn.strain_state == "plane_strain_cylindrical"
    assert pn.deflection_midspan(100.0) < ps.deflection_midspan(100.0)


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------
def test_beam_model_boundary_conditions():
    model = ib.beam_model(COARSE)
    spec = model.spec
    assert model.span_m == pytest.approx(spec.length_m)
    assert model.particle_count == np.prod(spec.counts())
    # supports: the two end columns at the neutral layer, full width
    idx = model.node_indices
    assert np.all(idx[model.support_nodes, 0] % (spec.counts()[0] - 1) == 0)
    assert np.all(idx[model.support_nodes, 2] == model.neutral_index)
    assert model.z_m[model.neutral_index] == pytest.approx(spec.thickness_m / 2.0)
    # load: the central top column across the width
    assert np.all(idx[model.load_nodes, 0] == spec.counts()[0] // 2)
    assert np.all(idx[model.load_nodes, 2] == spec.counts()[2] - 1)
    assert len(model.load_nodes) == spec.counts()[1]


# --------------------------------------------------------------------------
# Rate-free equilibrium
# --------------------------------------------------------------------------
def test_static_solution_is_an_equilibrium_of_the_solver():
    model = ib.beam_model(COARSE)
    load = 50.0
    solution = ib.static_solution(model, load)
    assert solution.solver == "cg"
    assert solution.pcg_relative_residual < 1e-9
    # the solver's own force routine leaves the free DOFs balanced and the
    # supports carrying exactly the applied load
    # contact springs between non-bonded neighbours leave an O(1e-5 P) residual,
    # so the bound is relative to the applied load
    assert solution.residual_free_dof_N / load < 1e-4
    assert solution.reactions_N[2] == pytest.approx(load, rel=1e-5)
    assert abs(solution.reactions_N[0]) < 1e-3
    assert abs(solution.reactions_N[1]) < 1e-3


def test_deflection_is_linear_in_load_and_symmetric():
    model = ib.beam_model(COARSE)
    theory = ib.beam_theory(model.spec, model.properties)
    w1 = ib.static_solution(model, 20.0, verify=False)
    w2 = ib.static_solution(model, 40.0, verify=False)
    profile1 = ib.deflection_profile(model, w1.displacement)["w_neutral_m"]
    profile2 = ib.deflection_profile(model, w2.displacement)["w_neutral_m"]
    assert np.allclose(profile2, 2.0 * profile1, rtol=1e-9, atol=1e-18)
    # symmetric about midspan, zero at the supports, downward
    assert profile1[0] == pytest.approx(0.0, abs=1e-15)
    assert profile1[-1] == pytest.approx(0.0, abs=1e-15)
    assert np.all(profile1 >= -1e-18)
    assert profile1 == pytest.approx(profile1[::-1], rel=1e-9, abs=1e-15)
    assert theory.deflection_midspan(20.0) > 0.0


def test_section_moment_matches_statics():
    """The bond forces crossing midspan must carry M = P L / 4."""
    model = ib.beam_model(MEDIUM)
    theory = ib.beam_theory(model.spec, model.properties)
    load = 40.0
    solution = ib.static_solution(model, load, verify=False)
    quarter = ib.section_moment(model, solution.displacement,
                                x_index=int(round(0.25 * (len(model.x_m) - 1))))
    expected = float(theory.moment(np.array([0.25 * model.span_m]), load)[0])
    assert quarter == pytest.approx(expected, rel=0.15)
    three_quarter = ib.section_moment(model, solution.displacement,
                                      x_index=int(round(0.75 * (len(model.x_m) - 1))))
    assert three_quarter == pytest.approx(expected, rel=0.15)


def test_first_bond_event_is_at_the_midspan_bottom_fibre():
    model = ib.beam_model(MEDIUM)
    theory = ib.beam_theory(model.spec, model.properties)
    anchor = ib.elastic_anchor(model, theory, probe_fraction=0.1, verify=False)
    first = anchor["first_failure"]
    assert first["first_break_is_tension"]
    assert first["first_break_z_over_h"] <= 0.15
    assert abs(first["first_break_x_over_L"] - 0.5) <= 0.10
    # the coarse test lattice is deliberately badly resolved (h/d = 4); the
    # spec-resolution (h/d = 20) value is 15.7 %
    assert first["rel_error_first_break_load"] < 0.75


def test_elastic_anchor_is_consistent():
    model = ib.beam_model(MEDIUM)
    theory = ib.beam_theory(model.spec, model.properties)
    anchor = ib.elastic_anchor(model, theory, verify=True)
    deflection = anchor["deflection"]
    assert deflection["comparison_point_count"] > 10
    # the lattice is stiffer than plane-stress beam theory but the shape agrees
    ratio = (anchor["deflection"]["midspan_timoshenko_m"]
             / anchor["deflection"]["midspan_dem_m"])
    assert 0.6 < ratio < 1.6
    # the anchor samples the midspan section, which sits on the load line; the
    # quarter-point section is checked in test_section_moment_matches_statics
    assert anchor["stress"]["rel_error_section_moment"] < 0.80
    assert abs(anchor["stress"]["sigma_max_dem_bottom_layer_Pa"]) > 0.0
    assert anchor["reaction_balance_error"] < 1e-4


def test_lattice_bending_bias_decreases_with_refinement():
    """Documented discretisation bias: the finer the lattice, the closer to EI."""
    ratios = []
    for spec in (COARSE, MEDIUM):
        model = ib.beam_model(spec)
        theory = ib.beam_theory(spec, model.properties)
        anchor = ib.elastic_anchor(model, theory, verify=False)
        ratios.append(anchor["deflection"]["midspan_timoshenko_m"]
                      / anchor["deflection"]["midspan_dem_m"])
    assert ratios[0] > ratios[1] > 1.0


# --------------------------------------------------------------------------
# Brittle damage cascade and crack path
# --------------------------------------------------------------------------
def test_fracture_cascade_initiates_at_bottom_midspan_and_grows_upwards():
    model = ib.beam_model(MEDIUM)
    theory = ib.beam_theory(model.spec, model.properties)
    cascade = ib.fracture_cascade(model, theory, max_events=4, events_per_step=1)
    assert cascade["event_count"] >= 2
    assert cascade["peak_load_N"] == pytest.approx(cascade["start_load_N"])
    report = ib.crack_report(cascade, model.spec)
    assert report["available"]
    initiation = report["initiation"]
    assert initiation["z_over_h"] <= 0.15
    assert abs(initiation["x_over_L"] - 0.5) <= 0.10
    # the compliance grows as bonds are removed
    assert cascade["compliance_growth"] >= 1.0
    assert cascade["history"][-1]["broken_bonds"] == cascade["broken_bonds"]


def test_crack_report_handles_empty_cascade():
    assert ib.crack_report({"events": []}, ib.IceBeamSpec()) == {"available": False}


# --------------------------------------------------------------------------
# Sensitivity and negative control
# --------------------------------------------------------------------------
def test_strength_sensitivity_is_linear():
    result = ib.stress_sensitivity(COARSE, scales=(0.8, 1.0, 1.2))
    rows = {round(row["bond_strength_scale"], 3): row for row in result["rows"]}
    base = rows[1.0]["first_break_load_N"]
    assert rows[0.8]["first_break_load_N"] == pytest.approx(0.8 * base, rel=1e-9)
    assert rows[1.2]["first_break_load_N"] == pytest.approx(1.2 * base, rel=1e-9)
    assert rows[1.2]["load_change_relative_to_base"] == pytest.approx(0.2, rel=1e-9)
    assert rows[0.8]["load_change_relative_to_base"] == pytest.approx(-0.2, rel=1e-9)


def test_unbonded_negative_control_collapses():
    model = ib.beam_model(COARSE)
    result = ib.unbonded_collapse(model, 5.0, steps=40, sample_every=10)
    assert result["unbonded_stiffness_nnz"] == 0
    assert result["unbonded_collapses"]
    assert result["unbonded"]["samples"][-1]["midspan_deflection_m"] > 0.0
    # with no bonds the load particles are in free fall: the paired bonded run
    # stays at rest in the same window
    assert result["unbonded"]["final_max_particle_speed_m_per_s"] > 0.0
    assert (result["unbonded"]["final_max_particle_speed_m_per_s"]
            > result["bonded"]["final_max_particle_speed_m_per_s"])


def test_explicit_drift_from_static_solution_is_small():
    model = ib.beam_model(COARSE)
    load = 5.0
    solution = ib.static_solution(model, load, verify=False)
    drift = ib.explicit_equilibrium_drift(model, load,
                                          displacement=solution.displacement,
                                          steps=30, sample_every=10)
    assert drift["relative_drift"] < 0.2
    assert np.isfinite(drift["peak_particle_speed_m_per_s"])


# --------------------------------------------------------------------------
# Acceptance plumbing
# --------------------------------------------------------------------------
def _toy_report(**overrides):
    base = {
        "anchor": {
            "probe_load_N": 200.0,
            "deflection": {"rel_error_midspan_vs_timoshenko": 0.01,
                           "rel_error_profile_max_vs_timoshenko": 0.02,
                           "rel_error_midspan_vs_euler_bernoulli": 0.02},
            "stress": {"rel_error_sigma_max_fit": 0.03,
                       "rel_error_sigma_max_bottom_layer": 0.1,
                       "rel_error_sigma_max_section_moment": 0.04,
                       "rel_error_profile_max_interior": 0.05,
                       "rel_error_section_moment": 0.04,
                       "rel_error_fit_slope": 0.05},
            "first_failure": {"rel_error_first_break_load": 0.05},
            "solution": {"residual_free_dof_N": 1e-9, "relative_force_balance": 1e-9},
        },
        "crack": {"initiation": {"at_bottom_fibre": True, "at_midspan": True,
                                 "x_over_L": 0.5, "z_over_h": 0.0},
                  "propagation": {"breaks_through_half_depth": True,
                                  "propagation_direction_upward": True,
                                  "crack_front_z_over_h": 0.9, "event_count": 12}},
        "sensitivity": {"rows": [{"bond_strength_scale": 0.8,
                                  "load_change_relative_to_base": -0.2},
                                 {"bond_strength_scale": 1.2,
                                  "load_change_relative_to_base": 0.2}]},
        "negative_control": {"unbonded_collapses": True, "bonded_holds": True,
                             "unbonded_stiffness_nnz": 0,
                             "unbonded_to_bonded_deflection_ratio": 100.0,
                             "unbonded_support_reaction_over_load": 0.0,
                             "bonded_support_reaction_over_load": 1.0,
                             "bonded": {"samples": [{"midspan_deflection_m": 1e-5}],
                                        "reference_midspan_deflection_m": 1e-5}},
        "equilibrium_drift": {"relative_drift": 0.01},
        "resolution_convergence": {"deflection_rel_change": 0.02},
    }
    base.update(overrides)
    return base


def test_evaluate_criteria_passes_a_clean_report():
    result = ib.evaluate_criteria(_toy_report())
    assert result["all_pass"], result["failed"]
    assert result["total"] == result["passed"]


def test_evaluate_criteria_reports_failures():
    report = _toy_report()
    report["anchor"]["deflection"]["rel_error_midspan_vs_timoshenko"] = 0.20
    report["crack"]["initiation"]["at_midspan"] = False
    result = ib.evaluate_criteria(report)
    assert not result["all_pass"]
    assert "deflection_midspan_vs_timoshenko" in result["failed"]
    assert "first_crack_at_midspan_bottom" in result["failed"]
    assert result["checks"]["deflection_midspan_vs_timoshenko"]["value"] == 0.20


def test_evaluate_criteria_sensitivity_band():
    report = _toy_report()
    report["sensitivity"]["rows"][1]["load_change_relative_to_base"] = 0.30
    result = ib.evaluate_criteria(report)
    assert "bond_strength_sensitivity_18_22pct" in result["failed"]