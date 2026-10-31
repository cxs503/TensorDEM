"""Tests for the W5-N03 ice-ridge keel grounding models.

Covers the Kerr elastic-foundation beam (analytic + mixed FEM), the discrete
bonded-particle keel (calibration, quasi-static indentation, dynamic kernel),
local crushing membrane solutions, the grounding criterion and the W5-N02 /
W3-N05 interfaces.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem import kerr_foundation as kf
from tensordem import keel_lattice as kl
from tensordem import local_crushing as lc
from tensordem import ridge_grounding as rg

LENGTH = 12.0
THICKNESS = 1.5
FOUNDATION_MODULUS = 2.0e7
E2D = 5.0e9


def foundation():
    return kf.KerrFoundation(upper=2 * FOUNDATION_MODULUS, lower=2 * FOUNDATION_MODULUS,
                             shear=FOUNDATION_MODULUS * (THICKNESS / 2) ** 2)


def beam_ei():
    return E2D * 1.0 * THICKNESS**3 / 12.0


# ---------------------------------------------------------------------------
# Kerr foundation / beam
# ---------------------------------------------------------------------------
def test_kerr_foundation_series_stiffness_and_validation():
    f = foundation()
    assert f.equivalent_winkler == pytest.approx(FOUNDATION_MODULUS)
    with pytest.raises(ValueError):
        kf.KerrFoundation(upper=0.0, lower=1.0e7)
    with pytest.raises(ValueError):
        kf.KerrFoundation(upper=1.0e7, lower=1.0e7, shear=-1.0)


def test_characteristic_length_matches_winkler_formula():
    f = foundation()
    ei = beam_ei()
    expected = (4.0 * ei / FOUNDATION_MODULUS) ** 0.25
    assert kf.characteristic_length(ei, f) == pytest.approx(expected)
    with pytest.raises(ValueError):
        kf.characteristic_length(-1.0, f)


def test_beam_section_properties():
    section = kf.BeamSection(young_modulus=E2D, thickness=THICKNESS)
    assert section.inertia == pytest.approx(THICKNESS**3 / 12.0)
    assert section.ei == pytest.approx(beam_ei())
    assert section.area == pytest.approx(THICKNESS)


def test_kerr_characteristic_roots_satisfy_polynomial():
    f = foundation()
    ei = beam_ei()
    roots = kf.kerr_characteristic_roots(ei, f)
    assert roots.size == 6
    alpha = 1.0 + f.upper / f.lower
    beta = f.shear / f.lower
    residual = (beta * ei * roots**6 - alpha * ei * roots**4
                + f.upper * beta * roots**2 - f.upper)
    assert np.max(np.abs(residual)) < 1e-6 * max(1.0, float(np.max(np.abs(f.upper))))


def test_winkler_limit_has_four_roots():
    f = kf.KerrFoundation(upper=4e7, lower=4e7, shear=0.0)
    roots = kf.kerr_characteristic_roots(beam_ei(), f)
    assert roots.size == 4


def test_analytic_vs_fem_below_three_percent():
    """Acceptance criterion (a): deflection error < 3%."""
    result = kf.compare_analytic_fem(LENGTH, beam_ei(), foundation(), 3.0e6, elements=240)
    assert result["deflection_profile_rel_error"] < 0.03
    assert result["peak_deflection_rel_error"] < 0.03
    assert result["pressure_peak_rel_error"] < 0.03


def test_analytic_vs_fem_winkler_limit():
    f = kf.KerrFoundation(upper=4e7, lower=4e7, shear=0.0)
    result = kf.compare_analytic_fem(LENGTH, beam_ei(), f, 3.0e6, elements=192)
    assert result["deflection_profile_rel_error"] < 0.03


def test_analytic_global_equilibrium():
    solution = kf.kerr_beam_analytic(LENGTH, beam_ei(), foundation(), 2.0e6, n_points=1201)
    assert solution.total_foundation_force == pytest.approx(2.0e6, rel=1e-4)
    assert solution.max_deflection > 0.0


def test_analytic_satisfies_governing_relations():
    """M-V and V-p relations of the beam + Kerr field, checked on exact arrays.

    The solution carries the analytic bending moment, shear and contact pressure,
    so one finite difference of exact quantities is enough: ``dM/dx = V`` and
    ``dV/dx = p`` must hold in the interior, which together with the boundary
    conditions verifies the whole solve.
    """
    solution = kf.kerr_beam_analytic(LENGTH, beam_ei(), foundation(), 2.0e6, n_points=2001)
    x = solution.x
    h = float(x[1] - x[0])
    interior = slice(20, 1980)
    moment, shear, pressure = solution.moment, solution.shear, solution.pressure
    d_moment = np.gradient(moment, h)
    scale = float(np.max(np.abs(shear[interior])))
    assert float(np.max(np.abs(d_moment[interior] - shear[interior]))) < 1e-3 * scale
    d_shear = np.gradient(shear, h)
    scale_p = float(np.max(np.abs(pressure[interior])))
    assert float(np.max(np.abs(d_shear[interior] - pressure[interior]))) < 1e-2 * scale_p
    # Kerr constitutive split: p = k_u (w - w_s)
    rebuilt = foundation().upper * (solution.deflection - solution.shear_layer)
    assert np.max(np.abs(rebuilt - pressure)) < 1e-6 * scale_p


def test_mesh_convergence_is_monotone():
    rows = kf.convergence_study(LENGTH, beam_ei(), foundation(), 2e6,
                                element_counts=(24, 48, 96, 192))
    errors = [row["deflection_profile_rel_error"] for row in rows]
    assert errors[0] > errors[-1]
    assert errors[-1] < 1e-4
    for earlier, later in zip(errors, errors[1:]):
        assert later < earlier


def test_response_is_linear_in_load():
    response = kf.load_deflection_response(LENGTH, beam_ei(), foundation(), [1e6, 2e6, 4e6],
                                           elements=96)
    ratios = [w / p for w, p in zip(response["peak_deflection_m"], response["load_N_per_m"])]
    assert max(ratios) - min(ratios) < 1e-6 * max(ratios)
    forces = response["foundation_force_N_per_m"]
    for force, load in zip(forces, response["load_N_per_m"]):
        assert force == pytest.approx(load, rel=2e-3)


def test_moving_load_scan_symmetry_and_end_effect():
    """A load near a free end deflects more than at mid-span; scan is symmetric."""
    scan = kf.moving_load_scan(LENGTH, beam_ei(), foundation(), 2e6,
                               [1.0, 3.0, 5.0, 6.0], elements=96)
    assert scan["worst_position_m"] == pytest.approx(1.0)
    envelope = np.asarray(scan["deflection_envelope_m"])
    assert envelope.max() == pytest.approx(scan["worst_deflection_m"], rel=1e-9)
    x = np.asarray(scan["x_m"])
    mid = float(np.interp(6.0, x, envelope))
    assert scan["worst_deflection_m"] > mid
    left = kf.kerr_beam_fem(LENGTH, beam_ei(), foundation(), 2e6, load_position=4.0, elements=96)
    right = kf.kerr_beam_fem(LENGTH, beam_ei(), foundation(), 2e6, load_position=8.0, elements=96)
    assert left.max_deflection == pytest.approx(right.max_deflection, rel=1e-9)


def test_shear_layer_reduces_deflection_vs_winkler():
    """The Kerr shear layer adds foundation stiffness for a bottom-mounted bed."""
    ei = beam_ei()
    kerr = kf.kerr_beam_analytic(LENGTH, ei, foundation(), 2e6)
    winkler = kf.kerr_beam_analytic(LENGTH, ei,
                                    kf.KerrFoundation(4e7, 4e7, 0.0), 2e6)
    assert kerr.max_deflection < winkler.max_deflection


# ---------------------------------------------------------------------------
# local crushing: rigid-plastic membrane
# ---------------------------------------------------------------------------
def test_rigid_plastic_membrane_scaling():
    panel = lc.PlatePanel()
    assert lc.rigid_plastic_membrane_pressure(0.05, panel) == pytest.approx(
        panel.membrane_rate * 0.05)
    deflection = lc.rigid_plastic_membrane_deflection(2e6, panel)
    assert lc.rigid_plastic_membrane_pressure(deflection, panel) == pytest.approx(2e6)
    with pytest.raises(ValueError):
        lc.rigid_plastic_membrane_pressure(-1.0, panel)


def test_variational_form_is_20_percent_softer_than_factor_eight():
    panel = lc.PlatePanel()
    for w in (0.02, 0.05, 0.1):
        assert lc.variational_membrane_pressure(w, panel) == pytest.approx(
            0.8 * lc.rigid_plastic_membrane_pressure(w, panel))


def test_membrane_strain_formula():
    panel = lc.PlatePanel()
    expected = 4.0 * 0.05**2 * (1 / panel.span_x**2 + 1 / panel.span_y**2)
    assert lc.membrane_strain(0.05, panel) == pytest.approx(expected)


def test_membrane_numerical_between_closed_forms():
    """Numerical membrane solution lies between the two rigid-plastic forms."""
    panel = lc.PlatePanel()
    pressure = 5.0e6
    result = lc.membrane_panel_energy(panel, pressure, grid=(9, 11), max_iterations=1200)
    closed = lc.rigid_plastic_membrane_deflection(pressure, panel)
    variational = pressure / lc.variational_membrane_pressure(1.0, panel)
    assert closed < result["central_deflection_m"] < variational * 1.02
    assert result["relative_error"] < 0.20


def test_crushing_damage_contract():
    panel = lc.PlatePanel()
    damage = lc.crushing_damage(5.0e6, panel)
    for key in ("deflection_m", "membrane_strain", "rupture", "absorbed_energy_J",
                "equivalent_force_N", "consumer"):
        assert key in damage
    assert damage["consumer"] == "W3-N05 collision damage"
    assert damage["deflection_m"] == pytest.approx(
        lc.rigid_plastic_membrane_deflection(5.0e6, panel))
    assert damage["rupture"] is False
    with pytest.raises(ValueError):
        lc.crushing_damage(-1.0, panel)


# ---------------------------------------------------------------------------
# discrete bonded-particle keel
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def calibration():
    return kl.calibrate_bond_stiffness(10.0, 1.0, 0.025, E2D)


def test_triangular_lattice_geometry():
    pos, s = kl.triangular_lattice(3.0, 0.6, 0.05)
    assert s == pytest.approx(0.1)
    assert pos.shape[1] == 2
    assert pos[:, 0].max() <= 3.0 + 1e-9
    realised = float(pos[:, 1].max())
    assert realised <= 0.6 + s * math.sqrt(3.0) / 2.0 + 1e-9   # one row rounding
    pairs = kl.neighbour_pairs(pos, s)
    assert pairs.shape[0] > 0
    assert int(pairs.max()) < pos.shape[0]


def test_uniaxial_modulus_is_linear_in_bond_stiffness(calibration):
    pos, s = kl.triangular_lattice(3.0, 1.0, 0.025)
    pairs = kl.neighbour_pairs(pos, s)
    k = 1.0e9
    first = kl.uniaxial_strain_test(pos, pairs, k)
    second = kl.uniaxial_strain_test(pos, pairs, 2 * k)
    assert second["E_2D_N_per_m"] == pytest.approx(2 * first["E_2D_N_per_m"], rel=1e-9)
    assert first["E_2D_N_per_m"] > 0.0


def test_calibration_reaches_target_modulus(calibration):
    assert calibration["relative_error"] < 1e-6
    assert calibration["realised_E_2D_N_per_m"] == pytest.approx(E2D, rel=1e-6)


def test_cantilever_bending_matches_section_formula(calibration):
    cant = calibration["cantilever"]
    expected = E2D * cant["thickness_m"]**3 / 12.0
    assert cant["EI_Nm2"] == pytest.approx(expected, rel=0.08)


def test_quasi_static_indentation_is_linear_and_monotone(calibration):
    result = kl.quasi_static_indentation(
        10.0, 1.0, 0.025, calibration["bond_stiffness_N_per_m"], FOUNDATION_MODULUS,
        punch_width=0.4, depths=[0.004, 0.008, 0.012])
    steps = result["steps"]
    compliances = [s["beam_deflection_m"] / s["reaction_N_per_m"] for s in steps]
    assert max(compliances) - min(compliances) < 1e-6 * max(compliances)
    forces = [s["reaction_N_per_m"] for s in steps]
    assert forces[0] < forces[1] < forces[2]
    rho = rg.spearman_monotonic([s["punch_depth_m"] for s in steps], forces)["spearman_rho"]
    assert rho == pytest.approx(1.0)


def test_quasi_static_indentation_rejects_bad_input():
    with pytest.raises(ValueError):
        kl.quasi_static_indentation(6.0, 0.8, 0.04, 1e9, FOUNDATION_MODULUS, depths=[])
    with pytest.raises(ValueError):
        kl.quasi_static_indentation(6.0, 0.8, 0.04, -1.0, FOUNDATION_MODULUS, depths=[0.01])


def test_dynamic_dem_punch_reaction_and_constraint():
    config = kl.KeelLatticeConfig(length=2.0, thickness=0.4, radius=0.05,
                                  bond_stiffness=1.0e9, foundation_modulus=2.0e7,
                                  punch_width=0.3, mass_scale=1.0, breaking_strain=1.0)
    dem = kl.KeelLatticeDEM(config)
    assert dem.dt > 0.0
    assert dem.initial_face_y > float(dem.pos[:, 1].max())
    info = dem.advance_punch(0.01, relax_steps=60, energy_tolerance=1e-2)
    assert info["penetration_m"] == pytest.approx(0.01)
    assert dem.reaction > 0.0
    penetrators = dem.punch_mask()
    assert bool(penetrators.any())
    profile = dem.pressure_profile(bins=11)
    assert len(profile["x_m"]) == 11


def test_dynamic_dem_kinetic_energy_decreases():
    config = kl.KeelLatticeConfig(length=2.0, thickness=0.4, radius=0.05,
                                  bond_stiffness=1.0e9, foundation_modulus=2.0e7,
                                  punch_width=0.3, breaking_strain=1.0)
    dem = kl.KeelLatticeDEM(config)
    dem.set_punch_face(dem.initial_face_y - 0.01)
    dem.step(5)
    initial = dem.kinetic_energy()
    dem.relax(max_steps=400, energy_tolerance=1e-6, min_steps=20)
    assert dem.kinetic_energy() < max(initial, 1e-12)


def test_fragment_size_distribution_of_intact_lattice():
    config = kl.KeelLatticeConfig(length=1.0, thickness=0.3, radius=0.05,
                                  bond_stiffness=1.0e9, foundation_modulus=0.0,
                                  punch_width=0.2, breaking_strain=1.0)
    dem = kl.KeelLatticeDEM(config)
    fragments = dem.fragment_size_distribution()
    assert fragments["clusters"] == 1
    assert fragments["largest_cluster_particles"] == dem.particles


# ---------------------------------------------------------------------------
# scenario, grounding criterion and interfaces
# ---------------------------------------------------------------------------
def test_kerr_foundation_mapping_from_keel():
    scenario = rg.RidgeGroundingScenario()
    mapped = rg.kerr_foundation_for_keel(scenario)
    assert mapped.equivalent_winkler == pytest.approx(scenario.foundation_modulus)
    assert mapped.shear == pytest.approx(scenario.foundation_modulus
                                        * (scenario.keel_thickness / 2) ** 2)


def test_spearman_monotonic_detects_trends():
    assert rg.spearman_monotonic([1, 2, 3, 4], [1, 4, 9, 16])["spearman_rho"] == pytest.approx(1.0)
    assert rg.spearman_monotonic([1, 2, 3, 4], [4, 3, 2, 1])["spearman_rho"] == pytest.approx(-1.0)
    assert abs(rg.spearman_monotonic([1, 2, 3, 4], [1, 3, 2, 4])["spearman_rho"]) < 1.0
    with pytest.raises(ValueError):
        rg.spearman_monotonic([1.0], [1.0])


def test_grounding_probability_is_monotone_decreasing():
    scenario = rg.RidgeGroundingScenario()
    result = rg.grounding_probability([4.0, 8.0, 12.0, 20.0, 40.0], scenario)
    assert result["monotone_decreasing"] is True
    probabilities = [row["grounding_probability"] for row in result["rows"]]
    assert probabilities[0] > 0.9
    assert probabilities[-1] < 0.05
    for value in probabilities:
        assert 0.0 <= value <= 1.0


def test_grounding_probability_validation():
    scenario = rg.RidgeGroundingScenario()
    with pytest.raises(ValueError):
        rg.grounding_probability([], scenario)
    with pytest.raises(ValueError):
        rg.grounding_probability([-1.0], scenario)


def test_fragment_interface_roundtrip():
    distribution = rg.fragment_size_distribution([0.05, 0.1, 0.2], [1.0, 2.0, 1.0])
    assert distribution["number_median_m"] == pytest.approx(0.1)
    assert distribution["total_count"] == pytest.approx(4.0)
    mapping = rg.lattice_from_fragment_sizes(distribution, particle_radius_ratio=0.5)
    assert mapping["particle_radius_m"] == pytest.approx(0.05)
    with pytest.raises(ValueError):
        rg.fragment_size_distribution([0.0, 1.0])


def test_compare_keel_with_kerr_metric_behaviour():
    foundation_case = rg.kerr_foundation_for_keel(rg.RidgeGroundingScenario())
    ei = rg.RidgeGroundingScenario().section.ei
    compliance = kf.kerr_beam_analytic(rg.RidgeGroundingScenario().keel_length, ei,
                                       foundation_case, 1.0).max_deflection
    steps = [
        {"punch_depth_m": d, "keel_length_m": 12.0,
         "beam_deflection_m": d / 2.0, "reaction_N_per_m": (d / 2.0) / compliance}
        for d in (0.01, 0.02, 0.03)
    ]
    result = rg.compare_keel_with_kerr(steps, foundation_case, ei)
    assert result["max_relative_error"] < 1e-9
    assert result["criterion_15_percent_met"] is True
    steps[0]["reaction_N_per_m"] *= 1.5
    shifted = rg.compare_keel_with_kerr(steps, foundation_case, ei)
    assert shifted["max_relative_error"] > 0.15
    assert shifted["criterion_15_percent_met"] is False


def test_contact_damage_scales_with_pressure():
    scenario = rg.RidgeGroundingScenario()
    weak = rg.contact_damage(1.0e6, scenario)
    strong = rg.contact_damage(4.0e6, scenario)
    assert strong["deflection_m"] > weak["deflection_m"]
    assert strong["absorbed_energy_J"] > weak["absorbed_energy_J"]
    assert weak["patched_area_m2"] == pytest.approx(scenario.punch_width)