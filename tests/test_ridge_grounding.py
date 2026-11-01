"""Tests for the ice-ridge / pipeline / seabed grounding stack.

Coverage map (criterion (e) of the campaign): continuum limit equilibrium and
subgrade curve, DNV-OS-F101 section capacities, Winkler beam against the
closed-form Hetenyi solution, DEM regressions (repulsive contact sign, passive
stability, reproducible keel packing, ductile-bond rejection), the particle-level
micro -> macro calibration harness, and the coupled runner.
"""
import math
import os
import sys

import pytest
import torch

# make the working tree's sources importable even when an installed copy of
# ``tensordem`` points at a different checkout
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

from tensordem.ridge_dem import (RidgeAssembly, RidgeKeel, RidgeMaterial, _bond_pairs,
                                calibrate_rubble_strength, generate_keel_assembly)
from tensordem.ridge_geotech import (MohrCoulombSoil, PipeSection, SofteningVerticalSpring,
                                     bearing_capacity_factors, dnv_os_f101_utilization,
                                     hetenyi_infinite_beam_patch_load,
                                     hetenyi_infinite_beam_point_load, hetenyi_verification,
                                     solve_beam_on_winkler)
from tensordem.ridge_grounding import (GroundingScenario, GroundingResult,
                                       axial_footprint_loads, continuum_limit_capacity,
                                       load_displacement_metrics, pipe_stress_profile,
                                       run_grounding)


@pytest.fixture(autouse=True)
def _single_thread():
    torch.set_num_threads(1)


def _soil(**kwargs):
    defaults = dict(friction_angle_deg=28.0, cohesion=5.0e3, saturated_unit_weight=18.0e3,
                    young_modulus=10.0e6, water_unit_weight=10.06e3)
    defaults.update(kwargs)
    return MohrCoulombSoil(**defaults)


def _section():
    return PipeSection(0.914, 0.0159, 207.0e9, 448.0e6)


# --------------------------------------------------------------- limit equilibrium
def test_bearing_capacity_factors_match_textbook_values():
    n_c, n_q, n_gamma = bearing_capacity_factors(math.radians(30.0))
    assert n_q == pytest.approx(18.40, abs=0.02)
    assert n_c == pytest.approx(30.14, abs=0.02)
    assert n_gamma == pytest.approx(22.40, abs=0.02)
    n_c0, n_q0, n_gamma0 = bearing_capacity_factors(0.0)
    assert (n_c0, n_q0, n_gamma0) == (5.14, 1.0, 0.0)


def test_bearing_capacity_factors_increase_with_friction_and_validate():
    values = [bearing_capacity_factors(math.radians(phi))[1] for phi in (10.0, 20.0, 35.0)]
    assert values[0] < values[1] < values[2]
    with pytest.raises(ValueError):
        bearing_capacity_factors(math.radians(90.0))
    with pytest.raises(ValueError):
        bearing_capacity_factors(-0.1)


def test_undrained_vertical_capacity_uses_skempton_embedment():
    soil = _soil(friction_angle_deg=0.0, cohesion=30.0e3, undrained=True)
    shallow = soil.vertical_capacity_per_length(0.914, 0.0)
    deep = soil.vertical_capacity_per_length(0.914, 2.0)
    assert deep > shallow
    pressure = soil.vertical_pressure_capacity(0.914, 0.914)
    assert pressure == pytest.approx(30.0e3 * min(5.14 * 1.2, 9.0) * 1.2, rel=1e-9)


def test_effective_stress_and_pore_pressure_ratio_reduce_reaction():
    drained = _soil()
    pressurised = _soil(pore_pressure_ratio=0.5)
    assert drained.effective_vertical_stress(2.0) == pytest.approx(
        drained.effective_unit_weight * 2.0)
    assert pressurised.effective_vertical_stress(2.0) == pytest.approx(
        0.5 * drained.effective_vertical_stress(2.0))
    assert pressurised.vertical_capacity_per_length(0.914, 0.5) \
        < drained.vertical_capacity_per_length(0.914, 0.5)


def test_vertical_reaction_inverts_and_saturates():
    soil = _soil()
    for load in (1.0e4, 5.0e4, 1.5e5):
        displacement = soil.displacement_for_load(load, 0.914, 0.5)
        assert soil.vertical_reaction(displacement, 0.914, 0.5) == pytest.approx(load, rel=1e-6)
    capacity = soil.vertical_capacity_per_length(0.914, 0.5)
    assert math.isinf(soil.displacement_for_load(1.01 * capacity, 0.914, 0.5))
    with pytest.raises(ValueError):
        soil.vertical_reaction(-1.0, 0.914, 0.5)


def test_lateral_limit_pressure_undrained_cap():
    soil = _soil(friction_angle_deg=0.0, cohesion=20.0e3, undrained=True)
    shallow = soil.lateral_limit_pressure(0.5, 0.914)
    deep = soil.lateral_limit_pressure(4.0, 0.914)
    assert shallow < deep
    capped = soil.lateral_limit_pressure(50.0, 0.914)
    assert capped == pytest.approx(9.0 * 20.0e3 * 0.914)


def test_softening_spring_peak_then_residual():
    spring = SofteningVerticalSpring(_soil(), 0.914, 0.5)
    description = spring.describe()
    assert description["peak_reaction"] == pytest.approx(
        0.9 * description["ultimate_bearing_capacity"])
    rising = [spring.reaction(w) for w in (0.02, 0.1, 0.2)]
    assert rising[0] < rising[1] < rising[2]
    peak = spring.reaction(description["yield_displacement"])
    softening = spring.reaction(description["yield_displacement"] + 0.3)
    residual = spring.reaction(5.0)
    assert peak > softening > residual
    assert residual == pytest.approx(0.8 * description["peak_reaction"])


# ------------------------------------------------------------------- pipe section
def test_pipe_section_properties_match_closed_form():
    section = _section()
    thin_wall_area = math.pi * 0.914 * 0.0159
    assert section.area == pytest.approx(thin_wall_area, rel=0.02)
    assert section.inertia == pytest.approx(math.pi / 64.0 * (0.914 ** 4 - 0.8822 ** 4), rel=1e-9)
    assert section.plastic_moment == pytest.approx(
        448.0e6 * (0.914 - 0.0159) ** 2 * 0.0159, rel=1e-12)
    assert section.elastic_section_modulus == pytest.approx(2.0 * section.inertia / 0.914)
    with pytest.raises(ValueError):
        PipeSection(0.2, 0.15, 2.1e11, 4.5e8)


def test_dnv_os_f101_load_controlled_interaction():
    section = _section()
    assert dnv_os_f101_utilization(0.0, 0.0, section) == pytest.approx(0.0)
    assert dnv_os_f101_utilization(section.plastic_moment, 0.0, section) == pytest.approx(
        1.0 / 0.85)
    combined = dnv_os_f101_utilization(0.5 * section.plastic_moment,
                                       0.5 * section.plastic_axial_force, section)
    assert combined == pytest.approx(0.5 / 0.85 + (0.5 / 0.85) ** 2)
    with pytest.raises(ValueError):
        dnv_os_f101_utilization(1.0, 1.0, section, alpha_c=1.5)


# ---------------------------------------------------------------- Winkler beam
def test_beam_on_winkler_matches_hetenyi_point_load():
    report = hetenyi_verification()
    assert report["deflection_relative_error"] < 0.02
    assert report["moment_relative_error"] < 0.02


def test_beam_on_winkler_matches_hetenyi_patch_load():
    report = hetenyi_verification(load=2.0e5, patch_half_length=4.0)
    assert report["moment_relative_error"] < 0.05
    assert report["profile_deflection_relative_error"] < 0.05


def test_simply_supported_beam_without_foundation():
    length, inertia, modulus, load = 10.0, 0.0123, 2.1e11, 5.0e4
    nodes = 101
    solution = solve_beam_on_winkler([load] * nodes, inertia=inertia, young_modulus=modulus,
                                     length=length, node_count=nodes, foundation_modulus=1.0,
                                     end_condition="pinned")
    analytic_moment = load * length ** 2 / 8.0
    analytic_deflection = 5.0 * load * length ** 4 / (384.0 * modulus * inertia)
    assert float(solution["moment"].max()) == pytest.approx(analytic_moment, rel=0.01)
    assert float(solution["deflection"].max()) == pytest.approx(analytic_deflection, rel=0.01)


def test_beam_solver_validates_inputs():
    with pytest.raises(ValueError):
        solve_beam_on_winkler([1.0, 2.0], inertia=1.0, young_modulus=1.0, length=1.0,
                              node_count=3, foundation_modulus=1.0)
    with pytest.raises(ValueError):
        solve_beam_on_winkler([1.0] * 5, inertia=1.0, young_modulus=1.0, length=1.0,
                              node_count=5, foundation_modulus=1.0, end_condition="fixed")


# ------------------------------------------------------------------------- DEM
def test_contact_force_is_repulsive():
    material = RidgeMaterial(particle_radius=0.5)
    positions = torch.tensor([[0.0, 0.0], [0.9, 0.0]])
    assembly = RidgeAssembly(positions, torch.tensor([0.5, 0.5]),
                             torch.tensor([1.0, 1.0]), torch.zeros((0, 2), dtype=torch.long),
                             material)
    out = assembly.forces(0.0, update_memory=False)
    force = out["force"]
    # particle 0 sits left of particle 1 and must be pushed further left
    assert float(force[0, 0]) < 0.0
    assert float(force[1, 0]) > 0.0
    assert float(force[0, 0]) == pytest.approx(-float(force[1, 0]), rel=1e-12)


def test_passive_contact_assembly_stays_at_rest():
    material = RidgeMaterial(particle_radius=0.5)
    positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [1.5, 0.866]])
    assembly = RidgeAssembly(positions, torch.full((4,), 0.5), torch.full((4,), 700.0),
                             torch.zeros((0, 2), dtype=torch.long), material)
    for _ in range(150):
        assembly.step(material.recommended_dt)
    assert assembly.kinetic_energy() < 1.0


def test_keel_geometry_and_packing_are_reproducible():
    keel = RidgeKeel(10.0, 20.0, 0.12)
    assert keel.angle_deg == pytest.approx(math.degrees(math.atan(1.0)), rel=1e-9)
    assert keel.half_width(5.0) == pytest.approx(5.0)
    assert keel.cross_section_area == pytest.approx(100.0)
    material = RidgeMaterial(particle_radius=0.5)
    first = generate_keel_assembly(keel, material)
    second = generate_keel_assembly(keel, material)
    assert first["info"]["particles"] == second["info"]["particles"]
    torch.testing.assert_close(first["positions"], second["positions"])
    assert first["info"]["driven_particles"] > 0
    assert first["info"]["bonds"] > 0


def test_material_rejects_ductile_bonds():
    with pytest.raises(ValueError):
        RidgeMaterial(bond_stiffness=1.0e6, bond_tensile_strength=200.0e3)
    with pytest.raises(ValueError):
        RidgeMaterial(
            ice_density=1100.0)


def test_calibration_harness_returns_two_mohr_circles():
    material = RidgeMaterial(particle_radius=0.5, freeze_bond_fraction=1.0,
                             bond_tensile_strength=100.0e3, bond_shear_strength=200.0e3)
    report = calibrate_rubble_strength(material, columns=6, rows=6, loading_rate=0.25,
                                       strain_target=0.05, consolidation_steps=300,
                                       max_steps=600)
    assert len(report["specimens"]) == 2
    assert all(math.isfinite(row["peak_deviator_stress"]) for row in report["specimens"])
    assert math.isfinite(report["friction_angle_deg"])
    assert report["cohesion"] >= 0.0


# ------------------------------------------------------------ coupled workflow
def _small_scenario(**kwargs):
    defaults = dict(keel=RidgeKeel(4.0, 8.0, 0.12),
                    material=RidgeMaterial(particle_radius=0.5),
                    soil=_soil(friction_angle_deg=0.0, cohesion=30.0e3, undrained=True),
                    section=_section(), cover_depth=0.35, tip_clearance=0.1,
                    stroke=0.45, platen_velocity=0.3, contact_length=12.0)
    defaults.update(kwargs)
    return GroundingScenario(**defaults)


def test_scenario_validation_and_derived_geometry():
    scenario = _small_scenario(stroke=None)
    assert scenario.pipe_centre_height == pytest.approx(-(0.35 + 0.457))
    assert scenario.stroke == pytest.approx(scenario.tip_clearance + scenario.cover_depth + 0.3)
    scenario = _small_scenario()
    with pytest.raises(ValueError):
        _small_scenario(loading="diagonal")
    with pytest.raises(ValueError):
        _small_scenario(pipe_response="springy")
    with pytest.raises(ValueError):
        _small_scenario(platen_velocity=0.0)


def test_footprint_loads_integrate_to_total_load():
    x, loads = axial_footprint_loads(2.0e5, contact_length=12.0, node_count=241)
    spacing = float(x[1] - x[0])
    integrated = float(loads.sum()) * spacing
    assert integrated == pytest.approx(2.0e5, rel=0.05)
    _, cosine = axial_footprint_loads(2.0e5, contact_length=12.0, node_count=241,
                                      shape="cosine")
    assert float(cosine.max()) > float(loads.max())
    assert float(cosine[0]) == 0.0


def test_pipe_stress_profile_matches_analytic_winkler():
    scenario = _small_scenario()
    profile = pipe_stress_profile(6.0e5, scenario)
    assert profile["moment_relative_error"] < 0.15
    assert profile["peak_utilization"] > 0.0
    assert profile["converged"]
    assert len(profile["moment"]) == len(profile["x"])


def test_continuum_limit_capacity_frictionless_limit():
    value = continuum_limit_capacity(0.0, 0.0, surcharge=0.0, diameter=1.0)
    assert value == pytest.approx(0.0)
    cohesion_only = continuum_limit_capacity(10.0e3, 0.0, surcharge=0.0, diameter=2.0)
    assert cohesion_only == pytest.approx(5.14 * 10.0e3 * 2.0)
    with_surcharge = continuum_limit_capacity(0.0, 30.0, surcharge=1.0e5, diameter=1.0)
    assert with_surcharge == pytest.approx(18.40 * 1.0e5, rel=2e-3)


def test_load_displacement_metrics_detects_peak_and_softening():
    history = {"pipe_load_smoothed": [0, 10, 40, 90, 120, 100, 80, 60, 55, 55],
               "keel_penetration": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
               "pipe_displacement": [0.0] * 10}
    result = GroundingResult(scenario=None, history=history, info={}, capacity=120.0,
                             dt=1.0, steps=10, final_pipe_displacement=0.0)
    metrics = load_displacement_metrics(result, displacement_key="keel_penetration")
    assert metrics["peak_load_per_length"] == pytest.approx(120.0)
    assert metrics["softening_ratio"] > 0.3
    assert metrics["passed"]
    monotone = dict(history)
    monotone["pipe_load_smoothed"] = [0, 10, 20, 40, 60, 80, 100, 120, 140, 160]
    failing = load_displacement_metrics(
        GroundingResult(scenario=None, history=monotone, info={}, capacity=160.0,
                        dt=1.0, steps=10, final_pipe_displacement=0.0),
        displacement_key="keel_penetration")
    assert not failing["checks"]["peak_before_end"]
    assert not failing["passed"]


def test_run_grounding_small_case_is_finite_and_balanced():
    result = run_grounding(_small_scenario())
    assert result.steps > 100
    assert math.isfinite(result.capacity)
    assert max(result.history["pipe_load"]) >= 0.0
    assert len(result.history["pipe_load"]) == len(result.history["time"])
    assert max(result.history["drive_force"]) > 0.0
    assert result.info["particles"] > 8
    assert result.info["soil_spring"]["peak_reaction"] > 0.0
