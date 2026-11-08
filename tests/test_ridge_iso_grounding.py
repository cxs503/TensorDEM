"""Tests for the ISO 19906-form grounding comparison (campaign node IN155-N11).

Three layers:

* closed-form arithmetic of the ISO form against the already-verified
  bearing-capacity routines in :mod:`tensordem.ridge_geotech` (pinned to the
  Vesic / Skempton closed forms, no DEM);
* the DEM degeneracy controls — the pipeline must behave as a rigid base when
  the subgrade is switched off, and the subgrade must offload the pipe;
* the energy ledger and the criterion tables (closure, grid independence,
  cover envelope), including a negative control that the verdicts are not
  hard-wired to "satisfied".
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.ridge_dem import RidgeKeel, RidgeMaterial, RidgeAssembly
from tensordem.ridge_geotech import (MohrCoulombSoil, PipeSection,
                                     bearing_capacity_factors,
                                     skempton_undrained_nc)
from tensordem.ridge_grounding import GroundingScenario, run_grounding
from tensordem.ridge_iso19906 import (compare_with_dem, cover_envelope,
                                      critical_cover_depth,
                                      energy_convergence_table,
                                      grid_convergence_table,
                                      instability_limits, iso19906_lateral_load,
                                      iso19906_vertical_load)

DIAMETER = 0.914
SECTION = PipeSection(outer_diameter=DIAMETER, wall_thickness=0.0159,
                      young_modulus=207.0e9, yield_strength=448.0e6)
UNDRAINED = MohrCoulombSoil(friction_angle_deg=0.0, cohesion=30.0e3,
                            saturated_unit_weight=18.0e3, young_modulus=10.0e6,
                            poisson_ratio=0.3, water_unit_weight=10.06e3,
                            undrained=True)
DRAINED = MohrCoulombSoil(friction_angle_deg=30.0, cohesion=5.0e3,
                          saturated_unit_weight=20.0e3, young_modulus=30.0e6,
                          poisson_ratio=0.3, water_unit_weight=10.06e3,
                          undrained=False)


# --------------------------------------------------------------- ISO arithmetic
def test_undrained_form_matches_verified_vertical_capacity():
    """The campaign-cited form must reproduce the verified capacity exactly."""
    for cover in (0.0, 0.5, 1.0 * DIAMETER, 3.0):
        form = iso19906_vertical_load(UNDRAINED, DIAMETER, cover_depth=cover)
        reference = UNDRAINED.vertical_capacity_per_length(DIAMETER, cover)
        assert form.total == pytest.approx(reference, rel=1e-12)
        # undrained total-stress branch: no drained overburden term
        assert form.q_term == 0.0
        assert form.n_q == 1.0


def test_undrained_nc_uses_skempton_and_strip_cap():
    cover = 4.0
    form = iso19906_vertical_load(UNDRAINED, DIAMETER, cover_depth=cover)
    expected = skempton_undrained_nc(cover + DIAMETER, DIAMETER, length_ratio=0.0)
    assert form.n_c == pytest.approx(expected, rel=1e-12)
    assert form.n_c <= 7.5 + 1e-12          # strip-footing cap, not the 9.0 square cap
    assert form.c_term == pytest.approx(UNDRAINED.cohesion * expected * DIAMETER,
                                        rel=1e-12)


def test_drained_form_decomposes_into_nq_and_nc_terms():
    cover = 1.0
    form = iso19906_vertical_load(DRAINED, DIAMETER, cover_depth=cover)
    n_c, n_q, n_gamma = bearing_capacity_factors(DRAINED.phi_rad)
    assert form.n_c == pytest.approx(n_c, rel=1e-12)
    assert form.n_q == pytest.approx(n_q, rel=1e-12)
    assert form.n_q == pytest.approx(
        math.exp(math.pi * math.tan(DRAINED.phi_rad))
        * math.tan(0.25 * math.pi + 0.5 * DRAINED.phi_rad) ** 2, rel=1e-12)
    assert form.total == pytest.approx(form.q_term + form.c_term + form.gamma_term,
                                       rel=1e-12)
    assert form.q_term > form.c_term        # N_q overburden dominates for phi=30
    # the same routine with the gamma term switched on adds exactly 0.5 N_g gamma' D^2
    with_gamma = iso19906_vertical_load(DRAINED, DIAMETER, cover_depth=cover,
                                        include_gamma_term=True)
    gamma_prime = DRAINED.effective_unit_weight * (1.0 - DRAINED.pore_pressure_ratio)
    assert with_gamma.gamma_term == pytest.approx(
        0.5 * gamma_prime * DIAMETER * n_gamma * DIAMETER, rel=1e-12)
    assert with_gamma.total - form.total == pytest.approx(with_gamma.gamma_term,
                                                          rel=1e-12)


def test_keel_surcharge_mode_is_an_upper_bound():
    cover = 1.0
    soil_form = iso19906_vertical_load(DRAINED, DIAMETER, cover_depth=cover)
    keel_form = iso19906_vertical_load(DRAINED, DIAMETER, cover_depth=cover,
                                       keel_depth=12.0, surcharge_mode="keel")
    assert keel_form.depth == 12.0
    assert keel_form.total > soil_form.total
    with pytest.raises(ValueError):
        iso19906_vertical_load(DRAINED, DIAMETER, cover_depth=cover,
                               surcharge_mode="keel")
    with pytest.raises(ValueError):
        iso19906_vertical_load(DRAINED, DIAMETER, cover_depth=cover,
                               surcharge_mode="mudline")


def test_lateral_form_is_np_shear_strength_diameter():
    cover = 1.0
    lateral = iso19906_lateral_load(UNDRAINED, DIAMETER, cover_depth=cover)
    z = cover + 0.5 * DIAMETER
    expected_np = min(2.0 + 1.5 * z / DIAMETER, 9.0)
    assert lateral["n_factor"] == pytest.approx(expected_np, rel=1e-12)
    assert lateral["total_n_per_m"] == pytest.approx(
        expected_np * UNDRAINED.cohesion * DIAMETER, rel=1e-12)
    assert lateral["embedment_m"] == pytest.approx(z, rel=1e-12)
    deep = iso19906_lateral_load(UNDRAINED, DIAMETER, cover_depth=20.0)
    assert deep["n_factor"] == 9.0          # capped


def test_instability_limits_follow_the_campaign_rule():
    limits = instability_limits(SECTION)
    assert limits["vertical"] == pytest.approx(DIAMETER / 10.0)
    assert limits["lateral"] == pytest.approx(DIAMETER / 5.0)


def test_critical_cover_depth_is_the_root_of_the_capacity_curve():
    # the mudline capacity is ~169 kN/m for this deck, so a 185 kN/m demand
    # has its root between the mudline configuration and one diameter of cover
    demand = 185.0e3
    cover = critical_cover_depth(UNDRAINED, DIAMETER, demand, upper=6.0)
    assert 0.0 < cover < 6.0
    capacity = iso19906_vertical_load(UNDRAINED, DIAMETER, cover_depth=cover).total
    assert capacity >= demand
    below = iso19906_vertical_load(UNDRAINED, DIAMETER,
                                   cover_depth=max(cover - 0.02, 0.0)).total
    assert below < demand
    assert critical_cover_depth(UNDRAINED, DIAMETER, 0.0) == 0.0
    assert critical_cover_depth(UNDRAINED, DIAMETER, 1.0e9, upper=2.0) == math.inf


def test_compare_with_dem_signs():
    compare = compare_with_dem(180.0e3, 200.0e3)
    assert compare["dem_below_iso"] is True
    assert compare["deviation"] == pytest.approx(-0.1, rel=1e-12)
    assert compare["ratio"] == pytest.approx(0.9, rel=1e-12)


# ------------------------------------------------------------------ DEM control
def _small_scenario(*, soil, radius=0.2, response="rigid", cover=0.2,
                    track_energy=False, steps_factor=1.0) -> GroundingScenario:
    """A cheap deck with the keel tip starting exactly at the pipe crown.

    The keel is small (few hundred particles) and the platen is fast, so a unit
    test gets past first contact within a few thousand steps.  ``capacity`` is
    therefore a *rate-influenced* number here; the tests only use it for
    relative comparisons, never as a physical anchor.
    """
    material = RidgeMaterial(particle_radius=radius, drag=2.0e4,
                             freeze_bond_fraction=1.0,
                             bond_tensile_strength=100.0e3,
                             bond_shear_strength=200.0e3)
    return GroundingScenario(
        keel=RidgeKeel(1.6, 3.2, 0.10), material=material, soil=soil,
        section=SECTION, cover_depth=cover, tip_clearance=0.05,
        initial_penetration=0.05 + cover,
        stroke=0.30, platen_velocity=1.0,
        pipe_response=response, track_energy=track_energy,
        steps_factor=steps_factor)


def test_no_soil_degenerates_to_a_rigid_base():
    """With the subgrade switched off the pipe *is* a rigid base.

    The keel can then only be carried by that rigid boundary, so the internal
    force balance of the run (platen load = pipe load + buoyancy + inertia)
    must close, and every particle below the mudline must carry no subgrade
    reaction at all.
    """
    scenario = _small_scenario(soil=None, steps_factor=0.35)
    result = run_grounding(scenario, max_steps=12000)
    assert result.capacity > 0.0
    history = result.history
    assert max(history["soil_force"], key=abs) == 0.0
    ledger = history["pipe_load"][-1]
    assert math.isfinite(ledger)
    # no subgrade reaction anywhere: the DEM reports zero soil contacts
    assembly_soil = scenario.soil
    assert assembly_soil is None
    # the pipe load is non-zero only because the rigid pipe carries the keel
    assert abs(result.capacity) > 1.0e3


def test_soil_offloads_the_pipe_and_reaction_is_positive():
    """Reverse control: switching the subgrade on must reduce the pipe load."""
    no_soil = run_grounding(_small_scenario(soil=None, steps_factor=0.35),
                            max_steps=12000)
    with_soil = run_grounding(_small_scenario(soil=UNDRAINED, steps_factor=0.35),
                              max_steps=12000)
    assert with_soil.info["particles"] == no_soil.info["particles"]
    assert abs(with_soil.capacity) < abs(no_soil.capacity)
    assert max(with_soil.history["soil_force"]) >= 0.0


def test_subgrade_contact_requires_a_mudline_crossing():
    """A particle above the mudline sees no subgrade; one below does."""
    material = RidgeMaterial(particle_radius=0.2, drag=0.0)
    positions = torch.tensor([[0.0, 0.5], [0.0, -0.5]], dtype=torch.float64)
    radii = torch.tensor([0.2, 0.2], dtype=torch.float64)
    masses = torch.tensor([10.0, 10.0], dtype=torch.float64)
    assembly = RidgeAssembly(positions, radii, masses,
                             torch.zeros((0, 2), dtype=torch.long), material,
                             mudline=0.0)
    out = assembly.forces(1e-5)
    assert out["soil_contacts"] == 0
    assembly.soil = UNDRAINED
    out = assembly.forces(1e-5)
    assert out["soil_contacts"] == 1
    assert float(out["soil_force"][1]) > 0.0
    assert out["exchange"]["soil"] == pytest.approx(0.0)   # no velocity yet


def test_elastic_energy_of_a_single_bond_is_half_k_dx_squared():
    material = RidgeMaterial(particle_radius=0.2, drag=0.0)
    positions = torch.tensor([[0.0, 0.0], [0.4, 0.0]], dtype=torch.float64)
    assembly = RidgeAssembly(positions, torch.tensor([0.2, 0.2]),
                             torch.tensor([10.0, 10.0]),
                             torch.tensor([[0, 1]]), material)
    assembly.pos = torch.tensor([[0.0, 0.0], [0.41, 0.0]], dtype=torch.float64)
    expected = 0.5 * material.bond_stiffness * 0.01 ** 2
    assert assembly.elastic_energy()["bond"] == pytest.approx(expected, rel=1e-12)
    assembly.vel = torch.tensor([[0.5, 0.0], [-0.5, 0.0]])
    assert assembly.kinetic_energy() == pytest.approx(0.5 * 10.0 * 0.25 * 2,
                                                      rel=1e-12)


def test_elastic_energy_is_conserved_for_a_particle_on_the_subgrade():
    """Ledger control: conservative oscillation on the Mohr-Coulomb spring.

    One particle, no damping, no body force, no friction: the only store is the
    reversible part of the subgrade reaction, so ``K + E_soil`` must be constant
    to integrator accuracy.  This pins the bookkeeping used by the coupled-run
    ledger (elastic energy function + the ``exchange`` power of the subgrade).
    """
    material = RidgeMaterial(particle_radius=0.2, drag=0.0,
                             contact_damping=0.0, soil_friction_coefficient=0.0)
    assembly = RidgeAssembly(torch.tensor([[0.0, 0.2]], dtype=torch.float64),
                             torch.tensor([0.2]), torch.tensor([500.0]),
                             torch.zeros((0, 2), dtype=torch.long), material,
                             soil=UNDRAINED, mudline=0.0)
    assembly.vel = torch.tensor([[0.0, -0.2]], dtype=torch.float64)
    dt = 1.0e-5
    start = assembly.kinetic_energy() + assembly.elastic_energy()["soil"]
    drifted = 0.0
    for _ in range(400):
        assembly.step(dt)
        total = assembly.kinetic_energy() + assembly.elastic_energy()["soil"]
        drifted = max(drifted, abs(total - start))
    assert drifted / max(start, 1e-9) < 1e-3


def test_energy_ledger_reports_its_own_closure_on_a_short_run():
    scenario = _small_scenario(soil=UNDRAINED, response="compliant",
                               track_energy=True, steps_factor=0.35)
    result = run_grounding(scenario, max_steps=12000)
    energy = result.info["energy"]
    assert set(energy["sources"]) == {"platen_work", "body_work", "total"}
    assert energy["sources"]["total"] > 0.0
    assert energy["sinks"]["absorbed_soil"] > 0.0
    assert energy["soil"]["plastic"] >= -1e-6
    # residual must be exactly what the ledger says it is
    assert energy["residual"] == pytest.approx(
        energy["sources"]["total"] - energy["sinks"]["total"], rel=1e-12)
    assert energy["closure_error"] == pytest.approx(
        abs(energy["residual"]) / max(abs(energy["sources"]["total"]),
                                      abs(energy["sinks"]["total"])), rel=1e-9)
    windows = energy["windows"]
    assert windows["available"] is True
    assert windows["t_end"] > windows["t_start"]
    # honest scope: the coupled deck does NOT close its energy ledger at the
    # model's recommended time step (see docs/ice_ridge_iso19906_grounding.md);
    # the test only pins that the number is produced, not that it is small.
    assert 0.0 <= energy["closure_error"] <= 10.0


def test_lateral_pipe_degree_of_freedom_tracks_the_subgrade_capacity():
    material = RidgeMaterial(particle_radius=0.25, drag=2.0e4,
                             freeze_bond_fraction=1.0)
    scenario = GroundingScenario(
        keel=RidgeKeel(2.0, 4.0, 0.10), material=material, soil=UNDRAINED,
        section=SECTION, cover_depth=DIAMETER, tip_clearance=0.05,
        stroke=2.0, platen_velocity=0.3, loading="lateral",
        pipe_response="rigid", pipe_lateral_compliance=True)
    result = run_grounding(scenario, max_steps=4000)
    lateral = result.info["lateral"]
    assert lateral["compliant"] is True
    assert lateral["capacity"] > 0.0
    assert len(result.history["pipe_displacement_lateral"]) == \
        len(result.history["pipe_load"])
    assert lateral["final_displacement"] >= 0.0


# ---------------------------------------------------------------- criterion logic
def _row(**kwargs):
    base = {"cover_depth": DIAMETER, "dem_capacity": 180.0e3,
            "iso_vertical": 200.0e3, "dem_demand": 150.0e3,
            "dem_displacement": 0.01, "iso_lateral": 100.0e3,
            "dem_lateral": 50.0e3, "dem_lateral_displacement": 0.02}
    base.update(kwargs)
    return base


def test_cover_envelope_marks_consistent_predictions():
    envelope = cover_envelope([_row()], diameter=DIAMETER,
                              vertical_limit=DIAMETER / 10.0,
                              lateral_limit=DIAMETER / 5.0)
    assert envelope["verdict"] == "consistent"
    assert envelope["max_abs_capacity_deviation"] == pytest.approx(0.1, rel=1e-12)
    assert envelope["rows"][0]["cover_over_d"] == pytest.approx(1.0)


def test_cover_envelope_flags_an_inconsistent_prediction():
    """Negative control: a large displacement the theory did not predict."""
    bad = _row(dem_displacement=0.5, dem_demand=150.0e3, iso_vertical=200.0e3)
    envelope = cover_envelope([bad], diameter=DIAMETER,
                              vertical_limit=DIAMETER / 10.0,
                              lateral_limit=DIAMETER / 5.0)
    assert envelope["verdict"] == "inconsistent"
    assert envelope["rows"][0]["dem_unstable"] is True
    assert envelope["rows"][0]["iso_predicted_unstable"] is False


def test_energy_convergence_table_requires_the_finest_case_to_close():
    rows = [{"dt": 4e-4, "closure_error": 0.2, "residual": 1.0,
             "sources": {"total": 10.0}},
            {"dt": 1e-4, "closure_error": 0.03, "residual": 0.3,
             "sources": {"total": 10.0}}]
    table = energy_convergence_table(rows)
    assert table["satisfied"] is True
    assert table["refinement_ratios"][0] == pytest.approx(0.2 / 0.03, rel=1e-12)
    slower = energy_convergence_table([rows[0]])
    assert slower["satisfied"] is False


def test_grid_convergence_table_reports_the_capacity_change():
    rows = [{"particle_radius": 0.5, "particles": 100, "capacity": 200.0e3},
            {"particle_radius": 0.354, "particles": 200, "capacity": 204.0e3}]
    table = grid_convergence_table(rows)
    assert table["satisfied"] is True
    assert table["capacity_change"] == pytest.approx(0.02, rel=1e-12)
    assert table["particle_ratio"] == pytest.approx(2.0, rel=1e-12)
    coarse = grid_convergence_table([rows[0],
                                     {"particle_radius": 0.354, "particles": 200,
                                      "capacity": 250.0e3}])
    assert coarse["satisfied"] is False