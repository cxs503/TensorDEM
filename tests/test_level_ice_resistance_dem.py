"""Level-ice resistance campaign: anchor formulas, sheet mechanics, controls.

The suite has three jobs: (i) pin the engineering anchors and their provenance,
(ii) pin the bonded-particle section mechanics against closed-form beam /
foundation solutions and against the assembled stiffness matrix, and (iii)
exercise the negative controls that must *fail* (invalid configs / no strength /
no advance).  Nothing here duplicates the campaign artifact; the campaign itself
is run by ``scripts/run_level_ice_campaign.py`` and audited by
``docs/LEVEL_ICE_RESISTANCE_DEM.md``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.level_ice import campaign as C
from tensordem.level_ice import formula as F
from tensordem.level_ice.sheet import LevelIceSheet, SheetConfig


# ---------------------------------------------------------------------------
# anchors and provenance
# ---------------------------------------------------------------------------
def test_ledger_hash_is_stable_hex():
    digest = F.ledger_sha256()
    assert isinstance(digest, str) and len(digest) == 64
    assert all(ch in "0123456789abcdef" for ch in digest)
    assert digest == F.ledger_sha256()          # deterministic
    assert F.COEFFICIENT_LEDGER["literature_agreement"][1] == "WITHHELD"


def test_lindqvist_dimensional_identities():
    """R_crush ~ sigma_c h^2, R_bend ~ sigma_f B h^2 / L, R_sub ~ rho g h^2 B."""
    ice = F.IceState(thickness_m=1.0)
    base = F.lindqvist_components(ice, F.ICEBREAKER, 0.0)
    bigger_c = F.lindqvist_components(
        F.IceState(thickness_m=1.0, compressive_strength_pa=3.0e6), F.ICEBREAKER)
    assert bigger_c["crushing_N"] == pytest.approx(2.0 * base["crushing_N"])
    assert bigger_c["bending_N"] == pytest.approx(base["bending_N"])
    bigger_f = F.lindqvist_components(
        F.IceState(thickness_m=1.0, flexural_strength_pa=1.0e6), F.ICEBREAKER)
    assert bigger_f["bending_N"] == pytest.approx(2.0 * base["bending_N"])
    thick = F.lindqvist_components(F.IceState(thickness_m=2.0), F.ICEBREAKER)
    assert thick["crushing_N"] / base["crushing_N"] == pytest.approx(4.0)
    assert thick["bending_N"] / base["bending_N"] == pytest.approx(4.0)
    assert thick["submersion_N"] / base["submersion_N"] == pytest.approx(4.0)


def test_added_drag_is_quadratic_in_speed():
    ice = F.IceState(thickness_m=1.0)
    low = F.added_drag(ice, F.ICEBREAKER, 1.0)
    high = F.added_drag(ice, F.ICEBREAKER, 2.0)
    assert high / low == pytest.approx(4.0)


def test_component_split_sums_to_total():
    payload = F.lindqvist_components(F.IceState(thickness_m=1.2), F.TRANSPORT, 2.5)
    parts = (payload["crushing_N"] + payload["bending_N"]
             + payload["submersion_N"] + payload["added_drag_N"])
    assert parts == pytest.approx(payload["total_N"], rel=1e-12)
    assert sum(payload[f"{name}_fraction"] for name in
               ("crushing", "bending", "submersion")) < 1.0


def test_negative_control_zero_thickness_anchor_rejected():
    """NEGATIVE CONTROL: an ice sheet of zero thickness cannot be evaluated."""
    with pytest.raises(ValueError):
        F.IceState(thickness_m=0.0)
    with pytest.raises(ValueError):
        F.HullForm(length_m=0.0)


def test_negative_control_zero_strength_gives_zero_resistance():
    """NEGATIVE CONTROL: with zero strengths the anchor collapses to the drag term."""
    zero = F.IceState(thickness_m=1.0, compressive_strength_pa=1e-12,
                      flexural_strength_pa=1e-12)
    payload = F.lindqvist_components(zero, F.ICEBREAKER, 0.0)
    assert payload["crushing_N"] < 1e-6
    assert payload["bending_N"] < 1e-6
    # the submersion term is buoyancy driven and must survive
    assert payload["submersion_N"] > 1.0


def test_riska_anchor_agrees_with_three_field_baseline():
    """The Riska anchor must not duplicate a second source of truth."""
    from tensordem.threefield.resistance import (Environment, HullGeometry,
                                                 IceProperties, riska_resistance)
    ice = F.IceState(thickness_m=1.0)
    ours = F.riska_components(ice, F.ICEBREAKER, 2.0)
    theirs = riska_resistance(
        IceProperties(thickness_m=1.0), HullGeometry(
            length_m=F.ICEBREAKER.length_m, beam_m=F.ICEBREAKER.beam_m,
            draught_m=F.ICEBREAKER.draught_m,
            waterline_entry_angle_deg=F.ICEBREAKER.entry_angle_deg,
            stem_angle_deg=F.ICEBREAKER.stem_angle_deg,
            friction_coefficient=F.ICEBREAKER.friction_coefficient),
        2.0, Environment())
    assert ours["total_N"] == pytest.approx(theirs["total_N"], rel=1e-12)


def test_reference_table_hash_and_missing_cells():
    rows = F.level_ice_cases()
    assert len(rows) == 10
    assert all(len(row["source_sha256"]) == 64 for row in rows)
    assert len({row["source_sha256"] for row in rows}) == 1
    thin = [r for r in rows if r["test_id"] == "401"][0]
    assert thin["measured_mean_resistance_table3_kN"] == 150.0
    assert thin["measured_std_kN"] is None            # not reported, not invented
    assert thin["experimental_reference_table8_kN"] == 152.0


def test_reference_audit_reports_published_formula_errors():
    audit = F.reference_summary()
    assert audit["n_cases"] == 10
    lind = audit["columns"]["lindqvist_prediction_kN"]
    assert lind["n"] == 10
    assert 0.0 < lind["mape_pct"] < 100.0


def test_published_column_structure_fit_is_reported_with_residual():
    fit = F.fit_hull_to_published()
    assert fit["n_cases"] == 10
    assert fit["rms_error_pct"] > 0.0
    assert "WITHHELD" in fit["grade"]


def test_pearson_and_power_law_helpers():
    assert F.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert F.pearson([1, 2, 3], [-2, -4, -6]) == pytest.approx(-1.0)
    assert F.power_law_exponent([1.0, 2.0, 4.0], [3.0, 12.0, 48.0]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# section mechanics
# ---------------------------------------------------------------------------
def test_force_assembly_is_minus_stiffness_times_displacement():
    """The dynamic bond forces must equal ``-K u`` to machine precision."""
    config = SheetConfig(thickness_m=1.0, particle_size_m=0.25, length_m=2.0,
                         buoyancy=False, drag_coefficient=0.0,
                         flexural_strength_pa=1e15, compressive_strength_pa=1e15,
                         shear_strength_pa=1e15)
    sheet = LevelIceSheet(config)
    rng = np.random.default_rng(7)
    sheet.u = rng.normal(scale=1e-6, size=(sheet.n, 3))
    fx, fz, mz = sheet.bond_forces()
    stiffness = sheet.stiffness_matrix().toarray()
    dofs = np.stack((sheet.u[:, 0], sheet.u[:, 1], sheet.u[:, 2]), axis=1).ravel()
    reference = -stiffness @ dofs
    dynamic = np.stack((fx, fz, mz), axis=1).ravel()
    assert np.abs(dynamic - reference).max() <= 1e-10 * max(
        np.abs(reference).max(), 1.0)


def test_cantilever_deflection_matches_euler_bernoulli():
    config = SheetConfig(thickness_m=1.0, particle_size_m=0.125, length_m=16.0,
                         buoyancy=False, drag_coefficient=0.0)
    sheet = LevelIceSheet(config)
    sheet.clamp_column(0)
    load = 1.0e4
    loads = np.zeros((sheet.n, 2))
    tip = sheet.grid_x == sheet.nx - 1
    loads[tip, 1] = -load / tip.sum()
    displacement = sheet.static_displacements(loads)
    modulus = config.elastic_modulus_pa * config.out_of_plane_width_m
    inertia = config.thickness_m**3 / 12.0
    analytic = -load * config.length_m**3 / (3.0 * modulus * inertia)
    assert displacement[tip, 1].mean() / analytic == pytest.approx(1.0, abs=0.06)


def test_buoyancy_foundation_matches_beam_on_elastic_foundation():
    config = SheetConfig(thickness_m=1.0, particle_size_m=0.25, length_m=120.0)
    sheet = LevelIceSheet(config)
    load = 1.0e4
    loads = np.zeros((sheet.n, 2))
    mid = (sheet.grid_x == sheet.nx // 2) & (sheet.grid_z == sheet.nz // 2)
    loads[mid, 1] = -load / mid.sum()
    displacement = sheet.static_displacements(loads)
    foundation = config.water_density_kg_m3 * config.gravity_m_s2
    inertia = config.elastic_modulus_pa * config.thickness_m**3 / 12.0
    beta = (foundation / (4.0 * inertia)) ** 0.25
    analytic = -load * beta / (2.0 * foundation)
    assert displacement[mid, 1].min() / analytic == pytest.approx(1.0, abs=0.06)


def test_negative_control_timestep_above_stability_limit_rejected():
    """NEGATIVE CONTROL: an over-large explicit step must be refused."""
    config = SheetConfig(thickness_m=1.0, particle_size_m=0.25, length_m=4.0)
    reference = LevelIceSheet(config)
    with pytest.raises(ValueError):
        LevelIceSheet(SheetConfig(thickness_m=1.0, particle_size_m=0.25,
                                  length_m=4.0, dt=10.0 * reference.dt))


def test_negative_control_zero_thickness_sheet_rejected():
    """NEGATIVE CONTROL: a zero-thickness sheet is not a valid configuration."""
    with pytest.raises(ValueError):
        SheetConfig(thickness_m=0.0)
    with pytest.raises(ValueError):
        SheetConfig(thickness_m=1.0, particle_size_m=0.0)


def test_mass_scaling_increases_the_stable_step():
    base = LevelIceSheet(SheetConfig(thickness_m=1.0, particle_size_m=0.25,
                                     length_m=4.0))
    scaled = LevelIceSheet(SheetConfig(thickness_m=1.0, particle_size_m=0.25,
                                       length_m=4.0, mass_scale=25.0))
    assert scaled.dt / base.dt == pytest.approx(5.0, rel=0.05)


def test_hull_split_is_additive_and_bounded():
    config = SheetConfig(thickness_m=1.0, particle_size_m=0.25, length_m=12.0,
                         mass_scale=400.0)
    sheet = LevelIceSheet(config)
    sheet.hull_x0 = 0.30
    for _ in range(400):
        hull = sheet.step()
    assert hull["total"] == pytest.approx(hull["crushing"] + hull["flexure"]
                                          + hull["friction"], rel=1e-12)
    assert hull["max_penetration"] >= 0.0


def test_zero_speed_hull_produces_no_resistance():
    spec = C.CaseSpec(hull="icebreaker", thickness_m=1.0, speed_m_s=1e-9,
                      layers=2, length_m=4.0, advance_m=1e-9 * 1e-3 * 50.0,
                      mass_scale=400.0, label="zero-speed")
    case = C.run_case(spec)
    assert case["mean_total_n_per_m"] == pytest.approx(0.0, abs=1e-9)


def test_bending_capacity_dimensional_scaling():
    check = C.dimensional_scaling_check()
    assert check["sigma_exponent"] == pytest.approx(1.0, abs=0.05)
    assert check["h_exponent"] == pytest.approx(2.0, abs=0.10)
    assert check["pass"] is True
    for case in check["cases"]:
        assert case["ratio"] == pytest.approx(1.0, abs=0.10)


def test_gci_helper_recovers_known_order_and_index():
    first, second = 1.0, 1.08
    third = second + 0.25 * ((second - first) / first) * second  # p = 2, r = 2
    values = [first, second, third]
    result = C._gci(values, [1, 2, 4], [])
    assert result["observed_order"] == pytest.approx(2.0, abs=0.05)
    assert 0.0 < result["gci_pct"] < 15.0


def test_map_to_ship_uses_declared_widths():
    case = {"spec": {"thickness_m": 1.0}, "crushing_n_per_m": 1000.0,
            "flexure_n_per_m": 100.0, "friction_n_per_m": 50.0}
    mapped = C.map_to_ship(case, F.ICEBREAKER)
    assert mapped["crush_width_m"] == pytest.approx(1.0)
    assert mapped["total_N"] == pytest.approx(1000.0 * 1.0 + 150.0 * 20.0)
    wide = C.map_to_ship(case, F.ICEBREAKER, crush_width_factor=2.0)
    assert wide["total_N"] > mapped["total_N"]


def test_case_record_schema():
    spec = C.CaseSpec(hull="icebreaker", thickness_m=1.0, speed_m_s=3.0,
                      layers=2, length_m=6.0, advance_m=0.6, mass_scale=400.0,
                      label="schema")
    case = C.run_case(spec)
    for key in ("spec", "config", "dt_s", "mean_total_n_per_m",
                "crushing_n_per_m", "flexure_n_per_m", "friction_n_per_m",
                "history", "n_steps", "broken_fraction"):
        assert key in case
    assert len(case["history"]["total_n_per_m"]) == len(case["history"]["advance_m"])
    assert case["n_steps"] > 0
    assert math.isfinite(case["mean_total_n_per_m"])