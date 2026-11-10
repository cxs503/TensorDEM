"""Tests for the broken-ice direct-navigation validation surface.

The suite covers the contact laws against their closed forms, the ship outline
geometry, the field construction, the transit runner end-to-end (including the
field-free control and the determinism of a seeded run), and the criterion
machinery (C1-C7) on synthetic inputs so that a verdict can be checked without
running a ship-scale case.
"""
from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from tensordem.broken_ice_resistance.anchors import (
    broken_ice_closed_form, lindqvist_anchor_scores, lindqvist_reference,
    mt_uikku_level_ice_cases)
from tensordem.broken_ice_resistance.contact import (BrokenIceChannel,
                                                     PenaltyIceChannel)
from tensordem.broken_ice_resistance.hull import ShipForm
from tensordem.broken_ice_resistance.ledger import (CASE_SPECS, MATERIAL,
                                                    STANDARD_CASE, CaseSpec,
                                                    MaterialState, ledger_sha256)
from tensordem.broken_ice_resistance.simulate import run_case
from tensordem.broken_ice_resistance.validation import (
    THRESHOLDS, _monotone_increasing, _pearson, count_tests, evaluate_criteria)

SMALL = CaseSpec(name="small", domain_length_m=60.0, domain_width_m=45.0,
                 mean_floe_radius_m=3.0, time_step_s=5.0e-3, speed_m_s=3.0)


def _channel(spec=SMALL, law="crushing") -> BrokenIceChannel:
    return BrokenIceChannel.from_case(replace(spec, law=law))


# --- ledger / specification -------------------------------------------------
def test_ledger_is_graded_and_hash_is_stable() -> None:
    digest = ledger_sha256()
    assert len(digest) == 64 and digest == ledger_sha256()
    grades = {grade for _v, grade, _t in
              __import__("tensordem.broken_ice_resistance.ledger",
                         fromlist=["COEFFICIENT_LEDGER"]).COEFFICIENT_LEDGER.values()}
    assert grades <= {"DERIVED", "CITED", "CONVENTION", "WITHHELD"}
    assert "CITED" in grades and "DERIVED" in grades


def test_material_is_the_mt_uikku_sheet_iii_state() -> None:
    rows = {r["test_id"]: r for r in mt_uikku_level_ice_cases()}
    row = rows["301"]
    assert MATERIAL["compressive_strength_pa"] == pytest.approx(
        1e3 * row["compressive_strength_kPa"], rel=1e-3)
    assert MATERIAL["flexural_strength_pa"] == pytest.approx(
        1e3 * row["flexural_strength_kPa"], rel=1e-3)
    assert len(rows) == 10
    # documented data gap: no broken-ice channel case is shipped
    assert all(r["ice_type"] == "level" for r in rows.values())


def test_case_spec_rejects_unphysical_input() -> None:
    with pytest.raises(ValueError):
        CaseSpec(concentration=1.5)
    with pytest.raises(ValueError):
        CaseSpec(law="spring")
    with pytest.raises(ValueError):
        CaseSpec(hull_form="barge")
    with pytest.raises(ValueError):
        CaseSpec(thickness_m=0.0)
    with pytest.raises(ValueError):
        MaterialState(ice_density_kg_m3=1100.0)


# --- ship outline -----------------------------------------------------------
def test_bow_length_follows_the_entry_angle() -> None:
    ship = ShipForm(length=105.0, beam=26.0, entry_angle_deg=18.0)
    assert ship.bow_length == pytest.approx(0.5 * 26.0 / math.tan(math.radians(18.0)))
    block = ShipForm(length=105.0, beam=26.0, form="block")
    assert block.bow_length == 0.0


def test_signed_distance_sign_and_outward_normals() -> None:
    ship = ShipForm(length=105.0, beam=26.0, entry_angle_deg=18.0, x_start=5.0)
    x_stem = ship.x_centre(0.0)
    inside = torch.tensor([[x_stem - 0.5 * ship.length, 0.0]], dtype=torch.float64)
    starboard = torch.tensor([[x_stem - 0.5 * ship.length, 20.0]], dtype=torch.float64)
    ahead = torch.tensor([[x_stem + 30.0, 0.0]], dtype=torch.float64)
    s_in, _ = ship.signed_distance(inside, x_stem)
    s_out, n_out = ship.signed_distance(starboard, x_stem)
    s_ahead, n_ahead = ship.signed_distance(ahead, x_stem)
    assert float(s_in[0]) < 0.0                      # inside the waterline
    assert float(s_out[0]) == pytest.approx(7.0, abs=1e-9)   # 20 - 13 = 7 m off the side
    assert float(n_out[0, 1]) == pytest.approx(1.0, abs=1e-9)
    assert float(s_ahead[0]) > 0.0 and float(n_ahead[0, 0]) > 0.0


def test_wedge_bow_deflects_ice_sideways_not_broadside() -> None:
    ship = ShipForm(length=105.0, beam=26.0, entry_angle_deg=18.0, x_start=5.0)
    x_stem = ship.x_centre(0.0)
    probe = torch.tensor([[x_stem - 0.5 * ship.bow_length, 0.0]], dtype=torch.float64)
    _s, normal = ship.signed_distance(probe, x_stem)
    # a floe on the bow centreline sees an inclined face, i.e. a lateral normal
    assert abs(float(normal[0, 1])) > 0.1


# --- contact laws -----------------------------------------------------------
def test_crushing_law_matches_the_hertzian_chord_force() -> None:
    sim = _channel()
    pen = torch.tensor([0.02, 0.05], dtype=torch.float64)
    radius = torch.tensor([5.0, 5.0], dtype=torch.float64)
    thickness = torch.tensor([1.0, 1.0], dtype=torch.float64)
    force, stiffness = sim._normal_force(pen, radius, thickness)
    eps = sim.chord_regularisation_m
    expected = (MATERIAL["compressive_strength_pa"] * 1.0
                * 2.0 * torch.sqrt(2.0 * 5.0 * (pen + eps)))
    assert torch.allclose(force, expected, rtol=1e-9)
    assert torch.all(torch.diff(stiffness) < 0)      # softening chord: dF/ddelta falls


def test_crushing_law_caps_the_contact_width_at_the_floe_diameter() -> None:
    sim = _channel()
    pen = torch.tensor([5.0], dtype=torch.float64)   # deep engagement
    force, _ = sim._normal_force(pen, torch.tensor([5.0], dtype=torch.float64),
                                 torch.tensor([1.0], dtype=torch.float64))
    assert float(force[0]) == pytest.approx(
        MATERIAL["compressive_strength_pa"] * 1.0 * 2.0 * 5.0, rel=1e-9)


def test_penalty_and_crushing_laws_agree_at_the_calibration_point() -> None:
    """Both laws are calibrated at delta = 1 % of the floe diameter."""
    crushing = _channel()
    penalty = PenaltyIceChannel.from_case(SMALL.with_(law="penalty"))
    diameter = 2.0 * 5.0
    pen = torch.tensor([0.01 * diameter], dtype=torch.float64)
    radius = torch.tensor([5.0], dtype=torch.float64)
    thickness = torch.tensor([1.0], dtype=torch.float64)
    f_crushing, _ = crushing._normal_force(pen, radius, thickness)
    f_penalty, _ = penalty._normal_force(pen, radius, thickness)
    # the Hertzian chord at the calibration penetration equals the linear spring
    assert float(f_crushing[0]) == pytest.approx(float(f_penalty[0]), rel=0.05)


def test_penalty_law_is_linear_in_penetration() -> None:
    sim = PenaltyIceChannel.from_case(SMALL.with_(law="penalty"))
    pen = torch.tensor([0.01, 0.2], dtype=torch.float64)
    force, stiffness = sim._normal_force(
        pen, torch.tensor([5.0, 5.0], dtype=torch.float64),
        torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert float(force[1] / force[0]) == pytest.approx(20.0, rel=1e-9)
    assert float(stiffness[0]) == pytest.approx(float(stiffness[1]), rel=1e-12)


# --- field construction -----------------------------------------------------
def test_field_realises_the_requested_concentration_without_overlap() -> None:
    sim = _channel()
    stats = sim.field_statistics()
    assert stats["realised_concentration"] == pytest.approx(0.6, abs=0.02)
    delta = sim.pos[:, None, :] - sim.pos[None, :, :]
    dist = torch.sqrt((delta ** 2).sum(dim=-1) + 1e-30)
    overlap = (sim.radius[:, None] + sim.radius[None, :]) - dist
    overlap = overlap - torch.diag_embed(torch.diagonal(overlap))
    assert float(overlap.max()) < 1e-3
    assert sim.n > 5


def test_null_field_has_one_unreachable_floe() -> None:
    sim = BrokenIceChannel.from_case(CASE_SPECS["no_ice"])
    assert sim.n == 1
    assert float(sim.radius[0]) < 0.1


# --- transit runner ---------------------------------------------------------
def _tiny(spec: CaseSpec, **changes) -> CaseSpec:
    base = dict(domain_length_m=120.0, domain_width_m=60.0,
                mean_floe_radius_m=4.0, time_step_s=6.0e-3, record_every=50)
    base.update(changes)
    return replace(spec, **base)


@pytest.mark.parametrize("name", ["tiny_speed_2", "tiny_speed_4"])
def test_transit_resistance_is_positive_and_increases_with_speed(name) -> None:
    speed = 2.0 if name.endswith("2") else 4.0
    spec = _tiny(SMALL, name=name, speed_m_s=speed)
    result = run_case(spec)
    assert math.isfinite(result.mean_kN) and result.mean_kN > 0.0
    # audit: hull power = drag + friction, so the friction share is a fraction
    assert -0.2 <= result.friction_power_share <= 1.2
    if name == "tiny_speed_4":
        slower = run_case(_tiny(SMALL, name="tiny_speed_2", speed_m_s=2.0))
        assert result.mean_kN > slower.mean_kN


def test_field_free_control_gives_zero_resistance() -> None:
    result = run_case(_tiny(STANDARD_CASE, name="no_ice_tiny",
                            null_field=True, concentration=0.0))
    assert result.mean_kN == pytest.approx(0.0, abs=1e-9)


def test_seeded_transit_is_reproducible() -> None:
    spec = _tiny(SMALL, name="repro", speed_m_s=4.0, record_every=200)
    first = run_case(spec)
    second = run_case(spec)
    assert first.mean_kN == pytest.approx(second.mean_kN, rel=1e-12)
    assert first.floe_count == second.floe_count


# --- anchors ----------------------------------------------------------------
def test_lindqvist_reference_structure_and_speed_trend() -> None:
    slow = lindqvist_reference(STANDARD_CASE)
    fast = lindqvist_reference(STANDARD_CASE.with_(speed_m_s=5.0))
    assert slow["total_N"] > 0.0
    assert fast["total_N"] > slow["total_N"]
    assert slow["total_N"] == pytest.approx(
        slow["crushing_N"] + slow["bending_N"] + slow["submersion_N"]
        + slow["added_drag_N"], rel=1e-12)


def test_broken_ice_closed_form_suppresses_bending_for_small_floes() -> None:
    payload = broken_ice_closed_form(STANDARD_CASE)
    assert payload["bending_suppressed"] is True      # 10 m floe < bending length
    assert payload["bending_N"] == 0.0
    assert payload["total_kN"] == pytest.approx(
        (payload["crushing_N"] + payload["submersion_N"] + payload["momentum_N"]) / 1e3)


def test_lindqvist_implementation_spans_the_published_column() -> None:
    scores = lindqvist_anchor_scores()
    assert scores["case_count"] == 10
    assert scores["mean_abs_rel_error_vs_published"] < 0.60
    for row in scores["cases"]:
        assert row["implemented_lindqvist_kN"] > 0.0


# --- criteria machinery -----------------------------------------------------
def test_pearson_and_monotone_helpers() -> None:
    assert _pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert _pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(_pearson([1.0], [1.0]))
    assert _monotone_increasing([1.0, 1.1, 1.2])
    assert not _monotone_increasing([1.0, 1.2, 1.1])


def test_criteria_flip_with_the_resistance_level() -> None:
    """C1 must pass only when the DEM sits inside the predeclared 15 % band."""
    lindqvist = lindqvist_reference(CASE_SPECS["standard"])["total_kN"]
    from tensordem.broken_ice_resistance.simulate import CaseResult

    def fake(name: str, resistance: float) -> CaseResult:
        spec = CASE_SPECS[name]
        return CaseResult(
            spec=spec.as_dict(), hull=ShipForm().as_dict(), mean_kN=resistance,
            std_kN=0.0, sem_kN=0.0, peak_kN=resistance, min_kN=resistance,
            contact_count_mean=0.0, floe_speed_mean_m_s=0.0, drag_power_kW=0.0,
            hull_power_kW=resistance * spec.speed_m_s, friction_power_share=0.5,
            kinematic_energy_change_kW=0.0, stationarity_drift=0.0,
            floe_count=1, realised_concentration=spec.concentration,
            steady_window_s=(0.0, 1.0), simulated_time_s=1.0, steps=1,
            wall_time_s=0.0)

    good = {name: fake(name, lindqvist * 0.95) for name in CASE_SPECS}
    verdict = evaluate_criteria(good, test_count=12, tests_passed=True)
    assert verdict["criteria"]["C1"]["verdict"] == "PASS"
    assert verdict["criteria"]["C4"]["verdict"] == "WITHHELD"

    bad = dict(good)
    bad["standard"] = fake("standard", lindqvist * 1.5)
    verdict_bad = evaluate_criteria(bad, test_count=12, tests_passed=True)
    assert verdict_bad["criteria"]["C1"]["verdict"] == "FAIL"


def test_criterion_c7_counts_the_suite() -> None:
    assert count_tests(__file__) >= THRESHOLDS["C7_min_tests"]
    verdict = evaluate_criteria(
        {name: CaseResult_stub(name) for name in CASE_SPECS},
        test_count=count_tests(__file__), tests_passed=True)
    assert verdict["criteria"]["C7"]["verdict"] == "PASS"


def CaseResult_stub(name: str):
    """Minimal stand-in for a case result (criteria machinery only)."""
    from tensordem.broken_ice_resistance.simulate import CaseResult
    spec = CASE_SPECS[name]
    value = 1.0 if spec.concentration > 0.0 else 0.0
    return CaseResult(
        spec=spec.as_dict(), hull={}, mean_kN=value, std_kN=0.0, sem_kN=0.0,
        peak_kN=value, min_kN=value, contact_count_mean=0.0,
        floe_speed_mean_m_s=0.0, drag_power_kW=0.0, hull_power_kW=value,
        friction_power_share=0.5, kinematic_energy_change_kW=0.0,
        stationarity_drift=0.0, floe_count=1,
        realised_concentration=spec.concentration, steady_window_s=(0.0, 1.0),
        simulated_time_s=1.0, steps=1, wall_time_s=0.0)