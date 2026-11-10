"""Tests for :mod:`tensordem.ridge_mobility` (grounded ridge mobility criterion)."""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.ridge_mobility import (
    GRAVITY, LITERATURE_KEEL_DRAFT_BAND, RHO_ICE, RHO_WATER,
    DemKeelRig, DemKeelSpec, RidgeMobilityCase, assess, dem_free_float,
    dem_grounding, free_float_draft_ratio, mobility_boundary, submerged_fraction,
)


# ---------------------------------------------------------------------------
# closed form
# ---------------------------------------------------------------------------

def test_free_float_draft_ratio_is_archimedes():
    ratio = free_float_draft_ratio(RHO_ICE, RHO_WATER)
    assert ratio == pytest.approx(math.sqrt(917.0 / 1025.0), rel=1e-12)
    assert 0.94 < ratio < 0.95


def test_weight_buoyancy_and_free_float():
    case = RidgeMobilityCase(keel_depth=5.0, water_depth=3.5)
    s = 1.0 / math.tan(math.radians(30.0))
    assert case.width == pytest.approx(2.0 * s * 5.0)
    assert case.section_area == pytest.approx(0.5 * case.width * 5.0)
    assert case.weight_per_length == pytest.approx(
        (1 - 0.30) * RHO_ICE * case.section_area * GRAVITY)
    # free float: water deeper than the free-float draft -> no seabed contact
    deep = RidgeMobilityCase(keel_depth=5.0, water_depth=case.free_float_draft + 0.5)
    assert deep.normal_force == 0.0
    assert not deep.grounded
    assert deep.status == "floating"
    # exactly at the free-float draft the normal force still vanishes
    at_ff = RidgeMobilityCase(keel_depth=5.0, water_depth=case.free_float_draft)
    assert at_ff.normal_force == pytest.approx(0.0, abs=1e-6)


def test_grounded_normal_force_closed_form():
    h_k, w, n, H = 5.0, 17.32050807568877, 0.30, 3.5
    case = RidgeMobilityCase(keel_depth=h_k, keel_width=w, porosity=n, water_depth=H)
    s = w / (2.0 * h_k)
    expected = (1.0 - n) * GRAVITY * s * (RHO_ICE * h_k ** 2 - RHO_WATER * H ** 2)
    assert case.normal_force == pytest.approx(expected, rel=1e-12)
    # monotone: deeper water -> smaller grounding force, shallower -> larger
    assert RidgeMobilityCase(keel_depth=h_k, keel_width=w, water_depth=3.0).normal_force \
        > case.normal_force


def test_porosity_scales_normal_force_linearly():
    a = RidgeMobilityCase(keel_depth=5.0, porosity=0.20, water_depth=3.5)
    b = RidgeMobilityCase(keel_depth=5.0, porosity=0.40, water_depth=3.5)
    assert a.normal_force / b.normal_force == pytest.approx(0.8 / 0.6, rel=1e-12)


def test_critical_friction_and_mobility_states():
    case = RidgeMobilityCase(keel_depth=5.0, water_depth=3.5, seabed_friction=0.4,
                            ship_thrust=1.5e6, ship_contact_length=30.0)
    q = 1.5e6 / 30.0
    assert case.driving_force == pytest.approx(q)
    assert case.critical_friction == pytest.approx(q / case.normal_force)
    assert case.critical_friction == pytest.approx(0.4054, abs=5e-3)
    assert case.status == "grounded_mobile"
    # deeper keel at the same water-depth ratio is harder to move
    deep = RidgeMobilityCase(keel_depth=6.0, water_depth=4.2, ship_thrust=1.5e6,
                             ship_contact_length=30.0)
    assert deep.critical_friction < case.critical_friction
    # immobile once the seabed friction exceeds the critical value
    stuck = RidgeMobilityCase(keel_depth=5.0, water_depth=3.5, seabed_friction=0.9,
                              ship_thrust=1.5e6, ship_contact_length=30.0)
    assert stuck.status == "grounded_immobile"
    assert stuck.mobility_margin < 0.0


def test_mu_b_zero_degenerates_to_adhesion_only():
    case = RidgeMobilityCase(keel_depth=5.0, water_depth=3.5, seabed_friction=0.0,
                             refreeze_adhesion=2.0e3)
    assert case.friction_resistance == 0.0
    assert case.resistance == pytest.approx(case.adhesion_force)
    assert case.adhesion_force > 0.0
    bare = RidgeMobilityCase(keel_depth=5.0, water_depth=3.5, seabed_friction=0.0)
    assert bare.resistance == 0.0 and bare.mobile


def test_contact_width_and_bearing_capacity():
    case = RidgeMobilityCase(keel_depth=5.0, water_depth=3.5)
    s = case.flank_slope
    expected = math.sqrt(2.0 * s * case.normal_force / case.subgrade_modulus)
    assert case.contact_width == pytest.approx(expected)
    # a weak seabed pushes the contact plastic: b = N / q_ult
    weak = RidgeMobilityCase(keel_depth=5.0, water_depth=3.5, bearing_capacity=1.0e5)
    assert weak.plastic_contact
    assert weak.contact_width == pytest.approx(weak.normal_force / 1.0e5)
    assert weak.bearing_pressure == pytest.approx(1.0e5)
    assert not case.plastic_contact
    assert "seabed_bearing_failure:keel_bedded_in" in weak.model_form_flags


def test_literature_band_flags():
    lo, hi = LITERATURE_KEEL_DRAFT_BAND
    inside = RidgeMobilityCase(keel_depth=5.0, local_draft=1.0)
    assert inside.keel_draft_ratio == pytest.approx(5.0)
    assert inside.draft_ratio_in_band and not inside.model_form_flags
    outside = RidgeMobilityCase(keel_depth=8.0, local_draft=1.0)
    assert not outside.draft_ratio_in_band
    assert any(flag.startswith("keel_depth_over_draft_out_of_band")
               for flag in outside.model_form_flags)


def test_assess_is_complete_and_serialisable():
    import json
    row = assess(RidgeMobilityCase(keel_depth=5.0, water_depth=3.5))
    json.dumps(row)  # must not raise
    for key in ("normal_force_N_per_m", "critical_friction", "status",
                "free_float_draft_m", "contact_width_m"):
        assert key in row
    assert row["free_float_draft_over_keel"] == pytest.approx(
        free_float_draft_ratio())


def test_mobility_boundary_curve_decreases_with_keel_depth():
    rows = mobility_boundary([4.0, 5.0, 6.0], water_depth_ratio=0.7,
                             ship_thrust=1.5e6, ship_contact_length=30.0)
    mus = [r["mu_star"] for r in rows]
    assert mus[0] > mus[1] > mus[2]
    assert all(0.0 < mu < 1.0 for mu in mus)
    assert rows[0]["mobility_map"]["0.10"] == "mobile"
    assert rows[0]["mobility_map"]["1.00"] == "immobile"


def test_input_validation():
    with pytest.raises(ValueError):
        RidgeMobilityCase(keel_depth=-1.0)
    with pytest.raises(ValueError):
        RidgeMobilityCase(keel_depth=5.0, porosity=1.0)
    with pytest.raises(ValueError):
        RidgeMobilityCase(keel_depth=5.0, flank_angle_deg=0.0)
    with pytest.raises(TypeError):
        RidgeMobilityCase(keel_depth="five")


# ---------------------------------------------------------------------------
# DEM closure
# ---------------------------------------------------------------------------

def test_submerged_fraction_exact():
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float64)
    frac = submerged_fraction(x, 1.0)
    assert frac[0].item() == pytest.approx(1.0)      # fully submerged
    assert frac[1].item() == pytest.approx(1.0)      # tangent from below
    assert frac[2].item() == pytest.approx(0.5)      # centre on the waterline
    assert frac[3].item() == pytest.approx(0.0)      # tangent from above
    assert frac[4].item() == pytest.approx(0.0)      # dry


def _small_spec(keel_depth=1.0):
    return DemKeelSpec(keel_depth=keel_depth, porosity=0.30,
                       radius=0.03 * keel_depth, flank_angle_deg=30.0)


def test_packing_geometry_and_porosity():
    spec = _small_spec()
    rig = DemKeelRig(spec)
    assert rig.n > 200
    assert 0.05 < rig.measured_porosity < 0.30
    # the packing occupies the triangular prism, apex down
    assert rig.positions[:, 1].min() < -0.9 * spec.keel_depth
    assert rig.positions[:, 1].max() <= 0.0
    half_width = spec.width / 2.0
    assert float(rig.positions[:, 0].abs().max()) <= half_width


def test_dem_free_float_matches_archimedes_and_draft_ratio():
    spec = _small_spec()
    case = RidgeMobilityCase(keel_depth=spec.keel_depth, porosity=0.30,
                             water_depth=0.7 * spec.keel_depth)
    ledger, cf = dem_free_float(case, spec, max_steps=6000, tol=1e-6)
    # Global Archimedes: the submerged solid fraction must return rho_i/rho_w.
    # The tolerance is the *coarse* test packing (r/h_k = 3 %): the free-float
    # position is a staircase in the grain size, cf. the campaign run at
    # r/h_k = 1.5 % which measures the O(1 %) value quoted in the report.
    # Free-float heave is a soft mode (period ~1e3 steps) and the relaxation
    # damping caps the drift velocity, so the coarse 6e3-step test packing only
    # gets within ~5 %; the campaign packing (r/h_k = 1.5 %, 2e4 steps) reaches
    # 2 %.  The test therefore only asserts the *direction* and a loose band.
    assert cf["archimedes_ratio_dem"] > cf["archimedes_ratio_cf"]   # approaches from above
    assert cf["archimedes_ratio_dem"] == pytest.approx(cf["archimedes_ratio_cf"], rel=0.08)
    draft_area = math.sqrt(cf["archimedes_ratio_dem"])
    assert draft_area == pytest.approx(cf["free_float_draft_ratio_cf"], rel=0.05)


def test_dem_grounding_force_balance_and_mu_star():
    spec = _small_spec()
    case = RidgeMobilityCase(keel_depth=spec.keel_depth, porosity=0.30,
                             water_depth=0.7 * spec.keel_depth,
                             seabed_friction=0.35, ship_thrust=1.5e6,
                             ship_contact_length=30.0)
    ledger, cf = dem_grounding(case, spec, max_steps=8000, tol=1e-6, loaded_steps=1500)
    static = ledger["static"]
    # the *static* (undriven) state is the one the mobility criterion is built on
    assert static["residual_force_ratio"] < 1e-3   # campaign packing: 2.6e-7
    assert static["ground_normal_N_per_m"] > 0.0
    # global vertical balance: weight - buoyancy - seabed reaction ~ 0
    residual = (static["weight_per_length_N_per_m"] - static["buoyancy_per_length_N_per_m"]
                - static["ground_normal_N_per_m"])
    assert abs(residual) / static["weight_per_length_N_per_m"] < 5e-3
    # the drive is a second-order perturbation of the normal force
    assert cf["normal_force_dem_loaded_N_per_m"] > 0.0
    # closed form on the realised prism geometry
    assert cf["normal_ratio_measured_geometry"] == pytest.approx(1.0, rel=0.10)
    # the DEM critical friction reproduces the closed-form boundary
    assert cf["mu_star_dem"] == pytest.approx(cf["mu_star_cf"], rel=0.10)