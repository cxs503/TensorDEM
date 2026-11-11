"""Tests for the vertical-section ice rubble pile-up DEM (campaign p227w2-N04).

Physics-level unit tests run always; the campaign-criterion tests read the
committed run artifact ``runs/m14600/ice_rubble_pileup/report.json`` and
recompute the criterion from the recorded numbers (no duplicated literals).
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensordem.ice_rubble_pileup_vertical import (
    FLOATING_CENTRE_OFFSET, PileupVerticalConfig, buoyant_vertical_force,
    cap_binds, fit_porosity_scaling, submerged_area,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runs" / "m14600" / "ice_rubble_pileup" / "report.json"
QUICK = dict(h=0.6, D=3.0, sheet_length_factor=5.0, platen_travel_factor=6.0,
             r_median_factor=0.16, r_bounds_factor=(0.10, 0.22), k_lin_factor=10.0)


def test_submerged_area_matches_numeric_integration():
    """Analytic submerged area vs a direct horizontal-chord quadrature."""
    for r in (0.1, 0.3, 0.5):
        for fc in (-1.0, -0.641, 0.0, 0.5, 1.0):
            z_c = fc * r
            analytic = float(submerged_area(r, z_c))
            x = np.linspace(-r, r, 20001)
            half = np.sqrt(np.maximum(r * r - x * x, 0.0))
            top = z_c + half
            bottom = z_c - half
            sub_depth = np.clip(np.minimum(top, 0.0) - bottom, 0.0, None)
            numeric = float(np.trapezoid(sub_depth, x))
            assert abs(analytic - numeric) < 2e-3 * math.pi * r * r


def test_buoyancy_gives_the_expected_draught_and_restoring_force():
    """Free-floating disc: equilibrium at A_sub/A = rho_ice/rho_water."""
    r, rho_ratio = 0.3, 900.0 / 1025.0
    z_eq = FLOATING_CENTRE_OFFSET * r
    f_eq = float(buoyant_vertical_force(r, z_eq))
    assert abs(f_eq) < 5e-3 * 900.0 * 9.81 * math.pi * r * r     # ~force balance
    assert float(submerged_area(r, z_eq)) / (math.pi * r * r) == pytest.approx(
        rho_ratio, abs=2e-3)
    # restoring: deeper -> stronger upward force, higher -> net downward force
    assert float(buoyant_vertical_force(r, -0.9 * r)) > float(buoyant_vertical_force(r, z_eq))
    assert float(buoyant_vertical_force(r, 0.0)) < 0.0     # half submerged is not enough
    assert float(buoyant_vertical_force(r, 0.9 * r)) < 0.0


def test_crushing_cap_must_bind_inside_the_crushable_zone():
    assert cap_binds(PileupVerticalConfig(k_lin_factor=10.0))
    assert not cap_binds(PileupVerticalConfig(k_lin_factor=3.0))


def test_fit_recovers_a_synthetic_exponent():
    n = np.linspace(0.55, 0.85, 9)
    k = 0.7
    y = k * n ** 1.6
    out = fit_porosity_scaling(n, y)
    assert out["alpha"] == pytest.approx(1.6, abs=1e-6)
    assert out["K"] == pytest.approx(k, rel=1e-6)
    assert out["R2"] > 0.999


def test_run_produces_a_positive_load_and_a_finite_pile():
    from tensordem.ice_rubble_pileup_vertical import run_vertical_pileup_case
    res = run_vertical_pileup_case(PileupVerticalConfig(**QUICK))
    assert res.F_max > 0.0
    assert np.all(np.isfinite(res.load))
    assert res.porosity == pytest.approx(res.porosity)  # not NaN
    assert 0.0 < res.porosity < 1.0
    assert res.pile_height > 0.5 * QUICK["h"]
    assert res.ratio_max > 0.0


@pytest.mark.skipif(not REPORT.exists(), reason="campaign artifact not built")
def test_campaign_iso_check_matches_recorded_artifact():
    rep = json.loads(REPORT.read_text())
    ck = rep["analysis"]["checks"]["C2_iso_order_of_magnitude"]
    base = ck["base_ratio_max"]
    assert ck["pass"] is (abs(base - 1.0) < 0.5)
    assert len(ck["all_base"]) == 4
    for row in ck["all_base"]:
        assert 0.1 < row["ratio_max"] < 5.0


@pytest.mark.skipif(not REPORT.exists(), reason="campaign artifact not built")
def test_campaign_porosity_and_monotonic_criteria_match_artifact():
    rep = json.loads(REPORT.read_text())
    ck = rep["analysis"]["checks"]
    fit = rep["analysis"]["porosity_fit"]
    assert ck["C3_porosity_scaling"]["pass"] is bool(
        fit["R2"] > 0.8 and 1.0 <= fit["alpha"] <= 2.0)
    assert ck["C5_monotonic_thickness"]["pass"] is all(
        ck["C5_monotonic_thickness"]["F_max"][i] < ck["C5_monotonic_thickness"]["F_max"][i + 1]
        for i in range(len(ck["C5_monotonic_thickness"]["F_max"]) - 1))
    assert ck["C6_monotonic_strength"]["pass"] is all(
        ck["C6_monotonic_strength"]["F_max"][i] < ck["C6_monotonic_strength"]["F_max"][i + 1]
        for i in range(len(ck["C6_monotonic_strength"]["F_max"]) - 1))
