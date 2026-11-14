"""Tests for the m20707 Beverloo campaign: parameter validation, formula checks, and protocol integrity.

These tests verify the campaign infrastructure without running the full DEM
(which takes minutes per case and belongs on the cluster).  The actual
Beverloo-law agreement is checked by the campaign + analysis scripts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "src")

import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from tensordem.silo_flow.beverloo import (
    BEVERLOO_C, BEVERLOO_K, DischargeCase, beverloo_prediction, fit_law, fit_power_law,
)
from tensordem.silo_flow.kernel import Material, SiloGeometry, damping_ratio


# ---------------------------------------------------------------------------
# 1. Reference coefficients
# ---------------------------------------------------------------------------
REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"


def test_reference_json_exists_and_is_valid():
    assert REF_PATH.exists(), f"Reference JSON not found: {REF_PATH}"
    ref = json.loads(REF_PATH.read_text())
    assert ref["law"].startswith("W = C")
    C = ref["coefficients"]["C"]
    k = ref["coefficients"]["k"]
    assert 0.55 <= C <= 0.60, f"C = {C} outside literature range [0.55, 0.60]"
    assert 1.2 <= k <= 1.6, f"k = {k} outside literature range [1.2, 1.6]"


def test_module_constants_match_reference():
    ref = json.loads(REF_PATH.read_text())
    assert BEVERLOO_C == pytest.approx(ref["coefficients"]["C"], rel=1e-6)
    assert BEVERLOO_K == pytest.approx(ref["coefficients"]["k"], rel=1e-6)


# ---------------------------------------------------------------------------
# 2. Beverloo prediction formula
# ---------------------------------------------------------------------------
def test_beverloo_prediction_matches_manual():
    C, k, d, rho_b, g = 0.58, 1.4, 0.003, 1400.0, 9.81
    D = 0.015  # D/d = 5
    Deff = D - k * d
    W_manual = C * rho_b * math.sqrt(g) * Deff ** 2.5
    W_func = beverloo_prediction(D, d, rho_b, c=C, k=k, gravity=g)
    assert W_func == pytest.approx(W_manual, rel=1e-12)


def test_beverloo_prediction_zero_below_critical_orifice():
    # D = k*d -> Deff = 0 -> W = 0
    assert beverloo_prediction(1.4 * 0.003, 0.003, 1400.0) == 0.0
    # D < k*d -> Deff < 0 -> W = 0
    assert beverloo_prediction(0.003, 0.003, 1400.0) == 0.0


def test_beverloo_prediction_scales_with_sqrt_gravity():
    D, d, rho_b = 0.015, 0.003, 1400.0
    W1 = beverloo_prediction(D, d, rho_b, gravity=9.81)
    W4 = beverloo_prediction(D, d, rho_b, gravity=9.81 / 4.0)
    assert W4 / W1 == pytest.approx(0.5, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Power-law fit
# ---------------------------------------------------------------------------
def test_fit_recovers_exact_beverloo_exponent():
    C, k, d, rho_b = 0.58, 1.4, 0.003, 1400.0
    D = np.array([0.012, 0.015, 0.018])
    W = np.array([beverloo_prediction(Di, d, rho_b, c=C, k=k) for Di in D])
    fits = fit_law(D, d, W, rho_b)
    fk = fits["fixed_k"]
    assert fk["exponent"] == pytest.approx(2.5, abs=0.01)
    assert fk["c"] == pytest.approx(C, rel=0.01)
    assert fk["r2"] > 0.999


def test_fit_flags_wrong_exponent():
    d, rho_b = 0.003, 1400.0
    D = np.array([0.012, 0.015, 0.018])
    # Artificial data with exponent 2.0 instead of 2.5
    W = 0.58 * rho_b * math.sqrt(9.81) * (D - 1.4 * d) ** 2.0
    fits = fit_law(D, d, W, rho_b)
    fk = fits["fixed_k"]
    assert abs(fk["exponent"] - 2.5) > 0.2  # should flag as wrong


# ---------------------------------------------------------------------------
# 4. Campaign case parameters
# ---------------------------------------------------------------------------
def test_campaign_cases_satisfy_beverloo_validity():
    """All D/d ratios must be >= 4 and silo_D/D >= 2.5."""
    from benchmarks.m20707_beverloo.campaign import BASE, CASES

    for cid, overrides in CASES.items():
        params = dict(BASE)
        params.update(overrides)
        d = params["particle_diameter"]
        D = params["orifice_diameter"]
        R = params["silo_radius"]
        D_over_d = D / d
        silo_D_over_D = 2 * R / D
        assert D_over_d >= 4.0, f"{cid}: D/d = {D_over_d:.1f} < 4"
        assert silo_D_over_D >= 2.5, f"{cid}: silo_D/D = {silo_D_over_D:.1f} < 2.5"


def test_primary_case_is_D5():
    from benchmarks.m20707_beverloo.campaign import CASES
    assert "D5" in CASES
    assert CASES["D5"]["orifice_diameter"] / 0.003 == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 5. DischargeCase validation
# ---------------------------------------------------------------------------
def test_discharge_case_rejects_orifice_larger_than_silo():
    with pytest.raises(ValueError):
        DischargeCase(orifice_diameter=0.060, particle_diameter=0.003,
                      silo_radius=0.025, fill_height=0.060)


def test_discharge_case_rejects_zero_orifice():
    with pytest.raises(ValueError):
        DischargeCase(orifice_diameter=0.0, particle_diameter=0.003,
                      silo_radius=0.025, fill_height=0.060)


def test_discharge_case_rejects_too_few_windows():
    with pytest.raises(ValueError):
        DischargeCase(orifice_diameter=0.015, particle_diameter=0.003,
                      silo_radius=0.025, fill_height=0.060, n_windows=1)


# ---------------------------------------------------------------------------
# 6. Material / geometry consistency
# ---------------------------------------------------------------------------
def test_material_damping_ratio_is_physical():
    # e = 0.3 -> zeta ≈ 0.22
    zeta = damping_ratio(0.3)
    assert 0.1 < zeta < 0.4


def test_material_dt_is_stable():
    mat = Material(radius=0.0015, kn=5e3, mu=0.5)
    # dt must be < 0.25 * contact period
    T = 2 * math.pi * math.sqrt(mat.mass_eff / mat.kn)
    assert mat.dt < 0.25 * T


def test_silo_geometry_volume_is_positive():
    geom = SiloGeometry(cylinder_radius=0.025, cylinder_height=0.075,
                        outlet_radius=0.0075)
    assert geom.volume_below(0.06) > 0.0
    assert geom.volume_below(0.0) == 0.0
