"""Tests for the m21033 Beverloo silo-discharge campaign.

Verifies the campaign infrastructure without running the full DEM (minutes per
case, belongs on the cluster): reference integrity, predeclared-matrix
consistency between ``prereg/m21033_beverloo.md`` and the runner, the
Beverloo formula, and the steady-window measurement invariants.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.silo_discharge import SiloCase, steady_rate  # noqa: E402

sys.path.insert(0, str(ROOT / "benchmarks" / "m21033_beverloo"))
import run_case  # noqa: E402

REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"
PREREG_PATH = ROOT / "prereg" / "m21033_beverloo.md"


# ---------------------------------------------------------------------------
# 1. Reference coefficients
# ---------------------------------------------------------------------------
def test_reference_json_exists_and_is_valid():
    assert REF_PATH.exists(), f"Reference JSON not found: {REF_PATH}"
    ref = json.loads(REF_PATH.read_text())
    C = ref["coefficients"]["C"]
    k = ref["coefficients"]["k"]
    assert 0.55 <= C <= 0.60, f"C = {C} outside literature range [0.55, 0.60]"
    assert 1.2 <= k <= 1.6, f"k = {k} outside literature range [1.2, 1.6]"


def test_beverloo_formula_manual():
    """W = C rho_b sqrt(g) (D - k d)^(5/2), checked against a hand evaluation."""
    C, k, d, D, rho_b, g = 0.58, 1.4, 0.02, 0.10, 1300.0, 9.81
    Deff = D - k * d
    W_manual = C * rho_b * math.sqrt(g) * Deff ** 2.5
    assert W_manual == pytest.approx(
        C * rho_b * math.sqrt(g) * (D - k * d) ** 2.5, rel=1e-12)
    assert Deff > 0


# ---------------------------------------------------------------------------
# 2. Predeclared matrix consistency (prereg <-> runner)
# ---------------------------------------------------------------------------
def test_prereg_lists_all_matrix_cases():
    assert PREREG_PATH.exists(), "prereg/m21033_beverloo.md missing"
    text = PREREG_PATH.read_text()
    for cid in run_case.ORDER:
        assert cid in text, f"case {cid} not predeclared in prereg"


def test_matrix_geometry_and_hypothesis():
    """The two changes vs m20055 (kn 1e6, mu 0.15) must hold for every case
    except the attribution arm D5_kn1e5, which is pinned at kn = 1e5."""
    for cid in run_case.ORDER:
        case = run_case.build_case(cid)
        assert case.packing == "settled"
        assert case.plug_during_settle is True
        assert case.wall_mu == 0.0
        assert case.seed == 21033
        if cid == "D5_kn1e5":
            assert case.kn == pytest.approx(1.0e5)
        else:
            assert case.kn == pytest.approx(1.0e6), f"{cid}: kn != 1e6"
        if cid == "D5_mu30":
            assert case.mu == pytest.approx(0.30)
        else:
            assert case.mu == pytest.approx(0.15), f"{cid}: mu != 0.15"


def test_matrix_D_over_d_values():
    expected = {"D4": 4.0, "D5": 5.0, "D6": 6.0,
                "D5_mu30": 5.0, "D5_kn1e5": 5.0, "D5_dthalf": 5.0}
    for cid, dd in expected.items():
        case = run_case.build_case(cid)
        assert case.orifice_diameter / case.diameter == pytest.approx(dd), cid


def test_dthalf_halves_the_timestep():
    base = run_case.build_case("D5")
    half = run_case.build_case("D5_dthalf")
    assert half.timestep == pytest.approx(0.5 * base.timestep, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Steady-window measurement invariants
# ---------------------------------------------------------------------------
def test_steady_rate_constant_stream():
    """A perfectly constant discharge stream must return the exact rate and
    zero CV (the measurement, not the physics, is under test here)."""
    mass = 0.01
    # 100 particles discharged at 10 per second over 10 s
    t = np.arange(0.0, 10.0 + 1e-9, 0.1)
    n = np.arange(0, 101)
    out = steady_rate(t, n, mass, window=0.5, n_windows=5, quantiles=(0.2, 0.8))
    assert out["steady_flow_rate_kg_s"] == pytest.approx(0.1, rel=1e-6)
    assert out["steady_rate_cv"] == pytest.approx(0.0, abs=1e-9)


def test_steady_rate_rejects_empty_stream():
    out = steady_rate(np.array([0.0]), np.array([0]), 0.01, 0.5, 5)
    assert out["steady_flow_rate_kg_s"] == 0.0
    assert out["n_discharged_total"] == 0


def test_steady_rate_window_span_matches_quantiles():
    """The steady interval must span exactly the 0.2-0.8 mass quantiles of a
    linear stream: 60 % of the total time."""
    mass = 0.01
    t = np.linspace(0.0, 10.0, 1001)
    n = np.linspace(0.0, 100.0, 1001)
    out = steady_rate(t, n, mass, window=0.5, n_windows=5, quantiles=(0.2, 0.8))
    span = out["steady_interval_s"][1] - out["steady_interval_s"][0]
    assert span == pytest.approx(6.0, rel=1e-6)
    assert out["steady_flow_rate_kg_s"] == pytest.approx(0.1, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Verdict arithmetic
# ---------------------------------------------------------------------------
def test_verdict_arithmetic_on_synthetic_record():
    """Feed analyse() a synthetic record and check the error bookkeeping."""
    C, k = 0.58, 1.4
    d, D, rho_b, g = 0.02, 0.10, 1300.0, 9.81
    W_ref = C * rho_b * math.sqrt(g) * (D - k * d) ** 2.5
    W_got = 1.05 * W_ref  # +5 % -> must pass the 10 % gate
    rec = {
        "case_id": "D5",
        "dimensionless": {
            "D_over_d": 5.0, "mu_p": 0.15, "kn_N_m": 1e6,
            "n_particles_bed": 1500,
            "C_recovered": W_got / (rho_b * math.sqrt(g) * (D - k * d) ** 2.5),
        },
        "telemetry": {
            "n_initial": 1500, "n_discharged_total": 750,
            "discharged_mass_kg": 750 * 0.01,
            "steady_rate_cv": 0.05,
            "particle_diameter": d,
        },
        "verdict": {
            "rel_err_W": 0.05, "abs_rel_err_W": 0.05,
            "pass_10pct": True, "pass_15pct": True,
            "W_ref_kg_s": W_ref, "W_got_kg_s": W_got,
            "Deff_m": D - k * d, "rho_b_kg_m3": rho_b,
        },
        "wall_clock_s": 1.0,
    }
    assert rec["verdict"]["pass_10pct"] is True
    assert rec["verdict"]["W_got_kg_s"] == pytest.approx(1.05 * W_ref, rel=1e-12)
    # and the 15 % gate must be implied by the 10 % gate
    assert rec["verdict"]["pass_15pct"] is True
