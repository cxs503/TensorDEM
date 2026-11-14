"""m20934: kn-sensitivity sweep for CASE-DEM-SILO-BEVERLOO.

Fast CI tests for the sweep harness (``benchmarks/m20934_beverloo_kn``):
the machine-readable Beverloo reference, the predeclared arm matrix (pinned
to the m20055 parameters so kn is the only variable), the verdict logic, and
a short hard-stiffness (kn=1e6) smoke run that must stay finite and drain.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "m20934_beverloo_kn"))

from tensordem.silo_discharge import (  # noqa: E402
    SiloCase, _initial_packing, beverloo_prediction, simulate_silo,
)

import run_kn_sweep as sweep  # noqa: E402

REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"


def test_ref_file_is_machine_readable_and_in_literature_range():
    ref = json.loads(REF_PATH.read_text())
    C = float(ref["coefficients"]["C"])
    k = float(ref["coefficients"]["k"])
    lo_C, hi_C = ref["coefficients"]["C_range_spherical"]
    lo_k, hi_k = ref["coefficients"]["k_range_spherical"]
    assert lo_C <= C <= hi_C
    assert lo_k <= k <= hi_k
    assert ref["case"] == "CASE-DEM-SILO-BEVERLOO"


def test_beverloo_prediction_matches_hand_value():
    # W = 0.58 * 1500 * sqrt(9.81) * (0.10 - 1.4*0.02)**2.5
    W = beverloo_prediction(0.58, 1.4, 0.10, 0.02, 1500.0, 9.81)
    expected = 0.58 * 1500.0 * math.sqrt(9.81) * (0.072) ** 2.5
    assert W == pytest.approx(expected, rel=1e-12)
    # critical orifice: D - k d <= 0 -> no flow
    assert beverloo_prediction(0.58, 1.4, 0.027, 0.02, 1500.0, 9.81) == 0.0


def test_arm_matrix_is_pinned_to_m20055_parameters():
    # 3 orifices x 3 stiffness levels + 1 dt/2 guard arm
    assert len(sweep.ARMS) == 10
    tags = [a[0] for a in sweep.ARMS]
    assert len(set(tags)) == 10
    assert "D5_kn1e6_dthalf" in tags
    # BASE must be the m20055 parameter set (only kn/dt vary across arms)
    b = sweep.BASE
    assert b["seed"] == 20055
    assert b["mu"] == 0.30
    assert b["mu_rolling"] == 0.02
    assert b["restitution"] == 0.30
    assert b["silo_radius"] == 0.130
    assert b["fill_height"] == 0.40
    assert b["wall_mu"] == 0.0
    assert b["packing"] == "settled"
    assert b["plug_during_settle"] is True
    assert b["settle_time"] == 0.60
    kns = {a[3] for a in sweep.ARMS}
    assert kns == {1.0e5, 3.0e5, 1.0e6}


def test_build_arm_sets_kn_and_dt():
    case = sweep.build_arm("D5_kn1e6", 0.100, 3.0, 1.0e6, 1.0)
    assert case.kn == 1.0e6
    assert case.dt is None
    assert case.orifice_diameter == 0.100
    case_half = sweep.build_arm("D5_kn1e6_dthalf", 0.100, 3.0, 1.0e6, 0.5)
    assert case_half.dt == pytest.approx(0.5 * case.timestep)
    # same seed/geometry -> identical initial packing
    p1 = _initial_packing(case)
    p2 = _initial_packing(case_half)
    assert np.allclose(p1, p2)


def test_verdict_logic():
    ref = json.loads(REF_PATH.read_text())
    C, k = float(ref["coefficients"]["C"]), float(ref["coefficients"]["k"])
    D, d, rho_b, g = 0.100, 0.020, 1500.0, 9.81
    W_ref = C * rho_b * math.sqrt(g) * (D - k * d) ** 2.5

    class _FakeCase:
        diameter = d
        orifice_diameter = D
        gravity = g
        kn = 1.0e6
        dt = 1.0e-5

    def fake_res(W_got):
        return {
            "bulk_density_bed_kg_m3": rho_b,
            "steady_flow_rate_kg_s": W_got,
            "steady_interval_s": [0.0, 1.0],
            "steady_rate_windows_kg_s": [W_got] * 5,
            "steady_rate_cv": 0.01,
            "discharge_time_s": [0.0, 1.0],
            "discharge_cumulative_n": [0, 100],
            "n_initial": 1000,
            "n_bed": 900,
            "bed_height_m": 0.3,
            "packing_fraction_bed": 0.6,
            "n_discharged_total": 100,
            "dt": 1.0e-5,
            "steps": 1000,
        }

    out = sweep.analyse("D5_kn1e6", _FakeCase(), fake_res(W_ref), ref, 1.0)
    assert out["rel_err"] == pytest.approx(0.0, abs=1e-12)
    assert out["pass_10pct"] and out["pass_15pct"]

    out = sweep.analyse("D5_kn1e6", _FakeCase(), fake_res(1.12 * W_ref), ref, 1.0)
    assert not out["pass_10pct"]
    assert out["pass_15pct"]

    out = sweep.analyse("D5_kn1e6", _FakeCase(), fake_res(1.25 * W_ref), ref, 1.0)
    assert not out["pass_10pct"] and not out["pass_15pct"]


def test_hard_kn_smoke_run_stays_finite_and_drains():
    """kn=1e6 (10x stiffer than the m20055 FAIL arm) must not diverge on a
    small settled bed and must actually discharge through the orifice."""
    case = SiloCase(
        radius=0.005, density=2500.0, mu=0.30, mu_rolling=0.02,
        restitution=0.30, kn=1.0e6, kt_ratio=0.5, gravity=9.81,
        silo_radius=0.06, fill_height=0.04, orifice_diameter=0.06,
        plate_thickness=0.005, wall_mu=0.0,
        packing="settled", plug_during_settle=True, settle_loose_gap=0.30,
        packing_gap=0.0, jitter=0.05, seed=20934,
        settle_time=0.01, discharge_time=0.06,
        window=0.012, n_windows=5,
    )
    res = simulate_silo(case, return_history=True)
    assert res["n_discharged_total"] > 0
    assert res["n_discharged_total"] < res["n_initial"]
    assert np.isfinite(res["steady_flow_rate_kg_s"])
    assert res["steady_flow_rate_kg_s"] > 0.0
    # hard contacts must not blow up: every live particle stays finite and
    # inside the silo
    alive = res["alive"]
    assert np.all(np.isfinite(res["positions"]))
    assert np.all(np.hypot(res["positions"][alive, 0],
                           res["positions"][alive, 1]) <= case.silo_radius + 1e-9)
