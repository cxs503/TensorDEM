"""Tests for the CASE-DEM-PLANE-SHEAR-MUIDI synthesis (Round 14).

Validates:
1. Reference law limits and monotonicity
2. Synthesis artifact is well-formed
3. Phi provenance audit is well-formed
4. Ledger verdict is honest (no silent pass)
5. Mu(I) errors are correctly computed
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
R14 = RUNS / "round14"

# Reference constants (single source of truth)
MU_S, MU_2, I_0 = 0.36, 0.84, 0.30
PHI_MAX_3D, PHI_MIN_3D = 0.64, 0.58


def ref_mu(I):
    return MU_S + (MU_2 - MU_S) / (I_0 / I + 1.0)


def ref_phi_3d(I):
    return PHI_MAX_3D - (PHI_MAX_3D - PHI_MIN_3D) / (I_0 / I + 1.0)


# ---------------------------------------------------------------------------
# Reference law tests
# ---------------------------------------------------------------------------

def test_ref_mu_limits():
    """mu(I) limits: I->0 gives mu_s, I->inf gives mu_2."""
    assert ref_mu(1e-9) == pytest.approx(MU_S, rel=1e-6)
    assert ref_mu(1e9) == pytest.approx(MU_2, rel=1e-6)


def test_ref_mu_monotone():
    """mu(I) must increase with I."""
    I_vals = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    mus = [ref_mu(I) for I in I_vals]
    assert all(b > a for a, b in zip(mus, mus[1:]))


def test_ref_phi_3d_limits():
    """phi_3d(I) limits: I->0 gives phi_max, I->inf gives phi_min."""
    assert ref_phi_3d(1e-9) == pytest.approx(PHI_MAX_3D, rel=1e-6)
    assert ref_phi_3d(1e9) == pytest.approx(PHI_MIN_3D, rel=1e-6)


def test_ref_phi_3d_decreasing():
    """phi_3d(I) must decrease with I (dilatancy)."""
    assert ref_phi_3d(1e-3) > ref_phi_3d(0.3)


# ---------------------------------------------------------------------------
# Synthesis artifact tests
# ---------------------------------------------------------------------------

def _load_r14():
    path = R14 / "result.json"
    assert path.exists(), "run synthesis_r14.py first"
    return json.loads(path.read_text())


def test_synthesis_is_well_formed():
    res = _load_r14()
    assert res["case"] == "CASE-DEM-PLANE-SHEAR-MUIDI"
    assert res["round"] == 14
    assert "mu_assessment" in res
    assert "phi_assessment" in res
    assert "ledger_verdict" in res
    assert "root_causes" in res
    assert "recommendations" in res
    assert "round_history" in res


def test_mu_assessment_has_errors():
    res = _load_r14()
    sweep = res["mu_assessment"]["sweep"]
    fc = res["mu_assessment"]["full_cell"]
    assert sweep["n_points"] == 6
    assert fc["n_points"] == 4
    assert len(sweep["errors"]) == 6
    assert len(fc["errors"]) == 4


def test_mu_errors_correctly_computed():
    """Spot-check: mu errors must match ref_mu computation."""
    res = _load_r14()
    for e in res["mu_assessment"]["sweep"]["errors"]:
        expected = 100.0 * (e["mu"] / ref_mu(e["I"]) - 1.0)
        assert e["err_pct"] == pytest.approx(expected, rel=1e-4)
    for e in res["mu_assessment"]["full_cell"]["errors"]:
        expected = 100.0 * (e["mu"] / ref_mu(e["I"]) - 1.0)
        assert e["err_pct"] == pytest.approx(expected, rel=1e-4)


def test_ledger_verdict_is_honest_fail():
    """The ledger verdict must NOT silently pass."""
    res = _load_r14()
    v = res["ledger_verdict"]
    assert "FAIL" in v["mu_I_curve"] or "HONEST" in v["mu_I_curve"]
    assert "WITHHELD" in v["phi_I_curve"] or "FAIL" in v["phi_I_curve"]
    assert "FAIL" in v["overall"]


def test_best_mu_point_within_10pct():
    """The best mu point (full_cell I=0.01) should be within 10%."""
    res = _load_r14()
    bp = res["mu_assessment"]["full_cell"]["best_point"]
    assert bp is not None
    assert abs(bp["err_pct"]) < 10.0, f"best point err={bp['err_pct']:.1f}%"


def test_worst_mu_error_above_10pct():
    """The worst mu error must be >10% (honest FAIL)."""
    res = _load_r14()
    worst_sweep = res["mu_assessment"]["sweep"]["worst_err_pct"]
    worst_fc = res["mu_assessment"]["full_cell"]["worst_err_pct"]
    assert worst_sweep > 10.0, f"sweep worst err={worst_sweep:.1f}%"
    assert worst_fc > 10.0, f"full_cell worst err={worst_fc:.1f}%"


# ---------------------------------------------------------------------------
# Phi provenance audit tests
# ---------------------------------------------------------------------------

def _load_phi_audit():
    path = RUNS / "phi_provenance_audit.json"
    assert path.exists(), "run phi_provenance_audit.py first"
    return json.loads(path.read_text())


def test_phi_audit_is_well_formed():
    res = _load_phi_audit()
    assert res["case"] == "CASE-DEM-PLANE-SHEAR-MUIDI"
    assert "points" in res
    assert "dilatancy" in res
    assert "verdict" in res


def test_phi_audit_has_both_sources():
    res = _load_phi_audit()
    sources = set(p["source"] for p in res["points"])
    assert "sweep" in sources
    assert "full_cell" in sources


def test_phi_audit_dilatancy_verdict():
    """Both sweep and full_cell should FAIL dilatancy (known defects)."""
    res = _load_phi_audit()
    assert res["dilatancy"]["sweep"]["ok"] is False
    assert res["dilatancy"]["full_cell"]["ok"] is False


def test_phi_audit_provenance_unverified():
    """The 3D phi magnitudes must be marked as PROVENANCE-UNVERIFIED."""
    res = _load_phi_audit()
    assert "PROVENANCE-UNVERIFIED" in res["verdict"]["phi_3d_magnitudes"]


# ---------------------------------------------------------------------------
# Round history completeness
# ---------------------------------------------------------------------------

def test_round_history_covers_r1_to_r14():
    res = _load_r14()
    history = res["round_history"]
    for i in range(1, 15):
        assert f"R{i}" in history, f"R{i} missing from round history"


# ---------------------------------------------------------------------------
# Recommendations are actionable
# ---------------------------------------------------------------------------

def test_recommendations_non_empty():
    res = _load_r14()
    assert len(res["recommendations"]) >= 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
