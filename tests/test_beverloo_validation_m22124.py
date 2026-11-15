"""CASE-DEM-SILO-BEVERLOO validation test — Beverloo law <10% criterion.

This test validates the primary case D5 (D/d = 5) of the silo_flow campaign
against the Beverloo correlation with the <10% tolerance required by the
feature ledger acceptance criteria.

The campaign uses the ``tensordem.silo_flow`` kernel (Cundall–Strack wall
friction, disordered packing, energy ledger) which has been shown to produce
physically correct discharge rates, unlike the ``silo_discharge`` kernel
which has a systematic +20% over-prediction due to wall proximity effects
(see m21274 falsified-hypothesis report).

Run the campaign first::

    PYTHONPATH=src python benchmarks/m20707_beverloo/campaign.py --case D5
    PYTHONPATH=src python benchmarks/m20707_beverloo/campaign.py --case D6
    PYTHONPATH=src python benchmarks/m20707_beverloo/campaign.py --case D5_g4

Then this test reads the results from ``runs/m20707_beverloo/`` and asserts
the <10% criterion.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "m20707_beverloo"
REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"


def _load_result(case_id: str) -> dict | None:
    """Load a campaign result JSON, returning None if not found."""
    p = RUN_DIR / f"{case_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# 1. Reference integrity
# ---------------------------------------------------------------------------
def test_reference_json_exists():
    assert REF_PATH.exists(), f"Reference JSON not found: {REF_PATH}"


def test_reference_coefficients_in_literature_range():
    ref = json.loads(REF_PATH.read_text())
    C = ref["coefficients"]["C"]
    k = ref["coefficients"]["k"]
    assert 0.55 <= C <= 0.60
    assert 1.2 <= k <= 1.6


# ---------------------------------------------------------------------------
# 2. Campaign result availability (skip if not yet run)
# ---------------------------------------------------------------------------
@pytest.fixture
def d5_result():
    r = _load_result("D5")
    if r is None:
        pytest.skip("D5 result not found — run campaign first")
    return r


@pytest.fixture
def d6_result():
    r = _load_result("D6")
    if r is None:
        pytest.skip("D6 result not found — run campaign first")
    return r


@pytest.fixture
def d5_g4_result():
    r = _load_result("D5_g4")
    if r is None:
        pytest.skip("D5_g4 result not found — run campaign first")
    return r


# ---------------------------------------------------------------------------
# 3. Primary case D5: <10% criterion (THE acceptance test)
# ---------------------------------------------------------------------------
def test_d5_beverloo_passes_10pct(d5_result):
    """Primary case D5 must satisfy |W - W_ref| / W_ref < 10%."""
    bev = d5_result["beverloo"]
    assert bev["pass_10pct"], (
        f"D5 FAIL: W_dem={d5_result['result']['flow_rate_kg_s']:.6f} kg/s, "
        f"W_ref={bev['W_ref_kg_s']:.6f} kg/s, "
        f"rel_err={bev['rel_err_pct']:+.1f}% (tolerance <10%)"
    )


def test_d5_recovered_C_in_literature_range(d5_result):
    """C recovered from D5 must be in the extended literature range [0.45, 0.75]."""
    C_rec = d5_result["beverloo"]["C_recovered"]
    assert 0.45 <= C_rec <= 0.75, (
        f"C_recovered={C_rec:.3f} far outside literature range [0.55, 0.60]"
    )


def test_d5_particles_discharged(d5_result):
    """D5 must have at least some particles discharged (not fully arching)."""
    n_exit = d5_result["result"]["particles_exited"]
    assert n_exit > 0, "No particles discharged (arching?)"


# ---------------------------------------------------------------------------
# 4. Secondary case D6: also <10%
# ---------------------------------------------------------------------------
def test_d6_beverloo_passes_10pct(d6_result):
    """D6 should also satisfy <10% (secondary check point)."""
    bev = d6_result["beverloo"]
    assert bev["pass_10pct"], (
        f"D6 FAIL: rel_err={bev['rel_err_pct']:+.1f}% (tolerance <10%)"
    )


# ---------------------------------------------------------------------------
# 5. Gravity scaling: W ∝ √g
# ---------------------------------------------------------------------------
def test_gravity_scaling(d5_result, d5_g4_result):
    """W(g/4) / W(g) must be close to 0.5 (within ±10%)."""
    W_g = d5_result["result"]["flow_rate_kg_s"]
    W_g4 = d5_g4_result["result"]["flow_rate_kg_s"]
    ratio = W_g4 / W_g if W_g > 0 else float("nan")
    assert abs(ratio - 0.5) < 0.10, (
        f"Gravity scaling FAIL: W(g/4)/W(g)={ratio:.3f}, expected 0.500"
    )


# ---------------------------------------------------------------------------
# 6. Power-law exponent check (requires D4, D5, D6)
# ---------------------------------------------------------------------------
def test_beverloo_exponent_near_5halves():
    """If 3+ orifice sizes available, the fitted exponent must be near 2.5."""
    results = {}
    for cid in ["D4", "D5", "D6"]:
        r = _load_result(cid)
        if r is not None:
            results[cid] = r
    if len(results) < 3:
        pytest.skip("Need D4, D5, D6 results for exponent fit")

    import numpy as np
    from tensordem.silo_flow.beverloo import fit_law

    D = np.array([results[cid]["case"]["orifice_diameter"] for cid in ["D4", "D5", "D6"]])
    W = np.array([results[cid]["result"]["flow_rate_kg_s"] for cid in ["D4", "D5", "D6"]])
    rho = float(np.mean([results[cid]["result"]["rho_bulk"] for cid in ["D4", "D5", "D6"]]))
    d = results["D4"]["case"]["particle_diameter"]

    fits = fit_law(D, d, W, rho)
    exp = fits["fixed_k"]["exponent"]
    assert abs(exp - 2.5) <= 0.3, (
        f"Exponent={exp:.3f}, target=2.5, deviation={abs(exp-2.5):.3f} > 0.3"
    )


# ---------------------------------------------------------------------------
# 7. Summary file generation
# ---------------------------------------------------------------------------
def test_summary_json_exists_after_analysis():
    """After running analyse.py, summary.json must exist."""
    summary_path = RUN_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("summary.json not found — run analyse.py first")
    summary = json.loads(summary_path.read_text())
    assert "verdict" in summary or "cases" in summary
