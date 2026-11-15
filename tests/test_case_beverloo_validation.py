"""CASE-DEM-SILO-BEVERLOO: comprehensive validation of the Beverloo discharge law.

This test suite validates the Beverloo orifice-discharge law (Beverloo, Leniger
& van de Velde, 1961) against DEM simulations.  It has two layers:

**Layer 1 — Analytic unit tests** (always runnable, no DEM needed):
  - Beverloo correlation formula correctness
  - Dimensional consistency
  - Edge cases (D ≤ kd, zero gravity, etc.)
  - Power-law fit inversion
  - C recovery from known W

**Layer 2 — Campaign result validation** (requires pre-run campaign):
  - Primary case D5: |W − W_ref| / W_ref < 10%
  - Recovered C in literature range
  - Power-law exponent near 5/2
  - Gravity scaling W ∝ √g
  - Window steadiness (CV < 0.15)

Run the campaign first::

    PYTHONPATH=src python benchmarks/m20707_beverloo/campaign.py --all
    PYTHONPATH=src python benchmarks/m20707_beverloo/analyse.py

Then run this test::

    pytest tests/test_case_beverloo_validation.py -v
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "m20707_beverloo"
REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"

# ---------------------------------------------------------------------------
# Layer 1: Analytic unit tests (no DEM needed)
# ---------------------------------------------------------------------------

class TestBeverlooCorrelation:
    """Test the Beverloo formula W = C rho_b sqrt(g) (D - kd)^(5/2)."""

    def test_known_value(self):
        """Verify the correlation against a hand-computed reference."""
        from tensordem.silo_flow.beverloo import beverloo_prediction
        # C=0.58, rho_b=1600, g=9.81, D=0.015, d=0.003, k=1.4
        # Deff = 0.015 - 1.4*0.003 = 0.015 - 0.0042 = 0.0108
        # W = 0.58 * 1600 * sqrt(9.81) * 0.0108^2.5
        Deff = 0.0108
        W_expected = 0.58 * 1600.0 * math.sqrt(9.81) * Deff ** 2.5
        W_got = beverloo_prediction(0.015, 0.003, 1600.0, c=0.58, k=1.4)
        assert W_got == pytest.approx(W_expected, rel=1e-12)

    def test_dimensional_consistency(self):
        """W must have units kg/s: [C]·[kg/m³]·[m/s²]^0.5·[m]^2.5 = kg/s."""
        from tensordem.silo_flow.beverloo import beverloo_prediction
        # Use SI base units and check the result is in a physically reasonable range
        W = beverloo_prediction(0.015, 0.003, 1600.0)
        # For a 15mm orifice with 3mm particles, W should be O(0.01-0.1) kg/s
        assert 0.001 < W < 1.0, f"W={W} is outside the physically reasonable range"

    def test_zero_effective_diameter(self):
        """When D = k·d, the effective diameter is zero and W must be zero."""
        from tensordem.silo_flow.beverloo import beverloo_prediction
        # D = 1.4 * 0.003 = 0.0042
        W = beverloo_prediction(0.0042, 0.003, 1600.0, k=1.4)
        assert W == 0.0

    def test_subcritical_diameter(self):
        """When D < k·d, the effective diameter is negative and W must be zero."""
        from tensordem.silo_flow.beverloo import beverloo_prediction
        W = beverloo_prediction(0.003, 0.003, 1600.0, k=1.4)
        assert W == 0.0

    def test_gravity_scaling(self):
        """W must scale as sqrt(g): W(g/4) / W(g) = 0.5."""
        from tensordem.silo_flow.beverloo import beverloo_prediction
        W_g = beverloo_prediction(0.015, 0.003, 1600.0, gravity=9.81)
        W_g4 = beverloo_prediction(0.015, 0.003, 1600.0, gravity=9.81 / 4.0)
        assert W_g4 / W_g == pytest.approx(0.5, rel=1e-12)

    def test_larger_orifice_gives_larger_rate(self):
        """Monotonicity: increasing D must increase W."""
        from tensordem.silo_flow.beverloo import beverloo_prediction
        W_small = beverloo_prediction(0.012, 0.003, 1600.0)
        W_large = beverloo_prediction(0.018, 0.003, 1600.0)
        assert W_large > W_small

    def test_density_scaling(self):
        """W must scale linearly with bulk density."""
        from tensordem.silo_flow.beverloo import beverloo_prediction
        W1 = beverloo_prediction(0.015, 0.003, 1600.0)
        W2 = beverloo_prediction(0.015, 0.003, 3200.0)
        assert W2 / W1 == pytest.approx(2.0, rel=1e-12)


class TestPowerLawFit:
    """Test the power-law fitting routines."""

    def test_exact_5halves_recovery(self):
        """If the data follows W = A·Deff^2.5 exactly, the fit must recover p=2.5."""
        from tensordem.silo_flow.beverloo import fit_power_law
        Deff = np.array([0.005, 0.010, 0.015])
        A = 0.58 * 1600.0 * math.sqrt(9.81)
        W = A * Deff ** 2.5
        fit = fit_power_law(Deff, W)
        assert fit["exponent"] == pytest.approx(2.5, rel=1e-6)
        assert fit["amplitude"] == pytest.approx(A, rel=1e-6)
        assert fit["r2"] == pytest.approx(1.0, abs=1e-10)

    def test_fit_with_noise(self):
        """With 5% noise, the exponent must still be within 0.3 of 2.5."""
        from tensordem.silo_flow.beverloo import fit_power_law
        rng = np.random.default_rng(42)
        Deff = np.array([0.005, 0.010, 0.015, 0.020, 0.025])
        A = 0.58 * 1600.0 * math.sqrt(9.81)
        W = A * Deff ** 2.5 * (1.0 + 0.05 * rng.standard_normal(len(Deff)))
        fit = fit_power_law(Deff, W)
        assert abs(fit["exponent"] - 2.5) < 0.3

    def test_fit_law_with_fixed_k(self):
        """fit_law with fixed k=1.4 must recover the correct exponent."""
        from tensordem.silo_flow.beverloo import fit_law
        D = np.array([0.012, 0.015, 0.018])
        d = 0.003
        rho = 1600.0
        C = 0.58
        k = 1.4
        W = np.array([C * rho * math.sqrt(9.81) * (Di - k * d) ** 2.5 for Di in D])
        fits = fit_law(D, d, W, rho)
        assert fits["fixed_k"]["exponent"] == pytest.approx(2.5, rel=1e-6)
        assert fits["fixed_k"]["c"] == pytest.approx(C, rel=1e-3)

    def test_fit_requires_at_least_two_points(self):
        from tensordem.silo_flow.beverloo import fit_power_law
        with pytest.raises(ValueError, match="at least two"):
            fit_power_law(np.array([0.01]), np.array([0.1]))


class TestCRecovery:
    """Test that C can be recovered from a known W."""

    def test_c_recovery_from_exact_w(self):
        """Given the exact Beverloo W, C_recovered must equal C."""
        from tensordem.silo_flow.beverloo import beverloo_prediction
        C, k = 0.58, 1.4
        rho_b = 1600.0
        D, d = 0.015, 0.003
        W = beverloo_prediction(D, d, rho_b, c=C, k=k)
        Deff = D - k * d
        C_rec = W / (rho_b * math.sqrt(9.81) * Deff ** 2.5)
        assert C_rec == pytest.approx(C, rel=1e-12)


class TestDischargeCaseValidation:
    """Test DischargeCase parameter validation."""

    def test_orifice_larger_than_silo_rejected(self):
        from tensordem.silo_flow.beverloo import DischargeCase
        with pytest.raises(ValueError, match="orifice"):
            DischargeCase(orifice_diameter=0.06, silo_radius=0.025)

    def test_fill_height_below_orifice_rejected(self):
        from tensordem.silo_flow.beverloo import DischargeCase
        with pytest.raises(ValueError, match="fill_height"):
            DischargeCase(orifice_diameter=0.015, fill_height=0.010)

    def test_negative_diameter_rejected(self):
        from tensordem.silo_flow.beverloo import DischargeCase
        with pytest.raises(ValueError):
            DischargeCase(orifice_diameter=-0.01)

    def test_too_few_windows_rejected(self):
        from tensordem.silo_flow.beverloo import DischargeCase
        with pytest.raises(ValueError, match="two"):
            DischargeCase(orifice_diameter=0.015, n_windows=1)


# ---------------------------------------------------------------------------
# Layer 2: Campaign result validation (requires pre-run campaign)
# ---------------------------------------------------------------------------

def _load_result(case_id: str) -> dict | None:
    p = RUN_DIR / f"{case_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_summary() -> dict | None:
    p = RUN_DIR / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


class TestReferenceIntegrity:
    """The reference JSON must exist and contain valid coefficients."""

    def test_reference_json_exists(self):
        assert REF_PATH.exists(), f"Reference JSON not found: {REF_PATH}"

    def test_reference_coefficients_in_literature_range(self):
        ref = json.loads(REF_PATH.read_text())
        C = ref["coefficients"]["C"]
        k = ref["coefficients"]["k"]
        assert 0.55 <= C <= 0.60, f"C={C} outside [0.55, 0.60]"
        assert 1.2 <= k <= 1.6, f"k={k} outside [1.2, 1.6]"


class TestPrimaryCaseD5:
    """Primary case D5: the main acceptance test."""

    @pytest.fixture
    def d5(self):
        r = _load_result("D5")
        if r is None:
            pytest.skip("D5 result not found — run campaign first")
        return r

    def test_d5_beverloo_passes_10pct(self, d5):
        """PRIMARY ACCEPTANCE: |W − W_ref| / W_ref < 10%."""
        bev = d5["beverloo"]
        assert bev["pass_10pct"], (
            f"D5 FAIL: W_dem={d5['result']['flow_rate_kg_s']:.6f} kg/s, "
            f"W_ref={bev['W_ref_kg_s']:.6f} kg/s, "
            f"rel_err={bev['rel_err_pct']:+.1f}% (tolerance <10%)"
        )

    def test_d5_recovered_C_in_literature_range(self, d5):
        """C recovered from D5 must be in the extended range [0.45, 0.75]."""
        C_rec = d5["beverloo"]["C_recovered"]
        assert 0.45 <= C_rec <= 0.75, (
            f"C_recovered={C_rec:.3f} far outside literature range"
        )

    def test_d5_particles_discharged(self, d5):
        """D5 must have discharged particles (no arching)."""
        n_exit = d5["result"]["particles_exited"]
        assert n_exit > 0, "No particles discharged (arching?)"

    def test_d5_bulk_density_reasonable(self, d5):
        """Bulk density must be in the range for random sphere packing."""
        rho_b = d5["result"]["rho_bulk"]
        assert 1200 < rho_b < 2000, (
            f"rho_bulk={rho_b:.0f} kg/m³ outside expected range [1200, 2000]"
        )

    def test_d5_window_steadiness(self, d5):
        """Coefficient of variation of window rates must be < 0.15."""
        wins = d5["result"]["flow_rate_windows_kg_s"]
        if len(wins) < 2:
            pytest.skip("Not enough windows for CV")
        cv = float(np.std(wins, ddof=1) / max(np.mean(wins), 1e-30))
        assert cv < 0.15, f"Window CV={cv:.3f} > 0.15: flow not steady"

    def test_d5_energy_balance(self, d5):
        """Energy balance residual must be small (closed ledger)."""
        energy = d5["result"].get("energy", {})
        residual = energy.get("residual", None)
        if residual is None:
            pytest.skip("Energy balance not recorded")
        assert abs(residual) < 0.05, f"Energy residual={residual:.2e} too large"


class TestSecondaryCases:
    """Secondary cases: D4, D6 for power-law check."""

    @pytest.fixture
    def d4(self):
        r = _load_result("D4")
        if r is None:
            pytest.skip("D4 result not found")
        return r

    @pytest.fixture
    def d6(self):
        r = _load_result("D6")
        if r is None:
            pytest.skip("D6 result not found")
        return r

    def test_d6_beverloo_passes_10pct(self, d6):
        """D6 should also satisfy <10%."""
        bev = d6["beverloo"]
        assert bev["pass_10pct"], (
            f"D6 FAIL: rel_err={bev['rel_err_pct']:+.1f}%"
        )

    def test_d4_beverloo_passes_15pct(self, d4):
        """D4 (D/d=4, near the lower validity limit) should pass at 15%."""
        bev = d4["beverloo"]
        assert bev["pass_15pct"], (
            f"D4 FAIL: rel_err={bev['rel_err_pct']:+.1f}% (tolerance <15%)"
        )


class TestPowerLawExponent:
    """Power-law exponent from D4/D5/D6 must be near 5/2."""

    def test_exponent_near_5halves(self):
        summary = _load_summary()
        if summary is None:
            pytest.skip("Summary not found — run analyse.py first")
        exp_check = summary.get("exponent_check")
        if exp_check is None:
            pytest.skip("Exponent check not in summary")
        exp = exp_check["exponent_fixed_k"]
        assert abs(exp - 2.5) <= 0.3, (
            f"Exponent={exp:.3f}, target=2.5, deviation={abs(exp-2.5):.3f} > 0.3"
        )

    def test_r2_above_threshold(self):
        summary = _load_summary()
        if summary is None:
            pytest.skip("Summary not found")
        exp_check = summary.get("exponent_check")
        if exp_check is None:
            pytest.skip("Exponent check not in summary")
        r2 = exp_check["r2_fixed_k"]
        assert r2 >= 0.90, f"R²={r2:.3f} < 0.90"


class TestGravityScaling:
    """W(g/4) / W(g) must be close to 0.5."""

    def test_gravity_scaling_ratio(self):
        d5 = _load_result("D5")
        d5g4 = _load_result("D5_g4")
        if d5 is None or d5g4 is None:
            pytest.skip("D5 and/or D5_g4 results not found")
        W_g = d5["result"]["flow_rate_kg_s"]
        W_g4 = d5g4["result"]["flow_rate_kg_s"]
        ratio = W_g4 / W_g if W_g > 0 else float("nan")
        assert abs(ratio - 0.5) < 0.10, (
            f"Gravity scaling FAIL: W(g/4)/W(g)={ratio:.3f}, expected 0.500"
        )


class TestSummaryVerdict:
    """The summary.json must contain a clear verdict."""

    def test_summary_exists(self):
        summary = _load_summary()
        if summary is None:
            pytest.skip("summary.json not found — run analyse.py first")
        assert "verdict" in summary or "cases" in summary

    def test_verdict_is_pass(self):
        summary = _load_summary()
        if summary is None:
            pytest.skip("summary.json not found")
        verdict = summary.get("verdict", {})
        if not verdict:
            pytest.skip("No verdict in summary")
        assert verdict.get("pass_10pct", False), (
            f"Verdict: {verdict.get('verdict', 'UNKNOWN')}, "
            f"rel_err={verdict.get('rel_err_pct', 'N/A')}%"
        )
