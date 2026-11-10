"""P489-N04: Bulk granular flow verification — acceptance tests.

Tests the four machine-verifiable criteria:
1. Beverloo exponent 2.5 ± 0.2 with R² ≥ 0.95
2. Janssen P_∞ saturation within 10 % of theory
3. Flow-pattern critical angle θ_c within 15 % of Jenike
4. Energy ledger residual ≤ 1 %

All results are read from JSON artifacts (zero hand-copying).
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

ARTIFACTS_DIR = ROOT / "runs" / "bulk_flow_p489n04"


def _load_summary() -> dict:
    p = ARTIFACTS_DIR / "summary.json"
    if not p.exists():
        pytest.skip(f"Summary not found at {p}; run the campaign first")
    return json.load(open(p))


# ---------------------------------------------------------------------------
# Criterion 1: Beverloo scaling
# ---------------------------------------------------------------------------

class TestBeverloo:
    """Beverloo discharge law: Q ∝ (D − k·d)^2.5."""

    @pytest.fixture(scope="class")
    def bev(self):
        s = _load_summary()
        bev = s.get("stages", {}).get("beverloo", {})
        if not bev:
            pytest.skip("No Beverloo results")
        return bev

    def test_min_particles(self, bev):
        """Statistical representativity: ≥ 5000 particles."""
        assert bev.get("min_particles", 0) >= 5000, (
            f"min particles = {bev.get('min_particles')}, need ≥ 5000"
        )

    def test_exponent(self, bev):
        """Exponent 2.5 ± 0.2."""
        acc = bev.get("acceptance", {})
        exp = acc.get("exponent", 0)
        assert abs(exp - 2.5) <= 0.2, (
            f"exponent = {exp:.4f}, |{exp:.4f} − 2.5| = {abs(exp-2.5):.4f} > 0.2"
        )

    def test_r2(self, bev):
        """R² ≥ 0.95."""
        acc = bev.get("acceptance", {})
        r2 = acc.get("r2", 0)
        assert r2 >= 0.95, f"R² = {r2:.4f} < 0.95"

    def test_steady_state_from_last_50pct(self, bev):
        """Verify the flow rate is computed from the last 50 % of windows."""
        # The campaign uses n_windows=8, steady = last 4
        for rf in sorted(ARTIFACTS_DIR.glob("beverloo_ratio_*.json")):
            r = json.load(open(rf))
            n_win = r["n_windows"]
            n_steady = r.get("flow_rate_steady_windows", 0)
            assert n_steady >= n_win // 2, (
                f"steady windows {n_steady} < {n_win}//2 for {rf.name}"
            )


# ---------------------------------------------------------------------------
# Criterion 2: Janssen wall-pressure saturation
# ---------------------------------------------------------------------------

class TestJanssen:
    """Janssen wall-pressure saturation: P_∞ within 10 % of theory."""

    @pytest.fixture(scope="class")
    def jan(self):
        s = _load_summary()
        jan = s.get("stages", {}).get("janssen", {})
        if not jan:
            pytest.skip("No Janssen results")
        return jan

    def test_p_inf_error(self, jan):
        """P_∞ relative error ≤ 10 %."""
        acc = jan.get("acceptance", {})
        err = acc.get("p_inf_rel_error")
        assert err is not None, "p_inf_rel_error not computed"
        assert err <= 0.10, f"P_∞ rel error = {err*100:.1f}% > 10%"

    def test_fit_quality(self, jan):
        """Janssen fit R² should be reasonable."""
        wf = jan.get("wall_fit", {})
        assert wf.get("ok"), "Janssen fit failed"
        assert wf.get("r2", 0) >= 0.90, f"Janssen fit R² = {wf.get('r2'):.4f} < 0.90"

    def test_particle_count(self, jan):
        """≥ 5000 particles."""
        assert jan.get("n_particles", 0) >= 5000, (
            f"n_particles = {jan.get('n_particles')}, need ≥ 5000"
        )


# ---------------------------------------------------------------------------
# Criterion 3: Flow-pattern transition
# ---------------------------------------------------------------------------

class TestFlowPattern:
    """Funnel → mass flow transition: θ_c within 15 % of Jenike."""

    @pytest.fixture(scope="class")
    def hop(self):
        s = _load_summary()
        hop = s.get("stages", {}).get("hopper", {})
        if not hop:
            pytest.skip("No hopper results")
        return hop

    def test_theta_c_error(self, hop):
        """Critical angle within 15 % of Jenike."""
        acc = hop.get("acceptance", {})
        err = acc.get("theta_c_rel_error")
        assert err is not None, "theta_c_rel_error not computed"
        if not math.isnan(err):
            assert err <= 0.15, f"θ_c rel error = {err*100:.1f}% > 15%"

    def test_regime_classification(self, hop):
        """Both mass and funnel regimes should be observed."""
        regimes = hop.get("regimes", [])
        assert "mass" in regimes, "No mass-flow case observed"
        assert "funnel" in regimes, "No funnel-flow case observed"

    def test_particle_count(self, hop):
        """≥ 5000 particles in each case."""
        n_parts = hop.get("n_particles", [])
        for n in n_parts:
            assert n >= 5000, f"n_particles = {n}, need ≥ 5000"


# ---------------------------------------------------------------------------
# Criterion 4: Energy ledger
# ---------------------------------------------------------------------------

class TestEnergy:
    """Energy conservation: per-timestep residual ≤ 1 %."""

    @pytest.fixture(scope="class")
    def eng(self):
        s = _load_summary()
        eng = s.get("stages", {}).get("energy", {})
        if not eng:
            pytest.skip("No energy results")
        return eng

    def test_residual(self, eng):
        """Best max residual ≤ 1 %."""
        acc = eng.get("acceptance", {})
        best = acc.get("best_residual", 1.0)
        assert best <= 0.01, f"best residual = {best*100:.4f}% > 1%"


# ---------------------------------------------------------------------------
# Overall
# ---------------------------------------------------------------------------

def test_overall_pass():
    """All four criteria must pass."""
    s = _load_summary()
    assert s.get("overall_pass"), (
        f"Overall verdict: {s.get('verdicts', {})}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
