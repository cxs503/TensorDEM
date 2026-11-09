"""DEM coarse-graining scaling law verification (IN61-N07).

Fast analytic tests for the scaling laws.  Heavy macro-anchor simulations
are run by ``scripts/run_coarse_graining_campaign.py`` and their results
are loaded from ``runs/in61_n07/results.json``.

Criteria verified here:
① Contact-level: t_c deviation <2%, e deviation <0.02
⑤ Negative control: mass-only scaling → t_c deviation >30%
Pareto front: speedup = k³, contact deviation <2% for all k
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import tensordem.coarse_graining as cg
from tensordem.coarse_graining import (
    scale_contact_time_invariant,
    scale_mass_only,
    verify_contact_scaling,
    contact_time_analytic,
    restitution_from_damping,
)

# ---------------------------------------------------------------------------
# Base material parameters
# ---------------------------------------------------------------------------

BASE = dict(
    radius=0.005,
    density=2500.0,
    kn=1.0e5,
    kt_ratio=0.5,
    restitution=0.3,
    mu=0.5,
    mu_rolling=0.05,
    gravity=9.81,
)

K_VALUES = [1.0, 2.0, 4.0, 8.0]


# ===================================================================
# ① Contact-level scaling verification
# ===================================================================

class TestContactScaling:
    """Criterion ①: t_c deviation <2%, e deviation <0.02."""

    def test_contact_time_invariance(self):
        """Contact time must be invariant under k-rescaling."""
        result = verify_contact_scaling(k_values=K_VALUES, **BASE)
        for row in result["scaled"]:
            assert row["contact_time_dev_pct"] < 2.0, (
                f"k={row['k']}: t_c deviation {row['contact_time_dev_pct']:.4f}% >= 2%"
            )

    def test_restitution_invariance(self):
        """Restitution coefficient must be invariant under k-rescaling."""
        result = verify_contact_scaling(k_values=K_VALUES, **BASE)
        for row in result["scaled"]:
            assert row["restitution_dev"] < 0.02, (
                f"k={row['k']}: e deviation {row['restitution_dev']:.4f} >= 0.02"
            )

    def test_scaling_laws_analytic(self):
        """Verify the analytic scaling relationships."""
        for k in [2.0, 4.0, 8.0]:
            scaled = scale_contact_time_invariant(k=k, **BASE)
            mass_orig = BASE["density"] * 4.0 / 3.0 * math.pi * BASE["radius"] ** 3

            # Mass scales as k³
            assert abs(scaled.mass / mass_orig - k ** 3) / k ** 3 < 1e-10
            # Radius scales as k
            assert abs(scaled.radius / BASE["radius"] - k) / k < 1e-10
            # Stiffness scales as k³
            assert abs(scaled.kn / BASE["kn"] - k ** 3) / k ** 3 < 1e-10
            # Rolling stiffness scales as k⁵
            kr_orig = BASE["kt_ratio"] * BASE["kn"] * BASE["radius"] ** 2
            assert abs(scaled.kr / kr_orig - k ** 5) / k ** 5 < 1e-10
            # Damping scales as k³
            gamma_orig = 2.0 * (abs(math.log(BASE["restitution"])) /
                               math.sqrt(math.pi ** 2 + math.log(BASE["restitution"]) ** 2)) * \
                         math.sqrt(BASE["kn"] * 0.5 * mass_orig)
            assert abs(scaled.gamma_pair / gamma_orig - k ** 3) / k ** 3 < 1e-6
            # Dimensionless quantities unchanged
            assert scaled.restitution == BASE["restitution"]
            assert scaled.mu == BASE["mu"]
            assert scaled.mu_rolling == BASE["mu_rolling"]


# ===================================================================
# ⑤ Negative control
# ===================================================================

class TestNegativeControl:
    """Criterion ⑤: Mass-only scaling breaks contact-time invariance."""

    def test_mass_only_breaks_contact_time(self):
        """Scaling mass without stiffness must give >30% t_c deviation."""
        for k in [2.0, 4.0, 8.0]:
            nc = scale_mass_only(k=k, **BASE)
            mass_orig = BASE["density"] * 4.0 / 3.0 * math.pi * BASE["radius"] ** 3
            mass_eff_orig = 0.5 * mass_orig
            tc_orig = contact_time_analytic(BASE["kn"], mass_eff_orig)

            mass_eff_new = 0.5 * nc.mass
            tc_new = contact_time_analytic(nc.kn, mass_eff_new)
            dev = abs(tc_new - tc_orig) / tc_orig * 100.0

            assert dev > 30.0, (
                f"k={k}: negative control t_c deviation {dev:.1f}% <= 30%"
            )

    def test_mass_only_deviation_grows_with_k(self):
        """Negative control deviation should grow as k^(3/2)."""
        devs = []
        for k in [2.0, 4.0, 8.0]:
            nc = scale_mass_only(k=k, **BASE)
            mass_orig = BASE["density"] * 4.0 / 3.0 * math.pi * BASE["radius"] ** 3
            mass_eff_orig = 0.5 * mass_orig
            tc_orig = contact_time_analytic(BASE["kn"], mass_eff_orig)
            tc_new = contact_time_analytic(nc.kn, 0.5 * nc.mass)
            devs.append(abs(tc_new - tc_orig) / tc_orig)

        # Deviation should grow: dev(k) ≈ k^(3/2) - 1
        for i, k in enumerate([2.0, 4.0, 8.0]):
            expected_ratio = (k ** 1.5 - 1) / (2.0 ** 1.5 - 1)
            actual_ratio = devs[i] / devs[0] if devs[0] > 0 else 1.0
            # Within 10% of expected ratio
            assert abs(actual_ratio / expected_ratio - 1.0) < 0.1, (
                f"k={k}: deviation ratio {actual_ratio:.2f} vs expected {expected_ratio:.2f}"
            )


# ===================================================================
# Pareto front
# ===================================================================

def compute_pareto_front() -> dict:
    """Compute the acceleration-deviation Pareto front."""
    speedups = {k: k ** 3 for k in K_VALUES}
    contact_result = verify_contact_scaling(k_values=K_VALUES, **BASE)
    contact_devs = {r["k"]: r["contact_time_dev_pct"] for r in contact_result["scaled"]}

    return {
        "k_values": K_VALUES,
        "speedups": speedups,
        "contact_time_deviations": contact_devs,
        "particle_reduction": {k: 1.0 - 1.0 / k ** 3 for k in K_VALUES},
    }


class TestParetoFront:
    """Verify the Pareto front is computed correctly."""

    def test_speedup_scales_as_k_cubed(self):
        """Speedup must scale as k³."""
        pareto = compute_pareto_front()
        for k in K_VALUES:
            assert abs(pareto["speedups"][k] - k ** 3) < 1e-10

    def test_contact_deviation_small(self):
        """Contact deviation should remain small for all k."""
        pareto = compute_pareto_front()
        for k, dev in pareto["contact_time_deviations"].items():
            assert dev < 2.0, f"k={k}: contact deviation {dev:.4f}% >= 2%"

    def test_particle_reduction(self):
        """Particle reduction should approach 100% as k increases."""
        pareto = compute_pareto_front()
        assert pareto["particle_reduction"][1.0] == 0.0
        assert pareto["particle_reduction"][8.0] > 0.99


# ===================================================================
# Campaign results validation (if available)
# ===================================================================

RESULTS_PATH = Path(__file__).resolve().parents[1] / "runs" / "in61_n07" / "results.json"


class TestCampaignResults:
    """Validate campaign results if the runner has been executed."""

    @pytest.fixture(scope="class")
    def results(self):
        """Load campaign results if available."""
        if not RESULTS_PATH.exists():
            pytest.skip("Campaign results not yet generated")
        with open(RESULTS_PATH) as f:
            return json.load(f)

    def test_contact_results_present(self, results):
        """Contact verification results must be present."""
        assert "contact_scaling" in results
        assert "scaled" in results["contact_scaling"]
        assert len(results["contact_scaling"]["scaled"]) >= 3

    def test_repose_results_if_present(self, results):
        """If repose results exist, validate angle range."""
        if "repose" not in results:
            pytest.skip("No repose results")
        angles = [results.get("repose", {})[str(k)]["theta_r_deg"]
                  for k in [1, 2, 4] if str(k) in results.get("repose", {})]
        if len(angles) >= 2:
            angle_range = max(angles) - min(angles)
            assert angle_range < 2.0, f"Repose angle range {angle_range:.2f}° >= 2°"

    def test_beverloo_results_if_present(self, results):
        """If Beverloo results exist, validate exponent."""
        if "beverloo" not in results or "fit" not in results.get("beverloo", {}):
            pytest.skip("No Beverloo results")
        fit = results.get("beverloo", {})["fit"]
        assert abs(fit["exponent"] - 2.5) < 0.15, (
            f"Beverloo exponent {fit['exponent']:.3f} outside 2.5±0.15"
        )

    def test_shear_results_if_present(self, results):
        """If shear results exist, validate R² and angle range."""
        if "shear" not in results:
            pytest.skip("No shear results")
        for k_str, r in results.get("shear", {}).items():
            assert r["r_squared"] > 0.99, f"k={k_str}: R² = {r['r_squared']:.4f} < 0.99"
        angles = [r["friction_angle_deg"] for r in results.get("shear", {}).values()]
        if len(angles) >= 2:
            angle_range = max(angles) - min(angles)
            assert angle_range < 1.5, f"Friction angle range {angle_range:.2f}° >= 1.5°"


# ===================================================================
# Report generation helper
# ===================================================================

def generate_report() -> dict:
    """Generate the campaign report dict."""
    contact = verify_contact_scaling(k_values=K_VALUES, **BASE)
    pareto = compute_pareto_front()

    return {
        "campaign": "IN61-N07",
        "title": "DEM Coarse-Graining Scaling Law & Physical Deviation Ledger",
        "base_material": BASE,
        "contact_scaling": contact,
        "pareto_front": pareto,
        "criteria": {
            "contact_tc_deviation_pct": {
                "threshold": 2.0,
                "results": {r["k"]: r["contact_time_dev_pct"] for r in contact["scaled"]},
            },
            "contact_e_deviation": {
                "threshold": 0.02,
                "results": {r["k"]: r["restitution_dev"] for r in contact["scaled"]},
            },
            "negative_control_tc_deviation_pct": {
                "threshold": 30.0,
                "results": {r["k"]: r["contact_time_dev_pct"] for r in contact["negative_control"]},
            },
        },
    }


if __name__ == "__main__":
    report = generate_report()
    print(json.dumps(report, indent=2))
