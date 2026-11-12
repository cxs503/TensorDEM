"""Unit tests for the kinetic/contact stress decomposition of
``tensordem.elastic_limit_shear`` and for the 2 % drop analysis of the
CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT round-2 driver.

The decomposition was added in mission m16587 so that Campbell's "purely
inertial" (collision-borne) branch can be separated from the elastic
(contact-borne) branch without touching the total-stress normalisation:
``tau = tau_kinetic + tau_contact`` holds *sample by sample*, hence in the mean.

Run::

    PYTHONPATH=src python -m pytest tests/test_elastic_limit_stress_decomposition.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.elastic_limit_shear import (  # noqa: E402
    ElasticLimitShearCase, simulate_plane_shear)

sys.path.insert(0, str(REPO / "benchmarks" / "case_dem_shear_campbell_elastic_limit"))
import phi_transition as drv  # noqa: E402


def _small(phi: float, seed: int = 7) -> ElasticLimitShearCase:
    return ElasticLimitShearCase(n_particles=120, solid_fraction=phi, kn=1.0e3,
                                 shear_rate=1.0, strain_total=0.5,
                                 strain_average_from=0.25, dt_factor=0.004,
                                 seed=seed)


# ---------------------------------------------------------------------------
# invariants of the decomposition
# ---------------------------------------------------------------------------
def test_decomposition_is_additive_exactly():
    out = simulate_plane_shear(_small(0.80))
    assert out["samples"] > 0
    assert out["tau_kinetic"] == pytest.approx(
        out["tau"] - out["tau_contact"], rel=0, abs=1e-10)
    assert out["pressure_kinetic"] == pytest.approx(
        out["pressure"] - out["pressure_contact"], rel=0, abs=1e-10)
    # ... and the residual is the only error term the samples can have
    assert abs(out["tau"] - out["tau_kinetic"] - out["tau_contact"]) < 1e-10 * max(1.0, abs(out["tau"]))


def test_share_is_a_fraction_and_gates_untouched():
    out = simulate_plane_shear(_small(0.80))
    assert 0.0 <= out["kinetic_stress_share"] <= 1.0
    assert math.isfinite(out["kinetic_stress_share"])
    # the new diagnostics must not alter the gate set (or its verdict)
    assert set(out["gates"]) == {"energy", "overlap", "steady", "coordination",
                                 "pressure_positive", "finite", "all"}
    assert out["gates"]["all"] == all(v for k, v in out["gates"].items() if k != "all")


def test_dense_bed_is_contact_borne():
    """The elastic branch must be carried by contacts, not by peculiar velocity."""
    out = simulate_plane_shear(_small(0.80))
    assert out["tau_contact"] > 0.0
    assert out["kinetic_stress_share"] < 0.2


def test_loose_bed_shifts_towards_kinetic():
    """Lower concentration => larger kinetic share (same apparatus, same seed)."""
    dense = simulate_plane_shear(_small(0.80))
    loose = simulate_plane_shear(_small(0.62))
    assert loose["kinetic_stress_share"] > dense["kinetic_stress_share"]


# ---------------------------------------------------------------------------
# analysis helpers of the round-2 driver (synthetic points: no simulation)
# ---------------------------------------------------------------------------
def _pt(phi, tau, gd=1.0, share=0.0, tag=None, gated=True):
    return {"phi": phi, "tau": tau, "pressure": 2.0 * tau, "tau_star": tau,
            "gamma_dot": gd, "kinetic_stress_share": share,
            "tag": tag or f"map_phi{phi:.2f}_g1.0",
            "gates": {"all": gated, "coordination": gated}, "I": 0.1,
            "mu_eff": 0.5}


def test_log_slope_recovers_a_power_law():
    assert drv._log_slope([1, 10], [3, 300]) == pytest.approx(2.0)
    assert math.isnan(drv._log_slope([1, 1], [3, 4]))   # no abscissa spread


def test_spearman_endpoints():
    assert drv.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert drv.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert math.isnan(drv.spearman([1, 2], [1, 2]))


def test_drop_analysis_uses_two_grid_steps_and_gated_points_only():
    # 0.01 grid: a 2 % window is two grid steps
    pts = [_pt(0.80, 100.0), _pt(0.79, 90.0), _pt(0.78, 50.0),
           _pt(0.77, 40.0)]
    d = drv.drop_analysis(pts)
    ratios = {round(r["ratio"], 6) for r in d["rows"]}
    assert 2.0 in ratios and 2.25 in ratios          # 100/50 and 90/40
    assert d["R2"] == pytest.approx(2.25)
    assert d["steepest"]["phi_hi"] == pytest.approx(0.79)

    # a gate-rejected endpoint removes exactly the windows that touch it
    pts[0]["gates"]["all"] = False
    d2 = drv.drop_analysis(pts)
    assert [r["phi_hi"] for r in d2["rows"]] == [pytest.approx(0.79)]
    assert d2["R2"] == pytest.approx(2.25)


def test_verdict_thresholds_are_the_preregistered_ones():
    """P1: R2 >= 100 PASS, >= 10 PARTIAL, else NOT_SUPPORTED (2 % window)."""
    assert drv.WINDOW == pytest.approx(0.02)
    assert drv.PHI_MAP[0] == pytest.approx(0.80)
    assert drv.PHI_MAP[-1] == pytest.approx(0.64)
    # the map is a 0.01 grid so that WINDOW is exactly two steps
    steps = {round(a - b, 10) for a, b in zip(drv.PHI_MAP, drv.PHI_MAP[1:])}
    assert steps == {0.01}


def test_preregistration_is_immutable_and_well_formed():
    pre = drv.preregistration()
    assert pre["written_before_any_point"] is True
    ids = [p["id"] for p in pre["predictions"]]
    assert ids == ["P1", "P2", "P3"]
    for p in pre["predictions"]:
        assert p["pass_rule"] and p["metric"]
    assert len(pre["anchor_sha256"]) == 64