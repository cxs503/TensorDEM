"""GAP-DEM-FRICTION m15159: machine rules of the direct-shear measurement.

Fast, no-DEM tests of the post-processing contract (the physics runs are covered
by ``tests/test_shear_cell_friction.py`` and by the committed run artifacts).
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.m15159.study import (  # noqa: E402
    DEGENERATE_BAND,
    REF_MU,
    benchmark,
    classify,
    envelope,
)


def test_classify_rejects_unsteady_window():
    ok, deg, why = classify(0.36, 0.5, drift_rel=0.24, stationarity_tol=0.15)
    assert not ok and not deg
    assert any("not steady" in r for r in why)


def test_classify_accepts_steady_non_degenerate():
    ok, deg, why = classify(0.3817, 0.5, drift_rel=-0.13, stationarity_tol=0.15)
    assert ok and not deg and why == []


@pytest.mark.parametrize("mu_p", [0.3, 0.5, 0.7])
def test_classify_flags_interface_sliding(mu_p):
    """mu == mu_p to within the band means the platen slid on the top layer."""
    ok, deg, why = classify(mu_p * 1.001, mu_p, drift_rel=0.0)
    assert deg and not ok
    assert any("degenerate" in r for r in why)
    assert DEGENERATE_BAND[0] <= 1.001 <= DEGENERATE_BAND[1]


def test_classify_rejects_nan():
    ok, deg, why = classify(float("nan"), 0.5, drift_rel=float("nan"))
    assert not ok and not deg and why


def _rec(tag, sigma, tau, mu_p=0.5, drift=-0.05, usable=True, degenerate=False):
    return {
        "tag": tag,
        "case": {"mu_p": mu_p, "gamma": 1.0, "sigma_target_pa": sigma, "geom": "medium"},
        "bed": {"n_particles": 390},
        "level": {"sigma_target_pa": sigma, "sigma_pa": sigma, "tau_pa": tau,
                  "mu_macro": tau / sigma, "drift_rel": drift, "usable": usable,
                  "degenerate": degenerate, "reject_reasons": []},
        "repro_cmd": "PYTHONPATH=src python scripts/m15159/run_gap_dem_friction.py --tag %s" % tag,
    }


def test_benchmark_error_and_pass():
    rec = _rec("headline", 4000.0, 0.3817 * 4000.0)
    b = benchmark(rec)
    assert b["got"] == pytest.approx(0.3817, rel=1e-12)
    assert b["signed_rel_error"] == pytest.approx(0.3817 / REF_MU - 1.0, rel=1e-12)
    assert b["within_tolerance"] and b["pass"]
    # an off-anchor point must fail the 10% criterion
    bad = benchmark(_rec("x", 4000.0, 0.50 * 4000.0))
    assert not bad["within_tolerance"] and not bad["pass"]
    # a steady number is still not quotable when the gate rejects the run
    rejected = benchmark(_rec("y", 4000.0, 0.3817 * 4000.0, usable=False))
    assert rejected["within_tolerance"] and not rejected["pass"]


def test_envelope_separates_usable_levels():
    pts = [_rec("l1", 2000.0, 100.0 + 0.36 * 2000.0),
           _rec("l2", 4000.0, 100.0 + 0.36 * 4000.0),
           _rec("l3", 8000.0, 100.0 + 0.36 * 8000.0),
           _rec("bad", 8000.0, 9000.0, usable=False)]
    env = envelope(pts)
    assert env["status"] == "ok" and env["n_levels"] == 3
    assert env["tan_phi"] == pytest.approx(0.36, rel=1e-9)
    assert env["cohesion_pa"] == pytest.approx(100.0, rel=1e-9)
    assert env["r_squared"] == pytest.approx(1.0, abs=1e-12)


def test_committed_artifacts_are_machine_checkable():
    """The committed run artifacts must carry the numbers the tables quote."""
    from scripts.m15159.study import OUT_DIR

    if not os.path.isdir(OUT_DIR):
        pytest.skip("no committed run artifacts")
    files = [f for f in os.listdir(OUT_DIR) if f.endswith(".json") and f != "benchmark.json"]
    assert files, "expected at least one run artifact"
    for name in files:
        with open(os.path.join(OUT_DIR, name)) as fh:
            rec = json.load(fh)
        for key in ("case", "bed", "level", "history", "repro_cmd"):
            assert key in rec, (name, key)
        lv = rec["level"]
        assert lv["sigma_pa"] > 0 and lv["tau_pa"] > 0
        assert lv["mu_macro"] == pytest.approx(lv["tau_pa"] / lv["sigma_pa"], rel=1e-9)
        assert 0.5 < lv["sigma_pa"] / lv["sigma_target_pa"] < 1.6
        assert len(rec["history"]["strain"]) == len(rec["history"]["tau_pa"]) > 50
        usable, degenerate, _ = classify(lv["mu_macro"], rec["case"]["mu_p"], lv["drift_rel"])
        assert (usable, degenerate) == (lv["usable"], lv["degenerate"])