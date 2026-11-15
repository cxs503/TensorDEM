"""Regression tests for the m21803 additivity analyser and the R5 probe defect.

Two things are pinned here:

1. The pairing/tolerance logic of
   ``benchmarks/case_dem_biaxial_friction/analyse_rolling_surrogate.py``: a
   (mu, mu_r>0) run is only judged against the pure-friction run at mu+mu_r, and
   an unmeasured pair must stay ``measured=False`` instead of silently passing.
2. The step-count key of ``run_biaxial_shear_band`` is ``loading["steps"]``.
   Round 5's probe read ``loading["total_steps"]`` and therefore crashed after
   every (expensive) run, which is why no R5 calibration artefact ever existed.
   A source-level check keeps that specific bug from coming back.
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "case_dem_biaxial_friction"))

import analyse_rolling_surrogate as analyser  # noqa: E402


def _row(mu: float, mu_r: float, phi_cv: float) -> dict:
    return {"tag": f"mu{int(mu*100):03d}_mur{int(mu_r*100):02d}",
            "mu": mu, "mu_r": mu_r, "mu_eff": mu + mu_r, "phi_cv_deg": phi_cv}


def test_pair_is_matched_on_mu_plus_mu_r():
    rows = [_row(0.55, 0.20, 24.0), _row(0.75, 0.0, 24.4), _row(0.55, 0.0, 18.0)]
    out = analyser.additivity_rows(rows)
    assert len(out) == 1, "only the mu_r>0 run needs a partner"
    pair = out[0]
    assert pair["measured"] is True
    assert pair["merged_tag"] == "mu075_mur00"
    assert pair["delta_phi_cv_deg"] == pytest.approx(-0.4)


def test_unpaired_run_is_reported_as_unmeasured_not_passed():
    rows = [_row(0.55, 0.20, 24.0), _row(0.55, 0.0, 18.0)]  # no mu=0.75 control
    out = analyser.additivity_rows(rows)
    assert out and out[0]["measured"] is False
    assert "additive_within_tolerance" not in out[0]


def test_tolerance_is_the_pre_declared_constant():
    assert analyser.TOL_ADDITIVITY_DEG == pytest.approx(1.5)
    inside = [_row(0.55, 0.20, 24.0), _row(0.75, 0.0, 25.0)]
    outside = [_row(0.55, 0.20, 24.0), _row(0.75, 0.0, 27.0)]
    assert analyser.additivity_rows(inside)[0]["additive_within_tolerance"] is True
    assert analyser.additivity_rows(outside)[0]["additive_within_tolerance"] is False


def test_probe_scripts_use_the_real_step_key():
    """``loading`` exposes ``steps``; ``total_steps`` does not exist (R5 bug)."""
    for rel in ("scripts/m21110_probe_rolling_cs.py",
                "scripts/m21803_rolling_surrogate.py"):
        src = open(os.path.join(REPO, rel)).read()
        assert "total_steps" not in src, f"{rel} still reads a non-existent key"


def test_analyser_uses_isochoric_log_settings():
    """The isochoric critical-state detector needs log strain + isochoric BC."""
    src = open(os.path.join(REPO,
                            "benchmarks/case_dem_biaxial_friction",
                            "analyse_rolling_surrogate.py")).read()
    assert 'strain_measure="log"' in src
    assert 'boundary_control="isochoric"' in src