"""Tests for the literature-band verification (closes R7 §4.3 item 3 debt).

Three groups:

* the arithmetic of the adjudication (inside-band / relative-error / tolerance
  logic) on synthetic inputs whose answer is known by construction;
* the theoretical-window placement: a measured ``theta`` that equals a law's
  prediction at a known state must be reported inside that law's window;
* the case's published R7 band angle (``theta = 55.00 deg``) placed against the
  Desrues & Viggiani (2004) 55-60 deg band and the three-law window at the
  measured state (``phi ~ 35.45 deg``) — the numbers are the committed R7/R9
  evidence, re-derived here without a DEM cell.
"""
from __future__ import annotations

import math

import pytest

from tensordem.literature_band_verification import (
    LITERATURE_BANDS,
    adjudicate_band_against_literature,
    band_window_span,
    theoretical_band_window,
    verify_literature_anchor,
)


# ---------------------------------------------------------------------------
# arithmetic of the adjudication
# ---------------------------------------------------------------------------
def test_inside_band_and_tolerance():
    # theta = 55.0 is inside the 55-60 band and 8.33% from the 60 midpoint
    adj = adjudicate_band_against_literature(55.0)
    assert adj.inside_band is True
    assert adj.rel_error_vs_midpoint == pytest.approx(8.33 / 100.0, abs=1e-4)
    assert adj.within_15pct is True
    assert adj.within_3pct is False
    assert adj.verdict == "PASS"


def test_outside_band_fails():
    # theta = 40.0 is outside the 55-60 band and 33% from the midpoint
    adj = adjudicate_band_against_literature(40.0)
    assert adj.inside_band is False
    assert adj.rel_error_vs_midpoint == pytest.approx(0.3333, abs=1e-3)
    assert adj.within_15pct is False
    assert adj.verdict == "FAIL"


def test_midpoint_exact():
    # theta = 60.0 (the anchor midpoint) has zero relative error
    adj = adjudicate_band_against_literature(60.0)
    assert adj.rel_error_vs_midpoint == pytest.approx(0.0, abs=1e-12)
    assert adj.within_3pct is True


def test_band_registry_provenance():
    # the relation sources are 'verified'; the borrowed band is 'quoted'
    assert LITERATURE_BANDS["roscoe_1970"].provenance == "verified"
    assert LITERATURE_BANDS["arthur_1977"].provenance == "verified"
    assert LITERATURE_BANDS["desrues_viggiani_2004"].provenance == "quoted"


# ---------------------------------------------------------------------------
# theoretical-window placement
# ---------------------------------------------------------------------------
def test_window_equals_law_prediction():
    # at phi=30, psi=0 the three laws give 60 / 45 / 52.5; a measured 60 deg
    # (Mohr-Coulomb) must be inside the window
    pred = theoretical_band_window(30.0, 0.0)
    assert pred["mohr_coulomb"] == pytest.approx(60.0)
    assert pred["roscoe"] == pytest.approx(45.0)
    assert pred["arthur"] == pytest.approx(52.5)
    lo, hi = band_window_span(pred)
    assert lo == pytest.approx(45.0)
    assert hi == pytest.approx(60.0)

    adj = adjudicate_band_against_literature(60.0, phi_deg=30.0, psi_deg=0.0)
    assert adj.window_span == (45.0, 60.0)
    # qualitative agreement via the theoretical window even though 60 is the
    # band's own midpoint
    assert adj.verdict == "PASS"


def test_window_qualitative_agreement():
    # a measured 52.5 deg (Arthur at phi=30, psi=0) is outside the 55-60 band
    # but inside the three-law window -> qualitative PASS
    adj = adjudicate_band_against_literature(52.5, phi_deg=30.0, psi_deg=0.0)
    assert adj.inside_band is False
    assert adj.window_span == (45.0, 60.0)
    assert adj.verdict == "PASS"


# ---------------------------------------------------------------------------
# case's published R7 band angle vs the literature
# ---------------------------------------------------------------------------
def test_case_r7_band_angle_vs_literature():
    """The committed R7/R9 evidence: theta=55.00, phi~35.45, psi~4.1 (Rowe 32)."""
    # R7 ensemble: theta = 55.00 +- 2.07 deg (n=8), 8.33% from the 60 anchor
    # R9: measured phi (peak) = 35.45 deg; Rowe psi at phi_cv=32 = 4.14 deg
    res = verify_literature_anchor(55.00, phi_deg=35.45, psi_deg=4.14)
    assert res["theta_deg"] == pytest.approx(55.00)
    assert res["inside_band"] is True          # 55.00 in [55, 60]
    assert res["within_15pct"] is True         # 8.33% < 15%
    assert res["within_3pct"] is False         # 8.33% > 3% (honest)
    assert res["verdict"] == "PASS"
    # the three-law window at the measured state brackets the band angle
    assert res["window_span"][0] <= 55.00 <= res["window_span"][1]
    # Arthur (the R9-supported law) is the closest prediction
    arthur = res["theoretical_window"]["arthur"]
    assert abs(arthur - 55.00) < abs(res["theoretical_window"]["mohr_coulomb"] - 55.00)


def test_case_r7_band_angle_phi_only():
    """Without a psi reading the window is still reported; verdict unchanged."""
    res = verify_literature_anchor(55.00, phi_deg=35.45)
    assert res["verdict"] == "PASS"
    assert res["window_span"] is None          # no psi -> no window
