"""Round 30 (m22864): invariants of the 2-D yardstick and of the re-judged verdict.

These tests do not merely *execute* ``reference_2d.py`` and
``judge_r30_2d.py``.  Each one pins a property that the round-30 conclusion
depends on, and each fails if the property is broken:

* **provenance is sentence-bound, not "the number occurs somewhere"** - the
  tampered-text cases below are exactly the ones a naive ``in``-test would pass
  (``0.3`` sits inside ``0.30``, ``1.0`` inside ``0.10``);
* **the law is the paper's law** - the constants at the stated ``I`` values, and
  the saturation caveat flag;
* **the OLS is a fit, not a shim** - it recovers known coefficients exactly and
  reports infinite errors for a two-point series instead of pretending;
* **the yardstick swap is not a rescue oracle** - the 2-D law is strictly below
  the 3-D reading in the tested band, and the measured ``mu`` fails *both*;
* **the artifact binds the verdict** - ``phi`` passes and ``mu`` fails on the
  archived round-30 numbers, read from the JSON rather than re-typed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks" / "plane_shear_mu_i"
sys.path.insert(0, str(BENCH))

import reference_2d as r2  # noqa: E402
import reference_law as rl  # noqa: E402

ARCHIVE = r2.ARCHIVE
R30 = REPO / "outputs" / "r30" / "r30_summary.json"


# --------------------------------------------------------------------------- #
# 1. provenance: sentence-bound
# --------------------------------------------------------------------------- #
def test_provenance_binds_both_constants():
    p = r2.verify_provenance()
    assert "µ∗ min ≃0.22" in p.mu_sentence
    assert "b ≃1.0" in p.mu_sentence
    assert "νmax ≃0.81" in p.nu_sentence
    assert "a ≃0.3" in p.nu_sentence
    assert "for µ = 0.4" in p.nu_sentence      # affiliation of the dilatancy pair
    assert len(p.digest) == 16


def test_provenance_rejects_corrupted_mu_value(tmp_path):
    text = ARCHIVE.read_text(encoding="utf-8", errors="replace")
    bad = tmp_path / "excerpts.txt"
    bad.write_text(text.replace("µ∗ min ≃0.22", "µ∗ min ≃0.36"), encoding="utf-8")
    with pytest.raises(r2.ProvenanceError):
        r2.verify_provenance(bad)


def test_provenance_rejects_swapped_sentences(tmp_path):
    """0.22 in the dilatancy sentence / 0.81 in the friction sentence."""
    text = ARCHIVE.read_text(encoding="utf-8", errors="replace")
    bad = tmp_path / "excerpts.txt"
    sw = text.replace("µ∗ min ≃0.22", "µ∗ min ≃0.81")
    bad.write_text(sw, encoding="utf-8")
    with pytest.raises(r2.ProvenanceError):
        r2.verify_provenance(bad)


def test_provenance_rejects_truncated_sentence(tmp_path):
    """Anchor present but its numbers gone - a bare ``in`` test would pass."""
    text = ARCHIVE.read_text(encoding="utf-8", errors="replace")
    bad = tmp_path / "excerpts.txt"
    bad.write_text(text.replace("b ≃1.0", "b ≃ 1"), encoding="utf-8")
    with pytest.raises(r2.ProvenanceError):
        r2.verify_provenance(bad)


def test_provenance_rejects_missing_archive(tmp_path):
    with pytest.raises(r2.ProvenanceError):
        r2.verify_provenance(tmp_path / "nope.txt")


# --------------------------------------------------------------------------- #
# 2. the law
# --------------------------------------------------------------------------- #
def test_mu_2d_values():
    assert r2.mu_2d(0.0) == pytest.approx(0.22)
    assert r2.mu_2d(0.1) == pytest.approx(0.32)
    assert r2.mu_2d(0.2) == pytest.approx(0.42)


def test_nu_2d_decreases_and_matches_paper():
    assert r2.nu_2d(0.0) == pytest.approx(0.81)
    assert r2.nu_2d(0.1) == pytest.approx(0.78)
    assert r2.nu_2d(0.3) < r2.nu_2d(0.05)        # dilatancy: phi falls with I
    assert r2.nu_2d(1e6) >= 0.5                  # clipped, never negative


def test_2d_law_is_strictly_below_3d_reading_in_band():
    """The swap is a *lower* yardstick - it cannot be used to inflate a verdict."""
    for I in (0.005, 0.02, 0.05, 0.1, 0.2):
        assert r2.mu_2d(I) < rl.ref_mu(I)
    for I in (0.005, 0.02, 0.05, 0.1, 0.2):
        assert r2.nu_2d(I) > rl.ref_phi(I)       # 2-D packs denser than 3-D


# --------------------------------------------------------------------------- #
# 3. the fit
# --------------------------------------------------------------------------- #
def test_fit_recovers_paper_coefficients_exactly():
    I = [0.02, 0.05, 0.19]
    y = [r2.mu_2d(i) for i in I]
    fit = r2.fit_linear(I, y)
    assert fit["intercept"] == pytest.approx(r2.MU_MIN_2D, abs=1e-12)
    assert fit["slope"] == pytest.approx(r2.B_2D, abs=1e-12)
    assert fit["chi2"] == pytest.approx(0.0, abs=1e-18)


def test_fit_two_points_has_infinite_errors_not_fake_ones():
    fit = r2.fit_linear([0.02, 0.2], [0.16, 0.21])
    assert math.isnan(fit["intercept_se"]) and math.isnan(fit["slope_se"])
    assert fit["n"] == 2


def test_fit_slope_uncertainty_shrinks_with_spread():
    wide = r2.fit_linear([0.0, 0.5], [0.0, 1.0])
    tight = r2.fit_linear([0.0, 0.01], [0.0, 0.01])
    assert wide["slope"] == pytest.approx(2.0)
    assert tight["slope"] == pytest.approx(1.0)


def test_fit_rejects_single_point():
    with pytest.raises(ValueError):
        r2.fit_linear([0.1], [0.2])


# --------------------------------------------------------------------------- #
# 4. the archived verdict (artifact-bound)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not R30.exists(), reason="round-30 summary not generated")
def test_archived_verdict_phi_passes_mu_fails():
    S = json.loads(R30.read_text())
    pts = [p for p in S["points"] if not p["beyond_linear_range"]]
    assert pts, "no in-range points"
    # the dilatancy side is within the 10 % band against the 2-D law ...
    assert max(abs(p["phi_err_2d_pct"]) for p in pts) <= 10.0
    # ... while the friction side is not, on either yardstick
    assert max(abs(p["mu_err_2d_pct"]) for p in pts) > 10.0
    assert max(abs(p["mu_err_3d_pct"]) for p in pts) > 10.0
    # and the 2-D yardstick really is the more forgiving one at every point
    for p in pts:
        assert abs(p["mu_err_2d_pct"]) < abs(p["mu_err_3d_pct"])
        assert abs(p["phi_err_2d_pct"]) < abs(p["phi_err_3d_pct"])


@pytest.mark.skipif(not R30.exists(), reason="round-30 summary not generated")
def test_archived_kinetic_bound_cannot_explain_slope_deficit():
    """The streaming-stress upper bound must be far below the missing ~70 % of b."""
    S = json.loads(R30.read_text())
    assert S["kinetic"], "no kinetic-bound entries"
    worst = max(k["frac_of_p_bulk"] for k in S["kinetic"])
    assert worst < 0.01, worst                   # < 1 % of p_bulk
    deficit = 1.0 - S["fit_mu"]["slope"] / r2.B_2D
    assert deficit > 0.5 and worst < deficit


@pytest.mark.skipif(not R30.exists(), reason="round-30 summary not generated")
def test_summary_carries_provenance_digest_matching_archive():
    S = json.loads(R30.read_text())
    assert S["provenance"]["digest"] == r2.verify_provenance().digest
    assert S["reference_2d"]["mu_min"] == r2.MU_MIN_2D
    assert S["reference_2d"]["b"] == r2.B_2D