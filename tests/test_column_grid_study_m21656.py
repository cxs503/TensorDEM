"""Tests for :mod:`tensordem.granular_column.grid_study` (m21656).

The module is pure arithmetic, so every test is pinned against a **closed-form
synthetic ladder** whose order of accuracy is known exactly: for a sequence
``f(h) = f0 + C h**p`` the three-level Richardson estimator must return ``p``
and ``f0`` to machine precision, and the GCI must land in the asymptotic range.
A test that merely executes the code would not detect a wrong log-base or a
sign flip in ``(f3 - f2)/(f2 - f1)``; these do.
"""

from __future__ import annotations

import math

import pytest

from tensordem.granular_column import grid_study as gs


def synthetic(p=2.0, f0=1.0, c=0.5, r=2.0, n=3):
    """Ladder of ``n`` levels with exact order ``p`` (finest ``h = 1``)."""
    levels = []
    for k in range(n):
        h = r ** (n - 1 - k)          # coarse -> fine
        levels.append((h, f0 + c * h ** p))
    return levels


# --------------------------------------------------------------------------
# exact recovery of the order and of the extrapolated value
# --------------------------------------------------------------------------
@pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("r", [1.5, 2.0, 4.0])
def test_observed_order_exact(p, r):
    (h1, f1), (h2, f2), (h3, f3) = synthetic(p=p, r=r)
    got = gs.observed_order(h1, h2, h3, f1, f2, f3)
    assert got["monotone"] is True
    assert got["p"] == pytest.approx(p, rel=1e-9, abs=1e-12)
    assert got["r"] == pytest.approx(r, rel=1e-9)


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_richardson_recovers_exact_value(p):
    (h1, f1), (h2, f2), (h3, f3) = synthetic(p=p, f0=7.25, c=-0.3)
    got = gs.observed_order(h1, h2, h3, f1, f2, f3)
    assert gs.richardson_extrapolate(f2, f3, got["p"], got["r"]) == pytest.approx(7.25, rel=1e-9)


def test_gci_asymptotic_ratio_is_one():
    """In the asymptotic range ``GCI_mid / (r**p GCI_fine) -> 1``."""
    # c = 1e-6 keeps every level within 1e-5 of f0 = 1, i.e. genuinely
    # asymptotic; the ratio must then be 1 to machine precision.
    levels = synthetic(p=2.0, f0=1.0, c=1.0e-6, r=2.0, n=3)
    (h1, f1), (h2, f2), (h3, f3) = levels
    got = gs.observed_order(h1, h2, h3, f1, f2, f3)
    est = gs.gci(f1, f2, f3, got["p"], got["r"])
    assert est["asymptotic_ratio"] == pytest.approx(1.0, rel=1e-4)
    assert est["f_extrapolated"] == pytest.approx(1.0, rel=1e-9)


def test_gci_value_matches_closed_form():
    """c = 0.5, p = 2, r = 2: |f3-f2|/f3 = 1, r**p - 1 = 3."""
    (h1, f1), (h2, f2), (h3, f3) = synthetic(p=2.0, f0=1.0, c=0.5, r=2.0, n=3)
    got = gs.observed_order(h1, h2, h3, f1, f2, f3)
    est = gs.gci(f1, f2, f3, got["p"], got["r"])
    assert est["u_rel"] == pytest.approx(gs.ASME_FS * 1.0 / 3.0, rel=1e-12)
    # off the asymptotic range the ratio is NOT 1: (6/3)/(4 * 1.5/1.5) = 0.5
    assert est["asymptotic_ratio"] == pytest.approx(0.5, rel=1e-12)


def test_gci_rejects_zero_order():
    with pytest.raises(gs.GridStudyError):
        gs.gci(1.0, 2.0, 3.0, 0.0, 2.0)


# --------------------------------------------------------------------------
# the failure modes that matter for the column-collapse ladder
# --------------------------------------------------------------------------
def test_non_monotone_is_not_given_an_order():
    """The anchor's a=3 sequence (+8.1, +15.4, +11.1 %) is non-monotone."""
    r_out = [3.8110 * 1.0809, 3.7746 * 1.1540, 3.7895 * 1.1106]
    h = [1.0, 0.5, 0.25]
    got = gs.observed_order(h[0], h[1], h[2], r_out[0], r_out[1], r_out[2])
    assert got["p"] is None
    assert got["monotone"] is False
    assert "non-monotone" in got["reason"]


def test_assess_ladder_falls_back_to_band_on_non_monotone():
    levels = [(0.48 / 8, 4.1193), (0.48 / 12, 4.3559), (0.48 / 16, 4.2087)]
    out = gs.assess_ladder(levels, reference=3.7895, tol=0.15)
    assert out["method"] == gs.LADDER_BAND
    assert out["u_rel"] == pytest.approx(
        0.5 * (4.3559 - 4.1193) / 4.2087, rel=1e-12)
    assert out["error_finest"] == pytest.approx((4.2087 - 3.7895) / 3.7895, rel=1e-12)
    assert out["adjudication"] == "in_band"      # +11.1 % < 15 %


def test_non_uniform_ratio_is_rejected():
    got = gs.observed_order(1.0, 0.5, 0.3, 1.0, 1.2, 1.3)
    assert got["p"] is None
    assert "non-uniform" in got["reason"]


def test_equal_successive_changes_rejected():
    got = gs.observed_order(4.0, 2.0, 1.0, 1.0, 1.1, 1.2)
    assert got["p"] is None
    assert got["monotone"] is True
    assert "p -> 0" in got["reason"]


def test_two_levels_is_insufficient_and_indeterminate():
    out = gs.assess_ladder([(1.0, 2.10), (0.5, 2.40)], reference=2.0, tol=0.15)
    assert out["method"] == gs.LADDER_INSUFFICIENT
    assert out["u_rel"] is None
    assert out["error_finest"] == pytest.approx(0.20)
    assert out["adjudication"] == "indeterminate"
    assert "fewer than three" in out["reason"]


# --------------------------------------------------------------------------
# adjudication semantics
# --------------------------------------------------------------------------
def test_adjudicate_in_band():
    assert gs.adjudicate(0.05, 0.10, 0.15) == "in_band"
    assert gs.adjudicate(0.30, 0.10, 0.15) == "out_of_band"
    assert gs.adjudicate(0.30, 0.00, 0.15) == "out_of_band"


def test_adjudicate_miss_inside_uncertainty_is_indeterminate():
    # the miss is 15.4 % and the band is +-7.3 % -> 15.4 - 7.3 = 8.1 % < 15 %
    assert gs.adjudicate(0.154, 0.073, 0.15) == "indeterminate"


def test_adjudicate_boundary_is_inclusive():
    assert gs.adjudicate(0.15, 0.0, 0.15) == "in_band"
    assert gs.adjudicate(-0.20, 0.04, 0.15) == "out_of_band"
    assert gs.adjudicate(-0.18, 0.04, 0.15) == "indeterminate"


def test_adjudicate_nan_is_indeterminate():
    assert gs.adjudicate(float("nan"), 0.1, 0.15) == "indeterminate"


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def test_duplicate_level_rejected():
    with pytest.raises(gs.GridStudyError):
        gs.assess_ladder([(1.0, 1.0), (1.0, 2.0)])


def test_single_level_rejected():
    with pytest.raises(gs.GridStudyError):
        gs.assess_ladder([(1.0, 1.0)])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_bad_spacing_rejected(bad):
    with pytest.raises(gs.GridStudyError):
        gs.assess_ladder([(bad, 1.0), (1.0, 2.0)])


def test_non_finite_value_rejected():
    with pytest.raises(gs.GridStudyError):
        gs.assess_ladder([(2.0, float("nan")), (1.0, 2.0)])


def test_refinement_ratios_order_independent():
    assert gs.refinement_ratios([(1.0, 0.0), (4.0, 0.0)]) == pytest.approx([4.0])
    assert gs.refinement_ratios([(4.0, 0.0), (1.0, 0.0)]) == pytest.approx([4.0])


# --------------------------------------------------------------------------
# monotone ladder end-to-end: GCI method selected, verdict survives
# --------------------------------------------------------------------------
def test_monotone_ladder_selects_gci():
    """A clean p = 1 ladder converging onto a reference that is 5 % away."""
    levels = synthetic(p=1.0, f0=2.0, c=0.2, n=3)
    out = gs.assess_ladder(levels, reference=1.905, tol=0.15)
    assert out["method"] == gs.LADDER_GCI
    assert out["order"]["p"] == pytest.approx(1.0, rel=1e-9)
    # f_finest = 2.2, reference 1.905 -> +15.5 % signed error, but the GCI is
    # 1.25 * (0.2/2.2) / (2 - 1) = 0.1136 so the miss is inside the scatter.
    assert out["error_finest"] == pytest.approx((2.2 - 1.905) / 1.905, rel=1e-12)
    assert out["adjudication"] == "indeterminate"
    assert out["error_minus_u"] < 0.15

# --------------------------------------------------------------------------
# triple selection: the designed uniform triple must be found, not the first 3
# --------------------------------------------------------------------------
def test_uniform_triples_finds_designed_triple():
    # grains-per-radius 4, 5, 8, 12, 16 -> d = 0.12, 0.096, 0.06, 0.04, 0.03
    hs = [0.12, 0.096, 0.06, 0.04, 0.03]
    levels = [(h, 1.0 + h) for h in hs]
    trips = gs.uniform_triples(levels)
    assert (0, 2, 4) in trips            # 4/8/16 -> r = 2 exactly
    assert (0, 1, 2) not in trips        # 4/5/8 -> 1.25 vs 1.6


def test_assess_ladder_uses_designed_triple():
    """A clean p = 2 sequence sampled at 4/8/16 plus filler rungs still gives GCI."""
    hs = [0.12, 0.096, 0.06, 0.04, 0.03]
    levels = [(h, 2.0 + 3.0 * h ** 2) for h in hs]
    out = gs.assess_ladder(levels)
    assert out["method"] == gs.LADDER_GCI
    assert out["triple"]["indices"] == [0, 2, 4]
    assert out["order"]["p"] == pytest.approx(2.0, rel=1e-9)


def test_assess_ladder_no_uniform_triple_falls_back():
    levels = [(1.0, 3.0), (0.6, 2.6), (0.35, 2.5)]
    out = gs.assess_ladder(levels)
    assert out["method"] == gs.LADDER_BAND
    assert out["triple"] is None
    assert "no uniform-ratio triple" in out["reason"]
