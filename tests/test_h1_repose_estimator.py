"""Ground-truth tests for the pile free-surface (repose angle) estimator.

The estimator is metrology, so it is tested against *synthetic piles with a known
slope* rather than against the simulation it is used on: a fit that cannot
recover a known cone angle cannot be trusted to accept or reject a DEM run.

Regression under test (H1 C2 / audit m15029): the shipped whole-profile least
squares fit averaged the flank together with the one-particle apron spread on
the floor and reported a known 30 deg cone as **16.05 deg** -- a 14 deg error
with no physics in it.  :func:`test_apron_does_not_flatten_the_flank` keeps the
naive fit as an explicit baseline and asserts that the flank fit removes that
error, which is what makes the test falsifiable rather than decorative.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.polyhedron_floe3d.pile_surface import (
    NO_PILE,
    bin_free_surface,
    fit_flank,
)

PARTICLE_RADIUS = 0.05
CONE_RADIUS = 0.30
APRON_HEIGHT = 0.05  # centres of a one-particle-thick apron, one radius up


def _synthetic_pile(angle_deg: float, n_bodies: int, *, apron_share: float = 0.0,
                    radius: float = CONE_RADIUS, seed: int = 3) -> dict:
    """Bodies lying on a cone of known slope, optionally with a flat apron.

    ``apron_share`` is the fraction of bodies that sit in a one-particle layer on
    the floor, spread over a disc 1.6x the pile radius -- the foot that every
    poured or collapsed pile carries.
    """
    rng = np.random.default_rng(seed)
    slope = math.tan(math.radians(angle_deg))
    n_apron = int(round(n_bodies * apron_share))
    xy, top = [], []
    for _ in range(n_bodies - n_apron):
        r = radius * math.sqrt(rng.random())
        theta = rng.uniform(0.0, 2.0 * math.pi)
        xy.append((r * math.cos(theta), r * math.sin(theta)))
        top.append((radius - r) * slope + PARTICLE_RADIUS)  # top of the body
    for _ in range(n_apron):
        r = 1.6 * radius * math.sqrt(rng.random())
        theta = rng.uniform(0.0, 2.0 * math.pi)
        xy.append((r * math.cos(theta), r * math.sin(theta)))
        top.append(PARTICLE_RADIUS)
    xy = np.asarray(xy)
    return bin_free_surface(np.hypot(xy[:, 0], xy[:, 1]), np.asarray(top))


def _naive_whole_profile_angle(profile: dict) -> float:
    """The pre-fix estimator: one least-squares line through every annulus."""
    slope, _ = np.polyfit(np.asarray(profile["radius"]), np.asarray(profile["height"]), 1)
    return math.degrees(math.atan(max(-slope, 0.0)))


@pytest.mark.parametrize("angle_deg", [20.0, 30.0, 40.0])
def test_recovers_known_cone_angle(angle_deg):
    """A clean cone is recovered to within 1 degree (relief >= 3 diameters)."""
    profile = _synthetic_pile(angle_deg, 400)
    fit = fit_flank(profile["radius"], profile["height"], profile["count"])
    assert fit["reason"] == "ok"
    assert fit["n_flank_bins"] >= 5
    assert fit["angle_deg"] == pytest.approx(angle_deg, abs=1.0)


@pytest.mark.parametrize("angle_deg", [30.0, 40.0])
def test_apron_does_not_flatten_the_flank(angle_deg):
    """A one-particle apron must not be averaged into the flank angle.

    Baseline (still computed): the naive whole-profile fit loses >5 deg.  The
    flank fit recovers the cone to within 3 deg.  Both directions are asserted,
    so the test fails if someone reverts to a whole-profile fit *and* if the
    flank fit starts chasing the apron.
    """
    profile = _synthetic_pile(angle_deg, 400, apron_share=0.4)
    naive = _naive_whole_profile_angle(profile)
    assert angle_deg - naive > 5.0, "apron baseline weakened: naive fit = %.2f" % naive
    fit = fit_flank(profile["radius"], profile["height"], profile["count"])
    assert fit["reason"] == "ok"
    assert abs(fit["angle_deg"] - angle_deg) <= 3.0
    # the apron is reported, not hidden
    assert fit["apron_body_share"] > 0.0


def test_shallow_pile_apron_error_is_bounded_and_declared():
    """A 20 deg pile is only ~2 bodies high: the declared error budget is 4 deg."""
    profile = _synthetic_pile(20.0, 400, apron_share=0.4)
    fit = fit_flank(profile["radius"], profile["height"], profile["count"])
    assert abs(fit["angle_deg"] - 20.0) <= 4.0
    assert _naive_whole_profile_angle(profile) < 18.0  # baseline is worse than that


def test_apron_is_excluded_from_the_fitted_flank():
    """The toe must fall inside the cone, not out in the apron."""
    profile = _synthetic_pile(35.0, 400, apron_share=0.4)
    fit = fit_flank(profile["radius"], profile["height"], profile["count"])
    assert fit["toe_radius_m"] < 1.05 * CONE_RADIUS
    assert fit["apex_radius_m"] < fit["toe_radius_m"]
    assert fit["flank_height_drop_m"] > 0.0


@pytest.mark.parametrize("angle_deg", [30.0, 40.0, 45.0])
def test_clean_cone_is_not_cut_into_a_shallow_flank(angle_deg):
    """A clean cone has no apron, so no break may be carved out of it.

    Regression: the BIC break search was free to place the break anywhere, so on
    a clean cone it could keep a short near-apex segment as the "flank" and dump
    the steeper outer part into the "apron" -- a 40 deg cone was reported as
    35.10 deg (5 deg of pure metrology error, and the *inverted* structural
    signature of an apron, which is flatter than the flank).  The admissibility
    rule rejects a split whose outward segment descends faster than the flank;
    ``require_flatter_apron=False`` keeps the pre-fix behaviour as the baseline
    (seed 7, the seed the committed evidence uses: at 400 bodies the defect is
    profile-dependent, so the contrast is pinned to the documented profile).
    """
    profile = _synthetic_pile(angle_deg, 400, seed=7)
    pre_fix = fit_flank(profile["radius"], profile["height"], profile["count"],
                        min_drop=PARTICLE_RADIUS, require_flatter_apron=False)
    fit = fit_flank(profile["radius"], profile["height"], profile["count"],
                    min_drop=PARTICLE_RADIUS)
    assert fit["reason"] == "ok"
    assert abs(fit["angle_deg"] - angle_deg) <= 1.5
    # baseline: the pre-fix split either lost > 2 deg or refused the angle
    # outright (the short apex segment it carved out had under one particle of
    # relief, so the relief gate fired on a clean cone -- both are defects)
    pre_fix_angle = pre_fix["angle_deg"]
    assert (not math.isfinite(pre_fix_angle)) or (angle_deg - pre_fix_angle > 2.0), (
        "baseline weakened: pre-fix fit = %r deg" % pre_fix_angle)
    # no admissible split -> the flank reaches the outermost annulus
    assert fit["toe_radius_m"] == pytest.approx(max(profile["radius"]))


def test_steeper_outboard_segment_is_never_reported_as_an_apron():
    """Structural signature check: an apron is flatter than the flank, always."""
    for angle_deg in (25.0, 35.0, 45.0):
        for share in (0.0, 0.2, 0.4):
            profile = _synthetic_pile(angle_deg, 400, apron_share=share)
            fit = fit_flank(profile["radius"], profile["height"], profile["count"],
                            min_drop=PARTICLE_RADIUS)
            if not math.isfinite(fit["angle_deg"]) or not math.isfinite(fit["apron_angle_deg"]):
                continue
            assert fit["apron_angle_deg"] <= fit["angle_deg"] + 1e-9, (angle_deg, share)


def test_monolayer_is_reported_as_no_pile_not_as_zero_degrees():
    """A dispersed monolayer has no flank: nan + reason, never a 0 deg angle.

    Regression for the old estimator, which turned a flat spread of floes into
    a "measured" repose angle of 0 deg.  The relief gate is the particle radius:
    a heap lower than one particle is not a heap.
    """
    profile = _synthetic_pile(30.0, 200, apron_share=1.0)
    fit = fit_flank(profile["radius"], profile["height"], profile["count"],
                    min_drop=PARTICLE_RADIUS)
    assert math.isnan(fit["angle_deg"])
    assert fit["reason"] == NO_PILE
    assert fit["n_flank_bins"] == 0
    # the naive fit would still have produced a number
    assert math.isfinite(_naive_whole_profile_angle(profile))


def test_too_few_bins_reports_no_pile():
    """Two annuli cannot define a slope: refuse instead of extrapolating."""
    fit = fit_flank([0.05, 0.15], [0.40, 0.25])
    assert math.isnan(fit["angle_deg"])
    assert fit["reason"] == NO_PILE


def test_angle_ordering_tracks_the_true_slope():
    """Recovered angles stay ordered when the true slope is ordered."""
    angles = [fit_flank(*(lambda p: (p["radius"], p["height"], p["count"]))(
        _synthetic_pile(a, 200, apron_share=0.4)))["angle_deg"] for a in (20.0, 30.0, 40.0)]
    assert angles[0] < angles[1] < angles[2]


def test_bin_profile_ignores_far_flung_outliers():
    """One body flung to 3 R must not become a data point of the profile."""
    rng = np.random.default_rng(11)
    radii = np.concatenate([rng.uniform(0.0, 0.30, 300), [0.90]])
    tops = np.concatenate([np.full(300, 0.20), [0.05]])
    profile = bin_free_surface(radii, tops)
    # the outlier sits alone in an outer annulus -> that annulus is dropped
    assert min(profile["height"]) > 0.15
    assert max(profile["radius"]) < 0.6 * 0.90