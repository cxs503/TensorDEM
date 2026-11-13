"""Tests for the floor-clearance ladder (mission m18243, CASE-PENETRATION R4).

Everything here is sub-second: the geometry accounting, the window ladder and the
clearance extrapolation are pure functions over arrays.  The one simulation-based
test reuses the *cached* ladder records and is skipped (not silently passed) when
they are absent, the same convention as ``test_cached_case_meets_acceptance``.
"""
import json
import math
import os

import numpy as np
import pytest

from tensordem.penetration_quasistatic import (
    G,
    REFERENCE,
    PenetrationConfig,
    brzinski_shape,
    preset,
)

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from penetration_deep_bed import (  # noqa: E402
    BOTTOM_MARGIN_OVER_D,
    SEMI_INFINITE_CLEARANCE_OVER_R,
    additive_verdict,
    bed_depth_for_clearance,
    build_config,
    fit_clearance_limit,
    floor_clearance_over_r,
    line_diagnostic,
    protocol_audit,
    static_audit,
    window_ladder,
)

ARTIFACT_DIR = os.path.join(ROOT, "artifacts", "case_penetration", "deep_bed")


def _synthetic_profile(alpha_true, radius=0.012, mu=0.40, rho_bulk=1490.0,
                       offset=0.0, z_max_over_r=1.6, z_min_over_r=0.02, n=400):
    z = np.linspace(z_min_over_r * radius, z_max_over_r * radius, n)
    S = mu * rho_bulk * G * brzinski_shape(z, radius)
    return z, alpha_true * S + offset


# ---------------------------------------------------------------------------
# 1. Geometry accounting
# ---------------------------------------------------------------------------

def test_floor_clearance_is_measured_from_surface_and_depth():
    # surface at 32 mm, deepest point 19.2 mm below it, R = 12 mm -> 1.07 R
    assert floor_clearance_over_r(0.032, 0.0192, 0.012) == pytest.approx(1.0667, rel=1e-3)
    # a deeper bed leaves proportionally more room under the intruder
    assert floor_clearance_over_r(0.060, 0.0192, 0.012) > floor_clearance_over_r(0.032, 0.0192, 0.012)
    with pytest.raises(ValueError):
        floor_clearance_over_r(0.032, 0.0192, 0.0)


def test_bed_depth_inversion_round_trips():
    R, d = 0.012, 0.004
    for clr in (2.0, 3.0, 4.5):
        depth = bed_depth_for_clearance(clr, R, 1.6, d, bottom_margin_over_d=0.0)
        # inverted from the *initial* fill height (no settlement, no margin)
        assert floor_clearance_over_r(depth, 1.6 * R, R) == pytest.approx(clr, rel=1e-9)
    # default margin is one grain diameter
    assert bed_depth_for_clearance(3.0, R, 1.6, d) == pytest.approx(
        (1.6 + 3.0) * R + BOTTOM_MARGIN_OVER_D * d)


def test_build_config_accepts_only_known_fields_and_rejects_thin_beds():
    cfg = build_config(3.0, rd=3.0, wall=4.0)
    assert cfg.intruder_radius_over_d == 3.0 and cfg.wall_half_width_over_R == 4.0
    assert cfg.bed_depth_over_d == pytest.approx(
        (1.6 + 3.0) * 3.0 + BOTTOM_MARGIN_OVER_D, rel=1e-9)   # (penetrate+clearance) R/d + margin
    with pytest.raises(SystemExit):
        build_config(3.0, not_a_field=1.0)
    # a negative clearance cannot be built at all
    with pytest.raises(ValueError):
        build_config(-0.5)


def test_shipped_presets_violate_the_semi_infinite_premise():
    """The defect this round fixes: every shipped preset is a thin bed."""
    audit = static_audit()
    assert audit["n_presets"] == 7
    assert audit["n_violating"] == 7
    clearances = [r["nominal_clearance_over_r"] for r in audit["presets"]]
    assert 0.5 < min(clearances) < 1.0 and 1.5 < max(clearances) < 2.5
    for row in audit["presets"]:
        assert row["semi_infinite_premise"] is False
        assert row["nominal_clearance_over_r"] < SEMI_INFINITE_CLEARANCE_OVER_R


def test_protocol_audit_warns_on_thin_bed_and_names_the_mechanism():
    cfg = PenetrationConfig()                       # shipped base: clearance 1.07 R
    audit = protocol_audit(z_surface=cfg.bed_depth, max_depth=1.6 * cfg.R, cfg=cfg)
    assert audit["floor_clearance_over_r"] == pytest.approx(1.0667, rel=1e-3)
    assert audit["semi_infinite_premise"] is False
    assert any("confined compression" in w for w in audit["warnings"])
    two = protocol_audit(z_surface=0.100, max_depth=1.6 * cfg.R, cfg=cfg)
    assert two["semi_infinite_premise"] is True
    assert not any("confined compression" in w for w in two["warnings"])


# ---------------------------------------------------------------------------
# 2. Window ladder / additive diagnostic
# ---------------------------------------------------------------------------

def test_window_ladder_recovers_a_pure_drag_law_without_a_tare():
    z, F = _synthetic_profile(REFERENCE["alpha"])
    ladder = window_ladder(z, F, 0.012, 0.40, 1490.0)
    assert len(ladder) >= 4
    for row in ladder:
        assert row["alpha"] == pytest.approx(REFERENCE["alpha"], rel=0.05)
        # a pure drag law needs no constant force once the window is fully on the
        # linear branch (windows straddling z = R mix the cubic Eq.(3c) in)
        if row["deep_branch"]:
            assert abs(row["tare_N"]) < 1e-9
            assert row["alpha_free"] == pytest.approx(REFERENCE["alpha"], rel=1e-6)
    assert additive_verdict(ladder)["verdict"] == "additive"


def test_line_diagnostic_separates_slope_from_a_constant_tare():
    """A constant tare must move the intercept, never the slope-derived alpha."""
    # deep branch only: Eq. (3d) is exactly linear there, so the split is exact
    z, F = _synthetic_profile(REFERENCE["alpha"], offset=0.5, z_min_over_r=1.0)
    pure = line_diagnostic(z, F, 0.012, 0.40, 1490.0)
    assert pure["tare_N"] == pytest.approx(0.5, abs=1e-9)
    assert pure["alpha_free"] == pytest.approx(REFERENCE["alpha"], rel=1e-9)
    assert pure["rms_N"] < 1e-9
    # the depth-independent check: the intercept is the theory value + tare
    assert pure["intercept_N"] == pytest.approx(
        pure["theory_intercept_N"] + 0.5, abs=1e-9)


def test_window_ladder_attributes_the_excess_to_an_additive_tare():
    """Constrained fit biased high, slope exactly right -> verdict ``additive``."""
    z, F = _synthetic_profile(REFERENCE["alpha"], offset=0.5)
    ladder = window_ladder(z, F, 0.012, 0.40, 1490.0)
    verdict = additive_verdict(ladder)
    assert verdict["verdict"] == "additive"
    assert verdict["tare_median_N"] == pytest.approx(0.5, abs=1e-6)
    assert verdict["rel_error_alpha_free"] < 1e-9
    deep = [r for r in ladder if r["deep_branch"]]
    assert deep, "the default windows must include fully-deep-branch windows"
    # the one-parameter estimator absorbs the tare and reports an alpha too high
    assert all(r["alpha"] > REFERENCE["alpha"] for r in deep)


def test_window_ladder_flags_a_multiplicative_error():
    """A wrong drag coefficient must be reported as multiplicative, not a tare."""
    z, F = _synthetic_profile(3.0 * REFERENCE["alpha"])
    ladder = window_ladder(z, F, 0.012, 0.40, 1490.0)
    verdict = additive_verdict(ladder)
    assert verdict["verdict"] == "multiplicative"
    assert verdict["rel_error_alpha_free"] > 1.0
    assert abs(verdict["tare_median_N"]) < 1e-6


def test_clearance_limit_extrapolates_a_known_thin_bed_correction():
    # alpha(c) = 35 + 60/c -> with c = 1, 2, 3, 4.5 the intercept must be 35
    c = np.array([1.0, 2.0, 3.0, 4.5])
    a = 35.0 + 60.0 / c
    out = fit_clearance_limit(c, a)
    assert out["alpha_inf"] == pytest.approx(35.0, rel=1e-6)
    assert out["slope_k"] == pytest.approx(60.0, rel=1e-6)
    assert out["rms"] < 1e-6 and out["stderr_intercept"] < 1e-6
    with pytest.raises(ValueError):
        fit_clearance_limit([1.0, 2.0], [95.0, 65.0])


# ---------------------------------------------------------------------------
# 3. Cached ladder records (skipped when absent -- never silently passed)
# ---------------------------------------------------------------------------

def _record_paths():
    if not os.path.isdir(ARTIFACT_DIR):
        return []
    return sorted(os.path.join(ARTIFACT_DIR, d, "result.json")
                  for d in os.listdir(ARTIFACT_DIR)
                  if os.path.exists(os.path.join(ARTIFACT_DIR, d, "result.json")))


@pytest.mark.skipif(not _record_paths(), reason="deep-bed records not built")
def test_cached_ladder_records_are_self_consistent():
    paths = _record_paths()
    assert len(paths) >= 3
    clearances, alphas = [], []
    for p in paths:
        with open(p) as fh:
            rec = json.load(fh)
        audit = rec["protocol_audit"]
        # provenance: measured geometry, dimensionless protocol numbers, ladder
        assert audit["floor_clearance_over_r"] > 0.0
        assert rec["alpha_ladder"], p
        assert rec["additive_verdict"]["verdict"] in ("additive", "multiplicative",
                                                      "undetermined")
        for row in rec["alpha_ladder"]:
            assert math.isfinite(row["tare_N"]) and math.isfinite(row["alpha_free"])
        assert rec["bed"]["settle_contacts"] > 0
        # every record states its own acceptance verdict consistently
        assert rec["status"] == ("PASS" if not rec["failed_criteria"] else "FAIL")
        clearances.append(audit["floor_clearance_over_r"])
        alphas.append(rec["metrics"]["alpha_fit"])
    # the ladder really spans the semi-infinite threshold
    assert max(clearances) >= SEMI_INFINITE_CLEARANCE_OVER_R
    assert min(clearances) < 2.0
    # and alpha must *decrease* with clearance if the floor was the culprit
    order = np.argsort(clearances)
    a = np.asarray(alphas)[order]
    assert a[0] > a[-1]