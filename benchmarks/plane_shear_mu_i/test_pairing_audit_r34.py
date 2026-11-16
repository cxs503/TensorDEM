"""Round 34: unit + regression tests for the stress-estimator pairing audit.

These are *artifact* tests: they do not run a simulation (each point costs ~65 s
and is archived in ``runs/pairing_r34``), they pin the properties the round
claims.  The one exception is the reproduction test, which compares the round-34
artifact bit-for-bit with the archived round-33 point it re-ran.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parents[1]
for _p in (str(REPO / "src"), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pairing_audit_r34 as R34  # noqa: E402
import reference_2d  # noqa: E402

RUNS = BENCH_DIR / "runs" / "pairing_r34"
R33 = BENCH_DIR / "runs" / "wall_translation_r33" / "smoke_filled_n310_wfixed_I0.05_s0.json"


def _run(name: str) -> dict:
    f = RUNS / name
    if not f.exists():
        pytest.skip(f"archived artifact missing: {f.name}")
    return json.loads(f.read_text())


@pytest.fixture(scope="module")
def judged():
    f = RUNS / "judged_r34.json"
    if not f.exists():
        pytest.skip("judged_r34.json missing (run pairing_audit_r34.py --judge-only)")
    return json.loads(f.read_text())


# --------------------------------------------------------------------------- #
# 1. the abscissa definition
# --------------------------------------------------------------------------- #
def test_i_from_rate_is_the_ruler_abscissa_and_falls_with_pressure():
    """``I = gammadot d / sqrt(p/rho)``: monotone decreasing in p, linear in rate."""
    g, d, rho = 6.97, 0.01, 2500.0
    assert R34.i_from(6500.0, g, d, rho) == pytest.approx(
        g * d / math.sqrt(6500.0 / rho))
    assert R34.i_from(6500.0, g, d, rho) > R34.i_from(9000.0, g, d, rho)
    assert R34.i_from(6500.0, 2 * g, d, rho) == pytest.approx(
        2 * R34.i_from(6500.0, g, d, rho))
    # a degenerate channel must give nan, never a silently imputed number
    assert math.isnan(R34.i_from(0.0, g, d, rho))
    assert math.isnan(R34.i_from(float("nan"), g, d, rho))


# --------------------------------------------------------------------------- #
# 2. the archived round-33 point is reproduced exactly (no physics was touched)
# --------------------------------------------------------------------------- #
def test_r34_rerun_reproduces_the_archived_r33_point_bit_for_bit():
    if not R33.exists():
        pytest.skip("round-33 artifact missing")
    a = json.loads(R33.read_text())
    b = _run("point_I0.05_s0.json")
    assert a["config_digest"] == b["config_digest"]
    for key in ("mu_bulk", "mu_wall", "phi_gap", "p_wall_over_P",
                "sigma_yy_over_P", "I_measured"):
        assert a[key] == pytest.approx(b.get(key, b.get("profile_derived", {})
                                             .get(key)), rel=0, abs=0), key
    # the only thing round 34 changed is that it *keeps* the profile
    assert "profile" in b and "profile" not in a


# --------------------------------------------------------------------------- #
# 3. the pairings are what they claim to be
# --------------------------------------------------------------------------- #
def test_self_consistent_pairings_take_mu_and_pressure_from_one_channel():
    out = _run("point_I0.05_s0.json")
    led = R34.stress_ledger(out)
    rows = R34.pairings(led)
    assert {r["mu_channel"] for r in rows} == set(R34.CHANNELS)
    for r in rows:
        mu_src, p_src = led[r["mu_channel"]], led[r["p_channel"]]
        assert r["mu"] == pytest.approx(mu_src["mu"])
        assert r["p"] == pytest.approx(p_src["p"])
        # the abscissa must be built from the *pressure* channel of the pairing,
        # not from the ordinate's channel -- that is the whole point of the audit
        assert r["I"] == pytest.approx(
            R34.i_from(p_src["p"], led["gamma_dot_measured"],
                       led["diameter"], led["density"]))
        if r["self_consistent"]:
            assert r["mu_channel"] == r["p_channel"]
        # every row carries its own ruler value, never a shared one
        assert r["mu_ref"] == pytest.approx(reference_2d.mu_2d(r["I"]))


def test_legacy_pairing_reproduces_the_error_round_33_published():
    """R33's Table B printed +18.8 % for this run, computed by hand path.

    Round 34 must reproduce it from the artifact, or the audit is not judging the
    same quantity the previous rounds published.
    """
    if not R33.exists():
        pytest.skip("round-33 artifact missing")
    a = json.loads(R33.read_text())
    I = a["I_measured"]
    err = 100.0 * (a["mu_bulk"] - reference_2d.mu_2d(I)) / reference_2d.mu_2d(I)
    assert err == pytest.approx(18.8, abs=0.15)
    # and round 34's legacy row is the same pairing
    judged = _run("point_I0.05_s0.json")
    led = R34.stress_ledger(judged)
    rows = R34.pairings(led)
    legacy = [r for r in rows if r["legacy"]]
    assert len(legacy) == 1
    assert legacy[0]["mu_channel"] == "bulk"
    assert legacy[0]["p_channel"] == "profile_contact"


# --------------------------------------------------------------------------- #
# 4. the pre-registered findings stay recorded (no silent flip)
# --------------------------------------------------------------------------- #
def test_kinetic_stress_share_stays_small_so_contact_and_total_are_interchangeable(judged):
    """P1 of ``prereg_r34.json`` was falsified: the share is <= 2 %, not > 15 %."""
    shares = [j["ledger"]["kinetic_share_p"] for j in judged]
    assert shares and all(math.isfinite(s) for s in shares)
    assert max(shares) < 0.05


def test_pressure_channels_agree_within_five_percent_at_the_low_rate(judged):
    """P2 falsified: momentum balance holds to a few percent at I_target = 0.05."""
    for j in judged:
        if abs(j["I_target"] - 0.05) > 1e-12:
            continue
        ps = [j["ledger"][c]["p"] for c in R34.CHANNELS]
        assert max(ps) / min(ps) < 1.10, (j["I_target"], j["seed"], ps)


def test_realisation_scatter_dominates_the_channel_spread(judged):
    """The round's headline: the seed scatter is the largest term in the budget."""
    lo = [j for j in judged if abs(j["I_target"] - 0.05) < 1e-12]
    assert len(lo) >= 4, "the pre-registered ensemble was 4 seeds at I_target=0.05"
    errs = []
    for j in lo:
        r = next(r for r in j["pairings"]
                 if r["self_consistent"] and r["mu_channel"] == "bulk")
        errs.append(r["err_pct"])
    spread_seed = max(errs) - min(errs)
    spread_channel = max(j["spread_self_consistent_pp"] for j in lo)
    assert spread_seed > spread_channel, (spread_seed, spread_channel)
    assert spread_seed > 10.0, "scatter must exceed the anchor's tolerance to matter"


def test_high_rate_points_do_not_reach_their_nominal_rate(judged):
    """The I >= 0.1 end of the anchor's range is not rate-controlled in this device."""
    hi = [j for j in judged if abs(j["I_target"] - 0.10) < 1e-12]
    assert hi, "the pre-registered high-rate points are missing"
    for j in hi:
        assert j["ledger"]["gamma_dot_measured"] / j["ledger"]["gamma_dot_nominal"] < 0.25


# --------------------------------------------------------------------------- #
# 5. the report is generated, not typed
# --------------------------------------------------------------------------- #
def test_tables_and_figure_are_generated_from_the_artifacts(judged):
    prereg = json.loads((BENCH_DIR / "prereg_r34.json").read_text())
    md = R34.tables(judged, prereg)
    for section in ("Table A", "Table B", "Table C", "Table D",
                    "Table E", "Table F", "Table G"):
        assert section in md
    fig = R34.figure(judged, RUNS / "fig_pairing_r34_test.png")
    assert fig.exists() and fig.stat().st_size > 10000
    fig.unlink()


def test_preregistration_is_complete_and_predates_the_decision():
    prereg = json.loads((BENCH_DIR / "prereg_r34.json").read_text())
    assert prereg["registered_before_runs"] is True
    assert {"P1_kinetic_share_of_normal_stress_gt_0.15", "P2_pressure_channel_spread_gt_1.15",
            "P3_error_spread_gt_10pp", "P4_falsifier"} <= set(prereg["predictions"])
    assert prereg["decisions_fixed_in_advance"]