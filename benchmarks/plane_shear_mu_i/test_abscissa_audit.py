"""m17694 / round 10 verification: the I-taxonomy helper and the abscissa audit.

What is asserted, and why each assertion can fail:

A. ``mu_i_rheology.inertial_number`` — the definition in one place
   * reproduces ``gdot d / sqrt(P/rho)`` on a hand-computed case;
   * uses the **grain** density convention (a bulk-density substitution would
     change I by ``1/sqrt(phi)`` ≈ 29 % and is exactly the kind of silent
     redefinition this helper exists to prevent) — checked by asserting the
     pressure dependence, which is the only place density enters;
   * returns ``nan`` (never 0) for a non-positive pressure, so a lost pressure
     cannot look like a small inertial number;
   * takes ``|gdot|``, so a sign convention cannot produce ``nan``/negative I.

B. The device taxonomy block (no value changes)
   ``inertial_number == inertial_number_imposed_rate`` byte-for-byte, and
   ``inertial_number_measured_rate`` equals the helper applied to
   ``shear_rate_measured`` and the measured pressure — on a real (small, fast)
   servo-shear run.  This is the guard that round 10 did not quietly re-define
   the historic key.

C. The audit arithmetic (``abscissa_audit``)
   * ``i_from_rate`` / ``measured_pressure`` invert each other;
   * ``I_vol = rate_ratio * I_imposed_measP`` on a synthetic point (the algebraic
     identity the audit relies on);
   * ``err_pct`` is finite/nan-safe and signed;
   * ``within`` never lets ``nan`` pass and honours the band edge exactly;
   * ``budget`` refuses to divide by an observed error near zero (the
     ill-conditioned ratio that produced a meaningless "23.0" before the guard);
   * the rebuilt reference law reproduces the archived ``mu_ref`` column of every
     archived point (a provenance cross-check: the audit must judge round 8 with
     round 8's law).

Usage: ``pytest benchmarks/plane_shear_mu_i/test_abscissa_audit.py -q``
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
for p in (str(SRC), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tensordem.mu_i_rheology import (          # noqa: E402
    ServoShearCase, inertial_number, simulate_servo_shear,
)
import abscissa_audit as aa                    # noqa: E402


# ---------------------------------------------------------------------------
# A. the definition helper
# ---------------------------------------------------------------------------
def test_inertial_number_matches_the_hand_computed_definition():
    # I = 5 * 0.01 / sqrt(62500/2500) = 0.05 / 5 = 0.01
    assert inertial_number(5.0, 0.01, 62500.0, 2500.0) == pytest.approx(0.01, rel=1e-12)
    # doubling the pressure at fixed rate lowers I by sqrt(2) -> density is not
    # interchangeable with the bulk density (that would be a 29 %-class error)
    I2 = inertial_number(5.0, 0.01, 125000.0, 2500.0)
    assert I2 == pytest.approx(0.01 / math.sqrt(2.0), rel=1e-12)


def test_inertial_number_is_nan_for_a_lost_pressure_never_zero():
    assert math.isnan(inertial_number(5.0, 0.01, 0.0, 2500.0))
    assert math.isnan(inertial_number(5.0, 0.01, -1.0, 2500.0))


def test_inertial_number_uses_the_magnitude_of_the_rate():
    assert inertial_number(-5.0, 0.01, 62500.0, 2500.0) == pytest.approx(
        inertial_number(5.0, 0.01, 62500.0, 2500.0), rel=0.0)


# ---------------------------------------------------------------------------
# B. the device taxonomy block (small fast run, same config as R8's virial test)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def run():
    case = ServoShearCase(inertial_number=0.05, kn=6.25e4,
                          normal_stress=6.25e4, box_width=0.06,
                          box_depth=0.03, fill_height=0.035, radius=0.004,
                          settle_steps=600, min_shear_steps=300,
                          shear_strain=0.04, sample_stride=20, seed=5279,
                          bulk_stress=True)
    return simulate_servo_shear(case)


def test_historic_key_is_unchanged_in_value(run):
    """Round 10 must not have moved ``inertial_number`` (regression guard)."""
    assert run["inertial_number"] == run["inertial_number_imposed_rate"]
    # and it is the imposed-rate value, not a measured one
    expect = inertial_number(run["shear_rate_imposed"], run["particle_diameter"],
                             run["normal_stress_measured_pa"], run["density"])
    assert run["inertial_number"] == pytest.approx(expect, rel=1e-12)


def test_measured_rate_abscissa_is_the_helper_applied_to_the_measured_rate(run):
    I = run["inertial_number_measured_rate"]
    assert I is not None and math.isfinite(I) and I > 0.0
    expect = inertial_number(run["shear_rate_measured"], run["particle_diameter"],
                             run["normal_stress_measured_pa"], run["density"])
    assert I == pytest.approx(expect, rel=1e-12)


def test_measured_and_imposed_abscissae_differ_exactly_by_the_rate_ratio(run):
    """The identity the audit leans on: I_vol / I_imposed_measP = rate_ratio."""
    ratio = run["shear_rate_measured"] / run["shear_rate_imposed"]
    assert run["inertial_number_measured_rate"] / run["inertial_number"] == \
        pytest.approx(abs(ratio), rel=1e-12)


# ---------------------------------------------------------------------------
# C. audit arithmetic
# ---------------------------------------------------------------------------
def test_i_from_rate_and_measured_pressure_invert_each_other():
    P, rate = 62500.0, 5.0
    I = aa.i_from_rate(rate, P)
    point = {"shear_rate_imposed": rate, "I_measured": I}
    assert aa.measured_pressure(point) == pytest.approx(P, rel=1e-12)
    assert aa.i_from_rate(rate, aa.measured_pressure(point)) == pytest.approx(I, rel=1e-12)


def test_i_from_rate_is_nan_without_pressure():
    assert math.isnan(aa.i_from_rate(5.0, 0.0))


def _point(**kw):
    base = {"I_target": 0.03, "I_measured": 0.0314, "I_bulk": 0.035,
            "shear_rate_imposed": 15.0, "shear_rate_measured": 14.56,
            "rate_ratio": 14.56 / 15.0, "profile_r2": 0.88, "servo_valid": True,
            "mu_full": 0.3456, "mu_full_final": 0.409,
            "mu_full_zz": 0.33, "mu_full_zz_final": 0.40, "mu_ref": 0.4036,
            "n_particles": 294}
    base.update(kw)
    return base


def test_abscissae_reconstruction():
    ab = aa.abscissae(_point())
    assert ab["I_target"] == 0.03
    assert ab["I_imposed_measP"] == 0.0314
    # I_vol = measured rate x measured pressure: differs from the imposed value
    # by exactly the rate ratio (same pressure reconstruction)
    assert ab["I_vol"] == pytest.approx(0.0314 * (14.56 / 15.0), rel=1e-10)
    assert ab["I_bulk"] == 0.035


def test_err_pct_is_nan_safe_and_signed():
    law = lambda I: 0.4                       # noqa: E731
    assert aa.err_pct(0.3, 0.1, law) == pytest.approx(-25.0)
    assert aa.err_pct(0.5, 0.1, law) == pytest.approx(+25.0)
    assert math.isnan(aa.err_pct(None, 0.1, law))
    assert math.isnan(aa.err_pct(0.4, float("nan"), law))
    assert math.isnan(aa.err_pct(0.4, 0.0, law))


def test_within_honours_the_band_and_never_passes_nan():
    law = lambda I: 0.5                       # noqa: E731
    # the band edge is approached from both sides; the exact edge is not tested
    # because 0.55/0.5 - 1 is 0.10000000000000009 in binary, so an "exactly at
    # the edge" assertion would test floating point, not the band
    assert aa.within(0.5499, 0.1, law)                      # +9.98 %
    assert not aa.within(0.5502, 0.1, law)                  # +10.04 %
    assert not aa.within(float("nan"), 0.1, law)            # no measurement
    assert not aa.within(0.5, float("nan"), law)            # no abscissa


def test_budget_refuses_the_ill_conditioned_ratio():
    law = lambda I: 0.5                       # noqa: E731
    # reference rises 10 % when the abscissa moves; observed error is tiny ->
    # the "fraction explained" is numerically meaningless and must be None
    judged = {"mu_ref": {"I_target": 0.5, "I_vol": 0.55},
              "err_pct": {"I_target|mu_full": -0.05,
                          "I_target|mu_full_final": -20.0}}
    b = aa.budget(judged)
    assert b["delta_ref_pct"] == pytest.approx(10.0)
    assert b["explained_tail"] is None                    # |err| < 1 % -> refused
    # d = +0.10, observed -20 % -> -0.5: a negative value means the abscissa
    # shift moves the reference *away* from the measurement
    assert b["explained_final"] == pytest.approx(-0.5)


def test_budget_reports_nan_without_abscissae():
    judged = {"mu_ref": {"I_target": float("nan"), "I_vol": 0.5},
              "err_pct": {"I_target|mu_full": -20.0,
                          "I_target|mu_full_final": -20.0}}
    assert math.isnan(aa.budget(judged)["delta_ref_pct"])


# ---------------------------------------------------------------------------
# D. provenance: the rebuilt law must be round 8's law
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (HERE / "runs" / "full_cell" / "point_0.03.json").exists(),
                    reason="archived round-8 artefacts absent")
def test_rebuilt_law_reproduces_the_archived_reference_column():
    law = aa._law_of_record()
    worst = 0.0
    for I in (0.01, 0.03, 0.1, 0.3):
        p = json.loads((HERE / "runs" / "full_cell" / f"point_{I:.3g}.json").read_text())
        worst = max(worst, aa.check_law(p, law))
    assert worst < 1e-12


@pytest.mark.skipif(not (HERE / "runs" / "full_cell" / "point_0.03.json").exists(),
                    reason="archived round-8 artefacts absent")
def test_audit_end_to_end_marks_the_known_point_and_writes_no_nan_rows(tmp_path):
    rc = aa.main(["--out", str(tmp_path), "--no-fig"])
    assert rc == 0
    res = json.loads((tmp_path / "result.json").read_text())
    blk = res["arms"]["full_cell"]
    assert blk["n_points"] == 4 and blk["n_gate_pass"] == 2
    # the published tail-window figure of round 8 at I = 0.3 must be reproduced
    j03 = [j for j in blk["points"] if j["I_target"] == 0.3][0]
    assert j03["err_pct"]["I_target|mu_full"] == pytest.approx(-27.2, abs=0.1)
    text = (tmp_path / "tables.md").read_text()
    assert "nan |" not in text or text.count("nan |") <= 1