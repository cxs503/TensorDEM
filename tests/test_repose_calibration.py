"""Tests for the DEM friction -> repose-angle calibration surface (m4970).

Coverage:
* analytic anchor A1: irrotational slider onset at ``arctan(mu)``;
* analytic anchor A2: rolling sphere onset at ``arctan(min(mu, mu_r))``;
* negative control: with ``mu_r = 0`` ideal spheres cannot build a heap;
* monotonicity of the simulated ``theta_r(mu)`` curve;
* PCHIP inverse round trip and off-node recovery accuracy (< 2 % on ``mu``);
* rejection of extrapolation above and below the calibrated bracket;
* rejection of non-monotone tables;
* the shipped calibration artifact must satisfy the same criteria.

The heavy forward runs use a reduced case (small column, short lift) so the file
stays inside the routine test budget; the full-size table is produced by
``scripts/calibrate_repose_dem3d.py``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import tensordem.repose_calibration as rc

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "repose_calibration" \
    / "repose_calibration.json"

# Reduced column-collapse case used by the tests (documented in the report).
# ~80 spheres: small enough for the routine test budget, large enough that the
# flank of the deposit is resolved by ~8 radial bins.
TEST_CASE = dict(radius=0.008, kn=6.0e4, cylinder_radius=0.032, fill_height=0.13,
                 packing_gap=0.01, jitter=0.002, hold_time=0.03, lift_time=0.30,
                 settle_time=0.40, restitution=0.05)

_CACHE: dict[tuple[float, float], dict] = {}


def _sim(mu: float, mu_rolling: float = 0.2) -> dict:
    """Cached forward run so several tests share the same expensive piles.

    The powder solver works on small tensors, for which the default BLAS thread
    count is counter-productive (thread launch dominates); one thread is ~7x
    faster here.  The previous setting is restored afterwards.
    """
    key = (round(mu, 6), round(mu_rolling, 6))
    if key not in _CACHE:
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            _CACHE[key] = rc.simulate(rc.ReposeCase(mu=mu, mu_rolling=mu_rolling,
                                                    **TEST_CASE))
        finally:
            torch.set_num_threads(threads)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# Analytic anchors
# ---------------------------------------------------------------------------

def test_irrotational_slider_onset_matches_arctan_mu():
    """Anchor A1: a sphere with frozen rotation slides at exactly arctan(mu).

    The bisection detects "slid" from a displacement threshold, so the measured
    onset carries a one-sided detection lag (the driving acceleration vanishes at
    the onset, so the displacement grows only quadratically there).  With
    ``duration = 0.2 s`` and ``slide_fraction = 0.25`` the lag is < 1 deg; it is
    asserted one-sided and reported in ``docs/repose_angle_calibration.md``.
    """
    for mu in (0.25, 0.6):
        result = rc.sliding_onset_angle(mu, radius=0.006, kn=3.0e4,
                                        restitution=0.15, duration=0.2,
                                        slide_fraction=0.25)
        assert -0.3 < result["error_deg"] < 1.2, result


def test_rolling_sphere_onset_matches_min_of_friction_coefficients():
    """Anchor A2: free sphere moves at arctan(min(mu, mu_r)); below it, static.

    Sliding equilibrium needs ``tan theta <= mu`` and rolling equilibrium needs
    ``tan theta <= mu_r``, so the sphere starts to move when the tilt exceeds
    ``arctan(min(mu, mu_r))``.  Checked from both sides with the same driver.
    """
    mu, mu_r = 0.6, 0.2
    critical = math.degrees(math.atan(min(mu, mu_r)))
    below = rc.sliding_displacement(mu, 0.8 * critical, radius=0.006, kn=3.0e4,
                                    restitution=0.15, duration=0.08,
                                    allow_rotation=True, mu_rolling=mu_r)
    above = rc.sliding_displacement(mu, 1.4 * critical, radius=0.006, kn=3.0e4,
                                    restitution=0.15, duration=0.08,
                                    allow_rotation=True, mu_rolling=mu_r)
    assert below["displacement"] < 1.0e-5, below
    assert above["displacement"] > 20.0 * max(below["displacement"], 1.0e-7), above


def test_zero_rolling_resistance_flattens_the_heap():
    """Negative control: with mu_r = 0 ideal spheres roll out of the pile.

    Same friction coefficient and same protocol, only the rolling resistance is
    removed: the deposit must be markedly flatter.  This proves that the rolling
    degrees of freedom are active in the powder solver (a translational-only
    friction law would leave the same heap in both runs).
    """
    with_rolling = _sim(0.60, mu_rolling=0.2)
    without = _sim(0.60, mu_rolling=0.0)
    assert with_rolling["theta_r_deg"] > 3.0, with_rolling
    assert without["theta_r_deg"] < with_rolling["theta_r_deg"] - 3.0, (with_rolling, without)
    assert without["apex_z"] < with_rolling["apex_z"], (with_rolling, without)


# ---------------------------------------------------------------------------
# Forward curve
# ---------------------------------------------------------------------------

def test_repose_angle_increases_with_sliding_friction():
    """theta_r(mu) must increase between two well-separated friction values."""
    low = _sim(0.35)
    high = _sim(0.70)
    assert low["theta_r_deg"] > 0.0, low
    assert high["theta_r_deg"] > low["theta_r_deg"] + 2.0, (low, high)
    for out in (low, high):
        # cohesionless kinematic bound: tan(theta_r) <= mu
        bound = rc.kinematic_bound_margin(out["theta_r_deg"], out["mu"])
        assert bound["satisfied"], (out, bound)


# ---------------------------------------------------------------------------
# Calibration table and inverse lookup
# ---------------------------------------------------------------------------

def _smooth_reference(mu: float) -> float:
    """Documented smooth forward map used to score the interpolation machinery."""
    return 3.0 + 34.0 * mu / (0.35 + mu)


def _reference_table(nodes=11) -> rc.ReposeCalibrationTable:
    mus = [0.10 + 0.05 * i for i in range(nodes)]
    return rc.ReposeCalibrationTable(
        mu=tuple(mus), theta_deg=tuple(_smooth_reference(m) for m in mus),
        mu_rolling=0.2, provenance={"kind": "synthetic-reference"})


def test_inverse_round_trip_and_off_node_recovery_below_two_percent():
    """Inverse lookup recovers off-node mu to better than 2 %."""
    table = _reference_table()
    for mu in table.mu:                      # node recovery is exact
        assert table.mu_from_theta(table.theta_from_mu(mu)) == pytest.approx(mu, rel=1e-9)
    worst = 0.0
    for mu in (0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525):
        theta = _smooth_reference(mu)
        recovered = table.mu_from_theta(theta)
        worst = max(worst, abs(recovered - mu) / mu)
    assert worst < 0.02, worst
    # Monotone PCHIP must not overshoot the node values.
    for mu in (0.125, 0.225, 0.325, 0.425, 0.525):
        theta = table.theta_from_mu(mu)
        assert table.theta_deg[0] <= theta <= table.theta_deg[-1]


def test_inverse_rejects_extrapolation_outside_bracket():
    """Negative case: queries outside the calibrated bracket are refused."""
    table = _reference_table()
    mu_min, mu_max = table.mu[0], table.mu[-1]
    theta_min, theta_max = table.theta_deg[0], table.theta_deg[-1]
    with pytest.raises(rc.ReposeExtrapolationError):
        table.mu_from_theta(theta_max + 1.0)
    with pytest.raises(rc.ReposeExtrapolationError):
        table.mu_from_theta(theta_min - 1.0)
    with pytest.raises(rc.ReposeExtrapolationError):
        table.theta_from_mu(mu_max + 0.01)
    with pytest.raises(rc.ReposeExtrapolationError):
        table.theta_from_mu(mu_min - 0.01)
    # exactly on the bracket edge remains admissible
    assert table.mu_from_theta(theta_max) == pytest.approx(mu_max, rel=1e-6)
    assert table.mu_from_theta(theta_min) == pytest.approx(mu_min, rel=1e-6)


def test_non_monotone_table_is_rejected():
    """Negative case: a non-monotone calibration table cannot be constructed."""
    with pytest.raises(rc.NonMonotoneTableError):
        rc.ReposeCalibrationTable(mu=(0.2, 0.3, 0.4), theta_deg=(10.0, 9.5, 12.0))
    with pytest.raises(rc.ReposeCalibrationError):
        rc.ReposeCalibrationTable(mu=(0.2, 0.3, 0.3), theta_deg=(10.0, 11.0, 12.0))


# ---------------------------------------------------------------------------
# Shipped artifact
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ARTIFACT.exists(), reason="calibration artifact not generated")
def test_shipped_artifact_is_self_consistent():
    """The exported calibration artifact must be internally consistent.

    This does *not* re-assert the 2 % end-to-end recovery target: the artifact
    records its own measured value and verdict (``inverse_recovery_under_2pct``),
    and the reported error must equal the exported per-run numbers — an honest
    FAIL must stay visible instead of being hidden by a relaxed assertion.
    """
    payload = json.loads(ARTIFACT.read_text())
    table = rc.ReposeCalibrationTable.from_json(payload["table"])
    assert all(b > a for a, b in zip(table.theta_deg, table.theta_deg[1:]))
    verdict = payload["verdict"]
    scored = [row for row in payload["inverse_recovery"] if row["rel_error"] is not None]
    worst = max((abs(row["rel_error"]) for row in scored), default=float("inf"))
    assert verdict["worst_inverse_rel_error"] == pytest.approx(worst, rel=1e-9)
    assert verdict["inverse_recovery_under_2pct"] == (worst < 0.02)
    # every non-recovered row must carry an explicit status, never a silent value
    for row in payload["inverse_recovery"]:
        assert row["status"] in ("recovered", "rejected_outside_bracket")
        if row["status"] == "rejected_outside_bracket":
            assert row["mu_recovered"] is None
    # every tabulated point honours the cohesionless kinematic bound
    for mu, theta in zip(table.mu, table.theta_deg):
        assert rc.kinematic_bound_margin(theta, mu)["satisfied"], (mu, theta)
    # the exclusion policy is recorded (monotone region only)
    assert verdict["n_table_nodes"] + verdict["n_excluded_nodes"] == len(payload["node_rows"])