"""Regression guard for the Hertz contract verification (docs/HERTZ_VERIFY.md).

Locks three things measured by ``scripts/verify_hertz_contract.py``:
  1. the rigid-counterpart reduced constants are exactly E/(1-nu^2) and R;
  2. the sphere-sphere pair magnitude currently sits at sqrt(2) x the module's own
     closed form (the shipped behaviour, an OPEN finding - when the source is fixed
     the expected value here becomes 1.0 and this test must be updated in the same
     commit);
  3. the realised restitution of the shipped Hertz pair law is the reported mapping
     bias, not the requested e.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "verify_hertz_contract", ROOT / "scripts" / "verify_hertz_contract.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

torch = pytest.importorskip("torch")


def test_rigid_counterpart_constants_are_closed_form():
    r = V.q2_rigid_constants()
    assert abs(r["e_star_relative_error"]) < 1e-15
    assert abs(r["r_star_relative_error"]) < 1e-15
    assert r["e_star_rigid_Pa"] == pytest.approx(V.E_MOD / (1.0 - V.NU ** 2), rel=1e-15)
    # sphere-sphere E* is exactly half of the rigid one
    assert r["e_star_sphere_sphere_ratio"] == pytest.approx(0.5, rel=1e-15)


def test_static_pair_force_offset_is_sqrt_two():
    """Documents the shipped 41.42 % offset; NOT an endorsement of it."""
    q1 = V.q1_static(fracs=(0.01, 0.1))
    ratio = [r["force_measured_N"] / r["force_hertz_N"] for r in q1["rows"]]
    for value in ratio:
        assert value == pytest.approx(math.sqrt(2.0), rel=1e-12)
    # slope check: F ~ delta^1.5 to round-off
    f = [r["force_measured_N"] for r in q1["rows"]]
    slope = math.log(f[1] / f[0]) / math.log(q1["rows"][1]["delta_m"] / q1["rows"][0]["delta_m"])
    assert slope == pytest.approx(1.5, abs=1e-9)


def test_restitution_mapping_bias_is_reproduced():
    ref = V.q3_restitution(e_targets=(0.9, 0.5), v0s=(0.05, 0.5))
    by_key = {(r["e_target"], r["v_in_m_s"]): r for r in ref["rows"]}
    # e=0.9: inside the 5 % bar but not zero
    assert by_key[(0.9, 0.5)]["e_measured"] == pytest.approx(0.91178, abs=5e-4)
    assert 1.0 < by_key[(0.9, 0.5)]["relative_error_percent"] < 2.0
    # e=0.5: breaks the 5 % bar by a factor > 3
    assert by_key[(0.5, 0.5)]["e_measured"] == pytest.approx(0.58254, abs=5e-4)
    assert by_key[(0.5, 0.5)]["relative_error_percent"] > 16.0
    # amplitude self-similarity: the bias does not depend on impact speed
    for e_t in (0.9, 0.5):
        lo, hi = by_key[(e_t, 0.05)]["e_measured"], by_key[(e_t, 0.5)]["e_measured"]
        assert abs(hi - lo) / lo < 1e-3
