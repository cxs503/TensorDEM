"""Tests for the seed-resolved temporal-bias adjudication of the RCP case.

The statistics are exercised on synthetic, closed-form examples so the tests do
not depend on any cluster run being present (the run set lives outside the
package and changes as the campaign grows).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "benchmarks", "case_rcp_deposition"))

from seed_bias_audit import (classify_bias, levels_present, load_runs,
                             seeds_needed, welch_diff)


def test_welch_diff_matches_hand_computation():
    a = np.array([0.640, 0.641])          # coarse, n = 2
    b = np.array([0.6395, 0.6405])        # fine
    w = welch_diff(a, b)
    assert w["n_coarse"] == 2 and w["n_fine"] == 2
    assert w["delta"] == pytest.approx(0.6400 - 0.6405)
    va = np.var(a, ddof=1) / 2
    vb = np.var(b, ddof=1) / 2
    assert w["sem"] == pytest.approx(math.sqrt(va + vb))
    # equal n and equal variance -> Welch df collapses to 2(n-1)
    assert w["df"] == pytest.approx(2.0, rel=1e-12) or w["df"] == pytest.approx(2.0)


def test_welch_diff_single_seed_falls_back_to_one_sided_scatter():
    # one level has a single seed -> sigma is borrowed from the other level and
    # the fallback is flagged, never silently presented as a Welch error
    w = welch_diff(np.array([0.640, 0.641]), np.array([0.6395]))
    assert w["error_model"] == "pooled_one_sided"
    assert w["sem"] == pytest.approx(7.0710678e-4 * math.sqrt(1 / 2 + 1 / 1), rel=1e-5)
    assert w["df"] == pytest.approx(1.0)
    # both levels single-seeded: no error estimate at all
    w2 = welch_diff(np.array([0.640]), np.array([0.6395]))
    assert w2["sem"] is None and w2["df"] is None
    assert w2["delta"] == pytest.approx(-5e-4)


def test_classify_converged_when_interval_inside_tolerance():
    # Delta = 0, tiny error -> certified
    v = classify_bias(0.0, 1e-5, 10.0, tol=5e-4)
    assert v["verdict"] == "CONVERGED"
    assert v["hi"] < 5e-4 and v["lo"] > -5e-4


def test_classify_biased_when_interval_fully_outside():
    v = classify_bias(-2e-3, 1e-5, 10.0, tol=5e-4)
    assert v["verdict"] == "BIASED"


def test_classify_undecidable_when_interval_straddles_tolerance():
    # the m21401 situation: point estimate just outside, error larger still
    v = classify_bias(-5.069e-4, 3.13e-4, 2.0, tol=5e-4)
    assert v["verdict"] == "UNDECIDABLE"
    assert v["lo"] < -5e-4 < v["hi"]


def test_classify_requires_stricter_interval_than_point_estimate():
    # |Delta| < tol but the interval is wide -> must NOT be certified
    v = classify_bias(1e-4, 4e-4, 5.0, tol=5e-4)
    assert v["verdict"] == "UNDECIDABLE"


def test_point_only_fallback_is_flagged():
    v = classify_bias(-5.069e-4, None, None, tol=5e-4)
    assert v["verdict"] == "BIASED"
    assert v["resolved_from"] == "point"


def test_seeds_needed_shrinks_with_scatter():
    assert seeds_needed(3.0e-5, 5.0e-4) < seeds_needed(3.0e-4, 5.0e-4)


def test_seeds_needed_is_monotone_and_reaches_target():
    n = seeds_needed(3.0e-4, 5.0e-4)
    from scipy import stats
    hw = stats.t.ppf(0.975, 2 * n - 2) * math.sqrt(2.0) * 3.0e-4 / math.sqrt(n)
    assert hw < 5.0e-4


def _fake_run(mu, kn, seed, dt_frac, phi, n=10000):
    return {"path": f"runs/x_{mu}_{kn}_{seed}_{dt_frac}.json", "n": n, "mu": mu,
            "kn": kn, "seed": seed, "dt_frac": dt_frac, "phi": phi,
            "phi_corr": phi, "settled": True}


def test_levels_present_requires_both_timesteps():
    runs = [_fake_run(0.0, 4000, 1, 1 / 20, 0.642),
            _fake_run(0.0, 4000, 1, 1 / 40, 0.641),
            _fake_run(0.1, 8000, 1, 1 / 20, 0.627)]
    assert levels_present(runs) == [(0.0, 4000)]


def test_audit_verdict_flips_from_undecidable_to_converged_with_more_seeds():
    from seed_bias_audit import audit
    # synthetic campaign: true bias 0, scatter 3e-4
    rng = np.random.default_rng(7)
    coarse = 0.6363 + 3e-4 * rng.standard_normal(3)
    one_fine = [_fake_run(0.0, 64000, 1, 1 / 40, coarse[0] - 5.069e-4)]
    many_fine = [_fake_run(0.0, 64000, s + 1, 1 / 40, coarse[s])
                 for s in range(3)]
    base = [_fake_run(0.0, 64000, s + 1, 1 / 20, coarse[s]) for s in range(3)]
    v1 = audit(base + one_fine)
    v2 = audit(base + many_fine)
    assert v1["rows"][0]["verdict"] == "UNDECIDABLE"           # n_fine = 1
    assert v2["rows"][0]["verdict"] == "CONVERGED"             # consistent seeds
    assert v2["all_converged"] is True


def test_real_run_set_is_readable_and_consistent():
    """Smoke test: the on-disk run set parses and every row is self-consistent."""
    runs = load_runs()
    if not runs:                                    # run set not shipped
        pytest.skip("no deposition runs on disk")
    rows = __import__("seed_bias_audit").audit(runs)["rows"]
    assert rows, "both timestep levels should exist for at least one level"
    for r in rows:
        assert r["verdict"] in ("CONVERGED", "BIASED", "UNDECIDABLE")
        assert r["n_coarse"] >= 1 and r["n_fine"] >= 1
        if r["ci_lo"] is not None:
            assert r["ci_lo"] < r["delta"] < r["ci_hi"]
            assert r["half_width"] > 0