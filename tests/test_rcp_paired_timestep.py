"""Tests for the common-random-numbers (paired) timestep-bias estimator.

Two kinds of claim are tested, and they are kept apart:

* **the premise** -- a coarse and a fine run at the same seed really start from
  the identical configuration (checked against the integrator's own placement
  routine, not by reading the code);
* **the statistics** -- the paired interval, its degeneracy rules and the
  variance reduction, on synthetic closed-form data, so no cluster run is
  required to run the suite.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks", "case_rcp_deposition"))

import paired_timestep_audit as pta
from paired_timestep_audit import (audit, pair_by_seed, paired_diff,
                                   seeds_needed_paired, variance_reduction)
from seed_bias_audit import CONF, DT_COARSE, DT_FINE, TOL_BIAS


# ---------------------------------------------------------------------------
# premise: same seed => same initial configuration, independent of dt_frac
# ---------------------------------------------------------------------------
def test_spec_geometry_is_dt_independent():
    from tensordem.deposition_packing import DepositionSpec
    a = DepositionSpec(n_particles=2000, seed=7, dt_frac=DT_COARSE)
    b = DepositionSpec(n_particles=2000, seed=7, dt_frac=DT_FINE)
    assert a.box_width == b.box_width
    assert a.initial_height == b.initial_height
    assert a.dt != b.dt  # the one thing that *is* supposed to differ


def test_initial_configuration_identical_across_timesteps():
    from tensordem.deposition_packing import (DepositionSpec,
                                              initial_configuration)
    out = []
    for dtf in (DT_COARSE, DT_FINE):
        s = DepositionSpec(n_particles=300, seed=11, dt_frac=dtf)
        pos, info = initial_configuration(s.n_particles, s.box_width,
                                          s.initial_height, s.radius, s.seed)
        out.append((pos, info))
    (p0, i0), (p1, i1) = out
    assert np.array_equal(p0, p1)          # bit-identical placement
    assert i0 == i1                        # same rejected-candidate count too


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _pairs(deltas, coarse, seed0=1, premise=True):
    return [{"seed": seed0 + i, "phi_coarse": float(c), "phi_fine": float(c + d),
             "delta": float(d), "premise_ok": premise}
            for i, (d, c) in enumerate(zip(deltas, coarse))]


def test_paired_diff_matches_hand_computation():
    d = np.array([1e-4, 3e-4, 2e-4, 2e-4])
    coarse = [0.64] * 4
    res = paired_diff(_pairs(d, coarse))
    assert res["n_pairs"] == 4
    assert res["delta"] == pytest.approx(d.mean())
    sd = d.std(ddof=1)
    assert res["std_paired"] == pytest.approx(sd)
    assert res["sem"] == pytest.approx(sd / math.sqrt(4))
    assert res["df"] == 3
    tc = stats.t.ppf(0.975, 3)
    assert res["lo"] == pytest.approx(d.mean() - tc * sd / 2)
    assert res["hi"] == pytest.approx(d.mean() + tc * sd / 2)
    assert res["error_model"] == "paired_crn"


def test_pairing_removes_between_seed_scatter():
    """Large seed-to-seed scatter, tiny within-pair difference.

    Unpaired Welch must be wide here; paired must be narrow.  This is the
    mechanical justification for the whole estimator.
    """
    offsets = np.array([-0.004, 0.003, -0.001, 0.002])
    deltas = np.array([2e-4, 2.2e-4, 1.9e-4, 2.1e-4])
    ps = _pairs(deltas, 0.64 + offsets)
    vr = variance_reduction(ps)
    assert vr["ratio"] is not None and vr["ratio"] > 5.0
    assert vr["sem_paired"] < vr["sem_unpaired"] / 5.0


def test_interval_verdicts_are_three_way():
    # tightly clustered, well inside the band -> CONVERGED from interval
    tight = paired_diff(_pairs([1e-5, 1.1e-5, 0.9e-5, 1.0e-5], [0.64] * 4))
    assert tight["verdict"] == "CONVERGED" and tight["resolved_from"] == "interval"
    # tightly clustered, well outside the band -> BIASED
    far = paired_diff(_pairs([5e-3, 5.1e-3, 4.9e-3, 5.0e-3], [0.64] * 4))
    assert far["verdict"] == "BIASED" and far["resolved_from"] == "interval"
    # scatter wider than the band -> UNDECIDABLE, never a pass
    noisy = paired_diff(_pairs([-1.5e-3, 1.5e-3, -1.0e-3, 2.0e-3], [0.64] * 4))
    assert noisy["verdict"] == "UNDECIDABLE"


def test_single_pair_is_a_point_estimate_not_a_certificate():
    res = paired_diff(_pairs([1e-5], [0.64]))
    assert res["n_pairs"] == 1
    assert res["sem"] is None and res["lo"] is None
    assert res["resolved_from"] == "point"
    assert res["verdict"] == "CONVERGED"   # vocabulary unchanged ...


def test_audit_never_certifies_from_a_point_estimate(monkeypatch):
    """... but the audit's ``certified`` flag must refuse it."""
    runs = [{"path": "x", "n": 10000, "mu": 0.0, "kn": 4000.0, "seed": 1,
             "dt_frac": DT_COARSE, "phi": 0.64, "phi_corr": 0.64,
             "settled": True},
            {"path": "y", "n": 10000, "mu": 0.0, "kn": 4000.0, "seed": 1,
             "dt_frac": DT_FINE, "phi": 0.64 + 1e-5, "phi_corr": 0.64,
             "settled": True}]
    monkeypatch.setattr(pta, "placement_fingerprint", lambda p: {"seed": 1})
    row = pta.audit_level(runs, 0.0, 4000.0)
    assert row["paired"]["verdict"] == "CONVERGED"
    assert row["certified"] is False


def test_premise_failure_drops_the_pair(monkeypatch):
    runs = [{"path": "a", "n": 10000, "mu": 0.0, "kn": 4000.0, "seed": s,
             "dt_frac": DT_COARSE, "phi": 0.64, "phi_corr": 0.64,
             "settled": True} for s in (1, 2)]
    runs += [{"path": "b", "n": 10000, "mu": 0.0, "kn": 4000.0, "seed": s,
              "dt_frac": DT_FINE, "phi": 0.6401, "phi_corr": 0.64,
              "settled": True} for s in (1, 2)]
    seen = {"n": 0}

    def fp(path):
        # call order per pair is (coarse, fine): seeds 1,1 then seeds 2,2
        seen["n"] += 1
        vals = {1: 26444, 2: 26444, 3: 26445, 4: 26446}
        return {"rejected_candidates": vals.get(seen["n"], 0),
                "seed_ok": True}

    monkeypatch.setattr(pta, "placement_fingerprint", fp)
    pairs = pair_by_seed(runs, 0.0, 4000.0)
    ok = [p for p in pairs if p["premise_ok"]]
    bad = [p for p in pairs if not p["premise_ok"]]
    assert len(ok) == 1 and len(bad) == 1
    res = paired_diff(pairs)
    assert res["dropped_seeds_premise_failed"] == [bad[0]["seed"]]
    assert res["n_pairs"] == 1


def test_seeds_needed_paired_matches_the_formula():
    sigma, margin = 3.9e-4, TOL_BIAS
    n = seeds_needed_paired(sigma, margin)
    tc = stats.t.ppf(1.0 - (1.0 - CONF) / 2.0, n - 1)
    assert tc * sigma / math.sqrt(n) < margin
    tc_prev = stats.t.ppf(1.0 - (1.0 - CONF) / 2.0, n - 2)
    assert tc_prev * sigma / math.sqrt(n - 1) >= margin   # tightest n
    assert seeds_needed_paired(0.0, margin) == 1


def test_paired_budget_is_no_worse_than_unpaired():
    """Same scatter: the paired estimator needs no more pairs than Welch."""
    sigma = 3.9e-4
    assert seeds_needed_paired(sigma, TOL_BIAS) <= 5


def test_verdict_vocabulary_is_not_extended():
    allowed = {"CONVERGED", "BIASED", "UNDECIDABLE", "NOT_RUN"}
    for deltas in ([1e-5, 1.1e-5, 0.9e-5], [5e-3, 5.1e-3, 4.9e-3],
                   [-2e-3, 2e-3, 0.0]):
        assert paired_diff(_pairs(deltas, [0.64] * 3))["verdict"] in allowed
    assert paired_diff([])["verdict"] == "NOT_RUN"


# ---------------------------------------------------------------------------
# archived runs: the premise holds on the real campaign data
# ---------------------------------------------------------------------------
def test_archived_runs_all_pair_on_identical_placement():
    res = audit()
    if not res["levels"]:
        pytest.skip("no archived run pairs in this checkout")
    assert res["summary"]["n_pairs_premise_failed"] == 0
    total_pairs = sum(r["paired"]["n_pairs"] for r in res["levels"])
    assert total_pairs >= 1
    for row in res["levels"]:
        for p in row["pairs"]:
            assert p["placement_coarse"] == p["placement_fine"]
            assert math.isclose(p["delta"],
                                p["phi_fine"] - p["phi_coarse"], abs_tol=1e-15)


def test_audit_json_is_serialisable():
    import json
    res = audit()
    json.dumps(res)   # raises if a numpy scalar leaked into the payload
    assert res["estimator"] == "paired_crn"
    assert res["tol_bias"] == TOL_BIAS