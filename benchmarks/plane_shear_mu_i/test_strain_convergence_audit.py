"""Tests for the round-16 strain-convergence audit (CASE-DEM-PLANE-SHEAR-MUIDI).

These assert *invariants* of the round-15 sweep, not merely that the analysis
runs:

* the step-budget model reproduces the archived strain to well under 1 %,
  which is what licenses the claim "the overrun is the pressure-recovery gate,
  not noise";
* the three points whose strain was overrun are exactly {0.03, 0.1, 0.3}, and
  the three that were honoured are {0.001, 0.003, 0.01};
* at I = 0.3 the recover gate spent *exactly* its whole 20 000-step budget --
  the arithmetic fingerprint of the defect;
* phi is stationary at every quotable point, and mu is stationary at none, so
  the mu acceptance may not be adjudicated from these runs;
* among the quotable points phi decreases with I (the dilatancy direction the
  reference law requires).
"""
from __future__ import annotations

import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import strain_convergence_audit as scra  # noqa: E402
import archive_strain_sweep as arc  # noqa: E402


@pytest.fixture(scope="module")
def runs():
    return scra.load_runs()


@pytest.fixture(scope="module")
def budget(runs):
    return scra.budget_table(runs)


# ---------------------------------------------------------------- availability
def test_six_archived_points_present(runs):
    assert len(runs) == 6
    assert [r["I_target"] for r in runs] == [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]


def test_provenance_names_the_cluster_jobs(runs):
    jobs = [r["provenance"]["cluster_job"] for r in runs]
    assert jobs == [f"cj{j}" for j in range(2950, 2956)]


def test_archive_index_matches_archived_series(runs, tmp_path):
    idx = os.path.join(scra.RUN_DIR, "strain_sweep_archive_index.json")
    assert os.path.exists(idx), "archive index missing"
    with open(idx) as fh:
        data = json.load(fh)
    assert data["jobs"] == [f"cj{j}" for j in range(2950, 2956)]
    for entry, run in zip(data["points"], runs):
        p = os.path.join(scra.RUN_DIR, entry["file"])
        assert os.path.exists(p)
        with open(p) as fh:
            disk = json.load(fh)
        assert disk["strain_achieved"] == pytest.approx(run["strain_achieved"])
        n = len(disk["series"]["strain"])
        assert n <= arc.KEEP_POINTS + 1, "series not thinned to budget"
        prov = disk["provenance"]
        assert prov["series_length_archived"] == n
        assert prov["series_downsample_factor"] >= 1


# ------------------------------------------------------- the budget model
def test_step_model_reproduces_archived_strain(budget):
    """The causal model must close on the data it claims to explain."""
    for b in budget:
        assert b["bookkeeping_error"] < 0.01, (
            f"I={b['I_target']}: model strain {b['strain_reconstructed']:.3f} "
            f"vs archived {b['strain_achieved']:.3f}")


def test_recover_gate_exhausted_exactly_at_high_I(budget):
    """I=0.3 spends the whole max_recover_steps default (20 000)."""
    high = next(b for b in budget if b["I_target"] == 0.3)
    assert high["recover_steps_spent"] == 20000


def test_budget_classification(budget):
    respected = [b["I_target"] for b in budget if b["respects_budget"]]
    violated = [b["I_target"] for b in budget if not b["respects_budget"]]
    assert respected == [0.001, 0.003, 0.01]
    assert violated == [0.03, 0.1, 0.3]


def test_overrun_is_monotone_in_I(budget):
    """Overrun factor must be non-decreasing with I over the sweep."""
    f = [b["overrun_factor"] for b in budget]
    assert all(x <= y + 1e-9 for x, y in zip(f, f[1:])), f
    assert f[-1] > 40.0, "the I=0.3 overrun should be order 50x"


def test_strain_budget_pure_function_on_synthetic_input():
    """No file dependency: the model must run on a constructed record."""
    rec = {"I_target": 0.02, "dt": 3.215e-5, "strain_requested": 2.0,
           "shear_steps_executed": 10000, "strain_achieved": 6.43}
    b = scra.strain_budget(rec)
    gd = scra.gamma_dot_of(0.02)
    assert b["gamma_dot"] == pytest.approx(500.0 * 0.02)
    assert b["strain_reconstructed"] == pytest.approx(10000 * gd * 3.215e-5)
    assert b["overrun_factor"] == pytest.approx(6.43 / 2.0)


def test_planning_floor_uses_device_formula():
    """planned = ceil(strain/(gamma_dot dt)) + settle guard."""
    for I in (0.001, 0.01, 0.3):
        gd = scra.gamma_dot_of(I)
        expected = int(math.ceil(2.0 / (gd * 3.215e-5))) + scra.SETTLE_GUARD
        rec = scra.strain_budget({"I_target": I, "dt": 3.215e-5,
                                  "strain_requested": 2.0,
                                  "shear_steps_executed": 1,
                                  "strain_achieved": gd * 3.215e-5})
        assert rec["planned_floor_steps"] == expected


# ------------------------------------------------------------- stationarity
def test_block_means_known_array():
    assert scra.block_means([1, 1, 2, 2, 3, 3, 4, 4], 4) == [1.0, 2.0, 3.0, 4.0]


def test_block_means_ignores_nan():
    assert scra.block_means([1.0, float("nan"), 3.0, 5.0], 2) == pytest.approx([1.0, 4.0])


def test_convergence_detects_flat_and_ramp():
    flat = {"series": {"x": [1.0] * 40}}
    assert scra.convergence(flat, "x")["converged"] is True
    ramp = {"series": {"x": [float(i) for i in range(1, 41)]}}
    c = scra.convergence(ramp, "x")
    assert c["converged"] is False
    assert c["terminal_drift_pct"] > 0


def test_phi_converged_at_every_quotable_point(runs, budget):
    by_i = {r["I_target"]: r for r in runs}
    for b in budget:
        if not b["respects_budget"]:
            continue
        c = scra.convergence(by_i[b["I_target"]])
        assert c["converged"], (b["I_target"], c["terminal_drift_pct"])


def test_mu_not_converged_anywhere_in_this_sweep(runs, budget):
    """A strain-2.0 window is far too short for the virial stress."""
    by_i = {r["I_target"]: r for r in runs}
    for b in budget:
        if not b["respects_budget"]:
            continue
        c = scra.convergence(by_i[b["I_target"]], key="full_mu")
        assert not c["converged"], (b["I_target"], c["terminal_drift_pct"])
        assert abs(c["terminal_drift_pct"]) > 1.0


def test_verdict_shape(runs):
    v = scra.verdict(runs)
    assert v["n_strain_respected"] == 3 and v["n_strain_violated"] == 3
    assert v["mu_usable_for_acceptance"] is False
    assert v["mu_converged_I"] == []
    assert sorted(v["phi_converged_I"]) == [0.001, 0.003, 0.01]


# --------------------------------------------------------- physics direction
def test_phi_decreases_with_I_among_quotable_points(runs, budget):
    """Dilatancy: the reference law and every admissible theory require it."""
    ref = scra.reference_compare(runs)
    quot = [r for r, b in zip(ref, budget) if b["respects_budget"]]
    phis = [r["phi"] for r in quot]
    assert all(a >= b for a, b in zip(phis, phis[1:])), phis


def test_reference_errors_are_computed_with_correct_sign(runs):
    ref = scra.reference_compare(runs)
    for r in ref:
        assert r["phi_err_pct"] == pytest.approx(
            100.0 * (r["phi"] - r["phi_ref"]) / r["phi_ref"])
    # the container phi sits below the reference band at every I in the sweep
    assert all(r["phi_err_pct"] < 0 for r in ref)


def test_quotable_phi_is_outside_the_ten_percent_band(runs, budget):
    """Documents the honest FAIL: convergence does not close the phi gap."""
    ref = scra.reference_compare(runs)
    quot = [r for r, b in zip(ref, budget) if b["respects_budget"]]
    assert len(quot) == 3
    for r in quot:
        assert abs(r["phi_err_pct"]) > 10.0, (r["I_target"], r["phi_err_pct"])


# ---------------------------------------------------------------- utilities
def test_thin_keeps_endpoints_and_budget():
    series = list(range(5000))
    out = arc.thin(series, 41)
    assert out[0] == 0 and out[-1] == 4999
    assert len(out) <= arc.KEEP_POINTS + 1


def test_markdown_table_contains_every_point(budget, runs):
    md = scra.markdown_table(budget, scra.reference_compare(runs))
    for b in budget:
        assert f"{b['I_target']:g}" in md
    assert "**NO**" in md and "yes" in md