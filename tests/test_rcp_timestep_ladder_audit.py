"""Invariants for the RCP temporal-convergence ladder audit (m21401).

The audit's claim is narrow but load-bearing: the hard-sphere intercept of
CASE-DEM-RCP-PACKING-FRACTION is only as good as the *time* convergence of the
stiffness levels that feed it.  These tests pin the three ways that statement
can silently become false:

* a coarse/fine pair with a real bias is reported as admissible (false PASS);
* runs of different ``N`` are averaged into one ladder level, which shifts the
  bias without any error being raised (this actually happened while writing the
  script -- the first version mixed N = 2000/4000/10000);
* a level that was never probed is counted as converged instead of ``NOT_RUN``.
"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(HERE, "benchmarks", "case_rcp_deposition")
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.join(HERE, "src"))

import timestep_ladder_audit as tla  # noqa: E402


def run(n, mu, kn, dt_frac, phi, seed=1):
    return {"path": f"runs/deposition_n{n}_mu{mu}_kn{kn}_seed{seed}.json",
            "n": n, "mu": mu, "kn": kn, "seed": seed, "dt_frac": dt_frac,
            "phi": phi, "phi_corr": phi, "settled": True}


# ---------------------------------------------------------------------------
# Richardson bracket
# ---------------------------------------------------------------------------
def test_richardson_bracket_contains_a_pure_first_order_signal():
    """phi(dt) = phi0 + C dt: the q=1 projection is exact, q=2 brackets it."""
    phi0, c = 0.6400, 0.02
    coarse, fine = phi0 + c * 1.0, phi0 + c * 0.5
    out = tla.richardson_dt(coarse, fine)
    assert out["by_order"]["q1"] == pytest.approx(phi0, abs=1e-12)
    assert out["lo"] <= phi0 <= out["hi"]
    assert out["delta"] == pytest.approx(fine - coarse)
    assert out["half_width"] > 0.0


def test_richardson_bracket_contains_a_pure_second_order_signal():
    phi0, c = 0.6300, 0.02
    coarse, fine = phi0 + c * 1.0 ** 2, phi0 + c * 0.5 ** 2
    out = tla.richardson_dt(coarse, fine)
    assert out["by_order"]["q2"] == pytest.approx(phi0, abs=1e-12)
    assert out["lo"] <= phi0 <= out["hi"]


def test_richardson_is_sign_aware_no_bias_means_no_correction():
    out = tla.richardson_dt(0.64, 0.64)
    assert out["delta"] == 0.0
    assert out["limit"] == pytest.approx(0.64, abs=1e-15)
    assert out["half_width"] == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# ladder assembly
# ---------------------------------------------------------------------------
def test_ladder_ignores_runs_at_other_particle_counts():
    """Regression: mixing N into one level shifts the bias with no error.

    The first version of the audit averaged the N = 2000/4000 runs of the same
    ``(mu, k_n, dt_frac)`` into the N = 10000 level mean and reported a
    ``+2.0e-3`` temporal bias at ``k_n = 4000`` that does not exist.
    """
    runs = [
        run(10000, 0.0, 4000, tla.DT_COARSE, 0.6426),
        run(4000, 0.0, 4000, tla.DT_COARSE, 0.6402),
        run(2000, 0.0, 4000, tla.DT_COARSE, 0.6384),
        run(10000, 0.0, 4000, tla.DT_FINE, 0.6427),
    ]
    lad = tla.ladder(runs, 0.0)
    row = [r for r in lad["rows"] if r["kn"] == 4000][0]
    assert row["n_coarse"] == 1
    assert row["phi_coarse"] == pytest.approx(0.6426)
    assert row["delta"] == pytest.approx(1e-4)


def test_ladder_level_without_fine_run_is_kept_but_unprobed():
    runs = [
        run(10000, 0.0, 4000, tla.DT_COARSE, 0.6426),
        run(10000, 0.0, 4000, tla.DT_FINE, 0.6427),
        run(10000, 0.0, 64000, tla.DT_COARSE, 0.6368),
    ]
    lad = tla.ladder(runs, 0.0)
    stiff = [r for r in lad["rows"] if r["kn"] == 64000][0]
    assert stiff["phi_fine"] is None
    assert stiff["admissible"] is None
    # an unprobed level still enters the ladder, at its coarse value
    assert stiff["level_used"] == pytest.approx(0.6368)
    assert [kn for kn, _ in lad["corrected"]] == [4000.0, 64000.0]


def test_ladder_separates_the_two_friction_tiers():
    runs = [
        run(10000, 0.0, 4000, tla.DT_COARSE, 0.6426),
        run(10000, 0.1, 4000, tla.DT_COARSE, 0.6267),
    ]
    assert tla.ladder(runs, 0.0)["raw"][0][1] == pytest.approx(0.6426)
    assert tla.ladder(runs, 0.1)["raw"][0][1] == pytest.approx(0.6267)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def _ana(runs):
    return tla.analyse(runs)


def _gate(gates, gid):
    hits = [g for g in gates if g["id"] == gid]
    assert hits, f"gate {gid} missing from {[g['id'] for g in gates]}"
    return hits[0]


def test_missing_probe_is_not_run_never_pass():
    runs = [
        run(10000, 0.0, 4000, tla.DT_COARSE, 0.6426),
        run(10000, 0.0, 4000, tla.DT_FINE, 0.6427),
        run(10000, 0.0, 64000, tla.DT_COARSE, 0.6368),
    ]
    gates = tla.build_gates(_ana(runs))
    assert _gate(gates, "T_bias_mu0_kn64000")["status"] == "NOT_RUN"
    assert _gate(gates, "T_bias_mu0_kn4000")["status"] == "PASS"
    # the stiff end of this ladder is unprobed -> coverage gate must fail
    assert _gate(gates, "T_stiff_end_covered_mu0")["status"] == "FAIL"


def test_bias_beyond_tolerance_fails_the_gate():
    noisy = tla.TOL_BIAS * 3.0
    runs = [
        run(10000, 0.0, 4000, tla.DT_COARSE, 0.6426),
        run(10000, 0.0, 4000, tla.DT_FINE, 0.6426 + noisy),
        run(10000, 0.0, 64000, tla.DT_COARSE, 0.6368),
        run(10000, 0.0, 64000, tla.DT_FINE, 0.6368),
    ]
    ana = _ana(runs)
    gates = tla.build_gates(ana)
    assert _gate(gates, "T_bias_mu0_kn4000")["status"] == "FAIL"
    # with a clean stiff level and a biased soft level the two-point
    # extrapolation still exists -- the gate, not a crash, is the verdict
    assert ana["intercepts"]["0"]["corrected"] is not None
    assert ana["intercepts"]["0"]["raw"] is not None


def test_intercept_shift_gate_flags_a_bias_that_moves_the_intercept():
    """A dt bias that grows with stiffness must show up as an intercept shift."""
    runs = []
    for kn, phi_c, phi_f in [(4000, 0.6426, 0.6427), (16000, 0.6379, 0.6385),
                             (64000, 0.6368, 0.6378)]:
        runs.append(run(10000, 0.0, kn, tla.DT_COARSE, phi_c))
        runs.append(run(10000, 0.0, kn, tla.DT_FINE, phi_f))
    ana = _ana(runs)
    shift = abs(ana["intercepts"]["0"]["corrected"]["limit"]
                - ana["intercepts"]["0"]["raw"]["limit"])
    gates = tla.build_gates(ana)
    g = _gate(gates, "T_intercept_shift_corrected_mu0")
    assert shift > tla.TOL_INTERCEPT_SHIFT
    assert g["status"] == "FAIL"


def test_single_level_ladder_cannot_produce_an_intercept():
    runs = [run(10000, 0.0, 4000, tla.DT_COARSE, 0.6426),
            run(10000, 0.0, 4000, tla.DT_FINE, 0.6427)]
    ana = _ana(runs)
    assert ana["intercepts"]["0"]["raw"] is None
    assert _gate(tla.build_gates(ana),
                 "T_intercept_shift_corrected_mu0")["status"] == "NOT_RUN"


# ---------------------------------------------------------------------------
# error table / claims
# ---------------------------------------------------------------------------
def test_error_table_answers_the_under_three_percent_question_per_row():
    runs = []
    for kn, phi in [(4000, 0.6426), (16000, 0.6379), (64000, 0.6368)]:
        runs.append(run(10000, 0.0, kn, tla.DT_COARSE, phi))
        runs.append(run(10000, 0.0, kn, tla.DT_FINE, phi + 1e-4))
    errs = tla.error_table(_ana(runs))
    assert errs
    for e in errs:
        assert e["under_3pct"] in {"yes", "no", "no admissible anchor"}
        assert e["verdict"] in {"PASS", "FAIL", "WITHHELD"}


def test_low_friction_tier_never_claims_accuracy():
    """mu=0.1 has no literature anchor, so the verdict must stay WITHHELD."""
    runs = []
    for kn, phi in [(4000, 0.6263), (16000, 0.6216), (64000, 0.6193)]:
        runs.append(run(10000, 0.1, kn, tla.DT_COARSE, phi))
        runs.append(run(10000, 0.1, kn, tla.DT_FINE, phi + 1e-4))
    errs = tla.error_table(_ana(runs))
    assert errs
    for e in errs:
        assert e["verdict"] == "WITHHELD"
        assert e["under_3pct"] == "no admissible anchor"


# ---------------------------------------------------------------------------
# the audit must stay wired to the artefacts it audits
# ---------------------------------------------------------------------------
def test_documented_plan_entries_match_the_probes_the_audit_expects():
    """plan.json declares the stiff-end temporal probes; the audit audits them."""
    plan = json.load(open(os.path.join(BENCH, "plan.json")))["cases"]
    stiff_dt = [c for c in plan if c.get("dt_frac") is not None and c["kn"] >= 16000]
    assert stiff_dt, "plan.json no longer declares a stiff-end temporal probe"
    for c in stiff_dt:
        assert c["dt_frac"] < tla.DT_COARSE
        assert c["n"] == tla.N_PRIMARY


def test_audit_reads_the_primary_estimator_of_the_merged_case():
    """The ladder must use the same estimator as the merged read-out (D1)."""
    src = open(os.path.join(BENCH, "timestep_ladder_audit.py")).read()
    assert 'm["phi_bulk"]' in src
    assert tla.N_PRIMARY == 10000