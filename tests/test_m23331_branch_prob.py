"""Pins the m23331 branch-probability estimator.

The estimator is the only thing this mission adds to the decision chain, so it is
tested the way an instrument is tested: against known values (Wilson interval,
Fisher exact), against every branch of the pre-registered verdict rule, and
against **planted mixtures** whose answer is known before the code runs.  A
classifier that can only say "pass" is not tested — two of the tests below force
it to fail.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "benchmarks" / "m23331_branch_prob"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import analyse as AZ  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Wilson score interval
# --------------------------------------------------------------------------- #
def test_wilson_known_values():
    # closed-form check: centre = (p + z^2/2n)/(1 + z^2/n), same half-width
    lo, hi = AZ.wilson(6, 6)
    assert lo == pytest.approx(0.609666, abs=2e-6)
    assert hi == 1.0
    lo, hi = AZ.wilson(0, 6)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(0.390334, abs=2e-6)
    lo, hi = AZ.wilson(3, 6)
    assert (lo, hi) == pytest.approx((0.187616, 0.812384), abs=2e-6)


def test_wilson_brackets_the_estimate_and_is_monotone_in_k():
    for n in range(1, 13):
        prev = -1.0
        for k in range(n + 1):
            lo, hi = AZ.wilson(k, n)
            assert -1e-12 <= lo <= k / n + 1e-12
            assert k / n - 1e-12 <= hi <= 1.0 + 1e-12
            assert k / n >= prev
            prev = k / n


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        AZ.wilson(7, 6)


# --------------------------------------------------------------------------- #
# 2. Fisher exact test (exact hypergeometric, no scipy)
# --------------------------------------------------------------------------- #
def test_fisher_extreme_table_matches_closed_form():
    # [[6,0],[0,6]]: p = 2 / C(12,6) = 2/924
    out = AZ.fisher_exact(6, 0, 0, 6)
    assert out["p_value"] == pytest.approx(2.0 / math.comb(12, 6), rel=1e-12)
    assert out["odds_ratio"] == math.inf


def test_fisher_identical_arms_is_one():
    assert AZ.fisher_exact(3, 3, 3, 3)["p_value"] == pytest.approx(1.0, rel=1e-12)


def test_fisher_is_symmetric_under_row_and_column_swap():
    base = AZ.fisher_exact(6, 0, 1, 5)["p_value"]
    assert AZ.fisher_exact(1, 5, 6, 0)["p_value"] == pytest.approx(base, rel=1e-12)
    assert base < 0.05
    # 5/1 vs 1/5 is NOT significant at 5 % (p = 0.0801) — the estimator must not
    # manufacture significance from a small table
    assert AZ.fisher_exact(5, 1, 1, 5)["p_value"] == pytest.approx(0.0801, abs=1e-3)


def test_fisher_no_difference_is_not_significant():
    assert AZ.fisher_exact(4, 2, 2, 4)["p_value"] > 0.05


# --------------------------------------------------------------------------- #
# 3. Every branch of the pre-registered verdict rule
# --------------------------------------------------------------------------- #
def _bp(n_flow, n):
    lo, hi = AZ.wilson(n_flow, n)
    return {"n": n, "n_flowing": n_flow, "p_flowing": n_flow / n,
            "ci95": [lo, hi], "pure": n_flow in (0, n)}


def test_verdict_b1_spelling_dominated():
    a, b = _bp(6, 6), _bp(0, 6)
    v = AZ.classify_branch(a, b, AZ.fisher_exact(6, 0, 0, 6))
    assert v["verdict"] == "B1_SPELLING_DOMINATED"


def test_verdict_b1_needs_its_own_significance():
    # amendment A1: 4/4 vs 0/1 is pure-and-opposite but Fisher p = 0.20, so it
    # must NOT be reported as a factor effect
    a, b = _bp(4, 4), _bp(0, 1)
    fe = AZ.fisher_exact(4, 0, 0, 1)
    assert fe["p_value"] == pytest.approx(0.2)
    assert AZ.classify_branch(a, b, fe)["verdict"] == "B3_INCONCLUSIVE"


def test_verdict_b2_both_arms_pure_and_equal():
    a, b = _bp(6, 6), _bp(6, 6)
    v = AZ.classify_branch(a, b, AZ.fisher_exact(6, 0, 6, 0))
    assert v["verdict"] == "B2_SEED_DOMINATED"
    a, b = _bp(0, 6), _bp(0, 6)
    v = AZ.classify_branch(a, b, AZ.fisher_exact(0, 6, 0, 6))
    assert v["verdict"] == "B2_SEED_DOMINATED"


def test_verdict_b2_both_arms_mixed_indistinguishable():
    a, b = _bp(3, 6), _bp(2, 6)
    fe = AZ.fisher_exact(3, 3, 2, 4)
    assert fe["p_value"] >= 0.05
    assert AZ.classify_branch(a, b, fe)["verdict"] == "B2_SEED_DOMINATED"


def test_verdict_b3_one_arm_pure_one_mixed():
    a, b = _bp(6, 6), _bp(1, 6)
    v = AZ.classify_branch(a, b, AZ.fisher_exact(6, 0, 1, 5))
    assert v["verdict"] == "B3_INCONCLUSIVE"


# --------------------------------------------------------------------------- #
# 4. Conditional statistics — and the fact that they can fail
# --------------------------------------------------------------------------- #
def test_conditional_pass_and_ci_arithmetic():
    rates = [0.0300, 0.0305, 0.0298, 0.0302]
    ref = 0.0305
    out = AZ.conditional_stats(rates, ref)
    n = 4
    sd = float(__import__("numpy").std(rates, ddof=1))
    half = 3.182 * sd / math.sqrt(n)
    assert out["n"] == n
    assert out["mean_kg_s"] == pytest.approx(sum(rates) / n)
    assert out["ci95_halfwidth_kg_s"] == pytest.approx(half, rel=1e-6)
    assert out["verdict"] == "PASS"


def test_conditional_wide_ci_is_inconclusive_not_fail():
    # 4 seeds with ~6.5 % CV: mean is unbiased but the t-interval is ~10 % wide
    rates = [0.0307498, 0.0266852, 0.0305731, 0.0288058]
    out = AZ.conditional_stats(rates, 0.0292425)
    assert out["accuracy_ok"] is True
    assert out["precision_ok"] is False
    assert out["verdict"] == "INCONCLUSIVE_CI_TOO_WIDE"


def test_required_n_matches_the_interval_width():
    cv = 0.065
    n = AZ.required_n(cv)
    assert n is not None
    prev = AZ.T975[n - 2] * cv / math.sqrt(n - 1)
    assert prev >= 0.10  # n-1 arms are not enough
    assert AZ.T975[n - 1] * cv / math.sqrt(n) < 0.10


def test_conditional_fails_a_20_percent_high_arm():
    # planted bias: the estimator must report FAIL, not absorb it
    rates = [0.0366, 0.0367, 0.0365, 0.0366]
    out = AZ.conditional_stats(rates, 0.0305)
    assert out["rel_error_vs_reference"] == pytest.approx(0.20, abs=0.01)
    assert out["verdict"] == "FAIL"


def test_conditional_refuses_to_grade_a_single_record():
    out = AZ.conditional_stats([0.0307], 0.0305)
    assert out["n"] == 1
    assert out["verdict"] == "NOT_DECIDABLE"
    out = AZ.conditional_stats([], 0.0305)
    assert out["verdict"] == "NOT_DECIDABLE"


def test_conditional_never_averages_an_arrested_record():
    # 0.0034 kg/s is an arch, not a rate: it must be filtered before averaging
    rates = [0.0307, 0.0306, 0.0034]
    out = AZ.conditional_stats([r for r in rates if r > 0.0], 0.0305)
    assert out["n"] == 3  # conditional_stats is fed FLOWING records only
    agg = AZ.conditional_stats([0.0307, 0.0306], 0.0305)
    assert agg["mean_kg_s"] == pytest.approx(0.03065)


# --------------------------------------------------------------------------- #
# 5. End-to-end on a planted mixture (load_records -> analyse)
# --------------------------------------------------------------------------- #
def _fake_record(arm, seed, w, arrested, ref=0.0305):
    return {
        "flow_rate_kg_s": w,
        "window_particles": [3, 3, 0, 0, 0, 0, 0, 0] if arrested else [60] * 8,
        "flow_rate_windows_kg_s": [w] * 8,
        "particles_exited": 6 if arrested else 480,
        "discharge_status": {"status": "ARRESTED" if arrested else "FLOWING",
                             "reasons": ["rate is 11% of the reference"] if arrested else [],
                             "rate_over_reference": w / ref,
                             "empty_window_fraction": 0.75 if arrested else 0.0},
        "branch_prob": {"arm": arm, "seed": seed, "dt_s": 1.86767398828964e-05,
                        "dt_bits": "3f53a1b0", "w_reference_kg_s": ref,
                        "bed_state_sha256": f"{arm}{seed}",
                        "rel_error_vs_beverloo": w / ref - 1.0},
    }


def _write(tmp, recs):
    d = Path(tmp) / "runs"
    d.mkdir(parents=True, exist_ok=True)
    for r in recs:
        f = d / f"{r['branch_prob']['arm']}_s{r['branch_prob']['seed']}.json"
        f.write_text(json.dumps(r))
    return d


def test_end_to_end_planted_b1(tmp_path):
    recs = ([_fake_record("F", s, 0.0307 + 1e-5 * s, False) for s in range(1, 7)]
            + [_fake_record("Q", s, 0.00336 + 1e-7 * s, True) for s in range(1, 7)])
    res = AZ.analyse(AZ.load_records(_write(tmp_path, recs)))
    assert res["branch_verdict"]["verdict"] == "B1_SPELLING_DOMINATED"
    assert res["arms"]["F"]["branch"]["p_flowing"] == 1.0
    assert res["arms"]["Q"]["branch"]["p_flowing"] == 0.0
    assert res["anchor"]["conditional"]["verdict"] == "PASS"
    assert res["arms"]["Q"]["conditional"]["verdict"] == "NOT_DECIDABLE"


def test_end_to_end_planted_seed_dominated(tmp_path):
    recs = []
    for arm in "FQ":
        for s in range(1, 7):
            arrested = s % 2 == 0  # seed parity decides, dt does not
            recs.append(_fake_record(arm, s, 0.0034 if arrested else 0.0306, arrested))
    res = AZ.analyse(AZ.load_records(_write(tmp_path, recs)))
    assert res["branch_verdict"]["verdict"] == "B2_SEED_DOMINATED"
    assert res["arms"]["F"]["branch"]["n_flowing"] == 3
    assert res["arms"]["Q"]["branch"]["n_flowing"] == 3


def test_summary_json_is_written_by_main(tmp_path):
    # 5 per arm so that the pure split is also significant (2/C(10,5) = 0.008)
    recs = ([_fake_record("F", s, 0.0307, False) for s in range(1, 6)]
            + [_fake_record("Q", s, 0.0034, True) for s in range(1, 6)])
    d = _write(tmp_path, recs)
    out = tmp_path / "summary.json"
    doc = tmp_path / "doc.md"
    assert AZ.main(["--dir", str(d), "--out", str(out), "--doc", str(doc),
                    "--no-figures"]) == 0
    res = json.loads(out.read_text())
    assert res["n_records"] == 10
    assert res["branch_verdict"]["verdict"] == "B1_SPELLING_DOMINATED"
    assert doc.read_text().startswith("#")


# --------------------------------------------------------------------------- #
# 6. The frozen case and the dt spellings are what the prereg claims
# --------------------------------------------------------------------------- #
def test_frozen_case_domain_inequalities():
    import run_case as RC

    f = RC.FROZEN
    d_over_d = f["orifice_diameter"] / f["particle_diameter"]
    silo_over_orifice = 2.0 * f["silo_radius"] / f["orifice_diameter"]
    h_over_d = f["fill_height"] / f["orifice_diameter"]
    assert d_over_d == pytest.approx(5.0) and d_over_d >= 4.0
    assert silo_over_orifice >= 2.5
    assert h_over_d >= 1.5


def test_dt_spellings_differ_by_exactly_one_ulp():
    import run_case as RC

    s = RC.dt_specs()
    assert s["ulp_gap_Q_minus_F"] == -1
    assert s["dt_F_s"] != s["dt_Q_s"]
    assert s["rel_gap"] == pytest.approx(1.8e-16, rel=0.5)
    # post-m23261 `Material.dt` *is* the Q spelling (dt_for_steps), F is legacy
    assert s["dt_Q_s"] == s["material"].dt
    assert s["dt_F_s"] != s["material"].dt


def test_shard_mapping_is_complete_and_unique():
    import run_case as RC

    names = RC.shard_names()
    assert len(names) == len(set(names)) == 12
    assert names[0] == "F_s1" and names[-1] == "Q_s6"
    for i, name in enumerate(names, 1):
        assert RC.parse_shard(name) == (name[0], int(name.split("_s")[1]))
        assert i <= 12