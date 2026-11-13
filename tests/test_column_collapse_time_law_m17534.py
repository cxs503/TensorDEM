"""Audit machinery + data regression for the *second* Lube (2004) law on
``CASE-COLUMN-COLLAPSE`` (m17534).

Two independent classes of checks:

1. **Machinery** — the reference formulas, the sign test, the log-log exponent
   fit and the definition-robustness rule are exercised on synthetic input, so
   a silent change in the analysis cannot pass unnoticed.
2. **Data regression** — the committed reduced provenance
   (``artifacts/column_collapse_time_law_m17534/cases_reduced.csv``, 25 round-5
   cases) is re-adjudicated here.  Every assertion below is a number quoted in
   ``docs/AUDIT_COLUMN_COLLAPSE_TIME_LAW_m17534.md``: if the audit report and
   the data ever diverge, this module fails.

The reduced table is committed so that the audit is reproducible without the
7.6 MB of raw per-case JSON (whose sha256 digests are in
``manifest_sha256.json``).
"""
from __future__ import annotations

import importlib.util
import math
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "analyze_column_collapse_time_law_m17534.py")
TABLE = os.path.join(ROOT, "artifacts", "column_collapse_time_law_m17534",
                     "cases_reduced.csv")


def _load():
    spec = importlib.util.spec_from_file_location("m17534_time_law", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


@pytest.fixture(scope="module")
def records():
    return M.load_reduced_csv(TABLE)


@pytest.fixture(scope="module")
def bundle(records):
    return M.adjudicate(records)


# ------------------------------------------------------------------ machinery
def test_reference_laws_match_the_paper():
    # Linear branch and its break point, then the inertial branch.
    assert M.lube_runout_ratio(0.5) == pytest.approx(1.62, abs=1e-12)
    assert M.lube_runout_ratio(1.0) == pytest.approx(2.24, abs=1e-12)
    assert M.lube_runout_ratio(1.7) == pytest.approx(1.0 + 1.6 * math.sqrt(1.7), abs=1e-12)
    assert M.lube_runout_ratio(3.0) == pytest.approx(1.0 + 1.6 * math.sqrt(3.0), abs=1e-12)
    # t_inf = 3 (a r_i / g)^(1/2)
    assert M.lube_emplacement_time(1.0, 0.48, 9.81) == pytest.approx(
        3.0 * math.sqrt(0.48 / 9.81), rel=1e-12)
    assert M.lube_emplacement_time(4.0, 0.12, 9.81) == pytest.approx(
        2.0 * M.lube_emplacement_time(1.0, 0.12, 9.81), rel=1e-12)


def test_reference_rejects_invalid_input():
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            M.lube_runout_ratio(bad)
    with pytest.raises(ValueError):
        M.lube_emplacement_time(1.0, 0.0, 9.81)


def test_sign_test_matches_binomial_tails():
    assert M.sign_test_p(25, 25) == pytest.approx(2.0 / 2 ** 25, rel=1e-12)
    assert M.sign_test_p(5, 5) == pytest.approx(2.0 / 2 ** 5, rel=1e-12)
    assert M.sign_test_p(10, 20) == pytest.approx(1.0, abs=0.6)


def test_exponent_fit_recovers_a_known_power_law():
    xs = [0.5, 1.0, 1.5, 2.0, 3.0]
    fit = M.fit_exponent(xs, [3.0 * x ** 0.5 for x in xs])
    assert fit["exponent"] == pytest.approx(0.5, abs=1e-12)
    assert fit["pearson_r"] == pytest.approx(1.0, abs=1e-12)


def test_definition_robust_violation_requires_all_three_fractions():
    def rec(label, errs):
        base = {
            "label": label, "aspect_ratio": 1.0, "aspect_measured": 1.0,
            "column_radius": 0.48, "column_radius_measured": 0.4727,
            "gravity": 9.81, "reference_emplacement_time": M.lube_emplacement_time(
                1.0, 0.48, 9.81), "mu_law": "constant",
        }
        for frac, err in zip(M.REACH_FRACTIONS, errs):
            key = {"0.9": "t_reach_f090", "0.95": "t_reach_f095", "0.99": "t_reach_f099"}[frac]
            base[key] = base["reference_emplacement_time"] * (1.0 + err)
        return base

    records = [rec("out_everywhere", (0.20, 0.20, 0.20)),
               rec("only_at_99", (-0.02, -0.02, 0.30))]
    bad = M.definition_robust_violations(records)
    assert [row["label"] for row in bad] == ["out_everywhere"]


def test_reduced_table_has_consistent_reference_columns(records):
    """The solver-stored reference time equals the paper formula (two independent
    code paths must agree) and no run was truncated before the front arrived."""
    assert len(records) == 25
    for rec in records:
        ref = M.lube_emplacement_time(rec["aspect_ratio"], rec["column_radius"],
                                      rec["gravity"])
        assert ref == pytest.approx(rec["reference_emplacement_time"], rel=1e-9)
        assert rec["runout_ratio_p995"] > 1.0
        assert rec["stop_reason"] == "at_rest", rec["label"]
        assert rec["t_reach"] < rec["time_end"], (
            f"{rec['label']}: front arrival coincides with the end of the run, "
            "which would make t_inf a truncation artefact")
        assert rec["t_reach_f090"] <= rec["t_reach_f095"] <= rec["t_reach_f099"]


# ------------------------------------------------------------- data regression
def test_runout_closure_reproduces_from_raw_provenance(bundle):
    """Round-5's 5/5 runout claim must reproduce from the stored numbers."""
    nominal = bundle["primary"]["runout_nominal"]
    assert nominal["n"] == 5 and nominal["n_pass"] == 5
    assert nominal["max_abs_error"] == pytest.approx(0.060, abs=0.005)


def test_runout_verdict_is_robust_to_the_aspect_convention(bundle):
    """Evaluating the law at the *measured* aspect ratio keeps 5/5 in band, but
    the margin grows: the nominal-a margin understates it by ~4 points."""
    measured = bundle["primary"]["runout_measured"]
    assert measured["n_pass"] == 5
    assert measured["max_abs_error"] == pytest.approx(0.099, abs=0.005)
    assert measured["max_abs_error"] > bundle["primary"]["runout_nominal"]["max_abs_error"]
    assert bundle["verdict"]["H_NOMINAL"]["rejected"] is False


def test_emplacement_time_law_is_not_satisfied(bundle):
    """The paper's second law fails on the closure arm at the strict arrival
    definition, and every case errs on the same side."""
    strict = bundle["primary"]["emplacement"]["0.99"]
    assert strict["n_pass"] == 1
    assert strict["n_positive"] == 5
    assert strict["median_error"] == pytest.approx(0.215, abs=0.01)
    assert bundle["verdict"]["H_TIME"]["rejected"] is True


def test_time_verdict_is_definition_sensitive(bundle):
    """Relaxing 'arrival' from 99 % to 90 % of the final front moves the median
    error by ~21 points: the t_inf verdict is dominated by an unregistered
    metric definition, so the honest statement is the shallow-case one below."""
    loose = bundle["primary"]["emplacement"]["0.9"]
    strict = bundle["primary"]["emplacement"]["0.99"]
    assert loose["n_pass"] == 4
    assert abs(loose["median_error"] - strict["median_error"]) > 0.15


def test_shallow_column_emplacement_time_fails_under_every_definition(bundle):
    """a = 0.5 is out of band at 90 %, 95 % and 99 % — a metric-independent
    violation, i.e. a real defect of the model at the shallow end."""
    rows = {row["label"]: row for row in bundle["emplacement"]["0.99"]["rows"]}
    shallow = [row for row in rows.values() if row["aspect"] == pytest.approx(0.5)]
    assert shallow, "a = 0.5 must be present in the closure arm"
    for row in shallow:
        assert row["error"] > 0.15
    violations = {row["label"] for row in bundle["definition_robust_violations"]}
    assert {row["label"] for row in shallow} <= violations
    assert bundle["verdict"]["H_TIME_SHALLOW"]["rejected"] is True


def test_time_scaling_matches_the_law_above_aspect_ratio_one(bundle):
    """For a >= 1 the fitted exponent is ~0.45-0.47, i.e. consistent with the
    a^(1/2) of the reference law; the shallow case is what pulls the whole-arm
    fit down to ~0.36."""
    scaling = bundle["scaling"]
    ge1 = scaling["t_reach_99_primary_a_ge_1"]["exponent"]
    assert 0.42 <= ge1 <= 0.52, ge1
    assert scaling["t_reach_99_primary"]["exponent"] < 0.40
    assert bundle["verdict"]["H_EXPONENT"]["rejected"] is False


def test_every_case_energy_ledger_is_finite(records):
    for rec in records:
        assert math.isfinite(rec["energy_residual_fraction"])
        assert rec["energy_residual_fraction"] < 0.25, rec["label"]

def test_manifest_binds_the_reduced_table_to_its_sources():
    """The reduced table must stay traceable: the manifest carries one sha256 per
    raw provenance file plus the digest of the committed CSV itself."""
    import hashlib
    import json

    out_dir = os.path.dirname(TABLE)
    with open(os.path.join(out_dir, "manifest_sha256.json")) as fh:
        manifest = json.load(fh)
    assert manifest["n_cases"] == 25
    assert len(manifest["sources"]) == 25
    for name, digest in manifest["sources"].items():
        assert len(digest) == 64 and name.endswith(".json")
    h = hashlib.sha256()
    with open(TABLE, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    assert manifest["artifacts"]["cases_reduced.csv"] == h.hexdigest()
