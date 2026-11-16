"""Adjudication of the pre-registered isochoric sensitivity matrix.

The round-3 document (``docs/CASE-DEM-BIAXIAL-FRICTION_R3_ISOCHORIC.md``) left
§5.5 / §9.3 open: the four sensitivity configurations (grid resolution, particle
friction, confining stress, strain rate) had never produced a stress path, so
the published ``phi_cv = 14.3 deg`` could not be separated from a discretisation
artefact.  ``prereg/PREREG_m22748_ISOCHORIC_SENSITIVITY.json`` fixed the
decision rules *before* those runs existed; this module tests the code that
applies them:

* the rules are read from the committed pre-registration and cannot be
  re-declared from inside the analyser (tolerances, decision keys and the
  outcome metric are pinned here);
* the adjudicator is pure, honours the 2.0 deg invariance bands exactly at the
  boundary, enforces the quasi-static inertial-number condition on the
  rate rule, and treats a missing configuration as ``pending`` rather than
  extrapolating;
* only a complete matrix may produce a verdict, and the verdict flips between
  ``H1_shape_supported`` (all rules pass) and ``H2_discretisation_supported``
  (any rule fails);
* when the measured matrix is committed
  (``results_m22832/sensitivity.json.gz``), the verdict and every ``phi_cv`` it
  rests on are pinned, so the document's §9.3 table cannot drift from the data.
"""
from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import math
import os

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANALYSER = os.path.join(REPO, "benchmarks", "case_dem_biaxial_friction",
                        "analyse_isochoric.py")
PREREG = os.path.join(REPO, "prereg", "PREREG_m22748_ISOCHORIC_SENSITIVITY.json")
CASE_DIR = os.path.join(REPO, "benchmarks", "case_dem_biaxial_friction")
BUNDLE = os.path.join(CASE_DIR, "results_m22832", "iso_sensitivity.json.gz")
EXPECTED = os.path.join(CASE_DIR, "results_m22832", "sensitivity_expected.json")

ANCHOR = "anchor_24x48_mu055_p100"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def analyser():
    return _load("analyse_isochoric_sensitivity", ANALYSER)


def case(tag: str, phi: float, *, mu: float = 0.55, inertial: float = 1.44e-3,
         rowe_err: float = 0.50, p_window: float = 325.0) -> dict:
    """Minimal stand-in for one analysed configuration."""
    return {
        "tag": tag,
        "config": {"mu": mu},
        "loading": {"inertial_number": inertial},
        "critical_state": {"phi_cv_deg": phi},
        "mean_pressure_in_window_kPa": p_window,
        "judge_primary_rowe": {"relative_error": rowe_err},
    }


def full_matrix(analyser, *,
                res=14.5, mu030=9.6, p200=14.1, rate=14.4,
                mu_rate_inertial=7.2e-4):
    return [
        case(ANCHOR, 14.3),
        case("res_16x32_mu055_p100", res),
        case("mu_24x48_mu030_p100", mu030, mu=0.30),
        case("pressure_24x48_mu055_p200", p200, p_window=640.0),
        case("rate_24x48_mu055_p100_slow", rate, inertial=mu_rate_inertial),
    ]


# --------------------------------------------------------------------------
# 1. the pre-registration itself is frozen
# --------------------------------------------------------------------------
def test_prereg_rules_are_pinned():
    with open(PREREG) as fh:
        prereg = json.load(fh)
    rules = prereg["decision_rules_predeclared"]
    assert set(rules) == {"resolution_invariance", "friction_monotonicity",
                          "pressure_invariance", "rate_invariance"}
    for key in ("resolution_invariance", "pressure_invariance", "rate_invariance"):
        assert "2.0 deg" in rules[key], key          # tolerance may not drift
    assert "strictly" in rules["friction_monotonicity"]
    assert "sin(phi_cv)/mu" in rules["friction_monotonicity"]
    assert "1e-3" in rules["rate_invariance"]
    # the outcome metric and the stopping rule are part of the contract
    assert "strain_measure='log'" in prereg["outcome_metric"]
    assert "no re-tuning" in prereg["stopping_rule"]
    assert set(prereg["pending_configurations"]) == {
        "res_16x32_mu055_p100", "mu_24x48_mu030_p100",
        "pressure_24x48_mu055_p200", "rate_24x48_mu055_p100_slow"}


def test_adjudicator_reads_tolerance_from_prereg(analyser):
    with open(PREREG) as fh:
        prereg = json.load(fh)
    assert analyser.INVARIANCE_TOL_DEG == 2.0
    assert "2.0 deg" in prereg["decision_rules_predeclared"]["resolution_invariance"]
    assert analyser.load_prereg()["mission"] == prereg["mission"]


# --------------------------------------------------------------------------
# 2. missing configurations are pending, never extrapolated
# --------------------------------------------------------------------------
def test_anchor_only_is_pending(analyser):
    adj = analyser.adjudicate_sensitivity([case(ANCHOR, 14.3)])
    assert adj["verdict"] == "pending"
    assert len(adj["pending_configs"]) == 4
    assert all(c["passed"] is None for c in adj["checks"])
    assert "no verdict" in adj["verdict_statement"]


def test_missing_anchor_is_an_error(analyser):
    with pytest.raises(ValueError):
        analyser.adjudicate_sensitivity([case("res_16x32_mu055_p100", 14.5)])


# --------------------------------------------------------------------------
# 3. invariance rules: boundary, sign, and the inertial-number gate
# --------------------------------------------------------------------------
def test_resolution_rule_at_exact_boundary(analyser):
    at_bound = analyser.adjudicate_sensitivity(
        [case(ANCHOR, 14.30), case("res_16x32_mu055_p100", 16.30)])
    check = next(c for c in at_bound["checks"]
                 if c["rule"] == "resolution_invariance")
    assert check["delta_deg"] == pytest.approx(2.0, abs=1e-9)
    assert check["passed"] is True                      # closed band, inclusive

    beyond = analyser.adjudicate_sensitivity(
        [case(ANCHOR, 14.30), case("res_16x32_mu055_p100", 16.31)])
    check = next(c for c in beyond["checks"]
                 if c["rule"] == "resolution_invariance")
    assert check["passed"] is False


def test_pressure_rule_uses_absolute_delta(analyser):
    adj = analyser.adjudicate_sensitivity(
        [case(ANCHOR, 14.30), case("pressure_24x48_mu055_p200", 11.80,
                                   p_window=640.0)])
    check = next(c for c in adj["checks"] if c["rule"] == "pressure_invariance")
    assert check["passed"] is False
    assert check["delta_deg"] == pytest.approx(-2.5, abs=1e-9)


def test_rate_rule_requires_quasi_static_inertial_number(analyser):
    # angle agrees, but the run is not quasi-static -> rule must fail
    fast = analyser.adjudicate_sensitivity(
        [case(ANCHOR, 14.30),
         case("rate_24x48_mu055_p100_slow", 14.40, inertial=2.0e-3)])
    check = next(c for c in fast["checks"] if c["rule"] == "rate_invariance")
    assert check["passed"] is False
    assert "inertial number" in check["note"]

    slow = analyser.adjudicate_sensitivity(
        [case(ANCHOR, 14.30),
         case("rate_24x48_mu055_p100_slow", 14.40, inertial=7.2e-4)])
    check = next(c for c in slow["checks"] if c["rule"] == "rate_invariance")
    assert check["passed"] is True


# --------------------------------------------------------------------------
# 4. friction rule: strict monotonicity + reported sin(phi_cv)/mu
# --------------------------------------------------------------------------
def test_friction_monotonicity_and_ratio(analyser):
    ok = analyser.adjudicate_sensitivity(
        [case(ANCHOR, 14.30, mu=0.55), case("mu_24x48_mu030_p100", 9.60, mu=0.30)])
    check = next(c for c in ok["checks"] if c["rule"] == "friction_monotonicity")
    assert check["passed"] is True
    assert check["sin_phi_over_mu"] == pytest.approx(
        math.sin(math.radians(9.60)) / 0.30, rel=1e-12)
    assert check["sin_phi_over_mu_baseline"] == pytest.approx(
        math.sin(math.radians(14.30)) / 0.55, rel=1e-12)

    inverted = analyser.adjudicate_sensitivity(
        [case(ANCHOR, 14.30, mu=0.55), case("mu_24x48_mu030_p100", 15.00, mu=0.30)])
    check = next(c for c in inverted["checks"]
                 if c["rule"] == "friction_monotonicity")
    assert check["passed"] is False


# --------------------------------------------------------------------------
# 5. only a complete matrix decides H1 vs H2
# --------------------------------------------------------------------------
def test_complete_passing_matrix_supports_shape_hypothesis(analyser):
    adj = analyser.adjudicate_sensitivity(full_matrix(analyser))
    assert adj["verdict"] == "H1_shape_supported"
    assert adj["pending_configs"] == []
    assert all(c["passed"] for c in adj["checks"])
    assert "shape/kinematics" in adj["verdict_statement"]


def test_one_failing_rule_flips_to_discretisation(analyser):
    adj = analyser.adjudicate_sensitivity(
        full_matrix(analyser, res=18.0))         # refining the grid moves phi up
    assert adj["verdict"] == "H2_discretisation_supported"
    assert "configuration dependent" in adj["verdict_statement"]


def test_verdict_withholds_when_a_config_is_still_missing(analyser):
    rows = [r for r in full_matrix(analyser) if r["tag"] != "pressure_24x48_mu055_p200"]
    adj = analyser.adjudicate_sensitivity(rows)
    assert adj["verdict"] == "pending"
    assert adj["pending_configs"] == ["pressure_24x48_mu055_p200"]


def test_adjudication_is_pure(analyser):
    rows = full_matrix(analyser)
    snapshot = copy.deepcopy(rows)
    analyser.adjudicate_sensitivity(rows)
    assert rows == snapshot


def test_markdown_table_reports_every_rule(analyser):
    md = analyser.markdown_sensitivity_table(
        analyser.adjudicate_sensitivity(full_matrix(analyser)))
    for rule in ("resolution_invariance", "friction_monotonicity",
                 "pressure_invariance", "rate_invariance"):
        assert rule in md
    assert md.count("PASS") == 4
    assert "H1_shape_supported" in md


# --------------------------------------------------------------------------
# 6. the committed measured matrix (present only once the runs landed)
# --------------------------------------------------------------------------
def test_measured_matrix_matches_committed_verdict(analyser):
    if not os.path.exists(BUNDLE):
        pytest.skip("measured sensitivity matrix not committed yet")
    rows, _ = analyser.load_runs(os.path.dirname(BUNDLE))
    assert len(rows) == 5, sorted(r["tag"] for r in rows)
    cases = [analyser.analyse_one(raw) for raw in rows]
    adj = analyser.adjudicate_sensitivity(cases)
    assert adj["pending_configs"] == []
    assert os.path.exists(EXPECTED), "measured matrix committed without its pins"
    with open(EXPECTED) as fh:
        expected = json.load(fh)
    assert adj["verdict"] == expected["verdict"]
    for c in adj["checks"]:
        assert c["phi_cv_deg"] == pytest.approx(
            expected["phi_cv_deg"][c["config"]], abs=0.15), c["config"]
        assert c["passed"] == expected["passed"][c["rule"]], c["rule"]