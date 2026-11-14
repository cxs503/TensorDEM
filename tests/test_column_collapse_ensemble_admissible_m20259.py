"""Gates for m20259: initial-state admissibility + no vacuous pass.

Two independent classes:

1. **Machinery** (synthetic input) — the overlap census, the admissibility
   bracket, the tri-state ensemble adjudication.  The controls *must* fail:
   an overlapping packing is inadmissible, a one-member "ensemble" has an
   undefined sigma, and a NaN statistic must never be read as a pass.
2. **Data regression** — the committed bundle
   ``docs/artifacts/column_collapse_ensemble_m20259/`` is re-adjudicated from
   its own rows; assertions are relations (reference from the Lube formulas,
   verdicts from the pre-registered thresholds), never typed numbers.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "docs", "artifacts", "column_collapse_ensemble_m20259")
BUNDLE = os.path.join(ART, "summary.json")
CASES = os.path.join(ART, "cases_ensemble.csv")
M19922 = os.path.join(ROOT, "scripts", "analyze_column_collapse_ensemble_m19922.py")
M20259 = os.path.join(ROOT, "scripts", "analyze_column_collapse_ensemble_m20259.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------- geometry ---------------------------------- #
def test_overlap_census_flags_a_penetrating_pair():
    from tensordem.granular_column.admissibility import overlap_stats
    d = 0.06
    ok = np.array([[0.0, 0, 0], [d * 1.001, 0, 0]])
    bad = np.array([[0.0, 0, 0], [d * 0.98, 0, 0]])
    assert overlap_stats(ok, 0.03)["admissible"] is True
    st = overlap_stats(bad, 0.03)
    assert st["admissible"] is False
    assert st["overlap_pairs"] == 1
    assert st["max_overlap"] == pytest.approx(d * 0.02, rel=1e-12)


def test_admissibility_bracket_is_a_real_bracket():
    """The lattice is admissible, a large jitter is not, and the reported
    bracket separates them (no fitted threshold)."""
    from tensordem.granular_column.admissibility import max_admissible_jitter
    r = 0.03

    def build(j):
        # 2 % clearance lattice; a jitter of j*r_p per coordinate moves the
        # neighbour by 2*j*r_p, so the clearance is spent at j = 0.01
        pos = np.zeros((2, 3))
        pos[1, 0] = 2 * r * 1.02 - 2 * j * r
        return pos

    res = max_admissible_jitter(build, r, (0.0, 0.005, 0.05, 0.10))
    assert res["jitter_max_admissible"] == pytest.approx(0.005)
    assert res["jitter_first_rejecting"] == pytest.approx(0.05)
    assert res["rule"].startswith("overlap_pairs == 0")


def test_admissible_wave_gate_rejects_an_overlapping_spec():
    B = _load(os.path.join(ROOT, "benchmarks",
                          "case_column_collapse_ensemble_admissible_m20259.py"),
              "m20259_bench")
    spec = B.admissible_cases((0.5,))[0]
    assert B.gate(spec)["admissible"] is True
    spec_bad = dict(spec, params=dict(spec["params"], jitter=0.10))
    assert B.gate(spec_bad)["admissible"] is False


# --------------------- tri-state adjudication (m19922 script) -------------- #
T19922 = _load(os.path.join(ROOT, "tests",
                           "test_column_collapse_ensemble_m19922.py"), "t19922")


def test_single_realisation_scatter_is_undefined_not_zero():
    M = T19922.M
    ref = M.lube_2004_runout_ratio(1.0)
    rows = [M.reduce_case(T19922._raw(1.0, 0.05, 1, ref))]
    s = M.summarize(rows)[1.0]
    assert s["n_jitter"] == 1
    assert math.isnan(s["runout_sigma"])
    assert s["spread_ok"] is None and s["spread_testable"] is False
    v = M.hypothesis_verdicts({1.0: s}, {"all_match": True, "n_checked": 1})
    assert v["H_SPREAD"]["pass"] is None
    assert v["H_SPREAD"]["status"] == "insufficient"


def test_nan_jitter_shift_is_insufficient_not_pass():
    M = T19922.M
    ref = M.lube_2004_runout_ratio(0.5)
    rows = [M.reduce_case(T19922._raw(0.5, 0.0, 1, ref))]
    v = M.hypothesis_verdicts(M.summarize(rows), {"all_match": True, "n_checked": 1})
    assert v["H_JITTER"]["pass"] is None
    assert v["H_JITTER"]["status"] == "insufficient"


def test_missing_archive_makes_repro_insufficient():
    M = T19922.M
    v = M.hypothesis_verdicts({}, {"all_match": None})
    assert v["H_REPRO"]["pass"] is None
    assert v["H_REPRO"]["status"] == "insufficient"


# ----------------------------- data regression ----------------------------- #
pytestmark_bundle = pytest.mark.skipif(
    not os.path.exists(BUNDLE), reason="m20259 bundle not built yet")


@pytestmark_bundle
def test_bundle_rows_recompute_their_own_reference_and_error():
    B = _load(M20259, "m20259_reducer")
    with open(CASES) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 8
    with open(BUNDLE) as fh:
        bundle = json.load(fh)
    for r in rows:
        assert float(r["ref_runout_ratio"]) == pytest.approx(
            B.lube_2004_runout_ratio(float(r["aspect_nominal"])), abs=1e-12)
        assert float(r["err_runout"]) == pytest.approx(
            float(r["runout_ratio_p995"]) / float(r["ref_runout_ratio"]) - 1.0,
            abs=1e-12)
    assert bundle["n_cases"] == len(rows)


@pytestmark_bundle
def test_bundle_verdicts_follow_from_its_own_rows():
    B = _load(M20259, "m20259_reducer_v")
    with open(BUNDLE) as fh:
        bundle = json.load(fh)
    rows = bundle["rows"]
    again = B.adjudicate(rows)
    for h, v in again.items():
        assert v["status"] == bundle["verdicts"][h]["status"], h
        assert v["pass"] == bundle["verdicts"][h]["pass"], h
    # the pre-registered m19922 amplitudes are geometrically inadmissible
    bad = [r for r in rows if r["wave"] == "inadmissible" and r["jitter"] > 0]
    assert bad and all(r["overlap_pairs"] > 0 for r in bad)
    adm = [r for r in rows if r["wave"] == "admissible"]
    assert adm and all(r["overlap_pairs"] == 0 for r in adm)
    # verdict consistency: a fail is never reported as a pass
    for h, v in bundle["verdicts"].items():
        assert (v["status"] == "pass") == (v["pass"] is True), h
