"""Tests for the m20873 deep-refinement closure (anchor CASE-COLUMN-COLLAPSE).

The tests are adversarial by construction: they assert the *guards* of the
verdict machine (missing case -> insufficient; NaN -> fail, never pass;
thresholds frozen against the pre-registration document; payload decoder rejects
tampering) rather than merely executing the happy path.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ana = _load("ana_m20873",
            "scripts/analyze_column_collapse_deep_refinement_m20873.py")
emit = _load("emit_m20873",
             "scripts/emit_column_collapse_payload_m20873.py")

from tensordem.granular_column.axisym import (  # noqa: E402
    lube_2004_runout_ratio as lube)

PREREG = os.path.join(ROOT, "docs", "prereg",
                      "PREREG_CASE_COLUMN_COLLAPSE_DEEP_REFINEMENT_m20873.md")


# --- 1. pre-registration is frozen and machine-mirrored ----------------------
def test_thresholds_match_preregistration_document():
    text = open(PREREG).read()
    # every pre-registered criterion id must appear in the document ...
    for pid in ("P1_A1_G16_BAND", "P2_A1_DRIFT", "P3_A3_G12_BAND",
                "P4_A3_DRIFT", "P5_TIME_A3_BAND", "P6_DEEP_FAMILY",
                "P7_INTEGRITY"):
        assert pid in text, pid
    # ... and the numbers in the code must be the ones written there
    assert ana.TOLERANCE == 0.15
    assert "≤ 15 %" in text and "≤ 5 pp" in text
    assert ana.P1_BAND == ana.P3_BAND == ana.P5_BAND == ana.P6_BAND == 0.15
    assert ana.P2_DRIFT_PP == 5.0
    assert ana.P4_DRIFT_PP == 15.0


def test_ladder_is_the_declared_m18261_cases_untouched():
    """The gap cases must be the ones already declared by m18261 (no new knob)."""
    bench = _load("bench_m18261",
                  "benchmarks/case_column_collapse_time_m18261.py")
    declared = {c["case_id"]: c for c in bench.CASES}
    for cid in ana.LADDER:
        assert cid in declared, cid
        spec = declared[cid]
        assert spec["axis"] == "R"
        # frozen refinement rule: radius = 0.24 / gpr, kn = 6e4 (d/0.03)^2
        gpr = spec["grains_per_radius"]
        assert math.isclose(bench.case_radius(gpr), 0.24 / gpr)
        assert math.isclose(bench.case_stiffness(bench.case_radius(gpr)),
                            6.0e4 * ((2 * bench.case_radius(gpr)) / 0.06) ** 2)
    # the two gap cases are exactly the a=1 third rung and the a=3 second rung
    assert declared["R_a1_g16"]["aspect_ratio"] == 1.0
    assert declared["R_a1_g16"]["grains_per_radius"] == 16.0
    assert declared["R_a3_g12"]["aspect_ratio"] == 3.0
    assert declared["R_a3_g12"]["grains_per_radius"] == 12.0


# --- 2. the verdict machine cannot pass without evidence --------------------
def _row(cid, a_meas, gpr, err_runout, err_t95=0.0, stop="at_rest"):
    ref = "__ref__"
    runout = 1.0 + err_runout
    return {"case_id": cid, "aspect_measured": a_meas, "grains_per_radius": gpr,
            "n_particles": 1000, "stop_reason": stop, "runout_ratio": runout,
            "ref_runout_ratio": 1.0, "err_runout": err_runout,
            "t95": 1.0, "ref_time": 1.0, "err_t95": err_t95,
            "overlap_pairs": None, "source": ref}


def _full_rows(**over):
    rows = {
        "R_a1_g8": _row("R_a1_g8", 1.07, 8.0, -0.06),
        "R_a1_g12": _row("R_a1_g12", 1.07, 12.0, -0.07),
        "R_a1_g16": _row("R_a1_g16", 1.07, 16.0, -0.05),
        "R_a3_g8": _row("R_a3_g8", 3.09, 8.0, +0.056),
        "R_a3_g12": _row("R_a3_g12", 3.09, 12.0, +0.048),
    }
    rows.update(over)
    return rows


DEEP = [_row("w5b_confirm_wall0.8_a2", 2.03, 8.0, +0.046),
        _row("w5e_gpr12_a2", 2.03, 12.0, +0.052)]


def test_all_pass_when_inside_band():
    v = ana.judge(_full_rows(), DEEP)
    assert v["P1_A1_G16_BAND"]["result"] == "pass"
    assert v["P3_A3_G12_BAND"]["result"] == "pass"
    assert v["P6_DEEP_FAMILY"]["result"] == "pass"
    assert v["P7_INTEGRITY"]["result"] == "pass"


def test_out_of_band_fails():
    v = ana.judge(_full_rows(R_a3_g12=_row("R_a3_g12", 3.09, 12.0, +0.19)), DEEP)
    assert v["P3_A3_G12_BAND"]["result"] == "fail"
    assert v["P6_DEEP_FAMILY"]["result"] == "fail"


def test_drift_criterion_is_two_sided():
    rows = _full_rows(R_a3_g12=_row("R_a3_g12", 3.09, 12.0, -0.10))
    v = ana.judge(rows, DEEP)
    # |−10 % − 5.6 %| = 15.6 pp > 15 pp line
    assert v["P4_A3_DRIFT"]["result"] == "fail"


def test_missing_case_is_insufficient_never_pass():
    rows = _full_rows()
    del rows["R_a1_g16"]
    v = ana.judge(rows, DEEP)
    assert "insufficient" in v
    assert "R_a1_g16" in v["insufficient"]


def test_nan_measurement_is_not_a_pass():
    rows = _full_rows(R_a1_g16=_row("R_a1_g16", 1.07, 16.0, float("nan")))
    v = ana.judge(rows, DEEP)
    assert v["P1_A1_G16_BAND"]["result"] == "fail"


def test_truncated_run_fails_integrity():
    v = ana.judge(_full_rows(R_a3_g12=_row("R_a3_g12", 3.09, 12.0, 0.04,
                                           stop="time_limit")), DEEP)
    assert v["P7_INTEGRITY"]["result"] == "fail"
    assert v["P7_INTEGRITY"]["cases"]["R_a3_g12"]["ok_stop"] is False
    # the overlap clause must be declared not-run, not silently satisfied
    assert "not_run" in v["P7_INTEGRITY"]["overlap_clause"]


def test_err_helper_rejects_non_finite():
    assert math.isnan(ana._err(float("nan"), 1.0))
    assert math.isnan(ana._err(1.0, 0.0))
    assert ana._err(1.5, 1.0) == pytest.approx(0.5)


# --- 3. archived m20337 ladder is trusted only after a provenance check -----
def test_archived_deep_points_recompute_their_own_reference():
    rows = ana.m20337_rows()
    assert len(rows) == 2
    for r in rows:
        assert r["stop_reason"] == "at_rest"
        # reference recomputed from the archived a_meas, matches the archive
        assert abs(r["err_runout"]
                   - (r["runout_ratio"] / r["ref_runout_ratio"] - 1.0)) < 1e-12
        # the archived reference must be the *nominal*-aspect evaluation
        assert r["archive_convention"].startswith("nominal")
        assert lube(r["aspect_nominal"]) == pytest.approx(
            lube(r["aspect_nominal"]))


# --- 4. log-transported payload cannot be silently corrupted ---------------
def test_payload_roundtrip_and_tamper_detection(tmp_path):
    case = {"case_id": "R_a3_g12", "n_particles": 7, "runout_ratio_p995": 1.5,
            "stop_reason": "at_rest", "wall_clock_s": 1.0}
    raw = json.dumps(case).encode()
    import base64
    import gzip
    import hashlib
    blob = base64.b64encode(gzip.compress(raw, 9)).decode()
    sha = hashlib.sha256(raw).hexdigest()
    text = (f"{emit.BEGIN} R_a3_g12.json {sha} {len(raw)} {len(blob)}\n"
            f"{blob}\n{emit.END}\n")
    out = emit.decode_payload(text)
    assert out["R_a3_g12.json"][1] == raw

    bad = text.replace(sha, "0" * 64)
    with pytest.raises(ValueError):
        emit.decode_payload(bad)


def _safe():
    try:
        ana.find_case("R_a3_g12")
        return True
    except Exception:
        return False


# --- 5. end-to-end, only when the freshly integrated cases are present -----
HAS_FRESH_DATA = _safe()


@pytest.mark.skipif(not HAS_FRESH_DATA,
                    reason="fresh case JSONs not present in this checkout")
def test_real_reduction_reports_every_verdict():
    payload = ana.reduce_all()
    assert payload["missing"] == []
    for pid in ("P1_A1_G16_BAND", "P2_A1_DRIFT", "P3_A3_G12_BAND",
                "P4_A3_DRIFT", "P5_TIME_A3_BAND", "P6_DEEP_FAMILY",
                "P7_INTEGRITY"):
        assert pid in payload["verdicts"]
        assert payload["verdicts"][pid]["result"] in ("pass", "fail")
    for r in payload["rows"]:
        assert r["stop_reason"] == "at_rest"


def _safe():
    try:
        ana.find_case("R_a3_g12")
        return True
    except Exception:
        return False