"""Unit tests for the breakpoint-neighbourhood scan (m21007).

Pure-function coverage only: no DEM integration runs here (those are the
pre-registered cases in ``benchmarks/case_column_collapse_kink_m21007.py``).
What is asserted:

* the analytic facts of the Lube (2004) piecewise law used to interpret the
  scan (crossover, the law's own 0.71 % jump at the card's break, slope ratio);
* that the frozen closed arm / refinement semantics of m18261 are reused, not
  redefined;
* the pre-registered hypothesis thresholds and case set are frozen;
* every branch of :func:`judge` (pass / fail / incomplete), including the
  provenance defect guard (arrival time read from ``primary.fraction_0.9``,
  never from a column label).
"""
import importlib.util as _ilu
import json
import math
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "benchmarks" / "case_column_collapse_kink_m21007.py"


@pytest.fixture(scope="module")
def kn():
    spec = _ilu.spec_from_file_location("_cc_kink_m21007", _SCRIPT)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# analytic reference-law facts
# ---------------------------------------------------------------------------
def test_crossover_is_where_branches_intersect(kn):
    a_star = kn.crossover_aspect()
    assert a_star == pytest.approx((1.6 / 1.24) ** 2, rel=1e-15)
    assert a_star == pytest.approx(1.6649323621227892, rel=1e-12)
    # Below the crossover the linear branch is the lower one, above it the
    # square-root branch is; at the crossover they coincide.
    eps = 1e-9
    lin = lambda a: 1.0 + 1.24 * a
    rt = lambda a: 1.0 + 1.6 * math.sqrt(a)
    assert lin(a_star - eps) < rt(a_star - eps)
    assert rt(a_star + eps) < lin(a_star + eps)


def test_law_discontinuity_at_card_break(kn):
    out = kn.law_discontinuity()
    assert out["aspect"] == 1.7
    # The library law switches to the sqrt branch at exactly a = 1.7.
    assert kn.lube_2004_runout_ratio(1.7) == pytest.approx(out["value_sqrt_branch"])
    assert kn.lube_2004_runout_ratio(1.699999) == pytest.approx(
        out["value_linear_branch"], rel=1e-6)
    # The law's own jump at the card's break is 0.71 %, i.e. far below the 15 %
    # tolerance -> the break cannot explain any out-of-band DEM error.
    assert out["relative_jump"] == pytest.approx(0.007081725572298962, abs=1e-12)
    assert out["relative_jump"] < 0.01
    assert out["break_minus_crossover"] > 0.0
    # Slope contrast across the break is ~2x.
    assert out["slope_ratio_across_break"] == pytest.approx(2.020952745612821, rel=1e-12)


def test_law_discontinuity_rejects_nonpositive(kn):
    with pytest.raises(Exception):
        kn.law_discontinuity(0.0)


def test_local_slope_basic(kn):
    assert kn.local_slope(1.0, 2.0, 2.0, 4.0) == pytest.approx(2.0)
    assert kn.local_slope(1.0, 4.0, 1.0, 5.0) != kn.local_slope(1.0, 4.0, 1.0, 5.0)  # nan
    assert math.isnan(kn.local_slope(1.5, 3.0, 1.5, 3.5))


# ---------------------------------------------------------------------------
# frozen protocol reuse (no redefinition of the closed arm)
# ---------------------------------------------------------------------------
def test_pre_registered_case_set_is_frozen(kn):
    ids = [c["case_id"] for c in kn.NEW_CASES]
    assert ids == ["KN_a1.5_g12", "KN_a1.6_g12", "KN_a1.7_g12",
                   "KN_a1.8_g12", "KN_a1.7_g8"]
    aspects = sorted({c["aspect_ratio"] for c in kn.NEW_CASES})
    assert aspects == [1.5, 1.6, 1.7, 1.8]
    assert kn.TOLERANCE == 0.15
    assert kn.MAX_ERROR_JUMP_PP == 5.0
    assert kn.MIN_SLOPE_RATIO == 1.0


def test_closed_arm_and_refinement_semantics_reused(kn):
    """r_i fixed, r = r_i/(2 gpr), k_n scaled to keep the sag ratio: m18261."""
    assert kn.BASE_MOD.BASE["mu"] == 0.5
    assert kn.BASE_MOD.BASE["mu_rolling"] == 0.5
    assert kn.BASE_MOD.BASE["mu_wall"] == 0.8
    assert kn.BASE_MOD.BASE["jitter"] == 0.0
    assert kn.BASE_MOD.COLUMN_RADIUS == 0.48
    case12 = kn.BASE_MOD.make_case(kn.CASE_BY_ID["KN_a1.7_g12"])
    case8 = kn.BASE_MOD.make_case(kn.CASE_BY_ID["KN_a1.7_g8"])
    assert case12.radius == pytest.approx(0.48 / 24.0)
    assert case8.radius == pytest.approx(0.48 / 16.0)
    # dimensionless sag ratio delta/d = pi rho g d^2 / (6 k_n) is held fixed
    sag12 = math.pi * case12.density * case12.gravity * case12.radius ** 2 / (6 * case12.kn)
    sag8 = math.pi * case8.density * case8.gravity * case8.radius ** 2 / (6 * case8.kn)
    assert sag12 == pytest.approx(sag8, rel=1e-12)
    assert case12.aspect_ratio == 1.7


# ---------------------------------------------------------------------------
# point_row / provenance
# ---------------------------------------------------------------------------
def _fake_case(case_id, aspect, runout, ref, t90, gpr=12.0, t95=9.99):
    """Stored-case JSON shaped like the m18261 runner writes."""
    return {
        "case_id": case_id,
        "aspect_nominal": aspect,
        "aspect_measured": aspect,
        "n_particles": 100,
        "dt": 1e-4,
        "n_steps": 10,
        "stop_reason": "at_rest",
        "wall_clock_s": 1.0,
        "runout_ratio_p995": runout,
        "protocol": {"grains_per_radius": gpr},
        "reference": {
            "measured": {"runout_ratio": ref, "emplacement_time": 1.0},
            "nominal": {"runout_ratio": ref, "emplacement_time": 1.0},
        },
        # provenance trap: a mislabelled t95 column (the m20873 defect)
        "arrival": {"primary": {"fraction_0.9": t90, "t95": t95}},
        "energy": {"residual_fraction_of_potential": 0.05},
    }


def test_point_row_reads_primary_definition_not_labels(kn, tmp_path):
    path = tmp_path / "KN_a1.7_g12.json"
    path.write_text(json.dumps(_fake_case("KN_a1.7_g12", 1.7, 3.0, 3.0, 1.1, t95=7.0)))
    row = kn.point_row(str(path), "test")
    assert row["t90"] == pytest.approx(1.1)          # fraction_0.9, not t95
    assert row["err_runout"] == pytest.approx(0.0)
    assert row["err_t90"] == pytest.approx(0.1)


def test_point_row_rejects_zero_reference(kn, tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps(_fake_case("x", 1.7, 3.0, 0.0, 1.0)))
    assert math.isnan(kn.point_row(str(path), "test")["err_runout"])


# ---------------------------------------------------------------------------
# judge: every branch
# ---------------------------------------------------------------------------
def _rows(kn, errs, gpr8_err=None, runout_scale=1.0):
    """Build judge() input from target relative errors (times 100 = %)."""
    out = []
    for (case_id, aspect), err in zip(
            [("KN_a1.5_g12", 1.5), ("KN_a1.6_g12", 1.6),
             ("KN_a1.7_g12", 1.7), ("KN_a1.8_g12", 1.8)], errs):
        ref = kn.lube_2004_runout_ratio(aspect)
        runout = ref * (1.0 + err) * runout_scale
        out.append({"case_id": case_id, "aspect_nominal": aspect,
                    "aspect_measured": aspect, "grains_per_radius": 12.0,
                    "runout_ratio": runout, "ref_runout_ratio": ref,
                    "err_runout": err, "t90": 1.0, "ref_time": 1.0,
                    "err_t90": 0.0, "energy_residual": 0.05,
                    "stop_reason": "at_rest", "source": "test"})
    if gpr8_err is not None:
        ref = kn.lube_2004_runout_ratio(1.7)
        out.append({"case_id": "KN_a1.7_g8", "aspect_nominal": 1.7,
                    "aspect_measured": 1.7, "grains_per_radius": 8.0,
                    "runout_ratio": ref * (1.0 + gpr8_err), "ref_runout_ratio": ref,
                    "err_runout": gpr8_err, "t90": 1.0, "ref_time": 1.0,
                    "err_t90": 0.0, "energy_residual": 0.05,
                    "stop_reason": "at_rest", "source": "test"})
    return out


def test_judge_incomplete_without_all_gpr12_points(kn):
    rows = _rows(kn, [-0.02, -0.01, 0.0, 0.01], gpr8_err=0.0)[:3]
    v = kn.judge(rows)
    assert v["verdict"] == "incomplete"
    assert "KN_a1.8_g12" in v["missing"]


def test_judge_pass_path(kn):
    # errors continue smoothly through the break -> H1/H2 pass, band holds
    rows = _rows(kn, [-0.03, -0.02, -0.01, 0.01], gpr8_err=-0.015)
    v = kn.judge(rows)
    assert v["H1_band"]["verdict"] == "pass"
    assert v["H2_no_branch_artefact"]["verdict"] == "pass"
    assert v["H4_resolution"]["verdict"] == "pass"
    assert v["overall"] in ("pass", "incomplete")  # H3 depends on synthetic slopes


def test_judge_h1_fails_when_a_point_leaves_band(kn):
    rows = _rows(kn, [-0.03, 0.16, 0.0, 0.01], gpr8_err=0.0)
    v = kn.judge(rows)
    assert v["H1_band"]["verdict"] == "fail"
    assert v["H1_band"]["all_in_band"] is False
    assert v["H1_band"]["worst"] == pytest.approx(0.16)
    assert v["overall"] == "fail"


def test_judge_h2_fails_on_error_jump_across_break(kn):
    rows = _rows(kn, [-0.03, -0.02, 0.10, 0.11], gpr8_err=0.10)
    v = kn.judge(rows)
    assert v["H2_no_branch_artefact"]["verdict"] == "fail"
    assert v["H2_no_branch_artefact"]["jump_below_pp"] == pytest.approx(12.0, abs=1e-6)
    assert v["overall"] == "fail"


def test_judge_h4_incomplete_without_gpr8_control(kn):
    rows = _rows(kn, [-0.03, -0.02, -0.01, 0.01], gpr8_err=None)
    v = kn.judge(rows)
    assert v["H4_resolution"]["verdict"] == "incomplete"
    assert v["overall"] == "incomplete"


def test_judge_h4_fails_when_resolution_moves_the_point(kn):
    rows = _rows(kn, [-0.03, -0.02, -0.01, 0.01], gpr8_err=-0.20)
    v = kn.judge(rows)
    assert v["H4_resolution"]["verdict"] == "fail"
    assert v["H4_resolution"]["delta_pp"] == pytest.approx(19.0, abs=1e-6)


def test_judge_h3_reports_law_slopes(kn):
    rows = _rows(kn, [-0.03, -0.02, -0.01, 0.01], gpr8_err=-0.015)
    v = kn.judge(rows)
    law = v["law_discontinuity"]
    assert law["linear_slope_at_break"] == pytest.approx(1.24)
    assert law["sqrt_slope_at_break"] == pytest.approx(0.6135719910778964, rel=1e-12)
    assert "slope_ratio_short" in v["H3_slope_shape"]