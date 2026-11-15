"""Fast unit tests for the m20055 D5 dt-convergence ladder (prereg/p20055_dt_ladder.md).

These tests exercise the *analysis* of the ladder, not the DEM: they pin

* the three-level Richardson order estimator and the extrapolation,
* the literature-band arithmetic (k-range / C-range of Beverloo 1961),
* the predeclared verdict rule (PASS / FAIL / INCONCLUSIVE),
* the ladder case construction: only ``dt`` changes, everything else is identical,
* that the generated audit markdown carries numbers pulled from the records.

The heavy ladder runs themselves are filed under ``runs/m20055_beverloo_dt_ladder/``
and are reproduced with ``dt_ladder.py --level <level>``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "m20055_beverloo"
for p in (str(ROOT / "src"), str(BENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dt_ladder as L  # noqa: E402

DT0 = L.build_level_case("dt").timestep


def _record(level: str, divisor: int, W: float, *, cv: float = 0.08,
            W_ref: float = 2.8249, n_bed: int = 1500, phi: float = 0.52) -> dict:
    """Minimal stand-in for a per-level JSON (same keys summarise() reads)."""
    return {
        "case_id": f"D5_{level}",
        "ladder": {"level": level, "divisor": divisor, "dt": DT0 / divisor,
                   "dt_default": DT0, "dt_over_contact_period": 0.03535534 / divisor,
                   "steps": int(50078 * divisor), "base_case": "D5"},
        "verdict": {"W_got_kg_s": W, "W_ref_kg_s": W_ref,
                    "rel_err_W": (W - W_ref) / W_ref, "rho_b_kg_m3": 1300.0,
                    "Deff_m": 0.072},
        "telemetry": {"steady_rate_cv": cv, "packing_fraction_bed": phi,
                      "n_discharged_total": 900, "n_bed": n_bed,
                      "orifice_diameter": 0.100, "particle_diameter": 0.020},
    }


# ---------------------------------------------------------------------------
# 1. observed order and Richardson extrapolation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p_true", [1.0, 1.5, 2.0])
def test_observed_order_recovers_a_geometric_ladder(p_true):
    """W(dt) = W_inf + a dt^p -> the three-level estimator must return p."""
    W_inf, a = 2.80, 0.60
    # unit refinement variable (the estimator only sees the three rates)
    w = np.array([W_inf + a * (1.0 / div) ** p_true for div in (1, 2, 4)])
    out = L.observed_order(w)
    assert out["p"] == pytest.approx(p_true, rel=1e-9)
    assert out["reason"] == ""


def test_richardson_removes_a_pure_power_law_bias():
    """Extrapolating a synthetic p = 2 ladder must land on W_inf to round-off."""
    W_inf, a, p = 2.80, 0.60, 2.0
    w = np.array([W_inf + a * (1.0 / div) ** p for div in (1, 2, 4)])
    assert L.richardson(w, p) == pytest.approx(W_inf, rel=1e-12)
    # and with the order estimated from the data, not handed in
    assert L.richardson(w, L.observed_order(w)["p"]) == pytest.approx(W_inf, rel=1e-9)


def test_observed_order_is_undefined_when_not_monotone():
    """A non-monotone ladder must not yield an order (it would be meaningless)."""
    w = np.array([3.00, 3.20, 3.10])
    out = L.observed_order(w)
    assert out["p"] is None
    assert "non-monotone" in out["reason"]


def test_observed_order_is_undefined_with_two_levels_or_zero_difference():
    assert L.observed_order(np.array([3.4, 3.1]))["p"] is None
    assert L.observed_order(np.array([3.4, 3.4, 3.1]))["p"] is None
    assert L.observed_order(np.array([3.4, 3.1, 3.1]))["p"] is None


def test_richardson_needs_two_levels():
    with pytest.raises(ValueError):
        L.richardson(np.array([3.4]), 1.0)


# ---------------------------------------------------------------------------
# 2. literature-band arithmetic
# ---------------------------------------------------------------------------
def test_k_band_is_unity_at_the_cited_k_and_widens_away_from_it():
    D, d, W_ref = 0.100, 0.020, 2.8249
    band = L.k_band(W_ref, D, d)
    base = D - L.K_LIT * d
    lo = W_ref * ((D - L.K_LO * d) / base) ** 2.5
    hi = W_ref * ((D - L.K_HI * d) / base) ** 2.5
    assert band["W_min"] == pytest.approx(min(lo, hi), rel=1e-12)
    assert band["W_max"] == pytest.approx(max(lo, hi), rel=1e-12)
    assert band["W_min"] < W_ref < band["W_max"]
    # a smaller k means a larger effective orifice -> larger rate
    assert L.K_LO < L.K_LIT < L.K_HI
    assert lo > W_ref > hi


def test_k_band_matches_the_5_2_power_law_ratio():
    """Explicit check of the exponent: 1 % change of D_eff -> 2.5 % change of W."""
    D, d, W_ref = 0.100, 0.020, 1.0
    k_probe = L.K_LIT * 1.001
    band = L.k_band(W_ref, D, d, k_lo=k_probe, k_hi=k_probe)
    Deff_lit = D - L.K_LIT * d
    Deff_probe = D - k_probe * d
    assert band["fac_max"] == pytest.approx((Deff_probe / Deff_lit) ** 2.5, rel=1e-12)


def test_c_band_is_linear_and_centred_on_the_cited_c():
    band = L.c_band(2.8249)
    assert band["W_min"] == pytest.approx(2.8249 * L.C_LO / L.C_LIT, rel=1e-12)
    assert band["W_max"] == pytest.approx(2.8249 * L.C_HI / L.C_LIT, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. the predeclared verdict rule
# ---------------------------------------------------------------------------
def _summarise(Ws, *, cvs=None):
    levels = ["dt", "dthalf", "dtquarter"][: len(Ws)]
    divs = [1, 2, 4][: len(Ws)]
    cvs = cvs or [0.08] * len(Ws)
    return L.summarise([_record(lv, dv, w, cv=c)
                        for lv, dv, w, c in zip(levels, divs, Ws, cvs)])


def test_verdict_passes_when_the_limit_is_inside_the_platform_tolerance():
    s = _summarise([3.4069, 3.1036, 2.9800])
    assert s["verdict"] == "PASS-CONVERGED-IN-TOL"
    assert abs(s["limit"]["rel"]) < L.TOL_PASS
    assert s["limit"]["p_used"] == pytest.approx(
        L.observed_order(np.array([3.4069, 3.1036, 2.9800]))["p"])


def test_verdict_fails_when_the_limit_stays_out_of_tolerance():
    """A ladder converging to +25 % is a physical defect, not an artefact."""
    s = _summarise([3.92, 3.62, 3.50])       # limit ~ +24 % of 2.8249
    assert s["verdict"] == "FAIL-BUT-QUANTIFIED"
    assert abs(s["limit"]["rel"]) > L.TOL_PASS


def test_verdict_is_inconclusive_when_the_order_is_not_identifiable():
    s = _summarise([3.10, 3.40, 3.20])       # non-monotone
    assert s["verdict"] == "INCONCLUSIVE-NO-ORDER"
    assert s["limit"]["p_source"].startswith("fallback")


def test_noisy_levels_are_flagged_and_excluded_from_the_order():
    s = _summarise([3.4069, 3.1036, 2.9800], cvs=[0.08, 0.45, 0.08])
    assert s["noisy_levels"] == ["dthalf"]
    # order must come from the two clean levels -> documented p=1 fallback
    assert s["limit"]["p_source"].startswith("fallback")


def test_summary_keeps_the_geometry_and_the_reference_provenance():
    s = _summarise([3.4069, 3.1036, 2.98])
    assert s["geometry"]["D_over_d"] == pytest.approx(5.0)
    assert s["reference"]["C"] == L.C_LIT and s["reference"]["k"] == L.K_LIT
    assert "NOT" in s["reference"]["provenance"]      # cited, NOT re-digitized


def test_dt_effect_is_the_coarse_to_fine_change_of_the_error():
    s = _summarise([3.4069, 3.1036])
    rel = [(3.4069 - 2.8249) / 2.8249, (3.1036 - 2.8249) / 2.8249]
    assert s["dt_effect"] == pytest.approx(abs(rel[1] - rel[0]), rel=1e-12)


# ---------------------------------------------------------------------------
# 4. ladder case construction: dt is the only changed variable
# ---------------------------------------------------------------------------
def test_ladder_case_changes_only_the_time_step():
    base = L.build_level_case("dt")
    assert base.timestep == pytest.approx(DT0)
    for level, div in (("dthalf", 2), ("dtquarter", 4)):
        c = L.build_level_case(level)
        assert c.timestep == pytest.approx(DT0 / div, rel=1e-15)
        changed = [k for k in vars(c) if getattr(c, k) != getattr(base, k)]
        assert changed == ["dt"], f"{level} changed more than dt: {changed}"


def test_ladder_level_rejects_an_unknown_name():
    with pytest.raises(SystemExit):
        L.build_level_case("dt-sixteenth")
    assert L.DIVISOR["dtquarter"] == 4


def test_contact_period_is_dt_over_the_recorded_ratio():
    case = L.build_level_case("dt")
    # the case record files dt/T_contact; it must be dt / (2 pi sqrt(m/kn))
    T = L.contact_period(case)
    assert T == pytest.approx(2 * math.pi * math.sqrt(case.material.mass / case.kn))
    assert case.timestep / T < 0.05      # the filed 0.0354, i.e. the regime under test


# ---------------------------------------------------------------------------
# 5. plumbing: a real (tiny) DEM record must be consumable by summarise()
# ---------------------------------------------------------------------------
def test_a_real_tiny_run_record_is_summarisable():
    """Guards the schema between run_case.analyse() and the ladder summary.

    A 60 mm mini-silo (not the D5 case, no physics claim) is run end-to-end so
    that a rename in the telemetry/verdict keys cannot silently break the
    analysis: the record is built exactly as ``dt_ladder.run_level`` builds it.
    """
    import run_case
    from tensordem.silo_discharge import SiloCase

    tiny = SiloCase(radius=0.010, density=2500.0, mu=0.30, mu_rolling=0.02,
                    restitution=0.30, kn=1.0e5, silo_radius=0.050,
                    fill_height=0.060, orifice_diameter=0.065,
                    plate_thickness=0.010, wall_mu=0.0, packing="settled",
                    plug_during_settle=True, settle_loose_gap=0.30, jitter=0.05,
                    seed=20055, settle_time=0.10, discharge_time=0.10,
                    window=0.02, n_windows=5)
    res = run_case.simulate_silo(tiny)
    rec = run_case.analyse("D5_dt", tiny, res, L.REF, 0.0)
    rec["ladder"] = {"level": "dt", "divisor": 1, "dt": tiny.timestep,
                     "dt_default": tiny.timestep,
                     "dt_over_contact_period": tiny.timestep / L.contact_period(tiny),
                     "steps": int(res["steps"])}
    finer = {**rec, "ladder": {**rec["ladder"], "level": "dthalf", "divisor": 2,
                               "dt": tiny.timestep / 2, "steps": 2 * res["steps"]},
             "verdict": {**rec["verdict"], "W_got_kg_s": 0.97 * rec["verdict"]["W_got_kg_s"]}}
    s = L.summarise([rec, finer])
    assert len(s["rows"]) == 2
    assert s["rows"][0]["n_bed"] > 0 and s["rows"][0]["W"] > 0.0
    assert s["geometry"]["D_over_d"] == pytest.approx(3.25)
    assert L.render_doc(s).startswith("# Audit")
    assert s["verdict"] in ("PASS-CONVERGED-IN-TOL", "FAIL-BUT-QUANTIFIED",
                            "INCONCLUSIVE-NO-ORDER")


# ---------------------------------------------------------------------------
# 5b. the archived verdict table is parsed, not hand copied
# ---------------------------------------------------------------------------
_ARCH_MD = """
| case | D/d | mu_p | n_bed | H/D | silo_D/D | rho_b | phi | discharged | W | W_ref | err | C_rec | CV | <10% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D5 | 5.00 | 0.30 | 232 | 1.08 | 1.60 | 1118 | 0.447 | 64% | 3.4069 | 2.8257 | +20.6% | 0.699 | 0.080 | x |
"""


def test_archive_row_is_read_from_the_filed_table():
    row = L.parse_archive_row(_ARCH_MD, "D5")
    assert row["rho_b"] == pytest.approx(1118.0)
    assert row["W"] == pytest.approx(3.4069)
    assert row["C_rec"] == pytest.approx(0.699)


def test_archive_row_absent_returns_none():
    assert L.parse_archive_row(_ARCH_MD, "D9") is None
    assert L.parse_archive_row("no table here", "D5") is None


def test_archive_check_reports_the_deviation_of_the_rerun():
    rec = _record("dt", 1, 3.7084)
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(_ARCH_MD)
        path = fh.name
    try:
        res = L.archive_check(rec, doc=path)
    finally:
        os.unlink(path)
    assert res["available"] is True
    assert res["archive"]["W"] == pytest.approx(3.4069)
    assert res["delta"]["W"] == pytest.approx((3.7084 - 3.4069) / 3.4069, rel=1e-12)
    assert L.archive_check(rec, doc="/nonexistent/doc.md") == {"available": False}


# ---------------------------------------------------------------------------
# 6. the generated audit document
# ---------------------------------------------------------------------------
def test_render_doc_contains_only_numbers_from_the_records():
    Ws = [3.4069, 3.1036, 2.9800]
    s = _summarise(Ws)
    text = L.render_doc(s)
    for w in Ws:
        assert f"{w:.4f}" in text
    assert f"{s['limit']['W_inf']:.4f}" in text
    assert s["verdict"] in text
    assert "< 3 %" in text                 # the paper-grade column is present
    assert "prereg/p20055_dt_ladder.md" in text
    # the honest statement about the cited reference must survive rendering
    assert "NOT" in text


def test_analyse_all_writes_summary_doc_and_figure(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for lv, dv, w in (("dt", 1, 3.4069), ("dthalf", 2, 3.1036), ("dtquarter", 4, 2.9800)):
        (runs / f"{lv}.json").write_text(json.dumps(_record(lv, dv, w), default=float))
    doc = tmp_path / "AUDIT.md"
    fig = tmp_path / "fig.png"
    s = L.analyse_all(runs=runs, doc=doc, fig=fig)
    assert (runs / "ladder_summary.json").exists()
    assert json.loads((runs / "ladder_summary.json").read_text())["verdict"] == s["verdict"]
    assert doc.exists() and doc.read_text().startswith("# Audit")
    assert s["levels"] == ["dt", "dthalf", "dtquarter"]


def test_analyse_all_needs_at_least_two_levels(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "dt.json").write_text(json.dumps(_record("dt", 1, 3.4069), default=float))
    with pytest.raises(SystemExit):
        L.analyse_all(runs=runs, doc=tmp_path / "d.md", fig=tmp_path / "f.png")


def test_reference_json_is_still_the_single_source_of_the_coefficients():
    ref = json.loads((BENCH / "ref_beverloo_1961.json").read_text())
    assert L.C_LIT == float(ref["coefficients"]["C"])
    assert L.K_LIT == float(ref["coefficients"]["k"])
    assert L.K_LO == float(ref["coefficients"]["k_range_spherical"][0])
    assert L.K_HI == float(ref["coefficients"]["k_range_spherical"][1])