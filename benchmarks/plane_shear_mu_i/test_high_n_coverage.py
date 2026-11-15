"""Round 20 (m21290) guards for ``high_n_coverage.py``.

The round-20 artefact exists to close two gaps that earlier rounds left open,
and each guard below pins one of them so it cannot silently reopen:

Q1 range       the ledger acceptance is ``I = 1e-3 ... 0.3``; the round-18/19
               higher-N sweep sampled only ``I >= 1e-2``.  The target tuple
               must bracket the ledger range at both ends.
Q2 estimator   the verdict must not depend on a single hidden estimator: the
               wall / full-cell / bulk mu arms and the imposed / measured
               abscissae are all carried, and ``assemble`` must report every
               combination instead of one number.
Q3 unit        every residual is a *percentage* and the band is compared in
               percent (round-19 defect D-C must not be reintroduced).
Q4 single src  no local copy of the reference triple or of ``ref_mu``/
               ``ref_phi`` (round-19 defect D-A / D-B).
Q5 honest      a curve that misses the band must come out FAIL, a curve that
               sits on the law must come out PASS -- both are exercised, so the
               verdict function cannot be a rubber stamp.

No DEM run is needed: ``summarise`` and ``assemble`` are pure.
"""
from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parents[1] / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import high_n_coverage as C  # noqa: E402
import reference_law as REF  # noqa: E402

MODULE = HERE / "high_n_coverage.py"


# ---------------------------------------------------------------- fake device
def _res(mu_wall, *, n_samples=40, i_imposed=0.03, i_measured=0.012,
         phi=(0.60, 0.61, 0.59), mu_full=None, mu_bulk=None,
         p_over_zz=0.85):
    """Minimal stand-in for ``simulate_servo_shear``'s result dict."""
    series = [{"mu": mu_wall, "step": 10 * k} for k in range(n_samples)]
    return {
        "series": series,
        "inertial_number_imposed_rate": i_imposed,
        "inertial_number_measured_rate_final": i_measured,
        "full_mu_final": mu_full if mu_full is not None else mu_wall * 1.05,
        "bulk_mu_final": mu_bulk if mu_bulk is not None else mu_wall * 1.10,
        "phi_bed_steady": phi[0],
        "phi_extent_final": phi[1],
        "phi_container_final": phi[2],
        "full_anisotropy": p_over_zz,
        "normal_stress_measured_pa": 6.25e4,
        "normal_stress_applied_pa": 6.25e4,
        "shear_rate_ratio": 0.8,
        "shear_strain_achieved": 0.5,
        "n_particles": 1178,
        "coordination_number": 5.6,
        "servo_valid": True,
        "dt": 3.2e-5,
        "settle_steps": 3000,
        "shear_steps": 15000,
    }


def _row(I, mu_wall, **kw):
    rec = C.summarise(_res(mu_wall, i_imposed=I, i_measured=kw.pop("i_measured", I),
                           **kw), I)
    rec["wall_clock_s"] = 60.0
    return rec


# ------------------------------------------------------------------ Q1 range
def test_targets_cover_the_ledger_range_at_both_ends():
    assert C.I_TARGETS[0] == pytest.approx(1e-3, rel=1e-12)
    assert C.I_TARGETS[-1] == pytest.approx(0.3, rel=1e-12)
    assert tuple(C.I_TARGETS) == tuple(sorted(C.I_TARGETS))
    # the round-18/19 blind spot: no point below 1e-2 in that sweep
    assert min(C.I_TARGETS) < 1e-2


def test_prereg_records_the_range_and_the_band():
    assert C.PREREG["I_targets"][0] == pytest.approx(1e-3)
    assert C.PREREG["I_targets"][-1] == pytest.approx(0.3)
    assert C.PREREG["tolerance_pct"] == 10.0


# -------------------------------------------------------------- Q2 estimators
def test_all_mu_arms_and_abscissae_are_carried():
    rec = _row(0.03, 0.45)
    for arm in C.MU_ARMS:
        for ab in C.ABSCISSAE:
            assert f"err_pct__{arm}__{ab}" in rec
            assert f"mu_ref__{arm}__{ab}" in rec
    assert set(C.MU_ARMS) == {"mu_wall", "mu_full", "mu_bulk"}
    assert "I_measured_rate" in C.ABSCISSAE and "I_imposed_rate" in C.ABSCISSAE


def test_estimators_are_distinct_when_the_device_returns_distinct_values():
    rec = _row(0.03, 0.40)
    assert rec["mu_full"] != pytest.approx(rec["mu_wall"])
    assert rec["mu_bulk"] != pytest.approx(rec["mu_full"])


def test_measured_rate_abscissa_is_used_when_present_and_falls_back_when_not():
    rec = _row(0.03, 0.45, i_measured=0.012)
    assert rec["I_measured_rate"] == pytest.approx(0.012)
    assert rec["I_imposed_rate"] == pytest.approx(0.03)
    assert rec["I_abscissa_best"] == pytest.approx(0.012)
    # no profile fit -> the documented fallback keeps the arm alive
    stale = _res(0.45, i_imposed=0.03, i_measured=None)
    stale["inertial_number_measured_rate_final"] = None
    rec2 = C.summarise(stale, 0.03)
    assert math.isnan(rec2["I_measured_rate"])
    assert rec2["I_abscissa_best"] == pytest.approx(0.03)


def test_phi_estimators_are_reported_separately_with_spread(tmp_path):
    rows = [_row(1e-2, 0.40, i_measured=1e-2, phi=(0.60, 0.62, 0.58)),
            _row(3e-2, 0.45, i_measured=3e-2)]
    _write(rows, tmp_path)
    v = C.assemble(tmp_path)
    pr = v["phi"]["rows"][0]
    assert pr["estimators"]["phi_extent_final"] == pytest.approx(0.62)
    assert pr["spread_pct"] == pytest.approx(100.0 * (0.62 - 0.58) / ((0.60 + 0.62 + 0.58) / 3),
                                            rel=1e-9)
    assert v["phi"]["reference_status"].startswith("PROVENANCE-UNVERIFIED")


# ------------------------------------------------------------------- Q3 unit
def test_residuals_are_percent_not_fraction():
    I = 0.03
    mu = REF.ref_mu(I) * 1.09            # 9 % high -> inside a 10 % band
    rec = _row(I, mu, i_measured=I)
    assert rec["err_pct__mu_wall__I_measured_rate"] == pytest.approx(9.0, abs=1e-9)
    assert C.TOL_PCT == 10.0


def test_band_boundary_is_evaluated_in_percent(tmp_path):
    """A 9 % point passes, a 12 % point fails -- with TOL=0.10 both would fail."""
    I = [1e-2, 3e-2]
    ok_rows = [_row(i, REF.ref_mu(i) * 1.09, i_measured=i) for i in I]
    _write(ok_rows, tmp_path / "ok")
    v_ok = C.assemble(tmp_path / "ok")
    arm_ok = _arm(v_ok, "mu_wall", "I_measured_rate")
    assert arm_ok["within_band_all_points"] is True
    assert arm_ok["worst_abs_err_pct"] == pytest.approx(9.0, abs=1e-9)

    bad_rows = [_row(i, REF.ref_mu(i) * 1.12, i_measured=i) for i in I]
    _write(bad_rows, tmp_path / "bad")
    v_bad = C.assemble(tmp_path / "bad")
    arm_bad = _arm(v_bad, "mu_wall", "I_measured_rate")
    assert arm_bad["within_band_all_points"] is False
    assert arm_bad["worst_abs_err_pct"] == pytest.approx(12.0, abs=1e-9)
    assert v_bad["mu_passes_full_range"] is False
    assert v_bad["tolerance_pct"] == 10.0


# -------------------------------------------------------------- Q4 single src
def test_reference_comes_from_reference_law_not_a_local_copy():
    assert C.ref_mu is REF.ref_mu and C.ref_phi is REF.ref_phi
    assert (C.MU_S, C.MU_2, C.I_0) == (REF.MU_S, REF.MU_2, REF.I_0)
    assert (C.PHI_MAX, C.PHI_MIN) == (REF.PHI_MAX, REF.PHI_MIN)


def test_no_module_level_redefinition_of_the_law_or_the_triple():
    tree = ast.parse(MODULE.read_text())
    forbidden_fn = {"ref_mu", "ref_phi"}
    forbidden_const = {"MU_S", "MU_2", "I_0", "PHI_MAX", "PHI_MIN"}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert node.name not in forbidden_fn, f"local {node.name} shadow"
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    assert tgt.id not in forbidden_const, f"local {tgt.id} shadow"
                    assert tgt.id not in forbidden_fn


# ---------------------------------------------------------------- Q5 honest
def test_honest_fail_when_every_arm_misses_the_band(tmp_path):
    """Measured mu 25 % low at every point: the verdict must be FAIL, not PASS."""
    Is = list(C.I_TARGETS)
    rows = [_row(i, REF.ref_mu(i) * 0.75, i_measured=i,
                 mu_full=REF.ref_mu(i) * 0.75, mu_bulk=REF.ref_mu(i) * 0.75)
            for i in Is]
    _write(rows, tmp_path)
    v = C.assemble(tmp_path)
    assert v["mu_passes_full_range"] is False
    assert all(not a["within_band_all_points"] for a in v["mu_verdicts"])
    assert v["mu_best_arm"]["worst_abs_err_pct"] == pytest.approx(25.0, abs=1e-6)
    # the oracle must not claim provenance can rescue a 25 % structural miss
    assert v["envelope_can_rescue"] is False


def test_pass_when_the_curve_sits_on_the_law(tmp_path):
    rows = [_row(i, REF.ref_mu(i) * 1.02, i_measured=i,
                 mu_full=REF.ref_mu(i) * 0.98, mu_bulk=REF.ref_mu(i) * 1.01)
            for i in C.I_TARGETS]
    _write(rows, tmp_path)
    v = C.assemble(tmp_path)
    assert v["mu_passes_full_range"] is True
    assert v["range_complete"] is True
    assert v["mu_best_arm"]["worst_abs_err_pct"] < 3.0


def test_assembly_reports_range_completeness_and_particle_count(tmp_path):
    rows = [_row(i, REF.ref_mu(i), i_measured=i) for i in C.I_TARGETS]
    _write(rows, tmp_path)
    v = C.assemble(tmp_path)
    assert v["range_complete"] is True
    assert v["min_n_particles"] == pytest.approx(1178)
    assert v["ledger_range"] == [1e-3, 0.3]
    assert v["servo_valid_all"] is True
    assert (tmp_path / "result.json").exists()


def test_partial_range_is_flagged_not_silently_passed(tmp_path):
    rows = [_row(i, REF.ref_mu(i), i_measured=i) for i in (1e-2, 3e-2, 0.1, 0.3)]
    _write(rows, tmp_path)
    v = C.assemble(tmp_path)
    assert v["range_complete"] is False
    assert v["range_covered"] == [pytest.approx(1e-2), pytest.approx(0.3)]


def test_tables_render_every_matrix_and_no_placeholder(tmp_path):
    rows = [_row(i, REF.ref_mu(i) * k / len(C.I_TARGETS), i_measured=i)
            for k, i in enumerate(C.I_TARGETS, start=1)]
    _write(rows, tmp_path)
    v = C.assemble(tmp_path)
    md = C.tables_markdown(v)
    for heading in ("Table 1", "Table 2", "Table 3", "Table 4"):
        assert heading in md
    assert "pending" not in md.lower()
    # Table 2 must carry one row per (estimator, abscissa) and Table 3 one
    # column per cell plus the I-target column
    t2 = md.split("## Table 2")[1].split("## Table 3")[0]
    t2_rows = [ln for ln in t2.splitlines() if ln.startswith("| ") and "|" in ln]
    assert len(t2_rows) - 1 == len(C.MU_ARMS) * len(C.ABSCISSAE)
    t3 = md.split("## Table 3")[1].split("## Table 4")[0]
    header = [ln for ln in t3.splitlines() if ln.startswith("| ")][0]
    assert header.count("|") == len(C.MU_ARMS) * len(C.ABSCISSAE) + 2
    t1_body = [ln for ln in md.split("## Table 1")[1].split("## Table 2")[0].splitlines()
               if ln.startswith("| ") and not ln.startswith("| I target")]
    assert len(t1_body) == len(rows)


# ------------------------------------------------- round-19 vacuous-PASS fix
def test_round19_assembly_cannot_pass_vacuously_without_mid_points(tmp_path):
    """``all([])`` is True: a dropped mid-I shard must not read as P1/P2 pass."""
    import high_n_sweep as H19

    d = tmp_path / "h19"
    d.mkdir(parents=True, exist_ok=True)
    # only the high-I points: the round-18 mid-I prediction cannot be judged
    for I, mu in ((0.1, 0.45), (0.3, 0.55)):
        rec = _row(I, mu, i_measured=I)
        rec.update({
            "mu_meas": mu,
            "mu_std_rel_pct": 3.0,
            "err_mu_pct": 1.0,
            "err_mu_pct_at_imposed_I": 1.0,
            "phi_meas": 0.60,
            "err_phi_pct": 1.0,
            "phi_extent_final": 0.60,
            "phi_container_final": 0.60,
            "mu_ref": REF.ref_mu(I),
            "phi_ref": REF.ref_phi(I),
            "mu_ref_at_imposed_I": REF.ref_mu(I),
            "phi_ref_at_imposed_I": REF.ref_phi(I),
            "normal_stress_pa": 6.25e4,
            "pressure_control_error": 0.0,
            "shear_stress_pa": mu * 6.25e4,
            "abscissa": "I_measured_rate",
            "I_abscissa": I,
        })
        (d / f"point_{I:g}.json").write_text(json.dumps(rec))
    v = H19.assemble(d)
    assert v["mid_points_present"] is False
    assert v["P1_noise_floor"] is False
    assert v["P2_mu_convergence"] is False

    # with the mid points present the same code must still be able to pass
    for I in (1e-2, 3e-2):
        rec = _row(I, REF.ref_mu(I), i_measured=I)
        rec.update({"mu_meas": REF.ref_mu(I), "mu_std_rel_pct": 3.0,
                    "err_mu_pct": 1.0, "err_mu_pct_at_imposed_I": 1.0,
                    "phi_meas": 0.60, "err_phi_pct": 1.0,
                    "phi_extent_final": 0.60, "phi_container_final": 0.60,
                    "mu_ref": REF.ref_mu(I), "phi_ref": REF.ref_phi(I),
                    "mu_ref_at_imposed_I": REF.ref_mu(I),
                    "phi_ref_at_imposed_I": REF.ref_phi(I),
                    "normal_stress_pa": 6.25e4, "pressure_control_error": 0.0,
                    "shear_stress_pa": REF.ref_mu(I) * 6.25e4,
                    "abscissa": "I_measured_rate", "I_abscissa": I})
        (d / f"point_{I:g}.json").write_text(json.dumps(rec))
    v2 = H19.assemble(d)
    assert v2["mid_points_present"] is True
    assert v2["P1_noise_floor"] is True and v2["P2_mu_convergence"] is True


def test_partial_range_cannot_claim_a_full_range_pass(tmp_path):
    """A band-pass over 4 of 6 ledger points is not the ledger acceptance."""
    rows = [_row(i, REF.ref_mu(i), i_measured=i) for i in (1e-2, 3e-2, 0.1, 0.3)]
    _write(rows, tmp_path)
    v = C.assemble(tmp_path)
    assert v["range_complete"] is False
    assert v["mu_passes_full_range"] is False


# --------------------------------------------------------------------- utils
def _write(rows, d: Path):
    d.mkdir(parents=True, exist_ok=True)
    for r in rows:
        (d / f"point_{r['I_target']:g}.json").write_text(json.dumps(r))


def _arm(verdict, arm, abscissa):
    for v in verdict["mu_verdicts"]:
        if v["arm"] == arm and v["abscissa"] == abscissa:
            return v
    raise AssertionError(f"no verdict for {arm}/{abscissa}")