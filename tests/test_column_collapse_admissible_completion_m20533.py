"""CASE-COLUMN-COLLAPSE completion wave (m20533) — artifact + gate tests.

These tests do **not** re-run the DEM (the ten cases cost ~8 CPU-minutes); they
pin the published reduction to the archived case data, recompute every target
from the Lube (2004) formulas, and make sure that a verdict cannot drift away
from the rows it is supposed to summarise.  What they assert:

* the wave covers exactly the aspects m20259 declared ``incomplete_aspects``,
  with the *same* member tuples and the *same* ``CLOSING_ARM`` (no retuning);
* the admissibility gate still rejects the m19922 jitter amplitudes at the new
  aspects (geometry only, no time integration);
* every stored reference value / error / verdict in the CSV is reproduced from
  the formula and the measured data, for **both** reference variants;
* the per-aspect tri-state verdicts (pass / fail / insufficient) agree with the
  rows, and ``anchor_within_15pct_all_members`` agrees with the worst error —
  the summary is allowed to say FAIL, but not to disagree with its own data;
* the artifact manifest hashes match the committed files.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util as ilu
import json
import math
import pathlib

import pytest

from tensordem.granular_column.admissibility import overlap_stats
from tensordem.granular_column.axisym import (AxisymCase, build_axisymmetric_column,
                                             lube_2004_runout_ratio)

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "case_column_collapse_admissible_completion_m20533.py"
ART = ROOT / "docs" / "artifacts" / "column_collapse_completion_m20533"
ARCHIVE_CSV = (ROOT / "docs" / "artifacts" / "column_collapse_ensemble_m20259"
               / "cases_ensemble.csv")
BAND_TOL = 0.15
SPREAD_TOL = 0.05
SCALE_TOL = 0.05


def _bench():
    spec = ilu.spec_from_file_location("_case_column_completion_m20533", BENCH)
    mod = ilu.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _summary():
    with open(ART / "summary.json") as fh:
        return json.load(fh)


def _rows():
    with open(ART / "cases_completion.csv") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Pre-registration of the wave itself
# ---------------------------------------------------------------------------
def test_wave_covers_the_aspects_m20259_declared_incomplete():
    mod = _bench()
    assert tuple(mod.COMPLETION_ASPECTS) == (1.0, 2.0)
    with open(ROOT / "docs" / "artifacts" / "column_collapse_ensemble_m20259"
              / "summary.json") as fh:
        prev = json.load(fh)
    # the hole this wave fills, read from the previous mission's own summary
    assert sorted(prev["incomplete_aspects"]) == sorted(mod.COMPLETION_ASPECTS)
    ids = {c["id"] for c in mod.completion_cases()}
    for a in mod.COMPLETION_ASPECTS:
        for tag, _, _ in (mod.LATTICE,) + tuple(mod.MEMBERS):
            assert f"cmp_a{a:g}_{tag}" in ids


def test_members_and_arm_are_inherited_not_retuned():
    mod = _bench()
    from case_column_collapse_ensemble_m19922 import CLOSING_ARM as ARM22
    from case_column_collapse_ensemble_admissible_m20259 import MEMBERS as M20259
    assert mod.MEMBERS == M20259
    assert mod.CLOSING_ARM == ARM22
    specs = mod.completion_cases()
    assert len(specs) == 2 * (1 + len(M20259))
    for c in specs:
        for k, v in ARM22.items():
            assert c["params"][k] == v, (c["id"], k)


@pytest.mark.parametrize("jitter,admissible", [(0.0025, True), (0.005, True),
                                               (0.01, False)])
def test_admissibility_gate_at_the_new_aspect(jitter, admissible):
    """The admissible band found at a = 0.5 / 2.0 must also hold at a = 1.0."""
    case = AxisymCase(radius=0.03, density=2500.0, mu=0.5, mu_rolling=0.5,
                      mu_wall=0.8, restitution=0.5, kn=6.0e4, kt_ratio=0.5,
                      grains_per_radius=8.0, packing_gap=0.02, gravity=9.81,
                      dt_scale=1.0, duration_factor=8.0, min_time=0.5,
                      rest_ke_tol=1.0e-4, rest_hold=200, t_reach_fraction=0.99,
                      aspect_ratio=1.0, jitter=jitter, seed=505)
    pos, _ = build_axisymmetric_column(case)
    st = overlap_stats(pos, case.radius)
    assert st["admissible"] is admissible, st


# ---------------------------------------------------------------------------
# Published artifact: completeness and closure
# ---------------------------------------------------------------------------
def test_summary_closes_the_declared_gap():
    s = _summary()
    assert s["incomplete_aspects"] == []
    assert s["aspects_complete"] == ["0.5", "1", "2"]
    assert s["primary_reference_variant"].startswith("measured-a")
    for a in ("0.5", "1", "2"):
        v = s["verdicts"][a]
        assert v["n_members"] == 4, a           # four admissible members per aspect
        assert v["n_control"] == 1, a
        assert v["C_BAND"]["status"] != "insufficient", a


def test_every_member_passed_the_admissibility_gate():
    rows = _rows()
    members = [r for r in rows if float(r["jitter"]) > 0.0]
    assert len(members) == 12               # 3 aspects x 4 members
    for r in members:
        assert int(r["overlap_pairs"]) == 0, r["case"]
        assert r["admissible"] == "True", r["case"]
        assert r["stop_reason"] in ("at_rest", "time_limit"), r["case"]


# ---------------------------------------------------------------------------
# No hand-typed targets: everything is recomputed from the formula
# ---------------------------------------------------------------------------
def test_reference_values_and_errors_recompute_exactly():
    for r in _rows():
        a_nom = float(r["aspect_nominal"])
        a_meas = float(r["aspect_measured"])
        runout = float(r["runout_ratio_p995"])
        ref_nom = lube_2004_runout_ratio(a_nom)
        ref_meas = lube_2004_runout_ratio(a_meas)
        assert float(r["ref_nominal"]) == pytest.approx(ref_nom, rel=1e-12)
        assert float(r["ref_measured"]) == pytest.approx(ref_meas, rel=1e-12)
        assert float(r["err_nom"]) == pytest.approx(runout / ref_nom - 1.0, rel=1e-12)
        assert float(r["err_meas"]) == pytest.approx(runout / ref_meas - 1.0, rel=1e-12)
        assert r["verdict_nom"] == ("pass" if abs(runout / ref_nom - 1.0) <= BAND_TOL
                                    else "fail")
        assert r["verdict_meas"] == ("pass" if abs(runout / ref_meas - 1.0) <= BAND_TOL
                                     else "fail")
        assert (r["flip"] == "True") == (r["verdict_nom"] != r["verdict_meas"])


def test_measured_aspect_is_the_paper_variable():
    """a_meas = h_meas/r_i_meas, so the stored aspect must track the packing."""
    rows = _rows()
    for a in (0.5, 1.0, 2.0):
        grp = [r for r in rows if float(r["aspect_nominal"]) == a]
        vals = [float(r["aspect_measured"]) for r in grp]
        assert max(vals) - min(vals) <= 0.005 * a, (a, vals)  # jitter barely moves a_meas
        assert all(v > a for v in vals), a      # the HCP packing underfills the radius


# ---------------------------------------------------------------------------
# Verdicts must follow from the rows (tri-state, no silent pass)
# ---------------------------------------------------------------------------
def _status(ok, n, need):
    if n < need:
        return "insufficient"
    return "pass" if ok else "fail"


def test_per_aspect_verdicts_match_recomputation():
    s = _summary()
    rows = _rows()
    for a in (0.5, 1.0, 2.0):
        grp = [r for r in rows if float(r["aspect_nominal"]) == a]
        mem = [r for r in grp if float(r["jitter"]) > 0.0]
        ctl = [r for r in grp if float(r["jitter"]) == 0.0]
        v = s["verdicts"][f"{a:g}"]

        worst = max(abs(float(r["err_meas"])) for r in mem)
        assert v["C_BAND"]["status"] == _status(worst <= BAND_TOL, len(mem), 1), a
        assert v["C_BAND"]["detail"]["worst_abs_err_measured_a"] == pytest.approx(
            worst, rel=1e-12)
        assert v["C_BAND"]["detail"]["worst_abs_err_nominal_a"] == pytest.approx(
            max(abs(float(r["err_nom"])) for r in mem), rel=1e-12)

        vals = [float(r["runout_ratio_p995"]) for r in mem]
        mean = sum(vals) / len(vals)
        sigma = math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1))
        assert v["C_SPREAD"]["detail"]["sigma_rel"] == pytest.approx(sigma / mean, rel=1e-12)
        assert v["C_SPREAD"]["status"] == _status(sigma / mean <= SPREAD_TOL, len(mem), 2)

        lattice = float(ctl[0]["runout_ratio_p995"])
        shift = max(abs(x / lattice - 1.0) for x in vals)
        assert v["C_SCALE"]["detail"]["max_abs_shift_vs_lattice"] == pytest.approx(
            shift, rel=1e-12)
        assert v["C_SCALE"]["status"] == _status(shift <= SCALE_TOL, len(mem) + 1, 2)

        flips = sorted(r["case"] for r in grp if r["flip"] == "True")
        assert sorted(v["C_BAND"]["detail"]["variant_verdict_flips"]) == flips


def test_honest_fail_is_not_smoothed():
    """The summary may report outside-band results, but must say so."""
    s = _summary()
    rows = _rows()
    worst = max(abs(float(r["err_meas"])) for r in rows if float(r["jitter"]) > 0.0)
    assert s["worst_abs_err_measured_a"] == pytest.approx(worst, rel=1e-12)
    assert s["anchor_within_15pct_all_members"] is (worst <= BAND_TOL)
    # and the per-aspect statements must agree with the same rows
    for a in ("0.5", "1", "2"):
        v = s["verdicts"][a]
        assert v["anchor_within_15pct_all_rows"] is (
            v["max_abs_err_measured_a_all_rows"] <= BAND_TOL), a


def test_archive_rows_come_from_the_m20259_csv_unchanged():
    with open(ARCHIVE_CSV) as fh:
        src = {(r["case"], r["member"]): r for r in csv.DictReader(fh)}
    got = [r for r in _rows() if r["wave"].startswith("archive")]
    assert len(got) == 5                       # lattice + four admissible members
    for r in got:
        ref = src[(r["case"], r["member"])]
        assert float(r["runout_ratio_p995"]) == pytest.approx(
            float(ref["runout_ratio_p995"]), rel=1e-12)
        assert float(r["aspect_measured"]) == pytest.approx(
            float(ref["aspect_measured"]), rel=1e-12)


def test_manifest_hashes_match():
    with open(ART / "manifest_sha256.json") as fh:
        manifest = json.load(fh)
    for name, digest in manifest.items():
        h = hashlib.sha256((ART / name).read_bytes()).hexdigest()
        assert h == digest, name
    assert {"cases_completion.csv", "summary.json",
            "fig_band_margin.png", "fig_runout_vs_aspect.png"} <= set(manifest)