"""Tests for :mod:`thin_layer_branch_audit` (CASE-DEM-PLANE-SHEAR-MUIDI).

The audit is the reference-side half of the ``phi`` acceptance, so the tests
assert *invariants* the archived paper text fixes, not merely that the code
runs:

* ``nu_max`` 0.81 (thick) and 0.78 (thin) are distinct and the thin branch is
  always the smaller reference, hence always the *less* favourable of the two
  for a measured ``phi`` that sits above the reference;
* the ``mu`` verdict must never be attributed to the branch (thickness
  invariance is a property of the friction law, quoted in the archived text);
* the branch sensitivity flag is exactly "the two branches disagree on
  PASS/FAIL", so it cannot be true when both pass or both fail;
* the layer count is the geometric one, ``gap_eff / d``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import thin_layer_branch_audit as tlb  # noqa: E402


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_layer_count_is_gap_over_diameter():
    assert tlb.layer_count(0.14, 0.01) == pytest.approx(14.0)


def test_layer_count_rejects_zero_diameter():
    with pytest.raises(ValueError):
        tlb.layer_count(0.14, 0.0)


def test_rounds_24_31_cell_is_fourteen_layers():
    """gap=0.15, r=5e-3 -> gap_eff=0.14, d=0.01 -> 14 layers (the audited cell)."""
    gap, radius = 0.15, 5.0e-3
    d = 2.0 * radius
    assert tlb.layer_count(gap - d, d) == pytest.approx(14.0)


# --------------------------------------------------------------------------- #
# the two branches are distinct reference values
# --------------------------------------------------------------------------- #
def test_branch_constants_match_archived_paper_text():
    assert tlb.THICK_NU_MAX == pytest.approx(0.81)
    assert tlb.THIN_NU_MAX == pytest.approx(0.78)
    assert tlb.THIN_NU_MAX < tlb.THICK_NU_MAX


def test_phi_ref_branch_decreases_with_I():
    for nu_max in (tlb.THICK_NU_MAX, tlb.THIN_NU_MAX):
        lo = tlb.phi_ref_branch(1e-3, nu_max=nu_max)
        hi = tlb.phi_ref_branch(0.3, nu_max=nu_max)
        assert hi < lo, "dilatancy: phi must fall as I rises"


def test_thin_branch_is_the_stricter_reference_above_reference():
    """For a measured phi above both references, the thin branch errs larger."""
    I, phi = 0.05, 0.80
    row = tlb.judge_point(I=I, phi=phi)
    thick = row["branches"]["thick"]
    thin = row["branches"]["thin"]
    assert thick["nu_ref"] > thin["nu_ref"]
    assert abs(thin["err"]) > abs(thick["err"])


# --------------------------------------------------------------------------- #
# branch-sensitivity semantics
# --------------------------------------------------------------------------- #
def test_branch_sensitive_requires_a_flip():
    summary = tlb.judge_points([{"I": 0.05, "phi": 0.79}])
    flip = summary["verdict"]["thick"] != summary["verdict"]["thin"]
    assert summary["branch_sensitive"] is flip


def test_a_point_between_the_branches_is_sensitive():
    """A phi inside the 0.03 band around nu_ref can flip the verdict."""
    # thick ref at I=0.05: 0.81-0.015 = 0.795; thin ref: 0.765.
    # phi=0.868 is +9.2% of thick but +13.5% of thin -> fail/fail, so search
    # for a genuinely flipping phi instead of assuming one.
    found_flip = False
    for phi in [0.70 + 0.005 * k for k in range(60)]:
        row = tlb.judge_point(I=0.05, phi=phi)
        if row["branch_sensitive"]:
            found_flip = True
            assert row["branches"]["thick"]["within"] != row["branches"]["thin"]["within"]
    assert found_flip, "the two 0.03-apart branches must admit a flipping phi"


def test_far_from_reference_both_branches_fail_together():
    summary = tlb.judge_points([{"I": 0.05, "phi": 0.30}])
    assert summary["verdict"]["thick"] == summary["verdict"]["thin"] == "FAIL"
    assert summary["branch_sensitive"] is False


def test_empty_point_set_is_no_points_not_pass():
    summary = tlb.judge_points([])
    assert summary["verdict"]["thick"] == "NO_POINTS"
    assert summary["branch_sensitive"] is False


# --------------------------------------------------------------------------- #
# the ledger-facing statement
# --------------------------------------------------------------------------- #
def test_mu_verdict_never_attributed_to_branch():
    summary = tlb.judge_points([{"I": 0.05, "phi": 0.80}])
    comp = tlb.compare_branches(summary)
    assert "thickness-invariant" in comp["mu_statement"]
    assert "mu" not in comp["phi_statement"].replace("nu_max", "")


def test_statement_flags_sensitivity_when_present():
    summary = tlb.judge_points([{"I": 0.05, "phi": 0.783}])
    comp = tlb.compare_branches(summary)
    if comp["branch_sensitive"]:
        assert "BRANCH-SENSITIVE" in comp["phi_statement"]
    else:
        assert "BRANCH-INSENSITIVE" in comp["phi_statement"]


def test_compare_branches_reports_both_verdicts():
    summary = tlb.judge_points([{"I": 0.05, "phi": 0.80}])
    comp = tlb.compare_branches(summary)
    assert set(comp["phi_verdict_by_branch"]) == {"thick", "thin"}
    assert comp["layer_separation"]


# --------------------------------------------------------------------------- #
# build_audit wiring
# --------------------------------------------------------------------------- #
def test_build_audit_carries_geometry_and_points():
    pts = [{"I": 0.05, "phi": 0.80, "file": "x.json", "I_source": "I_bulk"}]
    audit = tlb.build_audit(pts, gap_eff=0.14, diameter=0.01)
    assert audit["geometry"]["layers"] == pytest.approx(14.0)
    assert audit["summary"]["n_points"] == 1
    assert audit["summary"]["rows"][0]["file"] == "x.json"
    assert audit["comparison"]["phi_verdict_by_branch"]
    md = tlb.render_tables(audit)
    assert "表 A1" in md and "表 A3" in md


def test_render_tables_contains_no_placeholder_dashes_for_numbers():
    pts = [{"I": 0.05, "phi": 0.80, "file": "x.json", "I_source": "I_bulk"}]
    md = tlb.render_tables(tlb.build_audit(pts, gap_eff=0.14, diameter=0.01))
    # the numeric row must show the interpolated I, not a dash
    assert "| x.json | 0.05000 |" in md

def test_cli_gap_means_total_gap_not_gap_eff(tmp_path):
    """``--gap`` is the wall-to-wall gap, so the audited cell is 14 layers."""
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    (run_dir / "prof_r31_w0.5_I0.05_s0.json").write_text(
        '{"phi_gap": 0.80, "I_bulk": 0.05, "polydispersity": 0.5,'
        ' "profile_derived": {"I_measured": 0.05}}')
    rc = tlb.main(["--run-dir", str(run_dir),
                   "--out-dir", str(tmp_path / "out"),
                   "--gap", "0.15", "--diameter", "0.01"])
    assert rc == 0
    import json
    audit = json.loads((tmp_path / "out" / "thin_layer_branch_audit.json").read_text())
    assert audit["geometry"]["layers"] == pytest.approx(14.0)
    assert audit["summary"]["n_points"] == 1
