"""R4 of CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT: the elastic-limit coefficient.

The physics claim, the derivation and the artefact map are in
``docs/CASE_DEM_SHEAR_CAMPBELL_ELASTIC_LIMIT_R4_ELASTIC_COEFFICIENT.md``; the
driver is ``benchmarks/case_dem_shear_campbell_elastic_limit/ell_coefficient.py``.

These tests do two different jobs and it matters which is which:

* **unit / algebra** -- the virial and fabric reductions are checked against a
  hand-computable configuration, the ``kappa -> d`` inversion is round-tripped and
  the saturation fit is checked on synthetic data with a known asymptote;
* **honesty / reproducibility** -- the shipped ``analysis.json`` is recomputed
  from the shipped point files, and the two honest FAILs (P4, P5) are asserted *as
  FAIL*.  A later change that quietly turns them into PASS without new runs must
  break this suite.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path as _Path
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "case_dem_shear_campbell_elastic_limit"
sys.path.insert(0, str(BENCH))

import ell_coefficient as ec  # noqa: E402


# --------------------------------------------------------------------- algebra
def test_diameter_from_kappa_round_trips_the_case_definition():
    kn, rho, gamma_dot, d = 1.0e4, 1.0, 1.0, 1.2
    kappa = kn / (rho * d ** 3 * gamma_dot ** 2)
    assert ec.diameter_from_kappa(kn, kappa, rho, gamma_dot) == pytest.approx(d, rel=1e-12)


def test_contact_geometry_reproduces_a_hand_computable_pair():
    """Two disks overlapping on the diagonal: the virial and fabric are exact."""
    d = 1.0
    r = np.array([d / 2, d / 2])
    lx = ly = 20.0 * d          # far enough that the min-image is the direct pair
    eps = 0.1 * d               # overlap
    s = math.sqrt(2.0) / 2.0
    off = math.sqrt(2.0) * (d - eps) / 2.0
    x = np.array([5.0, 5.0 - off])      # compressive quadrant (n_x n_y < 0 raw)
    y = np.array([5.0, 5.0 + off])
    kn = 1.0
    geo = ec.contact_geometry(x, y, r, lx, ly, 0.0, kn)
    assert geo["n_contacts"] == 1
    # normal points i -> j, i.e. (-s, +s); the raw n_x n_y is -0.5 and the reported
    # (physical, compressive-quadrant) fabric is the flipped +0.5
    assert geo["fabric_xy"] == pytest.approx(0.5, rel=1e-12)
    assert geo["mean_overlap_ratio_snapshot"] == pytest.approx(eps / d, rel=1e-12)
    assert geo["coordination_snapshot"] == pytest.approx(1.0, rel=1e-12)
    # sigma_yx of one compressive contact: force on i is -k delta n, branch vector
    # (d-eps) n -> sigma_yx = +k delta (d-eps) n_x n_y / area (positive here, as in a
    # sheared packing: the compressive axis is the (-x,+y) diagonal)
    expect = kn * eps * (d - eps) * s * s / (lx * ly)
    assert geo["sigma_yx_normal"] == pytest.approx(expect, rel=1e-12)
    # pressure is the same quantity for this single diagonally oriented contact
    assert geo["pressure_normal"] == pytest.approx(expect, rel=1e-12)


def test_factored_identity_error_is_exactly_the_overlap_it_neglects():
    """The factorised form replaces the separation by d; for a uniform packing the
    error is therefore exactly (1 - delta/d) and must be *predictable*, not fudged."""
    d, eps, kn = 1.0, 0.02, 3.0
    r = np.full(4, d / 2)
    lx = ly = 4.0 * d
    # a 2x2 block compressed along both diagonals: all pairs touching on the diagonal
    off = math.sqrt(2.0) * (d - eps) / 2.0
    x = np.array([1.0, 1.0 + off, 1.0 + off, 1.0])
    y = np.array([1.0, 1.0 + off, 1.0, 1.0 + off])
    geo = ec.contact_geometry(x, y, r, lx, ly, 0.0, kn)
    phi = 2.0 * math.pi * geo["mean_r2"] * 4 / (lx * ly)
    a_normal = geo["sigma_yx_normal"] * d / kn
    a_factored = (phi * geo["coordination_snapshot"] * d ** 3
                  / (2.0 * math.pi * geo["mean_r2"])) * geo["overlap_weighted_fabric_xy"]
    assert a_factored == pytest.approx(a_normal * (1.0 - eps / d), rel=1e-10)


def test_saturation_fit_recovers_a_known_asymptote():
    a_inf, c, g = 0.0130, 0.18, 0.5
    kappa = np.logspace(2.5, 5.8, 11)
    a = a_inf + c * kappa ** (-g)
    fit = ec.fit_saturation(kappa, a, [1e-4] * kappa.size)
    assert fit["A_inf"] == pytest.approx(a_inf, rel=2e-3)
    assert fit["g"] == pytest.approx(g, abs=2e-2)


def test_coefficient_is_tau_star_over_kappa():
    point = {"kn": 1.0e6, "kappa": 578703.7037037038, "tau": 11310.0, "gamma_dot": 1.0}
    assert ec.coefficient(point) == pytest.approx(point["tau"] * 1.2 / point["kn"], rel=1e-12)


# ------------------------------------------------------- shipped artefacts
@pytest.fixture(scope="module")
def shipped():
    path = _Path(ec.RUNS) / "analysis.json"
    if not path.exists():
        pytest.skip("R4 artefacts not run in this checkout")
    with open(path) as fh:
        return json.load(fh)


def test_prereg_was_written_before_the_runs_and_is_hashed(shipped):
    with open(_Path(ec.RUNS) / "prereg.json") as fh:
        prereg = json.load(fh)
    assert prereg["script_sha256"] == shipped["prereg_script_sha256"]
    assert [p["id"] for p in prereg["predictions"]] == ["P1", "P2", "P3", "P4", "P5"]
    assert prereg["predictions"][0]["threshold"] == 0.15
    assert prereg["grid"]["phi"] == list(ec.PHI_GRID)


def test_gated_runs_are_used_and_rejected_runs_are_reported(shipped):
    """9 runs were made; the two phi=0.71 seeds that fail the coordination gate are
    reported as rejected (never re-run until they pass) and stay out of the fit."""
    pts = ec.load_points(ec.RUNS, "fab_")
    assert len(pts) == 9
    gated = [p for p in pts if p["gates"]["all"]]
    assert len(gated) == shipped["leg_b_n_gated"]
    assert len(gated) + len(shipped["leg_b_rejected"]) == 9
    for rej in shipped["leg_b_rejected"]:
        assert "coordination" in rej["failed"]
        assert rej["coordination"] < 1.5
    assert sorted({p["phi"] for p in gated}) == list(ec.PHI_GRID)


def test_verdicts_are_reproducible_from_the_shipped_point_files(shipped):
    fresh = ec.analyse()
    assert fresh["verdicts"] == shipped["verdicts"]
    assert fresh["leg_a_fit"]["A_inf"] == pytest.approx(shipped["leg_a_fit"]["A_inf"], rel=1e-12)
    assert fresh["leg_a_top_spread"] == pytest.approx(shipped["leg_a_top_spread"], rel=1e-12)
    assert fresh["leg_c_worst_residual"] == pytest.approx(shipped["leg_c_worst_residual"], rel=1e-12)


def test_the_two_honest_fails_stay_fails(shipped):
    """P4/P5 were pre-registered as FAIL; flipping them needs new runs, not edits."""
    assert shipped["verdicts"]["P4"] == "FAIL"
    assert shipped["verdicts"]["P5"] == "FAIL"
    assert shipped["leg_b_a_xy_spread"] > shipped["tolerance"]
    assert shipped["leg_c_worst_residual"] > shipped["tolerance"]


def test_the_coefficient_is_not_compared_to_a_paywalled_table(shipped):
    leg = shipped["coefficient_leg"]
    assert "WITHHELD" in leg["campbell_table"]
    assert leg["status"].startswith("MEASURED")


def test_no_key_quantity_is_claimed_below_three_percent(shipped):
    """The 3% question is answered explicitly: nothing is demonstrated at 3%."""
    assert shipped["leg_a_top_spread"] > 0.03
    assert shipped["leg_a_A_inf_vs_stiffest_rel"] > 0.03
    assert shipped["leg_c_worst_residual"] > 0.03
    assert shipped["leg_b_factored_residual_absmax"] > 0.03


def test_saturation_band_is_inside_the_ledger_tolerance(shipped):
    assert shipped["verdicts"]["P1"] == "PASS"
    assert shipped["leg_a_top_spread"] <= shipped["tolerance"]
    assert shipped["verdicts"]["P2"] == "PASS"
    assert shipped["leg_a_A_inf_vs_stiffest_rel"] <= shipped["tolerance"]


def test_virial_reconstruction_agrees_with_the_run_on_every_fabric_point():
    """The independent contact enumeration must not fight the solver's estimator."""
    for p in ec.load_points(ec.RUNS, "fab_"):
        # normal-force share of the shear stress: physical, in (0, 1]
        assert 0.0 < p["f_norm"] <= 1.02, p["tag"]
        # the single snapshot's coordination is only approximately the run's time
        # average: this residual IS the single-snapshot caveat (measured, not assumed)
        assert abs(p["snapshot_vs_timeavg_residual"]) < 0.25, p["tag"]
        # normal forces alone must under-carry the shear stress (friction adds to it)
        assert p["f_norm"] < 1.0, p["tag"]
        # the branch vector is the centre separation, so its shortening is delta/dist
        assert 0.0 < p["geometry"]["branch_shortening"] < 0.1, p["tag"]