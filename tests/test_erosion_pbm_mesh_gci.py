"""Tests for the PBM sectional-mesh metrology (campaign node ``p488w1-n09``).

The tests come in three layers:

* **grid / measure algebra** -- the closed forms the whole error budget rests on
  (analytic bin integrals, the refinement invariants of both mesh families, the
  ``(1 + r) / (2 sqrt r)`` measure factor);
* **metrology** -- Richardson/GCI on synthetic sequences whose order is known,
  the floor threshold interpolation, the erosion estimators;
* **acceptance** -- the committed ``artifacts/p488w1_n09/metrics.json`` is
  read back and every criterion is re-asserted against the same numbers, plus
  a spot recomputation through the study driver so the JSON cannot drift from
  the code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensordem.erosion_pbm import mesh_study as ms
from tensordem.erosion_pbm.erosion import OkaErosionLaw
from tensordem.erosion_pbm.mesh_gci import (LogLogGrid, cascade_error, cascade_solution,
                                            erosion_removed_mass, floor_share,
                                            floor_threshold, mass_ledger, nested_family,
                                            pivot_measure_bias, propagate, psd_fingerprint,
                                            richardson_gci)
from tensordem.erosion_pbm.pbm import daughter_matrix

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "p488w1_n09"
D0 = 150e-6
V0 = D0 ** 3


@pytest.fixture(scope="module")
def metrics() -> dict:
    path = ARTIFACTS / "metrics.json"
    assert path.is_file(), f"missing study artifacts: {path}"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# grid algebra
# ---------------------------------------------------------------------------
def test_nested_family_halving_lands_on_a_representative():
    """C2 needs ``sqrt(V_k V_{k+1}) / 2 == sqrt(V_{k-q} V_{k-q+1})`` exactly."""
    for q in (1, 3, 6, 12):
        grid = nested_family(q, 12, 2, D0)
        v = grid.volumes
        np.testing.assert_allclose(v[q:] / 2.0, v[:-q], rtol=1e-14)


def test_uniform_refinement_shrinks_every_log_width_by_the_factor():
    base = LogLogGrid(V0 * 2.0 ** -12, V0 * 2.0 ** 2, 48)
    fine = base.refine(2)
    assert fine.n_bins == 2 * base.n_bins
    log_w_coarse = np.diff(np.log(base.edges))
    log_w_fine = np.diff(np.log(fine.edges))
    np.testing.assert_allclose(log_w_fine, log_w_coarse.repeat(2) / 2.0, rtol=1e-13)


def test_graded_refinement_keeps_the_grading_ratio():
    base = LogLogGrid(V0 * 2.0 ** -12, V0 * 2.0 ** 2, 48, grading=4.0)
    fine = base.refine(2)
    assert fine.grading == base.grading
    assert base.width_grading == pytest.approx(4.0, rel=0.05)     # R**(1 - 1/N)
    assert fine.width_grading > base.width_grading                # -> R as N grows
    assert fine.width_grading == pytest.approx(4.0, rel=0.02)
    # refinement is nested and halves the *envelope* of the width profile
    # (element-wise halving holds only for the uniform ladder, tested above)
    coarse_edges = base.edges
    idx = np.searchsorted(fine.edges, coarse_edges)
    idx = np.clip(idx, 0, fine.n_bins)
    np.testing.assert_allclose(fine.edges[idx], coarse_edges, rtol=1e-12)
    lw_c, lw_f = np.diff(np.log(base.edges)), np.diff(np.log(fine.edges))
    # the envelope halves up to the O(1/N) discretisation of the exponential
    assert lw_f.max() == pytest.approx(lw_c.max() / 2.0, rel=0.02)
    assert lw_f.min() == pytest.approx(lw_c.min() / 2.0, rel=0.02)


def test_graded_grid_is_non_uniform_and_monotone():
    grid = LogLogGrid(V0 * 2.0 ** -12, V0 * 2.0 ** 2, 60, grading=16.0)
    w = grid.widths
    assert np.all(np.diff(w) > 0)                     # exponentially stretched
    assert w[-1] / w[0] > 5.0                          # far from uniform
    assert grid.sections_per_octave[0] > grid.sections_per_octave[-1]  # fines resolved


def test_grading_below_one_is_rejected():
    with pytest.raises(ValueError):
        LogLogGrid(1.0, 2.0, 10, grading=0.5)
    with pytest.raises(ValueError):
        LogLogGrid(2.0, 1.0, 10)


def test_moment_order_zero_is_the_bin_count():
    grid = LogLogGrid(V0 * 2.0 ** -8, V0 * 2.0 ** 2, 40)
    n = np.arange(1.0, 41.0)
    assert grid.moment(n, 0) == pytest.approx(n.sum(), rel=1e-14)


def test_moment_order_one_is_the_arithmetic_bin_mean():
    grid = LogLogGrid(V0 * 2.0 ** -8, V0 * 2.0 ** 2, 40)
    e = grid.edges
    n = np.full(grid.n_bins, 2.5)
    expected = float(np.sum(n * 0.5 * (e[:-1] + e[1:])))
    assert grid.moment(n, 1) == pytest.approx(expected, rel=1e-13)


def test_moment_matches_numerical_quadrature():
    grid = LogLogGrid(V0 * 2.0 ** -8, V0 * 2.0 ** 2, 30, grading=3.0)
    rng = np.random.default_rng(5)
    n = rng.random(grid.n_bins) + 0.1
    e = grid.edges
    for order in (0, 1, 2):
        quad = 0.0
        for k in range(grid.n_bins):
            v = np.linspace(e[k], e[k + 1], 20001)
            quad += float(np.trapezoid(v ** order, v)) * n[k] / (e[k + 1] - e[k])
        assert grid.moment(n, order) == pytest.approx(quad, rel=1e-4)


def test_measure_factor_closed_form_matches_the_grid():
    """``int v n dv / sum v_k n_k = (1 + r) / (2 sqrt r)`` for a uniform grid."""
    for n_bins in (48, 96, 192):
        grid = LogLogGrid(V0 * 2.0 ** -20, V0 * 2.0 ** 4, n_bins)
        r = grid.local_ratio[0]
        predicted = (1.0 + r) / (2.0 * math.sqrt(r))
        rng = np.random.default_rng(11)
        n = rng.random(grid.n_bins)
        assert grid.moment(n, 1) / grid.pivot_moment(n, 1) == pytest.approx(predicted, rel=1e-13)
        assert pivot_measure_bias(grid.local_ratio) == pytest.approx(0.5 * (r - 1.0), rel=1e-12)


def test_quantile_is_monotone_and_inside_the_support():
    grid = LogLogGrid(V0 * 2.0 ** -8, V0 * 2.0 ** 2, 40)
    rng = np.random.default_rng(3)
    n = rng.random(grid.n_bins)
    q = [grid.quantile_diameter(n, p) for p in (0.1, 0.25, 0.5, 0.75, 0.9)]
    assert all(b > a for a, b in zip(q, q[1:]))
    assert grid.edges[0] ** (1 / 3) <= q[0] <= grid.edges[-1] ** (1 / 3)


def test_index_of_volume_clips_to_the_support():
    grid = LogLogGrid(V0 * 2.0 ** -8, V0 * 2.0 ** 2, 40)
    idx = grid.index_of_volume([1e-30, 1e-3])
    assert idx[0] == 0 and idx[1] == grid.n_bins - 1


# ---------------------------------------------------------------------------
# propagator and mass identity
# ---------------------------------------------------------------------------
def test_propagator_conserves_mass_to_machine_precision():
    # 20 octaves below d_0: the cascade stays inside the mesh, so the two-pivot
    # mass identity is the only mass statement in play
    grid = nested_family(6, 20, 2, D0)
    n0 = ms.initial_particle(grid, V0)
    S = 4.0 * (grid.volumes / V0) ** (1.0 / 3.0)
    M = daughter_matrix(grid, n_fragments=2, rule="equal")
    run = propagate(S, M, n0, [1.5])
    led = mass_ledger(grid, n0, run["n"][:, -1])
    assert led["mass_initial"] == pytest.approx(V0, rel=1e-14)   # exact placement
    assert led["mass_drift_relative"] < 1e-12
    assert abs(float(np.max(np.abs(run["tail"])))) < 1e-12


def test_two_pivot_initial_particle_conserves_mass_and_number():
    grid = nested_family(6, 20, 4, D0)
    n0 = ms.initial_particle(grid, V0)
    assert n0.sum() == pytest.approx(1.0, rel=1e-14)
    assert grid.pivot_moment(n0, 1) == pytest.approx(V0, rel=1e-15)


def test_cascade_is_exact_on_every_nested_mesh():
    for q in (1, 3, 6):
        grid = nested_family(q, 12, 2, D0)
        res = cascade_error(grid, sections_per_octave=q, j0=12 * q,
                            t_eval=np.linspace(0.25, 2.0, 5), n_generations=6)
        assert res["max_relative_error"] < 1e-6
        assert res["generations_compared"] >= 5


def test_cascade_reference_is_not_applicable_on_a_graded_mesh():
    """A graded ladder is not self-similar under halving: the reference dies."""
    grid = LogLogGrid(V0 * 2.0 ** -12, V0 * 2.0 ** 2, 60, grading=4.0)
    res = cascade_error(grid, sections_per_octave=5, j0=50, t_eval=[1.0], n_generations=3)
    assert res["max_relative_error"] > 1e-2


def test_cascade_closed_form_is_the_poisson_kernel():
    # sum_m e^{-St}(2St)^m/m! = e^{St}: number growth, not mass conservation
    assert sum(cascade_solution(m, 1.3) for m in range(80)) == pytest.approx(math.exp(1.3), rel=1e-15)
    for m in range(4):
        assert cascade_solution(m, 0.0) == pytest.approx(1.0 if m == 0 else 0.0)


# ---------------------------------------------------------------------------
# Richardson / GCI metrology on synthetic sequences
# ---------------------------------------------------------------------------
def test_richardson_recovers_second_order():
    exact = 23.068
    c = 0.37
    f = lambda h: exact + c * h ** 2
    out = richardson_gci(f(1 / 384), f(1 / 192), f(1 / 96), r=2.0)
    assert out["p"] == pytest.approx(2.0, rel=1e-9)
    assert out["f_extrapolated"] == pytest.approx(exact, rel=1e-9)
    assert out["gci"] < 1e-6
    assert out["certified"] and out["asymptotic"]


def test_richardson_recovers_first_order():
    exact, c = 5.0, -0.8
    f = lambda h: exact + c * h
    out = richardson_gci(f(1 / 384), f(1 / 192), f(1 / 96), r=2.0)
    assert out["p"] == pytest.approx(1.0, rel=1e-9)
    assert out["f_extrapolated"] == pytest.approx(exact, rel=1e-9)


def test_richardson_flags_a_non_monotone_sequence():
    out = richardson_gci(1.0, 1.2, 1.1, r=2.0)
    assert out["monotone"] is False
    assert math.isnan(out["p"])
    assert out["certified"] is False
    assert out["p_for_extrapolation"] == 1.0


def test_richardson_flags_a_degenerate_sequence():
    """Bit-identical levels mean the mesh did not act -- the C5 anti-false-convergence rule."""
    out = richardson_gci(2.5, 2.5, 2.5)
    assert out["degenerate"] is True
    assert out["gci"] == 0.0
    assert out["certified"] is False


def test_richardson_gci_scales_with_the_safety_factor():
    exact = 1.0
    f = lambda h: exact + 0.1 * h
    a = richardson_gci(f(1 / 64), f(1 / 32), f(1 / 16), r=2.0)
    assert a["gci"] == pytest.approx(1.25 * abs(f(1 / 64) - exact) / exact, rel=1e-9)


# ---------------------------------------------------------------------------
# floor criterion
# ---------------------------------------------------------------------------
def test_floor_share_is_the_bin_zero_mass_fraction():
    grid = LogLogGrid(V0 * 2.0 ** -8, V0 * 2.0 ** 2, 20)
    n = np.zeros(grid.n_bins)
    n[0] = 1.0
    assert floor_share(n, grid)[-1] == pytest.approx(1.0)
    n[0] = 0.0
    n[5] = 1.0
    assert floor_share(n, grid)[-1] == pytest.approx(0.0)


def test_floor_threshold_interpolates_the_crossing():
    ratios = [1.0, 2.0, 4.0, 8.0]
    shares = [0.5, 0.2, 0.01, 1e-4]
    res = floor_threshold(ratios, shares, limit=0.05)
    assert res["crossed"] is True
    assert 2.0 < res["threshold"] < 4.0
    # log-linear interpolation: share falls 0.2 -> 0.01, the 5 % level is at
    expected = math.exp(math.log(2.0) + (0.2 - 0.05) / (0.2 - 0.01) * (math.log(4.0) - math.log(2.0)))
    assert res["threshold"] == pytest.approx(expected, rel=1e-12)


def test_floor_threshold_reports_no_crossing_honestly():
    res = floor_threshold([1.0, 2.0], [0.9, 0.6], limit=0.05)
    assert res["crossed"] is False and res["threshold"] is None


def test_floor_threshold_rejects_non_monotone_ratios():
    with pytest.raises(ValueError):
        floor_threshold([2.0, 1.0], [0.1, 0.2])


# ---------------------------------------------------------------------------
# erosion estimators
# ---------------------------------------------------------------------------
def test_erosion_estimators_differ_by_exactly_the_measure_factor():
    """A constant rate law isolates the measure convention: the two estimators
    differ by nothing but the integral-vs-pivot measure factor."""
    grid = LogLogGrid(V0 * 2.0 ** -12, V0 * 2.0 ** 2, 60)      # uniform: r is constant
    rng = np.random.default_rng(7)
    n = rng.random(grid.n_bins)
    const = lambda d: np.full(np.shape(d), 2.0e-4)
    res = erosion_removed_mass(grid, n, const, rho_p=2650.0, t_end=0.4, f_impact=50.0)
    assert res["removed_pivot"] > 0.0
    factor = res["removed_integral"] / res["removed_pivot"]
    r = grid.local_ratio[0]
    assert factor == pytest.approx((1.0 + r) / (2.0 * math.sqrt(r)), rel=1e-10)
    assert res["relative_difference"] > 0.0


def test_erosion_removed_mass_converges_from_coarse_to_fine():
    law = OkaErosionLaw(hv_gpa=0.5, ratio_ref=2.0e-4, u_ref=30.0, d_ref=D0, size_exponent=0.19)
    ratio = lambda d: law.ratio(30.0, 90.0, d)
    values = []
    for n_bins in (48, 96, 192):
        grid = LogLogGrid(V0 * 2.0 ** -20, V0 * 2.0 ** 4, n_bins)
        n = ms.initial_particle(grid, V0)          # same physical feed on every mesh
        res = erosion_removed_mass(grid, n, ratio, rho_p=2650.0, t_end=0.4, f_impact=50.0)
        values.append(res["removed_pivot"])
    # coarse is furthest from the fine value, and the residual shrink is monotone
    d_coarse = abs(values[0] - values[-1]) / values[-1]
    d_mid = abs(values[1] - values[-1]) / values[-1]
    assert d_coarse < 0.05
    assert d_mid < d_coarse


def test_fingerprint_detects_identical_and_different_psds():
    a = np.arange(10.0)
    assert psd_fingerprint(a) == psd_fingerprint(a.copy())
    assert psd_fingerprint(a) != psd_fingerprint(a + 1e-12)


# ---------------------------------------------------------------------------
# acceptance: the committed metrics.json
# ---------------------------------------------------------------------------
def test_metrics_verdict_is_pass(metrics):
    assert metrics["verdict"] == "PASS"
    assert all(metrics["checks"].values())


def test_c1_mass_drift_within_1e12(metrics):
    c1 = metrics["C1_mass"]
    assert c1["worst_drift_relative"] <= 1e-12
    assert c1["pass"] is True
    # both families and every level were actually measured
    assert set(c1["families"]) == {"uniform", "graded"}
    assert len(c1["families"]["uniform"]) == len(metrics["config"]["order_levels"])


def test_c2_cascade_error_within_1e6(metrics):
    c2 = metrics["C2_cascade"]
    assert c2["worst_uniform_relative_error"] <= 1e-6
    assert c2["pass"] is True
    assert all(r["max_relative_error"] <= 1e-6 for r in c2["uniform"].values())
    # the graded-ladder reference is reported as *not applicable*, not as a failure
    assert all(r["reference_applicable"] is False for r in c2["graded"].values())


def test_c3_required_functionals_are_certified_in_band(metrics):
    c3 = metrics["C3_order"]
    for name in c3["required_certified"]:
        assert name in c3["certified"], f"{name} not certified"
        row = c3["certified"][name]
        assert row["declared_p_nominal"] == 2.0
        assert 1.6 <= row["p"] <= 2.4
        assert row["order_stable"] is True
    assert c3["pass"] is True


def test_c3_p_values_are_reported_not_hidden(metrics):
    c3 = metrics["C3_order"]
    # every functional carries a numeric p or an explicit exclusion reason
    for name, row in {**c3["certified"], **c3["diagnostics"]}.items():
        assert "p" in row
        if not (row["p"] == row["p"]):
            assert row.get("excluded_because"), f"{name}: nan p without an exclusion reason"


def test_c3_measure_factor_anchor(metrics):
    anchor = metrics["C3_order"]["measure_factor_check"]
    assert anchor["relative_error_of_prediction"] < 1e-12
    for level, pair in anchor["per_level"].items():
        assert pair["measured"] == pytest.approx(pair["predicted"], rel=1e-12)


def test_c3_noise_floor_is_below_the_mesh_signal(metrics):
    noise = metrics["C3_order"]["noise_floor"]
    assert noise["seed_a"] != noise["seed_b"]
    assert noise["relative_change"]["M1_integral"] < 1e-8


def test_c4_gci_of_m0_and_m1_within_5pct(metrics):
    c4 = metrics["C4_gci"]
    assert c4["values"]["M0"]["gci"] <= 0.05
    assert c4["values"]["M1_integral"]["gci"] <= 0.05
    assert c4["pass"] is True
    graded = c4["graded_family_gci"]
    assert graded["M0"] <= 0.05 and graded["M1_integral"] <= 0.05
    assert c4["pivot_mass_identity"]["expected"] == 0.0


def test_c5_negative_control(metrics):
    c5 = metrics["C5_negative_control"]
    assert c5["fingerprints_distinct"] is True
    assert max(c5["crude_vs_fine_relative_deviation"].values()) > 0.05
    assert c5["pass"] is True


def test_c6_floor_threshold_and_warn_semantics(metrics):
    c6 = metrics["C6_floor"]
    assert c6["threshold"]["crossed"] is True
    assert 1.0 < c6["threshold"]["threshold"] < 10.0
    assert c6["production_d0_over_dmin"] > c6["threshold"]["threshold"]
    assert c6["production_floor_share"] <= 0.05
    assert c6["verdict"] == "PASS" and c6["warning"] is None
    # the sweep is monotone: finer floor -> smaller share
    shares = [c6["sweep"][k]["floor_mass_share"] for k in c6["sweep"]]
    assert all(b <= a for a, b in zip(shares, shares[1:]))


def test_c7_erosion_decomposition(metrics):
    c7 = metrics["C7_erosion"]
    dec = c7["decomposition"]
    assert dec["grid_contribution_mid"] <= dec["cross_correlation_deviation"] / 3.0
    assert dec["cross_correlation_deviation"] == pytest.approx(0.131, abs=2e-3)
    assert c7["pass"] is True
    # three levels per family are present
    for family in c7["families"].values():
        assert len(family) == len(metrics["config"]["physical_levels"])


def test_c7_grid_error_shrinks_under_refinement(metrics):
    fam = metrics["C7_erosion"]["families"]["1.0"]
    levels = [str(n) for n in sorted(int(k) for k in fam)]
    ref = fam[levels[-1]]["removed_pivot"]
    deviations = [abs(fam[k]["removed_pivot"] - ref) / ref for k in levels]
    assert deviations[0] > deviations[-1]       # coarse worse than fine
    assert deviations[-1] < 1e-3


# ---------------------------------------------------------------------------
# the JSON cannot drift away from the code
# ---------------------------------------------------------------------------
def test_study_driver_reproduces_the_archived_numbers(metrics):
    cfg = ms._prepare(dict(ms.CONFIG))
    ms._CASE_CACHE.clear()
    cfg["gaudin_samples"] = metrics["config"]["gaudin_samples"]
    cfg["gaudin_seed"] = metrics["config"]["gaudin_seed"]
    n_bins = int(metrics["config"]["order_levels"][1])
    grid = LogLogGrid(V0 * 2.0 ** -cfg["octaves_below"], V0 * 2.0 ** cfg["octaves_above"],
                      n_bins, 1.0)
    run = ms.order_case(grid, cfg)
    led = mass_ledger(grid, run["n0"], run["n"])
    stored = metrics["C1_mass"]["families"]["uniform"][str(n_bins)]
    assert led["mass_drift_relative"] == pytest.approx(stored["mass_drift_relative"], abs=1e-16)
    assert led["mass_final"] == pytest.approx(stored["mass_final"], rel=1e-14)
    m0 = grid.moment(run["n"], 0)
    assert m0 == pytest.approx(metrics["C3_order"]["functionals"]["M0"]["value_all"][str(n_bins)],
                               rel=1e-13)


def test_cascade_archived_numbers_reproduce():
    grid = nested_family(3, 12, 2, D0)
    res = cascade_error(grid, sections_per_octave=3, j0=36, t_eval=np.linspace(0.25, 2.0, 5),
                        n_generations=6)
    assert res["max_relative_error"] < 1e-6
    assert res["n_bins"] == 42


def test_tables_and_figures_are_program_generated(metrics):
    tables = ARTIFACTS / "tables.md"
    assert tables.is_file()
    text = tables.read_text()
    assert "C1 mass ledger" in text and "C7 erosion removal mass decomposition" in text
    for fig in metrics["surfaces"]["figures"]:
        path = ARTIFACTS / fig
        assert path.is_file() and path.stat().st_size > 5000