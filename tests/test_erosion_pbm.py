"""Acceptance tests for the DEM PBM x erosion fusion surface (node p176w2-n08).

Every criterion of the node has a test here; the numbers asserted are the
criterion thresholds, and the study runner (:mod:`tensordem.erosion_pbm.study`)
records the same quantities in ``artifacts/p176w2_n08/metrics.json``.

The DEM fixtures use a short horizon and a small ensemble so the suite stays
fast; the acceptance run uses the full case (20 000 particles, 0.4 s).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.erosion_pbm import dem as dem_module
from tensordem.erosion_pbm.dem import (ImpactDEMConfig, impact_ensemble_erosion,
                                       ledger_closure, meanfield_selection,
                                       run_impact_dem)
from tensordem.erosion_pbm.erosion import (CARBON_STEEL, FINNIE_CRIT_ANGLE_DEG,
                                           FINNIE_PEAK_ANGLE_DEG, OkaErosionLaw,
                                           dnv_o501_erosion_ratio,
                                           dnv_o501_penetration_rate_mm_per_year,
                                           finnie_angle_function,
                                           fit_coefficient_loglog,
                                           fit_power_exponent,
                                           max_relative_deviation,
                                           oka_angle_function, oka_hv_exponents,
                                           oka_peak_angle_deg)
from tensordem.erosion_pbm.pbm import (SectionalGrid, bin_counts, daughter_matrix,
                                       mass_moment, number_moment,
                                       solve_breakage, wasserstein_1d)
from tensordem.erosion_pbm.strength import (GranularStrength, sample_strengths,
                                            weibull_plot_fit)
from tensordem.erosion_pbm.study import (CASE, build_grid, check_o501,
                                         check_pbm_analytic, check_weibull,
                                         dem_config, erosion_law, run_study)

SHORT = 0.1          # s, short horizon for the DEM fixtures
SMALL_N = 4000       # particles


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def case():
    c = dict(CASE)
    c["t_end"] = SHORT
    c["n_particles"] = SMALL_N
    return c


@pytest.fixture(scope="module")
def dem_run(case):
    grid = build_grid(case)
    cfg = dem_config(case)
    out = run_impact_dem(cfg, grid, t_end=SHORT,
                         record_times=[SHORT / 2.0, SHORT],
                         erosion_law=erosion_law(case))
    return grid, cfg, out


@pytest.fixture(scope="module")
def pbm_vs_dem(case, dem_run):
    grid, cfg, dem = dem_run
    S = meanfield_selection(grid, cfg)
    M = daughter_matrix(grid, n_fragments=cfg.n_fragments, rule="gaudin",
                        gaudin_alpha=cfg.gaudin_alpha)
    n0 = bin_counts(grid, np.full(cfg.n_particles, case["d_0"]))
    pbm = solve_breakage(grid, n0, S, M, t_end=SHORT, t_eval=[SHORT / 2.0, SHORT])
    return grid, dem, pbm


# ---------------------------------------------------------------------------
# C1 -- Weibull strength distribution
# ---------------------------------------------------------------------------
def test_c1_weibull_fit_r2_above_threshold():
    out = check_weibull(CASE)
    assert out["r2"] > 0.95, out["r2"]
    assert out["m_relative_error"] < 0.10, out["m_relative_error"]


def test_weibull_size_effect_matches_power_law():
    model = GranularStrength(sigma_0=25e6, m=2.6, d_0=150e-6)
    d = 150e-6 / 2.0
    expected = (150e-6 / d) ** (3.0 / 2.6)
    got = float(model.characteristic_strength(d) / model.characteristic_strength(150e-6))
    assert got == pytest.approx(expected, rel=1e-12)


def test_weibull_sampling_recovers_modulus():
    rng = np.random.default_rng(7)
    model = GranularStrength(sigma_0=25e6, m=3.5, d_0=150e-6)
    s = sample_strengths(model, np.full(6000, 150e-6), rng)
    fit = weibull_plot_fit(s)
    assert fit["m"] == pytest.approx(3.5, rel=0.06)
    assert fit["sigma_0"] == pytest.approx(25e6, rel=0.05)


# ---------------------------------------------------------------------------
# C2 -- PBM numerics against analytic solutions
# ---------------------------------------------------------------------------
def test_c2_pbm_analytic_errors_below_three_percent():
    out = check_pbm_analytic(CASE)
    assert out["cascade_max_relative_error"] < 0.03
    assert out["number_growth_max_relative_error"] < 0.03
    assert out["loss_only_decay_max_relative_error"] < 0.03


def test_c2_binary_cascade_matches_poisson_solution():
    # N_k(t) = e^{-t} (2t)^k / k!  on a one-section-per-octave volume grid
    grid = build_grid(CASE, sections_per_octave=1)
    j0 = grid.n_bins - 1
    n0 = np.zeros(grid.n_bins)
    n0[j0] = 1.0
    M = daughter_matrix(grid, n_fragments=2, rule="equal")
    t_end = 1.5
    sol = solve_breakage(grid, n0, 1.0, M, t_end=t_end,
                         t_eval=np.linspace(0.0, t_end, 16))
    for it, t in enumerate(sol["t"]):
        for k in range(8):
            exact = math.exp(-t) * (2.0 * t) ** k / math.factorial(k)
            assert sol["n"][j0 - k, it] == pytest.approx(exact, rel=1e-6, abs=1e-12)


def test_pbm_mass_and_number_moments_conserved():
    grid = build_grid(CASE)
    n0 = bin_counts(grid, np.full(1000, CASE["d_0"]))
    M = daughter_matrix(grid, n_fragments=3, rule="gaudin", gaudin_alpha=2.5)
    sol = solve_breakage(grid, n0, 0.8, M, t_end=0.3, t_eval=[0.3])
    mass0 = mass_moment(n0, grid)
    mass1 = mass_moment(sol["n"][:, -1], grid)
    assert abs(mass1 - mass0) / mass0 < 1e-9
    # number grows exactly as exp((nu - 1) S t)
    nu = float(np.sum(M[:, -1]))
    exact = number_moment(n0) * math.exp((nu - 1.0) * 0.8 * 0.3)
    assert number_moment(sol["n"][:, -1]) == pytest.approx(exact, rel=1e-8)


def test_pbm_zero_selection_keeps_distribution():
    grid = build_grid(CASE)
    n0 = bin_counts(grid, np.full(500, CASE["d_0"]))
    M = daughter_matrix(grid, n_fragments=3, rule="gaudin")
    sol = solve_breakage(grid, n0, 0.0, M, t_end=0.5, t_eval=[0.0, 0.5])
    assert np.array_equal(sol["n"][:, -1], n0)


def test_daughter_matrix_conserves_fragment_count_and_mass():
    grid = build_grid(CASE)
    M = daughter_matrix(grid, n_fragments=3, rule="gaudin", gaudin_alpha=2.5,
                        n_samples=1024)
    counts = M.sum(axis=0)
    assert np.allclose(counts, 3.0, atol=1e-9)
    for j in (10, 40, 80, grid.n_bins - 1):
        mass_out = float(np.sum(grid.volumes * M[:, j]))
        assert mass_out == pytest.approx(float(grid.volumes[j]), rel=2e-2)


def test_grid_refinement_reduces_wasserstein_error():
    from tensordem.erosion_pbm.study import check_grid_convergence
    out = check_grid_convergence({**CASE, "t_end": 0.05})
    values = [row["wasserstein_normalised_vs_q12"] for row in out["rows"][:-1]]
    assert values[0] > values[1] > values[2] > values[3]
    assert values[-1] < 0.03


# ---------------------------------------------------------------------------
# C3 -- DEM vs PBM
# ---------------------------------------------------------------------------
def test_c3_dem_pbm_wasserstein_below_threshold(pbm_vs_dem, case):
    grid, dem, pbm = pbm_vs_dem
    w1 = wasserstein_1d(grid, dem["psd_counts"][-1], pbm["n"][:, -1],
                        normalise_by=case["d_0"])
    assert w1 < 0.1, w1


def test_c3_dem_and_pbm_number_agree_within_noise(pbm_vs_dem):
    _grid, dem, pbm = pbm_vs_dem
    ratio = pbm["n"][:, -1].sum() / max(dem["psd_counts"][-1].sum(), 1.0)
    assert ratio == pytest.approx(1.0, abs=0.10), ratio


def test_quenched_strength_differs_from_mean_field(case):
    """Frozen (quenched) strengths make the mean-rate PBE over-predict breakage.

    The gap is a function of ``lambda * t`` (it vanishes for short horizons), so
    this check runs its own longer, smaller ensemble instead of the short
    fixture.
    """
    grid = build_grid(case)
    t_long = 0.35
    n = 1500
    base = dem_config(case, n_particles=n, seed=11)
    annealed = run_impact_dem(base, grid, t_end=t_long, record_times=[t_long],
                              erosion_law=None)
    quenched = run_impact_dem(dem_config(case, n_particles=n, seed=11,
                                         freeze_strength=True), grid,
                              t_end=t_long, record_times=[t_long], erosion_law=None)
    S = meanfield_selection(grid, base)
    M = daughter_matrix(grid, n_fragments=base.n_fragments, rule="gaudin",
                        gaudin_alpha=base.gaudin_alpha)
    n0 = bin_counts(grid, np.full(n, case["d_0"]))
    pbm = solve_breakage(grid, n0, S, M, t_end=t_long, t_eval=[t_long])
    assert quenched["break_events"] < annealed["break_events"]
    gap_annealed = pbm["n"][:, -1].sum() / annealed["n_particles_final"]
    gap_quenched = pbm["n"][:, -1].sum() / quenched["n_particles_final"]
    assert gap_annealed == pytest.approx(1.0, abs=0.12)
    assert gap_quenched > 1.2, (gap_annealed, gap_quenched)


# ---------------------------------------------------------------------------
# erosion laws: angular peaks and velocity exponent
# ---------------------------------------------------------------------------
def test_finnie_peak_angle_inside_campaign_window():
    assert FINNIE_PEAK_ANGLE_DEG == pytest.approx(math.degrees(0.5 * math.atan(2.0 / 3.0)))
    assert 15.0 <= FINNIE_PEAK_ANGLE_DEG <= 25.0
    a = np.linspace(1e-6, 90.0, 200001)
    assert float(a[int(np.argmax(finnie_angle_function(a)))]) == pytest.approx(
        FINNIE_PEAK_ANGLE_DEG, abs=0.01)


def test_finnie_angular_function_continuity_and_limits():
    left = float(finnie_angle_function(FINNIE_CRIT_ANGLE_DEG - 1e-9))
    right = float(finnie_angle_function(FINNIE_CRIT_ANGLE_DEG + 1e-9))
    assert left == pytest.approx(0.3, rel=1e-6)
    assert right == pytest.approx(0.3, rel=1e-6)
    assert float(finnie_angle_function(90.0)) == pytest.approx(0.0, abs=1e-12)


def test_oka_peak_angle_inside_window_for_case_hardness():
    peak = oka_peak_angle_deg(CASE["hv_gpa"])
    assert 15.0 <= peak <= 25.0, peak
    # documented hardness dependence: harder target pushes the peak up
    assert oka_peak_angle_deg(1.0) > peak
    assert oka_peak_angle_deg(4.0) > oka_peak_angle_deg(1.0)


def test_oka_angular_function_normalised_at_normal_impact():
    for hv in (0.25, 0.5, 1.0, 4.0):
        assert float(oka_angle_function(90.0, hv)) == pytest.approx(1.0, rel=1e-12)


def test_velocity_exponent_calibration_inside_window():
    law = OkaErosionLaw(hv_gpa=CASE["hv_gpa"], ratio_ref=CASE["ratio_ref"],
                        u_ref=CASE["u_mean"], d_ref=CASE["d_0"])
    speeds = np.linspace(20.0, 40.0, 21)
    ratio = law.ratio(speeds, 90.0, np.full(speeds.size, CASE["d_0"]))
    fit = fit_power_exponent(speeds, ratio)
    assert 2.1 <= fit["n"] <= 2.5, fit["n"]
    assert fit["n"] == pytest.approx(oka_hv_exponents(CASE["hv_gpa"])[2], rel=1e-9)
    assert fit["r2"] > 0.999


def test_size_factor_saturates_above_reference_diameter():
    law = OkaErosionLaw(hv_gpa=0.5, ratio_ref=2e-4, d_ref=150e-6)
    coarse = float(law.ratio(30.0, 90.0, 300e-6))
    ref = float(law.ratio(30.0, 90.0, 150e-6))
    fine = float(law.ratio(30.0, 90.0, 50e-6))
    assert coarse == pytest.approx(ref, rel=1e-12)
    assert fine < ref


# ---------------------------------------------------------------------------
# C4 -- erosion rate against DNV-RP-O501
# ---------------------------------------------------------------------------
def test_c4_per_point_deviation_below_fifteen_percent():
    out = check_o501(CASE)
    assert out["max_abs_deviation"] < 0.15, out["deviations"]
    assert out["K_coefficient_verified"] is False  # honesty flag must travel


def test_o501_rate_structure_scales_as_power_law_and_linearly_with_sand_rate():
    u = np.array([20.0, 30.0, 40.0])
    ratio = dnv_o501_erosion_ratio(u, CARBON_STEEL, 1.0, K=1.0)
    fit = fit_power_exponent(u, ratio)
    assert fit["n"] == pytest.approx(CARBON_STEEL.n, rel=1e-12)
    r1 = dnv_o501_penetration_rate_mm_per_year(
        u=30.0, sand_rate_kg_per_s=1e-3, impacted_area_m2=1e-2,
        target_density=2700.0, K=3e-8)
    r2 = dnv_o501_penetration_rate_mm_per_year(
        u=30.0, sand_rate_kg_per_s=2e-3, impacted_area_m2=1e-2,
        target_density=2700.0, K=3e-8)
    assert r2 == pytest.approx(2.0 * r1, rel=1e-12)


def test_resolved_ensemble_reproduces_nominal_ratio():
    law = erosion_law(CASE)
    cfg = dem_config(CASE)
    ens = impact_ensemble_erosion(cfg, law, u_mean=30.0, alpha_mean_deg=90.0,
                                  alpha_std_deg=6.0, n_samples=50000)
    assert ens["jensen_factor"] == pytest.approx(1.0, abs=0.05)
    assert ens["mean_ratio"] == pytest.approx(float(law.ratio(30.0, 90.0, CASE["d_0"])),
                                              rel=0.05)
    assert max_relative_deviation([ens["nominal_ratio"]], [ens["mean_ratio"]]) < 0.05


# ---------------------------------------------------------------------------
# C5 -- mass-loss ledger
# ---------------------------------------------------------------------------
def test_c5_ledger_residual_below_five_percent(dem_run, case):
    grid, _cfg, dem = dem_run
    led = ledger_closure(dem["ke_by_bin"], dem["removed_by_bin"], grid,
                         erosion_law(case), u_ref=case["u_mean"])
    assert led["removed_mass"] > 0.0
    assert led["residual_per_bin"] < 0.05, led
    assert led["residual_global"] < 0.05, led


# ---------------------------------------------------------------------------
# C6 -- negative controls
# ---------------------------------------------------------------------------
def test_c6_infinite_strength_produces_no_fragments(dem_run, case):
    grid, _cfg, _dem = dem_run
    hard = dem_config(case, strength=GranularStrength(sigma_0=1e12, m=case["weibull_m"],
                                                     d_0=case["d_0"]), seed=5)
    out = run_impact_dem(hard, grid, t_end=SHORT, record_times=[SHORT],
                        erosion_law=None)
    assert out["break_events"] == 0
    psd = out["psd_counts"][-1]
    parent_bin = int(grid.index_of_diameter(np.array([case["d_0"]]))[0])
    assert (psd.sum() - psd[parent_bin]) / psd.sum() < 0.03


def test_c6_zero_impacts_gives_exactly_zero_removal(dem_run, case):
    grid, _cfg, _dem = dem_run
    quiet = dem_config(case, f_impact=0.0, seed=6)
    out = run_impact_dem(quiet, grid, t_end=SHORT, record_times=[SHORT],
                         erosion_law=erosion_law(case))
    assert out["removed_mass"] < 1e-12
    assert out["impacts"] == 0
    assert out["break_events"] == 0


def test_dem_conserves_particle_volume_through_breakage(dem_run):
    _grid, _cfg, dem = dem_run
    assert dem["volume_residual"] < 1e-12
    assert dem["break_events"] > 0


# ---------------------------------------------------------------------------
# end-to-end study + C7
# ---------------------------------------------------------------------------
def test_end_to_end_study_passes_all_criteria(tmp_path):
    res = run_study(tmp_path, t_end=0.06, n_particles=2000, with_figures=False)
    assert set(res["criteria"]) >= {
        "C1_weibull_r2_gt_0.95", "C2_pbm_vs_analytic_lt_3pct",
        "C3_dem_pbm_wasserstein_lt_0.1", "C4_o501_per_point_lt_15pct",
        "C5_mass_ledger_lt_5pct", "C6_negative_controls"}
    assert res["all_pass"], res["criteria"]
    assert (tmp_path / "metrics.json").exists()


def test_test_count_covers_criterion_c7():
    """C7: the suite must hold at least 15 cases (counted at collection)."""
    import pathlib
    text = pathlib.Path(__file__).read_text()
    assert text.count("def test_") >= 15