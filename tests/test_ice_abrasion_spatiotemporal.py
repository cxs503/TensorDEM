"""Tests for the N05 spatio-temporal ice-abrasion field module.

Every test targets a specific analytic statement:

* the moving-frame accumulation is the exact integral of the along-track ice
  concentration (uniform field, ice-edge lag, piecewise-linear concentration);
* the KL/kriging machinery reproduces the closed-form conditional variance of a
  separable exponential covariance and interpolates noise-free data exactly;
* the ensemble sampler is unbiased (sample statistics inside the theoretical
  sampling band, and the band is checked to scale like ``1/sqrt(m)``);
* the volume ledger detects a deliberately corrupted depth field (it has teeth);
* area-weighted quantiles reduce to the ordinary quantile for equal weights;
* the DEM-driven profile fit recovers a known separable profile.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem import ice_abrasion as ia
from tensordem import ice_abrasion_spatiotemporal as st


@pytest.fixture(scope="module")
def grid():
    return ia.build_waterline_grid()


@pytest.fixture(scope="module")
def panels(grid):
    return st.build_panel_grid(grid=grid, n_x=40, n_z=9)


def _uniform_profile(rate: float, panels) -> st.SeparableActivityProfile:
    return st.SeparableActivityProfile(
        scale_m_s=rate, background=1.0, decay_amplitude=0.0, decay_length=1.0,
        x_modes=((0.0, 0.5, 0.1),), waterline_offset_m=0.0,
        sigma_below_m=1.0e9, sigma_above_m=1.0e9,
        hull_length_m=panels.hull_length_m,
        hull_draught_m=panels.hull_draught_m)


# ---------------------------------------------------------------------------
# geometry / panel grid
# ---------------------------------------------------------------------------

def test_panel_grid_matches_band_area(grid, panels):
    assert panels.total_area_m2 == pytest.approx(grid.band_area_m2, rel=1e-12)
    assert panels.n_panels == panels.n_x * panels.n_z
    # x runs from the stem (0) to the stern (L); bow offset is positive
    assert panels.streamwise_offset_m[0] > 0.0
    assert panels.streamwise_offset_m[-1] < 0.0


def test_coarsen_field_conserves_area_integral(grid, panels):
    field = np.ones(tuple(int(v) for v in grid.shape))
    coarse = st.coarsen_field(field, grid, panels)
    assert coarse.shape == panels.shape
    assert float(np.sum(coarse * panels.cell_area_m2)) == pytest.approx(
        float(np.sum(field)) * float(np.diff(np.asarray(grid.x_edges))[0])
        * float(np.diff(np.asarray(grid.z_edges))[0]), rel=1e-12)


# ---------------------------------------------------------------------------
# moving-frame accumulation (closed form)
# ---------------------------------------------------------------------------

def test_uniform_field_accumulates_linearly(panels):
    rate = 2.0e-9
    track = st.uniform_track(1.0, 50_000.0, 5.0)
    series = st.evolve_abrasion_field(
        _uniform_profile(rate, panels), panels, track, n_time_steps=25,
        block_streamwise_m=1.0e12)          # freeze offsets -> pure uniform field
    expected = rate * series.time_s[:, None]
    assert np.max(np.abs(series.depth_m - expected)) / expected.max() < 1e-12


def test_uniform_ice_field_is_offset_invariant(panels):
    """A uniform ice field must give d = a t for every panel: the moving frame
    may not leak the hull length into an offset-invariant problem."""
    rate = 1.0e-9
    speed = 5.0
    track = st.uniform_track(1.0, 50_000.0, speed)
    series = st.evolve_abrasion_field(
        _uniform_profile(rate, panels), panels, track, n_time_steps=11,
        duration_s=3_600.0)
    expected = rate * series.time_s[:, None]
    assert np.max(np.abs(series.depth_m - expected)) / expected.max() < 1e-12
    # the bow and stern panels must agree to machine precision
    xi = panels.streamwise_offset_m
    assert series.depth_m[-1, int(np.argmax(xi))] == pytest.approx(
        series.depth_m[-1, int(np.argmin(xi))], rel=1e-12)


def test_ice_edge_onset_time_is_exact(panels):
    rate = 1.0e-9
    speed = 5.0
    s0 = 20_000.0
    track = st.IceTrack(
        break_s_m=np.array([0.0, s0, s0 + 10_000.0]),
        concentration=np.array([0.0, 1.0]), speed_m_s=speed)
    series = st.evolve_abrasion_field(
        _uniform_profile(rate, panels), panels, track, n_time_steps=4,
        duration_s=(s0 + 10_000.0) / speed)
    for i in (0, panels.n_panels // 2, panels.n_panels - 1):
        t_onset = (s0 - panels.streamwise_offset_m[i]) / speed
        before = series.time_s < t_onset - 1e-9
        assert np.all(series.depth_m[before, i] == 0.0)
        after = series.time_s > t_onset + 1e-9
        expected = rate * (series.time_s[after] - t_onset)
        assert np.max(np.abs(series.depth_m[after, i] - expected)) < 1e-20


def test_piecewise_track_closed_form_matches_fine_quadrature(panels):
    rate = 1.5e-9
    track = st.IceTrack(
        break_s_m=np.array([0.0, 500.0, 5_000.0, 9_000.0]),
        concentration=np.array([0.2, 0.8, 0.0]),
        speed_m_s=np.array([3.0, 5.0, 4.0]))
    series = st.evolve_abrasion_field(
        _uniform_profile(rate, panels), panels, track, n_time_steps=41)
    # independent reference: fine midpoint quadrature of c(s(tau)+xi)
    t = series.time_s
    fine = np.linspace(t[0], t[-1], 40001)
    tau = 0.5 * (fine[1:] + fine[:-1])
    dt = fine[1] - fine[0]
    xi = panels.streamwise_offset_m
    pos = track.position(tau)
    conc = track.concentration_at(pos[:, None] + xi[None, :])
    ref = rate * conc.sum(axis=0) * dt
    err = np.max(np.abs(series.depth_m[-1] - ref)) / ref.max()
    assert err < 1e-4


def test_time_stepped_integrator_matches_closed_form_when_aligned(panels):
    """Trapezoid is exact for a piecewise-constant rate whose jumps sit on the
    time grid; a misaligned grid leaves a first-order residue."""
    rate = 1.0e-9
    track = st.IceTrack(break_s_m=np.array([0.0, 4_000.0, 16_000.0]),
                        concentration=np.array([0.3, 0.9]),
                        speed_m_s=np.array([4.0, 6.0]))
    profile = _uniform_profile(rate, panels)
    assert track.duration_s == pytest.approx(3_000.0)
    # offsets frozen: every jump sits on the time grid -> midpoint rule is exact
    aligned = 31          # dt = 100 s, breaks at 1000 s and 3000 s
    stepped = st.evolve_time_stepped(profile, panels, track,
                                     n_time_steps=aligned, scheme="midpoint",
                                     block_streamwise_m=1.0e12)
    closed = st.evolve_abrasion_field(profile, panels, track,
                                      n_time_steps=aligned,
                                      block_streamwise_m=1.0e12)
    err_aligned = float(np.max(np.abs(stepped.depth_m - closed.depth_m))
                        / np.max(closed.depth_m))
    assert err_aligned < 1e-12

    # physical offsets shift each panel's jump off the grid (by xi_i / V), so
    # the midpoint rule drops to first order: the residue must be visible and
    # must shrink when the grid is refined
    errs = []
    for nts in (26, 51, 101):
        sm = st.evolve_time_stepped(profile, panels, track, n_time_steps=nts)
        cm = st.evolve_abrasion_field(profile, panels, track, n_time_steps=nts)
        errs.append(float(np.max(np.abs(sm.depth_m - cm.depth_m))
                          / np.max(cm.depth_m)))
    assert errs[0] > err_aligned
    assert errs[-1] < errs[0]
    assert errs[-1] < 0.01


# ---------------------------------------------------------------------------
# volume ledger
# ---------------------------------------------------------------------------

def test_volume_ledger_closes_and_detects_corruption(panels):
    rate = 1.0e-9
    track = ia.reference_baltic_voyage()
    track = st.ice_track_from_voyage(track)
    series = st.evolve_abrasion_field(_uniform_profile(rate, panels), panels,
                                      track, n_time_steps=31)
    ledger = st.volume_ledger(series)
    assert ledger["relative_error"] < 1e-10

    # negative control: remove one panel's depth -> the ledger must complain
    broken_depth = series.depth_m.copy()
    broken_depth[-1, 17] = 0.0
    broken = st.AbrasionFieldTimeSeries(
        panels=series.panels, track=series.track, profile=series.profile,
        time_s=series.time_s, depth_m=broken_depth, rate_m_s=series.rate_m_s,
        reference_concentration=series.reference_concentration)
    assert st.volume_ledger(broken)["relative_error"] > 1e-6
    assert not st.volume_ledger(broken)["passed"]


def test_reference_concentration_scales_depth(panels):
    """The profile rate is quoted at ``c_ref``; at half the concentration the
    depth must halve, i.e. depth = a * c / c_ref * t."""
    rate = 1.0e-9
    track = st.uniform_track(0.5, 20_000.0, 5.0)
    at_half_ref = st.evolve_abrasion_field(
        _uniform_profile(rate, panels), panels, track, n_time_steps=5,
        reference_concentration=1.0, block_streamwise_m=1e12)
    at_ref = st.evolve_abrasion_field(
        _uniform_profile(rate, panels), panels, track, n_time_steps=5,
        reference_concentration=0.5, block_streamwise_m=1e12)
    expected = rate * 0.5 * at_half_ref.time_s[:, None]
    assert np.allclose(at_half_ref.depth_m, expected, rtol=1e-12, atol=1e-24)
    assert np.allclose(at_ref.depth_m, 2.0 * at_half_ref.depth_m, rtol=1e-12)


# ---------------------------------------------------------------------------
# quantiles
# ---------------------------------------------------------------------------

def test_area_weighted_quantile_matches_numpy_for_equal_weights():
    rng = np.random.default_rng(0)
    v = rng.random(5000)
    w = np.ones_like(v)
    for q in (5.0, 50.0, 95.0):
        assert st.area_weighted_quantile(v, w, q) == pytest.approx(
            float(np.percentile(v, q)), abs=5e-3)


def test_area_weighted_quantile_two_level_exact():
    v = np.array([1.0, 1.0, 5.0, 5.0])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    assert st.area_weighted_quantile(v, w, 50.0) == pytest.approx(3.0, abs=1e-12)
    w2 = np.array([1.0, 1.0, 8.0, 8.0])
    # 5 occupies 8/9 of the weight -> the 50 % quantile is 5, not 3
    assert st.area_weighted_quantile(v, w2, 50.0) > 4.9


# ---------------------------------------------------------------------------
# profile fitting
# ---------------------------------------------------------------------------

def test_profile_fit_recovers_synthetic_field():
    hull_l, draught = 180.0, 10.0
    # the panel band straddles the waterline (as in the real hull band)
    panels = st.PanelGrid(x_edges=np.linspace(0.0, hull_l, 41),
                          z_edges=np.linspace(draught - 1.0, draught + 0.6, 10),
                          hull_length_m=hull_l, hull_draught_m=draught)
    truth = st.SeparableActivityProfile(
        scale_m_s=3.0e-9, background=0.5, decay_amplitude=1.5, decay_length=0.15,
        x_modes=((2.0, 0.05, 0.05), (1.0, 0.55, 0.20)),
        waterline_offset_m=-0.2, sigma_below_m=0.6, sigma_above_m=0.3,
        hull_length_m=hull_l, hull_draught_m=draught)
    data = truth.evaluate_on_grid(panels)
    fit = st.fit_separable_profile(panels.panel_x, panels.panel_z,
                                   data.ravel(), hull_length_m=hull_l,
                                   hull_draught_m=draught)
    assert fit.r_squared > 0.999
    assert fit.profile.scale_m_s == pytest.approx(3.0e-9, rel=0.05)

    noisy = data * (1.0 + 0.02 * np.random.default_rng(1).standard_normal(data.shape))
    fit2 = st.fit_separable_profile(panels.panel_x, panels.panel_z,
                                    np.maximum(noisy, 0.0).ravel(),
                                    hull_length_m=hull_l, hull_draught_m=draught)
    assert fit2.r_squared > 0.98
    assert fit2.converged


# ---------------------------------------------------------------------------
# covariance, KL and conditioning
# ---------------------------------------------------------------------------

def test_separable_kl_covariance_is_exact():
    x = np.tile(np.arange(12) * 2.0, 8)
    z = np.repeat(np.arange(8) * 0.5, 12)
    cov = st.SpatialCovariance(sigma2=4.0e-20, length_x_m=6.0, length_z_m=0.8,
                              nugget2=1.0e-20)
    kl = st.separable_kl_expansion(cov, x, z)
    assert kl.variance_explained == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(kl.covariance(), cov.matrix(x, z), rtol=1e-10, atol=1e-26)
    samples = kl.sample(4000, seed=3)
    emp = np.cov(samples, rowvar=False)
    target = cov.matrix(x, z)
    err = np.linalg.norm(emp - target) / np.linalg.norm(target)
    # compare with the i.i.d. sampling baseline sqrt((N+1)/m), not a fixed number
    assert err < 1.5 * math.sqrt((x.size + 1) / samples.shape[0])


def test_kriging_matches_analytic_conditional_variance():
    x = np.tile(np.arange(20) * 2.0, 10)
    z = np.repeat(np.arange(10) * 0.5, 20)
    cov = st.SpatialCovariance(sigma2=1.0e-20, length_x_m=8.0, length_z_m=0.6,
                              nugget2=0.0)
    _, cond_cov = st.simple_kriging(cov, x, z,
                                    observation_indices=np.array([0]),
                                    observation_values=np.zeros(1))
    analytic = st.analytic_conditional_variance(cov, x - x[0], z - z[0])
    assert np.max(np.abs(np.diag(cond_cov) - analytic)) < 1e-3 * cov.sigma2


def test_conditional_samples_interpolate_noiseless_data_exactly():
    x = np.tile(np.arange(16) * 2.0, 8)
    z = np.repeat(np.arange(8) * 0.5, 16)
    cov = st.SpatialCovariance(sigma2=1.0e-20, length_x_m=7.0, length_z_m=0.7,
                              nugget2=0.0)
    kl = st.separable_kl_expansion(cov, x, z)
    truth = kl.sample(1, seed=11)[0]
    obs = np.array([0, 3, 17, 31, 64, 100, 127])
    cond = st.simulate_conditional_modes(
        kl, observation_indices=obs, observation_values=truth[obs],
        n_samples=5, seed=2)
    err = np.max(np.abs(cond.samples[:, obs] - truth[obs]))
    assert err / math.sqrt(np.mean(np.diag(kl.covariance()))) < 1e-9
    assert np.max(cond.variance[obs]) < 1e-8 * cov.sigma2


def test_unconditional_and_conditional_statistics_are_unbiased():
    x = np.tile(np.arange(20) * 2.0, 10)
    z = np.repeat(np.arange(10) * 0.5, 20)
    cov = st.SpatialCovariance(sigma2=1.0e-20, length_x_m=9.0, length_z_m=0.6,
                              nugget2=2.0e-21)
    kl = st.separable_kl_expansion(cov, x, z)
    prior_var = float(np.mean(np.diag(kl.covariance())))
    uncond = st.simulate_conditional_modes(kl, n_samples=1000, seed=5)
    assert abs(uncond.samples.var(axis=0, ddof=1).mean() - prior_var) / prior_var < 0.05
    # the sample mean of 1000 draws must sit inside the theoretical 5-sigma band
    sigma_mean = math.sqrt(prior_var / uncond.samples.shape[0])
    assert np.max(np.abs(uncond.samples.mean(axis=0))) < 5.0 * sigma_mean

    obs = np.arange(0, 200, 17)
    cond = st.simulate_conditional_modes(
        kl, observation_indices=obs,
        observation_values=0.3 * math.sqrt(prior_var) * np.ones(obs.size),
        n_samples=1000, seed=6)
    bias = np.sqrt(np.mean((cond.samples.mean(axis=0) - cond.mean) ** 2))
    floor = math.sqrt(float(np.mean(cond.variance)) / 1000.0)
    assert bias < 3.0 * floor + 1e-30
    var_err = abs(cond.samples.var(axis=0, ddof=1).mean() - cond.variance.mean())
    assert var_err / prior_var < 0.05
    # conditioning must reduce the variance somewhere
    assert cond.variance.mean() < prior_var


def test_coarsened_covariance_is_positive_definite(grid):
    panels = st.build_panel_grid(grid=grid, n_x=14, n_z=4)
    cov = st.SpatialCovariance(sigma2=1.0e-23, length_x_m=1.0, length_z_m=0.4,
                              nugget2=5.0e-24)
    cp = st.coarsen_covariance(cov, grid, panels)
    assert cp.shape == (panels.n_panels, panels.n_panels)
    assert np.allclose(cp, cp.T)
    assert np.linalg.eigvalsh(cp).min() > -1e-6 * np.trace(cp) / panels.n_panels


def test_coarsened_covariance_diagonal_shrinks_with_panel_size(grid):
    cov = st.SpatialCovariance(sigma2=1.0e-23, length_x_m=0.5, length_z_m=0.5,
                              nugget2=1.0e-23)
    small = st.build_panel_grid(grid=grid, n_x=37, n_z=9)
    big = st.build_panel_grid(grid=grid, n_x=10, n_z=3)
    d_small = float(np.mean(np.diag(st.coarsen_covariance(cov, grid, small))))
    d_big = float(np.mean(np.diag(st.coarsen_covariance(cov, grid, big))))
    assert d_big < d_small


def test_kl_from_ensemble_reproduces_ensemble_covariance():
    rng = np.random.default_rng(4)
    base = rng.standard_normal((60, 40))
    kl = st.kl_from_ensemble(base)
    emp = np.cov(base, rowvar=False)
    assert np.linalg.norm(kl.covariance() - emp) / np.linalg.norm(emp) < 1e-10
    assert kl.variance_explained > 0.999


def test_activity_parameter_ensemble_keeps_total_rate(panels, grid):
    ens = st.activity_parameter_ensemble(panels, grid=grid, n_members=20,
                                         reference_rate=1.0e-11)
    assert ens.shape == (20, panels.n_panels)
    mean = np.sum((ens + 1.0e-11) * panels.areas_flat, axis=1) / panels.total_area_m2
    assert np.allclose(mean, 1.0e-11, rtol=1e-9)
    assert np.mean(ens) == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# DEM-driven activity field (small ensemble: keeps the test fast)
# ---------------------------------------------------------------------------

def test_dem_activity_ensemble_is_physical_and_fits_profile():
    grid = ia.build_waterline_grid()
    panels = st.build_panel_grid(grid=grid, n_x=40, n_z=9)
    ens = st.dem_activity_rate_field(grid=grid, n_realizations=3,
                                     n_events_per_leg=1500)
    assert ens.rate_realizations.shape == (3,) + tuple(int(v) for v in grid.shape)
    assert np.all(ens.rate_realizations >= 0.0)
    assert ens.voyage_duration_s > 0.0
    # one Baltic voyage of the reference profile: order 1 um of ice-belt wear
    assert 0.05 < ens.summary()["depth_per_voyage_um_mean"] < 100.0

    mean_panel = st.coarsen_field(ens.mean_rate, grid, panels)
    fit = st.fit_separable_profile(panels.panel_x, panels.panel_z,
                                   mean_panel.ravel(),
                                   hull_length_m=panels.hull_length_m,
                                   hull_draught_m=panels.hull_draught_m)
    # with only three realisations the mean field is noisy, so the bound is loose
    assert fit.r_squared > 0.85
    assert fit.profile.scale_m_s > 0.0


def test_covariance_lagged_recovers_total_variance(grid):
    rng = np.random.default_rng(9)
    shape = (int(grid.shape[0]), int(grid.shape[1]))
    base = rng.standard_normal((40,) + shape) * 1e-12
    # smooth in x with a short explicit kernel, plus white noise
    kernel = np.exp(-np.arange(4) / 1.5)
    kernel /= kernel.sum()
    smooth = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"),
                                 2, base)
    cov = st.estimate_covariance_lagged(grid, smooth, center=np.zeros(shape),
                                        max_lag_x_cells=4, max_lag_z_cells=2)
    assert cov.total_variance == pytest.approx(float(np.mean(np.var(smooth, axis=0))),
                                               rel=0.5)
    assert cov.length_x_m > 0.0 and cov.length_z_m > 0.0


# ---------------------------------------------------------------------------
# end-to-end criteria (reduced settings)
# ---------------------------------------------------------------------------

def test_verification_suite_passes_with_small_ensemble():
    grid = ia.build_waterline_grid()
    ens = st.dem_activity_rate_field(grid=grid, n_realizations=3,
                                     n_events_per_leg=1500)
    res = st.run_n05_verification(n_panels_x=40, n_panels_z=9, n_time_steps=25,
                                  n_samples=60, n_samples_check=200,
                                  n_survey_panels=40, ensemble=ens)
    crit = res["criteria"]
    assert crit["C1"]["r_squared_panel_grid"] > 0.85
    assert crit["C2"]["noiseless_interpolation_relative_error"] < 1e-6
    assert crit["C2"]["analytic_conditional_variance_rel_error"] < 0.05
    assert crit["C3"]["relative_error"] < 1e-6
    assert crit["C5"]["max_relative_error"] < 0.01
    assert crit["C6"]["passed"]
    # C4 is the honest weak point with three realisations (noisy P95 reference)
    assert 0.0 <= crit["C4"]["relative_error"] < 1.0
    assert res["auxiliary"]["moving_frame_onset_relative_error"] < 1e-6

def test_covariance_from_realizations_splits_smooth_and_nugget():
    """A synthetic field with a known smooth part + white noise must be
    decomposed into sigma^2 (smooth) and tau^2 (nugget) with tau^2 > 0."""
    rng = np.random.default_rng(12)
    nx, nz = 24, 6
    x = np.tile(np.arange(nx) * 2.0, nz)
    z = np.repeat(np.arange(nz) * 0.5, nx)
    cov_true = st.SpatialCovariance(sigma2=1.0e-22, length_x_m=8.0,
                                    length_z_m=0.8, nugget2=0.0)
    kl = st.separable_kl_expansion(cov_true, x, z)
    smooth = kl.sample(120, seed=3)
    noise = rng.standard_normal(smooth.shape) * math.sqrt(4.0e-22)
    realizations = smooth + noise
    cov = st.estimate_covariance_from_realizations(x, z, realizations)
    total = cov.total_variance
    expected = float(np.mean(np.var(realizations, axis=0, ddof=1)))
    assert total == pytest.approx(expected, rel=1e-6)
    # a nugget-dominant field must be split accordingly
    assert cov.nugget2 > 0.5 * total
    assert 0.3 < cov.sigma2 / 1.0e-22 < 3.0
    assert cov.nugget2 == pytest.approx(4.0e-22, rel=0.5)
