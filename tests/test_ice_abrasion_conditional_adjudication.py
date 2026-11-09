"""Tests for :mod:`tensordem.ice_abrasion_conditional_adjudication`.

The synthetic fields here are pure numerics: a separable activity profile times a
*stationary* anisotropic Gaussian residual.  The physical DEM channel is exercised
by ``scripts/ag27w2_n06_verification.py`` and by the last test (a two-realisation
smoke test on the real wear-season solver).

Stationarity of the fixture matters: a causal AR(1) filter applied on a finite
grid has a variance that grows from the corner, so its draws are not from the
process the simulator models and any variance comparison would be meaningless.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tensordem import ice_abrasion_conditional_adjudication as adj
from tensordem import ice_abrasion_conditional_sim as legacy

NX, NZ = 64, 12
LENGTH_X, DEPTH_Z = 185.0, 11.5


def _profile(nx: int = NX, nz: int = NZ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, LENGTH_X, nx)
    z = np.linspace(0.0, DEPTH_Z, nz)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    a = (1.0e-8 * np.exp(-((xx - 150.0) / 30.0) ** 2)
         * np.exp(-np.abs(zz - 8.0) / 3.0) + 1.0e-10)
    return x, z, a


def _correlated_residual(nx: int, nz: int, seed: int,
                         len_x_cells: float = 2.0, len_z_cells: float = 1.0) -> np.ndarray:
    """Stationary anisotropic Gaussian field, synthesised in the spectral domain."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((nx, nz))
    kx = 2.0 * np.pi * np.fft.fftfreq(nx)[:, None]
    kz = 2.0 * np.pi * np.fft.fftfreq(nz)[None, :]
    transfer = 1.0 / np.sqrt(1.0 + (kx * len_x_cells) ** 2 + (kz * len_z_cells) ** 2)
    field = np.fft.ifft2(np.fft.fft2(white) * transfer).real
    field -= field.mean()
    return field / max(np.std(field), 1e-300)


def _field(seed: int, nx: int = NX, nz: int = NZ, sigma: float = 0.30) -> np.ndarray:
    _, _, a = _profile(nx, nz)
    r = _correlated_residual(nx, nz, seed)
    return a * (1.0 + sigma * r)


def _block_average(field: np.ndarray, fx: int, fz: int) -> np.ndarray:
    nx, nz = field.shape
    nx2, nz2 = nx // fx, nz // fz
    f = field[:nx2 * fx, :nz2 * fz]
    return f.reshape(nx2, fx, nz2, fz).mean(axis=(1, 3))


@pytest.fixture(scope="module")
def fields():
    truth = np.array([_field(1000 + k) for k in range(20)])
    source = _field(999)
    _, _, a = _profile()
    return truth, source, a


# ---------------------------------------------------------------------------
# 1. Percentile and decomposition
# ---------------------------------------------------------------------------

def test_weighted_percentile_uniform_matches_numpy():
    """Uniform weights -> the empirical P95 order statistic.

    The weighted estimator is a step function (it returns an observed value), so it
    agrees with ``np.percentile`` to within one sample quantum, not exactly.
    """
    rng = np.random.default_rng(0)
    v = rng.standard_normal(500)
    est = adj.weighted_percentile(v, np.ones_like(v), 95.0)
    ref = np.percentile(v, 95.0)
    quantum = np.max(np.diff(np.sort(v)))
    assert abs(est - ref) <= quantum
    assert est in set(np.sort(v).tolist())


def test_weighted_percentile_respects_weights():
    v = np.array([0.0, 1.0, 10.0])
    w = np.array([1.0, 1.0, 8.0])
    assert adj.weighted_percentile(v, w, 95.0) == pytest.approx(10.0)


def test_multiplicative_decomposition_reconstructs(fields):
    truth, source, a = fields
    d = adj.decompose_multiplicative(source, a)
    assert np.allclose(d.profile_values * d.residual, source, rtol=1e-12, atol=0.0)
    assert d.residual.mean() == pytest.approx(1.0, rel=1e-9)
    assert 0.0 < d.r_squared <= 1.0


# ---------------------------------------------------------------------------
# 2. Spectral simulation
# ---------------------------------------------------------------------------

def test_phase_randomisation_preserves_spectrum_and_mean():
    rng = np.random.default_rng(3)
    r = _correlated_residual(NX, NZ, 7)
    out = adj.phase_randomize_2d(r, rng)
    assert out.shape == r.shape and np.isrealobj(out)
    f_src = np.abs(np.fft.rfft2(r - r.mean()))
    f_out = np.abs(np.fft.rfft2(out - out.mean()))
    assert np.allclose(f_src, f_out, rtol=1e-10, atol=1e-12)
    assert out.mean() == pytest.approx(r.mean(), rel=1e-12)


def test_gaussian_synthesis_preserves_spectrum_in_expectation(fields):
    _, source, a = fields
    d = adj.decompose_multiplicative(source, a)
    psd = adj.periodogram_psd(d.residual)
    rng = np.random.default_rng(0)
    draws = np.array([adj.synthesize_gaussian_realization(psd, rng) for _ in range(150)])
    assert draws.var(axis=(1, 2)).mean() == pytest.approx(d.residual.var(), rel=0.05)
    fid = adj.autocorrelation_fidelity(d.residual, draws, max_lag=3)
    assert fid["lag_deviation_pct"] < 5.0
    assert fid["energy_error_pct"] < 2.0


def test_row_means_are_preserved(fields):
    _, source, a = fields
    sims = adj.simulate_abrasion_realizations(source, a, n_realizations=8, seed=11)
    src_row = source.mean(axis=0)
    for s in sims:
        np.testing.assert_allclose(s.mean(axis=0), src_row, rtol=1e-12, atol=0.0)
    assert sims.shape == (8,) + source.shape


def test_row_moment_modes_trade_conditioning_against_dispersion(fields):
    """Pinning every realisation's row moments conditions away part of the scatter."""
    truth, source, a = fields
    ref = adj.p95_variance_truth(truth)
    pinned = adj.p95_variance_conditional(source, a, n_realizations=120, seed=4,
                                          row_moments="mean")
    loose = adj.p95_variance_conditional(source, a, n_realizations=120, seed=4,
                                         row_moments="mean_ensemble")
    assert 0.7 <= pinned.p95_variance / ref.p95_variance <= 1.4
    assert loose.p95_variance >= pinned.p95_variance


# ---------------------------------------------------------------------------
# 3. Variance estimators vs the independent-repeat referee
# ---------------------------------------------------------------------------

def test_conditional_variance_matches_independent_repeats(fields):
    truth, source, a = fields
    ref = adj.p95_variance_truth(truth)
    cond = adj.p95_variance_conditional(source, a, n_realizations=200, seed=4,
                                        noise="stationary")
    ratio = cond.p95_variance / ref.p95_variance
    assert 0.7 <= ratio <= 1.4, ratio


def test_iid_bootstrap_is_blind_to_spatial_correlation(fields):
    """The i.i.d. bootstrap only sees the marginal distribution.

    Shuffling a field in space leaves its marginal untouched, so the i.i.d.
    bootstrap reports the same P95 variance for a correlated field and its shuffled
    twin.  That blindness is why its error bars could not be trusted.
    """
    _, source, _ = fields
    rng = np.random.default_rng(7)
    shuffled = rng.permutation(source.ravel()).reshape(source.shape)
    boot_corr = adj.p95_variance_bootstrap_iid(source, n_bootstrap=120, seed=4)
    boot_shuf = adj.p95_variance_bootstrap_iid(shuffled, n_bootstrap=120, seed=4)
    assert 0.8 <= boot_corr.p95_variance / boot_shuf.p95_variance <= 1.25


def test_block_bootstrap_reports_finite_estimates(fields):
    truth, source, _ = fields
    ref = adj.p95_variance_truth(truth)
    mb = adj.p95_variance_bootstrap_moving_block(source, block=(8, 3),
                                                 n_bootstrap=60, seed=4)
    assert np.isfinite(mb.p95_variance) and mb.p95_variance > 0.0
    assert ref.p95_variance > 0.0


def test_white_noise_control_not_worse_than_bootstrap():
    rng = np.random.default_rng(12)
    wn = np.abs(rng.normal(loc=1e-8, scale=3e-9, size=(NX, NZ)))
    _, _, a = _profile()
    cond = adj.p95_variance_conditional(wn, a, n_realizations=150, seed=2)
    boot = adj.p95_variance_bootstrap_iid(wn, n_bootstrap=150, seed=2)
    assert cond.p95_variance / boot.p95_variance >= 0.8


# ---------------------------------------------------------------------------
# 4. Resolution decomposition
# ---------------------------------------------------------------------------

def _coarse_pair(truth, source, blocks=(4, 3)):
    return (np.array([_block_average(f, *blocks) for f in truth]),
            _block_average(source, *blocks))


def test_resolution_decomposition_closes(fields):
    """The two physical terms must add up to the measured difference.

    The sampling term is estimated from held-out conditioning fields, so its
    Monte-Carlo noise shrinks with their number: with 4 fields the closure error is
    ~35 %, with 20 it is 8 %.  The physical 185x9 case in the driver closes at 9.9 %
    with 10 fields because its residual is far less correlated (more independent
    eddies).
    """
    truth, source, a = fields
    coarse_truth, coarse_src = _coarse_pair(truth, source)
    src_stack = np.array([_field(3000 + k) for k in range(20)])
    coarse_stack = np.array([_block_average(f, 4, 3) for f in src_stack])
    brk = adj.resolution_decomposition(
        truth, coarse_truth, src_stack, coarse_stack, n_realizations=150, seed=6,
        profile_fine=a, profile_coarse=_block_average(a, 4, 3),
        noise="stationary")
    assert brk.closure_error_pct < 10.0
    assert abs(brk.discretization_pct) > 0.5 * abs(brk.total_measured_pct)
    assert brk.share_discretization + brk.share_sampling == pytest.approx(1.0)
    # the interaction term is the closure residual, not an absorbed fudge factor
    assert brk.interaction_pct == pytest.approx(
        brk.total_measured_pct - brk.discretization_pct - brk.sampling_pct, rel=1e-12)


def test_resolution_decomposition_does_not_hide_failure():
    """A deliberately wrong profile must show up as a closure failure."""
    rng = np.random.default_rng(0)
    tf = np.array([np.abs(rng.normal(1.0, 0.1, (NX, NZ))) for _ in range(12)])
    tc = np.array([np.abs(rng.normal(1.0, 0.1, (NX // 4, NZ // 3))) for _ in range(12)])
    sf = tf[0] * 3.0
    sc = _block_average(tf[0] * 3.0, 4, 3)
    brk = adj.resolution_decomposition(tf, tc, sf, sc, n_realizations=50, seed=1,
                                       noise="stationary")
    assert np.isfinite(brk.closure_error_pct)
    assert brk.interaction_pct != pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. Reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility(fields):
    _, source, a = fields
    rep = adj.reproducibility_check(source, a, n_realizations=20, seed=8, n_seeds=4)
    assert rep["bitwise_identical"]
    assert rep["cross_seed_cv_pct"] < 5.0


def test_field_hash_is_stable_and_content_sensitive():
    rng = np.random.default_rng(1)
    f = rng.standard_normal((6, 4))
    assert adj.field_hash(f) == adj.field_hash(f.copy())
    assert adj.field_hash(f) != adj.field_hash(f * 2.0)
    assert adj.field_hash(f) == hashlib.sha256(
        np.ascontiguousarray(f, dtype="<f8").tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# 6. Regression guards: why the legacy criteria could not fail
# ---------------------------------------------------------------------------

def test_legacy_truth_channel_is_circular(fields):
    """``truth_channel_p95`` defines truth with the simulator it validates.

    Its "truth" repeats are calls of the same ``conditional_simulator`` with other
    seeds, so the C2 ratio it reports is a self-consistency ratio.  The guard pins
    that behaviour so the physical channel is never silently replaced by it again.
    """
    _, source, _ = fields
    x = np.linspace(0.0, LENGTH_X, source.shape[0])
    z = np.linspace(0.0, DEPTH_Z, source.shape[1])
    res = legacy.truth_channel_p95(x, z, source, n_realizations=40,
                                   n_truth_repeats=3, base_seed=5)
    # the legacy ratio is not even self-consistent: its "truth" variance (spread of
    # the mean P95 over simulator seeds) and its "internal" variance (split-half
    # differences) measure different things, so the guard only pins finiteness.
    assert np.isfinite(res.variance_ratio) and res.variance_ratio > 0.0


def test_legacy_resolution_closure_is_an_identity(fields):
    """Legacy C4 closure is zero by construction (sampling := total - disc)."""
    _, source, _ = fields
    x = np.linspace(0.0, LENGTH_X, source.shape[0])
    z = np.linspace(0.0, DEPTH_Z, source.shape[1])
    res = legacy.decompose_resolution_error(x, z, source, n_realizations=40, seed=3)
    assert res.sampling_error_pct == pytest.approx(
        max(0.0, res.total_error_pct - res.discretization_error_pct), rel=1e-12)
    assert res.closure_error_pct == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 7. Physical smoke test (real DEM wear season, coarse event count)
# ---------------------------------------------------------------------------

def test_physical_repeat_variance_is_positive():
    from tensordem import ice_abrasion as ia
    from tensordem import ice_abrasion_spatiotemporal as st

    grid = ia.build_waterline_grid()
    ens = st.dem_activity_rate_field(grid=grid, voyage=ia.reference_baltic_voyage(),
                                     n_realizations=2, n_events_per_leg=500)
    panels = st.build_panel_grid(grid=grid, n_x=92, n_z=5)
    # ``coarsen_field`` returns [z, x]; the adjudication module indexes [x, z]
    fields = np.array([st.coarsen_field(r, grid, panels).T for r in ens.rate_realizations])
    est = adj.p95_variance_truth(fields)
    assert fields.shape == (2, 92, 5)
    assert np.isfinite(est.p95_mean) and est.p95_mean > 0.0
    a, _, _ = adj.fit_profile_field(*adj._grid_axes(fields[0].shape), fields[0])
    cond = adj.p95_variance_conditional(fields[0], a, n_realizations=40, seed=1,
                                        noise="local_scale")
    assert np.isfinite(cond.p95_variance) and cond.p95_variance > 0.0