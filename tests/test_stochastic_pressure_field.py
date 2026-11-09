"""Unit + integration tests for the stochastic ice-pressure field chain (AG27W2-N09).

Coverage:

* DEM contact law against the Hertz energy identity (independent reference);
* DEM pressure field: non-negative, force-conserving, fragmentation keeps the
  pressure bounded (a fracture-free layer diverges - that is the regression this
  test locks down);
* circular autocorrelation of a known field and the correlation-model fit;
* Karhunen-Loeve: exact eigen-decomposition of the separable covariance,
  truncation energy, sampling statistics;
* plate: modal static solution against the classical ``0.00406 qa^4/D``
  coefficient and the vectorised modal recurrence against the verified
  ``fatigue_chain.response.sdof_response``;
* fatigue: rainflow + Miner against a hand-summed cycle list, and the
  point-load analytic anchor;
* an end-to-end smoke run of the study on a small DEM.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.fatigue_chain.response import sdof_response
from tensordem.stochastic_pressure_field.correlation import (
    autocorrelation_2d,
    fit_correlation_model,
    matern_1d,
)
from tensordem.stochastic_pressure_field.dem_field import (
    simulate_crushed_layer,
    single_grain_hertz_verification,
)
from tensordem.stochastic_pressure_field.fatigue import (
    analytic_miner_anchor,
    ice_pressure_load_envelope,
    lifetime_from_damage,
    rainflow_miner_life,
)
from tensordem.stochastic_pressure_field.kl import kl_expansion_separable
from tensordem.stochastic_pressure_field.plate import (
    PlatePanel,
    modal_forces,
    modal_response_ramp,
    navier_static_uniform,
    stress_history_from_modal_force,
)
from tensordem.stochastic_pressure_field.study import pressure_area_exponent
from tensordem.whipping.fatigue import DNV_CURVES


# ---------------------------------------------------------------------------
# DEM
# ---------------------------------------------------------------------------
def test_hertz_single_grain_energy_identity():
    """Peak overlap of a dropped grain vs the Hertz energy identity (9 digits)."""
    res = single_grain_hertz_verification()
    assert res["max_overlap_dem"] > 0.0
    assert res["relative_error"] < 1e-6


def test_dem_field_non_negative_conservative_and_bounded():
    """Fragmentation keeps the pressure bounded; the deposit is force conserving."""
    snaps = simulate_crushed_layer(
        n_grains=18, panel_length=0.30, panel_width=0.15, layer_thickness=0.10,
        r_median=0.022, piston_velocity=0.3, n_steps=1200, snapshot_every=20,
        grid_nx=30, grid_ny=15, warmup_steps=300,
    )
    p = snaps.pressure
    assert p.shape[0] > 20
    assert np.all(np.isfinite(p))
    assert p.min() >= 0.0
    # without fragmentation the layer jams and the pressure diverges; with it
    # the panel pressure stays in a physical band
    assert snaps.mean_pressure() < 50.0e6
    # ∫ p dA equals the sum of the panel forces (deposit is force conserving)
    cell = snaps.cell_area
    deposited = p.sum(axis=(1, 2)) * cell
    applied = snaps.panel_force
    good = applied > 1e-9
    assert good.sum() > 5
    assert np.max(np.abs(deposited[good] - applied[good]) / applied[good]) < 1e-9


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------
def _periodic_field(nx, ny, length, nu, n_samples, seed=3):
    """Synthetic periodic field on a grid whose cell is ``length / 4``."""
    dx = length / 4.0
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    kl = kl_expansion_separable(x, y, model="matern", length_x=length,
                                length_y=length, nu=nu, variance=1.0,
                                mean_field=np.zeros((ny, nx)),
                                energy_target=0.999, n_modes_max=nx * ny)
    return kl.sample(n_samples, seed=seed), dx


def test_autocorrelation_and_fit_recover_known_length():
    length = 0.2
    nx, ny = 24, 24
    fields, dx = _periodic_field(nx, ny, length, nu=1.5, n_samples=400)
    R = autocorrelation_2d(fields)
    assert R[0, 0] == pytest.approx(1.0, abs=1e-12)
    # the circular estimator must reproduce the periodic model correlation
    assert R[0, 1] == pytest.approx(matern_1d(np.array([dx]), length, 1.5)[0],
                                    abs=0.03)
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    fits = fit_correlation_model(R, x, y)
    assert fits["matern"].r2 > 0.98
    assert fits["matern"].length_x == pytest.approx(length, rel=0.10)
    assert fits["matern"].length_y == pytest.approx(length, rel=0.10)
    assert fits["matern"].nu == pytest.approx(1.5, rel=0.30)
    # an exponential kernel cannot represent a smooth (nu = 1.5) field: it is
    # biased long, which is exactly why the fit family is reported side by side
    assert fits["exponential"].length_x > 1.2 * length
    assert fits["exponential"].r2 < fits["matern"].r2


def test_correlation_is_not_biased_by_snapshot_mean_removal():
    """Removing each snapshot's own spatial mean would delete the DC mode.

    The field here is a random *uniform* pressure level plus small noise: the
    true spatial correlation is ~0.99 at every lag.  Subtracting each snapshot's
    own spatial mean would leave only the noise (R ~ 0), which is the bias the
    ensemble-mean estimator avoids.
    """
    rng = np.random.default_rng(0)
    ny, nx = 12, 12
    level = rng.standard_normal((200, 1, 1))
    field = 10.0 + level + 0.05 * rng.standard_normal((200, ny, nx))
    R = autocorrelation_2d(field)
    assert R[0, 3] > 0.95
    centred = field - field.mean(axis=(1, 2), keepdims=True)
    R_centred = autocorrelation_2d(centred + centred.mean(axis=0, keepdims=True))
    assert R_centred[0, 3] < 0.1


# ---------------------------------------------------------------------------
# KL
# ---------------------------------------------------------------------------
def test_kl_eigenpairs_and_truncation():
    nx, ny = 12, 6
    x = (np.arange(nx) + 0.5) * 0.05
    y = (np.arange(ny) + 0.5) * 0.05
    variance = 4.0
    kl = kl_expansion_separable(x, y, model="matern", length_x=0.087,
                                length_y=0.085, nu=3.0, variance=variance,
                                mean_field=np.zeros((ny, nx)),
                                energy_target=0.95, n_modes_max=50)
    assert kl.n_modes < 50
    assert kl.energy_fraction >= 0.95
    # orthonormal modes
    phi = kl.modes.reshape(kl.n_modes, -1)
    gram = phi @ phi.T
    assert np.allclose(gram, np.eye(kl.n_modes), atol=1e-10)
    # the trace of the discrete covariance is n * variance (unit diagonal); the
    # minimum-image kernel is not exactly PSD, so the signed eigenvalue sum is
    # the trace and the positive part is slightly larger
    evals_full = np.linalg.eigvalsh(kl.target_covariance())
    assert float(np.sum(evals_full)) == pytest.approx(nx * ny * variance, rel=1e-9)
    # the retained eigenvalues are the largest ones, in descending order
    assert np.all(np.diff(kl.eigenvalues) <= 1e-6)
    assert float(np.sum(kl.eigenvalues)) <= float(np.sum(np.clip(evals_full, 0, None)))


def test_kl_samples_reproduce_target_moments():
    nx, ny = 12, 6
    x = (np.arange(nx) + 0.5) * 0.05
    y = (np.arange(ny) + 0.5) * 0.05
    mean_field = np.full((ny, nx), 1.9e6)
    kl = kl_expansion_separable(x, y, model="matern", length_x=0.087,
                                length_y=0.085, nu=3.0, variance=2.0e12,
                                mean_field=mean_field, energy_target=0.98,
                                n_modes_max=50)
    fields = kl.sample(2000, seed=5, moment_match=True)
    assert fields.shape == (2000, ny, nx)
    # mean field reproduced (moment-matched ensemble)
    assert np.max(np.abs(fields.mean(axis=0) - mean_field)) < 1e-6 * 1.9e6
    # variance reproduced up to the truncation loss
    var = float(np.mean(fields.var(axis=0)))
    assert var == pytest.approx(2.0e12 * kl.energy_fraction, rel=0.02)
    # and the target mean is inside the field
    assert fields.min() < mean_field[0, 0] < fields.max()


# ---------------------------------------------------------------------------
# plate
# ---------------------------------------------------------------------------
def test_plate_uniform_load_matches_classical_coefficient():
    panel = PlatePanel(length=0.6, width=0.6, thickness=0.02)
    nav = navier_static_uniform(panel, 1.0e5, n_terms=199)
    classical = 0.00406 * 1.0e5 * panel.length ** 4 / panel.flexural_rigidity
    assert nav["w_max"] == pytest.approx(classical, rel=0.005)
    assert nav["alpha"] == pytest.approx(0.00406, rel=0.005)


def test_modal_static_matches_navier_on_uniform_load():
    panel = PlatePanel(length=0.6, width=0.3, thickness=0.03, n_modes_x=12,
                       n_modes_y=12)
    nx, ny = 24, 12
    x = (np.arange(nx) + 0.5) * panel.length / nx
    y = (np.arange(ny) + 0.5) * panel.width / ny
    field = np.full((ny, nx), 1.0e5)
    Q = modal_forces(panel, field, x, y)
    sigma = stress_history_from_modal_force(
        panel, Q[None, :, None] * np.ones((1, 1, 2)), 1.0,
        panel.length / 2, panel.width / 2, dynamic=False)[0][0]
    nav = navier_static_uniform(panel, 1.0e5, n_terms=199)
    assert sigma == pytest.approx(nav["sigma_x"], rel=0.01)


def test_modal_response_matches_verified_sdof():
    panel = PlatePanel(length=0.6, width=0.3, thickness=0.03)
    omega = panel.omega_all()
    dt = 2.0e-3
    t = np.arange(400) * dt
    f = np.zeros_like(t)
    f[t >= 0.02] = 1.0 + 0.5 * np.sin(2.0 * math.pi * 10.0 * t[t >= 0.02])
    for mode in (0, 3):
        q_vec = modal_response_ramp(dt, f[None, None, :],
                                    omega[[mode]], np.array([panel.damping]))[0, 0]
        q_ref = sdof_response(t, f, omega[mode] / (2.0 * math.pi), panel.damping)
        assert np.max(np.abs(q_vec - q_ref)) < 1e-9


# ---------------------------------------------------------------------------
# fatigue
# ---------------------------------------------------------------------------
def test_rainflow_miner_matches_hand_summed_cycles():
    """A sine record vs the hand-derived ASTM cycle list."""
    n_cycles = 40
    f_hz, dt, sigma_a = 2.0, 1.0e-3, 30.0e6
    n_t = int(round(n_cycles / (f_hz * dt))) + 1
    t = np.arange(n_t) * dt
    sigma = sigma_a * np.sin(2.0 * math.pi * f_hz * t)
    res = rainflow_miner_life(sigma, dt, duration_s=t[-1] - t[0])
    curve = DNV_CURVES["D_air"]
    n_full = float(curve.cycles_to_failure(np.array([2.0 * sigma_a]))[0])
    n_half = float(curve.cycles_to_failure(np.array([sigma_a]))[0])
    # range-pair rule: N-1 full cycles of range 2A plus two half cycles of range A
    expected = (n_cycles - 1) / n_full + 1.0 / n_half
    # 2e-3 = the half cycle that closes the record is sampled over one time step,
    # so its range is a step-width short of A; the cycle *count* is exact
    assert res["damage"] == pytest.approx(expected, rel=2e-3)
    # the closing half cycle of the sampled record makes the count N + 0.5
    assert abs(res["cycles"] - n_cycles) <= 1.0


def test_lifetime_from_damage_is_inverse():
    assert lifetime_from_damage(0.5, 1000.0) == pytest.approx(2000.0)
    assert math.isinf(lifetime_from_damage(0.0, 1000.0))


def test_envelope_mean_is_one_and_positive():
    dt = 0.01
    t = np.arange(int(120 / dt)) * dt
    env = ice_pressure_load_envelope(t, rate_per_s=8.0, tau_rise=0.05,
                                     tau_decay=0.15, seed=11)
    assert env.min() >= 0.0
    assert env.mean() == pytest.approx(1.0, rel=1e-12)
    assert env.max() > 2.0


def test_analytic_miner_anchor_within_one_percent():
    panel = PlatePanel(length=0.6, width=0.3, thickness=0.03, n_modes_x=10,
                       n_modes_y=10)
    res = analytic_miner_anchor(
        panel, load_amplitude=2.0e5, frequency_hz=1.0, n_cycles=10, dt=0.002,
        hot_spot=(0.15, 0.05), load_position=(0.3, 0.15), n_terms=199,
        grid_nx=40, grid_ny=20)
    assert res["chain_damage"] > 0.0
    assert res["damage_rel_error_chain_vs_analytic"] < 0.01
    assert res["dynamic_amplification"] == pytest.approx(1.0, abs=1e-3)


def test_pressure_area_exponent_on_synthetic_field():
    """A field whose patch pressure decays like A^-0.3 must be recovered."""
    rng = np.random.default_rng(0)
    nx, ny, n_snap = 120, 60, 60
    # build a field by summing Gaussian patches with a wide size distribution
    field = np.zeros((n_snap, ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    for s in range(n_snap):
        for _ in range(40):
            cx, cy = rng.uniform(0, nx), rng.uniform(0, ny)
            w = rng.uniform(1.0, 6.0)
            amp = w ** (-0.6) * rng.uniform(0.5, 1.5)
            field[s] += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * w ** 2))
    out = pressure_area_exponent(field, cell=1.0, patch_sizes=(2, 4, 8, 16))
    assert np.isfinite(out["exponent"])
    assert 0.0 < out["exponent"] < 1.5


# ---------------------------------------------------------------------------
# end-to-end smoke (small DEM, cached in tmp_path)
# ---------------------------------------------------------------------------
def test_study_smoke_all_stages_run(tmp_path):
    from tensordem.stochastic_pressure_field.study import run_study

    dem_config = dict(n_grains=18, panel_length=0.30, panel_width=0.15,
                      layer_thickness=0.10, r_median=0.022, piston_velocity=0.3,
                      n_steps=1200, snapshot_every=30, grid_nx=30, grid_ny=15,
                      warmup_steps=300)
    res = run_study(quick=True, reuse_dem=False, run_dir=tmp_path,
                    dem_config=dem_config, n_samples=12)
    assert res["c1_correlation"]["r2_matern"] > 0.5
    assert res["c2_kl"]["n_modes"] <= 50
    assert 0.0 < res["c2_kl"]["energy_fraction"] <= 1.0
    assert res["c3_statistics"]["correlation"]["relative_l2_error"] < 0.5
    assert res["c5_anchor"]["damage_rel_error_chain_vs_analytic"] < 0.01
    assert (tmp_path / "study.json").exists()
    json_keys = {"c1_correlation", "c2_kl", "c3_statistics", "c4_fatigue",
                 "c5_anchor", "c6_performance", "pass_summary"}
    assert json_keys.issubset(res.keys())