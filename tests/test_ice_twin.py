"""Tests for the ice-zone structural-health digital twin (``tensordem.ice_twin``).

Campaign node IN39-N10 acceptance criteria covered here:

* (a) modal-filtering analytic anchor: single-mode sinusoid, < 1 % at SNR > 20 dB,
* (b) load inversion: known ice load -> forward strain -> inverted load,
      peak error < 15 %, energy error < 10 %,
* (c) rigid/elastic separation at 0.5 Hz (energy shares),
* (d) fatigue: 1000 ice-load cycles, incremental vs single-pass rainflow < 10 %,
* (e) real-time: per-window latency < 50 ms at 100 Hz,
* (f) DEM comparison: correlation > 0.7 on >= 5 ice events,
* (g) negative controls (zero load -> zero response) and an SNR sweep 10..30 dB.

The p25/p29 regression lock (``tests/test_whipping_response.py``) is a
separate file and must keep passing untouched.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ice_twin import (FreeFreeBeamROM, IceTwin, IncrementalFatigue,
                                ModalFilter, compare_loads, correction_needed,
                                differentiate, energy_error,
                                invert_modal_force, peak_error,
                                reconstruct_total_force, rigid_elastic_split)
from tensordem.ice_twin.fatigue_update import turning_points_indexed
from tensordem.whipping.dem_load import DEMLoadConfig, dem_contact_force
from tensordem.whipping.fatigue import DNV_CURVES, miner_damage
from tensordem.whipping.rainflow import rainflow as rainflow_cycles

# --------------------------------------------------------------------------- #
# shared configuration (full-scale 200 m hull girder, 100 Hz monitoring)
# --------------------------------------------------------------------------- #
L = 200.0
EI = 1.0e14          # N m^2
RHO_A = 1.0e5        # kg/m
N_MODES = 5          # all below the 50 Hz Nyquist of the 100 Hz monitor
FS = 100.0
DT = 1.0 / FS
X_GAUGES = L * np.array([0.05, 0.15, 0.25, 0.35, 0.45,
                         0.55, 0.65, 0.75, 0.85, 0.95])
X_LOAD = 0.5 * L
PATCH_W = 4.0


@pytest.fixture(scope="module")
def rom() -> FreeFreeBeamROM:
    return FreeFreeBeamROM(L=L, EI=EI, rhoA=RHO_A, n_modes=N_MODES, zeta=0.01)


@pytest.fixture(scope="module")
def filter_(rom) -> ModalFilter:
    return ModalFilter(rom, X_GAUGES)


def _pulse_train(t, periods=3, period=0.5, width=0.1,
                 amps=(2.0e6, 3.5e6, 1.2e6), seed=0) -> np.ndarray:
    """Half-sine ice-load pulses (total force, N)."""
    rng = np.random.default_rng(seed)
    f = np.zeros_like(t)
    for k in range(periods):
        amp = amps[k % len(amps)] * rng.uniform(0.8, 1.2)
        tc = (k + 0.5) * period
        m = (t > tc - width / 2) & (t < tc + width / 2)
        f[m] = amp * np.sin(np.pi * (t[m] - tc + width / 2) / width)
    return f


def _invert(rom, filter_, eps, cutoff=None):
    """Modal filter -> modal forces -> total ice force (the online chain)."""
    if cutoff is None:
        cutoff = min(1.5 * float(rom.freq_hz[-1]), 0.45 * FS)
    q = filter_.extract(eps)
    qdot, qddot = differentiate(q, DT, cutoff)
    Fm = invert_modal_force(rom, q, qdot, qddot)
    return reconstruct_total_force(rom, Fm, X_LOAD, PATCH_W), q


# --------------------------------------------------------------------------- #
# ROM
# --------------------------------------------------------------------------- #
def test_rom_frequencies_match_closed_form(rom):
    bl = np.asarray([4.730040744862704, 7.853204624095838, 10.995607838001671,
                     14.137165491257464, 17.278759657399481])
    exact = bl**2 * math.sqrt(EI / (RHO_A * L**4))
    assert np.allclose(rom.omega_dry, exact, rtol=1e-12)


def test_rom_mass_normalization_and_orthogonality(rom):
    x = np.linspace(0.0, L, 4001)
    P = rom.phi(x)
    G = rom.rhoA * np.trapezoid(P[:, :, None] * P[None, :, :], x=x, axis=2)
    assert np.allclose(np.diag(G), 1.0, rtol=1e-8)
    assert np.abs(G - np.eye(rom.n_modes)).max() < 1e-6


def test_rom_strain_matrix_matches_finite_difference(rom):
    x = np.linspace(0.02 * L, 0.98 * L, 401)
    H = rom.strain_matrix(x)
    dphi = np.gradient(rom.phi(x), x, axis=1)
    assert np.abs(H.T - dphi).max() < 5e-4 * np.abs(dphi).max()


def test_rom_wet_modes_lower_frequencies_by_10_to_30_percent():
    dry = FreeFreeBeamROM(L=L, EI=EI, rhoA=RHO_A, n_modes=5)
    wet = FreeFreeBeamROM(L=L, EI=EI, rhoA=RHO_A, n_modes=5,
                         wet_scale=(0.5, 0.4, 0.3, 0.2, 0.1))
    drop = dry.omega / wet.omega - 1.0
    assert np.all((drop > 0.10) & (drop < 0.30))


# --------------------------------------------------------------------------- #
# modal filtering
# --------------------------------------------------------------------------- #
def test_modal_filter_single_mode_sinusoid_anchor(rom, filter_):
    """(a) single-mode sinusoid: recovered q_n error < 1 % at SNR = 20 dB."""
    n, A = 0, 1.0e-4
    t = np.arange(0.0, 10.0 + 1e-9, DT)
    q_true = np.zeros((rom.n_modes, t.size))
    q_true[n] = A * np.sin(rom.omega[n] * t)
    eps = rom.strain(q_true, X_GAUGES)
    sig = float(np.sqrt(np.mean(eps ** 2)))
    eps_n = eps + 0.1 * sig * np.random.default_rng(0).normal(size=eps.shape)
    q_fit = filter_.fit_mode(eps_n, t, n)
    err = np.linalg.norm(q_fit - q_true[n]) / np.linalg.norm(q_true[n])
    assert err < 0.01


@pytest.mark.parametrize("snr_db", [10, 15, 20, 25, 30])
def test_modal_filter_snr_sweep(rom, filter_, snr_db):
    """(g) SNR sweep 10..30 dB: error decreases monotonically, < 1 % at 20 dB."""
    n, A = 1, 1.0e-4
    t = np.arange(0.0, 10.0 + 1e-9, DT)
    q_true = np.zeros((rom.n_modes, t.size))
    q_true[n] = A * np.sin(rom.omega[n] * t)
    eps = rom.strain(q_true, X_GAUGES)
    sig = float(np.sqrt(np.mean(eps ** 2)))
    errs = []
    for seed in range(3):
        eps_n = eps + (10 ** (-snr_db / 20.0)) * sig * \
            np.random.default_rng(seed).normal(size=eps.shape)
        q_fit = filter_.fit_mode(eps_n, t, n)
        errs.append(np.linalg.norm(q_fit - q_true[n]) / np.linalg.norm(q_true[n]))
    assert max(errs) < (0.01 if snr_db >= 20 else 0.20)


def test_modal_filter_snr_sweep_monotone(rom, filter_):
    n, A = 1, 1.0e-4
    t = np.arange(0.0, 10.0 + 1e-9, DT)
    q_true = np.zeros((rom.n_modes, t.size))
    q_true[n] = A * np.sin(rom.omega[n] * t)
    eps = rom.strain(q_true, X_GAUGES)
    sig = float(np.sqrt(np.mean(eps ** 2)))
    means = []
    for snr_db in (10, 15, 20, 25, 30):
        errs = []
        for seed in range(3):
            eps_n = eps + (10 ** (-snr_db / 20.0)) * sig * \
                np.random.default_rng(seed).normal(size=eps.shape)
            q_fit = filter_.fit_mode(eps_n, t, n)
            errs.append(np.linalg.norm(q_fit - q_true[n]) /
                        np.linalg.norm(q_true[n]))
        means.append(float(np.mean(errs)))
    assert all(b < a for a, b in zip(means, means[1:]))
    assert means[2] < 0.01          # 20 dB


def test_modal_filter_zero_load_zero_response(rom, filter_):
    """(g) negative control: zero strain -> zero modal coordinates."""
    t = np.arange(0.0, 2.0 + 1e-9, DT)
    q = filter_.extract(np.zeros((X_GAUGES.size, t.size)))
    assert np.abs(q).max() == 0.0


def test_modal_filter_mode_separation(rom, filter_):
    n, A = 3, 1.0e-4
    t = np.arange(0.0, 10.0 + 1e-9, DT)
    q_true = np.zeros((rom.n_modes, t.size))
    q_true[n] = A * np.sin(rom.omega[n] * t)
    eps = rom.strain(q_true, X_GAUGES)
    q_hat = filter_.extract(eps)
    assert np.linalg.norm(q_hat[n]) / A > 0.99
    others = np.delete(q_hat, n, axis=0)
    assert np.abs(others).max() < 1e-12


def test_modal_filter_requires_enough_gauges(rom):
    with pytest.raises(ValueError):
        ModalFilter(rom, np.array([0.3 * L, 0.5 * L]))


# --------------------------------------------------------------------------- #
# load inversion
# --------------------------------------------------------------------------- #
def test_load_inversion_noise_free(rom, filter_):
    """(b) known ice load -> forward strain -> inverted load: < 15 % / < 10 %."""
    t = np.arange(0.0, 12.0 + 1e-9, DT)
    f = _pulse_train(t, periods=3, period=3.0, width=0.4)
    sol = rom.forward(f, t, X_LOAD, PATCH_W)
    eps = rom.strain(sol.q, X_GAUGES)
    f_hat, _ = _invert(rom, filter_, eps)
    assert peak_error(f_hat, f) < 0.15
    assert energy_error(f_hat, f) < 0.10


def test_load_inversion_20db(rom, filter_):
    """(b) with 20 dB measurement noise the same bounds must hold."""
    t = np.arange(0.0, 12.0 + 1e-9, DT)
    f = _pulse_train(t, periods=3, period=3.0, width=0.4)
    sol = rom.forward(f, t, X_LOAD, PATCH_W)
    eps = rom.strain(sol.q, X_GAUGES)
    sig = float(np.sqrt(np.mean(eps ** 2)))
    eps_n = eps + 0.1 * sig * np.random.default_rng(1).normal(size=eps.shape)
    f_hat, _ = _invert(rom, filter_, eps_n)
    assert peak_error(f_hat, f) < 0.15
    assert energy_error(f_hat, f) < 0.10


def test_load_inversion_zero_load_negative(rom, filter_):
    """(g) negative control: zero load -> zero inverted force."""
    t = np.arange(0.0, 4.0 + 1e-9, DT)
    f_hat, _ = _invert(rom, filter_, np.zeros((X_GAUGES.size, t.size)))
    assert np.abs(f_hat).max() < 1e-9


def test_load_inversion_metrics_sanity():
    f = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    assert peak_error(f + 0.1, f) == pytest.approx(0.1 / 3.0)
    assert energy_error(f, f) == 0.0
    assert energy_error(2.0 * f, f) == pytest.approx(3.0)   # |4-1|/1


# --------------------------------------------------------------------------- #
# rigid / elastic separation
# --------------------------------------------------------------------------- #
def test_separation_rigid_dominant():
    """(c) low-frequency (< 0.5 Hz) energy share > 80 % -> rigid body."""
    t = np.arange(0.0, 20.0 + 1e-9, DT)
    s = 1.0e-5 * np.sin(2 * np.pi * 0.2 * t) \
        + 0.1e-5 * np.sin(2 * np.pi * 4.0 * t)
    res = rigid_elastic_split(s, DT)
    assert res.frac_rigid > 0.80
    assert res.classification == "rigid_dominant"


def test_separation_elastic_dominant():
    """(c) high-frequency (> 0.5 Hz) energy share > 70 % -> elastic."""
    t = np.arange(0.0, 20.0 + 1e-9, DT)
    s = 0.2e-5 * np.sin(2 * np.pi * 0.2 * t) \
        + 1.0e-5 * np.sin(2 * np.pi * 4.0 * t) \
        + 0.6e-5 * np.sin(2 * np.pi * 7.0 * t)
    res = rigid_elastic_split(s, DT)
    assert res.frac_elastic > 0.70
    assert res.classification == "elastic_dominant"


def test_separation_energy_partition_and_reconstruction():
    t = np.arange(0.0, 20.0 + 1e-9, DT)
    s = 1.0e-5 * np.sin(2 * np.pi * 0.2 * t) \
        + 0.8e-5 * np.sin(2 * np.pi * 5.0 * t)
    res = rigid_elastic_split(s, DT)
    assert res.frac_rigid + res.frac_elastic == pytest.approx(1.0)
    assert np.abs(res.rigid + res.elastic - s).max() < 1e-12


# --------------------------------------------------------------------------- #
# fatigue
# --------------------------------------------------------------------------- #
def _stress_history_1000_cycles(rom, seed=42):
    """Deck stress from 1000 ice-load pulses (lognormal amplitudes)."""
    rng = np.random.default_rng(seed)
    period, width = 0.5, 0.1
    t = np.arange(0.0, 1000 * period + 1e-9, DT)
    f = np.zeros(t.size)
    amps = rng.lognormal(mean=13.8, sigma=0.3, size=1000)
    for k in range(1000):
        tc = (k + 0.5) * period
        m = (t > tc - width / 2) & (t < tc + width / 2)
        f[m] = amps[k] * 1.0e3 * np.sin(np.pi * (t[m] - tc + width / 2) / width)
    sol = rom.forward(f, t, X_LOAD, PATCH_W)
    w = rom.stress_weights(0.5 * L, E=2.0e11, y_deck=10.0)
    return t, w @ sol.q


def test_fatigue_1000_cycles_vs_rainflow(rom):
    """(d) 1000 ice-load cycles: incremental D vs single-pass rainflow < 10 %."""
    t, sigma = _stress_history_1000_cycles(rom)
    curve = DNV_CURVES["D_air"]
    cyc = rainflow_cycles(sigma)
    ref = miner_damage(curve, [c.range for c in cyc], [c.count for c in cyc])
    inc = IncrementalFatigue(curve)
    for k in range(0, sigma.size, 100):
        inc.update(sigma[k:k + 100])
    D_inc = inc.finish()
    assert abs(D_inc - ref["damage"]) / ref["damage"] < 0.10


def test_fatigue_zero_load_no_damage(rom):
    """(g) negative control: zero stress -> zero damage."""
    inc = IncrementalFatigue(DNV_CURVES["D_air"])
    for k in range(0, 1000, 100):
        inc.update(np.zeros(100))
    assert inc.finish() == 0.0


def test_fatigue_damage_monotone_in_amplitude(rom):
    t, sigma = _stress_history_1000_cycles(rom)
    curve = DNV_CURVES["D_air"]
    ds = []
    for scale in (0.5, 1.0, 2.0):
        inc = IncrementalFatigue(curve)
        for k in range(0, sigma.size, 100):
            inc.update(scale * sigma[k:k + 100])
        ds.append(inc.finish())
    assert ds[0] < ds[1] < ds[2]
    # S-N slope m = 3 below the knee: doubling the range ~ 8x the damage
    assert ds[2] / ds[1] > 4.0


def test_fatigue_remaining_life_consistent(rom):
    t, sigma = _stress_history_1000_cycles(rom)
    inc = IncrementalFatigue(DNV_CURVES["D_air"])
    for k in range(0, sigma.size, 100):
        inc.update(sigma[k:k + 100])
    D = inc.finish()
    assert inc.remaining_life_cycles == pytest.approx(1000.0 / D, rel=0.05)


# --------------------------------------------------------------------------- #
# DEM comparison
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def dem_events(tmp_path_factory):
    """Five small-lattice DEM ice events, cached on disk (first run ~1 min)."""
    cache = tmp_path_factory.mktemp("ice_twin") / "dem_events_small.npz"
    if cache.exists():
        z = np.load(cache)
        return [(z[f"t_{i}"], z[f"f_{i}"]) for i in range(5)]
    cfg = DEMLoadConfig(nx=8, ny=3, duration=0.8)
    out = {}
    for i in range(5):
        ti, fi, _ = dem_contact_force(cfg, seed=i + 1)
        out[f"t_{i}"] = ti
        out[f"f_{i}"] = fi
    np.savez_compressed(cache, **out)
    return [(out[f"t_{i}"], out[f"f_{i}"]) for i in range(5)]


def test_dem_compare_correlation_5_events(rom, filter_, dem_events):
    """(f) inverted vs DEM-predicted ice load: correlation > 0.7, 5 events."""
    corrs = []
    for t_dem, f_dem in dem_events:
        fs = 100.0
        t = np.arange(0.0, t_dem[-1], 1.0 / fs)
        f = np.interp(t, t_dem, f_dem) * 1000.0      # model -> full scale
        sol = rom.forward(f, t, X_LOAD, PATCH_W)
        eps = rom.strain(sol.q, X_GAUGES)
        sig = float(np.sqrt(np.mean(eps ** 2)))
        eps_n = eps + 0.1 * sig * np.random.default_rng(7).normal(size=eps.shape)
        f_hat, _ = _invert(rom, filter_, eps_n)
        res = compare_loads(f_hat, f, DT)
        corrs.append(res.correlation)
    assert len(corrs) >= 5
    assert all(c > 0.7 for c in corrs)


def test_dem_compare_correction_trigger():
    t = np.arange(0.0, 2.0 + 1e-9, DT)
    f = np.sin(2 * np.pi * 2.0 * t)
    good = compare_loads(f * 1.05, f, DT)
    assert not correction_needed(good)
    bad = compare_loads(f * 1.4, f, DT)
    assert correction_needed(bad)
    assert bad.peak_error == pytest.approx(0.4)


def test_dem_compare_zero_reference():
    res = compare_loads(np.zeros(10), np.zeros(10), DT)
    assert res.correlation == 0.0 and res.peak_error == 0.0


# --------------------------------------------------------------------------- #
# the online twin
# --------------------------------------------------------------------------- #
def test_twin_realtime_latency_100hz(rom):
    """(e) per-window inversion latency < 50 ms at 100 Hz sampling."""
    twin = IceTwin(rom, X_GAUGES, X_LOAD, PATCH_W, fs=100.0, window_s=1.0,
                    snr_db=30.0)
    t = np.arange(0.0, 6.0 + 1e-9, DT)
    f = _pulse_train(t, periods=6, period=1.0, width=0.3)
    eps = twin.simulate_strain(f, t, snr_db=float("inf"))
    for k in range(0, eps.shape[1], twin.n_win):
        twin.update(eps[:, k:k + twin.n_win])
    assert twin.mean_latency_ms < 50.0
    assert twin.damage > 0.0


def test_twin_end_to_end(rom):
    """Full chain: DEM-like pulses -> strain -> inversion -> fatigue -> split."""
    twin = IceTwin(rom, X_GAUGES, X_LOAD, PATCH_W, fs=100.0, window_s=1.0,
                    snr_db=30.0)
    t = np.arange(0.0, 6.0 + 1e-9, DT)
    f = _pulse_train(t, periods=6, period=1.0, width=0.3)
    eps = twin.simulate_strain(f, t, snr_db=float("inf"))
    f_hat = np.zeros_like(f)
    fracs = []
    for k in range(0, eps.shape[1], twin.n_win):
        step = twin.update(eps[:, k:k + twin.n_win])
        f_hat[k:k + twin.n_win] = step.f_hat
        fracs.append(step.frac_rigid)
    res = compare_loads(f_hat, f, DT)
    assert res.correlation > 0.9
    assert peak_error(f_hat, f) < 0.15
    assert twin.damage > 0.0
    assert all(0.0 <= x <= 1.0 for x in fracs)


def test_twin_fatigue_accumulates_monotone(rom):
    twin = IceTwin(rom, X_GAUGES, X_LOAD, PATCH_W, fs=100.0, window_s=1.0,
                    snr_db=float("inf"))
    t = np.arange(0.0, 6.0 + 1e-9, DT)
    f = _pulse_train(t, periods=6, period=1.0, width=0.3)
    eps = twin.simulate_strain(f, t, snr_db=float("inf"))
    prev = 0.0
    for k in range(0, eps.shape[1], twin.n_win):
        step = twin.update(eps[:, k:k + twin.n_win])
        assert step.damage_total >= prev
        prev = step.damage_total
    assert prev > 0.0


def test_turning_points_indexed_matches_reference():
    from tensordem.whipping.rainflow import turning_points
    x = np.array([0.0, 1.0, 0.5, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5])
    idx, vals = turning_points_indexed(x)
    assert np.allclose(vals, turning_points(x))
    assert (np.diff(idx) > 0).all()
