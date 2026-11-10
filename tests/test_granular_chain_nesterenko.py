"""Tests for ``tensordem.granular_chain``: Nesterenko solitary waves + layer damper.

The suite is deliberately split into

* analytic/unit checks that pin the *derivations* (Hertz stiffness, dispersion
  consistency of the modified equation, ``cos^4`` closed-form quantities, the
  ``F^{1/6}`` and ``F^{-1/6}`` scalings, the fit utilities), and
* short DEM runs that pin the *physics* (exponent 1/6 on a real chain, energy
  conservation, the three negative controls, transmissibility of a single grain
  layer).

Full campaign numbers live in ``runs/p307w1_n06/report.json``
(``scripts/p307w1_n06_granular_chain.py``); these tests must stay fast enough for
the repo's default pytest run.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.granular_chain import (
    ChainConfig,
    HertzianChain,
    LayerConfig,
    GranularLayer,
    arrival_speed,
    energy_method_zeta,
    exponential_decay_length,
    fit_sdof_to_curve,
    hertz_stiffness,
    linear_sound_speed,
    loglog_slope,
    nesterenko_profile_strain,
    nesterenko_pulse_duration,
    nesterenko_speed,
    nesterenko_speed_from_strain,
    nesterenko_width,
    profile_l1_force,
    pulse_width_measured,
    sdof_transmissibility,
    stopband,
    window_for,
)
from tensordem.granular_chain.layer import zeta_from_bandwidth, zeta_from_peak
from tensordem.granular_chain.study import (
    SolitaryWaveStudy,
    SolitaryWaveStudyConfig,
    speed_from_monitors,
    symmetrised_L1,
)

# --------------------------------------------------------------------------
# analytic layer
# --------------------------------------------------------------------------


def test_hertz_stiffness_scales_with_sqrt_radius():
    """F = k delta^{3/2}: k = (4/3) E* sqrt(R*) and R* = R/2 for equal spheres."""
    e_star = 1.0e11
    k1 = hertz_stiffness(5e-3, 5e-3, e_star)
    k2 = hertz_stiffness(20e-3, 20e-3, e_star)
    assert k1 == pytest.approx(4.0 / 3.0 * e_star * math.sqrt(2.5e-3), rel=1e-12)
    # k ~ sqrt(R) for equal spheres
    assert k2 / k1 == pytest.approx(math.sqrt(20e-3 / 5e-3), rel=1e-12)
    # mixed radii use the harmonic-mean radius R* = R1 R2/(R1+R2)
    assert hertz_stiffness(2e-3, 2e-3, e_star) == pytest.approx(
        4.0 / 3.0 * e_star * math.sqrt(1e-3), rel=1e-12)


def test_nesterenko_speed_sixth_root_law():
    """``v_s ~ F^{1/6}`` and ``v_s^2 = (5/6)(k a^{5/2}/m) sqrt(s_m)``."""
    k, a, m = 7.33e9, 1.0e-2, 4.08e-3
    f1, f2 = 100.0, 1600.0
    v1, v2 = nesterenko_speed(f1, k, a, m), nesterenko_speed(f2, k, a, m)
    assert v2 / v1 == pytest.approx((f2 / f1) ** (1.0 / 6.0), rel=1e-12)
    # strain form is the same statement with F_m = k (a s_m)^{3/2}
    s_m = 1e-3
    f_m = k * (a * s_m) ** 1.5
    assert nesterenko_speed_from_strain(s_m, k, a, m) == pytest.approx(
        nesterenko_speed(f_m, k, a, m), rel=1e-12)
    assert nesterenko_speed_from_strain(s_m, k, a, m) == pytest.approx(
        math.sqrt(5.0 / 6.0 * (k * a ** 2.5 / m) * math.sqrt(s_m)), rel=1e-12)
    with pytest.raises(ValueError):
        loglog_slope([1.0], [1.0])


def test_width_is_amplitude_independent_and_pulse_duration_scales():
    """Width ``L = a sqrt(3)``; duration ``tau = pi L / v_s ~ F^{-1/6}``."""
    a = 1e-2
    assert nesterenko_width(a) == pytest.approx(a * math.sqrt(3.0), rel=1e-12)
    k, m = 7.33e9, 4.08e-3
    t1 = nesterenko_pulse_duration(100.0, k, a, m)
    t2 = nesterenko_pulse_duration(6400.0, k, a, m)   # 64x force = 2x speed
    # tau = pi L / v_s and L is fixed, so tau ~ F^{-1/6}
    assert t2 / t1 == pytest.approx(64.0 ** (-1.0 / 6.0), rel=1e-12)


def test_cos4_profile_closed_form():
    """``s = s_m cos^4((x-x_c)/L)`` with compact support and integral 3/8 pi L."""
    a = 1e-2
    L = nesterenko_width(a)
    x = np.linspace(-3 * a, 3 * a, 20001)
    prof = nesterenko_profile_strain(x, 1.0, 0.0, a)
    assert prof.max() == pytest.approx(1.0, abs=1e-12)
    # support = pi L
    inside = x[np.abs(x) <= 0.5 * math.pi * L]
    assert np.allclose(prof[np.abs(x) > 0.5 * math.pi * L + 1e-9], 0.0)
    integral = np.trapezoid(prof, x)
    assert integral == pytest.approx(3.0 * math.pi * L / 8.0, rel=2e-4)
    # value at one spacing: cos^4(a/L)
    assert nesterenko_profile_strain(np.array([a]), 1.0, 0.0, a)[0] == pytest.approx(
        math.cos(a / L) ** 4, rel=1e-12)


def test_modified_equation_matches_discrete_dispersion():
    """The 4th-order coefficient must reproduce ``omega^2 = 4 k_t/m sin^2(qa/2)``.

    ``m s_tt = A (s^{3/2})_{xx} + C (s^{3/2})_{xxxx}`` with ``C = A a^2/12`` is the
    linear-dispersion-consistent truncation; ``C = A a^2/24`` (a naive one-sided
    expansion of the force) is not, and would make the pulse 1.4x too narrow.
    That factor is what :func:`nesterenko_width` encodes (``L = a sqrt(3)``
    follows from ``L^2 = 36 C/A``).
    """
    a = 1.0
    for qa in (1e-3, 1e-2, 5e-2):
        exact = 4.0 * math.sin(0.5 * qa) ** 2
        truncated = qa ** 2 - qa ** 4 / 12.0          # C = A a^2/12
        naive = qa ** 2 - qa ** 4 / 24.0              # C = A a^2/24
        assert truncated == pytest.approx(exact, rel=1e-6)
        assert abs(exact - naive) > abs(exact - truncated)
    assert nesterenko_width(a) ** 2 == pytest.approx(36.0 * (a ** 2 / 12.0), rel=1e-12)


def test_linear_sound_speed_and_limit():
    """``c_0 = a sqrt(k_t/m)`` with ``k_t = 1.5 k sqrt(delta_0)``."""
    k, a, m, f0 = 7.33e9, 1e-2, 4.08e-3, 1.0e3
    d0 = (f0 / k) ** (2.0 / 3.0)
    assert linear_sound_speed(f0, k, a, m) == pytest.approx(
        a * math.sqrt(1.5 * k * math.sqrt(d0) / m), rel=1e-12)
    assert linear_sound_speed(0.0, k, a, m) == 0.0     # sonic vacuum
    # stronger precompression -> faster linear sound
    assert linear_sound_speed(4.0 * f0, k, a, m) > linear_sound_speed(f0, k, a, m)


def test_loglog_and_decay_fits_round_trip():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    y = 3.0 * x ** 0.25
    fit = loglog_slope(x, y)
    assert fit["exponent"] == pytest.approx(0.25, rel=1e-10)
    assert fit["prefactor"] == pytest.approx(3.0, rel=1e-10)
    assert fit["r2"] == pytest.approx(1.0, rel=1e-12)

    a = 1e-2
    n = np.arange(60, dtype=float)
    force = 100.0 * np.exp(-n * a / 0.5)
    dec = exponential_decay_length(force, a, 5, 55)
    assert dec["xi_m"] == pytest.approx(0.5, rel=1e-9)
    assert dec["slope_per_bead"] == pytest.approx(-a / 0.5, rel=1e-9)


def test_pulse_width_measure_recovers_cos4_integral_width():
    """``width_eq`` of a sampled ``cos^4`` equals ``3 pi L / 8`` within 3%."""
    a = 1e-2
    L = nesterenko_width(a)
    x = np.arange(-6, 7) * a
    s = nesterenko_profile_strain(x, 2.0, 0.0, a)
    rec = pulse_width_measured(s, x, 0.0)
    assert rec["width_eq_m"] == pytest.approx(3.0 * math.pi * L / 8.0, rel=0.03)
    assert rec["peak_strain"] == pytest.approx(2.0, rel=1e-12)


def test_symmetrised_l1_is_zero_for_the_analytic_profile():
    a = 1e-2
    x = np.arange(-5, 6) * a
    s = nesterenko_profile_strain(x, 1.0, 0.0, a)
    rec = symmetrised_L1(s, x, 5, a)
    assert rec["l1_even"] == pytest.approx(0.0, abs=1e-12)
    assert rec["asymmetry_D"] == pytest.approx(1.0, rel=1e-12)


def test_speed_from_monitors_removes_a_static_offset():
    """A large static force must not disturb the perturbation arrival fit."""
    a = 1e-2
    x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    t = x / 1200.0
    rec = {"monitor_histories": {}, "contact_x": x}
    tt = np.linspace(0.0, 1e-3, 2001)
    for i, (xi, ti) in enumerate(zip(x, t)):
        sig = 5000.0 + 50.0 * np.exp(-((tt - ti) / 2e-5) ** 2)
        rec["monitor_histories"][i] = np.column_stack([tt, sig])
    sp = speed_from_monitors(rec)
    assert sp["speed_m_s"] == pytest.approx(1200.0, rel=0.10)


# --------------------------------------------------------------------------
# chain physics
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def short_chain_run():
    chain = HertzianChain(ChainConfig(n=40, impact_velocity=1.0))
    rec = chain.run()
    return chain, rec


def test_chain_energy_is_conserved(short_chain_run):
    chain, rec = short_chain_run
    assert rec["energy"]["max_abs_residual"] < 2e-3
    assert rec["dt"] > 0 and rec["n_steps"] > 100


def test_chain_initial_geometry_and_hertz_law():
    chain = HertzianChain(ChainConfig(n=8))
    gaps = np.diff(chain.x0) - chain.spacing
    assert np.allclose(gaps, 0.0, atol=1e-12)          # just touching
    forces = chain.forces(chain.pos)
    assert np.allclose(forces, 0.0)                    # no precompression
    # pushing a bead in creates the 3/2 power law
    pos = chain.pos.copy()
    delta = 1e-5
    pos[3] -= delta                                    # compress contacts 2 and 3
    f = chain.forces(pos)
    assert f[2] == pytest.approx(chain.k[2] * delta ** 1.5, rel=1e-9)


def test_solitary_wave_speed_has_exponent_one_sixth():
    """Physics test: measured ``v_s(F)`` slope is 1/6 within 3%."""
    study = SolitaryWaveStudy(SolitaryWaveStudyConfig(
        base=ChainConfig(n=40), amplitudes=(0.5, 1.0, 2.0, 3.0),
        profile_amplitudes=(), disorder_sigmas=(0.05,), disorder_seeds=1,
        control_amplitudes=(0.5, 2.0)))
    scan = study.amplitude_scan()
    assert scan["slope_vs_force"]["r2"] > 0.999
    assert scan["slope_vs_force"]["exponent"] == pytest.approx(1.0 / 6.0, rel=0.03)
    assert scan["slope_vs_velocity"]["exponent"] == pytest.approx(0.2, rel=0.05)
    assert scan["energy_residual_max"] < 2e-2
    # analytic speed is reproduced in absolute terms
    assert scan["speed_vs_theory_max_rel"] < 0.05


def test_profile_matches_the_analytic_solution():
    study = SolitaryWaveStudy(SolitaryWaveStudyConfig(
        base=ChainConfig(n=40), amplitudes=(0.5, 1.0), profile_amplitudes=(1.0,),
        disorder_sigmas=(0.05,), disorder_seeds=1, control_amplitudes=(1.0,)))
    runs = study.amplitude_scan()["runs"]
    prof = next(r["profile"] for r in runs if "profile" in r)
    assert prof["l1_even"] < 0.10
    assert prof["width_eq_over_spacing"] == pytest.approx(
        prof["width_eq_theory_over_spacing"], rel=0.12)
    assert prof["asymmetry_D"] != pytest.approx(1.0, abs=0.01)   # real asymmetry


def test_negative_control_linear_chain_has_constant_speed():
    study = SolitaryWaveStudy(SolitaryWaveStudyConfig(
        base=ChainConfig(n=40), amplitudes=(0.5, 1.0), profile_amplitudes=(),
        disorder_sigmas=(0.05,), disorder_seeds=1, control_amplitudes=(0.5, 2.0)))
    ctl = study.controls()
    lin = ctl["linear_spring"]
    assert lin["pass"]
    assert abs(lin["speed_exponent_vs_velocity"]) < 0.05


def test_negative_control_precompressed_returns_to_linear_limit():
    study = SolitaryWaveStudy(SolitaryWaveStudyConfig(
        base=ChainConfig(n=40), amplitudes=(0.5, 1.0), profile_amplitudes=(),
        disorder_sigmas=(0.05,), disorder_seeds=1, control_amplitudes=(0.5, 2.0)))
    ctl = study.controls()
    pre = ctl["precompressed"]
    assert pre["pass"]
    for run in pre["runs"]:
        assert abs(run["deviation_from_c0"]) < 0.15


def test_negative_control_inelastic_kills_the_solitary_wave():
    study = SolitaryWaveStudy(SolitaryWaveStudyConfig(
        base=ChainConfig(n=40), amplitudes=(0.5, 1.0), profile_amplitudes=(),
        disorder_sigmas=(0.05,), disorder_seeds=1, control_amplitudes=(0.5, 1.0)))
    ctl = study.controls()
    assert ctl["inelastic"]["pass"]
    assert ctl["inelastic"]["far_near_force_ratio"][0] < 0.99


def test_disordered_chain_decays_faster_than_ordered():
    study = SolitaryWaveStudy(SolitaryWaveStudyConfig(
        base=ChainConfig(n=40), amplitudes=(0.5, 1.0), profile_amplitudes=(),
        disorder_sigmas=(0.05, 0.2), disorder_seeds=2,
        control_amplitudes=(0.5,)))
    dis = study.disorder_scan()
    xi = {r["sigma_radius"]: r["xi_m"] for r in dis["rows"]}
    assert xi[0.05] > xi[0.2]
    assert dis["slope_xi_vs_sigma"]["exponent"] < -1.0


# --------------------------------------------------------------------------
# layer / PID
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def settled_layer():
    layer = GranularLayer(LayerConfig(nrows=2, ncols=4))
    layer.settle(0.04)
    return layer


def test_layer_settles_and_has_positive_stiffness(settled_layer):
    st = settled_layer.static_stiffness()
    assert st["k_eff_N_m"] > 0.0
    assert st["fn_Hz"] > 0.0
    z = settled_layer.pos[:, 1]
    assert np.all(z >= -1e-6)          # nothing fell through the base
    assert settled_layer.z_lid > z.max()


def test_layer_energy_ledger_closes(settled_layer):
    rec = settled_layer.settle(0.01)
    assert abs(rec["residual"]) < 0.05


def test_layer_transmissibility_decays_above_resonance(settled_layer):
    fn = settled_layer.static_stiffness()["fn_Hz"]
    low = settled_layer.drive(0.8 * fn, cycles_measure=4, cycles_warmup=2)
    high = settled_layer.drive(6.0 * fn, cycles_measure=4, cycles_warmup=2)
    assert low["transmissibility"] > high["transmissibility"]
    assert high["transmissibility_dB"] < -6.0


def test_stopband_finder():
    pts = [{"frequency_Hz": f, "transmissibility_dB": db}
           for f, db in ((10, 0.0), (100, -5.0), (200, -25.0), (300, -30.0),
                         (400, -10.0))]
    band = stopband(pts)
    assert band["found"]
    assert band["f_lo_Hz"] == 200.0 and band["f_hi_Hz"] == 300.0
    assert band["min_dB"] == pytest.approx(-30.0)
    assert not stopband(pts, threshold_db=-60.0)["found"]


def test_sdof_zeta_identifications_round_trip():
    """Both identifications must return the zeta that generated the curve."""
    zeta_true, fn = 0.12, 60.0
    f = np.linspace(5.0, 400.0, 400)
    T = sdof_transmissibility(f, fn, zeta_true)
    peak = zeta_from_peak(float(T.max()))
    bw = zeta_from_bandwidth(f, T)
    fit = fit_sdof_to_curve(f, T)
    assert peak == pytest.approx(zeta_true, rel=0.05)
    assert bw["zeta"] == pytest.approx(zeta_true, rel=0.05)
    assert fit["zeta"] == pytest.approx(zeta_true, rel=0.05)
    assert fit["fn_Hz"] == pytest.approx(fn, rel=0.05)
    assert peak == pytest.approx(bw["zeta"], rel=0.10)


def test_energy_method_zeta_definition():
    """``zeta = dE / (4 pi E_peak)`` with ``E_peak = k X^2 / 2``."""
    k, X, zeta = 1000.0, 1e-3, 0.1
    e_peak = 0.5 * k * X ** 2
    dE = 4.0 * math.pi * zeta * e_peak
    assert energy_method_zeta(dE, k, X) == pytest.approx(zeta, rel=1e-12)
    assert math.isnan(energy_method_zeta(dE, k, 0.0))


def test_layer_config_derived_geometry():
    cfg = LayerConfig(ncols=5, nrows=3)
    assert cfg.thickness == pytest.approx(
        2 * cfg.radius + 2 * math.sqrt(3.0) * cfg.radius, rel=1e-12)
    assert 0.3 < cfg.filling_ratio < 0.95
    with pytest.raises(ValueError):
        LayerConfig(ncols=1)
    with pytest.raises(ValueError):
        LayerConfig(damping_ratio=1.5)


def test_chain_config_validation():
    with pytest.raises(ValueError):
        ChainConfig(n=2)
    with pytest.raises(ValueError):
        ChainConfig(restitution=1.5)
    with pytest.raises(ValueError):
        ChainConfig(sigma_radius=0.6)


def test_arrival_speed_fit_window():
    t = np.arange(20, dtype=float) * 1e-4
    x = 500.0 * t
    sp = arrival_speed(t, x, *window_for(20))
    assert sp["speed_m_s"] == pytest.approx(500.0, rel=1e-9)
    assert sp["r2"] == pytest.approx(1.0, rel=1e-12)


def test_profile_l1_force_matches_analytic_force_profile():
    a = 1e-2
    x = np.arange(-5, 6) * a
    f = np.zeros_like(x)
    # cos^6 force profile of the analytic solution
    from tensordem.granular_chain import nesterenko_profile_force
    f = nesterenko_profile_force(x, 100.0, 0.0, a)
    assert profile_l1_force(f, x, 100.0, 0.0, a) == pytest.approx(0.0, abs=1e-12)