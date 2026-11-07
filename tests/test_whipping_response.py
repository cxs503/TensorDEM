"""Tests for the ice-induced whipping chain (``tensordem.whipping``).

The suite pins down each link against an independent reference:

* 2-D BEM added mass against the closed-form ellipse values,
* beam-FEM dry frequencies against the free-free closed form,
* the piecewise-exact Duhamel integrator against analytic step response,
* the energy ledger (with a coarse-grid negative control that must fail),
* ASTM E1049-85 rainflow counting invariants,
* the DNV-RP-C203 curve D knee points and Miner scaling.

The DEM integration is exercised separately (``test_dem_record_*``) on a very
short case so the suite stays fast.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.whipping import fatigue, modes, pipeline, pulses, response, sections
from tensordem.whipping.dem_load import DEMLoadConfig, dem_contact_force
from tensordem.whipping.rainflow import rainflow as rainflow_cycles, turning_points


# --------------------------------------------------------------------------- #
# 2-D added mass (BEM)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_panels", [80])
def test_added_mass_semicircle_matches_closed_form(n_panels):
    pts = sections.ellipse_section(1.0, 1.0, n_panels, deep=False)
    res = sections.added_mass_per_length(pts, 1025.0)
    exact = sections.added_mass_semielipse_reference(1025.0, 1.0)
    assert abs(res["added_mass_per_length"] / exact - 1.0) < 1.0e-2
    assert res["rhs_residual"] < 1e-10


def test_added_mass_deep_ellipse_matches_closed_form():
    pts = sections.ellipse_section(1.0, 0.6, 80, deep=True)
    res = sections.added_mass_per_length(pts, 1025.0, free_surface=False)
    exact = sections.added_mass_ellipse_reference(1025.0, 1.0)
    assert abs(res["added_mass_per_length"] / exact - 1.0) < 2.0e-2


def test_added_mass_converges_with_panel_refinement():
    exact = sections.added_mass_semielipse_reference(1025.0, 1.0)
    errs = []
    for n in (40, 80):
        pts = sections.ellipse_section(1.0, 1.0, n, deep=False)
        errs.append(abs(sections.added_mass_per_length(pts, 1025.0)
                        ["added_mass_per_length"] / exact - 1.0))
    assert errs[1] < errs[0]


def test_box_section_added_mass_exceeds_flat_plate_bound():
    """A deep section entrains at least the flat-plate value rho pi B^2 / 8."""
    breadth = 2.0
    res = sections.added_mass_per_length(sections.box_section(breadth, 0.5, 40), 1025.0)
    lower_bound = 1025.0 * math.pi * breadth**2 / 8.0
    assert res["added_mass_per_length"] > lower_bound


def test_free_surface_contour_must_touch_the_waterline():
    pts = sections.ellipse_section(1.0, 1.0, 20, deep=True) + np.array([0.0, -2.0])
    with pytest.raises(ValueError):
        sections.added_mass_per_length(pts, 1025.0, free_surface=True)


# --------------------------------------------------------------------------- #
# hull-girder modes
# --------------------------------------------------------------------------- #
def test_beam_fem_frequencies_match_free_free_closed_form():
    ei, m_bar, length = 6.18e12, 203600.0, 150.0
    dry = modes.dry_modes(ei, m_bar, length, 60, 4)
    exact = modes.free_free_analytic_frequencies(ei, m_bar, length, 4)
    assert np.all(np.abs(dry.freqs / exact - 1.0) < 1.0e-4)


def test_mode_curvature_matches_finite_difference_of_the_shape():
    ei, m_bar, length = 6.18e12, 203600.0, 150.0
    dry = modes.dry_modes(ei, m_bar, length, 60, 2)
    x = 60.0
    h = 1.0e-2
    fd = (dry.shape(x + h, 0) - 2 * dry.shape(x, 0) + dry.shape(x - h, 0)) / h**2
    assert abs(fd[0] / dry.shape(x, 0, deriv=2)[0] - 1.0) < 1.0e-4


def test_wet_modes_lower_the_frequencies_by_10_to_30_percent():
    case = pipeline.ShipCase()
    dry = modes.dry_modes(case.youngs_modulus * case.inertia, case.mass_per_length,
                          case.length, case.n_elements, case.n_modes)
    mass = pipeline.hull_added_mass_per_length(case)
    wet = modes.wet_modes(dry, mass["m_a_of_x"])
    assert np.all(wet.freqs < dry.freqs)
    assert np.all((wet.reduction[:2] >= 0.10) & (wet.reduction[:2] <= 0.30))


def test_wet_frequency_drop_is_monotone_in_the_added_mass_relief():
    case = pipeline.ShipCase()
    dry = modes.dry_modes(case.youngs_modulus * case.inertia, case.mass_per_length,
                          case.length, case.n_elements, case.n_modes)
    drops = []
    for relief in (0.5, 0.75, 1.0):
        mass = pipeline.hull_added_mass_per_length(case, relief=relief)
        drops.append(modes.wet_modes(dry, mass["m_a_of_x"]).reduction[0])
    assert drops[0] < drops[1] < drops[2]


# --------------------------------------------------------------------------- #
# pulses
# --------------------------------------------------------------------------- #
def _synthetic_train(times, events):
    out = np.zeros_like(times)
    feats = []
    for k, (t0, t_rise, t_decay, peak) in enumerate(events):
        feat = pulses.PulseFeatures(index=k, t_onset=t0, t_peak=t0 + t_rise,
                                    t_end=t0 + t_rise + t_decay, peak=peak,
                                    rise_time=t_rise, decay_time=t_decay,
                                    width_fwhm=0.0, impulse=peak * 0.55 * (t_rise + t_decay),
                                    threshold=0.1 * peak)
        parl = pulses.ParametricPulse.from_features(feat)
        out += parl(times)
        feats.append(feat)
    return out


def test_pulse_detection_finds_every_event_with_its_features():
    t = np.arange(0.0, 2.0, 1.0 / 1000.0)
    events = [(0.2, 0.02, 0.09, 4.0e6), (0.8, 0.03, 0.12, 3.0e6), (1.4, 0.015, 0.08, 5.0e6)]
    f = _synthetic_train(t, events)
    found = pulses.detect_pulses(t, f)
    assert len(found) == 3
    for got, (_, _, _, peak) in zip(found, events):
        assert got.peak == pytest.approx(peak, rel=1e-6)
        assert got.rise_time > 0.0 and got.impulse > 0.0


def test_pulse_parametrisation_survives_a_detect_rebuild_round_trip():
    t = np.arange(0.0, 2.0, 1.0 / 2000.0)
    events = [(0.2, 0.02, 0.09, 4.0e6), (0.8, 0.03, 0.12, 3.0e6), (1.4, 0.015, 0.08, 5.0e6)]
    f = _synthetic_train(t, events)
    rebuilt = np.zeros_like(t)
    for feat in pulses.detect_pulses(t, f):
        rebuilt = rebuilt + pulses.ParametricPulse.from_features(feat)(t)
    a = pulses.detect_pulses(t, f)
    b = pulses.detect_pulses(t, rebuilt)
    assert len(a) == len(b) == 3
    for pa, pb in zip(a, b):
        assert pb.peak == pytest.approx(pa.peak, rel=5e-3)
        assert pb.impulse == pytest.approx(pa.impulse, rel=5e-2)


def test_parametric_pulse_reproduces_impulse_and_rise_time():
    t = np.arange(0.0, 1.0, 1.0 / 2000.0)
    f = _synthetic_train(t, [(0.2, 0.025, 0.1, 3.5e6)])
    feat = pulses.detect_pulses(t, f)[0]
    parl = pulses.ParametricPulse.from_features(feat)
    assert parl.impulse() == pytest.approx(feat.impulse, rel=1e-9)
    assert parl(t).max() == pytest.approx(feat.peak, rel=1e-9)
    assert parl.exponent > 0.0


def test_pulse_detection_rejects_an_empty_record():
    t = np.arange(0.0, 0.1, 1.0 / 1000.0)
    with pytest.raises(ValueError):
        pulses.detect_pulses(t, np.zeros_like(t))


def test_resample_uniform_produces_the_requested_rate():
    t = np.array([0.0, 0.3, 1.1])
    v = np.array([0.0, 3.0, 11.0])
    grid, vals = pulses.resample_uniform(t, v, 1000.0)
    assert grid[1] - grid[0] == pytest.approx(1e-3)
    assert vals[-1] == pytest.approx(11.0)


# --------------------------------------------------------------------------- #
# modal response and energy ledger
# --------------------------------------------------------------------------- #
def test_duhamel_integrator_is_exact_for_a_step_load():
    dt = 1.0e-4
    t = np.arange(0.0, 8.0, dt)
    omega = np.array([2.0 * math.pi * 0.5])
    sol = response.modal_response(t, np.ones((1, t.size)), omega, np.array([0.0]))
    exact = (1.0 - np.cos(omega[0] * t)) / omega[0] ** 2
    assert np.max(np.abs(sol.q[0] - exact)) < 1e-9


def _ledger_fixture(rate_hz):
    t = np.arange(0.0, 3.0, 1.0 / rate_hz)
    f = _synthetic_train(t, [(0.3, 0.02, 0.09, 12.0e6), (1.2, 0.025, 0.11, 9.0e6)])
    omega = 2.0 * math.pi * np.array([0.69, 1.83, 3.57])
    zeta = np.full(3, 0.015)
    shape = np.array([2.6e-4, 1.9e-4, 1.3e-4])
    sol = response.modal_response(t, shape[:, None] * f[None, :], omega, zeta)
    return response.energy_ledger(sol, f, shape, 1.0 / rate_hz)


def test_energy_ledger_closes_below_1e_6():
    led = _ledger_fixture(4000.0)
    assert led["residual"] < 1.0e-6
    assert led["work_input_simpson_J"] == pytest.approx(
        led["kinetic_J"] + led["strain_J"] + led["damping_J"], rel=1e-6)


def test_energy_ledger_coarse_grid_is_a_negative_control():
    """A 200 Hz grid must *not* reach the 1e-6 acceptance level."""
    led = _ledger_fixture(200.0)
    assert led["residual"] > 1.0e-6


def test_lowpass_and_bandpass_isolate_the_expected_frequency():
    dt = 1.0 / 1000.0
    t = np.arange(0.0, 4.0, dt)
    slow = np.sin(2 * math.pi * 0.3 * t)
    fast = np.sin(2 * math.pi * 5.0 * t)
    assert response.rms(response.bandpass(slow, dt, 1.0, 50.0)) < 0.05
    assert response.rms(response.lowpass(fast, dt, 1.0)) < 0.05
    assert response.rms(response.bandpass(fast + slow, dt, 1.0, 50.0)) == pytest.approx(
        response.rms(fast), rel=0.05)


# --------------------------------------------------------------------------- #
# rainflow counting
# --------------------------------------------------------------------------- #
def test_rainflow_of_a_sinusoid_gives_one_range_of_twice_the_amplitude():
    dt = 1.0 / 500.0
    t = np.arange(0.0, 10.0, dt)
    amp = 7.5
    cycles = rainflow_cycles(amp * np.sin(2 * math.pi * 1.0 * t))
    full = [c for c in cycles if c.count == 1.0]
    assert full
    dominant = [c for c in full if abs(c.range / (2 * amp) - 1.0) < 1e-3]
    assert len(dominant) >= 0.8 * len(full)   # the end effects close at half range
    assert sum(c.count for c in cycles) == pytest.approx(10.0, abs=0.6)


def test_rainflow_count_invariant_for_a_random_walk():
    rng = np.random.default_rng(7)
    signal = np.cumsum(rng.normal(size=400))
    cycles = rainflow_cycles(signal)
    n_turning = turning_points(signal).size
    assert sum(c.count for c in cycles) == pytest.approx(0.5 * (n_turning - 1), rel=1e-12)


def test_rainflow_counts_an_inner_cycle_inside_an_outer_one():
    # 0 -> 10 -> 2 -> 8 -> 0 : a small 6-unit cycle inside a 10-unit swing
    cycles = rainflow_cycles(np.array([0.0, 10.0, 2.0, 8.0, 0.0]))
    ranges = sorted(round(c.range, 6) for c in cycles)
    assert 6.0 in ranges


# --------------------------------------------------------------------------- #
# fatigue
# --------------------------------------------------------------------------- #
def test_dnv_curve_d_knee_points():
    curve = fatigue.DNV_CURVES["D_air"]
    assert curve.knee_stress_MPa() == pytest.approx(52.6, abs=0.1)
    assert float(curve.cycles_to_failure(np.array([curve.knee_stress()]))[0]) == pytest.approx(
        1.0e7, rel=1e-2)
    assert float(curve.cycles_to_failure(np.array([100e6]))[0]) == pytest.approx(1.46e6, rel=1e-2)


def test_miner_damage_is_linear_in_the_cycle_count():
    curve = fatigue.DNV_CURVES["D_air"]
    d1 = fatigue.miner_damage(curve, [80e6], [1000.0])["damage"]
    d2 = fatigue.miner_damage(curve, [80e6], [2000.0])["damage"]
    assert d2 == pytest.approx(2.0 * d1, rel=1e-12)
    # analytic cross-check: N = a1 dS^-3 in the m1 branch (dS in MPa)
    expected = 1000.0 / (curve.a1 * 80.0 ** -curve.m1)
    assert d1 == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- #
# DEM record and the full chain
# --------------------------------------------------------------------------- #
def test_dem_record_is_sampled_at_least_at_1khz_and_lasts_a_second():
    config = DEMLoadConfig(duration=1.2, nx=8, ny=3)
    t, f, info = dem_contact_force(config)
    assert 1.0 / (t[1] - t[0]) >= 1000.0
    assert info["dem_steps"] > 100
    assert np.all(np.isfinite(f))
    assert f.max() > 0.0
    assert info["particle_count"] == config.nx * config.ny


def test_full_chain_returns_every_criterion_and_a_closed_ledger():
    t = np.arange(0.0, 3.0, 1.0 / 1000.0)
    f = _synthetic_train(t, [(0.3, 0.02, 0.09, 12.0e6), (1.2, 0.025, 0.11, 9.0e6),
                             (2.2, 0.018, 0.07, 10.0e6)])
    result = pipeline.assess(t, f, pipeline.ShipCase(), integration_rate_hz=2000.0)
    crit = result["criteria"]
    for key in ("a_in_band_10_30_pct", "b_in_band_10_50_ms", "c_in_band_20_40_pct",
                "d_in_band_15_35_pct", "e_residual_below_1e_6",
                "f_negative_control_red", "g_all_below_20_pct"):
        assert isinstance(crit[key], bool)
    assert crit["e_residual_below_1e_6"] is True
    assert crit["f_negative_control_red"] is True
    assert result["load"]["pulse_count"] == 3
    assert result["modes"]["wet_freqs_Hz"][0] < result["modes"]["dry_freqs_Hz"][0]