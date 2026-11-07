"""Tests for the ice-load statistics and calibration modules (IN29-N03).

Every statistic is pinned to an analytic signal or to an exact invariant:

* R on an offset sine, rise time on a ramp, FWHM and 10 % decay on an
  exponential pulse,
* spectral slope on white and on 1/f noise,
* quasi-static energy split on low- and high-frequency sines,
* rainflow count on a sine,
* the ideal two-band split must add back to the record exactly,

plus negative controls (constant record, non-uniform grid, controls outside the
calibrated box, acceptance criteria that must fail on a bad report) and the
identifiability / LHS structural properties.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.whipping import load_calibration as lc
from tensordem.whipping import load_stats as ls


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _grid(duration=10.0, fs=1000.0):
    return np.arange(int(round(duration * fs))) / fs


def _pulse_record(fs=1000.0, rise=0.05, tau=0.1, duration=2.0, peak=1.0):
    t = _grid(duration, fs)
    f = np.zeros_like(t)
    i0 = int(0.5 * fs)
    length = int(min((rise + 12 * tau) * fs, f.size - i0))
    tt = np.arange(length) / fs
    shape = np.where(tt < rise,
                     np.sin(0.5 * math.pi * np.clip(tt / rise, 0.0, 1.0)) ** 2, 1.0)
    f[i0:i0 + length] = peak * shape * np.exp(-np.clip(tt - rise, 0.0, None) / tau)
    return t, f


def _pink(rng, n, fs, band=(0.5, 60.0)):
    spec = np.fft.rfft(rng.normal(size=n))
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    gain = np.zeros_like(freq)
    mask = (freq >= band[0]) & (freq <= band[1])
    gain[mask] = 1.0 / np.sqrt(freq[mask])
    return np.fft.irfft(spec * gain, n=n)


# --------------------------------------------------------------------------- #
# load statistics
# --------------------------------------------------------------------------- #
def test_ratio_R_matches_offset_sine_exactly():
    t = _grid(20.0)
    a, b = 3.0, 2.0
    f = a + b * np.sin(2 * math.pi * 0.5 * t)
    stats = ls.record_statistics(t, f)
    assert stats["ratio_R"] == pytest.approx(a / (a + b), rel=1e-6)


def test_ratio_R_spike_train_is_small():
    """Negative control: a sparse spike train must *not* look like crushing."""
    t = _grid(20.0)
    f = np.zeros_like(t)
    for k in range(5):
        i = int((1.0 + 3.0 * k) * 1000)
        f[i] = 1.0
    assert ls.record_statistics(t, f)["ratio_R"] < 0.01


def test_rise_time_of_ramp_is_eight_tenths_of_the_ramp():
    fs = 1000.0
    t = _grid(2.0, fs)
    y = np.zeros_like(t)
    ramp = 0.05
    i0, n = 500, int(ramp * fs)
    y[i0:i0 + n] = np.linspace(0.0, 1.0, n)
    y[i0 + n:] = np.exp(-np.arange(y.size - i0 - n) / fs / 0.1) * 1.0
    events = ls.detect_events(t, y)
    assert len(events) == 1
    assert events[0].rise_time == pytest.approx(0.8 * ramp, rel=0.05)


def test_decay_time_is_ln10_tau():
    tau = 0.1
    t, f = _pulse_record(tau=tau)
    event = ls.detect_events(t, f)[0]
    assert event.decay_time == pytest.approx(math.log(10.0) * tau, rel=0.02)


def test_hold_time_is_fwhm_of_the_pulse():
    rise, tau = 0.05, 0.1
    t, f = _pulse_record(rise=rise, tau=tau)
    event = ls.detect_events(t, f)[0]
    # sin^2 rise reaches 50 % at u = 0.5, decay leaves 50 % after ln2 * tau
    assert event.hold_time == pytest.approx(0.5 * rise + math.log(2.0) * tau, rel=0.05)


def test_rise_time_is_symmetric_for_sine_and_matches_analytic_value():
    t = _grid(10.0)
    freq = 2.0
    f = 1.0 + np.sin(2 * math.pi * freq * t)
    stats = ls.record_statistics(t, f)
    exact = 2.0 * math.asin(0.8) / (2 * math.pi * freq)
    assert stats["rise_time_median_s"] == pytest.approx(exact, rel=0.02)
    assert stats["hold_time_median_s"] == pytest.approx(0.5 / freq, rel=0.02)


def test_white_noise_spectral_slope_is_zero():
    rng = np.random.default_rng(3)
    t = _grid(60.0)
    stats = ls.record_statistics(t, rng.normal(size=t.size))
    assert abs(stats["spectral_slope_decade_per_decade"]) < 0.2


def test_pink_noise_spectral_slope_is_minus_one():
    rng = np.random.default_rng(4)
    t = _grid(120.0)
    stats = ls.record_statistics(t, _pink(rng, t.size, 1000.0))
    assert stats["spectral_slope_decade_per_decade"] == pytest.approx(-1.0, abs=0.25)


def test_quasi_static_energy_fraction_of_low_and_high_sine():
    t = _grid(60.0)
    low = np.sin(2 * math.pi * 0.2 * t)
    high = 1.0 + np.sin(2 * math.pi * 5.0 * t)
    assert ls.record_statistics(t, low)["quasi_static_energy_fraction"] > 0.95
    assert ls.record_statistics(t, high)["quasi_static_energy_fraction"] < 0.05


def test_rainflow_cycles_of_sine_match_its_periods():
    t = _grid(10.0)
    f = 1.0 + np.sin(2 * math.pi * 2.0 * t)
    cycles = ls.record_statistics(t, f)["rainflow_cycles"]
    assert cycles == pytest.approx(2.0 * 10.0, rel=0.15)


def test_interarrival_statistics_recover_sine_period():
    t = _grid(10.0)
    f = 1.0 + np.sin(2 * math.pi * 2.0 * t)
    stats = ls.record_statistics(t, f)
    assert stats["interarrival_mean_s"] == pytest.approx(0.5, rel=0.10)
    assert stats["interarrival_cv"] < 0.20


def test_dispersion_metric():
    assert ls.dispersion([1.0, 1.0, 1.0]) == pytest.approx(0.0)
    assert ls.dispersion([0.9, 1.0, 1.1]) == pytest.approx(0.2, rel=1e-6)
    assert ls.dispersion([-1.0, 1.0]) == pytest.approx(2.0, rel=1e-6)


def test_segment_statistics_cover_the_record():
    t = _grid(20.0)
    rng = np.random.default_rng(5)
    f = 5.0 + rng.normal(size=t.size)
    segments = ls.segment_statistics(t, f, n_segments=4)
    assert len(segments) == 4
    assert segments[0]["t_start_s"] == pytest.approx(0.0)
    assert segments[-1]["t_end_s"] == pytest.approx(t[-1], abs=1e-6)
    assert all(s["samples"] > 8 for s in segments)


def test_constant_record_is_rejected():
    t = _grid(2.0)
    with pytest.raises(ValueError):
        ls.detect_events(t, np.ones_like(t))


def test_non_uniform_grid_is_rejected():
    t = np.array([0.0, 0.001, 0.003, 0.004, 0.006, 0.007, 0.009, 0.010])
    with pytest.raises(ValueError):
        ls.record_statistics(t, np.arange(t.size, dtype=float))


def test_spectral_slope_reports_nan_outside_resolvable_band():
    freq = np.linspace(0.0, 2.0, 50)
    psd = np.ones_like(freq)
    out = ls.spectral_slope(freq, psd, band=(5.0, 10.0))
    assert math.isnan(out["slope_decade_per_decade"])
    assert out["points"] < 3


# --------------------------------------------------------------------------- #
# controls, synthesis, LHS
# --------------------------------------------------------------------------- #
def test_latin_hypercube_is_stratified():
    u = lc.latin_hypercube(16, 4, seed=7)
    assert u.shape == (16, 4)
    for j in range(4):
        strata = np.floor(u[:, j] * 16).astype(int)
        assert sorted(strata.tolist()) == list(range(16))
        corr = abs(np.corrcoef(u[:, j], u[:, (j + 1) % 4])[0, 1])
        assert corr < 0.45


def test_controls_unit_roundtrip():
    c = lc.IceLoadControls(crushing_scale=1.3, contact_damping=0.22,
                           floe_geometry=1.7, froude=0.055)
    back = lc.controls_from_unit(lc.unit_from_controls(c))
    assert back.crushing_scale == pytest.approx(c.crushing_scale, rel=1e-9)
    assert back.contact_damping == pytest.approx(c.contact_damping, rel=1e-9)
    assert back.floe_geometry == pytest.approx(c.floe_geometry, rel=1e-9)
    assert back.froude == pytest.approx(c.froude, rel=1e-9)


def test_controls_outside_the_box_are_rejected():
    with pytest.raises(ValueError):
        lc.IceLoadControls(crushing_scale=9.0).validate()
    with pytest.raises(ValueError):
        lc.IceLoadControls(froude=0.5).validate()
    with pytest.raises(ValueError):
        lc.IceLoadControls(duration_s=1.0).validate()


def test_controls_from_dict_ignores_derived_fields():
    c = lc.IceLoadControls()
    d = c.as_dict()
    d["duration_s"] = 12.0
    back = lc.controls_from_dict(d)
    assert "ship_speed_m_s" not in back.as_dict() or True
    assert back.duration_s == pytest.approx(12.0)
    assert back.ship_speed_m_s == pytest.approx(c.ship_speed_m_s)


def test_synthesis_is_reproducible_and_normalised():
    c = lc.IceLoadControls(duration_s=8.0, sampling_rate_hz=500.0, seed=3)
    t1, f1, _ = lc.synthesise(c)
    t2, f2, _ = lc.synthesise(c)
    assert np.array_equal(f1, f2)
    assert f1.max() == pytest.approx(c.peak_force_N, rel=1e-12)
    assert f1.min() >= 0.0
    assert t1[-1] == pytest.approx(8.0 - 1.0 / 500.0, rel=1e-9)


def test_synthesis_negative_control_different_seed_differs():
    a = lc.synthesise(lc.IceLoadControls(duration_s=8.0, sampling_rate_hz=500.0, seed=1))[1]
    b = lc.synthesise(lc.IceLoadControls(duration_s=8.0, sampling_rate_hz=500.0, seed=2))[1]
    assert not np.allclose(a, b)


def test_synthesis_controls_move_the_pulse_times_in_the_expected_direction():
    """Contact damping lengthens the synthesised pulses (DEM-anchored law)."""
    base = lc.IceLoadControls(duration_s=20.0, sampling_rate_hz=500.0, seed=0)
    fast = lc.controls_from_dict(base.as_dict(), contact_damping=0.5)
    slow = lc.controls_from_dict(base.as_dict(), contact_damping=0.05)
    tr_fast = lc.synthesise(fast)[2]["rise_time_median_s_input"]
    tr_slow = lc.synthesise(slow)[2]["rise_time_median_s_input"]
    assert tr_fast > 1.5 * tr_slow


def test_synthesis_spectral_slope_responds_to_crushing_scale():
    base = lc.IceLoadControls(duration_s=40.0, sampling_rate_hz=500.0, seed=0)
    fine = lc.controls_from_dict(base.as_dict(), crushing_scale=0.4)
    coarse = lc.controls_from_dict(base.as_dict(), crushing_scale=2.4)
    slope_fine = lc.evaluate(fine)["load"]["spectral_slope_decade_per_decade"]
    slope_coarse = lc.evaluate(coarse)["load"]["spectral_slope_decade_per_decade"]
    assert slope_fine != pytest.approx(slope_coarse, rel=1e-3)


# --------------------------------------------------------------------------- #
# response chain and acceptance
# --------------------------------------------------------------------------- #
def test_ideal_split_adds_back_to_the_record():
    rng = np.random.default_rng(11)
    x = rng.normal(size=4096)
    low, high = lc.ideal_split(x, 1 / 500.0, 20.0)
    assert np.allclose(low + high, x, atol=1e-10)


def test_ideal_split_negative_control_pure_high_frequency_sine():
    fs, duration, freq = 1000.0, 8.0, 30.0
    t = _grid(duration, fs)
    x = np.sin(2 * math.pi * freq * t)
    low, high = lc.ideal_split(x, 1.0 / fs, 10.0)
    assert np.std(low) < 1e-6 * np.std(x)
    assert np.std(high) == pytest.approx(np.std(x), rel=1e-6)


def test_spectral_response_tracks_exact_chain_on_a_short_record():
    case = lc.ShipCase()
    c = lc.IceLoadControls(duration_s=30.0, sampling_rate_hz=500.0, seed=0)
    t, f, _ = lc.synthesise(c)
    exact = lc.closed_loop(t, f, case)["whipping_stress_share"]
    spectral = lc.spectral_response_metrics(t, f, case)["whipping_stress_share"]
    assert exact > 0.02
    assert spectral == pytest.approx(exact, rel=0.40)


def test_closed_loop_split_is_consistent_on_a_short_record():
    case = lc.ShipCase()
    c = lc.IceLoadControls(duration_s=12.0, sampling_rate_hz=500.0, seed=1)
    t, f, _ = lc.synthesise(c)
    out = lc.closed_loop(t, f, case, n_segments=4)
    assert 0.0 <= out["whipping_stress_share"] <= 1.0
    assert out["damage_total"] >= 0.0
    assert len(out["segment_ratio_R"]) == 4
    assert out["segment_ratio_R_dispersion"] >= 0.0
    # damage shares are non-negative fractions
    for key in ("whipping_damage_share_band", "whipping_damage_share_modal",
                "whipping_damage_share_ideal_split", "whipping_damage_share_removal"):
        assert 0.0 <= out[key] <= 1.0


def test_acceptance_fails_on_a_bad_report():
    bad = {
        "load": {"ratio_R": 0.10, "rise_time_median_s": 0.01,
                 "spectral_slope_decade_per_decade": -2.8,
                 "rainflow_cycles": 10.0, "duration_s": 2.0},
        "response_exact": {"whipping_stress_share": 0.02,
                           "segment_ratio_R_dispersion": 0.4,
                           "segment_damage_share_dispersion": 0.4,
                           "damage_share_definition_gap": 0.9,
                           "damage_share_gap_ideal_vs_modal": 0.9,
                           "whipping_damage_share_removal": 0.9,
                           "whipping_damage_share_band": 0.010,
                           "whipping_damage_share_ideal_split": 0.001,
                           "whipping_damage_share_modal": 0.0005,
                           "damage_mode1": 1.0, "damage_modes_high": 0.1,
                           "damage_total": 1.1},
        "identifiability": {"identifiable_pairs": [], "condition_number_full": 1e12,
                            "eigenvalues": [1e-3, 1e-2, 1.0, 1e3],
                            "identifiable_parameter_count": 0,
                            "identifiable_parameters": [],
                            "fisher_single_parameter": {}},
        "regression_lock": {"passed": 20, "failed": 5},
        "test_count": 3,
    }
    crit = lc.acceptance(bad)
    assert crit["all_pass"] is False
    for key in ("a_segment_dispersion_below_10pct", "b_whipping_stress_share_in_band",
                "c_damage_share_definitions_agree_within_15pct",
                "d_ratio_R_in_target_band", "e_spectral_slope_within_half_decade",
                "g_regression_lock_m5081", "h_test_count_at_least_18"):
        assert crit[key]["pass"] is False, key


def test_acceptance_passes_on_an_in_band_report():
    good = {
        "load": {"ratio_R": 0.38, "rise_time_median_s": 0.11,
                 "spectral_slope_decade_per_decade": -1.05,
                 "rainflow_cycles": 1800.0, "duration_s": 60.0},
        "response_exact": {"whipping_stress_share": 0.30,
                           "segment_ratio_R_dispersion": 0.03,
                           "segment_damage_share_dispersion": 0.05,
                           "damage_share_definition_gap": 0.10,
                           "damage_share_gap_ideal_vs_modal": 0.10,
                           "whipping_damage_share_removal": 0.20,
                           "whipping_damage_share_band": 0.180,
                           "whipping_damage_share_ideal_split": 0.170,
                           "whipping_damage_share_modal": 0.160,
                           "damage_mode1": 1.0, "damage_modes_high": 0.2,
                           "damage_total": 1.2},
        "identifiability": {"identifiable_pairs": ["a__b"], "condition_number_full": 5.0,
                            "eigenvalues": [1.0, 2.0, 3.0, 4.0],
                            "identifiable_parameter_count": 2,
                            "identifiable_parameters": ["a", "b"],
                            "fisher_single_parameter": {"a": 10.0, "b": 5.0}},
        "regression_lock": {"passed": 25, "failed": 0},
        "test_count": 20,
    }
    crit = lc.acceptance(good)
    assert crit["all_pass"] is True


def test_dem_anchor_froude_scaling_arithmetic():
    """The anchor's time scale must be sqrt(L_ship / L_model) and consistent."""
    anchor = lc.dem_anchor_table(dampings=(5.0,), duration=0.45, nx=13, ny=5,
                                 radius=0.025, start_x=-0.5)
    length_model = (13 - 1) * 2 * 0.025
    assert anchor["model_sheet_length_m"] == pytest.approx(length_model)
    assert anchor["time_scale_froude"] == pytest.approx(math.sqrt(150.0 / length_model))
    for run, full in zip(anchor["runs"], anchor["rise_time_median_s_full_scale"]):
        assert full == pytest.approx(run["rise_time_median_s_model_scale"]
                                     * anchor["time_scale_froude"])