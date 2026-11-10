"""Tests of the dynamic (wave-propagation) granular verification arm.

The point of these tests is that every estimator is checked against a case with a
*known answer* (a synthetic ring-down, a synthetic resonance curve, a lattice with
a computable gap), so a green suite means the measurement chain can be trusted
before it is used on a DEM specimen.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.granular_wave import dynamic as dyn
from tensordem.granular_wave.engine import GranularPacking, GranularWaveConfig


# --------------------------------------------------------------------------
# initial state
# --------------------------------------------------------------------------
def test_loose_lattice_has_positive_clearance():
    cfg = GranularWaveConfig(nx=3, ny=3, nz=6)
    init = dyn.build_loose_lattice(cfg, gap=1.06, jitter=0.05)
    assert init["clearance_m"] > 0.0
    pos = init["positions"]
    radius = init["radii"]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    rs = (radius[:, None] + radius[None, :]).numpy()
    np.fill_diagonal(d, np.inf)
    assert float((rs - d).max()) < 0.0, "initial lattice must not interpenetrate"


def test_loose_lattice_rejects_overlapping_gap():
    cfg = GranularWaveConfig(nx=3, ny=3, nz=4)
    with pytest.raises(ValueError):
        dyn.build_loose_lattice(cfg, gap=1.0, jitter=0.05)
    with pytest.raises(ValueError):
        dyn.build_loose_lattice(cfg, gap=1.0001, jitter=0.10)


def test_loose_lattice_has_no_contacts_sim():
    cfg = GranularWaveConfig(nx=3, ny=3, nz=4)
    init = dyn.build_loose_lattice(cfg)
    sim = GranularPacking(cfg, init=init)
    assert sim.contact_network()["coordination_number"] == 0.0
    assert float(sim.pressures().max()) == 0.0


def test_default_engine_lattice_would_overlap():
    """The engine's own start state is *not* contact free (that is the defect)."""
    cfg = GranularWaveConfig(nx=4, ny=4, nz=6)
    sim = GranularPacking(cfg)
    pos = sim.pos
    radius = sim.radius
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    rs = (radius[:, None] + radius[None, :]).numpy()
    np.fill_diagonal(d, np.inf)
    overlap = float((rs - d).max())
    assert overlap > 0.0, "expected the shipped lattice to interpenetrate"


# --------------------------------------------------------------------------
# timestep
# --------------------------------------------------------------------------
def test_safe_timestep_scales_as_fourth_root_of_overlap():
    cfg = GranularWaveConfig(nx=3, ny=3, nz=4)
    sim = GranularPacking(cfg)
    dt1 = dyn.safe_timestep(sim, max_overlap_fraction=0.01)["dt_s"]
    dt4 = dyn.safe_timestep(sim, max_overlap_fraction=0.04)["dt_s"]
    # k ~ sqrt(delta) so the period scales as delta^(-1/4)
    assert dt1 / dt4 == pytest.approx(4.0 ** 0.25, rel=1e-9)


def test_safe_timestep_is_shorter_than_a_quarter_period():
    cfg = GranularWaveConfig(nx=3, ny=3, nz=4)
    sim = GranularPacking(cfg)
    info = dyn.safe_timestep(sim, max_overlap_fraction=0.01, factor=0.10)
    assert info["dt_s"] > 0.0
    assert info["k_max_N_m"] > 0.0
    assert info["m_min_kg"] > 0.0
    period = 2.0 * math.pi * math.sqrt(info["m_min_kg"] / info["k_max_N_m"])
    assert info["dt_s"] == pytest.approx(0.10 * period, rel=1e-12)


# --------------------------------------------------------------------------
# signal estimators against known answers
# --------------------------------------------------------------------------
def _synthetic_ringdown(frequency=250.0, q=12.0, dt=1.0e-5, duration=0.02, amp=1e-3):
    t = np.arange(0.0, duration, dt)
    envelope = amp * np.exp(-math.pi * frequency * t / q)
    return t, envelope * np.sin(2.0 * math.pi * frequency * t), dt


def test_analytic_envelope_has_no_wraparound():
    """A zero-padded envelope must not leak the record end into t = 0."""
    t, sig, dt = _synthetic_ringdown()
    env = dyn._analytic_envelope(sig)
    assert env[0] < 0.75 * env.max()
    # the envelope of a decaying sine is the exponential itself, up to the ripple
    # of a Hilbert envelope over a short record; the *decay rate* is what the Q
    # estimator uses and it must come out within a few percent
    ref = 1e-3 * np.exp(-math.pi * 250.0 * t / 12.0)
    assert np.abs(env / ref).max() < 1.5
    q = dyn.decay_q(env, dt, 250.0)
    assert q["gamma_1_s"] == pytest.approx(math.pi * 250.0 / 12.0, rel=0.05)


def test_zero_crossing_frequency_recovers_known_value():
    _, sig, dt = _synthetic_ringdown(frequency=250.0, q=20.0)
    band = dyn._bandpass(sig, dt, 0.75 * 250.0, 1.25 * 250.0)
    out = dyn.frequency_from_zero_crossings(band, dt)
    assert out["frequency_hz"] == pytest.approx(250.0, rel=2e-3)
    assert out["n_crossings"] > 5


def test_spectral_peak_recovers_known_value():
    _, sig, dt = _synthetic_ringdown(frequency=180.0, q=20.0)
    out = dyn.spectral_peak(sig, dt, f_lo=100.0, f_hi=300.0)
    assert out["peak_hz"] == pytest.approx(180.0, rel=5e-3)


def test_decay_q_recovers_known_q():
    t, sig, dt = _synthetic_ringdown(frequency=250.0, q=12.0)
    out = dyn.decay_q(dyn._analytic_envelope(sig), dt, 250.0)
    assert out["q"] == pytest.approx(12.0, rel=0.05)
    assert out["tau_s"] > 0.0
    assert out["gamma_1_s"] > 0.0


def test_lorentzian_fit_recovers_f0_and_q():
    f0, q = 250.0, 15.0
    fs = np.linspace(0.6 * f0, 1.4 * f0, 9)
    amps = 1.0 / np.sqrt((f0 ** 2 - fs ** 2) ** 2 + (f0 * fs / q) ** 2)
    fit = dyn.lorentzian_fit(fs, amps, f_bounds=(0.55 * f0, 1.45 * f0))
    assert fit["f0_hz"] == pytest.approx(f0, rel=5e-3)
    assert fit["q_bandwidth"] == pytest.approx(q, rel=0.05)
    assert fit["half_power_width_hz"] == pytest.approx(f0 / q, rel=0.05)


def test_lorentzian_fit_rejects_too_few_points():
    fit = dyn.lorentzian_fit([100.0, 200.0], [1.0, 0.5], f_bounds=(90.0, 210.0))
    assert math.isnan(fit["f0_hz"])


# --------------------------------------------------------------------------
# receivers
# --------------------------------------------------------------------------
def test_receiver_layout_covers_the_whole_cross_section():
    cfg = GranularWaveConfig(nx=4, ny=4, nz=12)
    sim = GranularPacking(cfg)
    layout = dyn.receiver_layout(sim, (0.30, 0.80), thickness_factor=3.0)
    n_per_slab = [len(idx) for idx in layout["slabs"]]
    # 16 particles per layer, a 3-radii-thick slab holds 1-2 layers
    assert all(10 <= n <= 40 for n in n_per_slab), n_per_slab
    assert layout["centres"][0] < layout["centres"][1]
    assert layout["distance"] > 0.0


# --------------------------------------------------------------------------
# preparation protocol
# --------------------------------------------------------------------------
def test_densify_raises_the_equilibrium_pressure():
    cfg = GranularWaveConfig(nx=4, ny=4, nz=10)
    init = dyn.build_loose_lattice(cfg)
    sim = GranularPacking(cfg, init=init)
    sim.dt = dyn.safe_timestep(sim)["dt_s"]
    dcfg = dyn.DynamicConfig(hold_steps=60, hold_window=15, max_jam_strain=0.26)
    loose = dyn.prepare_loose(sim, dcfg)
    assert loose["pressure_Pa"] == 0.0
    jam = dyn.densify(sim, 5.0e4, dcfg, factor=1.4, check_every=5)
    assert jam["applied_strain"] > 0.0
    assert jam["solid_fraction"] > 0.45
    assert jam["equilibrium_pressure_Pa"] > 0.0
    assert sim.contact_network()["coordination_number"] > 0.0


def test_equilibrate_reports_a_drifting_history_for_a_transient_state():
    cfg = GranularWaveConfig(nx=4, ny=4, nz=10)
    init = dyn.build_loose_lattice(cfg)
    sim = GranularPacking(cfg, init=init)
    sim.dt = dyn.safe_timestep(sim)["dt_s"]
    dcfg = dyn.DynamicConfig(hold_steps=60, hold_window=15, max_jam_strain=0.26)
    dyn.prepare_loose(sim, dcfg)
    dyn.densify(sim, 5.0e4, dcfg, factor=1.4, check_every=5)
    eq = dyn.equilibrate(sim, dcfg, max_blocks=4, tol=0.01)
    assert eq["blocks"] >= 1
    assert len(eq["history_Pa"]) == eq["blocks"]
    assert eq["pressure_Pa"] > 0.0


def test_ringdown_needs_a_positive_length_and_returns_finite_numbers():
    cfg = GranularWaveConfig(nx=4, ny=4, nz=10)
    init = dyn.build_loose_lattice(cfg)
    sim = GranularPacking(cfg, init=init)
    sim.dt = dyn.safe_timestep(sim)["dt_s"]
    dcfg = dyn.DynamicConfig(hold_steps=60, hold_window=15, max_jam_strain=0.26,
                             ringdown_cycles=3, sweep_points=3,
                             sweep_warmup_cycles=1, sweep_measure_cycles=1)
    dyn.prepare_loose(sim, dcfg)
    dyn.densify(sim, 5.0e4, dcfg, factor=1.4, check_every=5)
    ring = dyn.ringdown_velocity(sim, mode="P", v_ref=200.0, cfg=dcfg)
    assert ring["column_length_m"] > 0.0
    assert np.isfinite(ring["speed_m_s"])
    assert ring["frequency_hz"] > 0.0
    assert ring["n_steps"] > 0
    # the measured mode frequency must reproduce the speed it is converted from
    assert ring["speed_m_s"] == pytest.approx(2.0 * ring["column_length_m"]
                                              * ring["frequency_hz"], rel=1e-9)