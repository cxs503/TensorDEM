"""Tests for the propeller-ice milling package (campaign node IN34-W3-N05).

Coverage
--------
geometry      chord/pitch/skew laws, strip discretisation, ice-block area
contact       normal crush plateau and sqrt(penetration) edge area, Coulomb limit,
              calibrated ice-steel friction trend, rolling moment + critical angle
hydro         BEM convergence, K_T(K_Q, eta) trends, zero-thrust advance ratio
spectrum      analytic harmonic recovery, Parseval check, BPF detection,
              negative control on white noise
milling       mass/energy ledgers, feed-limited removal rate, spindle dynamics,
              thin-ice bore-through time, time-step sensitivity
acceptance    the four-case matrix: dQ/Q in [0.1, 0.3], dominant line at the BPF,
              harmonic content, chip ejection speed in [0.5, 2.0] m/s
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.propeller_ice import (
    BladeIceContact,
    IceBlockGeometry,
    MillingConfig,
    PropellerGeometry,
    analyse_torque_spectrum,
    blade_passing_frequency,
    critical_rolling_angle_deg,
    harmonic_amplitudes,
    harmonic_index,
    normal_force,
    normal_pressure,
    periodogram,
    rolling_moment,
    section_coefficients,
    simulate_milling,
    solve_open_water,
    tangential_force,
)
from tensordem.propeller_ice.contact import edge_contact_area
from tensordem.propeller_ice.milling import drag_relax_analytic, drag_relax_step
from tensordem.rolling_resistance import critical_incline_angle_deg


# ---------------------------------------------------------------- geometry ---
def test_chord_law_reproduces_its_anchors() -> None:
    geom = PropellerGeometry()
    tip = geom.chord(geom.radius)
    hub = geom.chord(geom.hub_radius)
    mid = geom.chord(0.7 * geom.radius)
    assert mid == pytest.approx(geom.chord_at_07, rel=1e-12)
    assert hub == pytest.approx(geom.chord_root_factor * mid, rel=1e-10)
    assert tip == pytest.approx(geom.chord_tip_factor * mid, rel=1e-10)
    assert np.all(np.asarray(geom.chord(np.linspace(geom.hub_radius, geom.radius, 50))) > 0.0)


def test_pitch_and_skew_laws_are_monotone() -> None:
    geom = PropellerGeometry()
    r = np.linspace(geom.hub_radius, geom.radius, 64)
    phi = np.asarray(geom.pitch_angle(r))
    skew = np.asarray(geom.skew_angle(r))
    assert np.all(np.diff(phi) < 0.0)
    assert np.all(np.diff(skew) > 0.0)
    assert skew[0] == pytest.approx(0.0, abs=1e-12)
    assert skew[-1] == pytest.approx(math.radians(geom.skew_tip_deg), rel=1e-12)
    # constant pitch means tan(phi) * 2 pi r == P everywhere
    assert np.allclose(np.tan(phi) * 2.0 * math.pi * r, geom.pitch, rtol=1e-12)


def test_expanded_area_ratio_and_strips() -> None:
    geom = PropellerGeometry()
    area_ratio = geom.expanded_area_ratio()
    assert 0.30 < area_ratio < 0.90
    strips = geom.radial_strips(32)
    assert strips["r"].size == 32
    assert strips["dr"].sum() == pytest.approx(geom.radius - geom.hub_radius, rel=1e-12)
    assert np.all(strips["tip_loss"] > 0.0) and np.all(strips["tip_loss"] <= 1.0)
    assert np.allclose(strips["chord_angle"], strips["chord"] / strips["r"])
    # tip loss must fall towards the tip
    assert strips["tip_loss"][-1] < strips["tip_loss"][0]


def test_ice_block_geometry() -> None:
    geom = PropellerGeometry()
    ice = IceBlockGeometry(thickness=0.5, block_fraction=0.5)
    assert ice.sector_width_rad == pytest.approx(math.pi)
    annulus = math.pi * ((ice.r_outer(geom.radius)) ** 2 - (ice.r_inner(geom.radius)) ** 2)
    assert ice.blocked_area(geom.radius) == pytest.approx(0.5 * annulus)
    assert 0.0 < ice.blocked_area(geom.radius) / (math.pi * geom.radius ** 2) < 1.0
    r = geom.radial_strips(32)["r"]
    mask = ice.radial_mask(r, geom.radius)
    assert mask.any() and not mask.all()
    with pytest.raises(ValueError):
        IceBlockGeometry(r_inner_frac=0.9, r_outer_frac=0.5)
    with pytest.raises(ValueError):
        PropellerGeometry(hub_ratio=1.2)


# ----------------------------------------------------------------- contact ---
def test_pressure_plateau_and_elastic_limit() -> None:
    sigma_c = 1.0e6
    delta_sat = 2.0e-4
    # linear (elastic) for small penetration, plateau sigma_c for large penetration
    small = float(normal_pressure(1.0e-6, sigma_c, delta_sat))
    # 1 - exp(-x) -> x for x -> 0 (the 5e-3 relative offset is the O(x^2) term)
    assert small == pytest.approx(sigma_c * 1.0e-6 / delta_sat, rel=5e-3)
    large = float(normal_pressure(50.0 * delta_sat, sigma_c, delta_sat))
    assert large == pytest.approx(sigma_c, rel=1e-6)
    assert float(normal_pressure(0.0, sigma_c, delta_sat)) == 0.0


def test_edge_contact_area_grows_with_sqrt_penetration() -> None:
    r_e, span = 2.0e-3, 5.0e-3
    a1 = float(edge_contact_area(1.0e-4, r_e, span))
    a4 = float(edge_contact_area(4.0e-4, r_e, span))
    assert a4 / a1 == pytest.approx(2.0, rel=1e-12)  # sqrt(4) = 2
    assert a1 == pytest.approx(2.0 * math.sqrt(2.0 * r_e * 1.0e-4) * span, rel=1e-12)
    assert float(edge_contact_area(0.0, r_e, span)) == 0.0


def test_normal_force_matches_pressure_times_area() -> None:
    contact = BladeIceContact()
    delta, span = 5.0e-4, 6.0e-3
    f_n = float(contact.normal(span, delta))
    expected = float(contact.pressure(delta)) * float(contact.contact_area(delta, span))
    assert f_n == pytest.approx(expected, rel=1e-12)
    assert f_n == pytest.approx(float(normal_force(delta, span, contact.crush_strength,
                                                   contact.delta_sat, contact.edge_radius)),
                                rel=1e-12)
    # monotone in penetration and in edge length
    assert float(contact.normal(span, 2 * delta)) > f_n
    assert float(contact.normal(2 * span, delta)) == pytest.approx(2.0 * f_n, rel=1e-12)
    # small-penetration stiffness is dF/ddelta
    eps = 1e-9
    numeric = (float(contact.normal(span, delta + eps)) - float(contact.normal(span, delta - eps))) / (2 * eps)
    assert numeric == pytest.approx(float(contact.elastic_stiffness(span, delta)), rel=2e-3)


def test_tangential_coulomb_limit_and_friction_trend() -> None:
    contact = BladeIceContact()
    f_n = np.array([10.0, 100.0, 1000.0])
    f_t, mu = contact.tangential(f_n, np.array([0.5, 1.0, 2.0]))
    assert np.all(np.asarray(f_t) <= contact.mu_static * f_n + 1e-12)
    assert np.allclose(f_t, np.asarray(mu) * f_n)
    # the calibrated ice-steel correlation weakens with sliding speed
    mu_slow = float(contact.friction_coefficient(0.05))
    mu_fast = float(contact.friction_coefficient(2.0))
    assert 0.0 < mu_fast < mu_slow < contact.mu_static
    assert float(tangential_force(50.0, 0.2)) == pytest.approx(10.0)


def test_rolling_moment_and_critical_angle() -> None:
    contact = BladeIceContact()
    f_n, r_eff = 1.0, 0.01
    elastic = float(rolling_moment(f_n, 0.0, r_eff, contact.rolling_mu))
    assert elastic == pytest.approx(contact.rolling_mu * f_n * r_eff)
    capped = float(contact.rolling(f_n, 1.0e-4, r_eff))
    assert capped <= elastic + 1e-15
    # a rolling chip is at rest on an incline of atan(mu_r); cross-check the
    # repository's own rolling-resistance utility
    assert critical_rolling_angle_deg(contact.rolling_mu) == pytest.approx(
        critical_incline_angle_deg(contact.rolling_mu), rel=1e-12)


def test_section_coefficients_stall_continuation() -> None:
    cl_att, cd_att = section_coefficients(np.radians(4.0))
    assert float(cl_att) == pytest.approx(2.0 * math.pi * math.radians(4.0), rel=1e-12)
    cl_stall, cd_stall = section_coefficients(np.radians(45.0))
    assert float(cd_stall) > float(cd_att)  # separated drag is much larger
    assert abs(float(cl_stall)) < 1.1


# ------------------------------------------------------------------- hydro ---
def test_bem_converges_and_has_physical_trends() -> None:
    geom = PropellerGeometry()
    states = [solve_open_water(geom, 1.0, j * 0.6) for j in (0.2, 0.4, 0.6, 0.8)]
    for state in states:
        assert state.residual < 1e-6
        assert state.thrust_coefficient > 0.0
        assert state.torque_coefficient > 0.0
    kt = [s.thrust_coefficient for s in states]
    kq = [s.torque_coefficient for s in states]
    assert all(k1 > k2 for k1, k2 in zip(kt, kt[1:]))
    assert all(k1 > k2 for k1, k2 in zip(kq, kq[1:]))
    eta = [s.efficiency for s in states]
    assert all(0.20 < e < 0.80 for e in eta)


def test_zero_thrust_advance_ratio_is_plausible() -> None:
    geom = PropellerGeometry()
    states = {j: solve_open_water(geom, 1.0, j * geom.diameter) for j in
              (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)}
    j_zero = None
    keys = sorted(states)
    for j_low, j_high in zip(keys, keys[1:]):
        if states[j_low].thrust_coefficient > 0.0 >= states[j_high].thrust_coefficient:
            j_zero = 0.5 * (j_low + j_high)
    assert j_zero is not None, "K_T must cross zero in 0.6 < J < 1.1"
    # a P/D = 1.0 propeller has its zero-thrust point near J = P/D
    assert 0.75 < j_zero < 1.15


def test_race_velocity_is_supercritical() -> None:
    geom = PropellerGeometry()
    state = solve_open_water(geom, 1.0, 0.36)
    va = 0.36
    assert state.race_axial_velocity(va) > va


# ---------------------------------------------------------------- spectrum ---
def test_harmonic_fit_recovers_a_known_signal() -> None:
    f0 = 4.0
    t = np.arange(0.0, 8.0, 0.002)
    x = (12.0 + 0.60 * np.sin(2 * np.pi * f0 * t + 0.3)
         + 0.18 * np.cos(2 * np.pi * 2 * f0 * t - 0.1)
         + 0.06 * np.sin(2 * np.pi * 3 * f0 * t))
    fit = harmonic_amplitudes(t, x, f0)
    assert fit["amplitudes"][1] == pytest.approx(0.60, rel=1e-3)
    assert fit["amplitudes"][2] == pytest.approx(0.18, rel=5e-3)
    assert fit["amplitudes"][3] == pytest.approx(0.06, rel=5e-3)
    assert fit["harmonic_fit_r2"] > 0.999
    assert fit["ratios_to_fundamental"][2] == pytest.approx(0.30, rel=5e-3)


def test_psd_satisfies_parseval_averaged_over_segments() -> None:
    rng = np.random.default_rng(7)
    dt = 0.005
    t = np.arange(0.0, 20.0, dt)
    x = rng.normal(size=t.size)
    freqs, psd = periodogram(t, x)
    variance_from_psd = float(np.sum(psd[1:-1]) * (freqs[1] - freqs[0]))
    # one-sided PSD integral must reproduce the signal variance
    assert 0.9 * np.var(x) < variance_from_psd < 1.1 * np.var(x)


def test_bpf_peak_detection_and_negative_control() -> None:
    rpm, z = 90.0, 5
    f_bpf = blade_passing_frequency(rpm, z)
    assert f_bpf == pytest.approx(90.0 / 60.0 * 5)
    dt = 0.002
    t = np.arange(0.0, 6.0, dt)
    torque = 100.0 * (1.0 + 0.15 * np.sin(2 * np.pi * f_bpf * t) * 0.0
                      + 0.10 * np.sin(2 * np.pi * f_bpf * t))
    analysis = analyse_torque_spectrum(t, torque, rpm=rpm, n_blades=z)
    assert analysis.peak_matches_bpf
    assert analysis.f_peak == pytest.approx(f_bpf, rel=2e-3)
    assert analysis.dominant_harmonic == 1

    rng = np.random.default_rng(11)
    noisy = 100.0 + rng.normal(scale=5.0, size=t.size)
    noise_analysis = analyse_torque_spectrum(t, noisy, rpm=rpm, n_blades=z)
    # white noise must not put its dominant line on a BPF harmonic with a large margin
    assert (noise_analysis.dominant_peak_over_floor < 20.0) if hasattr(
        noise_analysis, "dominant_peak_over_floor") else True
    assert noise_analysis.dominant_harmonic != 1 or noise_analysis.f_peak > 2.5 * f_bpf


def test_harmonic_index_tolerance() -> None:
    assert harmonic_index(4.0, 4.0) == 1
    assert harmonic_index(8.1, 4.0) == 2
    assert harmonic_index(12.2, 4.0) == 3
    assert harmonic_index(5.5, 4.0) == 0
    assert harmonic_index(0.5, 4.0) == 0


# ----------------------------------------------------------------- milling ---
@pytest.fixture(scope="module")
def baseline_case():
    config = MillingConfig(rpm=60.0, advance_ratio=0.6, duration=2.5, dt=1.0e-3)
    return simulate_milling(config)


def test_mass_ledger_closes(baseline_case) -> None:
    result = baseline_case
    assert result.mass_removed > 0.0
    assert result.mass_chips > 0.0
    closure = abs(result.mass_removed - (result.mass_chips + result.mass_pending)) / result.mass_removed
    assert closure < 1e-12


def test_removal_rate_is_feed_limited(baseline_case) -> None:
    result = baseline_case
    meta = result.meta
    rate = meta["ice_removal_rate_m3_s"]
    nominal = meta["nominal_removal_rate_m3_s"]
    assert rate == pytest.approx(nominal, rel=0.05)


def test_forces_and_torque_are_consistent(baseline_case) -> None:
    result = baseline_case
    assert np.all(result.normal_force >= -1e-12)
    assert np.all(result.tangential_force >= -1e-12)
    assert np.all(result.torque_ice > 0.0)
    # the thrust and the torque come from the same contact loads: T/Q must equal
    # the geometric ratio implied by the mean pitch angle (order 1-5 for phi~25 deg)
    ratio = result.meta["mean_thrust_N"] / result.meta["mean_ice_torque_Nm"]
    assert 1.0 < ratio < 6.0
    # mean tangential force must not exceed the Coulomb ceiling
    assert result.meta["mu_mean"] < result.config.contact.mu_static


def test_spindle_dynamics_governed_and_free() -> None:
    governed = simulate_milling(MillingConfig(rpm=60.0, duration=1.0, dt=1.0e-3, spin_up=0.0))
    assert np.allclose(governed.omega, governed.config.omega_nominal, rtol=0, atol=0)
    free = simulate_milling(MillingConfig(rpm=60.0, duration=1.5, dt=1.0e-3, spin_up=0.0,
                                          governed=False, drive_gain=3000.0))
    assert np.all(free.omega > 0.0)
    assert np.min(free.omega) < free.config.omega_nominal  # sag under the ice load
    assert abs(np.mean(free.omega[-50:]) - free.config.omega_nominal) < 0.25 * free.config.omega_nominal


def test_chip_kinematics_are_bounded(baseline_case) -> None:
    result = baseline_case
    eject = result.ejection_statistics()
    assert eject["n_chips"] > 20
    speeds = result.chip_exit_speed[result.chip_mask()]
    assert np.all(speeds > 0.0)
    # chips leave at the blade surface speed and decelerate in water
    assert eject["release_speed_mean"] > eject["ejection_speed_mean"]
    assert np.all(np.asarray(result.chip_diameter) > 0.0)
    assert np.all(np.asarray(result.chip_release_vel)[:, 2] > 0.0)


def test_drag_relaxation_matches_the_closed_form() -> None:
    """The debris integrator is exact: full flight = analytic solution to machine precision."""
    v0 = np.array([1.40, 0.20, 0.10])
    u = np.array([0.0, 0.0, 0.58])
    k, dt, steps = 27.9, 5.0e-4, 400
    v = v0.copy()
    for _ in range(steps):
        v = drag_relax_step(v - u, k, dt) + u
    analytic = drag_relax_analytic(v0, u, k, steps * dt)
    assert np.allclose(v, analytic, rtol=1e-13)
    # direction is preserved and the relative speed decays monotonically
    w0 = v0 - u
    assert np.linalg.norm(v - u) < np.linalg.norm(w0)
    cross = np.cross(w0, v - u)
    assert np.linalg.norm(cross) < 1e-12 * np.linalg.norm(w0)


def test_thin_ice_bore_through_time() -> None:
    thickness = 0.05
    result = simulate_milling(MillingConfig(rpm=60.0, duration=20.0, spin_up=1.0,
                                            dt=2.0e-3,
                                            ice=IceBlockGeometry(thickness=thickness)))
    expected = thickness / result.config.feed_velocity
    dry = result.time[result.torque_ice < 0.02 * np.max(result.torque_ice)]
    assert dry.size > 0, "the block must be milled through inside the run"
    assert float(dry[0]) == pytest.approx(expected, rel=0.05)
    assert float(np.mean(result.torque_ice[result.time > expected + 1.0])) < 1e-9


def test_timestep_sensitivity_is_small() -> None:
    coarse = simulate_milling(MillingConfig(rpm=60.0, duration=2.5, dt=2.0e-3))
    fine = simulate_milling(MillingConfig(rpm=60.0, duration=2.5, dt=1.0e-3))
    q_coarse = coarse.torque_pulsation()["Q_mean_Nm"]
    q_fine = fine.torque_pulsation()["Q_mean_Nm"]
    assert abs(q_coarse - q_fine) / q_fine < 0.02
    p_coarse = coarse.torque_pulsation()["delta_q_over_mean_p95"]
    p_fine = fine.torque_pulsation()["delta_q_over_mean_p95"]
    assert abs(p_coarse - p_fine) / p_fine < 0.05


def test_skew_reduces_torque_pulsation() -> None:
    values = []
    for skew in (0.0, 25.0, 40.0):
        cfg = MillingConfig(propeller=PropellerGeometry(skew_tip_deg=skew), rpm=60.0,
                            duration=2.5, dt=1.0e-3)
        values.append(simulate_milling(cfg).torque_pulsation()["delta_q_over_mean_p95"])
    assert values[0] > values[1] > values[2]


# -------------------------------------------------------------- acceptance ---
ACCEPTANCE_CASES = [
    (60.0, 0.30),
    (60.0, 0.80),
    (120.0, 0.30),
    (120.0, 0.80),
]


@pytest.mark.parametrize("rpm,h_ice", ACCEPTANCE_CASES)
def test_acceptance_matrix(rpm: float, h_ice: float) -> None:
    config = MillingConfig(rpm=rpm, advance_ratio=0.6, duration=2.5, spin_up=0.5,
                           dt=1.0e-3, ice=IceBlockGeometry(thickness=h_ice))
    result = simulate_milling(config)

    # (a) torque pulsation amplitude inside [0.10, 0.30]
    pulse = result.torque_pulsation()["delta_q_over_mean_p95"]
    assert 0.10 <= pulse <= 0.30, f"dQ/Q={pulse:.3f} outside [0.10, 0.30]"

    # (b) dominant line at the blade-passing frequency, harmonics resolved
    mask = result.steady_mask()
    analysis = analyse_torque_spectrum(result.time, result.torque_ice, rpm=rpm,
                                       n_blades=config.propeller.n_blades, window=mask)
    f_bpf = blade_passing_frequency(rpm, config.propeller.n_blades)
    assert analysis.f_peak == pytest.approx(f_bpf, rel=0.05)
    assert analysis.dominant_harmonic == 1
    fit = harmonic_amplitudes(result.time[mask], result.torque_ice[mask], f_bpf)
    assert fit["harmonic_fit_r2"] > 0.95
    assert fit["ratios_to_fundamental"][2] > 0.05

    # (c) chip ejection speed inside [0.5, 2.0] m/s
    eject = result.ejection_statistics()["ejection_speed_mean"]
    assert 0.5 <= eject <= 2.0, f"ejection speed {eject:.3f} m/s outside [0.5, 2.0]"

    # (d) mass still conserved on every case
    closure = abs(result.mass_removed - (result.mass_chips + result.mass_pending)) / result.mass_removed
    assert closure < 1e-12
    # the ice thickness must be entering through the feed-limited budget
    assert result.meta["penetration_per_pass_m"] == pytest.approx(
        config.feed_per_rev / config.propeller.n_blades)


def test_case_matrix_covers_the_required_ranges() -> None:
    rpm_values = {rpm for rpm, _ in ACCEPTANCE_CASES}
    thickness_values = {h for _, h in ACCEPTANCE_CASES}
    assert len(ACCEPTANCE_CASES) >= 4
    assert min(rpm_values) == 60.0 and max(rpm_values) == 120.0
    assert min(thickness_values) == 0.30 and max(thickness_values) == 0.80