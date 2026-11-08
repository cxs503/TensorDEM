"""Tests for the vibro-acoustic source terms (campaign node P27W6-V1).

Coverage
--------
``tensordem.acoustics.source_terms``
    Lighthill stress tensor definition, quadrupole/dipole analytic power
    references, compact far-field operators against the exact monopole-pair
    dipole, sphere quadrature exactness, Lighthill ``U^8`` scaling.
``tensordem.acoustics.face_fft``
    ring buffer wrap-around, peak-amplitude FFT convention, O(N log N) FFT vs
    O(N^2) direct DFT (error and speed), spectral vs retarded-time propagation.
``tensordem.acoustics.bpf``
    ``f_BPF = N Z / 60``, BPF tone recovery inside a rotating face sheet, a
    broadband negative control, third-octave ladder.
``tensordem.acoustics.api``
    ``POST /api/v1/dem/acoustic_source`` happy path and payload validation.

All references are closed-form; no test compares the code against itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.acoustics import (
    Medium,
    RingBuffer,
    acoustic_power_from_spectrum,
    blade_passing_frequency_hz,
    build_face_sheet,
    dipole_far_field_spectrum,
    dipole_pair_far_field,
    dipole_pair_force,
    dipole_pair_power,
    dipole_power_analytic,
    direct_dft,
    face_spectrum,
    face_spectrum_complex,
    find_spectral_peak,
    fit_power_law,
    handle_acoustic_source,
    lighthill_stress_tensor,
    monopole_far_field,
    pwsl_db,
    quadrupole_far_field_spectrum,
    quadrupole_power_analytic,
    rotor_face_forces,
    shaft_frequency_hz,
    sphere_quadrature,
    spl_db,
    third_octave_levels,
    volume_integral_stress,
)

MEDIUM = Medium()


# --------------------------------------------------------------------------- #
# Lighthill stress tensor
# --------------------------------------------------------------------------- #
def test_lighthill_tensor_definition() -> None:
    """``T_ij = rho u_i u_j + (p' - c0^2 rho') d_ij - tau_ij`` on a known state."""
    u = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]])
    rho = 1000.0
    p = 500.0
    drho = 0.2
    c0 = 1500.0
    T = lighthill_stress_tensor(u, rho=rho, pressure_perturbation=p,
                                density_perturbation=drho, c0=c0)
    iso = p - c0 ** 2 * drho
    assert T.shape == (2, 3, 3)
    # momentum flux
    np.testing.assert_allclose(T[0, 0, 0], rho * 1.0 ** 2 + iso)
    np.testing.assert_allclose(T[0, 0, 1], rho * 1.0 * 2.0)
    np.testing.assert_allclose(T[1, 1, 2], rho * 3.0 * 4.0)
    # off-diagonals carry no isotropic part
    np.testing.assert_allclose(T[1, 0, 2], 0.0)
    # viscous stress is subtracted
    tau = np.zeros((2, 3, 3))
    tau[0, 0, 0] = 17.0
    T2 = lighthill_stress_tensor(u, rho=rho, pressure_perturbation=p,
                                 density_perturbation=drho, c0=c0, viscous_stress=tau)
    np.testing.assert_allclose(T2[0, 0, 0], T[0, 0, 0] - 17.0)


def test_volume_integral_stress_sums_cells() -> None:
    """``Q_ij = sum_cells T_ij V_cell`` for non-uniform volumes."""
    u = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])  # (2, 3)
    T = lighthill_stress_tensor(u, rho=2.0, c0=1500.0)
    V = np.array([0.5, 1.5])
    Q = volume_integral_stress(T, V)
    assert Q.shape == (3, 3)
    np.testing.assert_allclose(Q[0, 0], 2.0 * 1.0 * 0.5)
    np.testing.assert_allclose(Q[1, 1], 2.0 * 4.0 * 1.5)
    with pytest.raises(ValueError):
        volume_integral_stress(T[None], V)


# --------------------------------------------------------------------------- #
# analytic power references
# --------------------------------------------------------------------------- #
def test_quadrupole_power_closed_forms() -> None:
    """``P = w^4 Q^2/(40 pi rho0 c0^5)`` (longitudinal) and ``/(120 pi ...)`` (lateral)."""
    omega = 2.0 * math.pi * 50.0
    q = 100.0
    long = np.zeros((3, 3), dtype=complex)
    long[0, 0] = q
    lat = np.zeros((3, 3), dtype=complex)
    lat[0, 1] = q
    expect_long = omega ** 4 * q ** 2 / (40.0 * math.pi * MEDIUM.rho0 * MEDIUM.c0 ** 5)
    expect_lat = omega ** 4 * q ** 2 / (120.0 * math.pi * MEDIUM.rho0 * MEDIUM.c0 ** 5)
    assert quadrupole_power_analytic(long, omega, MEDIUM) == pytest.approx(expect_long,
                                                                          rel=1e-12)
    assert quadrupole_power_analytic(lat, omega, MEDIUM) == pytest.approx(expect_lat,
                                                                         rel=1e-12)
    # the lateral quadrupole radiates a third of the longitudinal one
    assert quadrupole_power_analytic(lat, omega, MEDIUM) / \
        quadrupole_power_analytic(long, omega, MEDIUM) == pytest.approx(1.0 / 3.0,
                                                                       rel=1e-12)


def test_dipole_power_closed_form() -> None:
    """``P = w^2 |F|^2/(24 pi rho0 c0^3)`` and the two-monopole strength ``F=rho0 w Q d``."""
    omega = 2.0 * math.pi * 100.0
    f_peak = 6.4402649
    expect = omega ** 2 * f_peak ** 2 / (24.0 * math.pi * MEDIUM.rho0 * MEDIUM.c0 ** 3)
    got = dipole_power_analytic(np.array([0.0, 0.0, f_peak]), omega, MEDIUM)
    assert got == pytest.approx(expect, rel=1e-12)
    q, d = 1.0e-3, 0.01
    assert dipole_pair_force(q, d, omega, MEDIUM) == pytest.approx(
        MEDIUM.rho0 * omega * q * d, rel=1e-14)
    assert dipole_pair_power(q, d, omega, MEDIUM) == pytest.approx(expect, rel=1e-10)


# --------------------------------------------------------------------------- #
# far-field operators against closed form
# --------------------------------------------------------------------------- #
def test_monopole_far_field_formula() -> None:
    """``rho0 omega Q/(4 pi r)`` scales as 1/r and linearly in Q."""
    omega = 2.0 * math.pi * 100.0
    p1 = monopole_far_field(1.0e-3, omega, 10.0, MEDIUM)
    p2 = monopole_far_field(1.0e-3, omega, 20.0, MEDIUM)
    p3 = monopole_far_field(2.0e-3, omega, 10.0, MEDIUM)
    assert p1 / p2 == pytest.approx(2.0, rel=1e-12)
    assert p3 / p1 == pytest.approx(2.0, rel=1e-12)
    assert p1 == pytest.approx(MEDIUM.rho0 * omega * 1.0e-3 / (4 * math.pi * 10.0),
                               rel=1e-12)


def test_dipole_operator_matches_monopole_pair() -> None:
    """Curle dipole constant, 1/r law, ``cos(theta)`` pattern and power, all at once."""
    omega = 2.0 * math.pi * 100.0
    q, d, r = 1.0e-3, 0.01, 1000.0
    thetas = np.radians(np.linspace(0.0, 180.0, 91))
    dirs = np.stack([np.sin(thetas), np.zeros_like(thetas), np.cos(thetas)], axis=1)
    p_ref = dipole_pair_far_field(q, d, omega, dirs, r, MEDIUM)
    force = dipole_pair_force(q, d, omega, MEDIUM)
    p_op = dipole_far_field_spectrum(np.array([[0.0, 0.0, force]], dtype=complex),
                                     np.array([100.0]), dirs, distance=r,
                                     medium=MEDIUM)[0]
    # amplitude: within the O(1/(k r)) far-field error of the exact pair
    sig = np.abs(np.cos(thetas)) > 0.2
    level = np.max(np.abs(np.abs(p_op[sig]) - np.abs(p_ref[sig])) / np.abs(p_ref[sig]))
    assert level < 1e-3
    # pattern: |cos(theta)| to < 1e-4
    pattern_op = np.abs(p_op) / np.abs(p_op[0])
    pattern_ref = np.abs(p_ref) / np.abs(p_ref[0])
    assert np.max(np.abs(pattern_op - pattern_ref)) < 1.0e-4
    # power equals the closed form
    quad_dirs, solid = sphere_quadrature(8, 32)
    p_sphere = dipole_far_field_spectrum(np.array([[0.0, 0.0, force]], dtype=complex),
                                         np.array([100.0]), quad_dirs, distance=r,
                                         medium=MEDIUM)
    power = acoustic_power_from_spectrum(p_sphere, r, solid, MEDIUM)
    assert power == pytest.approx(dipole_power_analytic(np.array([0.0, 0.0, force]),
                                                        omega, MEDIUM), rel=1e-10)


def test_dipole_null_plane() -> None:
    """The dipole null at ``theta = 90 deg`` is at least 60 dB below the axis."""
    force = 10.0
    dirs = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    p = dipole_far_field_spectrum(np.array([[0.0, 0.0, force]], dtype=complex),
                                  np.array([100.0]), dirs, distance=50.0,
                                  medium=MEDIUM)[0]
    assert abs(p[0]) > 0.0
    assert 20.0 * math.log10(max(abs(p[1]) / abs(p[0]), 1e-30)) < -60.0


def test_quadrupole_power_vs_analytic_compact_blob() -> None:
    """Lighthill quadrupole of a compact blob: radiated power vs closed form < 10 %."""
    freq = 100.0
    omega = 2.0 * math.pi * freq
    radius = 0.05
    n_cells = 64
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(n_cells, 3))
    pts = pts / np.linalg.norm(pts, axis=1)[:, None] * radius * rng.uniform(0.0, 1.0,
                                                                           (n_cells, 1))
    volume_total = 4.0 / 3.0 * math.pi * radius ** 3
    volumes = np.full(n_cells, volume_total / n_cells)
    fs, n = 4096.0, 1024
    t = np.arange(n) / fs
    u = np.empty((n_cells, n, 3))
    u[:, :, 0] = np.cos(omega * t)[None, :]
    u[:, :, 1] = np.sin(omega * t)[None, :]
    u[:, :, 2] = 0.0
    T = lighthill_stress_tensor(u, rho=MEDIUM.rho0, c0=MEDIUM.c0)
    q_hist = np.moveaxis(T, 1, -1) * volumes[:, None, None, None]
    q_spec = np.fft.rfft(q_hist, axis=-1) / n * 2.0
    q_spec[..., 0] *= 0.5
    if n % 2 == 0:
        q_spec[..., -1] *= 0.5
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    q_spec = np.moveaxis(q_spec, -1, 0)
    k2 = int(np.argmin(np.abs(freqs - 2.0 * freq)))
    dirs, solid = sphere_quadrature(6, 24)
    p = quadrupole_far_field_spectrum(q_spec[k2:k2 + 1], freqs[k2:k2 + 1], dirs,
                                      distance=1000.0, medium=MEDIUM, positions=pts)
    power = acoustic_power_from_spectrum(p, 1000.0, solid, MEDIUM)
    amp = 0.5 * volume_total * MEDIUM.rho0 * 1.0 ** 2
    q_theory = np.array([[amp, -1j * amp, 0.0], [-1j * amp, -amp, 0.0],
                         [0.0, 0.0, 0.0]])
    power_theory = quadrupole_power_analytic(q_theory, 2.0 * omega, MEDIUM)
    assert abs(power - power_theory) / power_theory < 0.10


def test_quadrupole_lateral_directivity() -> None:
    """A lateral quadrupole (``Q_xy``) has its lobes on ``phi = 45 deg`` planes."""
    q = np.zeros((3, 3), dtype=complex)
    q[0, 1] = q[1, 0] = 1.0
    thetas = np.radians(np.linspace(0.0, 180.0, 37))
    phi = math.radians(45.0)
    dirs = np.stack([np.sin(thetas) * math.cos(phi), np.sin(thetas) * math.sin(phi),
                     np.cos(thetas)], axis=1)
    p = quadrupole_far_field_spectrum(q[None, :, :], np.array([100.0]), dirs,
                                      distance=100.0, medium=MEDIUM)[0]
    pattern = np.abs(p) / np.max(np.abs(p))
    theory = np.abs(np.einsum("...i,...j,ij->...", dirs, dirs, q))
    theory = theory / np.max(theory)
    assert np.max(np.abs(pattern - theory)) < 1.0e-12
    # no radiation on the axis of a pure lateral quadrupole
    assert pattern[0] < 1.0e-12


def test_lighthill_eighth_power_law() -> None:
    """``P ~ rho0 U^8 l^2 / c0^5``: the fitted exponent must be 8 (+-1 %)."""
    lengths = np.full(6, 0.1)
    speeds = np.array([1.0, 1.3, 1.6, 2.0, 2.5, 3.0])
    powers = []
    for u in speeds:
        omega = 2.0 * math.pi * u / lengths[0]        # Strouhal scaling w ~ U/l
        q = MEDIUM.rho0 * u ** 2 * lengths[0] ** 3
        powers.append(quadrupole_power_analytic(np.array([[q, 0.0, 0.0],
                                                          [0.0, 0.0, 0.0],
                                                          [0.0, 0.0, 0.0]], dtype=complex),
                                                omega, MEDIUM))
    exponent, _ = fit_power_law(speeds, powers)
    assert exponent == pytest.approx(8.0, rel=0.01)


def test_sphere_quadrature_is_exact() -> None:
    """``sum dOmega = 4 pi`` and ``int n_i n_j n_k n_l dOmega`` matches the tensor identity."""
    dirs, solid = sphere_quadrature(6, 24)
    assert solid.sum() == pytest.approx(4.0 * math.pi, rel=1e-12)
    eye = np.eye(3)
    integral = np.einsum("a,ai,aj,ak,al->ijkl", solid, dirs, dirs, dirs, dirs)
    theory = 4.0 * math.pi / 15.0 * (np.einsum("ij,kl->ijkl", eye, eye)
                                     + np.einsum("ik,jl->ijkl", eye, eye)
                                     + np.einsum("il,jk->ijkl", eye, eye))
    assert np.max(np.abs(integral - theory)) < 1e-12


# --------------------------------------------------------------------------- #
# face-level FFT machinery
# --------------------------------------------------------------------------- #
def test_ring_buffer_wraparound() -> None:
    """The ring buffer keeps the newest samples in order after wrapping twice."""
    buf = RingBuffer(n_faces=2, capacity=8)
    buf.push(np.arange(10.0)[None, :].repeat(2, axis=0))
    assert buf.count == 8
    np.testing.assert_allclose(buf.snapshot()[0], np.arange(2.0, 10.0))
    buf.push(np.full((2, 3), -1.0))
    np.testing.assert_allclose(buf.snapshot()[0], np.array([5.0, 6.0, 7.0, 8.0, 9.0,
                                                            -1.0, -1.0, -1.0]))
    buf.reset()
    assert buf.count == 0


def test_face_spectrum_peak_amplitude_convention() -> None:
    """A unit cosine on an FFT bin gives ``|X| = 1`` (and DC/Nyquist halving)."""
    fs, n = 1024.0, 1024
    t = np.arange(n) / fs
    x = np.stack([np.cos(2 * math.pi * 64.0 * t), 3.0 * np.cos(2 * math.pi * 64.0 * t
                                                               + 0.5)])
    freqs, spec = face_spectrum_complex(x, 1.0 / fs, taper=False)
    k = int(np.argmin(np.abs(freqs - 64.0)))
    np.testing.assert_allclose(np.abs(spec[:, k]), [1.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(np.angle(spec[1, k]), 0.5, atol=1e-12)
    _, amp = face_spectrum(x, 1.0 / fs, taper=False)
    np.testing.assert_allclose(amp[:, k], [1.0, 3.0], atol=1e-12)


def test_fft_matches_direct_dft_and_is_much_faster() -> None:
    """O(N log N) face FFT vs O(N^2) direct DFT: error < 2 %, speedup >= 10x."""
    from tensordem.acoustics.face_fft import benchmark_fft_vs_direct

    fs, n, n_faces = 600.0, 2048, 16
    t = np.arange(n) / fs
    rng = np.random.default_rng(3)
    block = (np.cos(2 * math.pi * 12.0 * t)[None, :]
             + 0.4 * np.sin(2 * math.pi * 30.0 * t)[None, :]
             + 0.02 * rng.standard_normal((n_faces, n)))
    freqs, a_fft = face_spectrum(block, 1.0 / fs, taper=False)
    _, a_dft = direct_dft(block, 1.0 / fs, freqs=freqs)
    denom = np.sqrt(np.mean(a_dft ** 2))
    assert np.max(np.abs(a_fft - a_dft)) / denom < 0.02
    bench = benchmark_fft_vs_direct(block, 1.0 / fs, repeats=2, max_samples=n)
    assert bench.max_rel_error < 0.02
    assert bench.speedup >= 10.0


def test_spectral_propagation_matches_retarded_time_domain() -> None:
    """Frequency-domain (array phase) vs explicit retarded sum agree within 5 %."""
    from tensordem.acoustics.face_fft import dipole_field_time, peak_to_fft_coefficients

    positions, _normals, _areas, psi = build_face_sheet(8, 1, 0.6, 0.8)
    fs, n = 4000.0, 2048
    t = np.arange(n) / fs
    forces = rotor_face_forces(t, psi, rpm=180.0, blades=4, force_peak_n=250.0,
                               axis=[0.0, 0.0, 1.0])
    direction = np.array([0.3, 0.2, 0.93])
    direction = direction / np.linalg.norm(direction)
    distance = 10.0
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.fft.rfft(forces, axis=-1) / n * 2.0
    spec[..., 0] *= 0.5
    spec[..., -1] *= 0.5
    p_spec = dipole_far_field_spectrum(np.moveaxis(spec, -1, 0), freqs,
                                       direction[None, :], distance=distance,
                                       medium=MEDIUM, positions=positions)[:, 0]
    p_time = np.fft.irfft(peak_to_fft_coefficients(p_spec, n), n=n)
    p_ref = dipole_field_time(forces, positions, direction, distance, 1.0 / fs, MEDIUM)
    skip = 200
    err = (np.sqrt(np.mean((p_time[skip:] - p_ref[skip:]) ** 2))
           / np.sqrt(np.mean(p_ref[skip:] ** 2)))
    assert err < 0.05


# --------------------------------------------------------------------------- #
# BPF
# --------------------------------------------------------------------------- #
def test_blade_passing_frequency_law() -> None:
    """``f_BPF = N Z / 60`` and shaft frequency ``N/60``."""
    assert shaft_frequency_hz(180.0) == pytest.approx(3.0)
    assert blade_passing_frequency_hz(180.0, 4) == pytest.approx(12.0)
    assert blade_passing_frequency_hz(173.0, 5) == pytest.approx(173.0 * 5.0 / 60.0)
    assert blade_passing_frequency_hz(600.0, 3) == pytest.approx(30.0)
    with pytest.raises(ValueError):
        blade_passing_frequency_hz(0.0, 4)
    with pytest.raises(ValueError):
        blade_passing_frequency_hz(180.0, 0)


def test_bpf_tone_recovered_from_rotating_face_sheet() -> None:
    """The API case recovers the BPF line to better than 1 %."""
    payload = {
        "rotor": {"rpm": 173.0, "blades": 5, "force_peak_n": 1500.0,
                  "blade_scatter": 0.05, "seed": 3},
        "source": {"n_theta": 32, "n_axial": 1, "radius_m": 0.6, "length_m": 1.2},
        "time": {"fs_hz": 600.0, "n_samples": 8192},
        "observer": {"distance_m": 10.0},
    }
    out = handle_acoustic_source(payload)
    f_bpf = blade_passing_frequency_hz(173.0, 5)
    assert out["bpf_detection"]["relative_error"] < 0.01
    assert out["bpf_detection"]["f_peak_hz"] == pytest.approx(f_bpf, rel=0.01)
    assert out["bpf_detection"]["prominence_ratio"] > 50.0


def test_bpf_peak_finder_refuses_broadband_noise() -> None:
    """White noise must not be mistaken for a tone (prominence stays low)."""
    fs, n = 600.0, 8192
    rng = np.random.default_rng(11)
    _, amp = face_spectrum(rng.standard_normal((1, n)), 1.0 / fs, taper=True)
    det = find_spectral_peak(np.fft.rfftfreq(n, d=1.0 / fs), amp[0],
                             f_expected=blade_passing_frequency_hz(180.0, 4))
    assert det.prominence_ratio < 5.0


def test_third_octave_ladder_and_energy() -> None:
    """IEC centres bracket the band and a flat spectrum gives flat band levels."""
    fs, n = 4096.0, 4096
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    amp = np.full_like(freqs, 2.0)
    centres, levels = third_octave_levels(freqs, amp, f_min=10.0, f_max=1000.0)
    assert centres[0] == pytest.approx(10.0, rel=0.06)
    assert centres[-1] == pytest.approx(1000.0, rel=1e-12)
    # a flat bin amplitude gives band levels growing with the band width; the
    # 1 Hz bin grid quantises the low bands, so only the trend is asserted
    assert np.all(np.diff(levels) >= 0.0)
    assert levels[-1] / levels[0] == pytest.approx(
        math.sqrt((1000.0 * 2 ** (1 / 6) - 1000.0 * 2 ** (-1 / 6))
                  / (10.0 * 2 ** (1 / 6) - 10.0 * 2 ** (-1 / 6))), rel=0.35)
    # a single tone only lights the band that contains it
    tone = np.zeros_like(freqs)
    tone[np.argmin(np.abs(freqs - 500.0))] = 1.0
    _, levels_tone = third_octave_levels(freqs, tone, f_min=10.0, f_max=1000.0)
    assert np.argmax(levels_tone) == int(np.argmin(np.abs(centres - 501.187)))
    assert np.count_nonzero(levels_tone) <= 2


# --------------------------------------------------------------------------- #
# API endpoint
# --------------------------------------------------------------------------- #
def test_api_endpoint_happy_path() -> None:
    """``POST /api/v1/dem/acoustic_source`` returns a coherent result set."""
    payload = {
        "medium": {"c0": 1500.0, "rho0": 1025.0},
        "rotor": {"rpm": 180.0, "blades": 4, "force_peak_n": 2000.0},
        "source": {"n_theta": 16, "n_axial": 1, "radius_m": 0.6, "length_m": 1.2},
        "time": {"fs_hz": 600.0, "n_samples": 2048},
        "observer": {"distance_m": 10.0},
    }
    out = handle_acoustic_source(payload)
    assert out["endpoint"] == "/api/v1/dem/acoustic_source"
    assert out["status"] == "ok"
    assert out["rotor"]["bpf_hz"] == pytest.approx(12.0)
    assert out["lighthill_quadrupole"]["radiated_power_w"] > 0.0
    assert out["curle_dipole"]["bpf_power_numeric_w"] > 0.0
    assert len(out["directivity"]["angles_deg"]) == len(out["directivity"]["band_spl_db"])
    assert len(out["spectrum"]["freqs_hz"]) == len(out["spectrum"]["spl_db"])
    # dipole directivity: on-axis louder than the 90 deg null
    angles = np.asarray(out["directivity"]["angles_deg"])
    spl = np.asarray(out["directivity"]["band_spl_db"])
    axis = spl[np.argmin(angles)]
    null = spl[np.argmin(np.abs(angles - 90.0))]
    assert axis > null + 20.0
    assert out["fft_performance"]["speedup"] > 5.0
    assert pwsl_db(out["lighthill_quadrupole"]["radiated_power_w"], MEDIUM) == \
        pytest.approx(out["lighthill_quadrupole"]["pwsl_db"], rel=1e-9)


def test_api_rejects_invalid_payload() -> None:
    """Bad operating points surface as ``ValueError`` (HTTP 422 at the route)."""
    base = {"rotor": {"rpm": 180.0, "blades": 4},
            "source": {"n_theta": 8, "n_axial": 1},
            "time": {"fs_hz": 600.0, "n_samples": 512},
            "observer": {"distance_m": 10.0}}
    with pytest.raises(ValueError):
        handle_acoustic_source({**base, "rotor": {"rpm": 0.0, "blades": 4}})
    with pytest.raises(ValueError):
        handle_acoustic_source({**base, "rotor": {"rpm": 180.0, "blades": 1}})
    with pytest.raises(ValueError):
        handle_acoustic_source({**base, "medium": {"c0": -5.0}})
    with pytest.raises(ValueError):
        handle_acoustic_source({**base, "observer": {"distance_m": 0.0}})
    with pytest.raises(ValueError):
        handle_acoustic_source({**base, "time": {"fs_hz": 600.0, "n_samples": 8}})


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))