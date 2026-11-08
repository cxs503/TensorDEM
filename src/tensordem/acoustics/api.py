"""API handler for ``POST /api/v1/dem/acoustic_source``.

The route turns a compact description of a rotating machine surface
(``medium`` + ``rotor`` + ``source`` + ``time`` + ``observer``) into

* the Curle dipole field of the surface load (BPF tone ladder, directivity),
* the Lighthill quadrupole field of the wake momentum flux (power, PWSL),
* a one-third-octave spectrum and a polar directivity cut,
* the timing of the O(N log N) face-level FFT against the O(N^2) reference.

Like the other TensorDEM endpoint handlers (e.g.
``tensordem.ice_abrasion.acoustic_*``) this is a pure function ``payload -> dict``
so the FastAPI route is a two-liner and invalid input surfaces as ``ValueError``
(mapped to HTTP 422 by the route layer).
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .bpf import (
    blade_passing_frequency_hz,
    find_spectral_peak,
    rotor_face_forces,
    rotor_face_turbulence,
    shaft_frequency_hz,
    third_octave_levels,
)
from .face_fft import benchmark_fft_vs_direct, face_spectrum, face_spectrum_complex
from .source_terms import (
    Medium,
    acoustic_power_from_spectrum,
    dipole_far_field_spectrum,
    lighthill_stress_tensor,
    pwsl_db,
    quadrupole_far_field_spectrum,
    quadrupole_power_analytic,
    spl_db,
    sphere_quadrature,
)

__all__ = ["ACOUSTIC_SOURCE_ENDPOINT", "handle_acoustic_source", "build_face_sheet"]

ACOUSTIC_SOURCE_ENDPOINT = "/api/v1/dem/acoustic_source"


# --------------------------------------------------------------------------- #
# payload helpers
# --------------------------------------------------------------------------- #
def _require_positive(payload: dict[str, Any], key: str, default: float,
                      where: str) -> float:
    value = float(payload.get(key, default))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{where}.{key} must be a positive finite number")
    return value


def build_face_sheet(n_theta: int = 12, n_axial: int = 3, radius: float = 0.6,
                     length: float = 1.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cylindrical face sheet around the shaft.

    Returns ``(positions, normals, areas, azimuth)`` for ``n_theta x n_axial``
    quadrilateral patches: the surface that carries the Curle loads and the
    wake control volumes of the Lighthill quadrupoles.
    """
    if n_theta < 3 or n_axial < 1:
        raise ValueError("n_theta >= 3 and n_axial >= 1 are required")
    if radius <= 0.0 or length <= 0.0:
        raise ValueError("radius and length must be positive")
    psi = 2.0 * math.pi * np.arange(n_theta) / n_theta
    xs = np.linspace(-0.5 * length, 0.5 * length, n_axial + 1)
    positions, normals, areas, azimuth = [], [], [], []
    d_area = (2.0 * math.pi * radius / n_theta) * (length / n_axial)
    for i in range(n_axial):
        xc = 0.5 * (xs[i] + xs[i + 1])
        for p in psi:
            positions.append([xc, radius * math.cos(p), radius * math.sin(p)])
            normals.append([0.0, math.cos(p), math.sin(p)])
            areas.append(d_area)
            azimuth.append(p)
    return (np.asarray(positions, dtype=float), np.asarray(normals, dtype=float),
            np.asarray(areas, dtype=float), np.asarray(azimuth, dtype=float))


def _axis_frame(axis: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal frame ``(e1, e2, e3)`` with ``e3`` along ``axis``."""
    e3 = np.asarray(axis, dtype=float).ravel()
    if e3.size != 3 or not np.any(e3):
        raise ValueError("source.force_axis must be a non-zero 3-vector")
    e3 = e3 / np.linalg.norm(e3)
    seed = np.array([1.0, 0.0, 0.0]) if abs(e3[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(e3, seed)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    return e1, e2, e3


def _observer_directions(angles_deg: Any, axis: Any = (0.0, 0.0, 1.0),
                         azimuth_deg: float = 0.0) -> np.ndarray:
    """Polar cut at ``azimuth_deg`` around ``axis``: ``theta`` from ``angles_deg``.

    ``theta`` is measured from the load/rotor axis, so ``theta = 0`` is the on-axis
    (maximum dipole) direction and ``theta = 90`` sits in the dipole null plane.
    """
    angles = np.asarray(angles_deg, dtype=float).ravel()
    if angles.size < 3:
        raise ValueError("observer.angles_deg needs at least 3 angles")
    if np.any(angles < 0.0) or np.any(angles > 180.0):
        raise ValueError("observer angles must lie in [0, 180] deg (polar cut)")
    e1, e2, e3 = _axis_frame(axis)
    th = np.radians(angles)
    ph = math.radians(float(azimuth_deg))
    radial = np.cos(ph) * e1[None, :] + np.sin(ph) * e2[None, :]
    return (np.sin(th)[:, None] * radial + np.cos(th)[:, None] * e3[None, :])


# --------------------------------------------------------------------------- #
# endpoint
# --------------------------------------------------------------------------- #
def handle_acoustic_source(payload: dict[str, Any]) -> dict[str, Any]:
    """Full acoustic-source analysis for one operating point."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    t_start = time.perf_counter()

    medium_payload = payload.get("medium", {}) or {}
    medium = Medium(
        c0=_require_positive(medium_payload, "c0", 1500.0, "medium"),
        rho0=_require_positive(medium_payload, "rho0", 1025.0, "medium"),
        p_ref=_require_positive(medium_payload, "p_ref", 1.0e-6, "medium"),
        p_power_ref=_require_positive(medium_payload, "p_power_ref", 1.0e-12, "medium"),
    )

    rotor = payload.get("rotor", {}) or {}
    rpm = _require_positive(rotor, "rpm", 180.0, "rotor")
    blades_raw = rotor.get("blades", 4)
    if int(blades_raw) != blades_raw or int(blades_raw) < 2:
        raise ValueError("rotor.blades must be an integer >= 2")
    blades = int(blades_raw)
    force_peak = _require_positive(rotor, "force_peak_n", 2000.0, "rotor")
    harmonic_ratios = rotor.get("harmonic_ratios", None)
    force_axis = rotor.get("force_axis", [0.0, 0.0, 1.0])
    blade_scatter = float(rotor.get("blade_scatter", 0.05))
    if blade_scatter < 0.0 or blade_scatter > 0.5:
        raise ValueError("rotor.blade_scatter must lie in [0, 0.5]")

    source = payload.get("source", {}) or {}
    n_theta = int(source.get("n_theta", 32))
    n_axial = int(source.get("n_axial", 2))
    radius = _require_positive(source, "radius_m", 0.6, "source")
    length = _require_positive(source, "length_m", 1.2, "source")
    mixing_thickness = _require_positive(source, "mixing_thickness_m", 0.15, "source")
    u_rms = _require_positive(source, "u_rms_m_s", 2.0, "source")
    u_mean = source.get("u_mean_m_s", [4.0, 0.0, 0.0])
    broadband = float(source.get("broadband_fraction", 0.05))
    if broadband < 0.0 or broadband > 1.0:
        raise ValueError("source.broadband_fraction must lie in [0, 1]")

    time_payload = payload.get("time", {}) or {}
    fs = _require_positive(time_payload, "fs_hz", 600.0, "time")
    n_samples = int(time_payload.get("n_samples", 8192))
    if n_samples < 256:
        raise ValueError("time.n_samples must be >= 256")

    obs = payload.get("observer", {}) or {}
    distance = _require_positive(obs, "distance_m", 10.0, "observer")
    angles = obs.get("angles_deg", [0, 30, 45, 60, 90, 120, 135, 150, 180])
    obs_azimuth = float(obs.get("azimuth_deg", 0.0))

    dt = 1.0 / fs
    t = np.arange(n_samples) * dt
    positions, _normals, areas, azimuth = build_face_sheet(n_theta, n_axial, radius, length)
    directions = _observer_directions(angles, force_axis, obs_azimuth)

    # --- sources ---------------------------------------------------------
    # blade-to-blade loading scatter: the mechanism that makes a compact rotor
    # radiate at the BPF at all (a perfectly Z-periodic load cancels, see bpf.py)
    scatter = 1.0 + blade_scatter * np.random.default_rng(
        int(rotor.get("seed", 0))).standard_normal(positions.shape[0])
    # ``force_peak_n`` is the *total* surface load, shared by the face sheet
    force_per_face = force_peak / float(positions.shape[0])
    forces = rotor_face_forces(t, azimuth, rpm=rpm, blades=blades,
                               force_peak_n=force_per_face, axis=force_axis,
                               harmonic_ratios=harmonic_ratios,
                               blade_scatter=scatter)
    velocity = rotor_face_turbulence(t, azimuth, rpm=rpm, blades=blades, u_rms=u_rms,
                                     u_mean=u_mean, harmonic_ratios=harmonic_ratios,
                                     rng=np.random.default_rng(int(rotor.get("seed", 0)) + 1),
                                     broadband_fraction=broadband,
                                     blade_scatter=scatter)

    # --- Curle dipole ----------------------------------------------------
    t_dip = time.perf_counter()
    freqs, f_spec_faces = face_spectrum_complex(forces.reshape(-1, n_samples), dt,
                                                taper=True)
    f_spec_faces = f_spec_faces.reshape(forces.shape[0], 3, -1)
    p_dip = dipole_far_field_spectrum(np.moveaxis(f_spec_faces, -1, 0), freqs, directions,
                                      distance=distance, medium=medium,
                                      positions=positions)
    dip_seconds = time.perf_counter() - t_dip
    p_dip_rms = np.sqrt(0.5) * np.abs(p_dip)     # per-bin peak amplitude -> RMS
    spl_spec = spl_db(p_dip_rms, medium)         # (n_freq, n_dir)

    f_bpf = blade_passing_frequency_hz(rpm, blades)
    f_shaft = shaft_frequency_hz(rpm)
    fwd = int(np.argmin(np.abs(np.asarray(angles, dtype=float))))
    detection = find_spectral_peak(freqs, p_dip_rms[:, fwd], f_expected=f_bpf)
    tones = []
    for k in range(1, 4):
        try:
            det = find_spectral_peak(freqs, p_dip_rms[:, fwd], f_expected=k * f_bpf)
        except ValueError:
            continue
        tones.append({
            "harmonic": k,
            "f_expected_hz": k * f_bpf,
            "f_peak_hz": det.f_peak,
            "relative_error": det.relative_error,
            "spl_db": float(spl_db(det.peak_amplitude, medium)),
            "prominence_ratio": det.prominence_ratio,
        })

    centres, band = third_octave_levels(freqs, p_dip_rms[:, fwd])
    band_spl = spl_db(np.maximum(band, 1.0e-30), medium)

    # --- Lighthill quadrupole (wake momentum flux) -----------------------
    t_lb = time.perf_counter()
    u_face = np.moveaxis(velocity, -1, 0)                    # (N, n_faces, 3)
    rho = np.full(u_face.shape[:2], medium.rho0)
    T = lighthill_stress_tensor(u_face, rho=rho, c0=medium.c0)  # (N, n_faces, 3, 3)
    volumes = areas * mixing_thickness
    q_hist = np.moveaxis(T, 0, -1) * volumes[:, None, None, None]
    q_spec = np.fft.rfft(q_hist, axis=-1) / n_samples * 2.0   # peak amplitude
    q_spec[..., 0] *= 0.5
    if n_samples % 2 == 0:
        q_spec[..., -1] *= 0.5
    q_spec = np.moveaxis(q_spec, -1, 0)                        # (n_freq, n_face, 3, 3)
    p_quad = quadrupole_far_field_spectrum(q_spec, freqs, directions, distance=distance,
                                           medium=medium, positions=positions)
    lb_seconds = time.perf_counter() - t_lb

    quad_dirs, solid = sphere_quadrature(6, 24)
    p_sphere = quadrupole_far_field_spectrum(q_spec, freqs, quad_dirs, distance=distance,
                                             medium=medium, positions=positions)
    power_band = acoustic_power_from_spectrum(p_sphere, distance, solid, medium)
    pwsl = float(pwsl_db(power_band, medium))

    # Coherent (all faces in phase) upper bound at the BPF line: the rotating
    # Z-lobe pattern is an azimuthal multipole and must radiate below it.  The
    # shortfall is reported as a suppression in dB, not treated as an error.
    k_bpf = int(np.argmin(np.abs(freqs - f_bpf)))
    q_coherent = np.sum(np.abs(q_spec[k_bpf]), axis=0)
    p_bound = quadrupole_power_analytic(q_coherent, 2.0 * math.pi * freqs[k_bpf], medium)
    band_mask = (freqs >= f_bpf * 0.9) & (freqs <= f_bpf * 1.1)
    power_bpf_band = acoustic_power_from_spectrum(p_sphere[band_mask], distance, solid, medium)
    mode_suppression_db = (10.0 * math.log10(power_bpf_band / p_bound)
                           if p_bound > 0.0 and power_bpf_band > 0.0 else float("nan"))

    # dipole reference at the BPF line: coherent (all faces in phase) upper bound
    f_coherent = np.sum(np.abs(f_spec_faces[:, :, k_bpf]), axis=0)
    from .source_terms import dipole_power_analytic
    dip_power_bound = dipole_power_analytic(f_coherent, 2.0 * math.pi * f_bpf, medium)
    p_dip_sphere = dipole_far_field_spectrum(
        np.moveaxis(f_spec_faces, -1, 0)[k_bpf:k_bpf + 1],
        freqs[k_bpf:k_bpf + 1], quad_dirs,
        distance=distance, medium=medium, positions=positions)
    dip_power_numeric = acoustic_power_from_spectrum(p_dip_sphere, distance, solid, medium)
    dip_mode_suppression_db = (10.0 * math.log10(dip_power_numeric / dip_power_bound)
                               if dip_power_bound > 0.0 and dip_power_numeric > 0.0
                               else float("nan"))

    # --- O(N) face-level FFT vs O(N^2) direct DFT ------------------------
    bench_block = forces.reshape(-1, n_samples)
    bench = benchmark_fft_vs_direct(bench_block, dt, repeats=2, max_samples=1024)

    total_directivity = 10.0 * np.log10(
        np.maximum(np.sum(10.0 ** (spl_spec[:, :] / 10.0), axis=0), 1.0e-30))
    # report the polar cut on an 80 dB dynamic range (the dipole null is exact)
    directivity_floor = float(np.max(total_directivity)) - 80.0
    total_directivity = np.maximum(total_directivity, directivity_floor)

    elapsed = time.perf_counter() - t_start
    return {
        "endpoint": ACOUSTIC_SOURCE_ENDPOINT,
        "status": "ok",
        "medium": {"c0_m_s": medium.c0, "rho0_kg_m3": medium.rho0,
                   "p_ref_pa": medium.p_ref, "p_power_ref_w": medium.p_power_ref},
        "rotor": {"rpm": rpm, "blades": blades, "shaft_hz": f_shaft,
                  "bpf_hz": f_bpf, "force_peak_n": force_peak},
        "source": {"n_faces": int(positions.shape[0]), "radius_m": radius,
                   "length_m": length, "mixing_thickness_m": mixing_thickness,
                   "u_rms_m_s": u_rms, "face_area_m2": float(areas[0]),
                   "sample_rate_hz": fs, "n_samples": n_samples,
                   "frequency_resolution_hz": float(freqs[1] - freqs[0])},
        "bpf_detection": detection.to_dict(),
        "tones": tones,
        "spectrum": {
            "freqs_hz": [float(v) for v in freqs],
            "spl_db": [float(v) for v in spl_spec[:, fwd]],
            "third_octave_centres_hz": [float(v) for v in centres],
            "third_octave_spl_db": [float(v) for v in band_spl],
            "observer_angle_deg": float(angles[fwd]),
        },
        "directivity": {
            "angles_deg": [float(v) for v in np.asarray(angles, dtype=float)],
            "band_spl_db": [float(v) for v in total_directivity],
        },
        "lighthill_quadrupole": {
            "radiated_power_w": power_band,
            "pwsl_db": pwsl,
            "bpf_band_power_w": power_bpf_band,
            "bpf_in_phase_bound_w": p_bound,
            "bpf_mode_suppression_db": mode_suppression_db,
            "t_ij_rms_pa": float(np.sqrt(np.mean(
                np.sum(T.reshape(-1, 3, 3) ** 2, axis=(1, 2))))),
            "mach_number": float(u_rms / medium.c0),
        },
        "curle_dipole": {
            "force_rms_n": float(np.sqrt(np.mean(np.sum(forces ** 2, axis=1)))),
            "total_force_rms_n": float(np.sqrt(np.mean(np.sum(forces, axis=1) ** 2))),
            "bpf_power_numeric_w": dip_power_numeric,
            "bpf_in_phase_bound_w": dip_power_bound,
            "bpf_mode_suppression_db": dip_mode_suppression_db,
            "bpf_pwsl_db": float(pwsl_db(dip_power_numeric, medium)),
        },
        "fft_performance": bench.to_dict(),
        "timing": {"dipole_s": dip_seconds, "lighthill_s": lb_seconds,
                   "total_s": elapsed},
        "limitations": [
            "compact-source (far-field) approximation: kr_far >> 1, L << lambda",
            "no non-linear propagation, no refraction/shear shielding, no sea-surface image",
            "isotropic medium, single homogeneous c0 and rho0",
            "Lighthill quadrupole keeps rho u_i u_j only (entropy term p' - c0^2 rho' set to 0)",
        ],
    }