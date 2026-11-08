"""Acceptance battery for the vibro-acoustic source terms (node P27W6-V1).

Five machine-checkable criteria, each returning a number and a pass flag:

(a) Lighthill quadrupole radiated power vs the closed-form compact-source value
    (``omega^4 Q_ij Q_kl <n_i n_j n_k n_l> / (32 pi^2 rho0 c0^5)``), error < 10 %;
(b) Curle dipole directivity vs the exact oscillating-sphere solution
    (``cos theta`` pattern plus absolute level), error < 10 %;
(c) blade-passing-frequency peak detected in the face-level spectrum,
    error < 1 % (two operating points, plus a broadband negative control);
(d) O(N log N) face-level FFT vs the O(N^2) direct DFT: amplitude error < 2 %
    and wall-clock ratio >= 10x, with the spectral propagation also checked
    against the explicit retarded-time summation;
(e) the pytest module of this capability (>= 12 tests) -- the count is collected
    from pytest when it is importable, otherwise from a source scan.

Everything is computed here and written to JSON by
``scripts/verify_acoustic_source_terms.py``; no number is hand-copied into the
documentation.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .api import handle_acoustic_source
from .bpf import blade_passing_frequency_hz, find_spectral_peak
from .face_fft import (
    RingBuffer,
    benchmark_fft_vs_direct,
    dipole_field_time,
    face_spectrum,
    quadrupole_field_time,
)
from .source_terms import (
    Medium,
    acoustic_power_from_spectrum,
    dipole_far_field_spectrum,
    dipole_power_analytic,
    lighthill_stress_tensor,
    dipole_pair_far_field,
    dipole_pair_force,
    dipole_pair_power,
    monopole_far_field,
    quadrupole_far_field_spectrum,
    quadrupole_power_analytic,
    sphere_quadrature,
)

__all__ = [
    "Criterion",
    "run_acceptance",
    "compact_quadrupole_case",
    "compact_dipole_case",
    "finite_extent_dipole_case",
    "bpf_cases",
    "fft_cases",
]


@dataclass
class Criterion:
    """One acceptance criterion: measured value, target, pass flag, detail."""

    ident: str
    name: str
    value: float
    target: float
    passed: bool
    comparison: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.ident,
            "name": self.name,
            "value": self.value,
            "target": self.target,
            "comparison": self.comparison,
            "passed": bool(self.passed),
            "detail": self.detail,
        }


# --------------------------------------------------------------------------- #
# (a) Lighthill quadrupole: compact rotating-stress blob
# --------------------------------------------------------------------------- #
def compact_quadrupole_case(
    *,
    medium: Medium | None = None,
    freq: float = 100.0,
    u_amp: float = 1.0,
    blob_radius: float = 0.05,
    n_cells: int = 200,
    fs: float = 4096.0,
    n_samples: int = 4096,
    n_theta_quad: int = 6,
    n_phi_quad: int = 24,
    distance: float = 1000.0,
    seed: int = 3,
) -> dict[str, Any]:
    """Lighthill quadrupole of a compact blob with ``u = U(cos wt, sin wt, 0)``.

    ``T_ij = rho u_i u_j`` has its acoustic content at ``2 omega``; the closed-form
    strength there is ``Q(2w) = V rho U^2/2 * [[1, -i, 0], [-i, -1, 0], [0,0,0]]``.
    """
    medium = medium or Medium()
    rng = np.random.default_rng(seed)
    # uniform points inside a sphere (rejection sampling)
    pts = []
    while len(pts) < n_cells:
        cand = rng.uniform(-1.0, 1.0, size=(4 * n_cells, 3))
        keep = cand[np.linalg.norm(cand, axis=1) <= 1.0]
        pts.extend(keep[: n_cells - len(pts)].tolist())
    positions = blob_radius * np.asarray(pts[:n_cells], dtype=float)
    volume_total = 4.0 / 3.0 * math.pi * blob_radius ** 3
    volumes = np.full(n_cells, volume_total / n_cells)

    t = np.arange(n_samples) / fs
    omega = 2.0 * math.pi * freq
    u = np.empty((n_cells, n_samples, 3), dtype=float)
    u[:, :, 0] = u_amp * np.cos(omega * t)[None, :]
    u[:, :, 1] = u_amp * np.sin(omega * t)[None, :]
    u[:, :, 2] = 0.0
    T = lighthill_stress_tensor(u, rho=medium.rho0, c0=medium.c0)   # (cells, N, 3, 3)
    q_hist = np.moveaxis(T, 1, -1) * volumes[:, None, None, None]   # (cells, 3, 3, N)
    q_spec = np.fft.rfft(q_hist, axis=-1) / n_samples * 2.0
    q_spec[..., 0] *= 0.5
    if n_samples % 2 == 0:
        q_spec[..., -1] *= 0.5
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    q_spec = np.moveaxis(q_spec, -1, 0)                             # (n_freq, cells, 3, 3)

    dirs, solid = sphere_quadrature(n_theta_quad, n_phi_quad)
    p_sphere = quadrupole_far_field_spectrum(q_spec, freqs, dirs, distance=distance,
                                             medium=medium, positions=positions)
    k2 = int(np.argmin(np.abs(freqs - 2.0 * freq)))
    power_numeric = acoustic_power_from_spectrum(p_sphere[k2:k2 + 1], distance, solid,
                                                 medium)

    # closed form at 2*omega (peak/complex convention)
    amp = 0.5 * volume_total * medium.rho0 * u_amp ** 2
    q_theory = np.array([[amp, -1j * amp, 0.0],
                         [-1j * amp, -amp, 0.0],
                         [0.0, 0.0, 0.0]], dtype=complex)
    power_theory = quadrupole_power_analytic(q_theory, 2.0 * omega, medium)
    # transverse plane through phi = 45 deg isolates the |sin^2 theta| lobe
    phi = math.radians(45.0)
    thetas = np.radians(np.linspace(0.0, 180.0, 91))
    pol = np.stack([np.sin(thetas) * math.cos(phi), np.sin(thetas) * math.sin(phi),
                    np.cos(thetas)], axis=1)
    p_pol = quadrupole_far_field_spectrum(q_spec[k2:k2 + 1], freqs[k2:k2 + 1], pol,
                                          distance=distance, medium=medium,
                                          positions=positions)[0]
    pattern_numeric = np.abs(p_pol) / np.max(np.abs(p_pol))
    q_eff = q_spec[k2].sum(axis=0)
    pattern_theory = np.abs(np.einsum("...i,...j,ij->...", pol, pol, q_eff))
    pattern_theory = pattern_theory / np.max(pattern_theory)
    return {
        "freq_hz": freq,
        "acoustic_freq_hz": float(freqs[k2]),
        "power_numeric_w": power_numeric,
        "power_theory_w": power_theory,
        "relative_error": abs(power_numeric - power_theory) / power_theory,
        "blob_radius_m": blob_radius,
        "kR": 2.0 * math.pi * (2.0 * freq) / medium.c0 * blob_radius,
        "directivity_theta_deg": np.degrees(thetas).tolist(),
        "directivity_numeric": pattern_numeric.tolist(),
        "directivity_theory": pattern_theory.tolist(),
        "directivity_max_abs_error": float(np.max(np.abs(pattern_numeric
                                                       - pattern_theory))),
        "quadrature": {"n_theta": n_theta_quad, "n_phi": n_phi_quad},
    }


# --------------------------------------------------------------------------- #
# (b) Curle dipole: oscillating sphere, exact solution
# --------------------------------------------------------------------------- #
def compact_dipole_case(
    *,
    medium: Medium | None = None,
    freq: float = 100.0,
    volume_velocity: float = 1.0e-3,
    separation: float = 0.01,
    distance: float = 1000.0,
    n_theta: int = 181,
) -> dict[str, Any]:
    """Curle dipole operator vs the exact compact dipole (two monopoles of +/-Q).

    The reference is built from the textbook monopole Green's function
    ``rho_0 omega^2 Q/(4 pi r)`` only; the equivalent force ``F = rho_0 omega Q d``
    is then fed to :func:`dipole_far_field_spectrum`.  Agreement to ``O((kd)^2)``
    validates the ``omega/(4 pi c_0)`` constant, the ``1/r`` law, the ``cos theta``
    pattern and the radiated power of the implementation.  The complex fields are
    compared, so a wrong retarded-phase sign would show up immediately.
    """
    medium = medium or Medium()
    omega = 2.0 * math.pi * freq
    thetas = np.radians(np.linspace(0.0, 180.0, n_theta))
    dirs = np.stack([np.sin(thetas), np.zeros_like(thetas), np.cos(thetas)], axis=1)

    p_ref = dipole_pair_far_field(volume_velocity, separation, omega, dirs, distance,
                                  medium)
    force = dipole_pair_force(volume_velocity, separation, omega, medium)
    f_lumped = np.array([[0.0, 0.0, force]], dtype=complex)
    p_op = dipole_far_field_spectrum(f_lumped, np.array([freq]), dirs,
                                     distance=distance, medium=medium)[0]

    scale = float(np.max(np.abs(p_ref)))
    sig = np.abs(np.cos(thetas)) > 0.2
    level_error = (np.abs(np.abs(p_op[sig]) - np.abs(p_ref[sig]))
                   / np.abs(p_ref[sig]))
    pattern_op = np.abs(p_op) / np.abs(p_op[0])
    pattern_ref = np.abs(p_ref) / np.abs(p_ref[0])

    quad_dirs, quad_solid = sphere_quadrature(8, 32)
    p_op_sphere = dipole_far_field_spectrum(f_lumped, np.array([freq]), quad_dirs,
                                            distance=distance, medium=medium)[0]
    power_op = acoustic_power_from_spectrum(p_op_sphere[None, :], distance, quad_solid,
                                            medium)
    p_ref_sphere = dipole_pair_far_field(volume_velocity, separation, omega, quad_dirs,
                                         distance, medium)
    power_ref_numeric = acoustic_power_from_spectrum(p_ref_sphere[None, :], distance,
                                                     quad_solid, medium)
    power_theory = dipole_pair_power(volume_velocity, separation, omega, medium)
    return {
        "freq_hz": freq,
        "separation_m": separation,
        "kd": omega / medium.c0 * separation,
        "volume_velocity_m3_s": volume_velocity,
        "equivalent_force_n": force,
        "monopole_level_pa": monopole_far_field(volume_velocity, omega, distance, medium),
        "directivity_theta_deg": np.degrees(thetas).tolist(),
        "directivity_operator": pattern_op.tolist(),
        "directivity_reference": pattern_ref.tolist(),
        "directivity_max_abs_error": float(np.max(np.abs(pattern_op - pattern_ref))),
        "level_max_relative_error": float(np.max(level_error)),
        "complex_max_relative_error": float(np.max(np.abs(p_op - p_ref)) / scale),
        "power_operator_w": power_op,
        "power_reference_numeric_w": power_ref_numeric,
        "power_analytic_w": power_theory,
        "power_relative_error": abs(power_op - power_theory) / power_theory,
        "reference_power_relative_error": (abs(power_ref_numeric - power_theory)
                                           / power_theory),
    }


def finite_extent_dipole_case(
    *,
    medium: Medium | None = None,
    radius: float = 0.1,
    length: float = 0.4,
    force_peak: float = 1.0,
    distance: float = 1000.0,
    n_theta_faces: int = 16,
    n_axial: int = 3,
    frequencies: tuple[float, ...] = (25.0, 50.0, 100.0, 200.0, 400.0, 800.0),
) -> dict[str, Any]:
    """Compactness study: a uniform in-phase axial load on a finite cylinder.

    The face sheet has a *finite axial extent* ``L``, so the retarded phases
    ``e^{i k x_f cos(theta)}`` no longer factor out of the directivity and the
    pattern departs from the point-dipole ``|cos theta|`` by ``O((kL/2)^2)``.
    That departure is the honest bound on the compact-source approximation used
    everywhere else in this package.
    """
    from .api import build_face_sheet

    medium = medium or Medium()
    positions, _normals, areas, psi = build_face_sheet(n_theta_faces, n_axial,
                                                       radius, length)
    # uniform axial load spread over the face sheet (net force = force_peak)
    weight = np.full(positions.shape[0], force_peak / positions.shape[0])
    force_total = float(np.sum(weight))
    del psi
    thetas = np.radians(np.linspace(0.0, 180.0, 73))
    dirs = np.stack([np.sin(thetas), np.zeros_like(thetas), np.cos(thetas)], axis=1)
    quad_dirs, quad_solid = sphere_quadrature(8, 32)
    rows = []
    for freq in frequencies:
        omega = 2.0 * math.pi * freq
        f_spec = np.zeros((1, positions.shape[0], 3), dtype=complex)
        f_spec[0, :, 2] = weight
        p = dipole_far_field_spectrum(f_spec, np.array([freq]), dirs, distance=distance,
                                      medium=medium, positions=positions)[0]
        pattern = np.abs(p) / np.abs(p[0])
        p_sphere = dipole_far_field_spectrum(f_spec, np.array([freq]), quad_dirs,
                                             distance=distance, medium=medium,
                                             positions=positions)
        power = acoustic_power_from_spectrum(p_sphere, distance, quad_solid, medium)
        power_analytic = dipole_power_analytic(np.array([0.0, 0.0, force_total]), omega,
                                               medium)
        rows.append({
            "freq_hz": freq,
            "kL_over_2": omega / medium.c0 * 0.5 * length,
            "kR": omega / medium.c0 * radius,
            "directivity_max_abs_error": float(np.max(np.abs(pattern
                                                             - np.abs(np.cos(thetas))))),
            "power_w": power,
            "power_analytic_w": power_analytic,
            "power_relative_error": abs(power - power_analytic) / power_analytic,
        })
    scaled = np.array([r["kL_over_2"] for r in rows])
    err = np.array([r["directivity_max_abs_error"] for r in rows])
    exponent = float(np.polyfit(np.log(scaled), np.log(np.maximum(err, 1e-16)), 1)[0])
    return {
        "radius_m": radius,
        "length_m": length,
        "n_faces": int(positions.shape[0]),
        "force_total_n": force_total,
        "rows": rows,
        "error_exponent_in_kL": exponent,
        "max_error_kL_over_2_le_0p1": float(np.max(err[scaled <= 0.1])),
    }


# --------------------------------------------------------------------------- #
# (c) BPF detection
# --------------------------------------------------------------------------- #
def bpf_cases(*, seeds: tuple[int, ...] = (0, 7)) -> dict[str, Any]:
    """Detect the BPF line in the face-level spectrum at two operating points."""
    cases = []
    for rpm, blades, seed in ((180.0, 4, seeds[0]), (173.0, 5, seeds[1])):
        payload = {
            "medium": {"c0": 1500.0, "rho0": 1025.0},
            "rotor": {"rpm": rpm, "blades": blades, "force_peak_n": 2000.0,
                      "force_axis": [0.0, 0.0, 1.0], "blade_scatter": 0.05,
                      "seed": seed},
            "source": {"n_theta": 32, "n_axial": 2, "radius_m": 0.6, "length_m": 1.2,
                       "u_rms_m_s": 2.0, "mixing_thickness_m": 0.15},
            "time": {"fs_hz": 600.0, "n_samples": 8192},
            "observer": {"distance_m": 10.0},
        }
        out = handle_acoustic_source(payload)
        cases.append({
            "rpm": rpm,
            "blades": int(blades),
            "f_bpf_hz": blade_passing_frequency_hz(rpm, blades),
            "f_peak_hz": out["bpf_detection"]["f_peak_hz"],
            "relative_error": out["bpf_detection"]["relative_error"],
            "prominence_ratio": out["bpf_detection"]["prominence_ratio"],
            "tones": out["tones"],
            "spectrum_freqs": out["spectrum"]["freqs_hz"],
            "spectrum_spl": out["spectrum"]["spl_db"],
            "directivity": out["directivity"],
            "lighthill": out["lighthill_quadrupole"],
            "curle": out["curle_dipole"],
            "fft_performance": out["fft_performance"],
            "third_octave": {"centres_hz": out["spectrum"]["third_octave_centres_hz"],
                             "spl_db": out["spectrum"]["third_octave_spl_db"]},
        })

    # symmetry control: with a perfectly Z-periodic load the BPF multipole cancels
    payload = {
        "medium": {"c0": 1500.0, "rho0": 1025.0},
        "rotor": {"rpm": 180.0, "blades": 4, "force_peak_n": 2000.0,
                  "force_axis": [0.0, 0.0, 1.0], "blade_scatter": 0.0, "seed": 0},
        "source": {"n_theta": 32, "n_axial": 2, "radius_m": 0.6, "length_m": 1.2,
                   "u_rms_m_s": 2.0, "mixing_thickness_m": 0.15},
        "time": {"fs_hz": 600.0, "n_samples": 8192},
        "observer": {"distance_m": 10.0},
    }
    sym = handle_acoustic_source(payload)
    symmetric = {
        "blade_scatter": 0.0,
        "bpf_tone_spl_db": sym["tones"][0]["spl_db"],
        "bpf_mode_suppression_db": sym["curle_dipole"]["bpf_mode_suppression_db"],
        "prominence_ratio": sym["bpf_detection"]["prominence_ratio"],
        "relative_error": sym["bpf_detection"]["relative_error"],
    }

    # negative control: broadband-only load must not fake a tone
    t = np.arange(8192) / 600.0
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(t.size)
    freqs, amp = face_spectrum(noise[None, :], 1.0 / 600.0, taper=True)
    neg = find_spectral_peak(freqs, amp[0], f_expected=blade_passing_frequency_hz(180.0, 4))
    return {
        "cases": cases,
        "symmetric_control": symmetric,
        "max_relative_error": max(c["relative_error"] for c in cases),
        "negative_control": {
            "prominence_ratio": neg.prominence_ratio,
            "f_peak_hz": neg.f_peak,
            "relative_error": neg.relative_error,
            "rejected": neg.prominence_ratio < 5.0,
        },
    }


# --------------------------------------------------------------------------- #
# (d) O(N log N) face-level FFT vs O(N^2) direct DFT
# --------------------------------------------------------------------------- #
def fft_cases(*, n_faces: int = 64, n_samples: int = 4096,
              dt: float = 1.0 / 600.0) -> dict[str, Any]:
    """Error and speed of the face-level FFT, plus a scaling study."""
    rng = np.random.default_rng(5)
    t = np.arange(n_samples) * dt
    tone = (np.cos(2 * math.pi * 12.0 * t)[None, :]
            + 0.5 * np.cos(2 * math.pi * 24.0 * t + 0.7)[None, :])
    block = tone + 0.02 * rng.standard_normal((n_faces, n_samples)) \
        + 0.05 * rng.standard_normal(n_faces)[:, None]
    bench = benchmark_fft_vs_direct(block, dt, repeats=3, max_samples=n_samples)

    scaling = []
    for n in (256, 512, 1024, 2048, 4096):
        b = benchmark_fft_vs_direct(block[:, :n], dt, repeats=1, max_samples=n)
        scaling.append({"n_samples": n, "fft_seconds": b.fft_seconds,
                        "direct_seconds": b.direct_seconds, "speedup": b.speedup,
                        "rms_rel_error": b.rms_rel_error})

    n_fit = np.array([s["n_samples"] for s in scaling], dtype=float)
    exp_fft, _ = np.polyfit(np.log(n_fit), np.log([s["fft_seconds"] for s in scaling]), 1)
    exp_dft, _ = np.polyfit(np.log(n_fit), np.log([s["direct_seconds"] for s in scaling]), 1)

    # propagation equivalence: spectral path vs explicit retarded-time summation
    from .api import build_face_sheet
    from .bpf import rotor_face_forces
    positions_src, _n, _a, psi = build_face_sheet(8, 1, 0.6, 0.8)
    tt = np.arange(2048) / 4000.0
    forces = rotor_face_forces(tt, psi, rpm=180.0, blades=4, force_peak_n=250.0,
                               axis=[0.0, 0.0, 1.0])
    d = np.array([0.3, 0.2, 0.93])
    d = d / np.linalg.norm(d)
    freqs = np.fft.rfftfreq(tt.size, d=1.0 / 4000.0)
    f_spec = np.fft.rfft(forces, axis=-1) / tt.size * 2.0
    f_spec[..., 0] *= 0.5
    f_spec[..., -1] *= 0.5
    p_spec = dipole_far_field_spectrum(np.moveaxis(f_spec, -1, 0), freqs, d[None, :],
                                       distance=10.0, medium=Medium(),
                                       positions=positions_src)[:, 0]
    from .face_fft import peak_to_fft_coefficients
    p_spec_t = np.fft.irfft(peak_to_fft_coefficients(p_spec, tt.size), n=tt.size)
    p_ref = dipole_field_time(forces, positions_src, d, 10.0, 1.0 / 4000.0, Medium())
    skip = 200
    prop_err = float(np.sqrt(np.mean((p_spec_t[skip:] - p_ref[skip:]) ** 2))
                     / np.sqrt(np.mean(p_ref[skip:] ** 2)))

    # quadrupole propagation equivalence (second derivative in the time path)
    u = np.zeros((positions_src.shape[0], tt.size, 3))
    u[:, :, 0] = 2.0 * np.cos(2 * math.pi * 40.0 * tt)[None, :]
    u[:, :, 1] = 1.0 * np.sin(2 * math.pi * 40.0 * tt)[None, :]
    T = lighthill_stress_tensor(u, rho=Medium().rho0, c0=Medium().c0)
    q_hist = np.moveaxis(T, 1, -1)                                  # (faces, 3, 3, N)
    q_spec = np.fft.rfft(q_hist, axis=-1) / tt.size * 2.0
    q_spec[..., 0] *= 0.5
    q_spec[..., -1] *= 0.5
    p_q_spec = quadrupole_far_field_spectrum(np.moveaxis(q_spec, -1, 0), freqs, d[None, :],
                                             distance=10.0, medium=Medium(),
                                             positions=positions_src)[:, 0]
    p_q_spec_t = np.fft.irfft(peak_to_fft_coefficients(p_q_spec, tt.size), n=tt.size)
    p_q_ref = quadrupole_field_time(q_hist, positions_src, d, 10.0, 1.0 / 4000.0, Medium())
    prop_q_err = float(np.sqrt(np.mean((p_q_spec_t[skip:] - p_q_ref[skip:]) ** 2))
                       / np.sqrt(np.mean(p_q_ref[skip:] ** 2)))

    return {
        "benchmark": bench.to_dict(),
        "scaling": scaling,
        "scaling_exponents": {"fft": float(exp_fft), "direct_dft": float(exp_dft)},
        "propagation_error_dipole": prop_err,
        "propagation_error_quadrupole": prop_q_err,
        "max_rel_error": bench.max_rel_error,
        "speedup": bench.speedup,
    }


# --------------------------------------------------------------------------- #
# (e) test count
# --------------------------------------------------------------------------- #
def _count_tests(test_file: Path) -> dict[str, Any]:
    if not test_file.exists():
        return {"tests": 0, "source": "missing"}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", str(test_file)],
            capture_output=True, text=True, timeout=300,
        )
        for line in reversed(proc.stdout.strip().splitlines()):
            if "test" in line and "collected" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith("test") and i > 0 and parts[i - 1].isdigit():
                        return {"tests": int(parts[i - 1]), "source": "pytest"}
        for line in reversed(proc.stdout.strip().splitlines()):
            token = line.split(" ")[0].strip()
            if token.isdigit():
                return {"tests": int(token), "source": "pytest"}
    except Exception:  # pragma: no cover - pytest not importable in some sandboxes
        pass
    text = test_file.read_text(encoding="utf-8")
    return {"tests": text.count("def test_"), "source": "source-scan"}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_acceptance(
    *,
    test_file: Path | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Run all five criteria and return the machine-readable acceptance record."""
    quad = compact_quadrupole_case(n_theta_quad=6, n_phi_quad=24,
                                  n_samples=2048 if quick else 4096)
    quad_coarse = compact_quadrupole_case(n_theta_quad=4, n_phi_quad=12,
                                          n_samples=2048 if quick else 4096)
    dip = compact_dipole_case()
    dip_finite = finite_extent_dipole_case()
    bpf = bpf_cases()
    fft = fft_cases(n_faces=32 if quick else 64, n_samples=2048 if quick else 4096)
    tests = _count_tests(test_file) if test_file else {"tests": 0, "source": "not-checked"}

    criteria = [
        Criterion(
            ident="a",
            name="Lighthill quadrupole radiated power vs analytic compact-source value",
            value=quad["relative_error"],
            target=0.10,
            passed=quad["relative_error"] < 0.10,
            comparison="<",
            detail={
                "power_numeric_w": quad["power_numeric_w"],
                "power_theory_w": quad["power_theory_w"],
                "kR": quad["kR"],
                "directivity_max_abs_error": quad["directivity_max_abs_error"],
                "power_error_coarse_quadrature": quad_coarse["relative_error"],
                "quadrature": quad["quadrature"],
            },
        ),
        Criterion(
            ident="b",
            name="Curle dipole directivity vs exact compact dipole (monopole pair)",
            value=max(dip["directivity_max_abs_error"],
                      dip["level_max_relative_error"],
                      dip["power_relative_error"],
                      dip_finite["max_error_kL_over_2_le_0p1"]),
            target=0.10,
            passed=max(dip["directivity_max_abs_error"],
                       dip["level_max_relative_error"],
                       dip["power_relative_error"],
                       dip_finite["max_error_kL_over_2_le_0p1"]) < 0.10,
            comparison="<",
            detail={
                "kd": dip["kd"],
                "complex_max_relative_error": dip["complex_max_relative_error"],
                "level_max_relative_error": dip["level_max_relative_error"],
                "pattern_max_abs_error": dip["directivity_max_abs_error"],
                "power_relative_error": dip["power_relative_error"],
                "reference_power_relative_error": dip["reference_power_relative_error"],
                "power_operator_w": dip["power_operator_w"],
                "power_analytic_w": dip["power_analytic_w"],
                "finite_extent_error_exponent_in_kL":
                    dip_finite["error_exponent_in_kL"],
                "finite_extent_rows": dip_finite["rows"],
            },
        ),
        Criterion(
            ident="c",
            name="BPF peak frequency error in the face-level spectrum",
            value=bpf["max_relative_error"],
            target=0.01,
            passed=bpf["max_relative_error"] < 0.01 and bpf["negative_control"]["rejected"],
            comparison="<",
            detail={
                "cases": [{"rpm": c["rpm"], "blades": c["blades"],
                           "f_bpf_hz": c["f_bpf_hz"], "f_peak_hz": c["f_peak_hz"],
                           "relative_error": c["relative_error"],
                           "prominence_ratio": c["prominence_ratio"]}
                          for c in bpf["cases"]],
                "negative_control": bpf["negative_control"],
                "symmetric_control": bpf["symmetric_control"],
            },
        ),
        Criterion(
            ident="d",
            name="O(N log N) face FFT vs O(N^2) direct DFT (amplitude error < 2 %, speedup >= 10x)",
            value=fft["max_rel_error"],
            target=0.02,
            passed=fft["max_rel_error"] < 0.02 and fft["speedup"] >= 10.0,
            comparison="<",
            detail={
                "speedup": fft["speedup"],
                "rms_rel_error": fft["benchmark"]["rms_rel_error"],
                "n_faces": fft["benchmark"]["n_faces"],
                "n_samples": fft["benchmark"]["n_samples"],
                "scaling_exponents": fft["scaling_exponents"],
                "propagation_error_dipole": fft["propagation_error_dipole"],
                "propagation_error_quadrupole": fft["propagation_error_quadrupole"],
            },
        ),
        Criterion(
            ident="e",
            name="pytest coverage of the capability (>= 12 tests)",
            value=float(tests["tests"]),
            target=12.0,
            passed=int(tests["tests"]) >= 12,
            comparison=">=",
            detail={"source": tests["source"]},
        ),
    ]
    return {
        "node": "P27W6-V1",
        "capability": "vibro-acoustic source terms (Lighthill quadrupole + Curle dipole + BPF)",
        "criteria": [c.to_dict() for c in criteria],
        "all_passed": all(c.passed for c in criteria),
        "cases": {"quadrupole": quad, "quadrupole_coarse": quad_coarse,
                  "dipole": dip, "dipole_finite_extent": dip_finite,
                  "bpf": bpf, "fft": fft, "tests": tests},
        "environment": {
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
    }