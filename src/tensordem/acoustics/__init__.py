"""Vibro-acoustic source terms for DEM face sheets (campaign node P27W6-V1).

Modules
-------
``source_terms``
    Lighthill stress tensor ``T_ij``, compact quadrupole / Curle dipole far
    fields, analytic power references and the oscillating-sphere exact solution.
``face_fft``
    ring buffer of per-face histories, O(N log N) face-level FFT, the O(N^2)
    direct-DFT reference and the explicit retarded-time propagation reference.
``bpf``
    blade-passing frequency ``f_BPF = N Z / 60``, the rotor pulse-train source
    model and spectral peak picking.
``api``
    handler for ``POST /api/v1/dem/acoustic_source``.
"""

from .api import ACOUSTIC_SOURCE_ENDPOINT, build_face_sheet, handle_acoustic_source
from .bpf import (
    ToneDetection,
    blade_passing_frequency_hz,
    default_harmonic_ratios,
    find_spectral_peak,
    rotor_face_forces,
    rotor_face_turbulence,
    shaft_frequency_hz,
    third_octave_levels,
)
from .face_fft import (
    RingBuffer,
    benchmark_fft_vs_direct,
    dipole_field_time,
    direct_dft,
    face_spectrum,
    face_spectrum_complex,
    propagation_equivalence,
    quadrupole_field_time,
)
from .source_terms import (
    Medium,
    acoustic_power_from_spectrum,
    dipole_far_field_spectrum,
    dipole_power_analytic,
    fit_power_law,
    lighthill_stress_tensor,
    dipole_pair_far_field,
    dipole_pair_force,
    dipole_pair_power,
    monopole_far_field,
    pwsl_db,
    quadrupole_directivity_theory,
    quadrupole_far_field_spectrum,
    quadrupole_power_analytic,
    sphere_quadrature,
    spl_db,
    volume_integral_stress,
)

__all__ = [
    "ACOUSTIC_SOURCE_ENDPOINT",
    "Medium",
    "RingBuffer",
    "ToneDetection",
    "acoustic_power_from_spectrum",
    "benchmark_fft_vs_direct",
    "blade_passing_frequency_hz",
    "build_face_sheet",
    "default_harmonic_ratios",
    "dipole_far_field_spectrum",
    "dipole_field_time",
    "dipole_power_analytic",
    "direct_dft",
    "face_spectrum",
    "face_spectrum_complex",
    "find_spectral_peak",
    "fit_power_law",
    "handle_acoustic_source",
    "lighthill_stress_tensor",
    "dipole_pair_far_field",
    "dipole_pair_force",
    "dipole_pair_power",
    "monopole_far_field",
    "propagation_equivalence",
    "pwsl_db",
    "quadrupole_directivity_theory",
    "quadrupole_far_field_spectrum",
    "quadrupole_field_time",
    "quadrupole_power_analytic",
    "rotor_face_forces",
    "rotor_face_turbulence",
    "shaft_frequency_hz",
    "sphere_quadrature",
    "spl_db",
    "third_octave_levels",
    "volume_integral_stress",
]