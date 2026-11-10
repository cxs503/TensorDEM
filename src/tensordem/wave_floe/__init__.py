"""Wave-loaded ice floe dynamics (wave-ice coupling verification suite).

The shipped wet coupling (:mod:`tensordem.buoyancy_wet`) provides linear Airy
kinematics and a Morison load per particle, but nothing in the tree ties them to
a rigid-body floe response, a sea-state spectrum, a multi-floe scattering model
or a wave-induced breakup threshold.  This package adds that chain and, with it,
the first machine-checkable anchors for *wave* driven ice:

* :mod:`tensordem.wave_floe.spectra` -- JONSWAP / Pierson-Moskowitz spectra,
  harmonic realisation, Welch PSD estimator, spectral fidelity metric;
* :mod:`tensordem.wave_floe.waves` -- finite-depth dispersion (Newton),
  regular and spectral kinematics, group velocity / energy flux;
* :mod:`tensordem.wave_floe.response` -- rigid disk-lattice floe in beam seas:
  time-domain (per-particle buoyancy and Morison load) vs frequency-domain
  (discrete and continuum transfer functions), RAO extraction, energy ledger;
* :mod:`tensordem.wave_floe.array` -- long-wave multiple scattering through a
  spaced floe array (transfer matrix), energy closure, attenuation curve;
* :mod:`tensordem.wave_floe.breakup` -- wave-curvature bending of a floating
  floe on a Winkler foundation -> critical wave height ``H_crit``;
* :mod:`tensordem.wave_floe.validation` -- the C1..C7 criterion runner.

Scope discipline: every anchor states the reference it is measured against.
Where the tree holds no external benchmark (WAMIT/AQWA RAO, digitised
transmission data) the report records an honest WITHHELD instead of a
self-consistent number dressed up as a validation.
"""

from __future__ import annotations

from .spectra import (
    Spectrum,
    jonswap,
    pierson_moskowitz,
    discretize,
    synthesize_eta,
    welch_psd,
    spectral_fidelity,
)
from .waves import (
    RegularWave,
    IrregularWave,
    solve_wavenumber,
    deep_water_wavenumber,
)
from .response import (
    FloeLattice,
    FloeResponseModel,
    FrequencyResponse,
    rao_from_time_series,
)
from .breakup import (
    WinklerBeam,
    critical_wave_height,
    theory_bending_stress,
    characteristic_wavenumber,
    finite_length_correction,
)
from .array import FloeArrayTransmission, transmission_vs_kh
from .validation import run_validation

__all__ = [
    "Spectrum",
    "jonswap",
    "pierson_moskowitz",
    "discretize",
    "synthesize_eta",
    "welch_psd",
    "spectral_fidelity",
    "RegularWave",
    "IrregularWave",
    "solve_wavenumber",
    "deep_water_wavenumber",
    "FloeLattice",
    "FloeResponseModel",
    "FrequencyResponse",
    "rao_from_time_series",
    "WinklerBeam",
    "critical_wave_height",
    "theory_bending_stress",
    "characteristic_wavenumber",
    "finite_length_correction",
    "FloeArrayTransmission",
    "transmission_vs_kh",
    "run_validation",
]