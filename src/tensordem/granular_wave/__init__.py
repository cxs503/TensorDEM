"""Micro-to-macro wave-speed toolkit for granular media (Hertz-Mindlin).

The package connects DEM contact parameters ``(k_n, k_t, mu, Z)`` to the
macroscopic elastic wave speeds / attenuation of a confined granular packing
and tests the Hertz-Mindlin effective-medium prediction (Digby 1981).

Modules
-------
``engine``  : Hertz-Mindlin packing + servo confining pressure + Z / stress
``emt``     : Digby (1981) / Hertz-Mindlin effective-medium predictions
``measure`` : travel-time (time domain) and sweep (frequency domain) wave speeds
``study``   : campaign runner (pressure scan, two-method cross-check, controls)
"""
from __future__ import annotations

from .engine import GranularWaveConfig, GranularPacking, build_packing
from .emt import digby_hertz_mindlin, poisson_from_velocities, moduli_from_velocities
from .measure import (
    PulseResult,
    SweepResult,
    pulse_wave_speed,
    sweep_wave_speed,
    cross_correlation_lag,
)
from .study import WaveSpeedStudy, WaveStudyConfig, run_wave_speed_study

__all__ = [
    "GranularWaveConfig",
    "GranularPacking",
    "build_packing",
    "digby_hertz_mindlin",
    "poisson_from_velocities",
    "moduli_from_velocities",
    "PulseResult",
    "SweepResult",
    "pulse_wave_speed",
    "sweep_wave_speed",
    "cross_correlation_lag",
    "WaveSpeedStudy",
    "WaveStudyConfig",
    "run_wave_speed_study",
]