"""Propeller-ice milling transient dynamics (DEM-scale contact, spindle, debris).

Public API
----------
``PropellerGeometry``, ``IceBlockGeometry``
    Parametric blade and ice-block geometry.
``BladeIceContact``
    Normal crushing, tangential ice-steel friction and chip rolling contact.
``solve_open_water``
    Blade-element / momentum open-water solve used for the shaft load and the
    race velocity that advects the debris.
``MillingConfig``, ``simulate_milling``, ``MillingResult``
    Transient milling simulation: shaft torque ``Q(t)``, thrust ``T(t)``, contact
    force ``F_c(t)``, chip trajectories and the mass/energy ledgers.
``analyse_torque_spectrum``
    BPF/harmonic spectral acceptance analysis of ``Q(t)``.

Typical use::

    from tensordem.propeller_ice import MillingConfig, simulate_milling, analyse_torque_spectrum
    result = simulate_milling(MillingConfig(rpm=60.0))
    analysis = analyse_torque_spectrum(result.time, result.torque_ice,
                                       rpm=60.0, n_blades=4, window=result.steady_mask())
"""

from __future__ import annotations

from .contact import (
    BladeIceContact,
    critical_rolling_angle_deg,
    normal_force,
    normal_pressure,
    rolling_moment,
    tangential_force,
)
from .geometry import IceBlockGeometry, PropellerGeometry
from .hydro import OpenWaterConfig, OpenWaterState, section_coefficients, solve_open_water
from .milling import MillingConfig, MillingResult, simulate_milling
from .spectrum import (
    SpectrumAnalysis,
    analyse_torque_spectrum,
    blade_passing_frequency,
    detrend,
    find_peaks_spectral,
    harmonic_amplitudes,
    harmonic_index,
    hann_window,
    periodogram,
)

__all__ = [
    "BladeIceContact",
    "IceBlockGeometry",
    "MillingConfig",
    "MillingResult",
    "OpenWaterConfig",
    "OpenWaterState",
    "PropellerGeometry",
    "SpectrumAnalysis",
    "analyse_torque_spectrum",
    "blade_passing_frequency",
    "critical_rolling_angle_deg",
    "detrend",
    "find_peaks_spectral",
    "harmonic_amplitudes",
    "harmonic_index",
    "hann_window",
    "normal_force",
    "normal_pressure",
    "periodogram",
    "rolling_moment",
    "section_coefficients",
    "simulate_milling",
    "solve_open_water",
    "tangential_force",
]