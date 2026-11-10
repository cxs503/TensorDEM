"""Strongly nonlinear granular-chain waves and granular-layer shock isolation.

Modules
-------
``nlsm``    analytic Nesterenko ``cos^4`` solitary wave (speed ``~F^{1/6}``,
            amplitude-independent width ``L = a sqrt(3)``) derived from the
            dispersion-corrected continuum limit, plus the linear (precompressed)
            sound speed limit and the log-log / exponential fits.
``chain``  1D Hertzian sphere chain, velocity Verlet with event restitution,
            energy ledger, contact peak-force/arrival ledger, strain snapshots.
``measure``arrival-time wave speed, profile L1 (contact sampled, even part),
            pulse width, disorder attenuation length.
``layer``  2D frictionless Hertzian packing between a driven base and a lid
            ("particle impact damper"), transmissibility ``T(f)``, SDOF ``zeta_eff``
            identification (energy ledger + curve fit) and the stop band.
``study``  campaign runners + the acceptance protocol of the p307w1-n06 node.

Quick start
-----------
>>> from tensordem.granular_chain import ChainConfig, HertzianChain, nesterenko_speed
>>> chain = HertzianChain(ChainConfig(n=40, impact_velocity=1.0))
>>> rec = chain.run()                       # doctest: +SKIP
>>> rec["arrival_speed_m_s"]                # doctest: +SKIP
"""
from __future__ import annotations

from .chain import ChainConfig, HertzianChain, run_chain
from .layer import (
    GranularLayer,
    LayerConfig,
    energy_method_zeta,
    fit_sdof_to_curve,
    sdof_transmissibility,
    stopband,
)
from .measure import (
    arrival_speed,
    best_width_profile_l1,
    peak_amplitude,
    profile_l1,
    profile_l1_force,
    pulse_width_measured,
    track_speed,
    window_for,
)
from .nlsm import (
    NesterenkoWave,
    estimate_run_time,
    exponential_decay_length,
    hertz_stiffness,
    impact_peak_force,
    linear_sound_speed,
    loglog_slope,
    nesterenko_peak_force_from_speed,
    nesterenko_profile_force,
    nesterenko_profile_strain,
    nesterenko_pulse_duration,
    nesterenko_speed,
    nesterenko_speed_from_strain,
    nesterenko_width,
)
from .study import (
    LayerStudy,
    LayerStudyConfig,
    SolitaryWaveStudy,
    SolitaryWaveStudyConfig,
    evaluate_layer,
    evaluate_solitary_wave,
    scaling_argument,
    speed_from_monitors,
    symmetrised_L1,
)

__all__ = [
    "ChainConfig", "HertzianChain", "run_chain",
    "GranularLayer", "LayerConfig", "energy_method_zeta", "fit_sdof_to_curve",
    "sdof_transmissibility", "stopband",
    "arrival_speed", "best_width_profile_l1", "peak_amplitude", "profile_l1",
    "profile_l1_force", "pulse_width_measured", "track_speed", "window_for",
    "NesterenkoWave", "estimate_run_time", "exponential_decay_length",
    "hertz_stiffness", "impact_peak_force", "linear_sound_speed", "loglog_slope",
    "nesterenko_peak_force_from_speed", "nesterenko_profile_force",
    "nesterenko_profile_strain", "nesterenko_pulse_duration", "nesterenko_speed",
    "nesterenko_speed_from_strain", "nesterenko_width",
    "LayerStudy", "LayerStudyConfig", "SolitaryWaveStudy",
    "SolitaryWaveStudyConfig", "evaluate_layer", "evaluate_solitary_wave",
    "scaling_argument", "speed_from_monitors", "symmetrised_L1",
]