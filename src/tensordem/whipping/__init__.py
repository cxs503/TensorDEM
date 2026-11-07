"""Ice-induced whipping: DEM contact pulses -> wet modes -> stress -> fatigue.

Public entry points::

    from tensordem.whipping import ShipCase, dem_contact_force, assess

    t, f, info = dem_contact_force()          # >= 1 kHz DEM ice-load record
    result = assess(t, f, ShipCase())         # full chain + acceptance criteria

Sub-modules
-----------
``pulses``    pulse segmentation and four-feature parametrisation
``sections``  2-D BEM section added mass (verified against closed forms)
``modes``     beam-FEM dry modes and the wet eigenproblem
``response``  piecewise-exact Duhamel modal superposition, filters, ledger
``rainflow``  ASTM E1049-85 cycle counting
``fatigue``   DNV-RP-C203 S-N curves and Miner damage
``dem_load``  DEM ice-load record generator
``pipeline``  the end-to-end assessment chain
"""

from .dem_load import DEMLoadConfig, dem_contact_force
from .fatigue import DNV_CURVES, SNCurve, miner_damage
from .modes import DryModes, WetModes, dry_modes, wet_modes
from .pipeline import ShipCase, assess, hull_added_mass_per_length
from .pulses import ParametricPulse, PulseFeatures, detect_pulses, resample_uniform
from .rainflow import rainflow as rainflow_cycles
from .sections import added_mass_per_length
from .response import (
    bandpass,
    energy_ledger,
    lowpass,
    modal_response,
    rms,
    stress_history,
)

__all__ = [
    "DEMLoadConfig",
    "DNV_CURVES",
    "DryModes",
    "ParametricPulse",
    "PulseFeatures",
    "SNCurve",
    "ShipCase",
    "WetModes",
    "added_mass_per_length",
    "assess",
    "bandpass",
    "dem_contact_force",
    "detect_pulses",
    "dry_modes",
    "energy_ledger",
    "hull_added_mass_per_length",
    "lowpass",
    "miner_damage",
    "modal_response",
    "rainflow_cycles",
    "resample_uniform",
    "rms",
    "stress_history",
    "wet_modes",
]