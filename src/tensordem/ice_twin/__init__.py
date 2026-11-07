"""Ice-zone structural-health monitoring & load-inversion digital twin.

Chain (per campaign node IN39-N10):

    strain gauges eps(x_i, t)
        -> modal coordinates q_n(t)            (modal_filter, least squares)
        -> modal forces F_n = m q'' + c q' + k q   (load_inversion)
        -> total ice force f_hat(t) + patch distribution
        -> rigid-body / elastic split at 0.5 Hz (separation)
        -> incremental Miner damage D(t)       (fatigue_update, windowed rainflow)
        -> online comparison vs the DEM forecast (dem_compare)

The structural model is the analytic free-free beam ROM of :mod:`.rom`
(dry or wet modes); the exact Duhamel integrator and the ASTM E1049-85 /
DNV-RP-C203 machinery are reused from :mod:`tensordem.whipping` (p25/p29).
"""

from .dem_compare import LoadComparison, compare_loads, correction_needed
from .fatigue_update import IncrementalFatigue, turning_points_indexed
from .load_inversion import (differentiate, energy_error, invert_modal_force,
                            peak_error, reconstruct_total_force, smooth)
from .modal_filter import ModalFilter, fit_sinusoid
from .rom import FREE_FREE_BETA_L, FreeFreeBeamROM, half_sine_patch
from .separation import RigidElasticSplit, rigid_elastic_split
from .twin import IceTwin, TwinStep

__all__ = [
    "FREE_FREE_BETA_L",
    "FreeFreeBeamROM",
    "IceTwin",
    "IncrementalFatigue",
    "LoadComparison",
    "ModalFilter",
    "RigidElasticSplit",
    "TwinStep",
    "compare_loads",
    "correction_needed",
    "differentiate",
    "energy_error",
    "fit_sinusoid",
    "half_sine_patch",
    "invert_modal_force",
    "peak_error",
    "reconstruct_total_force",
    "rigid_elastic_split",
    "smooth",
    "turning_points_indexed",
]
