"""Ice-region structural health monitoring and load-inversion digital twin.

The package implements the inverse chain of the whipping assessment:

    measured strain eps(x_k, t)
        -> modal filter            q_n(t)  = argmin ||eps - H q||^2
        -> load inversion          p_n(t)  = q_n'' + 2 zeta omega q_n' + omega^2 q_n
        -> physical projection     F_ice(t), f_ice(x, t)
        -> rigid/elastic split     < 0.5 Hz (load-following) vs > 0.5 Hz (vibration)
        -> fatigue update          rainflow + DNV-RP-C203 Miner damage, streaming
        -> DEM comparison          correlation / peak / energy error, model correction

Sub-modules
-----------
``model``          hull-girder dry/wet modal ROM and the strain-mode matrix ``H``
``modal_filter``   regularised least-squares inversion strain -> modal coordinates
``load_inversion`` time derivatives, modal force reconstruction, error metrics
``separation``     rigid-body (quasi-static) / elastic band separation
``fatigue_update`` streaming rainflow + Miner damage accumulator
``dem_compare``    DEM vs inverted load comparison and correction trigger
``twin``           online orchestrator (100 Hz chunks, latency accounting)
"""

from .dem_compare import compare_events, event_metrics
from .fatigue_update import FatigueAccumulator, StreamingRainflow
from .load_inversion import (
    LoadInversionResult,
    differentiate,
    invert_modal_force,
    load_error_metrics,
)
from .modal_filter import ModalFilter, ModalFilterResult
from .model import STRAIN_SENSORS, HullGirderROM, SHMConfig
from .separation import RigidElasticSplit, split_rigid_elastic
from .twin import IceLoadTwin, TwinResult

__all__ = [
    "FatigueAccumulator",
    "HullGirderROM",
    "IceLoadTwin",
    "LoadInversionResult",
    "ModalFilter",
    "ModalFilterResult",
    "RigidElasticSplit",
    "SHMConfig",
    "STRAIN_SENSORS",
    "StreamingRainflow",
    "TwinResult",
    "compare_events",
    "differentiate",
    "event_metrics",
    "invert_modal_force",
    "load_error_metrics",
    "split_rigid_elastic",
]