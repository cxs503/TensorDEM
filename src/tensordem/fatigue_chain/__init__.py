"""DEM contact-force time history -> rainflow -> S-N -> Miner damage chain.

Public surface
--------------
``contract``  : the F(t) export contract, its admissibility and energy checks.
``resample``  : sampling-rate compression (naive vs peak-preserving).
``response``  : exact SDOF / modal transfer of the force history to stress.
``damage``    : rainflow -> Miner chain, narrow-band Rayleigh compression.
``envelope``  : dD/D error envelope matrix and the minimum-sampling-rate law.
``cases``     : the three verification cases (single impact, two-ball, floe field).
"""

from .contract import (  # noqa: F401
    CORE_COLUMNS,
    ENERGY_COLUMNS,
    HULL,
    SCHEMA_ID,
    ContactForceHistory,
    ForceHistoryError,
    MissingForceHistory,
)
from .damage import (  # noqa: F401
    ChainResult,
    PowerLawSNCurve,
    chain_damage,
    damage_from_stress,
    narrowband_rayleigh_damage,
    statistics_compression,
)
from .envelope import (  # noqa: F401
    SamplingEnvelope,
    minimum_sampling_rate,
    sampling_envelope,
)
from .resample import peak_preserving_decimate, naive_decimate  # noqa: F401
from .response import ModalChannel, modal_stress_history, sdof_response  # noqa: F401

__all__ = [
    "SCHEMA_ID",
    "CORE_COLUMNS",
    "ENERGY_COLUMNS",
    "HULL",
    "ContactForceHistory",
    "ForceHistoryError",
    "MissingForceHistory",
    "ChainResult",
    "PowerLawSNCurve",
    "chain_damage",
    "damage_from_stress",
    "narrowband_rayleigh_damage",
    "statistics_compression",
    "SamplingEnvelope",
    "sampling_envelope",
    "minimum_sampling_rate",
    "peak_preserving_decimate",
    "naive_decimate",
    "ModalChannel",
    "modal_stress_history",
    "sdof_response",
]