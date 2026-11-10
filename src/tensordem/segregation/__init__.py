"""``__init__`` for the size-segregation validation surface (TensorDEM).

Two public surfaces live under this package:

* the **modular** bed/analysis surface (:mod:`packing`, :mod:`metrics`,
  :mod:`bed2d`, :mod:`analysis`, :mod:`transfer`, :mod:`campaign`), and
* the original **flat** binary-bed driver, kept importable as
  ``tensordem.segregation.legacy`` and re-exported here so that the historical
  ``from tensordem.segregation import run_segregation`` entry point keeps
  working after the flat module was folded into this package.
"""

from .packing import (
    bidisperse_radii,
    large_volume_fraction,
    mixture_area,
    random_sequential_pack,
)
from .metrics import (
    BedStatistics,
    mass_moment_audit,
    particle_measure,
    rise_velocity,
    segregation_indices,
    zonal_masses,
)
from .bed2d import BidisperseBed, BoxTooSmall, channel_walls

# Flat (legacy) driver surface -- must stay importable from the package root,
# otherwise ``tensordem.segregation`` shadowing the old flat module would break
# every existing caller (see tests/test_segregation.py).
from .legacy import (
    BinaryMaterial,
    SegregationResult,
    amplitude_from_gamma,
    build_pairs_poly,
    compute_gamma,
    compute_mass_conservation,
    compute_segregation_index,
    compute_tcg_gm_offset,
    contact_forces_poly,
    init_binary_bed,
    run_segregation,
    step_vv,
    vibration_base_vel,
    vibration_base_z,
    wall_forces,
)

__all__ = [
    # modular surface
    "bidisperse_radii",
    "large_volume_fraction",
    "mixture_area",
    "random_sequential_pack",
    "BedStatistics",
    "mass_moment_audit",
    "particle_measure",
    "rise_velocity",
    "segregation_indices",
    "zonal_masses",
    "BidisperseBed",
    "BoxTooSmall",
    "channel_walls",
    # legacy flat surface
    "BinaryMaterial",
    "SegregationResult",
    "amplitude_from_gamma",
    "build_pairs_poly",
    "compute_gamma",
    "compute_mass_conservation",
    "compute_segregation_index",
    "compute_tcg_gm_offset",
    "contact_forces_poly",
    "init_binary_bed",
    "run_segregation",
    "step_vv",
    "vibration_base_vel",
    "vibration_base_z",
    "wall_forces",
]