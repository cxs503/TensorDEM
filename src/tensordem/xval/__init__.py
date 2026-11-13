"""Cross-validation (XVAL) of TensorDEM laws against open-source implementations.

The reference laws in this package are *transcribed from the upstream source*
(files, commit and sha256 recorded in ``docs/xval/oss_snapshot/PROVENANCE.json``),
not from a paper summary: every formula carries the ``file:line`` it comes from.
"""
from .oss_rolling import (
    OVERLAP_TOL,
    SpringRolling,
    cdt_torque,
    oss_epsd2,
    rayleigh_timestep,
    tensordem_kernel,
    tangential_part,
)

__all__ = [
    "OVERLAP_TOL",
    "SpringRolling",
    "cdt_torque",
    "oss_epsd2",
    "rayleigh_timestep",
    "tensordem_kernel",
    "tangential_part",
]