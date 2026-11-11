"""Anisotropic (columnar, S2) sea-ice strength model for TensorDEM.

Campaign node ``p474w1-n08-dem-anisotropic`` (physical domain: columnar sea ice).

What the package provides
-------------------------
``fabric``
    Discrete c-axis ODF (:class:`~tensordem.ice_anisotropy.fabric.FabricODF`),
    the 2nd/4th-order orientation tensors it implies, and a JSON
    record/query ledger (:class:`~tensordem.ice_anisotropy.fabric.FabricLedger`).
``slip_systems``
    Schmid-law, slip-system-resolved reference model (basal + pyramidal glide
    over a measured fabric) -- the independent physics the quadratic surface is
    scored against, with its mean-field and Sachs (lower-bound) conventions.
``yield_surface``
    Transversely isotropic Hill surface (:class:`HillTI`) with
    ``(sigma_parallel, sigma_perp, tau_transverse)``, exact von Mises
    degeneration, and the fabric -> parameters closure
    (:func:`~tensordem.ice_anisotropy.yield_surface.strength_surface_from_fabric`).
``virtual_test``
    Discrete DEM directional uniaxial test used as the engine-level negative
    control.
``verification``
    Every acceptance criterion with its number, threshold and pass flag.
``visualization``
    Pole figures, strength-vs-angle figure, yield loci, DEM saturation curve.

Conventions
-----------
* the **column axis** (grain-growth direction) is the symmetry axis;
* ``theta`` is the load angle from that axis: 0 deg = load along the columns
  (the strong direction), 90 deg = across the columns;
* strengths are positive in compression, stresses in Pa;
* the strength model is pressure-insensitive and rate-independent (limitations
  are listed in the generated report).
"""

from .fabric import FabricLedger, FabricODF, columnar_fabric, cubic_orbit_directions
from .slip_systems import SlipSystemModel, calibrate_pyramidal_ratio, load_direction
from .verification import AnisotropyStudy, default_angles, run_verification
from .virtual_test import DirectionalTestConfig, micro_contrast_sweep, run_directional_test
from .yield_surface import (
    COLUMNAR_RATIO_RANGE,
    HillTI,
    calibrated_slip_reference,
    strength_surface_from_fabric,
)

__all__ = [
    "COLUMNAR_RATIO_RANGE",
    "AnisotropyStudy",
    "DirectionalTestConfig",
    "FabricLedger",
    "FabricODF",
    "HillTI",
    "SlipSystemModel",
    "calibrate_pyramidal_ratio",
    "calibrated_slip_reference",
    "columnar_fabric",
    "cubic_orbit_directions",
    "default_angles",
    "load_direction",
    "micro_contrast_sweep",
    "run_directional_test",
    "run_verification",
    "strength_surface_from_fabric",
]