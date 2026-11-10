"""Particle-level construction for the DEM layer.

Currently exposes the rigid multi-sphere ("clump") particle family used to
represent non-spherical / polyhedral-like grains:

* :mod:`tensordem.particle.clump` -- clump geometry (ellipsoidal templates with
  an explicit aspect ratio, single-sphere degeneration) and *union* mass
  properties (overlap counted once) verified against analytic sphere values.
* :mod:`tensordem.particle.clump_dem` -- the minimal but complete DEM engine
  (rigid-body clump dynamics, member-sphere contacts, energy ledger) together
  with the two macroscopic protocols used by the validation face: the
  lifted-box angle-of-repose test and the piston compression / force-chain test.
"""
from .clump import (  # noqa: F401
    ClumpParticle,
    SphereMember,
    clump_extents,
    ellipsoid_clump,
    equal_volume_sphere,
    polyhedron_clump,
    ellipsoid_volume,
    member_sum_mass_properties,
    quat_from_axis_angle,
    quat_integrate,
    quat_mul,
    quat_rotate,
    quat_to_matrix,
    random_quaternions,
    single_sphere,
    voxel_mass_properties,
)

__all__ = [
    "ClumpParticle",
    "SphereMember",
    "clump_extents",
    "ellipsoid_clump",
    "equal_volume_sphere",
    "polyhedron_clump",
    "ellipsoid_volume",
    "member_sum_mass_properties",
    "quat_from_axis_angle",
    "quat_integrate",
    "quat_mul",
    "quat_rotate",
    "quat_to_matrix",
    "random_quaternions",
    "single_sphere",
    "voxel_mass_properties",
]