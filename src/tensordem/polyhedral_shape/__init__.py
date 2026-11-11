"""Polyhedral particle shape characterisation, contact and packing validation.

Public surface
--------------
* :mod:`tensordem.polyhedral_shape.shapes` -- convex polyhedron / ellipsoid
  representation and closed-form shape descriptors (sphericity, roundness,
  aspect ratio, inradius/circumradius, exact volume moments);
* :mod:`tensordem.polyhedral_shape.contact_force` -- Hertzian force layer on top
  of the GJK/EPA geometry of :mod:`tensordem.convex_contact`, with the
  sphere-limit and zero-overlap negative controls;
* :mod:`tensordem.polyhedral_shape.packing` -- shape-dependent packing fraction
  ``phi(N_v, Psi)`` from two explicit protocols (dense compaction vs loose
  deposition);
* :mod:`tensordem.polyhedral_shape.api` -- two HTTP endpoints with 422
  validation.

All quantities are SI.
"""

from .shapes import (
    Ellipsoid,
    PolyhedralShape,
    ShapeDescriptors,
    cube_vertices,
    describe,
    describe_ellipsoid,
    named_shape,
    random_polyhedron,
    regular_dodecahedron_vertices,
    regular_icosahedron_vertices,
    regular_octahedron_vertices,
    regular_tetrahedron_vertices,
    sphericity,
    spherical_polyhedron,
    volume_equivalent_radius,
)
from .contact_force import (
    ContactForce,
    contact_force,
    hertz_force,
    hertz_sphere_sphere,
    normal_force_exponent,
    sphere_limit_convergence,
)
from .packing import (
    PHI_SPHERE_RCP,
    aligned_spheroid_affine_packing,
    PHI_SPHERE_RLP,
    PackingResult,
    pack_compaction_generic,
    pack_deposition,
    pack_spheres_compaction,
    packing_fraction_of_shapes,
    verify_affine_invariance,
)

__all__ = [
    "Ellipsoid",
    "PolyhedralShape",
    "ShapeDescriptors",
    "cube_vertices",
    "describe",
    "describe_ellipsoid",
    "named_shape",
    "random_polyhedron",
    "regular_dodecahedron_vertices",
    "regular_icosahedron_vertices",
    "regular_octahedron_vertices",
    "regular_tetrahedron_vertices",
    "sphericity",
    "spherical_polyhedron",
    "volume_equivalent_radius",
    "ContactForce",
    "contact_force",
    "hertz_force",
    "hertz_sphere_sphere",
    "normal_force_exponent",
    "sphere_limit_convergence",
    "PHI_SPHERE_RCP",
    "aligned_spheroid_affine_packing",
    "PHI_SPHERE_RLP",
    "PackingResult",
    "pack_deposition",
    "pack_spheres_compaction",
    "pack_compaction_generic",
    "packing_fraction_of_shapes",
    "verify_affine_invariance",
]