"""Non-spherical shape representation: analytic contact anchors and clump error bounds.

Campaign node P27W2-N07 (TensorDEM).  Two representations compete for a DEM of
broken ice, rubble or granular cargo:

* the **true shape** -- a superquadric ``|x/a|^n2 + |y/b|^n2)^(n1/n2) + |z/c|^n1 = 1``
  (``src/tensordem/superquadric.py``);
* the **N-sphere clump** -- a rigid union of ``N`` spheres, cheap because every
  contact is analytic.

This subpackage provides the *quantitative* side of that decision:

``hertz_ellipsoid``  : closed-form ellipsoid-on-plane Hertz contact with complete
                       elliptic integrals (stiffness anchor), a quadrature
                       residual verification that does not use those formulas,
                       and the superellipsoid-pole power-law indenter law
                       ``P ~ delta^(1+1/m)`` verified on Love's integral equation.
``clump_fit``        : greedy covering multi-sphere fit and its surface-deviation
                       convergence in ``N``.
``repose_channels``  : one integrator, three particle channels (sphere / clump /
                       true superquadric) for angle-of-repose comparisons, plus
                       the static-pile spurious-dissipation probe.
``verdict``          : the machine-readable verdict artefact
                       ``shape_representation_verdict.json``.
"""

from .hertz_ellipsoid import (  # noqa: F401
    EllipsoidPlaneContact,
    elliptic_terms,
    curvature_ratio_function,
    eccentricity_from_ratio,
    ellipsoid_plane_contact,
    hertz_stiffness,
    force_from_overlap,
    superellipsoid_pole_gap,
    pole_contact_exponent,
    displacement_quadrature,
    verify_closed_form,
    verify_elliptic_relation,
    axisymmetric_indenter_law,
)

__all__ = [
    "EllipsoidPlaneContact",
    "elliptic_terms",
    "curvature_ratio_function",
    "eccentricity_from_ratio",
    "ellipsoid_plane_contact",
    "hertz_stiffness",
    "force_from_overlap",
    "superellipsoid_pole_gap",
    "pole_contact_exponent",
    "displacement_quadrature",
    "verify_closed_form",
    "verify_elliptic_relation",
    "axisymmetric_indenter_law",
]