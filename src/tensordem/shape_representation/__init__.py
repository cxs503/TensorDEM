"""Non-spherical shape representation: analytic contact anchors and clump error bounds.

Campaign nodes P27W2-N07 and P40-N10 (TensorDEM).  Two representations compete for
a DEM of broken ice, rubble or granular cargo:

* the **true shape** -- a superquadric ``(|x/a|^n2 + |y/b|^n2)^(n1/n2) + |z/c|^n1 = 1``
  (``src/tensordem/superquadric.py``);
* the **N-sphere clump** -- a rigid union of ``N`` spheres, cheap because every
  contact is analytic.

This subpackage provides the *quantitative* side of that decision:

``hertz_ellipsoid``    : closed-form ellipsoid-on-plane Hertz contact with complete
                         elliptic integrals (stiffness anchor), a quadrature
                         residual verification that does not use those formulas,
                         and the superellipsoid-pole power-law indenter law
                         ``P ~ delta^(1+1/m)`` verified on Love's integral equation.
``elliptic_hertz``     : the P40-N10 contact contract -- closed form vs. an
                         independent Boussinesq-quadrature reference, plus the
                         no-name isotropic equivalent-radius shortcut whose error
                         the node quantifies.
``clump_fit``          : greedy covering multi-sphere fit and its surface-deviation
                         convergence in ``N``.
``clump_inscribed``    : inscribed (one-sided tangent) clump fit and its two-sided
                         surface error and volume fill.
``repose_channels``    : one integrator, three particle channels (sphere / clump /
                         true superquadric) for angle-of-repose comparisons, plus
                         the static-pile spurious-dissipation probe.
``pile2d``             : the 2-D pile test bed (``sphere`` channel), the free
                         surface repose estimator and the study driver.
``verdict``            : the machine-readable verdict artefact
                         ``shape_representation_verdict.json``.
"""

from ..superquadric import Superquadric  # noqa: F401
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
from .elliptic_hertz import (  # noqa: F401
    EllipticHertzResult,
    complete_elliptic_KE,
    elliptic_hertz,
    elliptic_hertz_quadrature,
    equivalent_radius_hertz,
    legendre_relation,
)
from .clump_inscribed import (  # noqa: F401
    ClumpDeviation,
    boundary_radius,
    clump_surface_deviation,
    fibonacci_directions,
    fit_inscribed_clump,
    inscribed_radius,
    superquadric_volume,
)
from .pile2d import (  # noqa: F401
    _Pile,                       # the test suite drives the integrator directly
    PileConfig2D,
    PileResult,
    repose_angle_from_pile,
    run_pile,
)

__all__ = [
    "Superquadric",
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
    "EllipticHertzResult",
    "complete_elliptic_KE",
    "elliptic_hertz",
    "elliptic_hertz_quadrature",
    "equivalent_radius_hertz",
    "legendre_relation",
    "ClumpDeviation",
    "boundary_radius",
    "clump_surface_deviation",
    "fibonacci_directions",
    "fit_inscribed_clump",
    "inscribed_radius",
    "superquadric_volume",
    "PileConfig2D",
    "PileResult",
    "repose_angle_from_pile",
    "run_pile",
]