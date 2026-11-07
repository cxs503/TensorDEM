"""Ice-structure contact mechanics: pressure field statistics and shape proxies.

Campaign node **IN27-S2** (TensorDEM).  The package closes three gaps in the
repository's ice-resistance chain:

``shapes``
    faceted ice fragments (convex polyhedra) versus their equal-volume sphere
    proxies, including the analytic rolling-resistance coefficient of a faceted
    body, ``mu_geo = <s_e / h_f>`` (area weighted).
``sim``
    a rigid **structure plane** (:math:`x=0`) loaded by an ice layer driven by a
    platen, i.e. the crushing-contact experiment whose wall traction is the
    quantity of engineering interest.  Contacts are elastic with a
    crushing-pressure cap ``sigma_c``; a spherical proxy emulates angularity
    with a rolling-resistance couple ``mu_r R F_n``.
``statistics``
    pressure fields ``p(y, z, t)`` on the structure, their spatial
    autocorrelation ``R(dy, dz)``, correlation length ``L_c``, spatial peak
    factor ``gamma`` and a Gumbel block-maxima extrapolation for design.
``proxy``
    the systematic bias ``delta(mu_r) = (F_poly - F_sphere) / F_poly`` of the
    sphere + ``mu_r`` proxy, its squared correlation with the polyhedral truth,
    the calibrated equivalence ``mu_r*``, and a self-contained Saltelli/Sobol
    global sensitivity analysis of the bias.

All numbers quoted by the modules are produced by the simulation itself; the
literature anchors are carried as *comparison* bands, never as fitting targets.
"""

from .shapes import (
    FragmentShape,
    fragment_volumes,
    polyhedron_shape,
    shape_pair,
    sphere_shape,
)
from .sim import CrushParams, IceWallCrush, pack_slab
from .statistics import (
    PressureStatistics,
    block_maxima,
    gumbel_fit,
    pressure_statistics,
    spatial_autocorrelation,
)
from .proxy import (
    ProxyBias,
    block_average,
    geometric_rolling_reference,
    proxy_bias,
    saltelli_sobol,
)

__all__ = [
    "FragmentShape",
    "sphere_shape",
    "polyhedron_shape",
    "shape_pair",
    "fragment_volumes",
    "CrushParams",
    "IceWallCrush",
    "pack_slab",
    "PressureStatistics",
    "pressure_statistics",
    "spatial_autocorrelation",
    "block_maxima",
    "gumbel_fit",
    "ProxyBias",
    "proxy_bias",
    "block_average",
    "geometric_rolling_reference",
    "saltelli_sobol",
]