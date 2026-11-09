"""Floe-size-distribution evolution: collision / breakage kernels and the
population balance equation, closed inside TensorDEM.

The campaign node (``p80w2-n05``) asks for the first *statistical-ensemble*
capability of this repository: not one floe, not one indentation event, but the
evolution of the floe size distribution ``n(D, t)`` of a broken-ice field.  The
causal chain is closed with DEM measurements only:

``soft_sphere``
    periodic soft-sphere DEM (optional breakable bonds) that *measures*
    contact-birth statistics, force-shell statistics and aggregates' fragment
    census.  It is the instrument, not a prescription.
``kernels``
    turns those measurements into the two closures the population balance needs:
    the Smoluchowski collision kernel ``K(D1, D2) = pi/4 (D1+D2)^2 <|v_rel|> g``
    and the breakage kernel ``B(D) = k_break D^alpha E^beta``, each with the
    DEM-measured data it was fitted or validated against.
``population_balance``
    fixed-pivot sectional solver of
    ``dn(D,t)/dt = C[n] - B[n] + S`` with exact mass conservation, inversion of a
    DEM census history into ``n(D, t)`` and the forecast/measurement comparison.
``hopper``
    the model-test anchor used for the size-scale comparison, with its evidence
    grade stated explicitly (raw experimental census: withheld).
``study``
    the production driver that runs the three DEM experiments with several seeds,
    assembles the ``C1..C7`` ledger and writes the JSON artefact and figures.

Everything is non-dimensional (see :mod:`tensordem.floe_population.soft_sphere`);
only ratios and dimensionless groups are claimed.
"""

from __future__ import annotations

from .soft_sphere import (
    ClusterCensus,
    ContactModel,
    SphereAssembly,
    build_floe,
    maxwellian_velocities,
    pair_distribution,
    place_floes,
    rotate_about,
)

__all__ = [
    "ClusterCensus",
    "ContactModel",
    "SphereAssembly",
    "build_floe",
    "maxwellian_velocities",
    "pair_distribution",
    "place_floes",
    "rotate_about",
]