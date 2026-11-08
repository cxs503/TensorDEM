"""Wet-ice particle cohesion for TensorDEM (P35-N2).

Physical scope
--------------
Dry ice granules interact through repulsion + friction only. Once a liquid phase
is present (pore water, brine, slush, melt layer) pendular **liquid bridges**
form at the particle necks and add a tensile/cohesive capacity.  This package
owns that ingredient for TensorDEM: it does not re-implement contact mechanics
(:mod:`tensordem.hertz_mindlin` / :mod:`tensordem.dem3d` keep their roles), it
only supplies the *bridge force law* plus the DEM-level cohesion assembly and the
validation campaign that accompanies it.

Sub-modules
-----------
``capillary``
    Axial capillary force of one pendular bridge: a calibrated closed form
    (DEM-grade, ~0.3 % peak error against the reference) plus the two anchors
    used to judge it — the classical small-bridge limit ``2 pi gamma R cos(theta)``
    and the exact axisymmetric Young-Laplace solution already shipped in
    :mod:`tensordem.liquid_bridge`.
``aggregation``
    Shear-flow agglomeration: particle-resolved cluster kinetics against the
    Smoluchowski coagulation equation.
``cohesion_cell``
    Cohesive 2D DEM tension cell used to measure the cohesion of wet vs dry
    granular ice (a constant-stress shear variant was attempted and abandoned --
    see the module docstring and the report, section 5).
``ice_resistance``
    Ice-resistance enhancement factor ``F_wet/F_dry`` from the measured cohesion
    and its micro-mechanical / Timco-Weeks reference branches.
``study``
    The validation campaign driver: every number in ``campaign.json`` is produced
    there.  ``figures`` renders the report figures from that JSON.

Evidence grading (per ecosystem convention)
-------------------------------------------
REAL : capillary closed form + exact Young-Laplace reference + DEM tension cell.
        Every number in the campaign JSON is produced by code in this package.
MOCK : none.
STUB : none.
WITHHELD : the *experimental* wet-vs-dry ice-rubble resistance ratio.  Timco &
        Weeks (2010) is a sea-ice property compendium and does not tabulate a
        granular wet/dry resistance ratio; the campaign therefore reports its
        comparison against the Timco-Weeks porosity/strength branch explicitly
        as a literature-relation comparison, not as an independent experiment
        (see ``docs/wet_ice_cohesion_P35N2.md`` section 6).
"""
from __future__ import annotations

from . import aggregation, capillary, cohesion_cell, ice_resistance, study  # noqa: F401

__all__ = ["aggregation", "capillary", "cohesion_cell", "ice_resistance", "study"]