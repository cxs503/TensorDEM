"""Granular hopper-discharge verification surface for TensorDEM (m7751).

Three physically independent DEM verification faces on **one** particle
inventory and **one** contact law set:

* ``repose``  -- free-surface angle of a collapsed column (``theta_r`` vs
  ``arctan(mu)``, monotonicity in ``mu``, frictionless negative control);
* ``discharge`` -- Beverloo orifice law ``W = C rho_b sqrt(g) (D0 - k d)^(5/2)``
  on a conical hopper for three orifice diameters;
* ``janssen`` -- wall-normal stress saturation ``sigma_w(z) =
  rho_b g R_h / mu_w [1 - exp(-mu_w K z / R_h)]`` for three wall frictions.

The module is deliberately *not* a wrapper around an analytic correlation: the
kernel integrates ``N`` spheres with the TensorDEM linear spring-dashpot +
Cundall-Strack tangential + optional rolling contact law against cylindrical,
conical and flat-plate walls, and every reported number is measured from the
particle trajectory (discharged mass balance, wall contact forces, free
surface geometry).

Sub-modules
-----------
``kernel``    case definition (geometry/params) + vectorised 3-D DEM kernel
``measures``  measurement operators (Beverloo fit, Janssen fit, repose angle)
``campaign``  multi-seed campaign driver + acceptance criteria C1..C9
"""
from __future__ import annotations

from .kernel import (
    GranularParams,
    HopperCase,
    HopperError,
    HopperGeometry,
    lattice_packing,
    simulate,
)
from .measures import (
    beverloo_fit,
    beverloo_scaling_fit,
    janssen_fit,
    janssen_analytic,
    repose_angle,
    saturation_metrics,
)
from .campaign import (
    campaign_plan,
    evaluate_criteria,
    run_case,
)

__all__ = [
    "GranularParams",
    "HopperCase",
    "HopperError",
    "HopperGeometry",
    "lattice_packing",
    "simulate",
    "beverloo_fit",
    "beverloo_scaling_fit",
    "janssen_fit",
    "janssen_analytic",
    "repose_angle",
    "saturation_metrics",
    "campaign_plan",
    "evaluate_criteria",
    "run_case",
]