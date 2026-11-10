"""Nonlocal granular rheology (NGF) and non-smooth contact dynamics (NSCD).

Modules
-------
``fluidity``      NGF closure: local ``mu(I)`` inversion, cooperativity length,
                  finite-volume operator, CG + Jacobi solver.
``couette``       2D annular (Couette) shear cell producing the rheology data.
``calibration``   two-scale calibration (wide band -> local law, thin band -> xi/d)
                  and pooled closure scoring.
``nscd``          Moreau time stepping: NLGS (sequential / colour-partitioned /
                  Jacobi) with Newton restitution and a Coulomb friction cone.
``validation``    NSCD verification tiers and adjudication vs a soft-sphere model.
``study``         driver: runs, calibration, figures, ``summary.json``.
"""
from .fluidity import (JopLaw, NGFError, NGFParams, NGFResult, RadialGrid,
                       cg_jacobi, local_fluidity, local_strain_rate,
                       mu_from_strain_rate, solve_ngf)

__all__ = ["JopLaw", "NGFError", "NGFParams", "NGFResult", "RadialGrid",
           "cg_jacobi", "local_fluidity", "local_strain_rate",
           "mu_from_strain_rate", "solve_ngf"]
