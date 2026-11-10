"""Floe-size-distribution evolution: DEM-calibrated breakage kernel + power law.

Package map
-----------
* :mod:`tensordem.fsd_evolution.cascade` -- DEM fragment cascade: measures
  ``parent -> daughters`` split events and the energy ledger of a
  quasi-static lattice-fracture run.
* :mod:`tensordem.fsd_evolution.kernel` -- turns the event catalogue into the
  breakage kernel ``K(d, d') = nu(d') beta(d | d')`` (daughter distribution,
  breakage-rate hazard, self-similarity and held-out residual tests).
* :mod:`tensordem.fsd_evolution.pbe` -- mass-conserving fixed-pivot population
  balance driven by that kernel, with analytic negative controls.
* :mod:`tensordem.fsd_evolution.study` -- campaign node driver: acceptance
  criteria, figures, JSON ledger and the paper-level report.
"""
from __future__ import annotations

from .cascade import (
    CascadeConfig,
    CascadeResult,
    FragmentationCascade,
    SplitEvent,
    fragment_perimeter,
    run_cascade,
)

__all__ = [
    "CascadeConfig",
    "CascadeResult",
    "FragmentationCascade",
    "SplitEvent",
    "fragment_perimeter",
    "run_cascade",
]