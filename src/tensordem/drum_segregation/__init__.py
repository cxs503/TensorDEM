"""Rotating-drum bidisperse granular segregation verification surface (p176w2-n08).

Sub-modules
-----------
``kernel2d``  two-dimensional soft-sphere DEM of a rotating drum (real physics)
``metrics``   segregation indices + permutation nulls (direction, degree)
``axial``     reduced-order axial banding model with analytic anchors
``sweeps``    declaration of the case matrix (pre-registered before the run)
``verification``  machine-checkable criteria C1-C6 over the case records
"""
from .kernel2d import DrumConfig, DrumRun, simulate, make_bed          # noqa: F401
from . import axial, metrics, sweeps, verification                     # noqa: F401

__all__ = ["DrumConfig", "DrumRun", "simulate", "make_bed",
           "axial", "metrics", "sweeps", "verification"]
