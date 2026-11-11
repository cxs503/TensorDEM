"""Periodic inclined-plane (chute) granular flow -- cross-validation case.

Implements the geometry and observables of Silbert et al., Phys. Rev. E **64**,
051302 (2001) with a single case specification that is shared with the
open-source reference codes used for the lateral (cross-tool) comparison; see
:mod:`tensordem.chute_flow.spec` for the parameter set, :mod:`.engine` for the
solver and :mod:`.analysis` for the tool-neutral post-processing.
"""
from __future__ import annotations

from .spec import ChuteCase, DEFAULT_CASE, make_base, make_initial_particles
from .engine import ChuteEngine
from .analysis import (average_profile_over_window, bagnold_fit, bin_profile,
                       classify_run, h_stop_from_scan, read_dump)

__all__ = [
    "ChuteCase", "DEFAULT_CASE", "make_base", "make_initial_particles",
    "ChuteEngine", "average_profile_over_window", "bagnold_fit", "bin_profile",
    "classify_run", "h_stop_from_scan", "read_dump",
]