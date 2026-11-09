"""Freeze bonds and ice arches: from a single bond to a self-sustaining arch.

The package has four layers, each with its own machine-checkable face:

``constitutive``
    :class:`FreezeBondLaw` -- the closed-form strength law
    ``sigma_fb = A dT^alpha (1 + beta S)^-1 t^gamma`` with the shear ratio and the
    fracture energy, in SI.
``cohesive``
    :class:`CohesiveBondSet` -- the vectorised mixed-mode cohesive bond element:
    damage history variable, linear softening that spends exactly ``G_f``,
    incremental work/dissipation ledger, and re-freezing (a reversible strength
    history that keeps the dissipated energy).
``assembly``
    bonded disk lattices (channel of ice rubble between frictional walls;
    confined biaxial cell), Cundall-Strack contacts and walls, internal forces.
``solver``
    quasi-static displacement-controlled loading with a closed energy ledger.
``force_network`` / ``arch``
    strong-force loops and the arch observables (cyclomatic number, spanning
    loops, load transmission to the downstream load cell).
``experiments``
    the campaign drivers: single-bond self-consistency, multi-bond energy
    closure, biaxial and channel tests as a function of ``kappa = sigma_fb /
    sigma_bulk``, the ``kappa_c`` detection and the self-sustain protocol.

Evidence grades are stated per module.  The freeze-bond *law* is a declared
hypothesis (self-consistent, not physically validated); the energy ledger and the
DEM bookkeeping are verified by ``tests/test_freeze_bond_*.py``.
"""
from __future__ import annotations

from .constitutive import FREEZE_BOND_LAW_REFERENCE, FreezeBondLaw
from .cohesive import BondEnvironment, CohesiveBondSet
from .assembly import (
    BondTopology,
    BondedSpecimen,
    FrictionState,
    SpecimenConfig,
    Wall,
    build_box_specimen,
    build_channel_specimen,
    bond_separations,
    commit_friction,
    contact_pairs,
    internal_forces,
)
from .solver import LoadingResult, StepRecord, load_steps, newton_solve, tangent_matrix
from .force_network import ForceNetwork, analyze_network, strong_force_subgraph

__all__ = [
    "FREEZE_BOND_LAW_REFERENCE",
    "FreezeBondLaw",
    "BondEnvironment",
    "CohesiveBondSet",
    "BondTopology",
    "BondedSpecimen",
    "SpecimenConfig",
    "FrictionState",
    "Wall",
    "build_box_specimen",
    "build_channel_specimen",
    "bond_separations",
    "commit_friction",
    "contact_pairs",
    "internal_forces",
    "LoadingResult",
    "StepRecord",
    "load_steps",
    "newton_solve",
    "tangent_matrix",
    "ForceNetwork",
    "analyze_network",
    "strong_force_subgraph",
]