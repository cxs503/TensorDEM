"""Ice-structure contact & friction verification surface (TensorDEM).

This package is the machine-checkable verification face for *ship ice
resistance produced by discrete-element contact mechanics*:

``contact``   Hertz normal elasticity, ice crushing pressure cap, Coulomb friction
``lattice``   bonded beam lattice (2-D longitudinal ice-sheet slice) with
              separate bending and crushing fracture modes
``bow``       rigid wedge-bow idealisation (rake plane + flat bottom)
``sim``       explicit dynamics + closed energy ledger
``anchors``   in-repository Lindqvist three-component anchor reduced to a slice
``campaign``  parameter campaigns, negative controls and the L1-L8 verdicts

Physical model
--------------
A level ice sheet of thickness ``h`` is discretised in a longitudinal slice of
out-of-plane width ``t`` (``t = h`` by default, matching the Lindqvist anchor's
implicit stem-contact width).  Particles are bonded by frame bonds, so the
sheet carries membrane *and* bending action; bonds fail in two distinct modes
(bending / crushing).  The rigid bow is a rake plane descending aft of the stem,
so the ship pushes the sheet down and forward — the physical icebreaking
mechanism.  Contact is Hertzian with an inelastic crushing pressure cap, so the
dominant dissipation (ice crushing) is measured rather than assumed.

Everything that does work is integrated explicitly, so the energy ledger is a
measurement.  See :mod:`tensordem.ice_structure_contact.sim` for the term list.
"""
from __future__ import annotations

from .anchors import fit_power_law, lindqvist_slice_anchor, relative_deviation, slice_anchor_from_case
from .bow import WedgeBow
from .contact import CrushingContact, CoulombFriction
from .lattice import IceMaterial, IceSheetLattice
from .sim import IceStructureContactSim, SimulationConfig

__all__ = [
    "CrushingContact",
    "CoulombFriction",
    "IceMaterial",
    "IceSheetLattice",
    "IceStructureContactSim",
    "SimulationConfig",
    "WedgeBow",
    "fit_power_law",
    "lindqvist_slice_anchor",
    "relative_deviation",
    "slice_anchor_from_case",
]