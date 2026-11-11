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
# ---------------------------------------------------------------------------
# m5754 / IN34-W3-N08 rolling-friction proxy & contact-pressure statistics.
#
# Formerly the top-level module ``src/tensordem/ice_structure_contact.py``.  A
# regular package directory shadows a same-named module, so after the m9819
# package landed that module became unreachable and every one of its 16 public
# contract symbols (plus 5 helpers the m5754 test imports) raised ImportError:
# ``tests/test_ice_structure_contact.py`` could not be collected at all.
#
# Fix (m14938, same class as m14449's freeze_bond repair): move the legacy
# module inside the package as :mod:`.rolling_proxy` and re-export its contract
# here, so the historical import path keeps working and the 29 m5754 tests run.
# ---------------------------------------------------------------------------
from .rolling_proxy import (
    ContactPressureField,
    IdentifiabilityResult,
    MonotonicScanResult,
    PolyhedronContactResult,
    RollingBiasResult,
    SpatialCorrelationResult,
    compute_contact_pressure_field,
    identifiability_analysis,
    ks_test_seeds,
    monotonic_scan,
    peak_factor,
    polyhedron_contact_force,
    polyhedron_hertz_degeneracy,
    rolling_bias_quantification,
    run_full_verification,
    spatial_correlation,
    # helpers used directly by tests/test_ice_structure_contact.py
    _estimate_angularity,
    _generate_contact_network,
    _is_sphere_vertices,
    _polyhedron_vertices_angular,
    _polyhedron_vertices_sphere,
)

__all__ += [
    "ContactPressureField",
    "IdentifiabilityResult",
    "MonotonicScanResult",
    "PolyhedronContactResult",
    "RollingBiasResult",
    "SpatialCorrelationResult",
    "compute_contact_pressure_field",
    "identifiability_analysis",
    "ks_test_seeds",
    "monotonic_scan",
    "peak_factor",
    "polyhedron_contact_force",
    "polyhedron_hertz_degeneracy",
    "rolling_bias_quantification",
    "run_full_verification",
    "spatial_correlation",
]
