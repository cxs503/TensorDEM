"""Shape-resolved FEM -> DEM handoff for level-ice fracture (campaign p302w2-n03).

What this package adds to the first handoff
-------------------------------------------
Campaign ``p302w1-n03`` established the *contract*: a Kirchhoff plate FEM solves
the bending failure of a level ice sheet, the fracture faces and floe polygons go
into ``failure_faces.json``, and the DEM side proves from the file alone that mass
and area are conserved (one equivalent disc per floe, zero velocity).

This package takes the same contract and pushes on the two things a handoff is
*for*:

* **shape** -- one disc of equivalent area is a mass carrier, not a fragment.  A
  floe is now seeded with an inscribed, mutually non-overlapping clump of discs
  (:mod:`.clump`), and the comparison with the equivalent disc is measured on a
  raster (``coverage``, ``iou``, ``spill_area``, inter-floe overlap) instead of
  being asserted.  The exact polygon footprint still travels with the body.
* **motion** -- a quasi-static FEM leaves the floes with zero velocity, which is
  wrong: the plate is bending when it breaks.  A per-floe rigid-plane fit of the
  nodal deflection field (:mod:`.kinematics`) transfers a heave offset, a heave
  velocity at the loading rate and pitch/roll rates, and reports the *rigidity
  residual* -- the part of the floe's own bending a rigid body cannot carry, which
  is the number that decides whether the discrete model is legitimate at all.

:mod:`.dynamics` then integrates the seeded bodies (heave oscillator with the
closed-form frequency ``sqrt(rho_w g / (rho_i t))``, and in-plane drag relaxation
against an exponential) and closes the books again *after* the run.

Artifacts of the production run (``tensordem.fem_dem_seeding.study``):
``failure_faces.json``, ``fem_node_state.json``, ``dem_bodies.json``,
``report_data.json`` and five figures.
"""
from __future__ import annotations

from .raster import Raster, polygon_mask, disc_mask, footprint_metrics, resolution_check
from .clump import (
    Disc,
    FloeBody,
    seed_inscribed_discs,
    seed_equivalent_disc,
    seed_floe_bodies,
    overlap_audit,
    body_ledger,
)
from .kinematics import (
    BodyKinematics,
    RigidPlaneFit,
    rigid_plane_fit,
    floe_node_mask,
    kinematic_transfer,
    read_node_state,
)
from .dynamics import (
    HeaveAnalytic,
    analytic_heave,
    integrate_heave,
    drag_relax,
    heave_audit,
    post_run_ledger,
)
from .study import (
    DEFAULT_OUT,
    LOADING_RATE_PA_PER_S,
    RASTER_H,
    boundary_identity,
    run_study,
    solve_and_write,
    write_figures,
)

__all__ = [
    "Raster",
    "polygon_mask",
    "disc_mask",
    "footprint_metrics",
    "resolution_check",
    "Disc",
    "FloeBody",
    "seed_inscribed_discs",
    "seed_equivalent_disc",
    "seed_floe_bodies",
    "overlap_audit",
    "body_ledger",
    "BodyKinematics",
    "RigidPlaneFit",
    "rigid_plane_fit",
    "floe_node_mask",
    "kinematic_transfer",
    "read_node_state",
    "HeaveAnalytic",
    "analytic_heave",
    "integrate_heave",
    "drag_relax",
    "heave_audit",
    "post_run_ledger",
    "DEFAULT_OUT",
    "LOADING_RATE_PA_PER_S",
    "RASTER_H",
    "boundary_identity",
    "run_study",
    "solve_and_write",
    "write_figures",
]