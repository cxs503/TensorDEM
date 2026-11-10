"""Three-dimensional slab slice of a broken-ice channel with an open lateral
boundary and a mass-conserving departure-flux ledger.

Why this module exists
----------------------
A 2-D (single horizontal slice, laterally closed, instant-clearance) channel
DEM inflates the level-ice resistance of MT Uikku by two orders of magnitude.
The two structural defects of that configuration are

1. **dimension** -- a single slice with reflective lateral walls leaves the
   crushed / displaced ice *no lateral relief path*.  The contact forces then
   saturate at the ice compressive-crushing limit of jammed floes instead of
   being set by the flexural breaking capacity of the sheet;
2. **instant clearance** -- floes pushed behind the hull are deleted from the
   node set, so the mass balance cannot be closed and the artifact is
   undetectable (a deleted floe is also a floe that never pushes back, and the
   *accounting* of the cleared channel is lost).

This module rebuilds the slice as a **slab**: ``n_layers`` (2-4) horizontal
slices stacked through the ice thickness with the total thickness equal to the
ice thickness ``h``, and the lateral boundary of the channel replaced by an
**opening plus an annular reservoir band**.  Floes that cross the opening stay
in the simulation (``in_reservoir``); nothing is deleted.  Every mass transfer
is written to a ledger (:class:`MassLedger`, :class:`FluxLedger`) so that

* the per-frame mass-conservation error and
* the lateral-flux closure ratio

are machine-checkable numbers rather than claims.

Physical model (declared closures)
----------------------------------
* **Layered slab.**  Layer ``k`` occupies the depth band ``[z_k, z_{k+1}]`` of
  the sheet and carries material thickness ``t_k = h/n`` (contact area and
  crushing cap scale with ``t_k``, so the ``n`` layers sum to the full-thickness
  crushing force).  Bending of the slab is *not* the sum of independent lamina
  bending: the flexural capacity per unit width is the full-slab
  ``sigma_f h**2 / 6`` and it is distributed over the layers by their share of
  the second moment of the linear stress profile,
  ``s_k = 4 (z_{k+1}**3 - z_k**3) / h**3`` (so ``sum_k s_k = 1``).  Independent
  lamina bending would scale as ``n * t**2 ~ 1/n`` and is kept only as a
  negative control (``layer_moment_coupling=False``).
* **Breaking capacity.**  A floe that is still part of the intact sheet detaches
  when the hull load across its edge reaches the repository's graded Lindqvist
  bending coefficient (``level_ice.formula.COEFFICIENT_LEDGER``:
  ``bending_efficiency = 0.25`` CONVENTION, lever arm ``L_c/1.5``), i.e. the
  per-unit-width breaking force is ``0.25 sigma_f h**2 / L_c`` with
  ``L_c = (E h**3 / (12 (1 - nu**2) rho_w g))**0.25``.  The floe edge width in
  the horizontal slice is ``2 r``; the layer carries the fraction ``s_k``.
* **Contact.**  Penalty normal law anchored on the ice compressive strength,
  ``k_n = sigma_c A_c(delta_ref) / delta_ref`` with the nominal line-contact
  area ``A_c(delta) = 2 t sqrt(2 r delta)`` (the same anchor as
  :mod:`tensordem.ice_channel`), and a crushing cap ``N <= sigma_c A_c(delta)``
  with the physical ceiling ``delta <= 0.5 r`` (a floe cannot crush beyond half
  its own radius -- this is the cap that the closed 2-D configuration rides
  on).  Coulomb tangential law with ``mu`` (ice-steel order, 0.10).
* **Quasi-static relaxation.**  Free floes are advanced by a damped Jacobi
  relaxation ``p <- p + omega F / k_n`` (no inertia); intact floes are pinned in
  the sheet and only translate with the drift.  Deterministic: same seed gives
  the same bit pattern.

Evidence grades
---------------
REAL: contact geometry, crushing cap, mass/flux ledger, slicing (all closed
form and unit-tested).  CONVENTION: the bending lever arm (inherited from the
repo ledger, graded CONVENTION there) and the friction coefficient.  The
submersion term is **not** resolved (no vertical floe dynamics) and enters the
resistance budget only as an explicitly declared closure with the same ledger
constants, reported separately from the DEM-resolved components.
"""
from .slab import (
    BENDING_EFFICIENCY,
    GRAVITY,
    ICE_DENSITY,
    WATER_DENSITY,
    HullFootprint,
    IceSheetProps,
    MassLedger,
    FluxLedger,
    SlabConfig,
    SlabChannelSim,
    RunSummary,
    layer_moment_shares,
    characteristic_length_m,
    bending_capacity_per_width_n_per_m,
)
from .uikku import (
    CRITERIA,
    UikkuCampaign,
    load_reference_cases,
    run_ablation,
    run_identifier,
    w_over_b_scan,
)

__all__ = [
    "BENDING_EFFICIENCY", "GRAVITY", "ICE_DENSITY", "WATER_DENSITY",
    "HullFootprint", "IceSheetProps", "MassLedger", "FluxLedger",
    "SlabConfig", "SlabChannelSim", "RunSummary",
    "layer_moment_shares", "characteristic_length_m",
    "bending_capacity_per_width_n_per_m",
    "CRITERIA", "UikkuCampaign", "load_reference_cases", "run_ablation",
    "run_identifier", "w_over_b_scan",
]