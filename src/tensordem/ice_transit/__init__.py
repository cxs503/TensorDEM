"""Steady-state broken-ice-field direct-navigation DEM with a closed energy ledger.

Why another surface
-------------------
The sibling package :mod:`tensordem.broken_ice_resistance` measured a 91 %
shortfall against the level-ice Lindqvist anchor and named two structural
causes: (i) a finite domain in which the entrained ice plug grows instead of
reaching a steady state, and (ii) a free lateral boundary that lets the pack
part with no pack pressure.  Its contact law also carries an energy-inconsistent
regularisation: the chord width is evaluated at ``delta + eps`` *without*
subtracting the resulting offset, so a floe that touches another one
experiences ``sigma_c h 2 sqrt(2 R eps)`` (1.1 MN for a 5 m floe) at zero
penetration.  The absolute resistance level of that surface is therefore set by
a numerical regularisation and not by ice mechanics.

This package rebuilds the transit measurement on three declared changes:

``law``        a **zero-offset crushing law**
               ``F = sigma_c h [2 sqrt(2 R (delta+eps)) - 2 sqrt(2 R eps)]``
               capped at ``2 R`` with ``eps = sigma_c R / E`` (the elastic
               penetration at which the ice edge reaches its crushing
               strength).  ``F(0) = 0`` and the closed-form potential
               ``U(delta) = int_0^delta F`` make the contact energy consistent
               (``dU/ddelta == F`` is a unit test).
``inlet``      a **recycling inlet** in the ship frame: floes that pass abaft the
               stern re-enter ahead of the stem.  The transit is therefore
               arbitrary long (>= 10 ship lengths) with a compact floe count,
               and the measurement is a genuine *steady state* rather than a
               growing plug.
``ledger``     a **contact-level energy ledger**: every contact reports its
               elastic potential and its dissipative power, so

                   W_hull_in + W_inj - W_rem = dE_stored + W_diss

               closes to a reported residual instead of being asserted.

Public surface
--------------
``laws``       crushing law + potential, instrumented contact kernels
``transit``    :class:`SteadyTransit` (field, recycling inlet, walls, ledger)
``reference``  Lindqvist anchor, broken-ice closures, concentration scaling
``campaign``   pre-registered case set, criteria, artefacts, figures, report

All quantities SI; forces in N, resistances reported in kN.
"""
from __future__ import annotations

__all__ = ["laws", "transit", "reference", "campaign"]