"""Integration of the four-ledger energy audit with the DEM main solver.

Why this module exists
----------------------
:mod:`tensordem.ice_structure_energy_ledger` owns the audit *physics* (the four
ledgers, their independent reference implementations and the acceptance
criteria).  A capability that is only exercised by its own benchmark is not yet
part of the solver: the audit has to be driven by the production DEM loop of
:mod:`tensordem.dem3d` (the main bonded-sphere ice solver) and it has to do so
without touching the trajectory it audits.

This module is that wiring:

* :class:`IceDEMEnergyAuditAdapter` reads one solver state per step and pushes
  the four ledgers into a :class:`~tensordem.ice_structure_energy_ledger.
  FourLedgerObserver`.
* :func:`attach_energy_audit` installs the adapter on a live solver; from then
  on ``IceDEM3D.step`` calls the adapter itself (one opt-in hook, default off).
* :func:`audit_solver_run` is the driver used by the benchmark: it advances the
  solver and returns the audit rows plus the closure verdict.

Trajectory neutrality
---------------------
The hook only reads.  Everything it touches is either a stored state array or
``IceDEM3D.diagnostics()``, which evaluates forces with
``update_fracture=False, update_history=False`` -- i.e. it neither commits
fracture, nor advances the Cundall-Strack tangential history, nor changes the
positions, so an audited run and an unaudited run are bit-for-bit identical
(this is asserted by the test suite).

Channel coverage (honesty rule)
-------------------------------
The adapter reports what the solver actually accumulates and nothing else:

* ``E_k`` -- recomputed here from ``sim.velocities`` and ``config.mass`` (an
  independent path, cross-checked against ``diagnostics()["kinetic_energy_J"]``;
  a mismatch beyond ``1e-9`` relative raises, so the audit cannot silently
  diverge from the solver's own accounting).
* ``E_s`` -- the solver's own spring potentials (bonds, particle contacts, tool
  contact, platens when enabled).
* ``E_f`` -- ``failure_energy_J`` accumulated by the solver at every bond break.
* ``E_v`` -- ``cumulative_bond_dissipation_J`` (bond dashpot).  Particle/tool
  dashpots are *not* accumulated by the solver: a run with ``restitution < 1``
  therefore leaves that dissipation inside the residual, and the residual
  reports it instead of hiding it.  Set ``restitution_coefficient = 1`` (the
  default) for a closure run.
* ``E_ext`` -- external work injected by a moving boundary, read from the host
  solver when it tracks it (``cumulative_bow_work_J`` /
  ``_cumulative_bow_work``); zero for a static structure, which is the case the
  audit is defined for.

Quantity names are SI; all arithmetic is done in float64.
"""
from __future__ import annotations

import math
from typing import Any

from .ice_structure_energy_ledger import FourLedgerObserver

__all__ = [
    "IceDEMEnergyAuditAdapter",
    "attach_energy_audit",
    "audit_solver_run",
    "EXTERNAL_WORK_ATTRS",
]

#: Attributes a host solver may expose for the external-work term ``E_ext``.
EXTERNAL_WORK_ATTRS = ("cumulative_bow_work_J", "_cumulative_bow_work")


class IceDEMEnergyAuditAdapter:
    """Drive a :class:`FourLedgerObserver` from a ``tensordem.dem3d`` solver."""

    def __init__(self, observer: FourLedgerObserver | None = None) -> None:
        self.observer = observer if observer is not None else FourLedgerObserver()
        self.cross_check_max = 0.0
        self.external_work_source: str | None = None
        #: Cumulative dissipation of dashpot channels the host solver does *not*
        #: accumulate itself (e.g. the tool/bow dashpot), integrated by the
        #: caller from public state and added to ledger 4.  Zero by default, so
        #: the residual keeps reporting any channel that is not accounted.
        self.extra_viscous_dissipation_J = 0.0

    # -- external work ------------------------------------------------------
    def _external_work(self, sim: Any) -> float:
        for name in EXTERNAL_WORK_ATTRS:
            if hasattr(sim, name):
                self.external_work_source = name
                return float(getattr(sim, name) or 0.0)
        self.external_work_source = None
        return 0.0

    # -- one step -----------------------------------------------------------
    def observe(self, sim: Any) -> dict[str, float]:
        """Sample ``sim``, append one four-ledger row and return it."""
        cfg = sim.config
        velocities = sim.velocities
        if hasattr(velocities, "detach"):
            velocities = velocities.detach().cpu().numpy()
        kinetic = 0.5 * float(cfg.mass) * float(
            (velocities * velocities).sum())
        diag = sim.diagnostics()
        reference = float(diag.get("kinetic_energy_J", 0.0))
        scale = max(abs(reference), abs(kinetic), 1.0e-30)
        cross = abs(kinetic - reference) / scale
        self.cross_check_max = max(self.cross_check_max, cross)
        if cross > 1.0e-9:
            raise ValueError(
                "audit kinetic energy disagrees with the solver diagnostics "
                f"({kinetic!r} vs {reference!r}, relative {cross:.3e})")

        strain_contact = (float(diag.get("particle_contact_energy_J", 0.0))
                          + float(diag.get("tool_contact_energy_J", 0.0)))
        strain_bond = float(diag.get("bond_elastic_energy_J", 0.0))
        failure = getattr(sim, "failure_energy_J", None)
        fracture = float(failure.sum()) if failure is not None else 0.0
        viscous = (float(getattr(sim, "cumulative_bond_dissipation_J", 0.0) or 0.0)
                   + float(self.extra_viscous_dissipation_J))

        out_of_ledger: dict[str, float] = {}
        # Channels the solver accumulates but that are deliberately outside the
        # four ledgers stay visible instead of being folded into E_v.
        drag_dissipation = getattr(sim, "cumulative_drag_dissipation_J", None)
        if drag_dissipation is not None:
            out_of_ledger["drag"] = float(drag_dissipation)

        return self.observer.observe(
            time=float(sim.time),
            kinetic_trans=kinetic,
            kinetic_rot=0.0,          # dem3d has translational DOF only
            strain_contact=strain_contact,
            strain_bond=strain_bond,
            fracture_energy=fracture,
            viscous_dissipation=viscous,
            external_work=self._external_work(sim),
            dt=float(sim.dt) if math.isfinite(float(sim.dt)) else None,
            out_of_ledger=out_of_ledger,
        )

    # -- verdict ------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        out = self.observer.summary()
        out["kinetic_cross_check_max"] = self.cross_check_max
        out["external_work_source"] = self.external_work_source
        out["extra_viscous_dissipation_J"] = float(self.extra_viscous_dissipation_J)
        return out

    @property
    def rows(self) -> list[dict[str, float]]:
        return self.observer.rows


def attach_energy_audit(sim: Any,
                        adapter: IceDEMEnergyAuditAdapter | None = None
                        ) -> IceDEMEnergyAuditAdapter:
    """Install the audit on a live ``IceDEM3D`` (``sim.energy_audit`` hook).

    The solver calls ``adapter.observe(sim)`` at the end of every ``step()``;
    the first row is recorded immediately so that ``E_total(0)`` is the state
    the solver starts from.
    """
    if not hasattr(sim, "energy_audit"):
        raise TypeError(
            f"{type(sim).__name__} has no 'energy_audit' hook; the four-ledger "
            "integration is available on tensordem.dem3d.IceDEM3D and its "
            "subclasses")
    adapter = adapter if adapter is not None else IceDEMEnergyAuditAdapter()
    adapter.observe(sim)          # reference state, before the first step
    sim.energy_audit = adapter
    return adapter


def audit_solver_run(sim: Any, n_steps: int,
                     adapter: IceDEMEnergyAuditAdapter | None = None
                     ) -> IceDEMEnergyAuditAdapter:
    """Advance ``sim`` by ``n_steps`` with the audit attached; return adapter."""
    adapter = attach_energy_audit(sim, adapter)
    for _ in range(int(n_steps)):
        sim.step()
    return adapter