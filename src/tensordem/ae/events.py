"""Bond-break event tracking and energy release accounting.

Each bond break in the DEM releases stored elastic energy E_rel. This module
records these events as AE proxies with timestamps, released energies, and
spatial coordinates.

Energy closure:
    The total released energy must equal the sum of individual bond failure
    energies, and the cumulative external work must balance:
        W_ext = E_kinetic + E_bond_elastic + E_contact + E_released + E_dissipated
    with relative closure error < 1%.

Single-bond exact check:
    For a single bond pulled monotonically from rest to failure at strain ε_b:
        E_rel = 0.5 * k * (ε_b * L0)^2
        ∫F·du = ∫₀^{ε_b·L0} k·u du = 0.5 * k * (ε_b * L0)^2
    These must agree to machine precision (relative diff < 1e-6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from ..dem3d import DEM3DConfig, IceDEM3D


@dataclass(frozen=True)
class AEEvent:
    """A single acoustic emission event from a bond break.

    Attributes:
        time_s: Simulation time of the bond break (s).
        bond_index: Index of the broken bond in the pair list.
        node_i: First particle index.
        node_j: Second particle index.
        position: Midpoint of the bond (m).
        energy_release_J: Stored elastic energy released (J).
        failure_mode: 1=tensile, 2=shear, 4=creep.
        failure_strain: Axial strain at failure (dimensionless).
    """
    time_s: float
    bond_index: int
    node_i: int
    node_j: int
    position: tuple[float, float, float]
    energy_release_J: float
    failure_mode: int
    failure_strain: float


@dataclass
class AEEventCatalog:
    """Chronological catalog of AE events from a DEM simulation.

    The catalog records every bond break with its energy release, enabling
    post-processing for AE event rates, amplitude distributions, and b-values.
    """
    events: list[AEEvent] = field(default_factory=list)
    _prev_alive: torch.Tensor | None = field(default=None, repr=False)
    _prev_bonded: torch.Tensor | None = field(default=None, repr=False)

    def snapshot(self, sim: IceDEM3D) -> None:
        """Record the current alive state for future break detection."""
        self._prev_alive = sim.alive.clone()
        self._prev_bonded = sim.bonded.clone()

    def detect_and_record(self, sim: IceDEM3D) -> list[AEEvent]:
        """Detect new bond breaks since last snapshot and record them.

        Returns the list of newly detected events (may be empty).
        """
        if self._prev_alive is None:
            self.snapshot(sim)
            return []

        # Bonds that were alive+bonded but are now dead
        was_alive = self._prev_alive & self._prev_bonded
        newly_broken = was_alive & ~sim.alive

        new_events: list[AEEvent] = []
        if not bool(newly_broken.any()):
            self.snapshot(sim)
            return new_events

        broken_indices = torch.nonzero(newly_broken, as_tuple=False).flatten()
        positions = sim.positions
        pairs = sim.pairs
        rest_lengths = sim.rest_lengths

        for idx in broken_indices.tolist():
            i, j = int(pairs[idx, 0].item()), int(pairs[idx, 1].item())
            pos = 0.5 * (positions[i] + positions[j])
            e_rel = float(sim.failure_energy_J[idx].item())
            f_mode = int(sim.failure_mode[idx].item())
            f_strain = float(sim.failure_extension_m[idx].item()) / float(rest_lengths[idx].item())
            t = float(sim.time)

            event = AEEvent(
                time_s=t,
                bond_index=idx,
                node_i=i,
                node_j=j,
                position=(float(pos[0].item()), float(pos[1].item()), float(pos[2].item())),
                energy_release_J=e_rel,
                failure_mode=f_mode,
                failure_strain=f_strain,
            )
            new_events.append(event)
            self.events.append(event)

        self.snapshot(sim)
        return new_events

    @property
    def total_energy_J(self) -> float:
        """Total released energy across all events (J)."""
        return sum(e.energy_release_J for e in self.events)

    @property
    def n_events(self) -> int:
        return len(self.events)

    def energies(self) -> list[float]:
        """List of released energies (J) in chronological order."""
        return [e.energy_release_J for e in self.events]

    def times(self) -> list[float]:
        """List of event times (s) in chronological order."""
        return [e.time_s for e in self.events]

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """Convert events to a list of dicts suitable for CSV export."""
        return [
            {
                "event_id": k,
                "time_s": e.time_s,
                "bond_index": e.bond_index,
                "node_i": e.node_i,
                "node_j": e.node_j,
                "x_m": e.position[0],
                "y_m": e.position[1],
                "z_m": e.position[2],
                "energy_release_J": e.energy_release_J,
                "failure_mode": e.failure_mode,
                "failure_strain": e.failure_strain,
            }
            for k, e in enumerate(self.events)
        ]


def single_bond_energy_release(bond_stiffness: float, breaking_strain: float,
                               rest_length: float) -> float:
    """Analytical released energy for a single bond broken at breaking_strain.

    E_rel = 0.5 * k * (ε_b * L0)^2

    This is the exact energy that should be recorded by the DEM solver when
    a single bond is pulled monotonically to its breaking strain.
    """
    extension = breaking_strain * rest_length
    return 0.5 * bond_stiffness * extension ** 2


def single_bond_work_integral(bond_stiffness: float, breaking_strain: float,
                              rest_length: float) -> float:
    """External work ∫F·du for pulling a single bond to breaking_strain.

    For a linear spring F = k*u, the work is:
        W = ∫₀^{ε_b·L0} k·u du = 0.5 * k * (ε_b * L0)^2

    This must equal single_bond_energy_release to machine precision.
    """
    extension = breaking_strain * rest_length
    return 0.5 * bond_stiffness * extension ** 2


def energy_closure_check(sim: IceDEM3D, catalog: AEEventCatalog,
                         initial_energy: dict[str, float] | None = None,
                         external_work_J: float = 0.0) -> dict[str, float]:
    """Check energy closure for the DEM simulation.

    Energy balance:
        W_ext + E_initial = E_kinetic + E_bond + E_contact + E_released + E_dissipated

    Returns a dict with closure error and components.
    """
    energy = sim.energy()
    e_kinetic = energy["kinetic_energy_J"]
    e_bond = energy["bond_elastic_energy_J"]
    e_contact = (energy.get("particle_contact_energy_J", 0.0)
                 + energy.get("tool_contact_energy_J", 0.0)
                 + energy.get("platen_contact_energy_J", 0.0))
    e_released = catalog.total_energy_J
    e_dissipated = float(sim.cumulative_bond_dissipation_J)

    e_current = e_kinetic + e_bond + e_contact + e_released + e_dissipated

    if initial_energy is not None:
        e_init = (initial_energy.get("kinetic_energy_J", 0.0)
                  + initial_energy.get("bond_elastic_energy_J", 0.0)
                  + initial_energy.get("particle_contact_energy_J", 0.0)
                  + initial_energy.get("tool_contact_energy_J", 0.0)
                  + initial_energy.get("platen_contact_energy_J", 0.0))
    else:
        e_init = 0.0

    total_in = external_work_J + e_init
    total_out = e_current
    closure_error = abs(total_out - total_in) / max(abs(total_in), 1e-30)

    return {
        "external_work_J": external_work_J,
        "initial_energy_J": e_init,
        "kinetic_energy_J": e_kinetic,
        "bond_elastic_energy_J": e_bond,
        "contact_energy_J": e_contact,
        "released_energy_J": e_released,
        "dissipated_energy_J": e_dissipated,
        "total_in_J": total_in,
        "total_out_J": total_out,
        "closure_relative_error": closure_error,
    }


__all__ = [
    "AEEvent", "AEEventCatalog",
    "single_bond_energy_release", "single_bond_work_integral",
    "energy_closure_check",
]
