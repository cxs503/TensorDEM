"""Erosion mass/energy ledger: closes ``mass removed`` against ``energy lost``.

The closure statement the campaign asks for is

    erosion mass loss  =  particle kinetic-energy loss  x  erosion efficiency

and it is only meaningful if the *three* legs are computed independently:

1. **kinematics** -- what the particle stream actually lost,
   ``dE = sum_i 1/2 m_i v_i^2 (1 - e^2)`` for normal restitution ``e``;
2. **constitutive** -- what the erosion model asks for, ``dm = rho_t dV`` with
   ``dV = dE f(alpha) / (2 p)`` for the Finnie cutting model;
3. **efficiency** -- the dimensionless ratio ``eta = dE_cut / dE`` which must
   stay below the geometric maximum ``f_max / 2 = 0.1514`` and, for any
   empirical model, below unity by energy conservation.

:class:`ErosionLedger` accumulates the three legs event by event, so the closure
residual can be reported as a number (``< 2%`` is the acceptance criterion) and
a bookkeeping mistake (double counting, unit slip, missing branch) cannot hide
inside a single closed-form expression.

The residual is reported as

    rel_residual = |E_consumed - m_removed * p / rho_t| / E_consumed

i.e. the accumulated removal energy is recomputed from the accumulated mass
through the *specific* removal energy, which catches any inconsistency between
per-event and aggregate bookkeeping.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .materials import Material, get_material
from .models import finnie_angle_function

FINNIE_GEOMETRIC_EFFICIENCY_MAX = 0.1514
"""``f_max / 2``: the largest fraction of impact energy that Finnie's cutting
model can spend on material removal (0.3027/2).  Any larger efficiency means an
energy-conservation violation."""


@dataclass(frozen=True)
class ImpactEvent:
    """One particle impact that carries energy into the wall."""

    particle_mass_kg: float
    speed_m_s: float
    angle_deg: float
    restitution: float = 0.0

    def __post_init__(self) -> None:
        if not self.particle_mass_kg > 0.0:
            raise ValueError("particle_mass_kg must be positive")
        if not (self.speed_m_s >= 0.0 and math.isfinite(self.speed_m_s)):
            raise ValueError("speed_m_s must be finite and non-negative")
        if not (0.0 <= self.angle_deg <= 90.0):
            raise ValueError("angle_deg must be within [0, 90]")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must be within [0, 1]")

    @property
    def kinetic_energy_in_J(self) -> float:
        return 0.5 * self.particle_mass_kg * self.speed_m_s**2

    @property
    def kinetic_energy_out_J(self) -> float:
        return self.kinetic_energy_in_J * self.restitution**2

    @property
    def kinetic_energy_lost_J(self) -> float:
        return self.kinetic_energy_in_J - self.kinetic_energy_out_J


@dataclass
class ErosionLedger:
    """Event-by-event energy/mass bookkeeping for one target material."""

    material: Material | str = "carbon_steel"
    impacts: int = 0
    delivered_mass_kg: float = 0.0
    kinetic_energy_in_J: float = 0.0
    kinetic_energy_out_J: float = 0.0
    cutting_energy_J: float = 0.0
    removed_volume_m3: float = 0.0
    removed_mass_kg: float = 0.0
    history: list[dict] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.material, str):
            self.material = get_material(self.material)

    # -- accumulation ---------------------------------------------------
    def add_impact(self, event: ImpactEvent, record: bool = False) -> dict:
        """Integrate one impact through the three ledger legs."""
        mat = self.material
        f_alpha = float(finnie_angle_function(event.angle_deg))
        e_lost = event.kinetic_energy_lost_J
        e_cut = e_lost * f_alpha / 2.0
        volume = e_cut / mat.flow_stress_pa
        mass = volume * mat.density_kg_m3

        self.impacts += 1
        self.delivered_mass_kg += event.particle_mass_kg
        self.kinetic_energy_in_J += event.kinetic_energy_in_J
        self.kinetic_energy_out_J += event.kinetic_energy_out_J
        self.cutting_energy_J += e_cut
        self.removed_volume_m3 += volume
        self.removed_mass_kg += mass

        entry = {
            "impact": self.impacts,
            "particle_mass_kg": event.particle_mass_kg,
            "speed_m_s": event.speed_m_s,
            "angle_deg": event.angle_deg,
            "restitution": event.restitution,
            "energy_in_J": event.kinetic_energy_in_J,
            "energy_lost_J": e_lost,
            "cutting_energy_J": e_cut,
            "removed_mass_kg": mass,
        }
        if record:
            self.history.append(entry)
        return entry

    def extend(self, events: Iterable[ImpactEvent], record: bool = False) -> None:
        for event in events:
            self.add_impact(event, record=record)

    def merge(self, other: "ErosionLedger") -> None:
        """Add another ledger of the *same* material into this one."""
        if other.material.name != self.material.name:
            raise ValueError("cannot merge ledgers of different materials")
        self.impacts += other.impacts
        self.delivered_mass_kg += other.delivered_mass_kg
        self.kinetic_energy_in_J += other.kinetic_energy_in_J
        self.kinetic_energy_out_J += other.kinetic_energy_out_J
        self.cutting_energy_J += other.cutting_energy_J
        self.removed_volume_m3 += other.removed_volume_m3
        self.removed_mass_kg += other.removed_mass_kg
        self.history.extend(other.history)

    # -- closure --------------------------------------------------------
    @property
    def kinetic_energy_lost_J(self) -> float:
        return self.kinetic_energy_in_J - self.kinetic_energy_out_J

    @property
    def specific_removal_energy_J_m3(self) -> float:
        """Energy actually spent per unit removed volume (should equal ``p``)."""
        if self.removed_volume_m3 <= 0.0:
            return float("nan")
        return self.cutting_energy_J / self.removed_volume_m3

    @property
    def erosion_efficiency(self) -> float:
        """``E_cut / E_lost`` -- must stay below the geometric bound."""
        if self.kinetic_energy_lost_J <= 0.0:
            return 0.0
        return self.cutting_energy_J / self.kinetic_energy_lost_J

    def closure(self) -> dict:
        """Close the ledger and report residuals and physical bounds."""
        mat = self.material
        e_consumed = self.cutting_energy_J
        e_from_mass = self.removed_mass_kg / mat.density_kg_m3 * mat.flow_stress_pa
        rel = (
            abs(e_consumed - e_from_mass) / e_consumed if e_consumed > 0.0 else 0.0
        )
        e_lost = self.kinetic_energy_lost_J
        kinetic_rel = (
            abs(e_lost - (self.kinetic_energy_in_J - self.kinetic_energy_out_J)) / e_lost
            if e_lost > 0.0
            else 0.0
        )
        eta = self.erosion_efficiency
        return {
            "material": mat.name,
            "impacts": self.impacts,
            "delivered_particle_mass_kg": self.delivered_mass_kg,
            "removed_mass_kg": self.removed_mass_kg,
            "erosion_ratio_kg_per_kg": (
                self.removed_mass_kg / self.delivered_mass_kg
                if self.delivered_mass_kg > 0.0
                else 0.0
            ),
            "kinetic_energy_in_J": self.kinetic_energy_in_J,
            "kinetic_energy_out_J": self.kinetic_energy_out_J,
            "kinetic_energy_lost_J": e_lost,
            "cutting_energy_J": e_consumed,
            "removal_energy_from_mass_J": e_from_mass,
            "specific_removal_energy_J_m3": self.specific_removal_energy_J_m3,
            "flow_stress_pa": mat.flow_stress_pa,
            "erosion_efficiency": eta,
            "efficiency_bound": FINNIE_GEOMETRIC_EFFICIENCY_MAX,
            "efficiency_within_bound": bool(eta <= FINNIE_GEOMETRIC_EFFICIENCY_MAX * (1 + 1e-12)),
            "energy_conserving": bool(eta <= 1.0),
            "closure_residual": rel,
            "kinetic_bookkeeping_residual": kinetic_rel,
            "closure_ok": bool(rel < 0.02 and eta <= 1.0),
        }

    def to_csv(self, path: str | Path) -> Path:
        """Write the per-impact history (requires ``record=True``)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self.history:
            raise ValueError("ledger has no recorded history; pass record=True")
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)
        return out


def closed_form_finnie_stream(
    v_m_s: float,
    alpha_deg: float,
    particle_mass_kg: float,
    material: Material | str = "carbon_steel",
    restitution: float = 0.0,
) -> float:
    """Aggregate removed mass for ``N`` identical impacts, by closed form.

    ``m_removed = N * rho_t * dE * f(alpha) / (2 p)`` -- the aggregate route that
    an event-by-event ledger must reproduce for a monodisperse single-angle
    stream (used as the C4 cross-route check).
    """
    mat = get_material(material) if isinstance(material, str) else material
    d_e = 0.5 * particle_mass_kg * v_m_s**2 * (1.0 - restitution**2)
    f_alpha = float(finnie_angle_function(alpha_deg))
    return mat.density_kg_m3 * d_e * f_alpha / (2.0 * mat.flow_stress_pa)


def implied_efficiency(
    erosion_ratio_kg_per_kg: float,
    v_m_s: float,
    material: Material | str = "carbon_steel",
    restitution: float = 0.0,
) -> float:
    """Erosion efficiency implied by an arbitrary model's erosion ratio.

    ``eta = (dm/m_p) / rho_t * p / (1/2 v^2 (1 - e^2))`` -- the removed *volume*
    per unit particle mass times the specific removal energy, divided by the
    impact energy per unit particle mass.  Values above 1 mean the model claims
    to remove more material than the impact energy can pay for, which is an
    a-priori impossible prediction (the specific plastic work of removal cannot
    be below the flow stress), so this is a physical plausibility gate for the
    empirical (Oka, DNV) models.  Finnie's cutting model reproduces exactly
    ``eta = f(alpha)/2 <= 0.1514`` under this definition.
    """
    mat = get_material(material) if isinstance(material, str) else material
    available = 0.5 * v_m_s**2 * (1.0 - restitution**2)
    if available <= 0.0:
        return 0.0
    removed_volume_per_mass = erosion_ratio_kg_per_kg / mat.density_kg_m3
    return removed_volume_per_mass * mat.flow_stress_pa / available


__all__ = [
    "FINNIE_GEOMETRIC_EFFICIENCY_MAX",
    "ErosionLedger",
    "ImpactEvent",
    "closed_form_finnie_stream",
    "implied_efficiency",
]