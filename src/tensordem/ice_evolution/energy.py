"""Energy budget and closure for the sintering + refreezing field (IN155-N09).

The control volume is the rubble layer.  Over a healing interval the balance
is

    E_latent + E_sinter = E_conduction + E_ambient + dE_sensible

with ``E_ambient = 0`` for a Dirichlet face held at the ambient temperature
(the ambient reservoir does no work).  The closure is

    |E_latent + E_sinter - E_conduction - dE_sensible| / (E_latent + E_sinter)

and must be < 10 % (criterion e).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyBudget:
    """Energy terms of one healing interval, per unit area (J/m^2)."""
    e_latent: float
    e_sinter: float
    e_conduction: float
    e_ambient: float = 0.0
    d_e_sensible: float = 0.0

    @property
    def e_released(self) -> float:
        return self.e_latent + self.e_sinter

    @property
    def e_removed(self) -> float:
        return self.e_conduction + self.e_ambient + self.d_e_sensible


def energy_closure(budget: EnergyBudget) -> float:
    """Relative closure of the energy balance (0 = exact)."""
    released = budget.e_released
    if released <= 0.0:
        # nothing happened: the balance is trivially closed
        return 0.0
    return abs(released - budget.e_removed) / released


def sinter_surface_energy_j_m3(neck_ratio: float, particle_radius_m: float,
                               n_contacts_per_m3: float,
                               surface_energy: float = 0.033) -> float:
    """Sintering surface energy released per unit layer volume.

    Each contact removes two hemispherical patches of area ``pi a^2`` from
    the free surface; with ``n_contacts_per_m3`` contacts per m^3 the
    released surface energy is ``2 pi a^2 gamma n_contacts``.
    """
    a = neck_ratio * particle_radius_m
    return 2.0 * 3.141592653589793 * a * a * surface_energy * n_contacts_per_m3
