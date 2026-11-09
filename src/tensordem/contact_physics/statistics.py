"""Ice-ridge morphology and contact-network statistics: adapter over N05.

Campaign node: ``P121W5-N05`` (ice-ridge morphology statistics).  The sampling
primitives are the verified ones; this module only fixes their signatures and
their input layout so that the integrated contact-physics march can request
"the morphology and the force-network statistics of this ridge at this operating
point" through one call.

Delegated primitives
-------------------
* :func:`tensordem.ridge_dem.generate_keel_assembly` -- deterministic bonded
  triangular-lattice keel at the requested porosity / freeze-bond fraction.
  Its ``info`` dict already reports the *realised* porosity after thinning,
  which is the morphology number quoted here (not the requested one).
* :func:`tensordem.ice_rubble_microstructure.coordination_number` and
  :func:`~tensordem.ice_rubble_microstructure.coordination_spectrum` -- mean
  load-bearing contacts per particle and its force-cutoff sweep.
* :func:`tensordem.ice_rubble_microstructure.force_chain_pdf` -- normalised
  contact-force PDF, coefficient of variation, strong fraction and the load
  fraction carried by the strong (``f > <f>``) contacts.
* :class:`tensordem.ridge_dem.RidgeKeel` -- keel flank angle, envelope
  cross-section area and the geometry-derived half width.

What is *not* claim: these are morphological and load-distribution statistics of
a lattice realisation, not a measurement of a full ridge-severity distribution.
The limits are listed in ``docs/contact_physics_module.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..ice_rubble_microstructure import (
    coordination_number,
    coordination_spectrum,
    force_chain_pdf,
)
from ..ridge_dem import RidgeKeel, RidgeMaterial, generate_keel_assembly

__all__ = ["RidgeMorphologyCase", "RidgeContactStatistics"]


@dataclass(frozen=True)
class RidgeMorphologyCase:
    """The ridge realisation whose morphology is sampled.

    Parameters
    ----------
    keel:
        Keel geometry (depth, width, porosity, sail height).
    material:
        Micro-parameters of the rubble lattice (particle radius, stiffnesses,
        freeze-bond fraction, seed).  The seed makes the realisation
        reproducible: the same case always yields the same lattice.
    tip_height:
        Height of the keel tip above the mudline at sampling time, m.
    """

    keel: RidgeKeel
    material: RidgeMaterial
    tip_height: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.keel, RidgeKeel):
            raise TypeError("keel must be a RidgeKeel")
        if not isinstance(self.material, RidgeMaterial):
            raise TypeError("material must be a RidgeMaterial")
        if not (math.isfinite(self.tip_height) and self.tip_height >= 0.0):
            raise ValueError("tip_height must be finite and nonnegative")


class RidgeContactStatistics:
    """Morphology + contact-network statistics of one ridge realisation.

    The expensive part (building the lattice) is done once in :meth:`assemble`
    and cached; :meth:`morphology` and :meth:`force_network` are then cheap and
    can be called once per statistics window by the integrated march.
    """

    def __init__(self, case: RidgeMorphologyCase) -> None:
        if not isinstance(case, RidgeMorphologyCase):
            raise TypeError("case must be a RidgeMorphologyCase")
        self.case = case
        self._assembled: dict[str, Any] | None = None

    # -- assembly ------------------------------------------------------------
    def assemble(self) -> dict[str, Any]:
        """Build (once) and return the keel assembly and its derived arrays."""
        if self._assembled is None:
            built = generate_keel_assembly(
                self.case.keel, self.case.material,
                tip_height=float(self.case.tip_height))
            bonds = built["assembly"].bonds
            i = np.asarray(bonds[:, 0].detach().cpu(), dtype=int) if len(bonds) else np.zeros(0, int)
            j = np.asarray(bonds[:, 1].detach().cpu(), dtype=int) if len(bonds) else np.zeros(0, int)
            positions = np.asarray(built["positions"].detach().cpu(), dtype=float)
            self._assembled = {
                "info": dict(built["info"]),
                "positions": positions,
                "i": i,
                "j": j,
                "n_particles": int(positions.shape[0]),
                "assembly": built["assembly"],
            }
        return self._assembled

    # -- morphology ----------------------------------------------------------
    def morphology(self) -> dict[str, float]:
        """Geometric and lattice statistics of the realisation.

        Returns
        -------
        dict with ``keel_depth_m``, ``keel_width_m``, ``keel_angle_deg``,
        ``cross_section_area_m2``, ``requested_porosity``, ``realised_porosity``
        (after thinning), ``n_particles``, ``n_bonds``, ``bond_density_1_per_m2``,
        ``mean_coordination`` (``Z`` of the bonded lattice, delegated to
        :func:`coordination_number`), ``driven_particles`` and
        ``particle_radius_m``.
        """
        built = self.assemble()
        info = built["info"]
        keel = self.case.keel
        contacts = {"i": built["i"], "j": built["j"],
                    "normal_force": np.ones(built["i"].size, dtype=float)}
        coordination = coordination_number(contacts, built["n_particles"])
        area = float(keel.cross_section_area)
        return {
            "keel_depth_m": float(keel.keel_depth),
            "keel_width_m": float(keel.keel_width),
            "keel_angle_deg": float(keel.angle_deg),
            "cross_section_area_m2": area,
            "requested_porosity": float(keel.porosity),
            "realised_porosity": float(info["porosity"]),
            "n_particles": float(built["n_particles"]),
            "n_bonds": float(info["bonds"]),
            "bond_density_1_per_m2": float(info["bonds"]) / max(area, 1e-30),
            "mean_coordination": float(coordination["Z"]),
            "driven_particles": float(info["driven_particles"]),
            "particle_radius_m": float(info["particle_radius"]),
        }

    def coordination_sweep(self) -> list[dict[str, float]]:
        """``Z`` versus the contact-force cutoff fraction (delegated sweep)."""
        built = self.assemble()
        contacts = {"i": built["i"], "j": built["j"],
                    "normal_force": np.ones(built["i"].size, dtype=float)}
        return coordination_spectrum(contacts, built["n_particles"])

    # -- force network -------------------------------------------------------
    def force_network(self, forces: Sequence[float], *,
                      n_bins: int = 20) -> dict[str, Any]:
        """Normalised contact-force statistics of one sampled window.

        Parameters
        ----------
        forces:
            Transmitted bond/contact forces (N) of the sampled window.  At least
            four positive values are required by the delegated PDF.
        n_bins:
            Number of PDF bins (forwarded unchanged).
        """
        array = np.asarray(forces, dtype=float)
        pdf = force_chain_pdf(array, n_bins=int(n_bins))
        keys = ("n_contacts", "mean_force", "coefficient_of_variation",
                "strong_fraction", "strong_load_fraction")
        return {key: (float(pdf[key]) if key != "n_contacts" else int(pdf[key]))
                for key in keys}

    def sample(self, forces: Sequence[float], *, n_bins: int = 20) -> dict[str, Any]:
        """Morphology + force-network statistics in one record."""
        record: dict[str, Any] = {"morphology": self.morphology(),
                                 "force_network": self.force_network(forces, n_bins=n_bins)}
        record["n_particles"] = int(self.assemble()["n_particles"])
        return record