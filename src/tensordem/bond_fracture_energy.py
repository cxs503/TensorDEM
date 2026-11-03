"""Closed-form bond fracture energy for the simple-cubic bonded-sphere lattice.

The 3-D bonded-sphere model stores a linear central spring on every neighbour
pair inside the initial cutoff: simple-cubic lattice, spacing ``a = 2 * radius``,
26 neighbours per interior particle (axial, face-diagonal and body-diagonal
bonds). A bond fails when its axial strain exceeds ``breaking_strain`` and the
solver records the elastic energy of that spring at the moment of detection,

    0.5 * bond_stiffness * (breaking_strain * rest_length) ** 2.

This module turns those two uncalibrated inputs (``bond_stiffness``,
``breaking_strain``) into an equivalent continuum fracture energy per unit
crack area, so the fracture parameter set can be compared with measured ice
fracture energy instead of being reported as a bare breaking strain.

Derivation (plane normal to a lattice axis).  One crack sheet of area A cuts one
lattice cell cross-section ``a**2`` per crossing bond family.  The 26-neighbour
set contributes nine bond families crossing that sheet (``dz = 1``,
``dx, dy in {-1, 0, 1}``), with rest lengths ``a * sqrt(1 + dx**2 + dy**2)``.
Summing the per-bond failure energies of those nine families and dividing by
``a**2`` gives

    G_f = 0.5 * k * eps_b**2 * sum(1 + dx**2 + dy**2) = 10.5 * k * eps_b**2

in J/m**2.  The lattice spacing cancels: the number of bonds per unit area grows
as ``1 / a**2`` while the stored energy per bond grows as ``a**2``, so the
equivalent fracture energy of a *fixed* ``(k, eps_b)`` pair is resolution
independent.  That is a property of the strain-based failure criterion, not of a
fracture-energy criterion; see ``docs/dem3d_bond_fracture_energy.md``.

All functions use SI units. Nothing here modifies the solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .dem3d import DEM3DConfig, IceDEM3D

#: Full 26-neighbour shell of the simple-cubic lattice used by
#: :class:`~tensordem.dem3d.IceDEM3D` (which stores each pair once).
FULL_NEIGHBOUR_SHELL: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dx == dy == dz == 0)
)


def crossing_offset_sum(axis: int = 2) -> float:
    """Return ``sum(rest_length**2) / spacing**2`` for bonds crossing one lattice cell.

    A crack sheet normal to ``axis`` is cut between two lattice layers. Exactly
    the neighbour offsets with ``component[axis] == +1`` connect the lower layer
    to the upper one (nine families for the 26-neighbour shell), and each family
    contributes one bond per cell cross-section ``a**2``. The returned value is
    21 for any of the three cube axes, by symmetry of the shell.
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    total = 0.0
    for offset in FULL_NEIGHBOUR_SHELL:
        if offset[axis] != 1:
            continue
        total += sum(component * component for component in offset)
    return total


#: Per-cell-crossing energy factor for a plane normal to a cube axis (21/2 = 10.5).
AXIS_PLANE_FACTOR = 0.5 * crossing_offset_sum(2)


def lattice_spacing(config: DEM3DConfig) -> float:
    """Return the simple-cubic lattice spacing ``a`` implied by the particle radius."""
    return 2.0 * config.radius


def bond_failure_energy_J(bond_stiffness: float, breaking_strain: float,
                          rest_length: float) -> float:
    """Elastic energy stored in one bond at the instant of tensile failure (J)."""
    for name, value in (("bond_stiffness", bond_stiffness),
                        ("breaking_strain", breaking_strain),
                        ("rest_length", rest_length)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return 0.5 * bond_stiffness * (breaking_strain * rest_length) ** 2


def analytic_fracture_energy_J_per_m2(bond_stiffness: float, breaking_strain: float,
                                      axis: int = 2) -> float:
    """Closed-form equivalent fracture energy of the lattice (J/m**2).

    Uses the idealised (edge-free) 26-neighbour lattice, so it is the limit the
    finite specimen counting below converges to as the specimen grows.
    """
    factor = 0.5 * crossing_offset_sum(axis)
    return factor * bond_stiffness * breaking_strain ** 2


def fracture_energy_from_config(config: DEM3DConfig, breaking_strain: float | None = None,
                                axis: int = 2) -> dict[str, Any]:
    """Count crossing bonds in the *actual* solver lattice and score per unit area.

    The specimen cross-section is taken as the lattice footprint
    ``(n_j - 1) * a`` by ``(n_k - 1) * a`` on the axes orthogonal to ``axis``,
    and the cut plane is placed halfway through the specimen. Finite specimens
    lose the bonds whose far endpoint falls outside the domain, so the measured
    value lies below the closed form by an ``O(1/n)`` edge deficit; the size
    series in the benchmark script reports that convergence.
    """
    strain = config.breaking_strain if breaking_strain is None else breaking_strain
    if not math.isfinite(strain) or strain <= 0:
        raise ValueError("breaking_strain must be finite and positive")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    sim = IceDEM3D(config)
    spacing = lattice_spacing(config)
    positions = sim.initial_positions
    if axis == 0:
        n_cross, n_area = config.ny, config.nz
    elif axis == 1:
        n_cross, n_area = config.nx, config.nz
    else:
        n_cross, n_area = config.nx, config.ny
    extent_area = (n_cross - 1) * spacing * (n_area - 1) * spacing
    if extent_area <= 0:
        raise ValueError("specimen has no cross-section on the chosen axis")

    # Cut between two adjacent lattice layers: coordinates along the axis are
    # multiples of ``a`` from ``-(n - 1) / 2 * a``, so a mid-layer plane would
    # contain the particles themselves.
    coord = positions[:, axis]
    layers = torch.unique(coord).sort().values
    if len(layers) < 2:  # pragma: no cover - guarded by config validation
        raise ValueError("specimen has fewer than two layers on the chosen axis")
    mid = (len(layers) - 1) // 2
    cut = 0.5 * (float(layers[mid]) + float(layers[mid + 1]))
    i, j = sim.pairs.T
    crossing = (coord[i] - cut) * (coord[j] - cut) < 0
    per_bond = 0.5 * config.bond_stiffness * (strain * sim.rest_lengths) ** 2
    released = float(per_bond[crossing].sum())
    analytic = analytic_fracture_energy_J_per_m2(config.bond_stiffness, strain, axis)
    measured = released / extent_area
    return {
        "axis": axis,
        "particle_count": int(len(positions)),
        "bond_count": int(len(sim.pairs)),
        "crossing_bond_count": int(crossing.sum()),
        "crossing_bonds_per_area": float(crossing.sum()) / extent_area,
        "cross_section_area_m2": extent_area,
        "lattice_spacing_m": spacing,
        "breaking_strain": strain,
        "bond_stiffness_N_per_m": config.bond_stiffness,
        "released_energy_J": released,
        "measured_fracture_energy_J_per_m2": measured,
        "analytic_fracture_energy_J_per_m2": analytic,
        "relative_error": abs(measured - analytic) / analytic,
    }


def required_breaking_strain(bond_stiffness: float, target_J_per_m2: float,
                             axis: int = 2) -> float:
    """Return the ``breaking_strain`` that reproduces a target fracture energy."""
    factor = 0.5 * crossing_offset_sum(axis)
    if not math.isfinite(target_J_per_m2) or target_J_per_m2 <= 0:
        raise ValueError("target fracture energy must be finite and positive")
    return math.sqrt(target_J_per_m2 / (factor * bond_stiffness))


def required_bond_stiffness(breaking_strain: float, target_J_per_m2: float,
                            axis: int = 2) -> float:
    """Return the ``bond_stiffness`` that reproduces a target fracture energy."""
    factor = 0.5 * crossing_offset_sum(axis)
    if not math.isfinite(target_J_per_m2) or target_J_per_m2 <= 0:
        raise ValueError("target fracture energy must be finite and positive")
    return target_J_per_m2 / (factor * breaking_strain ** 2)


def ice_fracture_energy_J_per_m2(stress_intensity_MPA_sqrt_m: float,
                                 youngs_modulus_GPa: float,
                                 poisson_ratio: float = 0.3) -> float:
    """Convert a measured fracture toughness into fracture energy (plane strain).

    ``G = K_Ic**2 * (1 - nu**2) / E``. This is the reference scale used to judge
    whether a ``(bond_stiffness, breaking_strain)`` pair represents a physically
    plausible ice fracture energy; it is not a validation of the DEM.
    """
    for name, value in (("stress_intensity_MPA_sqrt_m", stress_intensity_MPA_sqrt_m),
                        ("youngs_modulus_GPa", youngs_modulus_GPa)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0.0 <= poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must be in [0, 0.5)")
    k = stress_intensity_MPA_sqrt_m * 1e6  # Pa*sqrt(m)
    e = youngs_modulus_GPa * 1e9  # Pa
    return k * k * (1.0 - poisson_ratio ** 2) / e


@dataclass(frozen=True)
class FractureEnergyAudit:
    """Calibration summary for one bonded-particle parameter set."""

    config: DEM3DConfig
    measured: dict[str, Any]
    analytic_J_per_m2: float
    physical_band_J_per_m2: tuple[float, float]
    required_strain_for_band_midpoint: float

    def relative_to_band(self) -> float:
        midpoint = 0.5 * sum(self.physical_band_J_per_m2)
        return self.analytic_J_per_m2 / midpoint

    def as_dict(self) -> dict[str, Any]:
        return {
            "bond_stiffness_N_per_m": self.config.bond_stiffness,
            "breaking_strain": self.config.breaking_strain,
            "lattice_spacing_m": lattice_spacing(self.config),
            "analytic_fracture_energy_J_per_m2": self.analytic_J_per_m2,
            "measured_finite_specimen_J_per_m2": self.measured["measured_fracture_energy_J_per_m2"],
            "finite_specimen_relative_error": self.measured["relative_error"],
            "physical_band_J_per_m2": list(self.physical_band_J_per_m2),
            "ratio_to_band_midpoint": self.relative_to_band(),
            "required_breaking_strain_for_band_midpoint": self.required_strain_for_band_midpoint,
        }


def audit_config(config: DEM3DConfig | None = None,
                 physical_band_J_per_m2: tuple[float, float] = (0.23, 4.1)) -> FractureEnergyAudit:
    """Audit a parameter set against a physical fracture-energy band.

    The default band spans ``K_Ic = 0.05 ... 0.15 MPa*sqrt(m)`` at
    ``E = 5 ... 10 GPa``, ``nu = 0.3`` for sea ice (see
    :func:`ice_fracture_energy_J_per_m2`); pass another band explicitly when a
    measured value for the ice of interest is available.
    """
    cfg = DEM3DConfig() if config is None else config
    measured = fracture_energy_from_config(cfg)
    analytic = measured["analytic_fracture_energy_J_per_m2"]
    midpoint = 0.5 * sum(physical_band_J_per_m2)
    return FractureEnergyAudit(
        config=cfg,
        measured=measured,
        analytic_J_per_m2=analytic,
        physical_band_J_per_m2=physical_band_J_per_m2,
        required_strain_for_band_midpoint=required_breaking_strain(cfg.bond_stiffness, midpoint),
    )


__all__ = [
    "AXIS_PLANE_FACTOR", "FULL_NEIGHBOUR_SHELL", "FractureEnergyAudit",
    "analytic_fracture_energy_J_per_m2", "audit_config", "bond_failure_energy_J",
    "crossing_offset_sum", "fracture_energy_from_config",
    "ice_fracture_energy_J_per_m2", "lattice_spacing",
    "required_bond_stiffness", "required_breaking_strain",
]