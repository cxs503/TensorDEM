"""Measured structural / elastic metrics of a generated bonded-particle lattice.

This module is the *measurement* half of the HCP-packing verification: it only
post-processes a ``(positions, bond_indices, rest_lengths)`` triple that came out
of :mod:`tensordem.fracture_gen` and never re-uses the generator's geometry
formulas beyond the sphere radius and the cell volume.  The reference values it
is compared against (``pi/(3 sqrt 2)``, coordination number 12, the affine axial
modulus ``3 k / (2 sqrt 2 h)``) are written down in
``tests/test_fracture_gen_hcp_packing.py`` from hand derivation, not imported
from here.

Definitions
-----------
*coordination number* ``Z_i``
    Number of bonds incident to particle ``i`` (bond cutoff ``2.02 r``).

*interior particle*
    A particle whose coordination number equals the bulk value ``Z_bulk`` of the
    requested packing.  Using this -- instead of a geometric shell test -- keeps
    the elastic averages free of surface particles without special-casing the
    two lattice geometries.

*measured packing fraction*
    ``N * (4/3) pi r^3 / V_box`` with ``V_box`` the volume of the *domain* that
    was requested (not the bounding box of the centres).  Boundary trimming
    makes this converge to the exact lattice value from below as the lattice
    gets bigger, which is what the convergence table in
    ``docs/dem3d_fracture_hcp_packing.md`` reports.

*affine axial modulus* ``E``
    For an affine uniaxial strain ``eps`` along x the bond ``b`` of length
    ``l_b`` and unit direction ``n_b`` stretches by ``l_b eps n_{b,x}^2``, so the
    strain-energy density is ``(k eps^2 / 2 V) sum_b l_b^2 n_{b,x}^4`` and

        E_affine = (k / V) sum_b l_b^2 n_{b,x}^4 .

    With ``V = N V_cell`` and the sum restricted to interior particles, this is
    the affine (Voigt upper-bound) estimate that a central-force lattice
    attains under homogeneous strain.  It is exact for the infinite lattice and
    used here as the analytic anchor.
"""
from __future__ import annotations

from typing import Any

import torch

from .fracture_mass_mapping import (
    normalise_packing,
    packing_coordination_number,
    particle_cell_volume,
)

__all__ = [
    "bond_direction_cosines",
    "coordination_numbers",
    "coordination_histogram",
    "interior_mask",
    "min_pair_distance",
    "affine_axial_modulus_over_bond_stiffness",
    "structure_report",
]


def bond_direction_cosines(
    positions: torch.Tensor, bond_indices: torch.Tensor, rest_lengths: torch.Tensor
) -> torch.Tensor:
    """Unit bond vectors ``n_b = (x_j - x_i) / l_b``, shape ``(n_bonds, 3)``."""
    if len(bond_indices) == 0:
        return torch.empty((0, 3), dtype=positions.dtype, device=positions.device)
    delta = positions[bond_indices[:, 1]] - positions[bond_indices[:, 0]]
    return delta / rest_lengths[:, None]


def coordination_numbers(
    num_particles: int, bond_indices: torch.Tensor, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Per-particle bond count."""
    if len(bond_indices) == 0:
        return torch.zeros(num_particles, dtype=torch.long, device=device)
    return torch.bincount(
        torch.cat([bond_indices[:, 0], bond_indices[:, 1]]), minlength=num_particles
    )


def coordination_histogram(
    num_particles: int, bond_indices: torch.Tensor
) -> dict[int, int]:
    """``{coordination number: particle count}``."""
    counts = coordination_numbers(num_particles, bond_indices)
    histogram: dict[int, int] = {}
    for value, count in zip(*torch.unique(counts, return_counts=True)):
        histogram[int(value)] = int(count)
    return histogram


def interior_mask(
    num_particles: int,
    bond_indices: torch.Tensor,
    packing: str = "sc",
) -> torch.Tensor:
    """Mask of particles carrying the bulk coordination number."""
    bulk = packing_coordination_number(normalise_packing(packing))
    counts = coordination_numbers(num_particles, bond_indices)
    return counts == bulk


def min_pair_distance(positions: torch.Tensor) -> float:
    """Smallest centre-to-centre distance in the cloud (``inf`` for N < 2)."""
    if len(positions) < 2:
        return float("inf")
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(float("inf"))
    return float(distances.min())


def affine_axial_modulus_over_bond_stiffness(
    positions: torch.Tensor,
    bond_indices: torch.Tensor,
    rest_lengths: torch.Tensor,
    particle_radius: float,
    packing: str = "sc",
    *,
    interior_only: bool = True,
) -> float:
    """``E_affine / k`` in ``1/m`` for a uniaxial strain along x.

    See the module docstring for the derivation.  ``interior_only`` keeps only
    bonds whose *both* endpoints are bulk-coordinated particles, which removes
    the surface softening of a finite lattice.
    """
    n = len(positions)
    if len(bond_indices) == 0:
        raise ValueError("lattice has no bonds")
    n_hat = bond_direction_cosines(positions, bond_indices, rest_lengths)
    per_bond = rest_lengths ** 2 * n_hat[:, 0] ** 4
    if interior_only:
        mask = interior_mask(n, bond_indices, packing)
        if not bool(mask.any()):
            raise ValueError("lattice has no interior particles")
        # Each bond contributes half of its l^2 n_x^4 to each endpoint, so an
        # interior-interior bond is counted once and an interior-surface bond
        # half once -- the boundary bonds that a surface particle lacks are
        # therefore not silently dropped (which would bias E low).
        weight = 0.5 * (
            mask[bond_indices[:, 0]].to(per_bond.dtype)
            + mask[bond_indices[:, 1]].to(per_bond.dtype)
        )
        numerator = float((weight * per_bond).sum())
        n_active = int(mask.sum())
    else:
        numerator = float(per_bond.sum())
        n_active = n
    volume = n_active * particle_cell_volume(particle_radius, packing)
    return numerator / volume


def structure_report(
    positions: torch.Tensor,
    bond_indices: torch.Tensor,
    rest_lengths: torch.Tensor,
    particle_radius: float,
    domain_bounds: tuple[float, float, float, float, float, float],
    packing: str = "sc",
) -> dict[str, Any]:
    """One-stop structural audit of a generated lattice (machine-readable)."""
    packing = normalise_packing(packing)
    xmin, xmax, ymin, ymax, zmin, zmax = domain_bounds
    box_volume = abs((xmax - xmin) * (ymax - ymin) * (zmax - zmin))
    n = len(positions)
    sphere_volume = (4.0 / 3.0) * torch.pi * particle_radius ** 3
    bulk = packing_coordination_number(packing)
    mask = interior_mask(n, bond_indices, packing)
    lengths = torch.unique(rest_lengths.round(decimals=12)).tolist() if len(rest_lengths) else []
    # Volume actually spanned by the (boundary-trimmed) cloud, including the
    # outer half-diameter shell: this is the volume the lattice really fills and
    # the only denominator for which the measured packing fraction must equal
    # the exact lattice value at finite size.
    extent = positions.max(dim=0).values - positions.min(dim=0).values + 2.0 * particle_radius
    spanned_volume = float(extent[0] * extent[1] * extent[2])
    return {
        "packing": packing,
        "num_particles": n,
        "num_bonds": int(len(bond_indices)),
        "bond_lengths_m": [float(v) for v in lengths],
        "bulk_coordination_number": bulk,
        "num_interior_particles": int(mask.sum()),
        "interior_fraction": float(mask.sum()) / max(n, 1),
        "coordination_histogram": coordination_histogram(n, bond_indices),
        "min_pair_distance_m": min_pair_distance(positions),
        "sphere_volume_m3": float(sphere_volume),
        "box_volume_m3": float(box_volume),
        "measured_packing_fraction": float(n * sphere_volume / box_volume),
        "spanned_volume_m3": spanned_volume,
        "spanned_packing_fraction": float(n * sphere_volume / spanned_volume),
        "cell_volume_m3": particle_cell_volume(particle_radius, packing),
        "represented_volume_m3": n * particle_cell_volume(particle_radius, packing),
        "affine_modulus_over_k_1_per_m": affine_axial_modulus_over_bond_stiffness(
            positions, bond_indices, rest_lengths, particle_radius, packing
        ),
        "affine_modulus_over_k_all_bonds_1_per_m": (
            affine_axial_modulus_over_bond_stiffness(
                positions, bond_indices, rest_lengths, particle_radius, packing,
                interior_only=False,
            )
        ),
    }