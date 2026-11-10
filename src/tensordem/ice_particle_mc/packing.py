"""Specimen and ice-layer packings for the ice-particle Mohr-Coulomb verification.

All lengths are SI (m).  The packings are *jittered lattices*: a regular lattice
would expose only three bond orientations, and a Mohr-Coulomb criterion sampled
on three orientations cannot represent the continuum envelope.  Jitter (default
15% of the lattice spacing) randomises the bond-orientation set while keeping the
specimen non-overlapping, so the initial state carries no bond force.

Generated geometry is never trusted blindly: :func:`validate_cylinder` reports
the measured diameter / height / volume / coordination number so a report can
show that the digital specimen matches the requested geometry before any
mechanics is claimed (geometry is verified before strength).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = ["Specimen", "cylinder_specimen", "ice_layer", "bond_pairs", "relax_overlaps",
           "validate_cylinder", "coordination_number", "orientation_coverage"]


@dataclass(frozen=True)
class Specimen:
    """A jittered particle packing plus its candidate bonds (no mechanics yet)."""

    positions: torch.Tensor      # (N, 3)
    radii: torch.Tensor          # (N,)
    bond_i: torch.Tensor         # (M,)
    bond_j: torch.Tensor         # (M,)
    bond_length: torch.Tensor    # (M,)
    diameter: float
    height: float
    volume: float

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_bonds(self) -> int:
        return int(self.bond_i.shape[0])


def _lattice_offsets(nx: int, ny: int, nz: int, spacing: float) -> torch.Tensor:
    xs = (torch.arange(nx, dtype=torch.float64) - (nx - 1) / 2.0) * spacing
    ys = (torch.arange(ny, dtype=torch.float64) - (ny - 1) / 2.0) * spacing
    zs = torch.arange(nz, dtype=torch.float64) * spacing
    grid = torch.meshgrid(xs, ys, zs, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grid], dim=1)


def cylinder_specimen(diameter: float = 0.05, height: float = 0.10,
                      particle_radius: float = 0.0030,
                      spacing_fraction: float = 1.0,
                      jitter_fraction: float = 0.15,
                      radius_polydispersity: float = 0.10,
                      seed: int = 0,
                      bond_gap_fraction: float = 0.25) -> Specimen:
    """Jittered cylindrical specimen (defaults: D = 50 mm, H = 100 mm).

    ``spacing_fraction`` is the lattice spacing in units of ``2 r`` and is
    clamped to >= 1 (particles must not interpenetrate at t = 0); the bond reach
    ``(r_i + r_j)(1 + bond_gap_fraction)`` then connects the lattice.
    """
    if not (diameter > 0 and height > 0 and particle_radius > 0):
        raise ValueError("diameter, height and particle_radius must be positive")
    spacing = 2.0 * particle_radius * max(float(spacing_fraction), 1.0)
    nx = int(math.ceil(diameter / spacing)) + 1
    ny = nx
    nz = int(math.ceil(height / spacing)) + 1
    lattice = _lattice_offsets(nx, ny, nz, spacing)

    gen = torch.Generator().manual_seed(int(seed))
    jitter = torch.rand(lattice.shape, generator=gen, dtype=torch.float64) - 0.5
    lattice = lattice + jitter * (jitter_fraction * spacing)
    radial = lattice[:, :2].norm(dim=1)
    keep = radial <= (diameter / 2.0 - particle_radius)
    lattice = lattice[keep]

    radii = particle_radius * (
        1.0 + radius_polydispersity
        * (torch.rand(lattice.shape[0], generator=gen, dtype=torch.float64) - 0.5) * 2.0)
    order = torch.argsort(lattice[:, 2])
    positions = lattice[order].contiguous()
    radii = radii[order].contiguous()

    # the relaxation can push particles radially outward, and pulling them back
    # can re-create overlaps: alternate the two projections until both hold
    for _ in range(6):
        positions, _worst = relax_overlaps(positions, radii, iterations=120)
        centre = positions[:, :2].norm(dim=1).clamp_min(1e-12)
        limit = (diameter / 2.0) - radii
        excess = centre > limit
        if not bool(excess.any()):
            break
        scale = torch.where(excess, limit / centre, torch.ones_like(centre))
        positions = positions.clone()
        positions[:, :2] *= scale[:, None]
    # enforce the *requested* height exactly, after the relaxation (the
    # relaxation moves particles, so any earlier z scaling would be undone)
    positions = positions - positions[:, 2].min() + radii.max()
    z_extent = float(positions[:, 2].max() + radii.max())
    positions = positions.clone()
    positions[:, 2] *= float(height) / z_extent
    # final hard geometric clamp: no particle centre may sit outside
    # (D/2 - r), otherwise the measured diameter would exceed the request
    centre = positions[:, :2].norm(dim=1)
    limit = ((diameter / 2.0) - radii).clamp_min(1e-6)
    ratio = torch.where(centre > limit, limit / centre.clamp_min(1e-12),
                        torch.ones_like(centre))
    positions = positions.clone()
    positions[:, :2] *= ratio[:, None]
    height_eff = float(positions[:, 2].max() + radii.max())
    volume = math.pi * (diameter / 2.0) ** 2 * height_eff
    bi, bj, length = bond_pairs(positions, radii, gap_fraction=bond_gap_fraction)
    return Specimen(positions=positions, radii=radii, bond_i=bi, bond_j=bj,
                    bond_length=length, diameter=float(diameter),
                    height=height_eff, volume=volume)


def ice_layer(width: float = 0.30, depth: float = 0.30, height: float = 0.06,
              particle_radius: float = 0.020,
              jitter_fraction: float = 0.15,
              radius_polydispersity: float = 0.10,
              seed: int = 0,
              bond_gap_fraction: float = 0.25) -> Specimen:
    """Rectangular bonded ice layer (a level-ice slab for indentation tests)."""
    spacing = 2.0 * particle_radius
    nx = max(int(math.floor((width - 2 * particle_radius) / spacing)) + 1, 2)
    ny = max(int(math.floor((depth - 2 * particle_radius) / spacing)) + 1, 2)
    nz = max(int(math.floor((height - 2 * particle_radius) / spacing)) + 1, 2)
    lattice = _lattice_offsets(nx, ny, nz, spacing)
    gen = torch.Generator().manual_seed(int(seed))
    jitter = torch.rand(lattice.shape, generator=gen, dtype=torch.float64) - 0.5
    lattice = lattice + jitter * (jitter_fraction * spacing)
    lattice[:, 0] -= lattice[:, 0].min() - particle_radius
    lattice[:, 1] -= lattice[:, 1].min() - particle_radius
    lattice[:, 2] -= lattice[:, 2].min() - particle_radius
    radii = particle_radius * (
        1.0 + radius_polydispersity
        * (torch.rand(lattice.shape[0], generator=gen, dtype=torch.float64) - 0.5) * 2.0)
    lattice, _worst = relax_overlaps(lattice, radii)
    height_eff = float(lattice[:, 2].max() + radii.max())
    volume = float(width) * float(depth) * height_eff
    bi, bj, length = bond_pairs(lattice, radii, gap_fraction=bond_gap_fraction)
    return Specimen(positions=lattice.contiguous(), radii=radii.contiguous(),
                    bond_i=bi, bond_j=bj, bond_length=length,
                    diameter=float(width), height=height_eff, volume=volume)


def relax_overlaps(positions: torch.Tensor, radii: torch.Tensor,
                   iterations: int = 300,
                   reach_factor: float = 1.6) -> tuple[torch.Tensor, float]:
    """Position-based overlap removal (a cheap geometric relaxation).

    A jittered lattice leaves a handful of pairs closer than ``r_i + r_j``.  Those
    pairs are *not* bonds but they would become Hertz contacts with a large
    initial overlap, and the resulting force would blow the run up.  This
    relaxation pushes every overlapping pair apart along its axis before the bond
    graph is built; it changes geometry only, never mechanics.
    """
    pos = positions.clone()
    if pos.shape[0] < 2:
        return pos, 0.0
    for _ in range(iterations):
        i, j = torch.triu_indices(pos.shape[0], pos.shape[0], offset=1)
        delta = pos[j] - pos[i]
        distance = delta.norm(dim=1).clamp_min(1e-15)
        limit = (radii[i] + radii[j]) * reach_factor
        overlap = (radii[i] + radii[j]) - distance
        mask = overlap > 1e-9
        if not bool(mask.any()):
            break
        direction = delta / distance[:, None]
        correction = direction * (0.5 * overlap).clamp_min(0.0)[:, None]
        push = torch.zeros_like(pos)
        push.index_add_(0, i[mask], -correction[mask])
        push.index_add_(0, j[mask], correction[mask])
        pos = pos + push
    worst = 0.0
    if pos.shape[0] >= 2:
        i, j = torch.triu_indices(pos.shape[0], pos.shape[0], offset=1)
        distance = (pos[j] - pos[i]).norm(dim=1)
        worst = float(((radii[i] + radii[j]) - distance).max())
    return pos, worst


def bond_pairs(positions: torch.Tensor, radii: torch.Tensor,
               gap_fraction: float = 0.25
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All pairs closer than ``(r_i + r_j)(1 + gap_fraction)`` become bonds."""
    n = int(positions.shape[0])
    i, j = torch.triu_indices(n, n, offset=1)
    distance = (positions[j] - positions[i]).norm(dim=1)
    reach = (radii[i] + radii[j]) * (1.0 + gap_fraction)
    mask = distance <= reach
    return i[mask], j[mask], distance[mask]


def coordination_number(specimen: Specimen) -> float:
    """Mean number of bonds per particle."""
    if specimen.n_particles == 0:
        return 0.0
    degree = torch.bincount(torch.cat([specimen.bond_i, specimen.bond_j]),
                            minlength=specimen.n_particles)
    return float(degree.double().mean())


def validate_cylinder(positions: torch.Tensor, radii: torch.Tensor,
                      diameter: float, height: float) -> dict[str, float]:
    """Measured geometry of the digital specimen (must match the request)."""
    radial_extent = float((positions[:, :2].norm(dim=1) + radii).max())
    z_extent = float(positions[:, 2].max() + radii.max() - (positions[:, 2].min() - radii.min()))
    volume = float((4.0 / 3.0 * math.pi * radii.pow(3)).sum())
    bulk = math.pi * (diameter / 2.0) ** 2 * height
    return {
        "requested_diameter_m": float(diameter),
        "requested_height_m": float(height),
        "measured_diameter_m": 2.0 * radial_extent,
        "measured_height_m": z_extent,
        "diameter_error": abs(2.0 * radial_extent - diameter) / diameter,
        "height_error": abs(z_extent - height) / height,
        "solid_fraction": volume / bulk,
        "worst_overlap_m": _worst_overlap(positions, radii),
        "n_particles": float(positions.shape[0]),
    }


def orientation_coverage(specimen: Specimen, friction_angle_deg: float) -> dict[str, float]:
    """Bond-orientation coverage around the MC critical plane ``n_z^2 = (1-sin)/2``.

    A bond set that does not sample the critical orientation cannot express the
    continuum envelope; the error is second order in the angular gap, so this
    reports both the fraction of bonds inside ``+-0.1`` of the critical
    ``n_z^2`` and the distance from the closest bond to it.
    """
    if specimen.n_bonds == 0:
        return {"fraction_near_critical": 0.0, "closest_gap": float("nan")}
    delta = specimen.positions[specimen.bond_j] - specimen.positions[specimen.bond_i]
    normals = delta / delta.norm(dim=1, keepdim=True)
    x = (normals[:, 2] ** 2).double()
    critical = 0.5 * (1.0 - math.sin(math.radians(friction_angle_deg)))
    return {
        "critical_n_z_squared": critical,
        "fraction_near_critical": float(((x - critical).abs() < 0.1).double().mean()),
        "closest_gap": float((x - critical).abs().min()),
        "n_z_squared_p05": float(torch.quantile(x, 0.05)),
        "n_z_squared_p95": float(torch.quantile(x, 0.95)),
    }


def _worst_overlap(positions: torch.Tensor, radii: torch.Tensor) -> float:
    """Largest initial interpenetration of any pair (m); must be ~0."""
    if positions.shape[0] < 2:
        return 0.0
    i, j = torch.triu_indices(positions.shape[0], positions.shape[0], offset=1)
    distance = (positions[j] - positions[i]).norm(dim=1)
    return float(((radii[i] + radii[j]) - distance).max())
