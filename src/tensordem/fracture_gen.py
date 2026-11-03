"""Generate bonded-particle ice field from FEM failure faces.

This module implements the DEM side of SI-2 campaign: converting FEM cohesive
failure surfaces into a DEM ice field with pre-broken bonds. The input contract
is frozen and shared with TensorFEM SI-2/FEM leg.

Mass mapping: equal-volume spheres (frozen in prereg). Each DEM particle has
mass = density × (4/3 π r³). Total DEM mass = N_particles × particle_mass.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

import torch

from .fracture_mass_mapping import VALID_MASS_MODES, mass_mapping_report


# Supported mass mappings (see prereg/fracture_gen_prereg.json).
#   equal_volume_spheres - frozen legacy mapping, m = rho * (4/3) pi r^3
#                          (under-counts the represented cell mass by
#                          1 - pi/6 = 47.64 %)
#   cell_volume          - m = rho * (2r)^3, volume consistent
#   lattice_cell         - same mass as cell_volume, carried by an
#                          equivalent sphere of radius MASS_RADIUS_FACTOR*r
#                          so contact/buoyancy code sees a mass-consistent
#                          radius instead of the geometric one
MASS_MAPPINGS = frozenset({"equal_volume_spheres", "cell_volume", "lattice_cell"})
# Pre-audit ``mass_mode`` spellings accepted for backwards compatibility.
MASS_MAPPING_ALIASES = {"cell": "cell_volume", "sphere": "equal_volume_spheres"}
# Radius factor of the sphere whose volume equals the lattice cell (2r)^3.
MASS_RADIUS_FACTOR = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * 2.0
# Repository default: the volume-consistent mapping (frozen prereg intent, and
# the only mapping that reproduces the FEM continuum mass on an integer lattice).
DEFAULT_MASS_MAPPING = "cell_volume"
# Supported lateral face extents for bond-crossing detection.
FACE_EXTENTS = frozenset({"plane", "disc"})
# Largest particle count for which the O(N^2) vectorised pair search is used.
_VECTOR_PAIR_LIMIT = 5000


@dataclass
class FailureFace:
    """A single FEM failure face."""
    id: int
    area: float
    normal: torch.Tensor  # shape [3]
    centroid: torch.Tensor  # shape [3]
    G: float  # fracture energy release rate (J/m²)
    released_energy: float  # total energy released (J)
    # Explicit in-plane footprint radius (m).  ``None`` falls back to the
    # area-equivalent disc radius sqrt(area / pi).
    extent_radius: float | None = None


@dataclass
class FractureGenConfig:
    """Configuration for fracture generation."""
    density: float = 917.0  # ice density (kg/m³)
    particle_radius: float = 0.1  # particle radius (m)
    bond_stiffness: float = 2000.0  # bond stiffness (N/m)
    domain_bounds: tuple[float, float, float, float, float, float] | None = None
    # (xmin, xmax, ymin, ymax, zmin, zmax)
    # Mass mapping. The frozen prereg decision is "equal_volume_spheres"; the
    # alternative "cell_volume" assigns each particle the mass of its lattice
    # cell (density * (2r)^3) so that a lattice filling the domain reproduces
    # the FEM continuum mass exactly. See docs/dem3d_fracture_gen_audit.md.
    mass_mapping: str | None = None
    # Legacy pre-audit switch of ``tensordem.fracture_mass_mapping``:
    # ``None`` derives the mapping from ``mass_mapping``, "cell" means
    # lattice-cell mass, "sphere" the inscribed-sphere legacy mass.
    mass_mode: str | None = None
    # Lateral extent of a failure face used for bond-crossing detection.
    # "plane": infinite plane through centroid (frozen behaviour).
    # "disc":  bonds must additionally intersect the circular disc of area
    #          `face.area` centred on `face.centroid`.
    face_extent: str = "plane"
    # Footprint-aware bond cutting: equivalent to ``face_extent="disc"``.
    footprint_aware: bool = False
    # Additionally sever bonds with exactly one endpoint on the plane.
    on_plane_tolerance: float = 0.0

    def __post_init__(self):
        if self.density <= 0:
            raise ValueError("density must be positive")
        if self.particle_radius <= 0:
            raise ValueError("particle_radius must be positive")
        if self.bond_stiffness <= 0:
            raise ValueError("bond_stiffness must be positive")
        if self.mass_mapping is not None and self.mass_mapping not in (
            set(MASS_MAPPINGS) | set(MASS_MAPPING_ALIASES)
        ):
            raise ValueError(
                f"unknown mass_mapping {self.mass_mapping!r}; "
                f"expected one of {sorted(MASS_MAPPINGS)}"
            )
        if self.mass_mode is not None and self.mass_mode not in VALID_MASS_MODES:
            raise ValueError(
                f"mass_mode must be one of {VALID_MASS_MODES}, got {self.mass_mode!r}"
            )
        if not math.isfinite(self.on_plane_tolerance) or self.on_plane_tolerance < 0.0:
            raise ValueError("on_plane_tolerance must be finite and nonnegative")
        if self.face_extent not in FACE_EXTENTS:
            raise ValueError(
                f"unknown face_extent {self.face_extent!r}; "
                f"expected one of {sorted(FACE_EXTENTS)}"
            )

    @property
    def resolved_mass_mapping(self) -> str:
        """Canonical mass mapping name (``mass_mapping`` wins over ``mass_mode``)."""
        if self.mass_mapping is not None:
            return MASS_MAPPING_ALIASES.get(self.mass_mapping, self.mass_mapping)
        if self.mass_mode is not None:
            return MASS_MAPPING_ALIASES[self.mass_mode]
        return DEFAULT_MASS_MAPPING


class FractureField:
    """DEM ice field initialized from FEM failure faces.
    
    This field can be directly fed into IceDEM3D.step(external_forces=...).
    """
    
    def __init__(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        bond_indices: torch.Tensor,
        rest_lengths: torch.Tensor,
        alive: torch.Tensor,
        particle_radius: float,
        density: float,
        bond_stiffness: float,
        mass_mapping: str | None = None,
        face_extent: str = "plane",
        mass_mode: str | None = None,
        footprint_aware: bool = False,
        on_plane_tolerance: float = 0.0,
    ):
        if mass_mapping is not None and mass_mapping not in (
            set(MASS_MAPPINGS) | set(MASS_MAPPING_ALIASES)
        ):
            raise ValueError(f"unknown mass_mapping {mass_mapping!r}")
        if mass_mode is not None and mass_mode not in VALID_MASS_MODES:
            raise ValueError(
                f"mass_mode must be one of {VALID_MASS_MODES}, got {mass_mode!r}"
            )
        if face_extent not in FACE_EXTENTS:
            raise ValueError(f"unknown face_extent {face_extent!r}")
        if mass_mapping is not None:
            resolved = MASS_MAPPING_ALIASES.get(mass_mapping, mass_mapping)
        elif mass_mode is not None:
            resolved = MASS_MAPPING_ALIASES[mass_mode]
        else:
            resolved = DEFAULT_MASS_MAPPING
        self.mass_mapping = resolved
        # Keep the legacy attribute so pre-audit callers/tests still see a name.
        self.mass_mode = mass_mode
        self.face_extent = face_extent
        self.footprint_aware = bool(footprint_aware)
        self.on_plane_tolerance = float(on_plane_tolerance)
        self.positions = positions
        self.velocities = velocities
        self.bond_indices = bond_indices
        self.rest_lengths = rest_lengths
        self.alive = alive  # boolean mask, False = broken
        self.particle_radius = particle_radius
        self.density = density
        self.bond_stiffness = bond_stiffness
    
    @property
    def num_particles(self) -> int:
        return len(self.positions)
    
    @property
    def cell_volume(self) -> float:
        """Volume of one simple-cubic lattice cell, (2r)^3."""
        return (2.0 * self.particle_radius) ** 3

    @property
    def particle_cell_volume(self) -> float:
        """Alias of :attr:`cell_volume` (``fracture_mass_mapping`` naming)."""
        return self.cell_volume

    @property
    def represented_volume(self) -> float:
        """Ice volume represented by the lattice, N * (2r)^3."""
        return self.num_particles * self.cell_volume

    @property
    def mass_radius(self) -> float:
        """Radius that carries the particle mass.

        ``lattice_cell`` replaces the geometric radius by the radius of the
        sphere whose volume equals the lattice cell, so that mass-consistent
        code paths (contact, buoyancy) see ``m = rho * (2r)^3``.
        """
        if self.mass_mapping == "lattice_cell":
            return MASS_RADIUS_FACTOR * self.particle_radius
        return self.particle_radius

    @property
    def particle_mass(self) -> float:
        """Mass of a single particle under the selected mass mapping."""
        if self.mass_mapping in ("cell_volume", "lattice_cell"):
            return self.density * self.cell_volume
        return self.density * (4.0 / 3.0) * math.pi * self.mass_radius**3

    @property
    def volume_fraction(self) -> float:
        """Solid volume fraction implied by the mass mapping (1.0 = cell mass)."""
        if self.mass_mapping in ("cell_volume", "lattice_cell"):
            return 1.0
        return math.pi / 6.0

    @property
    def mass_mode_legacy(self) -> str:
        """``mass_mode`` spelling of the resolved mapping ("cell"/"sphere")."""
        return "sphere" if self.mass_mapping == "equal_volume_spheres" else "cell"
    
    @property
    def total_mass(self) -> float:
        """Total mass of all particles."""
        return self.num_particles * self.particle_mass
    
    @property
    def num_broken_bonds(self) -> int:
        """Number of broken bonds."""
        return (~self.alive).sum().item()
    
    @property
    def num_intact_bonds(self) -> int:
        """Number of intact bonds."""
        return self.alive.sum().item()
    
    def to_icedem3d_initial_state(self) -> dict[str, torch.Tensor]:
        """Convert to initial state dict for IceDEM3D."""
        return {
            'positions': self.positions,
            'velocities': self.velocities,
            'bond_indices': self.bond_indices,
            'rest_lengths': self.rest_lengths,
            'alive': self.alive,
        }


def load_failure_faces(path: str) -> list[FailureFace]:
    """Load failure faces from JSON file.
    
    Input format (frozen contract):
    {
        "faces": [
            {
                "id": int,
                "area": float,
                "normal": [x, y, z],
                "centroid": [x, y, z],
                "G": float,
                "released_energy": float
            }
        ],
        "units": "SI"
    }
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    if data.get('units') != 'SI':
        raise ValueError(f"Only SI units supported, got {data.get('units')}")
    
    faces = []
    for face_data in data['faces']:
        face = FailureFace(
            id=face_data['id'],
            area=face_data['area'],
            normal=torch.tensor(face_data['normal'], dtype=torch.float64),
            centroid=torch.tensor(face_data['centroid'], dtype=torch.float64),
            G=face_data['G'],
            released_energy=face_data['released_energy']
        )
        faces.append(face)
    
    return faces


def generate_particle_lattice(
    domain_bounds: tuple[float, float, float, float, float, float],
    particle_radius: float,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = 'cpu'
) -> torch.Tensor:
    """Generate a simple cubic lattice of particles.
    
    Particles are placed at grid points with spacing 2*particle_radius.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = domain_bounds
    for lo, hi in ((xmin, xmax), (ymin, ymax), (zmin, zmax)):
        if hi - lo < particle_radius:
            raise ValueError(
                "each domain extent must be at least one particle radius "
                f"({particle_radius} m); got {hi - lo} m"
            )
    spacing = 2.0 * particle_radius
    
    # Generate grid points (offset by radius to center particles in domain)
    x_coords = torch.arange(xmin + particle_radius, xmax, spacing, dtype=dtype, device=device)
    y_coords = torch.arange(ymin + particle_radius, ymax, spacing, dtype=dtype, device=device)
    z_coords = torch.arange(zmin + particle_radius, zmax, spacing, dtype=dtype, device=device)
    
    # Create meshgrid
    xx, yy, zz = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    positions = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
    
    return positions


def build_bond_topology(
    positions: torch.Tensor,
    particle_radius: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build bond topology for simple cubic lattice.

    Bonds connect nearest neighbours (distance = 2 * particle_radius). The
    pair search is a vectorised all-pairs distance matrix for small lattices
    and falls back to the incremental loop above ``_VECTOR_PAIR_LIMIT``
    particles to bound the O(N^2) distance-matrix footprint.
    """
    n_particles = len(positions)
    spacing = 2.0 * particle_radius
    if n_particles < 2:
        return (torch.empty((0, 2), dtype=torch.long, device=positions.device),
                torch.empty(0, dtype=positions.dtype, device=positions.device))

    if n_particles <= _VECTOR_PAIR_LIMIT:
        distances = torch.cdist(positions, positions)
        upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        close = (distances <= spacing * 1.01) & upper
        bond_indices = close.nonzero(as_tuple=False)  # row-major = (i, j) ascending
    else:  # pragma: no cover - only reached for very large lattices
        bond_pairs = []
        for i in range(n_particles):
            dist = torch.linalg.norm(positions[i + 1:] - positions[i], dim=1)
            for offset in (dist <= spacing * 1.01).nonzero(as_tuple=False).flatten().tolist():
                bond_pairs.append([i, i + 1 + offset])
        bond_indices = torch.tensor(bond_pairs, dtype=torch.long, device=positions.device)

    if len(bond_indices) == 0:
        return (torch.empty((0, 2), dtype=torch.long, device=positions.device),
                torch.empty(0, dtype=positions.dtype, device=positions.device))

    bond_indices = bond_indices.to(device=positions.device)
    rest_lengths = torch.linalg.norm(
        positions[bond_indices[:, 1]] - positions[bond_indices[:, 0]],
        dim=1
    )

    return bond_indices, rest_lengths


def face_footprint_radius(face: FailureFace) -> float:
    """In-plane footprint radius of a failure face (m).

    Uses the explicit ``extent_radius`` when the producer supplies one, else the
    area-equivalent disc radius sqrt(area / pi) of the frozen contract.
    """
    if face.extent_radius is not None:
        if face.extent_radius <= 0:
            raise ValueError("extent_radius must be positive")
        return float(face.extent_radius)
    if face.area <= 0:
        raise ValueError("area must be positive")
    return math.sqrt(face.area / math.pi)


def identify_bonds_crossing_face(
    bond_indices: torch.Tensor,
    positions: torch.Tensor,
    face: FailureFace,
    face_extent: str = "plane",
    footprint_aware: bool | None = None,
    on_plane_tolerance: float = 0.0,
) -> torch.Tensor:
    """Identify bonds that cross a failure face.

    A bond crosses a face if the two endpoints are on opposite sides of the
    face plane (defined by normal and centroid).

    ``face_extent="plane"`` (default, frozen behaviour) treats the face as an
    infinite plane, so every lattice bond straddling the plane is broken even
    if it lies far outside the physical face area.
    ``face_extent="disc"`` (or, equivalently, ``footprint_aware=True``)
    additionally requires the bond/plane intersection point to lie inside the
    circular disc of area ``face.area`` centred on ``face.centroid`` (in-plane
    radius :func:`face_footprint_radius`); this keeps the number of broken bonds
    consistent with the face area, and therefore with the fracture energy that
    face can release.

    ``on_plane_tolerance > 0`` also severs bonds with exactly one endpoint on the
    plane (|d| <= tolerance), which restores the expected cut when the failure
    plane passes through a particle layer instead of between two layers.
    """
    if face_extent not in FACE_EXTENTS:
        raise ValueError(
            f"unknown face_extent {face_extent!r}; expected one of {sorted(FACE_EXTENTS)}"
        )
    if not math.isfinite(on_plane_tolerance) or on_plane_tolerance < 0.0:
        raise ValueError("on_plane_tolerance must be finite and nonnegative")
    disc = face_extent == "disc" or bool(footprint_aware)
    if len(bond_indices) == 0:
        return torch.empty(0, dtype=torch.bool, device=positions.device)

    i_idx, j_idx = bond_indices.T
    pos_i = positions[i_idx]
    pos_j = positions[j_idx]

    # Move face data to same device/dtype as positions
    centroid = face.centroid.to(device=positions.device, dtype=positions.dtype)
    normal = face.normal.to(device=positions.device, dtype=positions.dtype)
    norm = torch.linalg.norm(normal)
    if not torch.isfinite(norm) or float(norm) == 0.0:
        raise ValueError("face normal must be a finite non-zero vector")
    normal = normal / norm

    # Signed distance from face plane: normal . (pos - centroid)
    dist_i = torch.sum((pos_i - centroid) * normal, dim=1)
    dist_j = torch.sum((pos_j - centroid) * normal, dim=1)

    # Bond crosses face if signs are opposite (one positive, one negative)
    crosses = (dist_i * dist_j) < 0

    if on_plane_tolerance > 0.0:
        near_i = dist_i.abs() <= on_plane_tolerance
        near_j = dist_j.abs() <= on_plane_tolerance
        crosses = crosses | (near_i & ~near_j) | (near_j & ~near_i)

    if disc and bool(crosses.any()):
        denom = dist_i - dist_j
        # crosses implies dist_i != dist_j, so the division is safe.
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        t = torch.where(crosses, dist_i / denom, torch.zeros_like(denom))
        x_cross = pos_i + t.unsqueeze(1) * (pos_j - pos_i)
        delta = x_cross - centroid
        # Remove the normal component to get the in-plane offset.
        in_plane = delta - torch.sum(delta * normal, dim=1, keepdim=True) * normal
        radius = face_footprint_radius(face)
        inside = torch.linalg.norm(in_plane, dim=1) <= radius * (1.0 + 1e-12)
        crosses = crosses & inside

    return crosses


def generate_fracture_field(
    failure_faces: list[FailureFace],
    config: FractureGenConfig
) -> FractureField:
    """Generate DEM ice field from FEM failure faces.
    
    1. Generate particle lattice filling the domain
    2. Build bond topology (nearest neighbors)
    3. Break bonds that cross failure faces
    """
    if config.domain_bounds is None:
        raise ValueError("domain_bounds must be specified")
    
    # Generate particle lattice
    positions = generate_particle_lattice(
        config.domain_bounds, 
        config.particle_radius
    )
    velocities = torch.zeros_like(positions)
    
    # Build bond topology
    bond_indices, rest_lengths = build_bond_topology(positions, config.particle_radius)
    
    # Initialize all bonds as alive
    alive = torch.ones(len(bond_indices), dtype=torch.bool, device=positions.device)
    
    # Break bonds at failure faces
    for face in failure_faces:
        crosses = identify_bonds_crossing_face(
            bond_indices,
            positions,
            face,
            face_extent=config.face_extent,
            footprint_aware=config.footprint_aware,
            on_plane_tolerance=config.on_plane_tolerance,
        )
        alive &= ~crosses
    
    return FractureField(
        positions=positions,
        velocities=velocities,
        bond_indices=bond_indices,
        rest_lengths=rest_lengths,
        alive=alive,
        particle_radius=config.particle_radius,
        density=config.density,
        bond_stiffness=config.bond_stiffness,
        mass_mapping=config.mass_mapping,
        face_extent=config.face_extent,
        mass_mode=config.mass_mode,
        footprint_aware=config.footprint_aware,
        on_plane_tolerance=config.on_plane_tolerance,
    )


class FractureLedger:
    """Accounting ledger for mass and energy conservation.

    The frozen prereg criteria c1/c2 compare DEM bookkeeping against itself
    (``unit_bond_energy`` was *defined* as ``total_released / num_broken``, so
    multiplying it back is a tautology). This ledger therefore also exposes the
    falsifiable lattice analytics: the continuum (FEM) mass of the domain under
    each mass mapping, and the fracture energy per bond implied by the face
    area, ``G * (2r)^2``. See docs/dem3d_fracture_gen_audit.md.
    """

    def __init__(self, fracture_field: FractureField, failure_faces: list[FailureFace]):
        self.field = fracture_field
        self.faces = failure_faces

    def mass_conservation_error(self, fem_mass: float) -> float:
        """Compute mass conservation error: |DEM_mass - FEM_mass|."""
        dem_mass = self.field.total_mass
        return abs(dem_mass - fem_mass)

    def mass_accounting(self, domain_volume: float | None = None) -> dict[str, Any]:
        """Mass bookkeeping against the FEM continuum mass rho * V.

        ``domain_volume`` defaults to the particle count times the lattice cell
        volume, i.e. the volume actually discretised by the lattice.  The
        continuum reference is ``density * domain_volume``.

        ``relative_error`` is signed (DEM relative to continuum, negative means
        mass is lost); ``absolute_relative_error`` is its magnitude and
        ``within_15pct`` is the 15 % acceptance gate of the platform.
        """
        field = self.field
        if domain_volume is not None and domain_volume <= 0:
            raise ValueError("domain_volume must be positive")
        covered = field.num_particles * field.cell_volume
        if domain_volume is None:
            domain_volume = covered
        fem_mass = field.density * domain_volume
        dem_mass = field.total_mass
        signed = (dem_mass - fem_mass) / fem_mass if fem_mass > 0 else float("nan")
        return {
            "mass_mapping": field.mass_mapping,
            "face_extent": field.face_extent,
            "volume_fraction": field.volume_fraction,
            "num_particles": field.num_particles,
            "domain_volume": domain_volume,
            "fem_continuum_mass": fem_mass,
            "continuum_mass": fem_mass,
            "dem_mass": dem_mass,
            "absolute_error": dem_mass - fem_mass,
            "relative_error": signed,
            "signed_relative_error": signed,
            "absolute_relative_error": abs(signed),
            "within_15pct": abs(signed) <= 0.15,
            # True only if the lattice cell volume equals the target volume,
            # i.e. no volume was lost to lattice quantisation.
            "lattice_volume_covered": covered,
        }

    def lattice_volume(self) -> float:
        """Volume occupied by the particle lattice, N * (2r)^3 (m^3)."""
        return self.field.num_particles * self.field.cell_volume

    def mass_mapping_report(
        self,
        fem_mass: float | None = None,
        tolerance: float = 0.01,
        domain_bounds: tuple[float, float, float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        """Audit the DEM mass against the FEM ice mass or rho * V_domain.

        Delegates to :func:`tensordem.fracture_mass_mapping.mass_mapping_report`.
        ``domain_bounds`` defaults to the bounding box of the realised particle
        positions expanded by one radius per face (each particle owns a full
        cell, its centre only spans (N-1) h per axis).
        """
        if domain_bounds is None:
            pos = self.field.positions
            lo = pos.min(dim=0).values.tolist()
            hi = pos.max(dim=0).values.tolist()
            r = self.field.particle_radius
            domain_bounds = (
                lo[0] - r, hi[0] + r, lo[1] - r, hi[1] + r, lo[2] - r, hi[2] + r
            )
        return mass_mapping_report(
            density=self.field.density,
            particle_radius=self.field.particle_radius,
            domain_bounds=domain_bounds,
            mass_mode=self.field.mass_mode_legacy,
            fem_mass=fem_mass,
            tolerance=tolerance,
            num_particles=self.field.num_particles,
        )

    def energy_accounting(self) -> dict[str, Any]:
        """Account for fracture energy.

        Returns dict with:
        - total_released_energy: sum of released_energy from all faces
        - num_broken_bonds: number of broken bonds in DEM field
        - unit_bond_energy: total_released_energy / num_broken_bonds
          (bookkeeping identity, not a physical check)
        - analytic_bond_count: lattice estimate of bonds crossing the faces,
          sum(area) / (2r)^2
        - analytic_unit_bond_energy: fracture energy a *single* bond should
          carry, G * (2r)^2, area-weighted over the faces
        - unit_bond_energy_relative_error: |implied - analytic| / analytic
        - input_consistency_error: sum |released_energy - G * area|; the input
          contract carries both fields, so they must agree unless the FEM side
          deliberately apportions energy differently
        """
        field = self.field
        spacing = 2.0 * field.particle_radius
        total_released = sum(face.released_energy for face in self.faces)
        num_broken = field.num_broken_bonds
        unit_bond_energy = total_released / num_broken if num_broken > 0 else 0.0

        total_area = sum(face.area for face in self.faces)
        analytic_bond_count = total_area / spacing ** 2 if total_area > 0 else 0.0
        if total_area > 0:
            weighted_G = sum(face.G * face.area for face in self.faces) / total_area
        else:
            weighted_G = 0.0
        analytic_unit_bond_energy = weighted_G * spacing ** 2
        if analytic_unit_bond_energy > 0:
            rel = abs(unit_bond_energy - analytic_unit_bond_energy) / analytic_unit_bond_energy
        else:
            rel = float("nan")
        input_consistency_error = sum(
            abs(face.released_energy - face.G * face.area) for face in self.faces
        )

        return {
            "total_released_energy": total_released,
            "num_broken_bonds": num_broken,
            "unit_bond_energy": unit_bond_energy,
            "analytic_bond_count": analytic_bond_count,
            "bond_count_relative_error": (
                abs(num_broken - analytic_bond_count) / analytic_bond_count
                if analytic_bond_count > 0 else float("nan")
            ),
            "analytic_unit_bond_energy": analytic_unit_bond_energy,
            "unit_bond_energy_relative_error": rel,
            "input_consistency_error": input_consistency_error,
            "face_extent": field.face_extent,
        }

    def check_bonds_at_faces(self) -> dict[str, Any]:
        """Verify that bonds crossing failure faces are broken.
        
        Returns dict with:
        - bonds_crossing_faces: total bonds that should be broken
        - bonds_actually_broken: bonds that are actually broken
        - all_face_bonds_broken: boolean, True if all face bonds are broken
        """
        # Re-identify bonds crossing each face
        bonds_should_break = torch.zeros(len(self.field.bond_indices), 
                                         dtype=torch.bool, 
                                         device=self.field.bond_indices.device)
        
        for face in self.faces:
            crosses = identify_bonds_crossing_face(
                self.field.bond_indices,
                self.field.positions,
                face,
                face_extent=self.field.face_extent,
                footprint_aware=self.field.footprint_aware,
                on_plane_tolerance=self.field.on_plane_tolerance,
            )
            bonds_should_break |= crosses
        
        bonds_actually_broken = ~self.field.alive
        
        return {
            'bonds_crossing_faces': bonds_should_break.sum().item(),
            'bonds_actually_broken': bonds_actually_broken.sum().item(),
            'all_face_bonds_broken': torch.all(bonds_should_break == bonds_actually_broken).item()
        }
