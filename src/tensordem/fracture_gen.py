"""Generate bonded-particle ice field from FEM failure faces.

This module implements the DEM side of SI-2 campaign: converting FEM cohesive
failure surfaces into a DEM ice field with pre-broken bonds. The input contract
is frozen and shared with TensorFEM SI-2/FEM leg.

Mass mapping: equal-volume spheres (frozen in prereg). A particle on the
simple-cubic lattice of spacing h = 2 r owns a cell of volume h³, so the
"equal-volume sphere" is the sphere that has that cell's volume: the volume
consistent particle mass is density × h³ (``mass_mode="cell"``, the default).
The inscribed-sphere mapping density × (4/3 π r³) is kept as
``mass_mode="sphere"`` for reproducing the original pre-audit numbers; it
under-counts the mass of the represented ice volume by 1 - π/6 = 47.64 %.
See ``tensordem.fracture_mass_mapping`` and
``docs/dem3d_fracture_mass_mapping_audit.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

import torch

from .fracture_mass_mapping import (
    VALID_MASS_MODES,
    mass_mapping_report,
    particle_cell_volume,
    particle_mass as _particle_mass,
)

#: Mass (inertial) radius per geometric radius for a volume-consistent particle
#: on the simple-cubic lattice ``h = 2 r``: ``r_m = (6/pi)^(1/3) r = 1.24070 r``
#: (equivalently ``(3/4 pi)^(1/3) h``).  Arithmetic identity:
#: ``4/3 pi r_m^3 = h^3``, so ``rho * 4/3 pi r_m^3 = rho * h^3`` to round-off.
MASS_RADIUS_FACTOR = (6.0 / math.pi) ** (1.0 / 3.0)

#: Accepted ``mass_mapping`` spellings -> internal ``mass_mode``.  The two
#: spellings are the ones already published by the two merged audit notes
#: (``docs/dem3d_fracture_gen_audit.md``, ``docs/fracture_gen_mass_footprint_audit.md``);
#: they name the same two mappings, so both are accepted and normalised.
MASS_MAPPING_ALIASES = {
    "equal_volume_spheres": "sphere",
    "equal_volume_sphere": "sphere",
    "cell_volume": "cell",
    "lattice_cell": "cell",
}

#: Canonical reporting name per internal mass mode.
MASS_MODE_TO_MAPPING = {"sphere": "equal_volume_spheres", "cell": "cell_volume"}

#: Supported face-extent models for bond severing.
VALID_FACE_EXTENTS = ("plane", "disc")


@dataclass
class FailureFace:
    """A single FEM failure face."""
    id: int
    area: float
    normal: torch.Tensor  # shape [3]
    centroid: torch.Tensor  # shape [3]
    G: float  # fracture energy release rate (J/m²)
    released_energy: float  # total energy released (J)
    # Optional producer-supplied face extent (radius, m).  ``None`` keeps the
    # frozen FEM->DEM JSON contract unchanged: the footprint radius is then
    # derived from ``area`` as sqrt(area / pi).
    extent_radius: float | None = None


@dataclass
class FractureGenConfig:
    """Configuration for fracture generation."""
    density: float = 917.0  # ice density (kg/m³)
    particle_radius: float = 0.1  # particle radius (m)
    bond_stiffness: float = 2000.0  # bond stiffness (N/m)
    domain_bounds: tuple[float, float, float, float, float, float] | None = None
    # (xmin, xmax, ymin, ymax, zmin, zmax)
    mass_mode: str = "cell"
    # "cell"   -> m = density * h^3, volume consistent (default, fulfils the
    #             frozen "equal-volume spheres" intent)
    # "sphere" -> m = density * (4/3) pi r^3, legacy pre-audit mapping that
    #             under-counts mass by 1 - pi/6 = 47.64 %
    # Public alias of ``mass_mode`` using the names of the published audit
    # notes: "cell_volume" == "lattice_cell" == mode "cell",
    # "equal_volume_spheres" == "equal_volume_sphere" == mode "sphere".
    mass_mapping: str | None = None
    # Bond-severing model: "plane" (frozen contract, the failure face is
    # treated as an infinite plane) or "disc" (only bonds whose intersection
    # point falls inside the face footprint are severed).
    face_extent: str = "plane"
    # Alias for face_extent="disc" kept for the published audit scripts.
    footprint_aware: bool | None = None
    # A bond with exactly one endpoint within this distance of the face plane
    # counts as severed; 0.0 is the frozen strict sign test, which cannot see
    # a cut surface that coincides with a particle layer.
    on_plane_tolerance: float = 0.0

    def __post_init__(self):
        if self.density <= 0:
            raise ValueError("density must be positive")
        if self.particle_radius <= 0:
            raise ValueError("particle_radius must be positive")
        if self.bond_stiffness <= 0:
            raise ValueError("bond_stiffness must be positive")
        if self.mass_mapping is not None:
            if self.mass_mapping not in MASS_MAPPING_ALIASES:
                raise ValueError(
                    "mass_mapping must be one of "
                    f"{tuple(MASS_MAPPING_ALIASES)}, got {self.mass_mapping!r}"
                )
            self.mass_mode = MASS_MAPPING_ALIASES[self.mass_mapping]
        if self.mass_mode not in VALID_MASS_MODES:
            raise ValueError(
                f"mass_mode must be one of {VALID_MASS_MODES}, got {self.mass_mode!r}"
            )
        self.mass_mapping = MASS_MODE_TO_MAPPING[self.mass_mode]
        if self.footprint_aware:
            self.face_extent = "disc"
        if self.face_extent not in VALID_FACE_EXTENTS:
            raise ValueError(
                f"face_extent must be one of {VALID_FACE_EXTENTS}, "
                f"got {self.face_extent!r}"
            )
        if self.on_plane_tolerance < 0:
            raise ValueError("on_plane_tolerance must be non-negative")


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
        mass_mode: str = "cell",
        mass_mapping: str | None = None,
        face_extent: str = "plane",
        on_plane_tolerance: float = 0.0,
    ):
        if mass_mapping is not None:
            if mass_mapping not in MASS_MAPPING_ALIASES:
                raise ValueError(
                    "mass_mapping must be one of "
                    f"{tuple(MASS_MAPPING_ALIASES)}, got {mass_mapping!r}"
                )
            mass_mode = MASS_MAPPING_ALIASES[mass_mapping]
        if mass_mode not in VALID_MASS_MODES:
            raise ValueError(
                f"mass_mode must be one of {VALID_MASS_MODES}, got {mass_mode!r}"
            )
        if face_extent not in VALID_FACE_EXTENTS:
            raise ValueError(
                f"face_extent must be one of {VALID_FACE_EXTENTS}, got {face_extent!r}"
            )
        self.mass_mode = mass_mode
        self.mass_mapping = MASS_MODE_TO_MAPPING[mass_mode]
        self.face_extent = face_extent
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
    def particle_mass(self) -> float:
        """Mass of a single particle for the configured mass mapping."""
        return _particle_mass(self.density, self.particle_radius, self.mass_mode)

    @property
    def mass_radius(self) -> float:
        """Inertial radius consistent with the configured mass mapping.

        ``sphere`` mode: the geometric radius (contact radius == mass radius).
        ``cell`` mode: ``(6/pi)^(1/3) r``, i.e. the radius of the sphere whose
        volume equals the lattice cell ``h^3``.  The contact radius stays
        ``particle_radius`` in both cases.
        """
        if self.mass_mode == "cell":
            return MASS_RADIUS_FACTOR * self.particle_radius
        return self.particle_radius

    @property
    def volume_fraction(self) -> float:
        """DEM particle volume / volume of the lattice cell each one owns."""
        return self.particle_mass / (self.density * self.particle_cell_volume)

    @property
    def particle_cell_volume(self) -> float:
        """Volume of the lattice cell owned by one particle, h^3."""
        return particle_cell_volume(self.particle_radius)

    @property
    def represented_volume(self) -> float:
        """Ice volume represented by the lattice, N * h^3."""
        return self.num_particles * self.particle_cell_volume
    
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
    
    Bonds connect nearest neighbors (distance = 2*particle_radius).
    """
    n_particles = len(positions)
    spacing = 2.0 * particle_radius
    limit = spacing * 1.01  # small tolerance for numerical errors

    # Find all pairs within bonding distance.  Blocked Euclidean distance
    # instead of the former Python double loop: O(N^2) distance reductions of
    # one scalar each were the dominant cost of every generator call (1000
    # particles took tens of seconds), which is what made the published audit
    # contract impractical to execute.  The pair set and its ordering
    # (i < j, lexicographic) are identical to the brute-force reference.
    block = 512
    pair_blocks = []
    for start in range(0, n_particles, block):
        stop = min(start + block, n_particles)
        diff = positions[start:stop].unsqueeze(1) - positions.unsqueeze(0)
        dist = torch.linalg.norm(diff, dim=2)
        rows = torch.arange(start, stop, device=positions.device).unsqueeze(1)
        cols = torch.arange(n_particles, device=positions.device).unsqueeze(0)
        keep = (dist <= limit) & (cols > rows)
        local_i, local_j = keep.nonzero(as_tuple=True)
        if local_i.numel():
            pair_blocks.append(
                torch.stack([local_i + start, local_j], dim=1).to(torch.long)
            )

    if not pair_blocks:
        return torch.empty((0, 2), dtype=torch.long, device=positions.device), \
               torch.empty(0, dtype=positions.dtype, device=positions.device)

    bond_indices = torch.cat(pair_blocks, dim=0)
    rest_lengths = torch.linalg.norm(
        positions[bond_indices[:, 1]] - positions[bond_indices[:, 0]],
        dim=1
    )

    return bond_indices, rest_lengths


def face_footprint_radius(face: FailureFace) -> float:
    """Radius of the circular footprint a failure face is modelled as.

    Uses the producer-supplied ``extent_radius`` when present, otherwise the
    area-equivalent disc ``sqrt(area / pi)``.  The FEM->DEM contract carries
    only area and centroid, so a disc is an assumption (see
    ``docs/fracture_gen_mass_footprint_audit.md`` §5).
    """
    if face.extent_radius is not None:
        if face.extent_radius <= 0:
            raise ValueError(
                f"extent_radius must be positive, got {face.extent_radius!r}"
            )
        return float(face.extent_radius)
    if face.area <= 0:
        raise ValueError(f"face area must be positive, got {face.area!r}")
    return math.sqrt(face.area / math.pi)


def identify_bonds_crossing_face(
    bond_indices: torch.Tensor,
    positions: torch.Tensor,
    face: FailureFace,
    face_extent: str = "plane",
    on_plane_tolerance: float = 0.0,
    footprint_aware: bool | None = None,
) -> torch.Tensor:
    """Identify bonds that cross a failure face.
    
    A bond crosses a face if the two endpoints are on opposite sides of the
    face plane (defined by normal and centroid).
    """
    if len(bond_indices) == 0:
        return torch.empty(0, dtype=torch.bool, device=positions.device)
    
    i_idx, j_idx = bond_indices.T
    pos_i = positions[i_idx]
    pos_j = positions[j_idx]
    
    # Move face data to same device as positions
    centroid = face.centroid.to(positions.device)
    normal = face.normal.to(positions.device)
    
    if float(torch.linalg.norm(normal)) <= 0.0:
        raise ValueError("face normal must be non-zero")
    if footprint_aware:
        face_extent = "disc"
    if face_extent not in VALID_FACE_EXTENTS:
        raise ValueError(
            f"face_extent must be one of {VALID_FACE_EXTENTS}, got {face_extent!r}"
        )
    tolerance = float(on_plane_tolerance)
    if tolerance < 0:
        raise ValueError("on_plane_tolerance must be non-negative")

    # Signed distance from face plane: normal · (pos - centroid)
    dist_i = torch.sum((pos_i - centroid) * normal, dim=1)
    dist_j = torch.sum((pos_j - centroid) * normal, dim=1)

    # Bond crosses face if signs are opposite (one positive, one negative)
    crosses = (dist_i * dist_j) < 0

    if tolerance > 0.0:
        # A bond with exactly one endpoint on the cut surface is separated by
        # it; a bond lying *inside* the surface (both endpoints on it) is not.
        near_i = dist_i.abs() <= tolerance
        near_j = dist_j.abs() <= tolerance
        crosses = crosses | (near_i ^ near_j)

    if face_extent == "disc":
        radius = face_footprint_radius(face)
        span = dist_j - dist_i
        safe = torch.where(
            span.abs() > 0.0, span, torch.ones_like(span)
        )
        # Parameter of the plane crossing along the bond; 0/1 when the crossing
        # is exactly at an endpoint (the tolerance case).
        fraction = torch.where(
            span.abs() > 0.0, -dist_i / safe, torch.zeros_like(span)
        )
        crossing = pos_i + fraction.unsqueeze(1) * (pos_j - pos_i)
        offset = crossing - centroid
        offset = offset - torch.sum(offset * normal, dim=1, keepdim=True) * normal
        crosses = crosses & (torch.linalg.norm(offset, dim=1) <= radius)

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
        mass_mode=config.mass_mode,
        face_extent=config.face_extent,
        on_plane_tolerance=config.on_plane_tolerance,
    )


class FractureLedger:
    """Accounting ledger for mass and energy conservation.
    
    Tracks:
    - Mass conservation: FEM mass vs DEM particle mass sum
    - Energy accounting: ∑released_energy vs broken_bonds × unit_bond_energy
    """
    
    def __init__(self, fracture_field: FractureField, failure_faces: list[FailureFace]):
        self.field = fracture_field
        self.faces = failure_faces
    
    def mass_conservation_error(self, fem_mass: float) -> float:
        """Compute mass conservation error: |DEM_mass - FEM_mass|."""
        dem_mass = self.field.total_mass
        return abs(dem_mass - fem_mass)

    def mass_mapping_report(
        self,
        fem_mass: float | None = None,
        tolerance: float = 0.01,
        domain_bounds: tuple[float, float, float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        """Audit the DEM mass against the FEM ice mass or rho * V_domain.

        Uses ``domain_bounds`` when given, otherwise the bounding box of the
        realised particle positions.  See
        ``tensordem.fracture_mass_mapping.mass_mapping_report``.
        """
        if domain_bounds is None:
            # Centres span only (N-1) h per axis; each particle also owns a
            # half-cell of half-width r on both sides, so expanding the centre
            # bounding box by r recovers the lattice extent N h.
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
            mass_mode=self.field.mass_mode,
            fem_mass=fem_mass,
            tolerance=tolerance,
            num_particles=self.field.num_particles,
        )
    
    def lattice_volume(self) -> float:
        """Volume physically covered by the realised particle lattice, N h^3."""
        return self.field.represented_volume

    def mass_accounting(
        self, domain_volume: float | None = None
    ) -> dict[str, Any]:
        """Compare DEM mass with the continuum mass ``rho * V`` it must carry.

        ``domain_volume`` defaults to the lattice-covered volume, which makes
        the number a pure material-mapping residual (the lattice quantisation
        error is reported separately as ``lattice_volume_covered`` vs
        ``fem_continuum_mass``).

        Returns signed ``relative_error`` (negative = DEM mass too low), its
        magnitude as ``absolute_relative_error``, and ``within_15pct`` for the
        platform acceptance bar.
        """
        covered = self.lattice_volume()
        if domain_volume is None:
            domain_volume = covered
        domain_volume = float(domain_volume)
        if not math.isfinite(domain_volume) or domain_volume <= 0.0:
            raise ValueError("domain_volume must be a positive finite volume")
        dem_mass = self.field.total_mass
        continuum_mass = self.field.density * domain_volume
        signed = (dem_mass - continuum_mass) / continuum_mass
        return {
            "mass_mapping": self.field.mass_mapping,
            "mass_mode": self.field.mass_mode,
            "mass_radius": self.field.mass_radius,
            "num_particles": self.field.num_particles,
            "domain_volume": domain_volume,
            "lattice_volume_covered": covered,
            "volume_fraction": self.field.volume_fraction,
            "dem_mass": dem_mass,
            "fem_continuum_mass": continuum_mass,
            # kept under the key used by benchmarks/fracture_gen_mass_footprint_audit.py
            "continuum_mass": continuum_mass,
            "relative_error": signed,
            "absolute_relative_error": abs(signed),
            "within_15pct": abs(signed) <= 0.15,
        }

    def energy_accounting(self) -> dict[str, Any]:
        """Account for fracture energy against falsifiable lattice analytics.

        Returns dict with:
        - total_released_energy: sum of released_energy from all faces
        - num_broken_bonds: number of broken bonds in DEM field
        - unit_bond_energy: total_released_energy / num_broken_bonds
        - analytic_bond_count: sum of A_i / h^2 (bonds an area-consistent cut
          through the lattice should sever)
        - analytic_unit_bond_energy: sum(G_i A_i) / analytic_bond_count, i.e.
          G_i h^2 for a single face
        - bond_count_relative_error: |broken - analytic| / analytic
        - unit_bond_energy_relative_error: |unit - analytic| / analytic
        - input_consistency_error: sum |released_energy - G * area|
        - face_extent: the severing model the field was generated with
        """
        total_released = sum(face.released_energy for face in self.faces)
        num_broken = self.field.num_broken_bonds

        if num_broken > 0:
            unit_bond_energy = total_released / num_broken
        else:
            unit_bond_energy = 0.0

        spacing = 2.0 * self.field.particle_radius
        spacing_sq = spacing * spacing
        analytic_bonds = sum(
            face.area / spacing_sq for face in self.faces if face.area > 0
        )
        analytic_unit = (
            sum(face.G * face.area for face in self.faces) / analytic_bonds
            if analytic_bonds > 0
            else 0.0
        )
        bond_count_rel = (
            abs(num_broken - analytic_bonds) / analytic_bonds
            if analytic_bonds > 0
            else 0.0
        )
        unit_rel = (
            abs(unit_bond_energy - analytic_unit) / analytic_unit
            if analytic_unit > 0
            else 0.0
        )
        input_consistency = sum(
            abs(face.released_energy - face.G * face.area) for face in self.faces
        )

        return {
            'total_released_energy': total_released,
            'num_broken_bonds': num_broken,
            'unit_bond_energy': unit_bond_energy,
            'analytic_bond_count': analytic_bonds,
            'analytic_unit_bond_energy': analytic_unit,
            'bond_count_relative_error': bond_count_rel,
            'unit_bond_energy_relative_error': unit_rel,
            'input_consistency_error': input_consistency,
            'face_extent': self.field.face_extent,
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
                on_plane_tolerance=self.field.on_plane_tolerance,
            )
            bonds_should_break |= crosses
        
        bonds_actually_broken = ~self.field.alive
        
        return {
            'bonds_crossing_faces': bonds_should_break.sum().item(),
            'bonds_actually_broken': bonds_actually_broken.sum().item(),
            'all_face_bonds_broken': torch.all(bonds_should_break == bonds_actually_broken).item()
        }
