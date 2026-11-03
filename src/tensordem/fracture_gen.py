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

# Audit vocabulary <-> internal mass modes.  ``docs/dem3d_fracture_gen_audit.md``
# names the two mappings by what they claim to conserve; ``mass_mode`` uses the
# shorter engineering names.  Both names always denote the same two laws, so a
# single alias table keeps the audit and the engine from drifting apart.
MASS_MAPPING_TO_MODE: dict[str, str] = {
    "equal_volume_spheres": "sphere",
    "cell_volume": "cell",
}
MODE_TO_MASS_MAPPING: dict[str, str] = {v: k for k, v in MASS_MAPPING_TO_MODE.items()}
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
    extent_radius: float | None = None
    # optional in-plane radius (m) of the face footprint; ``None`` means the
    # equal-area disc radius sqrt(area / pi).  Only used by face_extent="disc".


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
    mass_mapping: str | None = None
    # audit-facing alias of ``mass_mode``: "equal_volume_spheres" (= "sphere")
    # or "cell_volume" (= "cell").  ``None`` (default) defers to ``mass_mode``
    # so the fused engine contract keeps its single source of truth; an
    # explicit value overrides ``mass_mode``.
    face_extent: str = "plane"
    # "plane" -> a failure face cuts the whole infinite plane (frozen SI-2
    #            behaviour: the declared area is ignored)
    # "disc"  -> only bonds whose crossing point lies inside the equal-area
    #            footprint disc sqrt(area / pi) are severed (area-consistent)

    def __post_init__(self):
        if self.density <= 0:
            raise ValueError("density must be positive")
        if self.particle_radius <= 0:
            raise ValueError("particle_radius must be positive")
        if self.bond_stiffness <= 0:
            raise ValueError("bond_stiffness must be positive")
        if self.mass_mode not in VALID_MASS_MODES:
            raise ValueError(
                f"mass_mode must be one of {VALID_MASS_MODES}, got {self.mass_mode!r}"
            )
        if self.mass_mapping is not None and self.mass_mapping not in MASS_MAPPING_TO_MODE:
            raise ValueError(
                "mass_mapping must be one of "
                f"{tuple(MASS_MAPPING_TO_MODE)} or None, got {self.mass_mapping!r}"
            )
        if self.face_extent not in VALID_FACE_EXTENTS:
            raise ValueError(
                f"face_extent must be one of {VALID_FACE_EXTENTS}, got {self.face_extent!r}"
            )

    @property
    def effective_mass_mode(self) -> str:
        """Internal mass mode actually used (``mass_mapping`` wins if set)."""
        if self.mass_mapping is not None:
            return MASS_MAPPING_TO_MODE[self.mass_mapping]
        return self.mass_mode


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
        domain_bounds: tuple[float, float, float, float, float, float] | None = None,
        face_extent: str = "plane",
    ):
        if mass_mode not in VALID_MASS_MODES:
            raise ValueError(
                f"mass_mode must be one of {VALID_MASS_MODES}, got {mass_mode!r}"
            )
        if face_extent not in VALID_FACE_EXTENTS:
            raise ValueError(
                f"face_extent must be one of {VALID_FACE_EXTENTS}, got {face_extent!r}"
            )
        self.mass_mode = mass_mode
        self.domain_bounds = domain_bounds
        self.face_extent = face_extent
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
    def particle_cell_volume(self) -> float:
        """Volume of the lattice cell owned by one particle, h^3."""
        return particle_cell_volume(self.particle_radius)

    @property
    def mass_mapping(self) -> str:
        """Audit name of the active mass mapping (see ``MASS_MAPPING_TO_MODE``)."""
        return MODE_TO_MASS_MAPPING[self.mass_mode]

    @property
    def volume_fraction(self) -> float:
        """Volume fraction filled by the mapped particle volume.

        m_p / (density * h^3): ``pi/6 = 0.523599`` for the inscribed-sphere
        mapping, exactly ``1.0`` for the volume-consistent cell mapping.
        """
        return self.particle_mass / (self.density * self.particle_cell_volume)

    @property
    def domain_volume(self) -> float | None:
        """Volume of the configured generation domain (m^3), if known."""
        if self.domain_bounds is None:
            return None
        xmin, xmax, ymin, ymax, zmin, zmax = self.domain_bounds
        return (xmax - xmin) * (ymax - ymin) * (zmax - zmin)

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
    cutoff = spacing * 1.01  # small tolerance for numerical errors

    # Vectorised nearest-neighbour search: chunk the leading index so the
    # (chunk x n x 3) distance block stays bounded (~64 MB of float64), and
    # emit pairs in strictly lexicographic (i, j) order, i < j, which is the
    # order the equivalent double loop produces.  The previous per-pair loop
    # was O(n^2) Python and dominated every 10^3-particle audit run.
    device = positions.device
    chunk = max(1, int(6.0e6 / max(n_particles, 1)))
    out_i: list[torch.Tensor] = []
    out_j: list[torch.Tensor] = []
    for start in range(0, n_particles, chunk):
        stop = min(start + chunk, n_particles)
        block = positions[start:stop]
        dist = torch.linalg.norm(block[:, None, :] - positions[None, :, :], dim=2)
        # j > i within the chunk, and j > every index of the chunk
        cols = torch.arange(n_particles, device=device)
        rows = torch.arange(start, stop, device=device)
        mask = dist <= cutoff
        mask &= cols[None, :] > rows[:, None]
        ii, jj = mask.nonzero(as_tuple=True)
        if ii.numel():
            out_i.append(ii + start)
            out_j.append(jj)

    if not out_i:
        return torch.empty((0, 2), dtype=torch.long, device=device), \
               torch.empty(0, dtype=positions.dtype, device=device)

    i_all = torch.cat(out_i)
    j_all = torch.cat(out_j)
    bond_indices = torch.stack([i_all, j_all], dim=1).to(dtype=torch.long)
    rest_lengths = torch.linalg.norm(
        positions[bond_indices[:, 1]] - positions[bond_indices[:, 0]],
        dim=1
    )

    return bond_indices, rest_lengths


def face_footprint_radius(face: FailureFace) -> float:
    """In-plane radius (m) of the face footprint used by ``face_extent="disc"``.

    ``face.extent_radius`` when supplied, otherwise the equal-area disc radius
    ``sqrt(area / pi)``.  Non-positive radii are rejected: a zero-radius face
    could never break a bond, which would silently disable the face.
    """
    radius = face.extent_radius
    if radius is None:
        if face.area <= 0:
            raise ValueError(f"face {face.id}: area must be positive, got {face.area}")
        radius = math.sqrt(face.area / math.pi)
    radius = float(radius)
    if radius <= 0:
        raise ValueError(f"face {face.id}: footprint radius must be positive, got {radius}")
    return radius


def identify_bonds_crossing_face(
    bond_indices: torch.Tensor,
    positions: torch.Tensor,
    face: FailureFace,
    face_extent: str = "plane",
) -> torch.Tensor:
    """Identify bonds that cross a failure face.

    The plane test is the frozen one: two endpoints on opposite sides of the
    face plane (defined by normal and centroid).  ``face_extent="disc"`` adds
    the area-consistent footprint test: the plane crossing point must also lie
    inside the footprint disc of ``face_footprint_radius`` around the centroid.
    """
    if face_extent not in VALID_FACE_EXTENTS:
        raise ValueError(
            f"face_extent must be one of {VALID_FACE_EXTENTS}, got {face_extent!r}"
        )
    if len(bond_indices) == 0:
        return torch.empty(0, dtype=torch.bool, device=positions.device)
    
    i_idx, j_idx = bond_indices.T
    pos_i = positions[i_idx]
    pos_j = positions[j_idx]
    
    # Move face data to same device as positions
    centroid = face.centroid.to(positions.device)
    normal = face.normal.to(positions.device)
    
    norm = torch.linalg.norm(normal)
    if not float(norm) > 0.0:
        raise ValueError(f"face {face.id}: degenerate face normal {normal.tolist()}")
    normal = normal / norm

    # Signed distance from face plane: normal · (pos - centroid)
    dist_i = torch.sum((pos_i - centroid) * normal, dim=1)
    dist_j = torch.sum((pos_j - centroid) * normal, dim=1)
    
    # Bond crosses face if signs are opposite (one positive, one negative)
    crosses = (dist_i * dist_j) < 0

    if face_extent == "disc" and bool(crosses.any()):
        # Linear interpolation to the plane crossing point, then the in-plane
        # offset from the face centroid (normal component removed).
        t = dist_i[crosses] / (dist_i[crosses] - dist_j[crosses])
        crossing = pos_i[crosses] + t.unsqueeze(1) * (pos_j[crosses] - pos_i[crosses])
        offset = crossing - centroid
        offset = offset - (offset * normal).sum(dim=1, keepdim=True) * normal
        inside = torch.linalg.norm(offset, dim=1) <= face_footprint_radius(face)
        crosses = crosses.clone()
        crosses[crosses.clone()] = inside

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
            bond_indices, positions, face, face_extent=config.face_extent
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
        mass_mode=config.effective_mass_mode,
        domain_bounds=config.domain_bounds,
        face_extent=config.face_extent,
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
    
    def mass_accounting(self, domain_volume: float | None = None) -> dict[str, Any]:
        """Compare the DEM mass with the FEM continuum mass ``rho * V`` (c1).

        ``domain_volume`` defaults to the configured generation domain.  Keys:

        - ``mass_mapping``: audit name of the active mapping
        - ``num_particles`` / ``particle_mass`` / ``dem_mass``
        - ``lattice_volume_covered``: ``N * h^3`` actually discretised
        - ``volume_fraction``: mapped particle volume over its lattice cell
        - ``fem_continuum_mass``: ``density * domain_volume``
        - ``signed_relative_error`` / ``relative_error`` (``|.|``)

        The frozen criterion compared the DEM mass with its own sphere formula;
        this one compares it with the continuum mass the FEM face belongs to, so
        the ``pi/6 - 1 = -47.64 %`` deficit of ``equal_volume_spheres`` is
        visible instead of cancelled.
        """
        if domain_volume is None:
            domain_volume = self.field.domain_volume
        if domain_volume is None:
            raise ValueError(
                "domain_volume is required: the field was built without domain_bounds"
            )
        domain_volume = float(domain_volume)
        if not domain_volume > 0.0:
            raise ValueError(f"domain_volume must be positive, got {domain_volume}")

        fem_continuum_mass = self.field.density * domain_volume
        dem_mass = self.field.total_mass
        signed = (dem_mass - fem_continuum_mass) / fem_continuum_mass
        return {
            "mass_mapping": self.field.mass_mapping,
            "num_particles": self.field.num_particles,
            "particle_mass": self.field.particle_mass,
            "dem_mass": dem_mass,
            "fem_continuum_mass": fem_continuum_mass,
            "domain_volume": domain_volume,
            "lattice_volume_covered": self.field.represented_volume,
            "volume_fraction": self.field.volume_fraction,
            "signed_relative_error": signed,
            "relative_error": abs(signed),
        }

    def energy_accounting(self) -> dict[str, float]:
        """Account for fracture energy, against lattice analytics.

        The frozen c2 criterion divided the released energy by the broken-bond
        count and multiplied it back (an identity).  This report keeps that
        ratio but also states the *analytically expected* values, so the
        comparison can fail:

        - ``total_released_energy``: sum of ``released_energy`` over faces
        - ``num_broken_bonds``: broken bonds in the DEM field
        - ``unit_bond_energy``: ``total_released_energy / num_broken_bonds``
        - ``analytic_bond_count``: ``sum(area) / h^2``, the number of bonds a
          simple-cubic lattice of spacing ``h`` cuts on a face of that area
        - ``analytic_unit_bond_energy``: ``sum(G * area) / analytic_bond_count``
          = ``G * h^2`` for a single consistent face
        - ``unit_bond_energy_relative_error`` / ``bond_count_relative_error``:
          relative departures from those analytics (``nan`` when undefined)
        - ``input_consistency_error``: ``|sum(released) - sum(G * area)|``
        - ``face_extent``: which cut criterion produced the count
        """
        total_released = sum(face.released_energy for face in self.faces)
        num_broken = self.field.num_broken_bonds

        if num_broken > 0:
            unit_bond_energy = total_released / num_broken
        else:
            unit_bond_energy = 0.0

        spacing_sq = (2.0 * self.field.particle_radius) ** 2
        face_area = sum(face.area for face in self.faces)
        expected_released = sum(face.G * face.area for face in self.faces)
        analytic_bond_count = face_area / spacing_sq if spacing_sq > 0 else 0.0
        if analytic_bond_count > 0:
            analytic_unit = expected_released / analytic_bond_count
            bond_count_relative_error = abs(num_broken - analytic_bond_count) / analytic_bond_count
        else:
            analytic_unit = 0.0
            bond_count_relative_error = float("nan")
        if num_broken > 0 and analytic_unit > 0:
            unit_relative_error = abs(unit_bond_energy - analytic_unit) / analytic_unit
        else:
            unit_relative_error = float("nan")

        return {
            'total_released_energy': total_released,
            'num_broken_bonds': num_broken,
            'unit_bond_energy': unit_bond_energy,
            'analytic_bond_count': analytic_bond_count,
            'analytic_unit_bond_energy': analytic_unit,
            'unit_bond_energy_relative_error': unit_relative_error,
            'bond_count_relative_error': bond_count_relative_error,
            'input_consistency_error': abs(total_released - expected_released),
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
            )
            bonds_should_break |= crosses
        
        bonds_actually_broken = ~self.field.alive
        
        return {
            'bonds_crossing_faces': bonds_should_break.sum().item(),
            'bonds_actually_broken': bonds_actually_broken.sum().item(),
            'all_face_bonds_broken': torch.all(bonds_should_break == bonds_actually_broken).item()
        }
