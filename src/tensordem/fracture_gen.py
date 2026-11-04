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

#: Mass radius of a lattice particle relative to its contact radius, for
#: spacing ``h = 2 r``:  ``r_eq / r = (3 h^3 / (4 pi))^{1/3} / r = 2 (3/4pi)^{1/3}``
#: ``= 1.240701``.  A sphere of radius ``MASS_RADIUS_FACTOR * r`` at the field
#: density carries exactly the mass of the cubic cell ``h^3`` the particle owns,
#: which is what ``mass_mode="cell"`` assigns (see
#: ``docs/dem3d_fracture_mass_mapping_audit.md``).
MASS_RADIUS_FACTOR = 2.0 * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)

#: Accepted ``mass_mapping`` spellings -> canonical ``mass_mode``.
MASS_MAPPING_ALIASES = {
    "cell": "cell",
    "lattice_cell": "cell",
    "lattice-cell": "cell",
    "volume_cell": "cell",
    # The audit layer (``benchmarks/fracture_gen_audit.py``,
    # ``scripts/plot_fracture_gen_audit.py``, ``tests/test_fracture_gen_audit.py``)
    # spells the same two mappings "cell_volume" / "equal_volume_spheres"; accept
    # them so one mapping has one documented behaviour, not two spellings that
    # raise.
    "cell_volume": "cell",
    "sphere": "sphere",
    "equal_volume_sphere": "sphere",
    "equal_volume_spheres": "sphere",
    "equal-volume-sphere": "sphere",
    "frozen": "sphere",
}

#: Geometric tolerance (m) applied to the in-plane footprint radius test.
_FOOTPRINT_TOLERANCE = 1e-12

#: ``face_extent`` spellings -> ``footprint_aware``.  ``"plane"`` is the frozen
#: SI-2 criterion (the face is treated as an infinite plane, so the bond count is
#: ``A_plane / h^2`` regardless of the declared face area); ``"disc"`` breaks only
#: bonds whose crossing point projects inside the equal-area footprint disc of
#: radius ``sqrt(area/pi)``, so the count tracks ``pi R^2 / h^2``.
FACE_EXTENTS = {"plane": False, "disc": True}


def normalise_mass_mapping(value: str, name: str = "mass_mapping") -> str:
    """Canonicalise a mass-mapping spelling onto ``VALID_MASS_MODES``.

    Both the descriptive spellings used by the validation-debt audit
    (``"lattice_cell"``, ``"equal_volume_sphere"``) and the canonical
    ``mass_mode`` values (``"cell"``, ``"sphere"``) are accepted so that the
    published audit tables stay reproducible through a single keyword.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{name}/mass_mode must be a string, got {type(value).__name__}"
        )
    key = value.strip().lower()
    if key not in MASS_MAPPING_ALIASES:
        raise ValueError(
            f"{name}/mass_mode must be one of {sorted(MASS_MAPPING_ALIASES)} "
            f"(canonical mass_mode values {VALID_MASS_MODES}), got {value!r}"
        )
    return MASS_MAPPING_ALIASES[key]


def normalise_face_extent(value: str | None, footprint_aware: bool = False) -> bool:
    """Canonicalise ``face_extent`` ("plane" | "disc") onto ``footprint_aware``."""
    if value is None:
        return bool(footprint_aware)
    if not isinstance(value, str) or value not in FACE_EXTENTS:
        raise ValueError(
            f"face_extent must be one of {sorted(FACE_EXTENTS)}, got {value!r}"
        )
    aware = FACE_EXTENTS[value]
    if footprint_aware and not aware:
        raise ValueError(
            "face_extent='plane' contradicts footprint_aware=True: the frozen "
            "infinite-plane criterion is not footprint aware"
        )
    return aware


@dataclass
class FailureFace:
    """A single FEM failure face."""
    id: int
    area: float
    normal: torch.Tensor  # shape [3]
    centroid: torch.Tensor  # shape [3]
    G: float  # fracture energy release rate (J/m²)
    released_energy: float  # total energy released (J)
    #: In-plane extent of the face (m).  ``None`` means "derive from
    #: ``area`` as ``sqrt(area/pi)``", i.e. an equal-area disc.
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
    mass_mapping: str | None = None
    # Descriptive alias of ``mass_mode`` (accepts "lattice_cell",
    # "equal_volume_sphere", ...).  When given it wins over ``mass_mode``.
    footprint_aware: bool = False
    # True  -> break only bonds whose crossing point projects inside the face
    #          footprint disc (radius ``face_footprint_radius(face)``)
    # False -> frozen SI-2 criterion: every bond crossing the *infinite* plane
    face_extent: str | None = None
    # Descriptive alias of ``footprint_aware``: ``"plane"`` == False,
    # ``"disc"`` == True.  When given it wins, but a ``face_extent="plane"``
    # combined with an explicit ``footprint_aware=True`` is a contradiction and
    # is rejected instead of silently resolved.
    on_plane_tolerance: float = 0.0
    # > 0 -> endpoints within this distance of the face plane count as lying on
    #        it, so a plane coincident with a particle layer severs the two
    #        bonds per column that touch it instead of nothing.

    def __post_init__(self):
        if self.density <= 0:
            raise ValueError("density must be positive")
        if self.particle_radius <= 0:
            raise ValueError("particle_radius must be positive")
        if self.bond_stiffness <= 0:
            raise ValueError("bond_stiffness must be positive")
        if self.mass_mapping is not None:
            self.mass_mode = normalise_mass_mapping(self.mass_mapping, "mass_mapping")
        else:
            self.mass_mode = normalise_mass_mapping(self.mass_mode, "mass_mode")
        if not math.isfinite(self.on_plane_tolerance) or self.on_plane_tolerance < 0.0:
            raise ValueError("on_plane_tolerance must be non-negative and finite")
        self.footprint_aware = normalise_face_extent(self.face_extent, self.footprint_aware)


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
        footprint_aware: bool = False,
        on_plane_tolerance: float = 0.0,
        face_extent: str | None = None,
        mass_mapping: str | None = None,
    ):
        self.mass_mode = normalise_mass_mapping(mass_mode, "mass_mode")
        # Canonical mode plus the spelling the caller actually asked for, so an
        # audit can quote "cell_volume" while the physics uses "cell".
        if mass_mapping is None:
            self.mass_mapping = self.mass_mode
        else:
            normalise_mass_mapping(mass_mapping, "mass_mapping")  # validate only
            self.mass_mapping = str(mass_mapping)
        if not math.isfinite(on_plane_tolerance) or on_plane_tolerance < 0.0:
            raise ValueError("on_plane_tolerance must be non-negative and finite")
        self.footprint_aware = normalise_face_extent(face_extent, footprint_aware)
        self.on_plane_tolerance = on_plane_tolerance
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
        """Radius of the sphere that carries ``particle_mass`` at ``density``.

        ``mass_mode="cell"``   -> ``MASS_RADIUS_FACTOR * r`` (volume-equivalent
        radius ``r_eq = (3 h^3 / 4 pi)^{1/3}``, larger than the contact radius);
        ``mass_mode="sphere"`` -> ``r`` (the mass *is* the inscribed sphere's).
        """
        if self.mass_mode == "cell":
            return MASS_RADIUS_FACTOR * self.particle_radius
        return self.particle_radius

    @property
    def volume_fraction(self) -> float:
        """Volume fraction of the mass-carrying sphere in its lattice cell.

        ``particle_mass / (density * h^3)``: exactly 1.0 for the volume
        consistent ``"cell"`` mapping, ``pi/6`` for the inscribed-sphere
        ``"sphere"`` mapping.
        """
        cell_volume = self.particle_cell_volume
        if cell_volume <= 0.0:
            raise ValueError("lattice cell volume must be positive")
        return self.particle_mass / (self.density * cell_volume)

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
    def face_extent(self) -> str:
        """Canonical ``face_extent`` spelling of this field ("plane" | "disc")."""
        return "disc" if self.footprint_aware else "plane"

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

    # Find all pairs within bonding distance.  Done block-wise with ``cdist``
    # instead of a per-pair Python loop: the loop issued one ``torch.linalg.norm``
    # per pair (22 s for the 1000-particle audit lattice), which dominated every
    # caller.  The block sweep visits (i, j) in ascending row-major order, so the
    # pair order is identical to the double loop it replaces.
    block = 512
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    for start in range(0, n_particles, block):
        stop = min(start + block, n_particles)
        distances = torch.cdist(positions[start:stop], positions)
        block_i, block_j = torch.nonzero(distances <= cutoff, as_tuple=True)
        block_i = block_i + start
        keep = block_j > block_i
        rows.append(block_i[keep])
        cols.append(block_j[keep])

    if rows:
        bond_indices = torch.stack(
            (torch.cat(rows), torch.cat(cols)), dim=1
        ).to(dtype=torch.long, device=positions.device)
    else:
        bond_indices = torch.empty((0, 2), dtype=torch.long, device=positions.device)

    if len(bond_indices) == 0:
        return bond_indices, torch.empty(0, dtype=positions.dtype, device=positions.device)

    rest_lengths = torch.linalg.norm(
        positions[bond_indices[:, 1]] - positions[bond_indices[:, 0]],
        dim=1
    )

    return bond_indices, rest_lengths


def face_footprint_radius(face: FailureFace) -> float:
    """In-plane radius of a failure face, in metres.

    Uses the explicit ``extent_radius`` when the FEM side provides it, otherwise
    the equal-area disc radius ``sqrt(area / pi)``.  The footprint is what the
    DEM field may legitimately sever: a face of area ``A`` releases its energy
    over ``~pi R^2`` of lattice cross-section, not over the whole plane.
    """
    if face.extent_radius is not None:
        radius = float(face.extent_radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("face extent_radius must be positive and finite")
        return radius
    area = float(face.area)
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("face area must be positive and finite")
    return math.sqrt(area / math.pi)


def identify_bonds_crossing_face(
    bond_indices: torch.Tensor,
    positions: torch.Tensor,
    face: FailureFace,
    on_plane_tolerance: float = 0.0,
    footprint_aware: bool = False,
    face_extent: str | None = None,
) -> torch.Tensor:
    """Identify bonds that cross a failure face.
    
    A bond crosses a face if the two endpoints are on opposite sides of the
    face plane (defined by normal and centroid).

    Three criteria are available, in increasing physical fidelity:

    ``on_plane_tolerance == 0`` and ``footprint_aware is False`` (default)
        frozen SI-2 contract: strict sign test on the *infinite* plane.  It cuts
        every bond that straddles the plane, which over-breaks a small face by
        the ratio of the plane area to the footprint area.
    ``on_plane_tolerance > 0``
        endpoints within the tolerance count as lying on the plane: a bond with
        exactly one endpoint on the plane is severed (a plane coincident with a
        particle layer then severs the two bonds per column that touch it, i.e.
        ``2 n^2`` for an ``n x n`` cross-section), while a bond lying *inside*
        the plane (both endpoints on it) stays intact.
    ``footprint_aware is True``
        additionally restricts breakage to the disc of radius
        ``face_footprint_radius(face)`` around the face centroid, measured on
        the in-plane projection of the bond midpoint.  The severed count then
        tracks ``pi R^2 / h^2`` instead of the whole plane.
    """
    if on_plane_tolerance < 0.0:
        raise ValueError("on_plane_tolerance must be non-negative")
    footprint_aware = normalise_face_extent(face_extent, footprint_aware)
    normal_norm = float(torch.linalg.norm(face.normal))
    if not math.isfinite(normal_norm) or normal_norm <= 0.0:
        raise ValueError(
            "failure face normal must be non-degenerate (norm > 0), got "
            f"{face.normal.tolist()}"
        )
    if len(bond_indices) == 0:
        return torch.empty(0, dtype=torch.bool, device=positions.device)
    
    i_idx, j_idx = bond_indices.T
    pos_i = positions[i_idx]
    pos_j = positions[j_idx]
    
    # Move face data to same device as positions
    centroid = face.centroid.to(positions.device)
    normal = face.normal.to(positions.device)
    
    # Signed distance from face plane: normal · (pos - centroid)
    dist_i = torch.sum((pos_i - centroid) * normal, dim=1)
    dist_j = torch.sum((pos_j - centroid) * normal, dim=1)
    
    # Bond crosses face if signs are opposite (one positive, one negative)
    crosses = (dist_i * dist_j) < 0

    if on_plane_tolerance > 0.0:
        on_i = dist_i.abs() <= on_plane_tolerance
        on_j = dist_j.abs() <= on_plane_tolerance
        # exactly one endpoint on the plane -> severed; both on it -> coplanar
        crosses = crosses | (on_i != on_j)

    if footprint_aware:
        radius = face_footprint_radius(face)
        unit = normal / torch.linalg.norm(normal)
        midpoint = 0.5 * (pos_i + pos_j)
        offset = midpoint - centroid
        offset = offset - torch.sum(offset * unit, dim=1, keepdim=True) * unit
        radial = torch.linalg.norm(offset, dim=1)
        crosses = crosses & (radial <= radius + _FOOTPRINT_TOLERANCE)
    
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
            on_plane_tolerance=config.on_plane_tolerance,
            footprint_aware=config.footprint_aware,
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
        mass_mapping=config.mass_mapping,
        footprint_aware=config.footprint_aware,
        on_plane_tolerance=config.on_plane_tolerance,
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

    def mass_accounting(
        self,
        continuum_volume: float | None = None,
        tolerance: float = 0.15,
        *,
        domain_volume: float | None = None,
    ) -> dict[str, Any]:
        """Compare the DEM mass with the continuum mass of a reference volume.

        ``continuum_volume`` (alias ``domain_volume``) is the solid ice volume
        the field is meant to represent (m³), so the reference mass is
        ``density * continuum_volume``.  When it is omitted the volume the
        lattice itself covers is used, i.e. the particle bounding box grown by
        half a cell on every axis, which is ``N h^3`` for a full lattice.

        ``relative_error`` is signed, ``(dem - continuum) / continuum``, as the
        trunk benchmarks display it; ``relative_error_magnitude`` is its
        magnitude, which is what the 15 % gate and the quantisation audit grade.
        ``fem_continuum_mass`` is the spelling the audit tables use for
        ``continuum_mass``.  The
        acceptance bar is the platform's 15 % material-mapping bar; a mapping
        that fails it is reported as such and never silently rounded.
        """
        if domain_volume is not None:
            if continuum_volume is not None:
                raise ValueError(
                    "pass either continuum_volume or domain_volume, not both"
                )
            continuum_volume = domain_volume
        if continuum_volume is None:
            continuum_volume = self._lattice_volume()
        if not math.isfinite(continuum_volume) or continuum_volume <= 0.0:
            raise ValueError("continuum_volume must be positive and finite")
        continuum_mass = self.field.density * continuum_volume
        dem_mass = self.field.total_mass
        signed_error = (dem_mass - continuum_mass) / continuum_mass
        magnitude = abs(signed_error)
        return {
            "mass_mapping": self.field.mass_mapping,
            "mass_mode": self.field.mass_mode,
            "particle_mass": self.field.particle_mass,
            "mass_radius": self.field.mass_radius,
            "num_particles": self.field.num_particles,
            "lattice_volume_covered": self.field.represented_volume,
            "volume_fraction": self.field.volume_fraction,
            "dem_mass": dem_mass,
            "continuum_volume": continuum_volume,
            "continuum_mass": continuum_mass,
            "fem_continuum_mass": continuum_mass,
            "relative_error": signed_error,
            "relative_error_magnitude": magnitude,
            "tolerance": tolerance,
            "within_15pct": magnitude <= tolerance,
        }

    def _lattice_volume(self) -> float:
        """Volume covered by the particle lattice: bounding box + half a cell."""
        positions = self.field.positions
        if positions.numel() == 0:
            raise ValueError("cannot derive a volume from an empty particle field")
        spacing = 2.0 * self.field.particle_radius
        low = positions.min(dim=0).values
        high = positions.max(dim=0).values
        extents = (high - low) + spacing
        volume = 1.0
        for extent in extents.tolist():
            if not math.isfinite(extent) or extent <= 0.0:
                raise ValueError("particle field has a degenerate bounding box")
            volume *= extent
        return volume

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
    
    def energy_accounting(self) -> dict[str, Any]:
        """Account for fracture energy against lattice analytics.

        ``unit_bond_energy = sum(released) / num_broken`` is what the frozen SI-2
        criterion c2 multiplied back, so on its own it cannot fail.  This ledger
        therefore also reports the *independent* lattice references:

        - ``analytic_bond_count`` -- ``sum(area) / h^2`` bonds must be severed
          (``h = 2 r`` is the lattice spacing), independent of the DEM code path;
        - ``analytic_unit_bond_energy`` -- energy released per bond must be
          ``G h^2`` (area-weighted when faces carry different ``G``);
        - ``unit_bond_energy_relative_error`` / ``bond_count_relative_error``
          -- relative distance of the measured values from those references;
        - ``input_consistency_error`` -- ``sum |released_energy - G * area|``,
          i.e. whether the FEM-side input is self-consistent at all;
        - ``face_extent`` -- which crossing criterion produced the bond count
          ("plane": infinite plane, "disc": equal-area footprint).
        """
        total_released = sum(float(face.released_energy) for face in self.faces)
        num_broken = self.field.num_broken_bonds
        spacing = 2.0 * self.field.particle_radius
        spacing_sq = spacing * spacing
        total_area = sum(float(face.area) for face in self.faces)
        analytic_bond_count = total_area / spacing_sq if spacing_sq > 0.0 else 0.0
        if total_area > 0.0:
            weighted_g = sum(float(face.G) * float(face.area) for face in self.faces) / total_area
            analytic_unit_bond_energy = weighted_g * spacing_sq
        else:
            analytic_unit_bond_energy = 0.0
        unit_bond_energy = total_released / num_broken if num_broken > 0 else 0.0
        input_consistency_error = sum(
            abs(float(face.released_energy) - float(face.G) * float(face.area))
            for face in self.faces
        )
        return {
            'total_released_energy': total_released,
            'num_broken_bonds': num_broken,
            'unit_bond_energy': unit_bond_energy,
            'analytic_bond_count': analytic_bond_count,
            'analytic_unit_bond_energy': analytic_unit_bond_energy,
            'unit_bond_energy_relative_error': (
                abs(unit_bond_energy - analytic_unit_bond_energy) / analytic_unit_bond_energy
                if analytic_unit_bond_energy > 0.0 else 0.0
            ),
            'bond_count_relative_error': (
                abs(num_broken - analytic_bond_count) / analytic_bond_count
                if analytic_bond_count > 0.0 else 0.0
            ),
            'input_consistency_error': input_consistency_error,
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
                on_plane_tolerance=self.field.on_plane_tolerance,
                footprint_aware=self.field.footprint_aware,
            )
            bonds_should_break |= crosses
        
        bonds_actually_broken = ~self.field.alive
        
        return {
            'bonds_crossing_faces': bonds_should_break.sum().item(),
            'bonds_actually_broken': bonds_actually_broken.sum().item(),
            'all_face_bonds_broken': torch.all(bonds_should_break == bonds_actually_broken).item()
        }
