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

MASS_RADIUS_FACTOR = (6.0 / math.pi) ** (1.0 / 3.0)
"""``mass_radius / particle_radius`` for a volume-consistent sphere.

A particle on a simple-cubic lattice of spacing ``h = 2 r`` owns the cell
volume ``h^3``.  The sphere that carries that volume has radius
``(3 h^3 / 4 pi)^(1/3) = (6/pi)^(1/3) r = 1.240701 r``, i.e. the *mass* radius
is larger than the *contact* radius by this factor.  The legacy
"equal-volume spheres" mapping instead uses the contact radius, which
under-counts the represented mass by ``1 - pi/6 = 47.64 %``.
"""

VALID_MASS_MAPPINGS = ("cell_volume", "equal_volume_spheres")
"""Canonical mass-mapping names.

* ``cell_volume`` -- ``m = density * h^3`` (continuum consistent, the default).
* ``equal_volume_spheres`` -- ``m = density * (4/3) pi r^3`` (legacy frozen
  mapping, under-counts the represented ice mass by ``1 - pi/6``).
"""

DEFAULT_MASS_MAPPING = "cell_volume"

MASS_MAPPING_ALIASES = {
    "cell_volume": "cell_volume",
    "cell": "cell_volume",
    "lattice_cell": "cell_volume",
    "equal_volume_spheres": "equal_volume_spheres",
    "equal_volume_sphere": "equal_volume_spheres",
    "sphere": "equal_volume_spheres",
}
"""Accepted spellings.  ``mass_mode`` (``"cell"``/``"sphere"``) from the
mass-mapping audit and the ``lattice_cell``/``equal_volume_spheres`` vocabulary
of the footprint audit are aliases of the two canonical names above."""

VALID_FACE_EXTENTS = ("plane", "disc")
"""``plane`` = frozen infinite-plane cut; ``disc`` = area-consistent footprint."""


def canonical_mass_mapping(name: str) -> str:
    """Normalise a mass-mapping spelling to its canonical name."""
    if isinstance(name, str) and name in MASS_MAPPING_ALIASES:
        return MASS_MAPPING_ALIASES[name]
    raise ValueError(
        f"mass_mode/mass_mapping must be one of {VALID_MASS_MAPPINGS} "
        f"(aliases of mass_mapping: {sorted(MASS_MAPPING_ALIASES)}), got {name!r}"
    )


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
    # Optional in-plane extent of the face (m).  ``None`` means the producer did
    # not send one and the footprint radius falls back to the area-equivalent
    # disc ``sqrt(area/pi)``.  The frozen JSON contract is unchanged because the
    # field is optional; ``face_extent="plane"`` ignores it entirely.


@dataclass
class FractureGenConfig:
    """Configuration for fracture generation."""
    density: float = 917.0  # ice density (kg/m³)
    particle_radius: float = 0.1  # particle radius (m)
    bond_stiffness: float = 2000.0  # bond stiffness (N/m)
    domain_bounds: tuple[float, float, float, float, float, float] | None = None
    # (xmin, xmax, ymin, ymax, zmin, zmax)
    mass_mapping: str | None = None
    # None -> "cell_volume" (default).  Accepted spellings: see
    # MASS_MAPPING_ALIASES ("cell_volume"/"cell"/"lattice_cell",
    # "equal_volume_spheres"/"equal_volume_sphere"/"sphere").  Normalised in
    # __post_init__ into ``self.mass_mapping`` (canonical) + ``self.mass_mode``
    # ("cell"/"sphere", the spelling used by tensordem.fracture_mass_mapping).
    mass_mode: str | None = None
    # Legacy alias of ``mass_mapping``; kept because the mass-mapping audit and
    # ``FractureField.mass_mode`` use the "cell"/"sphere" vocabulary.
    face_extent: str = "plane"
    # "plane" -> frozen infinite-plane cut (independent of the declared area)
    # "disc"  -> cut only inside the footprint disc of the face
    #            (radius = extent_radius, else sqrt(area/pi))
    footprint_aware: bool = False
    # Convenience flag of the footprint audit: True is exactly face_extent="disc"
    # with the radius taken from FailureFace.extent_radius when it is present.
    on_plane_tolerance: float = 0.0
    # When > 0 a bond with exactly one endpoint within this distance of the face
    # plane counts as severed, which is what a cut coinciding with a particle
    # layer needs.  0 keeps the frozen strict sign test (and severs nothing on a
    # layer-coincident plane).

    def __post_init__(self):
        if self.density <= 0:
            raise ValueError("density must be positive")
        if self.particle_radius <= 0:
            raise ValueError("particle_radius must be positive")
        if self.bond_stiffness <= 0:
            raise ValueError("bond_stiffness must be positive")

        resolved: str | None = None
        if self.mass_mapping is not None:
            resolved = canonical_mass_mapping(self.mass_mapping)
        if self.mass_mode is not None:
            legacy = canonical_mass_mapping(self.mass_mode)
            if resolved is not None and legacy != resolved:
                raise ValueError(
                    "mass_mapping and mass_mode disagree: "
                    f"{resolved!r} vs {legacy!r}"
                )
            resolved = legacy
        self.mass_mapping = DEFAULT_MASS_MAPPING if resolved is None else resolved
        self.mass_mode = (
            "sphere" if self.mass_mapping == "equal_volume_spheres" else "cell"
        )

        if self.face_extent not in VALID_FACE_EXTENTS:
            raise ValueError(
                f"face_extent must be one of {VALID_FACE_EXTENTS}, "
                f"got {self.face_extent!r}"
            )
        tolerance = float(self.on_plane_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("on_plane_tolerance must be finite and non-negative")
        self.on_plane_tolerance = tolerance
        self.footprint_aware = bool(self.footprint_aware)

    @property
    def resolved_face_extent(self) -> str:
        """Extent actually used to cut bonds ("disc" if footprint_aware)."""
        return "disc" if self.footprint_aware else self.face_extent


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
        domain_bounds: tuple[float, float, float, float, float, float] | None = None,
        face_extent: str = "plane",
        on_plane_tolerance: float = 0.0,
    ):
        resolved = canonical_mass_mapping(mass_mode)
        if mass_mapping is not None:
            explicit = canonical_mass_mapping(mass_mapping)
            if explicit != resolved:
                raise ValueError(
                    f"mass_mapping {explicit!r} disagrees with mass_mode {mass_mode!r}"
                )
            resolved = explicit
        self.mass_mapping = resolved
        self.mass_mode = "sphere" if resolved == "equal_volume_spheres" else "cell"
        if face_extent not in VALID_FACE_EXTENTS:
            raise ValueError(
                f"face_extent must be one of {VALID_FACE_EXTENTS}, got {face_extent!r}"
            )
        self.face_extent = face_extent
        self.on_plane_tolerance = float(on_plane_tolerance)
        self.domain_bounds = domain_bounds
        self.domain_volume = (
            None
            if domain_bounds is None
            else (domain_bounds[1] - domain_bounds[0])
            * (domain_bounds[3] - domain_bounds[2])
            * (domain_bounds[5] - domain_bounds[4])
        )
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
        """Radius of the sphere that carries ``particle_mass``.

        Equal to ``particle_radius`` for the legacy sphere mapping and to
        ``MASS_RADIUS_FACTOR * particle_radius`` (1.240701) for the
        volume-consistent mapping: the *inertial* radius exceeds the *contact*
        radius by exactly the packing factor ``(6/pi)^(1/3)``.
        """
        return (3.0 * self.particle_mass / (4.0 * math.pi * self.density)) ** (1.0 / 3.0)

    @property
    def volume_fraction(self) -> float:
        """``represented_volume / domain_volume`` (raises if bounds unknown)."""
        if self.domain_volume is None:
            raise ValueError(
                "volume_fraction needs domain_bounds (this field was built without them)"
            )
        return self.represented_volume / self.domain_volume

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
            released_energy=face_data['released_energy'],
            extent_radius=face_data.get('extent_radius')
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

    # Find all pairs within bonding distance.  Row-chunked cdist in place of the
    # original double Python loop: the pair list (i < j, lexicographic) and the
    # bond tolerance are unchanged, but the cost is O(N^2) tensor ops instead of
    # O(N^2) Python iterations (1000 particles: ~30 s -> ~0.05 s).
    i_parts: list[torch.Tensor] = []
    j_parts: list[torch.Tensor] = []
    chunk = 512
    for start in range(0, n_particles, chunk):
        stop = min(start + chunk, n_particles)
        dist = torch.cdist(positions[start:stop], positions)
        rows = torch.arange(start, stop, device=positions.device).unsqueeze(1)
        cols = torch.arange(n_particles, device=positions.device).unsqueeze(0)
        hits = torch.nonzero((dist <= cutoff) & (cols > rows), as_tuple=False)
        if hits.numel():
            i_parts.append(hits[:, 0] + start)
            j_parts.append(hits[:, 1])

    if not i_parts:
        return torch.empty((0, 2), dtype=torch.long, device=positions.device), \
               torch.empty(0, dtype=positions.dtype, device=positions.device)

    bond_indices = torch.stack(
        [torch.cat(i_parts).to(dtype=torch.long), torch.cat(j_parts).to(dtype=torch.long)],
        dim=1,
    )
    rest_lengths = torch.linalg.norm(
        positions[bond_indices[:, 1]] - positions[bond_indices[:, 0]],
        dim=1
    )
    
    return bond_indices, rest_lengths


def face_footprint_radius(face: FailureFace) -> float:
    """In-plane radius of the disc that represents a failure face.

    Uses ``FailureFace.extent_radius`` when the producer supplied one, otherwise
    the area-equivalent disc ``sqrt(area / pi)``.  The area-only fallback is the
    best that the frozen ``area + centroid`` contract allows; for a square face
    it under-covers the corners by up to 12 % (see the footprint audit).
    """
    extent = getattr(face, "extent_radius", None)
    if extent is None:
        area = float(face.area)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError(
                f"face {getattr(face, 'id', '?')}: area must be finite and positive "
                "to derive a footprint radius"
            )
        return math.sqrt(area / math.pi)
    extent = float(extent)
    if not math.isfinite(extent) or extent < 0.0:
        raise ValueError("FailureFace.extent_radius must be finite and non-negative")
    return extent


def identify_bonds_crossing_face(
    bond_indices: torch.Tensor,
    positions: torch.Tensor,
    face: FailureFace,
    on_plane_tolerance: float = 0.0,
    face_extent: str = "plane",
) -> torch.Tensor:
    """Identify bonds that cross a failure face.

    Default (``face_extent="plane"``, ``on_plane_tolerance=0``) is the frozen
    criterion: a bond is severed when its endpoints straddle the face *plane*,
    independently of the declared face area.

    ``on_plane_tolerance > 0`` additionally severs a bond with exactly one
    endpoint within ``tolerance`` of the plane.  This is needed when the cut
    surface coincides with a particle layer: there the strict sign test sees a
    zero product and severs nothing.  A coplanar bond (both endpoints on the
    plane) is never severed -- it lies *in* the fracture surface.

    ``face_extent="disc"`` restricts the cut to the footprint disc of radius
    :func:`face_footprint_radius`, evaluated at the bond/plane intersection
    point.  This converts the infinite-plane over-break into an area-consistent
    cut.
    """
    if face_extent not in VALID_FACE_EXTENTS:
        raise ValueError(
            f"face_extent must be one of {VALID_FACE_EXTENTS}, got {face_extent!r}"
        )
    tolerance = float(on_plane_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("on_plane_tolerance must be finite and non-negative")

    # Move face data to same device/dtype as positions
    centroid = face.centroid.to(device=positions.device, dtype=positions.dtype)
    normal = face.normal.to(device=positions.device, dtype=positions.dtype)
    normal_norm = float(torch.linalg.norm(normal))
    if not math.isfinite(normal_norm) or normal_norm <= 0.0:
        raise ValueError(
            f"face {getattr(face, 'id', '?')}: normal must be non-zero and finite"
        )
    normal = normal / normal_norm

    if len(bond_indices) == 0:
        return torch.empty(0, dtype=torch.bool, device=positions.device)

    i_idx, j_idx = bond_indices.T
    pos_i = positions[i_idx]
    pos_j = positions[j_idx]

    # Signed distance from face plane: normal · (pos - centroid)
    dist_i = torch.sum((pos_i - centroid) * normal, dim=1)
    dist_j = torch.sum((pos_j - centroid) * normal, dim=1)

    # Frozen criterion: endpoints on opposite sides of the plane
    crosses = (dist_i * dist_j) < 0

    if tolerance > 0.0:
        on_i = dist_i.abs() <= tolerance
        on_j = dist_j.abs() <= tolerance
        # exactly one endpoint on the plane -> the cut passes through the bond
        crosses = crosses | (on_i ^ on_j)
        dist_i = torch.where(on_i, torch.zeros_like(dist_i), dist_i)
        dist_j = torch.where(on_j, torch.zeros_like(dist_j), dist_j)

    if face_extent == "disc":
        radius = face_footprint_radius(face)
        denom = dist_i - dist_j
        t = torch.where(
            denom != 0,
            dist_i / torch.where(denom != 0, denom, torch.ones_like(denom)),
            torch.full_like(denom, 0.5),
        )
        point = pos_i + t.unsqueeze(1) * (pos_j - pos_i)
        radial = torch.linalg.norm(point - centroid, dim=1)
        crosses = crosses & (radial <= radius + 1e-12)

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
    extent = config.resolved_face_extent
    for face in failure_faces:
        crosses = identify_bonds_crossing_face(
            bond_indices,
            positions,
            face,
            on_plane_tolerance=config.on_plane_tolerance,
            face_extent=extent,
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
        domain_bounds=config.domain_bounds,
        face_extent=extent,
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

    def mass_accounting(
        self,
        domain_volume: float | None = None,
        tolerance: float = 0.15,
    ) -> dict[str, Any]:
        """DEM mass vs the continuum mass ``rho * V_domain`` (signed).

        ``relative_error`` is signed, ``(M_dem - rho V) / (rho V)``: positive
        when the field over-counts the ice it replaces, negative when it
        under-counts.  ``within_15pct`` applies the platform acceptance bar to
        ``|relative_error|``.

        Two independent contributions are visible separately, which is the
        point of the audit: ``volume_fraction`` (lattice quantisation: the
        centres cover ``N h`` per axis, not the requested domain) and the
        mass-mapping deficit ``pi/6 - 1`` of the legacy sphere mapping.
        """
        volume = self.field.domain_volume if domain_volume is None else domain_volume
        if volume is None:
            raise ValueError(
                "domain_volume must be given (this field was built without domain_bounds)"
            )
        volume = float(volume)
        if not math.isfinite(volume) or volume <= 0.0:
            raise ValueError("domain_volume must be finite and positive")
        tol = float(tolerance)
        if not math.isfinite(tol) or tol < 0.0:
            raise ValueError("tolerance must be finite and non-negative")

        dem_mass = self.field.total_mass
        continuum = self.field.density * volume
        relative_error = (dem_mass - continuum) / continuum
        return {
            "mass_mapping": self.field.mass_mapping,
            "mass_mode": self.field.mass_mode,
            "num_particles": self.field.num_particles,
            "particle_mass": self.field.particle_mass,
            "mass_radius": self.field.mass_radius,
            "dem_mass": dem_mass,
            "fem_continuum_mass": continuum,
            "continuum_mass": continuum,
            "domain_volume": volume,
            "lattice_volume_covered": self.field.represented_volume,
            "volume_fraction": self.field.represented_volume / volume,
            "relative_error": relative_error,
            "tolerance": tol,
            "within_15pct": abs(relative_error) <= tol,
        }

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
        """Account for fracture energy and check it against lattice analytics.

        Frozen keys (kept unchanged for existing callers):

        - ``total_released_energy``: sum of ``released_energy`` over the faces
        - ``num_broken_bonds``: broken bonds in the DEM field
        - ``unit_bond_energy``: ``total_released_energy / num_broken_bonds``

        Falsifiable keys (this is what makes the shipped ``c2`` check
        non-vacuous -- the frozen check divided the released energy by the bond
        count it had itself produced, an algebraic identity):

        - ``analytic_bond_count``: ``sum(A_face) / (2r)^2``, the number of bonds
          a area-consistent cut must sever on a simple-cubic lattice
        - ``analytic_unit_bond_energy``: ``G_mean * (2r)^2``, the energy one
          broken bond stands for
        - ``bond_count_relative_error``, ``unit_bond_energy_relative_error``:
          magnitudes ``|measured - analytic| / analytic``
        - ``input_consistency_error``: ``max |released_energy - G * area|`` over
          the faces (J); a producer that ships inconsistent inputs is flagged
          here instead of silently redefining the energy scale
        - ``face_extent``: the cut criterion that produced the bond count
        """
        total_released = float(sum(face.released_energy for face in self.faces))
        num_broken = self.field.num_broken_bonds
        unit_bond_energy = total_released / num_broken if num_broken > 0 else 0.0

        spacing_sq = (2.0 * self.field.particle_radius) ** 2
        total_area = float(sum(face.area for face in self.faces))
        if total_area > 0.0:
            analytic_bond_count = total_area / spacing_sq
            g_mean = total_released / total_area
            analytic_unit_bond_energy = g_mean * spacing_sq
        else:
            analytic_bond_count = 0.0
            analytic_unit_bond_energy = float("nan")

        bond_count_relative_error = (
            abs(num_broken - analytic_bond_count) / analytic_bond_count
            if analytic_bond_count > 0.0
            else float("nan")
        )
        if math.isfinite(analytic_unit_bond_energy) and analytic_unit_bond_energy != 0.0:
            unit_bond_energy_relative_error = (
                abs(unit_bond_energy - analytic_unit_bond_energy)
                / abs(analytic_unit_bond_energy)
            )
        else:
            unit_bond_energy_relative_error = float("nan")

        input_consistency_error = max(
            (abs(face.released_energy - face.G * face.area) for face in self.faces),
            default=0.0,
        )

        return {
            'total_released_energy': total_released,
            'num_broken_bonds': num_broken,
            'unit_bond_energy': unit_bond_energy,
            'analytic_bond_count': analytic_bond_count,
            'analytic_unit_bond_energy': analytic_unit_bond_energy,
            'bond_count_relative_error': bond_count_relative_error,
            'unit_bond_energy_relative_error': unit_bond_energy_relative_error,
            'input_consistency_error': float(input_consistency_error),
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
                face_extent=self.field.face_extent,
            )
            bonds_should_break |= crosses
        
        bonds_actually_broken = ~self.field.alive
        
        return {
            'bonds_crossing_faces': bonds_should_break.sum().item(),
            'bonds_actually_broken': bonds_actually_broken.sum().item(),
            'all_face_bonds_broken': torch.all(bonds_should_break == bonds_actually_broken).item()
        }
