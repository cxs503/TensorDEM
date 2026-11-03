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

Audited surface (restored for the merged contracts
``tests/test_fracture_gen_audit.py`` and
``tests/test_fracture_gen_mass_footprint.py``):

* ``mass_mapping`` selects the mapping by its audited name
  (``"cell_volume"`` / ``"lattice_cell"`` vs ``"equal_volume_spheres"`` /
  ``"sphere"``); the frozen sphere mapping is not the default.
* ``face_extent`` selects how far a failure face reaches: ``None`` keeps the
  frozen infinite-plane criterion, ``"plane"`` cuts the whole cross-section
  exactly once per lattice column (a layer-coincident plane is tie-broken),
  ``"disc"`` (or ``footprint_aware=True``) restricts breakage to the footprint
  disc of radius ``sqrt(area/pi)`` or ``FailureFace.extent_radius``.
* ``on_plane_tolerance`` breaks bonds with exactly one endpoint on the plane.
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

#: Radius factor turning the geometric radius ``r`` into the radius of the
#: sphere carrying the mass of the lattice cell ``h = 2 r``:
#: ``(4/3) pi (k r)^3 = h^3``  ->  ``k = 2 (3 / (4 pi))^(1/3) = 1.240701``.
MASS_RADIUS_FACTOR = 2.0 * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)

#: Audited mapping name -> canonical ``mass_mode`` of
#: ``tensordem.fracture_mass_mapping``.
MASS_MAPPING_ALIASES = {
    "cell_volume": "cell",
    "lattice_cell": "cell",
    "equal_volume_spheres": "sphere",
    "sphere": "sphere",
}

#: Canonical reporting label for each ``mass_mode``.
_MASS_MODE_LABELS = {"cell": "cell_volume", "sphere": "equal_volume_spheres"}

#: Supported face-extent models (``None`` = frozen infinite plane).
FACE_EXTENTS = ("plane", "disc")


def face_footprint_radius(face: "FailureFace") -> float:
    """In-plane radius of the face footprint, in metres.

    ``extent_radius`` wins when the producer knows the real face extent;
    otherwise the declared area is interpreted as a disc, ``sqrt(A/pi)``.
    The frozen contract carries only area and centroid, so the shape is a
    modelling assumption and the resulting bond count is resolved only to
    ``O(1)`` per face.
    """
    if face.extent_radius is not None:
        if face.extent_radius <= 0.0:
            raise ValueError("extent_radius must be positive")
        return float(face.extent_radius)
    if face.area <= 0.0:
        raise ValueError("face area must be positive")
    return math.sqrt(face.area / math.pi)


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
    # In-plane footprint radius (m). ``None`` -> derived from ``area`` as a
    # disc, see ``face_footprint_radius``. Only used when the generator runs
    # with ``face_extent="disc"`` / ``footprint_aware=True``.


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
    # Audited mapping name, one of ``MASS_MAPPING_ALIASES``. When given it
    # overrides ``mass_mode``; ``None`` keeps the ``mass_mode`` default.
    face_extent: str | None = None
    # None    -> frozen infinite-plane crossing (strict sign test)
    # "plane" -> the face spans the cross-section; a layer-coincident plane is
    #            tie-broken so the cross-section is cut exactly once
    # "disc"  -> breakage limited to the footprint disc of the face
    footprint_aware: bool = False
    # Shortcut for ``face_extent="disc"``.
    on_plane_tolerance: float = 0.0
    # > 0 -> a bond with exactly one endpoint within this distance of the
    #        plane is broken (restores the bonds a coincident plane loses).

    def __post_init__(self):
        if self.density <= 0:
            raise ValueError("density must be positive")
        if self.particle_radius <= 0:
            raise ValueError("particle_radius must be positive")
        if self.bond_stiffness <= 0:
            raise ValueError("bond_stiffness must be positive")
        if self.mass_mapping is None:
            if self.mass_mode not in VALID_MASS_MODES:
                raise ValueError(
                    f"mass_mode must be one of {VALID_MASS_MODES}, "
                    f"got {self.mass_mode!r}"
                )
            label = _MASS_MODE_LABELS[self.mass_mode]
        else:
            if self.mass_mapping not in MASS_MAPPING_ALIASES:
                raise ValueError(
                    "mass_mapping must be one of "
                    f"{sorted(MASS_MAPPING_ALIASES)}, got {self.mass_mapping!r}"
                )
            label = self.mass_mapping
            self.mass_mode = MASS_MAPPING_ALIASES[label]
        #: Audited mapping label, reported by the ledger.
        self.mass_mapping_label = label
        if self.face_extent is not None and self.face_extent not in FACE_EXTENTS:
            raise ValueError(
                "face_extent must be one of "
                f"{[None, *FACE_EXTENTS]}, got {self.face_extent!r}"
            )
        if self.on_plane_tolerance < 0.0:
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
        face_extent: str | None = None,
        on_plane_tolerance: float = 0.0,
        domain_bounds: tuple[float, float, float, float, float, float] | None = None,
    ):
        if mass_mode not in VALID_MASS_MODES:
            raise ValueError(
                f"mass_mode must be one of {VALID_MASS_MODES}, got {mass_mode!r}"
            )
        self.mass_mode = mass_mode
        self.mass_mapping = (
            mass_mapping if mass_mapping is not None else _MASS_MODE_LABELS[mass_mode]
        )
        self.face_extent = face_extent
        self.on_plane_tolerance = on_plane_tolerance
        self.domain_bounds = domain_bounds
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
    def mass_radius(self) -> float:
        """Radius of the sphere carrying one particle's mass.

        For the volume-consistent ``cell`` mapping this is the equal-volume
        radius of the lattice cell, ``MASS_RADIUS_FACTOR * r``; the frozen
        sphere mapping uses the geometric radius itself.
        """
        if self.mass_mode == "cell":
            return MASS_RADIUS_FACTOR * self.particle_radius
        return self.particle_radius

    @property
    def domain_volume(self) -> float:
        """Volume of the requested domain, from ``domain_bounds``."""
        if self.domain_bounds is None:
            raise ValueError("domain_bounds are required for the domain volume")
        xmin, xmax, ymin, ymax, zmin, zmax = self.domain_bounds
        volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        if volume <= 0.0:
            raise ValueError("domain volume must be positive")
        return volume

    @property
    def volume_fraction(self) -> float:
        """Lattice volume covered by the particles over the domain volume.

        Reported to 12 decimals: the lattice volume is a float sum of ``N h^3``
        and the domain volume a product of float extents, so an exactly covered
        domain comes out as ``1.0 + O(1e-16)`` before rounding.
        """
        return round(self.represented_volume / self.domain_volume, 12)

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

    The pair search is vectorised over the ``n x n`` distance matrix instead of
    a Python double loop: the audit lattices run to thousands of particles, for
    which the loop was the dominant cost. ``torch.triu`` keeps one row-major
    ``(i, j)`` ordering with ``i < j``, so the returned pair list is identical
    to the brute-force enumeration.
    """
    n_particles = len(positions)
    spacing = 2.0 * particle_radius

    if n_particles < 2:
        return torch.empty((0, 2), dtype=torch.long, device=positions.device), \
               torch.empty(0, dtype=positions.dtype, device=positions.device)

    with torch.no_grad():
        delta = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist = torch.linalg.norm(delta, dim=2)
        adjacent = torch.triu(dist <= spacing * 1.01, diagonal=1)

    bond_pairs = torch.nonzero(adjacent, as_tuple=False)

    if len(bond_pairs) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=positions.device), \
               torch.empty(0, dtype=positions.dtype, device=positions.device)

    bond_indices = bond_pairs.to(dtype=torch.long, device=positions.device)
    rest_lengths = torch.linalg.norm(
        positions[bond_indices[:, 1]] - positions[bond_indices[:, 0]],
        dim=1
    )

    return bond_indices, rest_lengths


def identify_bonds_crossing_face(
    bond_indices: torch.Tensor,
    positions: torch.Tensor,
    face: FailureFace,
    face_extent: str | None = None,
    on_plane_tolerance: float = 0.0,
) -> torch.Tensor:
    """Identify bonds that cross a failure face.

    ``face_extent`` selects the extent model:

    ``None``
        Frozen contract: a bond crosses iff its endpoints lie strictly on
        opposite sides of the infinite face plane. A plane coincident with a
        particle layer severs **nothing** -- the degenerate case the audit
        reports.
    ``"plane"``
        The face spans the whole cross-section, so the cut must sever each
        lattice column exactly once. Straddling bonds are cut, and a bond
        leaving a layer-coincident plane towards the positive normal side is cut
        too. Both branches are independent of the bond's index order, so exactly
        one bond per column is selected even when no bond straddles the plane.
    ``"disc"``
        Strict crossing **and** the in-plane crossing point within
        ``face_footprint_radius(face)`` of the face centroid, so a small finite
        face no longer severs the whole plane.

    ``on_plane_tolerance > 0`` additionally breaks every bond with exactly one
    endpoint within that distance of the plane; bonds lying inside the plane
    stay intact.
    """
    if face_extent is not None and face_extent not in FACE_EXTENTS:
        raise ValueError(
            f"face_extent must be one of {[None, *FACE_EXTENTS]}, got {face_extent!r}"
        )
    if on_plane_tolerance < 0.0:
        raise ValueError("on_plane_tolerance must be non-negative")
    normal = face.normal.to(positions.device)
    norm = torch.linalg.norm(normal)
    if float(norm) <= 0.0:
        raise ValueError("failure-face normal must be non-degenerate")
    if len(bond_indices) == 0:
        return torch.zeros(0, dtype=torch.bool, device=positions.device)

    i_idx, j_idx = bond_indices.T
    pos_i = positions[i_idx]
    pos_j = positions[j_idx]

    # Move face data to same device as positions
    centroid = face.centroid.to(positions.device)
    normal = normal / norm

    # Signed distance from face plane: normal . (pos - centroid)
    dist_i = torch.sum((pos_i - centroid) * normal, dim=1)
    dist_j = torch.sum((pos_j - centroid) * normal, dim=1)

    # Bond crosses face if signs are opposite (one positive, one negative)
    crosses = (dist_i * dist_j) < 0
    if face_extent == "plane":
        crosses = (
            crosses
            | ((dist_i == 0) & (dist_j > 0))
            | ((dist_j == 0) & (dist_i > 0))
        )

    if face_extent == "disc":
        radius = face_footprint_radius(face)
        denom = dist_i - dist_j
        safe = denom.abs() > 0
        t = torch.where(
            safe,
            dist_i / torch.where(safe, denom, torch.ones_like(denom)),
            torch.zeros_like(denom),
        )
        point = pos_i + t.unsqueeze(1) * (pos_j - pos_i)
        offset = point - centroid
        in_plane = offset - torch.sum(offset * normal, dim=1, keepdim=True) * normal
        crosses = crosses & (torch.linalg.norm(in_plane, dim=1) <= radius + 1e-12)

    if on_plane_tolerance > 0.0:
        on_i = dist_i.abs() <= on_plane_tolerance
        on_j = dist_j.abs() <= on_plane_tolerance
        crosses = crosses | (on_i ^ on_j)

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
    extent = config.face_extent
    if config.footprint_aware:
        extent = "disc"
    for face in failure_faces:
        crosses = identify_bonds_crossing_face(
            bond_indices,
            positions,
            face,
            face_extent=extent,
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
        mass_mapping=config.mass_mapping_label,
        face_extent=extent,
        on_plane_tolerance=config.on_plane_tolerance,
        domain_bounds=config.domain_bounds,
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
    
    def mass_accounting(
        self,
        domain_volume: float | None = None,
        mass_mapping: str | None = None,
        tolerance: float = 0.15,
        fem_mass: float | None = None,
    ) -> dict[str, Any]:
        """Audit the DEM mass of this field against a continuum reference.

        The reference is ``fem_mass`` when supplied (TensorFEM's ice mass) and
        otherwise ``density * domain_volume``. ``relative_error`` is **signed**,
        ``(dem - reference) / reference``; ``absolute_relative_error`` is its
        magnitude and is what the 15 % platform acceptance bar is applied to.

        ``mass_mapping`` defaults to the mapping the field was built with, and
        when given it re-prices the particles without rebuilding the lattice.

        This is the criterion the frozen prereg check could not apply: it
        compared the DEM mass with the same sphere formula it was built from,
        which is an algebraic identity (see ``mass_conservation_error``).
        """
        if domain_volume is None:
            if self.field.domain_bounds is None:
                raise ValueError(
                    "domain_volume is required when the field carries no domain_bounds"
                )
            domain_volume = self.field.domain_volume
        if domain_volume <= 0.0:
            raise ValueError("domain_volume must be positive")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")

        if mass_mapping is None:
            mode = self.field.mass_mode
            label = self.field.mass_mapping
        else:
            if mass_mapping not in MASS_MAPPING_ALIASES:
                raise ValueError(
                    f"mass_mapping must be one of {sorted(MASS_MAPPING_ALIASES)}, "
                    f"got {mass_mapping!r}"
                )
            label = mass_mapping
            mode = MASS_MAPPING_ALIASES[label]

        density = self.field.density
        radius = self.field.particle_radius
        num_particles = self.field.num_particles
        unit_mass = _particle_mass(density, radius, mode)
        dem_mass = num_particles * unit_mass
        reference_mass = density * domain_volume if fem_mass is None else float(fem_mass)
        if reference_mass <= 0.0:
            raise ValueError("reference mass must be positive")

        relative_error = (dem_mass - reference_mass) / reference_mass
        return {
            "protocol": "tensordem-fracture-mass-accounting-v1",
            "mass_mapping": label,
            "mass_mode": mode,
            "num_particles": num_particles,
            "particle_radius_m": radius,
            "lattice_spacing_m": 2.0 * radius,
            "mass_radius_m": MASS_RADIUS_FACTOR * radius if mode == "cell" else radius,
            "particle_mass_kg": unit_mass,
            "lattice_volume_covered": num_particles * particle_cell_volume(radius),
            "domain_volume_m3": domain_volume,
            "dem_mass": dem_mass,
            "continuum_mass": reference_mass,
            "fem_continuum_mass": reference_mass,
            "reference_source": (
                "fem_mass" if fem_mass is not None else "density*domain_volume"
            ),
            "relative_error": relative_error,
            "absolute_relative_error": abs(relative_error),
            "tolerance": tolerance,
            "within_15pct": abs(relative_error) <= tolerance,
        }

    def energy_accounting(self) -> dict[str, Any]:
        """Account for fracture energy, with the falsifiable analytic keys.

        ``num_broken_bonds`` is compared with ``A / h^2`` (one bond per lattice
        column) and ``unit_bond_energy`` with ``G h^2`` (the energy one bond
        represents). ``input_consistency_error`` reports the face contract's own
        inconsistency, ``sum |released_energy - G area|``.

        The frozen criteria c2/c3 compared the measured energy with the same
        ``released_energy / num_broken_bonds`` identity, so they could not
        detect an over-broken face; these keys can.
        """
        total_released = sum(face.released_energy for face in self.faces)
        num_broken = self.field.num_broken_bonds
        spacing_sq = (2.0 * self.field.particle_radius) ** 2

        analytic_bond_count = sum(face.area / spacing_sq for face in self.faces)
        analytic_total_energy = sum(face.G * face.area for face in self.faces)
        analytic_unit_bond_energy = (
            analytic_total_energy / analytic_bond_count if analytic_bond_count > 0 else 0.0
        )
        unit_bond_energy = total_released / num_broken if num_broken > 0 else 0.0

        def _rel(measured: float, reference: float) -> float:
            if reference <= 0.0:
                return 0.0
            return (measured - reference) / reference

        return {
            "total_released_energy": total_released,
            "num_broken_bonds": num_broken,
            "unit_bond_energy": unit_bond_energy,
            "analytic_bond_count": analytic_bond_count,
            "analytic_unit_bond_energy": analytic_unit_bond_energy,
            "unit_bond_energy_relative_error": abs(
                _rel(unit_bond_energy, analytic_unit_bond_energy)
            ),
            "bond_count_relative_error": abs(
                _rel(float(num_broken), analytic_bond_count)
            ),
            "input_consistency_error": sum(
                abs(face.released_energy - face.G * face.area) for face in self.faces
            ),
            "face_extent": self.field.face_extent or "infinite_plane",
            "lattice_spacing_sq": spacing_sq,
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
                face
            )
            bonds_should_break |= crosses
        
        bonds_actually_broken = ~self.field.alive
        
        return {
            'bonds_crossing_faces': bonds_should_break.sum().item(),
            'bonds_actually_broken': bonds_actually_broken.sum().item(),
            'all_face_bonds_broken': torch.all(bonds_should_break == bonds_actually_broken).item()
        }
