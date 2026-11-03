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

#: Radius of the cell-volume-equivalent sphere, in units of the contact radius:
#: ``r_mass = (6/pi)^(1/3) r = 1.240701 r`` for lattice spacing ``h = 2 r``.
#: With this radius ``density * 4/3 pi r_mass^3 = density * h^3``, i.e. the
#: lattice cell mass, so the *contact* radius stays ``r`` (particles do not
#: overlap) while the mass matches the represented ice volume.
MASS_RADIUS_FACTOR = (6.0 / math.pi) ** (1.0 / 3.0)

#: Accepted values of :attr:`FractureGenConfig.mass_mapping`, mapped onto the
#: ``mass_mode`` vocabulary of :mod:`tensordem.fracture_mass_mapping`.
MASS_MAPPING_TO_MODE = {
    "equal_volume_sphere": "sphere",
    "equal_volume_spheres": "sphere",  # m2287 audit vocabulary
    "lattice_cell": "cell",
    "cell_volume": "cell",             # m2287 audit vocabulary
}

#: How a failure face is mapped onto the bond cut: ``"plane"`` severs every
#: bond crossing the infinite face plane (frozen behaviour), ``"disc"`` only
#: those whose cut point falls inside the face footprint disc.
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
    extent_radius: float | None = None  # optional footprint radius of the face (m)

    def __post_init__(self):
        if self.extent_radius is not None and self.extent_radius < 0.0:
            raise ValueError(
                f"extent_radius must be non-negative, got {self.extent_radius!r}"
            )


@dataclass
class FractureGenConfig:
    """Configuration for fracture generation."""
    density: float = 917.0  # ice density (kg/m³)
    particle_radius: float = 0.1  # particle radius (m)
    bond_stiffness: float = 2000.0  # bond stiffness (N/m)
    domain_bounds: tuple[float, float, float, float, float, float] | None = None
    # (xmin, xmax, ymin, ymax, zmin, zmax)
    mass_mode: str | None = None
    # "cell"   -> m = density * h^3, volume consistent (default, fulfils the
    #             frozen "equal-volume spheres" intent)
    # "sphere" -> m = density * (4/3) pi r^3, legacy pre-audit mapping that
    #             under-counts mass by 1 - pi/6 = 47.64 %
    mass_mapping: str = "lattice_cell"
    # "lattice_cell"         -> equivalent to mass_mode="cell" (volume
    #                           consistent; the shipped default, matching
    #                           docs/dem3d_fracture_mass_mapping_audit.md)
    # "equal_volume_sphere"  -> equivalent to mass_mode="sphere" (frozen
    #                           prereg mapping; under-counts the represented ice
    #                           volume by 1 - pi/6 = 47.64 %, opt-in only)
    # An explicitly supplied ``mass_mode`` wins over ``mass_mapping``.
    footprint_aware: bool = False
    # When True, a bond is only severed if the cut point lies inside the face
    # footprint disc (radius ``FailureFace.extent_radius`` or sqrt(A/pi)).
    on_plane_tolerance: float = 0.0
    # Half-width (m) of the "endpoint lies on the face plane" band. With 0 the
    # strict sign test is used, so a face plane coincident with a particle layer
    # severs nothing; a small positive value restores the 2 n^2 bonds.
    face_extent: str | None = None
    # "plane" -> infinite-plane cut (frozen default), "disc" -> footprint disc.
    # An explicitly supplied ``face_extent`` wins over ``footprint_aware``.

    def __post_init__(self):
        if self.density <= 0:
            raise ValueError("density must be positive")
        if self.particle_radius <= 0:
            raise ValueError("particle_radius must be positive")
        if self.bond_stiffness <= 0:
            raise ValueError("bond_stiffness must be positive")
        if self.mass_mapping not in MASS_MAPPING_TO_MODE:
            raise ValueError(
                "mass_mapping must be one of "
                f"{tuple(MASS_MAPPING_TO_MODE)}, got {self.mass_mapping!r}"
            )
        if self.mass_mode is None:
            self.mass_mode = MASS_MAPPING_TO_MODE[self.mass_mapping]
        if self.mass_mode not in VALID_MASS_MODES:
            raise ValueError(
                f"mass_mode must be one of {VALID_MASS_MODES}, got {self.mass_mode!r}"
            )
        if self.on_plane_tolerance < 0.0:
            raise ValueError(
                f"on_plane_tolerance must be non-negative, got {self.on_plane_tolerance!r}"
            )
        if self.face_extent is not None and self.face_extent not in VALID_FACE_EXTENTS:
            raise ValueError(
                f"face_extent must be one of {VALID_FACE_EXTENTS}, got {self.face_extent!r}"
            )
        if self.face_extent is None:
            self.face_extent = "disc" if self.footprint_aware else "plane"
        self.footprint_aware = self.face_extent == "disc"


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
        mass_mapping: str | None = None,
        face_extent: str | None = None,
        domain_bounds: tuple[float, float, float, float, float, float] | None = None,
    ):
        if mass_mode not in VALID_MASS_MODES:
            raise ValueError(
                f"mass_mode must be one of {VALID_MASS_MODES}, got {mass_mode!r}"
            )
        self.mass_mode = mass_mode
        self.positions = positions
        self.velocities = velocities
        self.bond_indices = bond_indices
        self.rest_lengths = rest_lengths
        self.alive = alive  # boolean mask, False = broken
        self.particle_radius = particle_radius
        self.density = density
        self.bond_stiffness = bond_stiffness
        # Breakage options used to build the field; the ledger re-identifies
        # bonds with the same options, so the "all face bonds broken" invariant
        # keeps its meaning.
        self.footprint_aware = bool(footprint_aware)
        self.on_plane_tolerance = float(on_plane_tolerance)
        # Vocabulary actually used by the caller, reported back by the ledger.
        self.mass_mapping = mass_mapping or (
            "lattice_cell" if mass_mode == "cell" else "equal_volume_sphere"
        )
        self.face_extent = face_extent or ("disc" if footprint_aware else "plane")
        self.domain_bounds = domain_bounds
    
    @property
    def num_particles(self) -> int:
        return len(self.positions)
    
    @property
    def particle_mass(self) -> float:
        """Mass of a single particle for the configured mass mapping."""
        return _particle_mass(self.density, self.particle_radius, self.mass_mode)

    @property
    def mass_radius(self) -> float:
        """Radius of the sphere that carries the particle mass.

        Equals the contact radius under the legacy ``sphere`` mapping and
        ``(6/pi)^(1/3) r = 1.240701 r`` under the volume-consistent ``cell``
        mapping. The geometric/contact radius is untouched either way.
        """
        if self.mass_mode == "cell":
            return self.particle_radius * MASS_RADIUS_FACTOR
        return self.particle_radius

    @property
    def particle_cell_volume(self) -> float:
        """Volume of the lattice cell owned by one particle, h^3."""
        return particle_cell_volume(self.particle_radius)

    @property
    def volume_fraction(self) -> float:
        """Mass volume of one particle over the cell volume it represents.

        1.0 for the volume-consistent ``cell`` mapping, ``pi/6 = 0.5236`` for
        the legacy inscribed-sphere mapping.
        """
        return (self.particle_mass / self.density) / self.particle_cell_volume

    @property
    def domain_volume(self) -> float | None:
        """Volume of the requested domain, when the config carried bounds."""
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
            released_energy=face_data['released_energy'],
            extent_radius=face_data.get('extent_radius'),
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

    Bonds connect nearest neighbours (distance <= 1.01 * 2 * particle_radius).
    The scan is blocked over particle index (``torch.cdist`` on a row block
    against the full set) instead of a Python double loop, so the cost is
    vectorised and stays usable for the 10^4-10^5 particle ice fields this
    generator is meant to produce. The resulting pair list is identical to the
    brute-force Python scan: pairs are unique, oriented ``i < j`` and sorted
    lexicographically by ``(i, j)``.
    """
    n_particles = len(positions)
    spacing = 2.0 * particle_radius
    cutoff = spacing * 1.01
    device = positions.device
    empty_pairs = torch.empty((0, 2), dtype=torch.long, device=device)
    empty_lengths = torch.empty(0, dtype=positions.dtype, device=device)
    if n_particles < 2:
        return empty_pairs, empty_lengths

    # Keep the distance block below ~20 M entries (160 MB in float64).
    block = max(1, min(n_particles, int(2.0e7 / n_particles) or 1))
    pair_i: list[torch.Tensor] = []
    pair_j: list[torch.Tensor] = []
    for start in range(0, n_particles, block):
        stop = min(start + block, n_particles)
        distances = torch.cdist(positions[start:stop], positions)
        rows, cols = torch.nonzero(distances <= cutoff, as_tuple=True)
        keep = cols > (rows + start)
        if bool(keep.any()):
            pair_i.append(rows[keep] + start)
            pair_j.append(cols[keep])

    if not pair_i:
        return empty_pairs, empty_lengths

    bond_indices = torch.stack(
        [torch.cat(pair_i), torch.cat(pair_j)], dim=1
    ).to(torch.long)
    # Lexicographic (i, j) order, matching the brute-force reference scan.
    order = torch.argsort(bond_indices[:, 0] * n_particles + bond_indices[:, 1])
    bond_indices = bond_indices[order]
    rest_lengths = torch.linalg.norm(
        positions[bond_indices[:, 1]] - positions[bond_indices[:, 0]], dim=1
    )
    return bond_indices, rest_lengths


def face_footprint_radius(face: FailureFace) -> float:
    """Radius (m) of the disc that stands in for the face footprint.

    Uses ``FailureFace.extent_radius`` when the producer supplies it (the FEM
    leg knows the cohesive-face extent) and otherwise the area-equivalent disc
    ``sqrt(area / pi)``. The frozen JSON contract is unchanged because the field
    is optional. Non-circular faces are represented by this disc: the bias is
    O(perimeter), see ``docs/fracture_gen_mass_footprint_audit.md`` §5.
    """
    if face.extent_radius is not None:
        if face.extent_radius < 0.0:
            raise ValueError(
                f"extent_radius must be non-negative, got {face.extent_radius!r}"
            )
        return float(face.extent_radius)
    if face.area <= 0.0:
        raise ValueError(f"area must be positive, got {face.area!r}")
    return math.sqrt(face.area / math.pi)


def identify_bonds_crossing_face(
    bond_indices: torch.Tensor,
    positions: torch.Tensor,
    face: FailureFace,
    *,
    footprint_aware: bool = False,
    on_plane_tolerance: float = 0.0,
    face_extent: str | None = None,
) -> torch.Tensor:
    """Identify bonds that cross a failure face.

    A bond crosses a face if the two endpoints are on opposite sides of the
    face plane (defined by normal and centroid).

    Two opt-in refinements are available and both default to the frozen
    behaviour:

    ``on_plane_tolerance``
        Half-width (m) of the band in which an endpoint counts as lying *on*
        the plane. With the strict sign test a cut plane coincident with a
        particle layer severs nothing, because one signed distance is exactly
        zero. With a small tolerance a bond is also severed when exactly one
        endpoint lies in the band (``2 n^2`` bonds for a layer-coincident cut);
        bonds lying inside the plane (both endpoints in the band) stay intact.
    ``footprint_aware``
        Only sever a bond if the cut point, projected onto the face plane, lies
        inside the footprint disc of radius :func:`face_footprint_radius`. The
        frozen criterion severs every bond crossing the *infinite* plane, which
        over-breaks a small finite face by up to the plane-area/footprint ratio.
    """
    if on_plane_tolerance < 0.0:
        raise ValueError(
            f"on_plane_tolerance must be non-negative, got {on_plane_tolerance!r}"
        )
    if face_extent is not None:
        if face_extent not in VALID_FACE_EXTENTS:
            raise ValueError(
                f"face_extent must be one of {VALID_FACE_EXTENTS}, got {face_extent!r}"
            )
        footprint_aware = face_extent == "disc"
    normal_np = face.normal.to(positions.device)
    if float(torch.linalg.norm(normal_np)) <= 0.0:
        raise ValueError("failure face normal must be non-zero")
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

    cut_points = None
    if on_plane_tolerance > 0.0:
        on_i = dist_i.abs() <= on_plane_tolerance
        on_j = dist_j.abs() <= on_plane_tolerance
        # exactly one endpoint inside the band -> the bond is severed there
        endpoint_cut = on_i ^ on_j
        crosses = crosses | endpoint_cut
        # cut point for the band case is the on-plane endpoint
        band_point = torch.where(on_i.unsqueeze(1), pos_i, pos_j)
        cut_points = band_point

    if footprint_aware:
        radius = face_footprint_radius(face)
        if cut_points is None:
            # interpolated zero crossing: dist_i + t (dist_j - dist_i) = 0
            denom = dist_i - dist_j
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            t = (dist_i / denom).unsqueeze(1)
            cut_points = pos_i + t * (pos_j - pos_i)
        offset = cut_points - centroid
        in_plane = offset - torch.sum(offset * normal, dim=1, keepdim=True) * normal
        inside = torch.linalg.norm(in_plane, dim=1) <= radius
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
        mass_mode=config.mass_mode,
        footprint_aware=config.footprint_aware,
        on_plane_tolerance=config.on_plane_tolerance,
        mass_mapping=config.mass_mapping,
        face_extent=config.face_extent,
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
    
    def lattice_volume(self) -> float:
        """Ice volume represented by the realised lattice, ``N * h^3`` (m^3)."""
        return self.field.represented_volume

    def mass_accounting(self, domain_volume: float | None = None) -> dict[str, Any]:
        """DEM mass against the continuum solid mass ``rho * domain_volume``.

        This is the *independent* baseline the frozen criterion 1 lacked: it
        compares the particle masses with the ice volume the field replaces, not
        with the same sphere formula used to build them. Returns the signed
        relative error and the platform gate flag (|error| <= 15 %).
        """
        if domain_volume is None:
            domain_volume = self.field.domain_volume
            if domain_volume is None:
                raise ValueError(
                    "domain_volume must be given when the field was generated "
                    "without domain_bounds"
                )
        if not (domain_volume > 0.0) or not math.isfinite(domain_volume):
            raise ValueError(
                f"domain_volume must be positive and finite, got {domain_volume!r}"
            )
        continuum_mass = self.field.density * float(domain_volume)
        dem_mass = self.field.total_mass
        relative_error = dem_mass / continuum_mass - 1.0
        return {
            "dem_mass": dem_mass,
            "dem_mass_kg": dem_mass,
            "continuum_mass": continuum_mass,
            "continuum_mass_kg": continuum_mass,
            "fem_continuum_mass": continuum_mass,
            "domain_volume_m3": float(domain_volume),
            "lattice_volume_m3": self.lattice_volume(),
            "lattice_volume_covered": self.lattice_volume(),
            "num_particles": self.field.num_particles,
            "particle_mass_kg": self.field.particle_mass,
            "volume_fraction": self.field.volume_fraction,
            "mass_mode": self.field.mass_mode,
            "mass_mapping": self.field.mass_mapping,
            "mass_radius_m": self.field.mass_radius,
            # signed: DEM mass minus continuum mass, relative to continuum
            "relative_error": relative_error,
            "within_15pct": abs(relative_error) <= 0.15,
        }

    def energy_accounting(self) -> dict[str, float]:
        """Account for fracture energy.
        
        Returns dict with:
        - total_released_energy: sum of released_energy from all faces
        - num_broken_bonds: number of broken bonds in DEM field
        - unit_bond_energy: total_released_energy / num_broken_bonds
        """
        total_released = sum(face.released_energy for face in self.faces)
        num_broken = self.field.num_broken_bonds
        
        if num_broken > 0:
            unit_bond_energy = total_released / num_broken
        else:
            unit_bond_energy = 0.0
        
        # Independent analytic baselines for the frozen criteria c2/c3:
        # a face of area A severs A/s^2 bonds of a lattice with spacing s, and
        # a unit bond carries G*s^2 of fracture energy.
        spacing_sq = (2.0 * self.field.particle_radius) ** 2
        analytic_bond_count = sum(face.area / spacing_sq for face in self.faces)
        analytic_unit_energy = (
            sum(face.G * face.area for face in self.faces) / analytic_bond_count
            if analytic_bond_count > 0.0
            else 0.0
        )
        unit_bond_relative_error = (
            abs(unit_bond_energy / analytic_unit_energy - 1.0)
            if analytic_unit_energy > 0.0
            else 0.0
        )
        bond_count_relative_error = (
            abs(num_broken - analytic_bond_count) / analytic_bond_count
            if analytic_bond_count > 0.0
            else 0.0
        )
        input_consistency_error = sum(
            abs(face.released_energy - face.G * face.area) for face in self.faces
        )

        return {
            'total_released_energy': total_released,
            'num_broken_bonds': num_broken,
            'unit_bond_energy': unit_bond_energy,
            'analytic_bond_count': analytic_bond_count,
            'analytic_unit_bond_energy': analytic_unit_energy,
            'unit_bond_energy_relative_error': unit_bond_relative_error,
            'bond_count_relative_error': bond_count_relative_error,
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
