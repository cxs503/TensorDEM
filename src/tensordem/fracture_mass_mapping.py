"""Mass / volume mapping audit for the FEM->DEM fracture-field generator.

Why this module exists
----------------------
``tensordem.fracture_gen`` fills a domain with particles on a *simple cubic*
lattice of spacing ``h = 2 r`` and documents its mass mapping as
"equal-volume spheres":

    m_p = rho * 4/3 pi r^3,      M_DEM = N * m_p

A particle on that lattice, however, *owns* a cubic Voronoi cell of volume
``h^3 = 8 r^3`` (the lattice covers the domain once per cell).  The sphere of
radius ``r`` is inscribed in that cube, not equal to it, so the ratio of the
summed particle volume to the ice volume the lattice represents is exactly the
simple-cubic packing fraction of equal spheres:

    V_sphere / V_cell = (4/3 pi r^3) / (2 r)^3 = pi / 6 = 0.523598775598...

i.e. the legacy mapping under-counts the mass of the represented ice volume by
``1 - pi/6 = 47.64 %``.  For any ship-ice problem this is a first-order bias:
inertia, added-mass ratio, floating draught (Archimedes) and the kinetic-energy
share of the ice field are all 52.4 % of their intended value.

Two internally consistent mappings are therefore supported explicitly:

``sphere``  (legacy, frozen SI-2 contract)
    ``m_p = rho * 4/3 pi r^3``.  Only consistent if the lattice spacing is
    ``h = (4 pi / 3)^{1/3} r = 1.61199 r`` (a *loose* lattice), or if the user
    accepts that the field represents 52.4 % of the geometric ice volume.

``cell``    (volume consistent, recommended)
    ``m_p = rho * h^3`` with ``h = 2 r``, i.e. the Voronoi/cell mass.  This is
    the standard DEM practice of separating the *contact* radius from the mass
    (O'Sullivan, *Particulate Discrete Element Modelling*, 2011, ch. 3): the
    contact radius stays ``r`` so that neighbours do not overlap, while the mass
    is taken from the volume of the region each particle represents.  With this
    mapping ``M_DEM = rho * N * h^3`` reproduces ``rho * V_domain`` exactly
    whenever the domain is an integer multiple of ``h``.

An equivalent reading of the ``cell`` mapping is a *volume-equivalent radius*
``r_eq = (3 h^3 / 4 pi)^{1/3} = 0.6203505 h``; ``equal_volume_radius`` returns it.

Secondary geometric bias (quantisation)
---------------------------------------
Particles are placed at ``xmin + r + k h < xmax`` so the number of particles per
axis is ``N = floor((W - r)/h) + 1``.  Unless ``W`` is an integer multiple of
``h`` the lattice covers ``N h`` instead of ``W``, giving an unavoidable
relative volume error ``(N h - W) / W``.  ``volume_quantisation_error`` reports
it analytically so it is never silently mixed into the material-mapping
residual.

All quantities are SI (kg, m, s).
"""
from __future__ import annotations

import math
from typing import Any

#: Volume ratio of an inscribed sphere to its simple-cubic cell: pi / 6.
SIMPLE_CUBIC_PACKING_FRACTION = math.pi / 6.0

#: Legacy sphere-mapping mass deficit of the represented ice volume.
SPHERE_MAPPING_DEFICIT = 1.0 - SIMPLE_CUBIC_PACKING_FRACTION

#: Volume ratio of a contact sphere to its hexagonal-close-packed (HCP) cell:
#: ``(4/3 pi r^3) / (4 sqrt(2) r^3) = pi / (3 sqrt 2) = 0.7404804896930611``.
HCP_PACKING_FRACTION = math.pi / (3.0 * math.sqrt(2.0))

#: Volume of the HCP cell owned by one contact sphere, ``4 sqrt(2) r^3``.
HCP_CELL_VOLUME_PER_RADIUS_CUBED = 4.0 * math.sqrt(2.0)

#: Accepted lattice packings of the fracture-field generator.
#: ``"sc"``  : simple cubic, spacing ``h = 2 r``, coordination number 6
#: ``"hcp"`` : hexagonal close packed (ABAB), nearest distance ``2 r``,
#:            coordination number 12
VALID_PACKINGS = ("sc", "hcp")

#: Accepted ``packing`` spellings -> canonical value.
PACKING_ALIASES = {
    "sc": "sc",
    "simple_cubic": "sc",
    "simple-cubic": "sc",
    "simplecubic": "sc",
    "cubic": "sc",
    "hcp": "hcp",
    "hexagonal": "hcp",
    "hexagonal_close_packed": "hcp",
    "hexagonal-close-packed": "hcp",
    "abab": "hcp",
}

VALID_MASS_MODES = ("sphere", "cell")

#: Spacing / radius ratio at which an inscribed sphere would be volume equal.
EQUIVALENT_SPACING_PER_RADIUS = (4.0 * math.pi / 3.0) ** (1.0 / 3.0)


def normalise_packing(value: str, name: str = "packing") -> str:
    """Canonicalise a lattice-packing spelling onto ``VALID_PACKINGS``."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    key = value.strip().lower()
    if key not in PACKING_ALIASES:
        raise ValueError(
            f"{name} must be one of {sorted(PACKING_ALIASES)} "
            f"(canonical values {VALID_PACKINGS}), got {value!r}"
        )
    return PACKING_ALIASES[key]


def packing_fraction(packing: str = "sc") -> float:
    """Volume fraction of equal contact spheres in the requested lattice.

    ``sc`` -> ``pi/6`` (0.5235987755982988); ``hcp`` -> ``pi/(3 sqrt 2)``
    (0.7404804896930611).  Both are exact closed forms (Gauss' 1831 sphere
    packing result for the close-packed density).
    """
    packing = normalise_packing(packing)
    if packing == "hcp":
        return HCP_PACKING_FRACTION
    return SIMPLE_CUBIC_PACKING_FRACTION


def expected_lattice_bond_lengths(packing: str = "sc") -> dict[str, float]:
    """Closed-form lattice lengths (in units of the contact radius ``r``).

    * ``sc``  : only one bond length, ``2 r``; cell edge ``2 r``.
    * ``hcp`` : nearest-neighbour distance ``2 r`` (in-plane and interlayer are
      equal by construction), in-plane row pitch ``sqrt(3) r``, interlayer
      pitch ``2 sqrt(2/3) r = 1.632993 r``.

    A touching-sphere HCP lattice is the ABAB stacking whose nearest-neighbour
    distance equals the sphere diameter, so the *contact* radius and the *cell*
    volume decouple exactly as in the cubic case.
    """
    packing = normalise_packing(packing)
    if packing == "hcp":
        return {
            "bond_length": 2.0,
            "in_plane_row_pitch": math.sqrt(3.0),
            "interlayer_pitch": 2.0 * math.sqrt(2.0 / 3.0),
            "cell_volume": HCP_CELL_VOLUME_PER_RADIUS_CUBED,
        }
    return {
        "bond_length": 2.0,
        "in_plane_row_pitch": 2.0,
        "interlayer_pitch": 2.0,
        "cell_volume": 8.0,
    }


def packing_coordination_number(packing: str = "sc") -> int:
    """Nearest-neighbour count of the bulk lattice (6 for sc, 12 for hcp)."""
    packing = normalise_packing(packing)
    return 12 if packing == "hcp" else 6


def mass_radius_factor(packing: str = "sc") -> float:
    """``r_eq / r``: radius of the sphere carrying the cell mass at the field density.

    ``sc``  -> ``2 (3/4pi)^{1/3} = 1.240701`` (cell ``8 r^3``),
    ``hcp`` -> ``(3 sqrt 2 / pi)^{1/3} = 1.105634`` (cell ``4 sqrt 2 r^3``).
    """
    return (3.0 / (4.0 * math.pi) * expected_lattice_bond_lengths(packing)["cell_volume"]) ** (1.0 / 3.0)


def lattice_spacing(particle_radius: float) -> float:
    """Nearest-neighbour spacing of the simple cubic lattice built by fracture_gen."""
    if particle_radius <= 0.0:
        raise ValueError("particle_radius must be positive")
    return 2.0 * particle_radius


def particle_cell_volume(particle_radius: float, packing: str = "sc") -> float:
    """Volume of the Voronoi cell owned by one lattice particle.

    ``sc``  : ``h^3`` with ``h = 2 r`` (``8 r^3``).
    ``hcp`` : ``4 sqrt(2) r^3`` (the close-packed cell; note the *contact* radius
              is unchanged, only the volume each particle represents shrinks).
    """
    packing = normalise_packing(packing)
    if packing == "hcp":
        return HCP_CELL_VOLUME_PER_RADIUS_CUBED * particle_radius ** 3
    h = lattice_spacing(particle_radius)
    return h ** 3


def particle_sphere_volume(particle_radius: float) -> float:
    """Volume of the inscribed sphere, ``4/3 pi r^3``."""
    if particle_radius <= 0.0:
        raise ValueError("particle_radius must be positive")
    return (4.0 / 3.0) * math.pi * particle_radius ** 3


def sphere_to_cell_volume_ratio() -> float:
    """``V_sphere / V_cell = pi/6`` for the as-built simple cubic lattice."""
    return SIMPLE_CUBIC_PACKING_FRACTION


def equal_volume_radius(spacing: float) -> float:
    """Radius whose sphere has the same volume as a cube of side ``spacing``.

    ``r_eq = (3 h^3 / (4 pi))^{1/3} = 0.62035 h``.
    """
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    return (3.0 * spacing ** 3 / (4.0 * math.pi)) ** (1.0 / 3.0)


#: Absolute tolerance used when an axis extent is (numerically) an exact
#: integer multiple of the lattice spacing; keeps the closed form aligned with
#: ``torch.arange(xmin + r, xmax, 2 r)``, which excludes the upper bound.
_AXIS_COUNT_TOL = 1e-9


def lattice_axis_count(width: float, particle_radius: float) -> int:
    """Analytic number of particles along one axis of extent ``width``.

    ``generate_particle_lattice`` places particles at ``r + k h < W``, so

        ``N = ceil((W - r) / h)``

    where an extent that is numerically an integer multiple of ``h`` counts the
    integer itself (the upper bound is excluded by the half-open range).  The
    tolerance ``_AXIS_COUNT_TOL`` absorbs floating-point noise; extents shorter
    than one particle radius return ``0`` and are rejected by the generator.
    """
    if width < 0.0:
        raise ValueError("width must be non-negative")
    h = lattice_spacing(particle_radius)
    ratio = (width - particle_radius) / h
    if ratio <= -_AXIS_COUNT_TOL:
        return 0
    return max(0, int(math.ceil(ratio - _AXIS_COUNT_TOL)))


def expected_lattice_counts(
    domain_bounds: tuple[float, float, float, float, float, float],
    particle_radius: float,
) -> tuple[int, int, int]:
    """Analytic particle counts ``(Nx, Ny, Nz)`` for the fracture_gen lattice."""
    xmin, xmax, ymin, ymax, zmin, zmax = domain_bounds
    return (
        lattice_axis_count(xmax - xmin, particle_radius),
        lattice_axis_count(ymax - ymin, particle_radius),
        lattice_axis_count(zmax - zmin, particle_radius),
    )


def domain_volume(domain_bounds: tuple[float, float, float, float, float, float]) -> float:
    """Geometric volume of the target domain, ``m^3``."""
    xmin, xmax, ymin, ymax, zmin, zmax = domain_bounds
    return abs((xmax - xmin) * (ymax - ymin) * (zmax - zmin))


def lattice_cell_volume(
    domain_bounds: tuple[float, float, float, float, float, float],
    particle_radius: float,
) -> float:
    """Volume represented by the populated lattice cells, ``N * h^3``."""
    nx, ny, nz = expected_lattice_counts(domain_bounds, particle_radius)
    return nx * ny * nz * particle_cell_volume(particle_radius)


def volume_quantisation_error(
    domain_bounds: tuple[float, float, float, float, float, float],
    particle_radius: float,
) -> float:
    """Relative error ``(N h^3 - V_domain) / V_domain`` of the lattice coverage."""
    v_domain = domain_volume(domain_bounds)
    if v_domain <= 0.0:
        raise ValueError("domain volume must be positive")
    return (lattice_cell_volume(domain_bounds, particle_radius) - v_domain) / v_domain


def analytic_bond_count(counts: tuple[int, int, int]) -> int:
    """Nearest-neighbour bond count of a periodic-free simple cubic lattice."""
    nx, ny, nz = counts
    return nx * ny * (nz - 1) + nx * (ny - 1) * nz + (nx - 1) * ny * nz


def particle_mass(
    density: float,
    particle_radius: float,
    mass_mode: str = "cell",
    packing: str = "sc",
) -> float:
    """Particle mass in kg for the requested mapping and lattice packing.

    ``mass_mode='cell'``   -> ``density * V_cell(packing)`` (volume consistent)
    ``mass_mode='sphere'`` -> ``density * 4/3 pi r^3`` (legacy frozen contract)

    ``packing`` changes the cell volume (``8 r^3`` for sc, ``4 sqrt 2 r^3`` for
    hcp) and therefore the ``cell`` mass; the ``sphere`` mass is
    packing-independent by definition.
    """
    if mass_mode not in VALID_MASS_MODES:
        raise ValueError(
            f"mass_mode must be one of {VALID_MASS_MODES}, got {mass_mode!r}"
        )
    if density <= 0.0:
        raise ValueError("density must be positive")
    if mass_mode == "cell":
        return density * particle_cell_volume(particle_radius, packing)
    return density * particle_sphere_volume(particle_radius)


def mass_mapping_report(
    density: float,
    particle_radius: float,
    domain_bounds: tuple[float, float, float, float, float, float],
    mass_mode: str = "cell",
    fem_mass: float | None = None,
    tolerance: float = 0.01,
    num_particles: int | None = None,
) -> dict[str, Any]:
    """Audit report for one (density, radius, domain) configuration.

    The reference mass is ``fem_mass`` when supplied (e.g. from TensorFEM), and
    otherwise ``density * V_domain``.  Returns the realised DEM mass, the
    relative error, its decomposition into the *material mapping* term and the
    *lattice quantisation* term, and a pass/fail against ``tolerance``.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    nx, ny, nz = expected_lattice_counts(domain_bounds, particle_radius)
    derived = nx * ny * nz
    if num_particles is None:
        n_particles = derived
        count_source = "analytic_lattice"
    else:
        n_particles = int(num_particles)
        if n_particles < 0:
            raise ValueError("num_particles must be non-negative")
        count_source = "caller"
    v_domain = domain_volume(domain_bounds)
    v_lattice = n_particles * particle_cell_volume(particle_radius)
    total_mass = n_particles * particle_mass(density, particle_radius, mass_mode)

    reference_mass = density * v_domain if fem_mass is None else float(fem_mass)
    if reference_mass <= 0.0:
        raise ValueError("reference mass must be positive")

    rel_error = (total_mass - reference_mass) / reference_mass
    quantisation = (v_lattice - v_domain) / v_domain
    material = total_mass / (density * v_lattice) - 1.0

    return {
        "protocol": "tensordem-fracture-mass-mapping-v1",
        "mass_mode": mass_mode,
        "density_kg_m3": density,
        "particle_radius_m": particle_radius,
        "lattice_spacing_m": lattice_spacing(particle_radius),
        "domain_bounds_m": list(domain_bounds),
        "lattice_counts": [nx, ny, nz],
        "num_particles": n_particles,
        "num_particles_source": count_source,
        "domain_volume_m3": v_domain,
        "lattice_volume_m3": v_lattice,
        "particle_mass_kg": particle_mass(density, particle_radius, mass_mode),
        "dem_total_mass_kg": total_mass,
        "reference_mass_kg": reference_mass,
        "reference_source": "fem_mass" if fem_mass is not None else "density*domain_volume",
        "relative_mass_error": rel_error,
        "lattice_quantisation_relative_error": quantisation,
        "material_mapping_relative_error": material,
        "tolerance": tolerance,
        "within_tolerance": abs(rel_error) <= tolerance,
        "sphere_to_cell_volume_ratio": sphere_to_cell_volume_ratio(),
        "equal_volume_radius_m": equal_volume_radius(lattice_spacing(particle_radius)),
        "analytic_bond_count": analytic_bond_count((nx, ny, nz)),
    }


def resolution_sweep(
    density: float,
    domain_bounds: tuple[float, float, float, float, float, float],
    particles_per_axis: tuple[int, ...] = (2, 3, 4, 5, 6),
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Sweep lattice resolution and report both mappings' mass errors.

    ``particles_per_axis`` is translated to ``r = W / (2 n)`` per axis using the
    smallest axis, so ``W = 2 n r`` is an exact integer multiple of ``h`` and the
    quantisation term is zero (isolating the material-mapping term).
    """
    xmin, xmax, ymin, ymax, zmin, zmax = domain_bounds
    width = min(xmax - xmin, ymax - ymin, zmax - zmin)
    rows = []
    for n in particles_per_axis:
        if n < 1:
            raise ValueError("particles_per_axis entries must be >= 1")
        radius = width / (2.0 * n)
        cell = mass_mapping_report(density, radius, domain_bounds, "cell", tolerance=tolerance)
        sphere = mass_mapping_report(density, radius, domain_bounds, "sphere", tolerance=tolerance)
        rows.append(
            {
                "particles_per_axis": n,
                "particle_radius_m": radius,
                "num_particles": cell["num_particles"],
                "cell_mode_relative_error": cell["relative_mass_error"],
                "cell_mode_within_tolerance": cell["within_tolerance"],
                "sphere_mode_relative_error": sphere["relative_mass_error"],
                "sphere_mode_within_tolerance": sphere["within_tolerance"],
                "lattice_quantisation_relative_error": cell[
                    "lattice_quantisation_relative_error"
                ],
            }
        )
    return {
        "protocol": "tensordem-fracture-mass-mapping-sweep-v1",
        "density_kg_m3": density,
        "domain_bounds_m": list(domain_bounds),
        "tolerance": tolerance,
        "sphere_to_cell_volume_ratio": sphere_to_cell_volume_ratio(),
        "cases": rows,
    }