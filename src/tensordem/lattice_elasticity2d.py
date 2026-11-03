"""Exact small-strain (Cauchy-Born) elasticity of the 2-D bonded-disk lattice.

Why this module exists
----------------------
``IceDEM`` (the two-dimensional bonded-disk engine used by the hull/ridge
benchmarks in ``docs/``) builds its lattice as *square-grid nearest neighbours
plus face diagonals* inside a cutoff of ``sqrt(2) * diameter``, i.e. an
eight-neighbour central-force network with a single ``bond_stiffness``.  The
three-dimensional sibling already has an affine-elasticity audit
(``docs/dem3d_lattice_elasticity.md``, ``docs/dem3d_lattice_elastic_constants.md``)
that converts ``bond_stiffness`` into engineering moduli and quantifies the
directional bias, but the two-dimensional engine -- the one the ship-ice
benchmarks actually use -- had **no reference modulus at all**.  Consequences:

* ``bond_stiffness`` could not be related to a Young's modulus;
* the specimen-direction bias was only qualitative (no number attached);
* nothing recorded that four-fold (square) bond symmetry is *not* sufficient
  for an isotropic 2-D continuum: it admits a cubic-type anisotropy with an
  independent shear constant.

This module supplies that missing reference.  For a central-force bond network
with bond stiffness ``k_b``, rest length ``l_b`` and unit direction ``n_b``, the
affine (uniform-strain, Cauchy-Born) stiffness tensor is

    C_ijkl = (1/A) * sum_b k_b l_b**2 n_i n_j n_k n_l ,

with ``A`` the area per particle (Born & Huang, *Dynamical Theory of Crystal
Lattices*, 1954).  For the eight-neighbour square lattice, counting the two
axis bonds and two face-diagonal bonds of one unit cell (area ``A = a**2``),
this collapses to the closed form

    C11 = C22 = k_a + k_d,      C12 = k_d,      C66 = k_d,

where ``k_a`` is the axis-bond stiffness and ``k_d`` the diagonal-bond
stiffness.  With the engine default ``k_d = k_a`` this gives
``C11 = 2 k_a``, ``C12 = C66 = k_a``.

Isotropy condition (this is the actionable part)
------------------------------------------------
Two-dimensional isotropy of a *central-force* network requires the Cauchy
relation ``C12 = C66`` (always true here) **and** ``C11 - C12 = 2 C66``.  The
engine default violates the second one by a factor of two.  It is restored
exactly by

    k_d = k_a / 2      (``ISOTROPIC_DIAGONAL_SCALE``),

which yields ``C11 = 1.5 k_a``, ``C12 = C66 = 0.5 k_a``, plane-stress
``E = 4 k_a / 3`` and ``nu = 1/3`` in every in-plane direction.  The default
lattice instead has ``E(45 deg)/E(0 deg) = 8/5`` (i.e. 60 % stiffer on the
diagonal) and a Poisson ratio that runs from ``0.20`` to ``0.50`` with the
loading direction.

Because ``bond_stiffness`` is a free modelling number (no measured ice value
fixes it), the isotropic variant costs nothing physically: it is the same
network with one scalar changed, and it is expressible without touching the
engine through :class:`tensordem.calibrated_dem.CalibratedIceDEM`, which takes
explicit per-bond stiffnesses.  :func:`isotropic_bond_stiffness` returns the
per-bond coefficient array for exactly that backend.

Conventions
-----------
SI units, plane stress, per unit out-of-plane thickness: ``C`` is in N/m, so a
traction (force per unit width) is ``sigma = C : epsilon`` and the
Pa-equivalent modulus is ``C / thickness``.  Strain is the symmetric small
strain tensor ``epsilon``; ``C66`` multiplies the engineering shear strain
``gamma = 2 * epsilon_xy``.

Every routine is deterministic and torch-based; no NumPy round trip is
required.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import torch

#: The eight neighbour directions of the engine lattice, in units of the
#: particle spacing ``a = 2 * radius``.
CANONICAL_NEIGHBOURS: tuple[tuple[int, int], ...] = (
    (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1),
)

#: Diagonal-bond stiffness scale used by ``IceDEM`` (all bonds share one k).
DEFAULT_DIAGONAL_SCALE = 1.0

#: Exact isotropy condition for the eight-neighbour square lattice.
ISOTROPIC_DIAGONAL_SCALE = 0.5


def _as_tensor(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.to(dtype=torch.float64)


def canonical_bonds(
    spacing: float,
    *,
    bond_stiffness: float = 1.0,
    diagonal_scale: float = DEFAULT_DIAGONAL_SCALE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(normals, rest_lengths, stiffness)`` of the eight lattice bonds.

    ``normals`` has shape ``(8, 2)``, ``rest_lengths`` and ``stiffness`` have
    shape ``(8,)``.  The four face diagonals carry ``bond_stiffness *
    diagonal_scale``.
    """
    if not (math.isfinite(spacing) and spacing > 0):
        raise ValueError("spacing must be finite and positive")
    if not (math.isfinite(bond_stiffness) and bond_stiffness > 0):
        raise ValueError("bond_stiffness must be finite and positive")
    if not (math.isfinite(diagonal_scale) and diagonal_scale > 0):
        raise ValueError("diagonal_scale must be finite and positive")
    normals, lengths, stiffness = [], [], []
    for dx, dy in CANONICAL_NEIGHBOURS:
        direction = torch.tensor([float(dx), float(dy)], dtype=torch.float64)
        is_diagonal = abs(dx) + abs(dy) == 2
        normals.append(direction / torch.linalg.vector_norm(direction))
        lengths.append(spacing * (math.sqrt(2.0) if is_diagonal else 1.0))
        stiffness.append(bond_stiffness * (diagonal_scale if is_diagonal else 1.0))
    return (
        torch.stack(normals),
        torch.tensor(lengths, dtype=torch.float64),
        torch.tensor(stiffness, dtype=torch.float64),
    )


def stiffness_components(
    bond_stiffness: float,
    diagonal_scale: float = DEFAULT_DIAGONAL_SCALE,
) -> dict[str, float]:
    """Closed-form in-plane stiffness constants (N/m, per unit thickness).

    ``C11 = C22 = k_a + k_d``, ``C12 = C66 = k_d`` with ``k_d = k_a *
    diagonal_scale``.  The per-particle area cancels because every bond length
    is a multiple of the spacing ``a`` and ``A = a**2``.
    """
    if not (math.isfinite(bond_stiffness) and bond_stiffness > 0):
        raise ValueError("bond_stiffness must be finite and positive")
    if not (math.isfinite(diagonal_scale) and diagonal_scale > 0):
        raise ValueError("diagonal_scale must be finite and positive")
    k_a = float(bond_stiffness)
    k_d = k_a * float(diagonal_scale)
    return {
        "c11": k_a + k_d,
        "c22": k_a + k_d,
        "c12": k_d,
        "c66": k_d,
        "axis_stiffness": k_a,
        "diagonal_stiffness": k_d,
    }


def voigt_sum_stiffness(
    normals: torch.Tensor,
    rest_lengths: torch.Tensor,
    stiffness: torch.Tensor,
    per_particle_area: float,
    *,
    weight: float = 1.0,
) -> torch.Tensor:
    """Generic Cauchy-Born summation ``(1/A) sum_b k_b l_b**2 n^4``.

    This is the *definition*, used here as an independent route to the closed
    form.  ``weight`` lets callers rescale the result (``0.5`` reproduces the
    "average over particles of a half neighbour sum" normalisation used when
    the bond list is one particle's neighbour sum instead of one unit cell).
    """
    normals = _as_tensor(normals)
    rest_lengths = _as_tensor(rest_lengths)
    stiffness = _as_tensor(stiffness)
    if normals.ndim != 2 or normals.shape[1] != 2:
        raise ValueError("normals must have shape (M, 2)")
    if rest_lengths.shape != normals.shape[:1] or stiffness.shape != normals.shape[:1]:
        raise ValueError("rest_lengths and stiffness must have shape (M,)")
    if not (math.isfinite(per_particle_area) and per_particle_area > 0):
        raise ValueError("per_particle_area must be finite and positive")
    weight_tensor = stiffness * rest_lengths.square() * weight
    return torch.einsum(
        "m,mi,mj,mk,ml->ijkl", weight_tensor, normals, normals, normals, normals
    ) / per_particle_area


def affine_stiffness_tensor(
    bond_stiffness: float,
    diagonal_scale: float = DEFAULT_DIAGONAL_SCALE,
    *,
    spacing: float = 1.0,
) -> torch.Tensor:
    """Affine stiffness tensor of the ideal lattice, shape ``(2, 2, 2, 2)``."""
    normals, rest_lengths, stiffness = canonical_bonds(
        spacing, bond_stiffness=bond_stiffness, diagonal_scale=diagonal_scale
    )
    # One unit cell holds two axis and two face-diagonal bonds; the neighbour
    # sum above lists four axis + four diagonal directions, so each is counted
    # once with weight 1/2.
    return voigt_sum_stiffness(
        normals, rest_lengths, stiffness, spacing * spacing, weight=0.5
    )


def rotate_stiffness(stiffness: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Rotate a fourth-order in-plane tensor into a frame turned by ``angle_rad``."""
    stiffness = _as_tensor(stiffness)
    if tuple(stiffness.shape) != (2, 2, 2, 2):
        raise ValueError("stiffness must have shape (2, 2, 2, 2)")
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    rotation = torch.tensor([[c, s], [-s, c]], dtype=torch.float64)
    return torch.einsum(
        "ai,bj,ck,dl,abcd->ijkl", rotation, rotation, rotation, rotation, stiffness
    )


def plane_stress_moduli(stiffness: torch.Tensor) -> dict[str, float]:
    """Young's modulus and Poisson ratio for loading along the local 1-axis.

    Returns the traction modulus (N/m) and the Pa-equivalent value needs the
    out-of-plane thickness: ``E / thickness``.
    """
    stiffness = _as_tensor(stiffness)
    c11 = float(stiffness[0, 0, 0, 0])
    c12 = float(stiffness[0, 0, 1, 1])
    c22 = float(stiffness[1, 1, 1, 1])
    if c11 <= 0 or c22 <= 0:
        raise ValueError("stiffness tensor is not positive definite")
    return {
        "youngs_modulus_N_per_m": c11 - c12 * c12 / c22,
        "poisson_ratio": c12 / c22,
        "shear_modulus_N_per_m": float(stiffness[0, 1, 0, 1]),
    }


def directional_moduli(
    theta_deg: Sequence[float],
    bond_stiffness: float,
    diagonal_scale: float = DEFAULT_DIAGONAL_SCALE,
    *,
    spacing: float = 1.0,
) -> list[dict[str, float]]:
    """Sweep the uniaxial (plane-stress) response over loading direction."""
    stiffness = affine_stiffness_tensor(
        bond_stiffness, diagonal_scale, spacing=spacing
    )
    rows = []
    for theta in theta_deg:
        rotated = rotate_stiffness(stiffness, math.radians(float(theta)))
        moduli = plane_stress_moduli(rotated)
        rows.append(
            {
                "theta_deg": float(theta),
                "youngs_modulus_N_per_m": moduli["youngs_modulus_N_per_m"],
                "poisson_ratio": moduli["poisson_ratio"],
                "coupling_c16": float(rotated[0, 0, 0, 1]),
            }
        )
    return rows


def anisotropy_report(
    bond_stiffness: float,
    diagonal_scale: float = DEFAULT_DIAGONAL_SCALE,
    *,
    spacing: float = 1.0,
    thickness: float = 1.0,
    n_theta: int = 181,
) -> dict[str, float]:
    """Quantify the direction dependence of the lattice's elastic response.

    ``zener_ratio`` is ``2 C66 / (C11 - C12)`` (exactly 1 for an isotropic
    central-force 2-D network); ``modulus_ratio`` is ``max E / min E`` over the
    first quadrant (``8/5`` for the engine default).
    """
    if n_theta < 5:
        raise ValueError("n_theta must be >= 5")
    components = stiffness_components(bond_stiffness, diagonal_scale)
    sweep = directional_moduli(
        [180.0 * i / (n_theta - 1) for i in range(n_theta)],
        bond_stiffness,
        diagonal_scale,
        spacing=spacing,
    )
    youngs = [row["youngs_modulus_N_per_m"] for row in sweep]
    poisson = [row["poisson_ratio"] for row in sweep]
    e_min = min(youngs)
    e_max = max(youngs)
    e_at_0 = sweep[0]["youngs_modulus_N_per_m"]
    e_at_90 = sweep[-1]["youngs_modulus_N_per_m"]
    e_at_45 = directional_moduli(
        [45.0], bond_stiffness, diagonal_scale, spacing=spacing
    )[0]["youngs_modulus_N_per_m"]
    isotropic = ISOTROPIC_DIAGONAL_SCALE
    report = {
        "diagonal_scale": float(diagonal_scale),
        "isotropic_diagonal_scale": isotropic,
        "c11_N_per_m": components["c11"],
        "c12_N_per_m": components["c12"],
        "c66_N_per_m": components["c66"],
        "zener_ratio_2d": 2.0 * components["c66"] / (components["c11"] - components["c12"]),
        "youngs_modulus_0deg_N_per_m": e_at_0,
        "youngs_modulus_45deg_N_per_m": e_at_45,
        "youngs_modulus_90deg_N_per_m": e_at_90,
        "youngs_modulus_max_N_per_m": e_max,
        "youngs_modulus_min_N_per_m": e_min,
        "modulus_ratio_max_min": e_max / e_min,
        "modulus_ratio_45_over_0": e_at_45 / e_at_0,
        "poisson_ratio_min": min(poisson),
        "poisson_ratio_max": max(poisson),
        "theta_at_max_modulus_deg": next(
            row["theta_deg"]
            for row in sorted(sweep, key=lambda r: r["theta_deg"])
            if row["youngs_modulus_N_per_m"] >= e_max * (1.0 - 1e-9)
        ),
        "coupling_c16_max_abs": max(abs(row["coupling_c16"]) for row in sweep),
        "youngs_modulus_pa_at_0deg": e_at_0 / thickness,
        "thickness_m": float(thickness),
    }
    return report


def bond_rest_lengths(sim: Any) -> torch.Tensor:
    """Rest lengths of the *bonded* pairs only (``sim.rest_lengths`` is pair-wise)."""
    rest = _as_tensor(sim.rest_lengths)
    bonded = getattr(sim, "bonded", None)
    if bonded is not None and rest.numel() != len(getattr(sim, "bond_indices")):
        rest = rest[torch.as_tensor(bonded).to(torch.bool)]
    return rest


def isotropic_bond_stiffness(
    sim: Any, *, axis_stiffness: float | None = None
) -> torch.Tensor:
    """Per-bond stiffness making an :class:`IceDEM` lattice exactly isotropic.

    Axis bonds keep ``axis_stiffness`` (default: the simulation's
    ``config.bond_stiffness``); face diagonals get half of it.  The returned
    array has shape ``(len(sim.bond_indices),)`` and can be handed straight to
    :class:`tensordem.calibrated_dem.CalibratedIceDEM` as ``bond_stiffness=``.
    """
    rest = bond_rest_lengths(sim)
    if rest.numel() == 0:
        raise ValueError("simulation has no bonds")
    spacing = float(rest.min())
    axis = (
        float(sim.config.bond_stiffness)
        if axis_stiffness is None
        else float(axis_stiffness)
    )
    if not (math.isfinite(axis) and axis > 0):
        raise ValueError("axis_stiffness must be finite and positive")
    is_diagonal = rest > spacing * (1.0 + 1e-9)
    scale = torch.where(
        is_diagonal,
        torch.full_like(rest, ISOTROPIC_DIAGONAL_SCALE),
        torch.ones_like(rest),
    )
    return scale * axis


def bulk_stiffness_from_sim(sim: Any) -> dict[str, Any]:
    """Affine stiffness of the solver's *own* bond network (route 3).

    Only fully coordinated interior particles are averaged, so the free-surface
    bond deficit does not bias the result.  Per-bond stiffness comes from
    ``sim.coefficients`` when the simulation is a ``CalibratedIceDEM`` and from
    ``config.bond_stiffness`` otherwise, i.e. from whatever law the engine will
    actually integrate.
    """
    if not hasattr(sim, "bond_indices"):
        raise TypeError("sim must be an IceDEM-like object with bond_indices")
    bond_indices = sim.bond_indices
    if bond_indices.numel() == 0:
        raise ValueError("simulation has no bonds")
    rest = bond_rest_lengths(sim)
    reference = _as_tensor(sim.reference_bond_vectors)
    spacing = float(rest.min())
    per_particle_area = spacing * spacing
    stiffness = getattr(sim, "coefficients", None)
    if stiffness is None:
        stiffness = torch.full_like(rest, float(sim.config.bond_stiffness))
    else:
        stiffness = _as_tensor(stiffness)
    normals = reference / rest[:, None]
    n_particles = len(sim.positions)
    neighbour_tensor = torch.einsum(
        "m,mi,mj,mk,ml->mijkl", stiffness * rest.square(), normals, normals, normals, normals
    )
    accumulated = torch.zeros((n_particles, 2, 2, 2, 2), dtype=torch.float64)
    i, j = bond_indices.T
    accumulated.index_add_(0, i, neighbour_tensor)
    accumulated.index_add_(0, j, neighbour_tensor)
    coordination = torch.zeros(n_particles, dtype=torch.float64)
    coordination.index_add_(0, i, torch.ones(len(i), dtype=torch.float64))
    coordination.index_add_(0, j, torch.ones(len(j), dtype=torch.float64))
    interior = coordination == coordination.max()
    if not bool(interior.any()):
        raise ValueError("simulation has no fully coordinated interior particle")
    tensor = accumulated[interior].mean(dim=0) * 0.5 / per_particle_area
    return {
        "stiffness": tensor,
        "interior_particles": int(interior.sum()),
        "coordination_max": int(coordination.max()),
        "coordination_bulk_fraction": float(interior.to(torch.float64).mean()),
        "spacing_m": spacing,
    }


def missing_bond_deficit(sim: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bonds a *bulk* particle would have but the free surface removes.

    Returns ``(normals, rest_lengths, stiffness)`` for every absent canonical
    neighbour, each counted once.  Used to turn the total bond energy of a
    finite patch back into a bulk strain-energy density.
    """
    cfg = sim.config
    nx, ny = int(cfg.nx), int(cfg.ny)
    if nx * ny != len(sim.positions):
        raise ValueError("simulation is not a full nx-by-ny grid; deficit undefined")
    spacing = 2.0 * float(cfg.radius)
    rest = bond_rest_lengths(sim)
    diagonal = rest > spacing * (1.0 + 1e-9)
    stiffness_axis = float(cfg.bond_stiffness)
    coefficients = getattr(sim, "coefficients", None)
    if coefficients is not None and bool(diagonal.any()):
        stiffness_diag = float(_as_tensor(coefficients)[diagonal].max())
    else:
        stiffness_diag = stiffness_axis
    normals, lengths, stiffness = [], [], []
    for index in range(nx * ny):
        iy, ix = divmod(index, nx)
        for dx, dy in CANONICAL_NEIGHBOURS:
            nxp, nyp = ix + dx, iy + dy
            if 0 <= nxp < nx and 0 <= nyp < ny:
                continue
            direction = torch.tensor([float(dx), float(dy)], dtype=torch.float64)
            is_diagonal = abs(dx) + abs(dy) == 2
            normals.append(direction / torch.linalg.vector_norm(direction))
            lengths.append(spacing * (math.sqrt(2.0) if is_diagonal else 1.0))
            stiffness.append(stiffness_diag if is_diagonal else stiffness_axis)
    if normals:
        return (
            torch.stack(normals),
            torch.tensor(lengths, dtype=torch.float64),
            torch.tensor(stiffness, dtype=torch.float64),
        )
    return (
        torch.zeros((0, 2), dtype=torch.float64),
        torch.zeros((0,), dtype=torch.float64),
        torch.zeros((0,), dtype=torch.float64),
    )


def applied_strain_energy_density(
    sim: Any, strain: torch.Tensor, *, include_deficit: bool = True
) -> float:
    """Strain-energy density of the engine's affine deformation by ``strain``.

    The deformation ``F = I + strain`` is applied to the solver's own
    ``initial_positions``; the total bond potential is read back from
    ``sim.mechanical_energy()`` (the engine's own energy routine).  Because a
    finite patch is missing surface bonds, the known canonical deficit
    (equated analytically through the same kinematic law) is added back so the
    result is a *bulk* density ``0.5 * C_ijkl * eps_ij * eps_kl``.
    """
    strain = _as_tensor(strain)
    if tuple(strain.shape) != (2, 2):
        raise ValueError("strain must have shape (2, 2)")
    original = sim.positions.clone()
    try:
        deformation = torch.eye(2, dtype=torch.float64) + strain
        sim.positions = torch.as_tensor(sim.initial_positions) @ deformation.T
        total = float(sim.mechanical_energy()["bond_J"])
        if include_deficit:
            normals, lengths, stiffness = missing_bond_deficit(sim)
            if normals.numel():
                # The deficit is a list of missing *neighbour slots*: each
                # absent slot would be one end of a bond, and every real bond is
                # counted once in ``bond_J``, so the missing energy is half the
                # slot sum.
                stretched = torch.linalg.vector_norm(normals @ deformation.T, dim=1)
                extension = lengths * (stretched - 1.0)
                total += 0.5 * float((0.5 * stiffness * extension.square()).sum())
        area = len(sim.positions) * (2.0 * float(sim.config.radius)) ** 2
        return total / area
    finally:
        sim.positions = original


def stiffness_from_strain_energy(
    sim: Any, *, step: float = 1.0e-6, include_deficit: bool = True
) -> torch.Tensor:
    """Recover the full in-plane stiffness tensor from the engine's energy.

    Central second differences of :func:`applied_strain_energy_density` along
    the three independent strain modes and one mixed mode.  ``step`` must stay
    far below the engine's ``breaking_strain`` so the response is elastic and no
    bond fails.
    """
    if not (math.isfinite(step) and 0 < step < 1e-3):
        raise ValueError("step must be finite and in (0, 1e-3)")
    if step > 0.1 * float(sim.config.breaking_strain):
        raise ValueError("step must be well below the engine breaking strain")

    def density(e11: float, e22: float, e12: float) -> float:
        strain = torch.tensor([[e11, e12], [e12, e22]], dtype=torch.float64)
        return applied_strain_energy_density(
            sim, strain, include_deficit=include_deficit
        )

    h = step
    zero = density(0.0, 0.0, 0.0)
    # u(e) = 0.5 * (C11 e11^2 + C22 e22^2 + 2 C12 e11 e22 + 4 C66 e12^2).
    c11 = (density(h, 0.0, 0.0) - 2.0 * zero + density(-h, 0.0, 0.0)) / (h * h)
    c22 = (density(0.0, h, 0.0) - 2.0 * zero + density(0.0, -h, 0.0)) / (h * h)
    c12 = (
        density(h, h, 0.0) - density(h, 0.0, 0.0)
        - density(0.0, h, 0.0) + zero
    ) / (h * h)
    c66 = 0.25 * (
        density(0.0, 0.0, h) - 2.0 * zero + density(0.0, 0.0, -h)
    ) / (h * h)
    tensor = torch.zeros((2, 2, 2, 2), dtype=torch.float64)
    tensor[0, 0, 0, 0] = c11
    tensor[1, 1, 1, 1] = c22
    tensor[0, 0, 1, 1] = c12
    tensor[1, 1, 0, 0] = c12
    tensor[0, 1, 0, 1] = c66
    tensor[1, 0, 0, 1] = c66
    tensor[0, 1, 1, 0] = c66
    tensor[1, 0, 1, 0] = c66
    return tensor


def tensor_error(measured: torch.Tensor, reference: torch.Tensor) -> float:
    """Relative Frobenius error between two fourth-order tensors."""
    measured = _as_tensor(measured)
    reference = _as_tensor(reference)
    scale = torch.linalg.vector_norm(reference)
    if float(scale) == 0.0:
        raise ValueError("reference tensor is zero")
    return float(torch.linalg.vector_norm(measured - reference) / scale)


__all__ = [
    "CANONICAL_NEIGHBOURS",
    "DEFAULT_DIAGONAL_SCALE",
    "ISOTROPIC_DIAGONAL_SCALE",
    "affine_stiffness_tensor",
    "anisotropy_report",
    "applied_strain_energy_density",
    "bulk_stiffness_from_sim",
    "bond_rest_lengths",
    "canonical_bonds",
    "directional_moduli",
    "isotropic_bond_stiffness",
    "missing_bond_deficit",
    "plane_stress_moduli",
    "rotate_stiffness",
    "stiffness_components",
    "stiffness_from_strain_energy",
    "tensor_error",
    "voigt_sum_stiffness",
]