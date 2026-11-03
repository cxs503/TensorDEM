"""Directional elastic constants of the simple-cubic bonded-sphere lattice.

``IceDEM3D`` is a central-force network of axial springs on a simple-cubic lattice
augmented with the twelve edge diagonals, and its module docstring states that the
directional stiffness "must be calibrated before engineering use". This module
supplies that calibration in closed form and a finite-sample measurement that can
be checked against it:

* :func:`periodic_elastic_constants` -- exact affine (Born) homogenisation of the
  bond network over the thirteen half-space neighbours used by the solver. The
  second-order elastic tensor follows directly from

      C_ijkl = (1 / a^3) * sum_bonds k * l^2 * n_i n_j n_k n_l

  where the sum runs over the thirteen bonds owned by one lattice node (the
  1/2 in the strain energy is cancelled by the 26/13 bond sharing), ``l`` is the
  bond rest length and ``n`` the bond unit vector. Because every bond is an axial
  spring, the tensor automatically obeys the Cauchy relation ``C12 = C44``.

* :func:`relaxed_uniaxial_modulus` -- the engineering Young's modulus of a finite
  sample: axial displacement prescribed on the two end faces, transverse
  displacements free, rigid-body modes removed. For a cubic crystal this is the
  uniaxial-stress modulus ``C11 - 2 C12^2 / (C11 + C12)``, which is measurably
  lower than the affine ``C11``.

No empirical coefficient is introduced anywhere; every number is a property of
the solver's own bond topology and ``bond_stiffness``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

#: The thirteen neighbour offsets owned by one node of the solver lattice
#: (six face neighbours plus twelve edge diagonals, half-space filtered exactly
#: as ``IceDEM3D`` builds them: dz in {0,1}, dy in {-1,0,1}, dx in {-1,0,1}).
HALF_SPACE_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dz in (0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dx == dy == dz == 0)
    and (dz > 0 or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0))
)


def lattice_spacing(radius: float) -> float:
    """Solver lattice spacing: neighbouring particle centres are two radii apart."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    return 2.0 * radius


def lattice_positions(nx: int, ny: int, nz: int, spacing: float) -> np.ndarray:
    """Particle centres of the solver's box lattice, centred on the origin."""
    for name, value in (("nx", nx), ("ny", ny), ("nz", nz)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 3:
            raise ValueError(f"{name} must be an integer >= 3")
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    grid = np.stack(
        np.meshgrid(
            np.arange(nx) - (nx - 1) / 2.0,
            np.arange(ny) - (ny - 1) / 2.0,
            np.arange(nz) - (nz - 1) / 2.0,
            indexing="ij",
        ),
        axis=-1,
    )
    return grid.reshape(-1, 3) * spacing


def lattice_bonds(nx: int, ny: int, nz: int) -> np.ndarray:
    """Bond index pairs, identical to the topology built by ``IceDEM3D``."""
    positions = lattice_positions(nx, ny, nz, 1.0)
    index = {(round(x), round(y), round(z)): n for n, (x, y, z) in
             enumerate(positions + np.array([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2]))}
    pairs: list[tuple[int, int]] = []
    for (ix, iy, iz) in [(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)]:
        for dx, dy, dz in HALF_SPACE_OFFSETS:
            jx, jy, jz = ix + dx, iy + dy, iz + dz
            if 0 <= jx < nx and 0 <= jy < ny and 0 <= jz < nz:
                a, b = index[(ix, iy, iz)], index[(jx, jy, jz)]
                pairs.append((min(a, b), max(a, b)))
    return np.asarray(sorted(pairs), dtype=np.int64)


def periodic_elastic_constants(stiffness: float, spacing: float) -> dict[str, float]:
    """Exact affine (Born) second-order elastic constants of the bond network.

    Returns the cubic constants ``C11``, ``C12``, ``C44`` in pascals together with
    the anisotropic ``C13``/``C66`` cross-checks (cubic symmetry demands
    ``C13 = C12`` and ``C66 = C44``).
    """
    if not math.isfinite(stiffness) or stiffness <= 0:
        raise ValueError("stiffness must be finite and positive")
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")

    offsets = np.asarray(HALF_SPACE_OFFSETS, dtype=float)
    direction = np.linalg.norm(offsets, axis=1)
    lengths = direction * spacing
    normals = offsets / direction[:, None]
    weights = stiffness * lengths**2

    def c(i: int, j: int, k: int, l: int) -> float:
        return float(np.sum(weights * normals[:, i] * normals[:, j]
                            * normals[:, k] * normals[:, l])) / spacing**3

    return {
        "C11_Pa": c(0, 0, 0, 0),
        "C22_Pa": c(1, 1, 1, 1),
        "C33_Pa": c(2, 2, 2, 2),
        "C12_Pa": c(0, 0, 1, 1),
        "C13_Pa": c(0, 0, 2, 2),
        "C44_Pa": c(0, 1, 0, 1),
        "C55_Pa": c(1, 2, 1, 2),
        "C66_Pa": c(0, 2, 0, 2),
        "bond_stiffness_N_per_m": stiffness,
        "lattice_spacing_m": spacing,
        "bond_count_per_node": float(len(HALF_SPACE_OFFSETS)),
    }


def derived_moduli(c11: float, c12: float, c44: float) -> dict[str, float]:
    """Engineering moduli and anisotropy measures of a cubic elastic tensor."""
    if c11 + c12 <= 0:
        raise ValueError("C11 + C12 must be positive")
    bulk = (c11 + 2.0 * c12) / 3.0
    young = c11 - 2.0 * c12**2 / (c11 + c12)
    poisson = c12 / (c11 + c12)
    # Isotropic-equivalent shear: Voigt is C44, Reuss is the compliancy-based bound.
    shear_voigt = c44
    shear_reuss = 5.0 * c44 * (c11 - c12) / (3.0 * c11 - 4.0 * c12 + 4.0 * c44) if (
        3.0 * c11 - 4.0 * c12 + 4.0 * c44) else float("nan")
    return {
        "E_Pa": young,
        "nu": poisson,
        "G_voigt_Pa": shear_voigt,
        "G_reuss_Pa": shear_reuss,
        "K_Pa": bulk,
        "zener_anisotropy": 2.0 * c44 / (c11 - c12) if c11 != c12 else float("inf"),
        "cauchy_residual_Pa": c12 - c44,
        "isotropic_poisson_from_cauchy": c12 / (c11 + c12),
    }


def relaxed_uniaxial_modulus(
    positions: np.ndarray,
    pairs: np.ndarray,
    stiffness: float,
    *,
    axis: int = 0,
    strain: float = 1e-4,
    control: str = "displacement",
    applied_stress: float = 1.0e4,
) -> dict[str, object]:
    """Uniaxial modulus of a finite sample with free lateral faces.

    Two protocols are supported because both are used by DEM practice:

    ``control="displacement"``
        Platen style: the axial degree of freedom of every node on both end faces
        is prescribed to a uniform displacement. The end faces stay planes, which
        suppresses shear relaxation near the platens.
    ``control="load"``
        Load style: the lower end face is held axially, a uniform axial traction
        is applied to the upper face, and every lateral degree of freedom is free.

    In both cases the lateral faces are free, so a finite sample reports the
    uniaxial modulus (not ``C11``). Boundary constraints make the measured value
    exceed the infinite-crystal modulus; the bias is quantified by the audit.
    """
    positions = np.asarray(positions, dtype=float)
    pairs = np.asarray(pairs, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (B, 2)")
    if not math.isfinite(stiffness) or stiffness <= 0:
        raise ValueError("stiffness must be finite and positive")
    if not math.isfinite(strain) or strain <= 0:
        raise ValueError("strain must be finite and positive")
    if control not in ("displacement", "load"):
        raise ValueError("control must be 'displacement' or 'load'")
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")

    n = len(positions)
    delta = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    length = np.linalg.norm(delta, axis=1)
    normal = delta / length[:, None]
    element = stiffness * normal[:, :, None] * normal[:, None, :]

    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for bond, (i, j) in enumerate(pairs):
        dofs = np.array([3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2])
        block = np.block([[element[bond], -element[bond]],
                          [-element[bond], element[bond]]])
        rows.append(np.repeat(dofs, 6))
        columns.append(np.tile(dofs, 6))
        values.append(block.ravel())
    k_global = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(3 * n, 3 * n),
    ).tocsr()

    axial = positions[:, axis]
    extent = float(axial.max() - axial.min())
    if extent <= 0:
        raise ValueError("sample has no extent along the loading axis")
    low_face = np.isclose(axial, axial.min())
    high_face = np.isclose(axial, axial.max())
    low_nodes = np.where(low_face)[0]
    high_nodes = np.where(high_face)[0]

    fixed = np.zeros(3 * n, dtype=bool)
    fixed[3 * low_nodes + axis] = True
    if control == "displacement":
        fixed[3 * high_nodes + axis] = True
    # Remove rigid-body modes normal to the axis without disturbing the stress state.
    for node in (low_nodes[0], low_nodes[-1]):
        for dof in range(3):
            if dof != axis:
                fixed[3 * node + dof] = True
    free = np.where(~fixed)[0]

    load = np.zeros(3 * n)
    transverse_axes = [d for d in (0, 1, 2) if d != axis]
    area = 1.0
    for dof in transverse_axes:
        area *= float(positions[:, dof].max() - positions[:, dof].min())
    prescribed_strain = float(strain)
    if control == "displacement":
        load[3 * high_nodes + axis] = prescribed_strain * extent
    else:
        load[3 * high_nodes + axis] = applied_stress * area / len(high_nodes)
        prescribed_strain = float("nan")

    displacement = np.where(fixed, load, 0.0)
    displacement[free] = sparse_linalg.spsolve(
        k_global[free][:, free],
        load[free] - k_global[free][:, fixed] @ displacement[fixed],
    )
    force = k_global @ displacement

    if control == "displacement":
        axial_strain = prescribed_strain
        modulus = float(force[3 * high_nodes + axis].sum()) / (area * axial_strain)
    else:
        axial_strain = float(displacement[3 * high_nodes + axis].mean()) / extent
        modulus = applied_stress / axial_strain

    poisson_components = []
    for dof in transverse_axes:
        high = np.isclose(positions[:, dof], positions[:, dof].max())
        low = np.isclose(positions[:, dof], positions[:, dof].min())
        span = float(positions[high, dof].mean() - positions[low, dof].mean())
        change = (displacement[3 * np.where(high)[0] + dof].mean()
                  - displacement[3 * np.where(low)[0] + dof].mean())
        poisson_components.append(-(change / span) / axial_strain)

    return {
        "E_Pa": float(modulus),
        "control": control,
        "poisson_ratio_measured": float(np.mean(poisson_components)),
        "poisson_components": [float(value) for value in poisson_components],
        "axial_strain": float(axial_strain),
        "axial_extent_m": extent,
        "cross_section_area_m2": area,
        "particle_count": int(n),
        "bond_count": int(len(pairs)),
        "max_free_dof_force_N": float(np.abs(force[free]).max()) if free.size else 0.0,
        "displacement_m": displacement.reshape(-1, 3),
        "force_N": force.reshape(-1, 3),
    }


def bond_stiffness_for_target_modulus(target_E: float, spacing: float,
                                      modulus_factor: float) -> float:
    """Bond stiffness that would give ``target_E`` with ``modulus_factor = E/(k/a)``."""
    for name, value in (("target_E", target_E), ("spacing", spacing),
                        ("modulus_factor", modulus_factor)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return target_E * spacing / modulus_factor
