"""Periodic (k = 0) Born-Huang internal-strain analysis of central-force lattices.

Motivation
----------
``docs/dem3d_lattice_elastic_constants.md`` reports affine (Born) cubic constants
for the ``IceDEM3D`` bond lattice and, in its section 4, a "relaxed" (non-affine)
response measured on finite specimens with free lateral faces. It concludes there
that the relaxed shear stiffness is "strongly non-affine" and that the affine
Zener anisotropy is "mostly an artifact", and it records one explicit gap in
section 6:

    "The rigorous alternative is a periodic k = 0 Born-Huang internal-strain
     calculation; it is not implemented here."

This module implements exactly that calculation and answers it.

Theory
------
For a Born-von Karman periodic cell (``M`` atoms per primitive cell, cell matrix
``cell``, undirected bond list with rest vectors ``r``, unit normals ``n`` and
stiffness ``k``), a homogeneous strain ``E`` maps the lattice to the affine
configuration ``u_i = E x_i``. Expanding the bond energy

    Phi = 0.5 * sum_bonds k * (|r + u_j - u_i| - |r|)^2

to second order around the affine state gives

    Phi(u) = Phi_affine + 0.5 * u^T K u - f^T u,

with the k = 0 dynamical (Hessian) matrix built from ``k n n`` blocks and the
affine force ``f_i = sum_{bonds at i} k |r| (n.E.n) n`` (``n`` pointing away from
``i``). The non-affine relaxation is ``u* = K^+ f`` and the internal-strain tensor
is ``Xi = du*/dE``. Hence

    Phi_relaxed = Phi_affine - 0.5 * f^T u*  <=  Phi_affine,

so the affine constants are an *upper bound* on the relaxed ones and the two
coincide **iff** the affine state is force balanced (``f = 0``).

Key structural fact used below: every atom of a Bravais lattice with a one-atom
basis sits at a centre of inversion (``+n`` and ``-n`` bonds carry equal weights
and opposite unit normals), so ``f = sum_26 k |r| (n.E.n) n`` cancels *exactly*
for every ``E``. A one-atom-basis central-force lattice therefore has
``Xi = 0`` identically: no internal strain, and relaxed bulk constants equal the
affine ones. Non-zero internal strain requires more than one atom per primitive
cell with inequivalent bonding; the 1-D diatomic chain is the smallest such case
and is validated here to machine precision against its closed form.

Contents
--------
* :func:`solver_half_space_offsets` / :func:`solver_supercell_bonds` -- the exact
  26-neighbour bond set of ``IceDEM3D`` replicated on a periodic torus.
* :func:`diatomic_chain_bonds` -- two-atom-basis control lattice with a closed
  form (relaxed modulus = harmonic mean of the two spring constants), used as a
  negative control so the engine cannot pass by being a silent no-op.
* :func:`bonded_lattice_analysis` -- affine constants, affine-force residual,
  k = 0 Hessian, internal-strain tensor and relaxed constants for any periodic
  bond list.
* :func:`free_surface_specimen` -- the *finite* free-surface protocol of the
  earlier audit, re-implemented with a sparse solve so the size ladder can be
  extended and fitted with more than one functional form.

Scope
-----
Small-strain linear elasticity of an intact bond network. No fracture, contacts,
dynamics, damping or ice-material claim. Every number is a property of the
solver's bond topology and ``bond_stiffness`` alone.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Iterable, Sequence

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

__all__ = [
    "SOLVER_HALF_SPACE_OFFSETS",
    "solver_half_space_offsets",
    "solver_all_offsets",
    "solver_supercell_bonds",
    "diatomic_chain_bonds",
    "diatomic_chain_closed_form",
    "bonded_lattice_analysis",
    "free_surface_specimen",
    "fit_surface_model",
    "STRAIN_MODES",
]

#: The thirteen half-space neighbour offsets owned by one node of the solver
#: lattice, rebuilt here independently of ``dem3d_lattice_elastic`` so that the
#: identity between the two sets is a *test*, not an assumption.
SOLVER_HALF_SPACE_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dz in (0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dx == dy == dz == 0)
    and (dz > 0 or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0))
)

#: Unit strain tensors used as probes. ``constant = 2 * energy_density`` for
#: these normalisations, i.e. ``xx`` -> C11, ``biaxial`` -> C11 + C12 and
#: ``shear`` (engineering shear gamma = 1, tensor component 1/2) -> C44.
STRAIN_MODES: dict[str, np.ndarray] = {
    "xx": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    "yy": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    "biaxial": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    "shear": np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
}


def solver_half_space_offsets() -> np.ndarray:
    """Thirteen half-space offsets as an integer array, shape ``(13, 3)``."""
    return np.array(SOLVER_HALF_SPACE_OFFSETS, dtype=np.int64)


def solver_all_offsets() -> np.ndarray:
    """All twenty-six neighbour offsets as an integer array, shape ``(26, 3)``."""
    offsets = [
        (dx, dy, dz)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dx == dy == dz == 0)
    ]
    return np.array(offsets, dtype=np.int64)


def _supercell_positions(supercell: Sequence[int], cell: np.ndarray) -> np.ndarray:
    """Cell origins in the node order used by :func:`_node_index` (i fastest)."""
    nx, ny, nz = supercell
    grid = [(i, j, k) for k in range(nz) for j in range(ny) for i in range(nx)]
    return np.asarray(grid, dtype=float) @ np.asarray(cell, dtype=float)


def _node_index(supercell: Sequence[int], ijk: np.ndarray) -> int:
    nx, ny, nz = supercell
    i, j, k = (int(v) % n for v, n in zip(ijk, (nx, ny, nz)))
    return (k * ny + j) * nx + i


def solver_supercell_bonds(
    spacing: float, supercell: Sequence[int] = (3, 3, 3), stiffness: float = 1.0
) -> dict[str, Any]:
    """Undirected bond list of the solver's 26-neighbour lattice on a torus.

    ``supercell`` counts primitive cells per axis; ``3`` is the smallest value
    that keeps the ``+n`` and ``-n`` images of a direction distinct on the torus
    (with ``2`` the ``(1,1,0)`` and ``(-1,-1,0)`` bonds would collapse onto the
    same node pair and the symmetry check would be weakened).
    """
    for name, value in (("spacing", spacing), ("stiffness", stiffness)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    supercell = tuple(int(v) for v in supercell)
    if len(supercell) != 3 or any(v < 3 for v in supercell):
        raise ValueError("supercell must hold three integers >= 3")
    cell = np.eye(3) * float(spacing)
    positions = _supercell_positions(supercell, cell)
    offsets = solver_all_offsets()
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    vectors: list[np.ndarray] = []
    for node, ijk in enumerate(
        [(i, j, k) for k in range(supercell[2]) for j in range(supercell[1])
         for i in range(supercell[0])]
    ):
        for offset in offsets:
            other = _node_index(supercell, np.array(ijk) + offset)
            key = (min(node, other), max(node, other))
            if key in seen:
                # Each physical bond is emitted once per endpoint direction; the
                # duplicate must carry the opposite vector.
                continue
            seen.add(key)
            pairs.append(key)
            if other == node:  # pragma: no cover - excluded by supercell >= 3
                raise ValueError("self bond on torus; increase supercell")
            vectors.append(offset * float(spacing) if node < other else -offset * float(spacing))
    return {
        "positions": positions,
        "pairs": np.array(pairs, dtype=np.int64),
        "vectors": np.array(vectors, dtype=float),
        "cell": cell,
        "cells": int(np.prod(supercell)),
        "stiffness": np.full(len(pairs), float(stiffness)),
        "coordination_number": int(len(offsets)),
        "bonds_per_cell": len(pairs) / float(np.prod(supercell)),
    }


def diatomic_chain_bonds(
    bond_length: float = 1.0,
    stiffness_one: float = 2.0,
    stiffness_two: float = 1.0,
    cells: int = 3,
    transverse_stiffness: float = 0.0,
) -> dict[str, Any]:
    """Two-atom-basis chain of alternating springs (internal-strain control).

    Cell matrix ``diag(2 * bond_length, bond_length, bond_length)`` with atoms at
    ``A = (0, 0, 0)`` and ``B = (bond_length, 0, 0)``. The bond ``A -> B`` has
    stiffness ``stiffness_one`` and ``B -> A(+1 cell)`` has ``stiffness_two``.
    With ``transverse_stiffness > 0`` same-species springs are added along ``y``
    and ``z``, which turn the object into a genuine three-dimensional lattice
    without changing the axial internal-strain answer (those springs are inert
    under ``eps_xx``).
    """
    for name, value in (("bond_length", bond_length), ("stiffness_one", stiffness_one),
                        ("stiffness_two", stiffness_two)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(transverse_stiffness) or transverse_stiffness < 0:
        raise ValueError("transverse_stiffness must be finite and non-negative")
    cells = int(cells)
    if cells < 3:
        raise ValueError("cells must be >= 3 so that +-pi images stay distinct")
    length = float(bond_length)
    cell = np.diag([2.0 * length, length, length])
    basis = np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]])
    # (atom_a, atom_b, cell_shift) triples.
    # A(i) -> B(i) with k1, and B(i) -> A(i + 1) with k2 (both of length L).
    specs: list[tuple[int, int, tuple[int, int, int]]] = [
        (0, 1, (0, 0, 0)),
        (1, 0, (1, 0, 0)),
    ]
    stiffness: list[float] = [float(stiffness_one), float(stiffness_two)]
    if transverse_stiffness > 0:
        for shift in ((0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            for atom in (0, 1):
                specs.append((atom, atom, shift))
                stiffness.append(float(transverse_stiffness))
    supercell = (cells, 3 if transverse_stiffness > 0 else 1,
                 3 if transverse_stiffness > 0 else 1)
    positions = []
    for k in range(supercell[2]):
        for j in range(supercell[1]):
            for i in range(supercell[0]):
                for atom in range(2):
                    positions.append(np.array([i, j, k], dtype=float) @ cell + basis[atom])
    positions = np.array(positions)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    vectors: list[np.ndarray] = []
    per_bond: list[float] = []
    for k in range(supercell[2]):
        for j in range(supercell[1]):
            for i in range(supercell[0]):
                base = ((k * supercell[1] + j) * supercell[0] + i) * 2
                for (atom_a, atom_b, shift), k_bond in zip(specs, stiffness):
                    ijk_b = np.array([i, j, k]) + np.array(shift)
                    wrapped = [ijk_b[0] % supercell[0], ijk_b[1] % supercell[1],
                               ijk_b[2] % supercell[2]]
                    node_a = base + atom_a
                    node_b = ((wrapped[2] * supercell[1] + wrapped[1]) * supercell[0]
                              + wrapped[0]) * 2 + atom_b
                    if node_a == node_b:  # pragma: no cover - transverse on torus
                        raise ValueError("self bond in diatomic control lattice")
                    key = (min(node_a, node_b), max(node_a, node_b))
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append((min(node_a, node_b), max(node_a, node_b)))
                    # Unwrapped bond vector in the direction ``min -> max``; using
                    # the wrapped coordinates would pick the long way round the
                    # torus for the ``-1`` cell shift.
                    vector = basis[atom_b] + np.asarray(shift, dtype=float) @ cell - basis[atom_a]
                    vectors.append(vector if node_a < node_b else -vector)
                    per_bond.append(k_bond)
    return {
        "positions": positions,
        "pairs": np.array(pairs, dtype=np.int64),
        "vectors": np.array(vectors, dtype=float),
        "cell": cell,
        "cells": int(np.prod(supercell)),
        "stiffness": np.array(per_bond, dtype=float),
    }


def diatomic_chain_closed_form(
    bond_length: float = 1.0, stiffness_one: float = 2.0, stiffness_two: float = 1.0
) -> dict[str, float]:
    """Closed form of the diatomic chain under uniaxial strain.

    With ``u`` the relaxation of atom ``A`` (``B`` moving by ``-u`` so that the
    cell centre of mass is fixed) and ``eps`` the axial strain, the two bond
    lengths are ``L (1 + eps) +- 2 u``; stationarity gives

        u_A / eps = L (k1 - k2) / (2 (k1 + k2))
        C11_relaxed = 2 k1 k2 / ((k1 + k2) L)        (harmonic mean)
        C11_affine  = (k1 + k2) / (2 L)              (arithmetic mean)

    both per unit cross-section ``L^2`` (cell volume ``2 L^3``).

    Note the relaxed/affine ratio ``4 k1 k2 / (k1 + k2)^2`` is strictly below one
    unless ``k1 = k2``: this is a genuine internal-strain system.
    """
    k1, k2, length = float(stiffness_one), float(stiffness_two), float(bond_length)
    return {
        "internal_strain_per_unit_strain": length * (k1 - k2) / (2.0 * (k1 + k2)),
        "C11_relaxed_Pa": 2.0 * k1 * k2 / ((k1 + k2) * length),
        "C11_affine_Pa": (k1 + k2) / (2.0 * length),
        "ratio_relaxed_to_affine": 4.0 * k1 * k2 / (k1 + k2) ** 2,
    }


def _hessian(n_atoms: int, pairs: np.ndarray, normals: np.ndarray,
             stiffness: np.ndarray) -> sparse.csr_matrix:
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for bond, (i, j) in enumerate(pairs):
        n = normals[bond]
        block = stiffness[bond] * np.outer(n, n)
        dofs_a = 3 * i + np.arange(3)
        dofs_b = 3 * j + np.arange(3)
        for (a, b, sign) in ((dofs_a, dofs_a, 1.0), (dofs_b, dofs_b, 1.0),
                             (dofs_a, dofs_b, -1.0), (dofs_b, dofs_a, -1.0)):
            rows.append(np.repeat(a, 3))
            cols.append(np.tile(b, 3))
            vals.append(sign * block.ravel())
    matrix = sparse.coo_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * n_atoms, 3 * n_atoms),
    ).tocsr()
    return matrix


def bonded_lattice_analysis(
    network: dict[str, Any],
    modes: Iterable[str] = ("xx", "biaxial", "shear"),
) -> dict[str, Any]:
    """Affine vs k = 0 relaxed constants, plus the internal-strain tensor.

    ``network`` is any dict with ``positions``, ``pairs``, ``vectors``, ``cell``,
    ``cells`` and ``stiffness`` (see :func:`solver_supercell_bonds` and
    :func:`diatomic_chain_bonds`). Unit strain probes are used throughout, so a
    constant is simply ``2 * energy_density``.
    """
    positions = np.asarray(network["positions"], dtype=float)
    pairs = np.asarray(network["pairs"], dtype=np.int64)
    vectors = np.asarray(network["vectors"], dtype=float)
    cell = np.asarray(network["cell"], dtype=float)
    cells = int(network["cells"])
    stiffness = np.asarray(network["stiffness"], dtype=float)
    if len(stiffness) == 1 and len(pairs) > 1:
        stiffness = np.full(len(pairs), float(stiffness[0]))
    length = np.linalg.norm(vectors, axis=1)
    normals = vectors / length[:, None]
    volume = abs(float(np.linalg.det(cell))) * cells

    def energy_density(extension: np.ndarray) -> float:
        return float(0.5 * (stiffness * extension**2).sum() / volume)

    hessian = _hessian(len(positions), pairs, normals, stiffness)
    results: dict[str, Any] = {
        "atoms": int(len(positions)),
        "bonds": int(len(pairs)),
        "cells": cells,
        "volume_m3": volume,
        "bonds_per_cell": len(pairs) / float(cells),
        "modes": {},
    }

    for mode in modes:
        strain = STRAIN_MODES[mode]
        # Affine bond extension ``l * (n . E . n)``, evaluated from the (unwrapped)
        # bond vectors: on a torus the node coordinates differ by lattice vectors,
        # so ``E (x_j - x_i)`` must not be taken from the wrapped positions.
        extension = np.einsum("ij,jk,ik->i", vectors, strain, normals)
        affine_energy = energy_density(extension)
        # Force on every atom in the affine state, n pointing away from the atom.
        force = np.zeros_like(positions)
        np.add.at(force, pairs[:, 0], (stiffness * extension)[:, None] * normals)
        np.add.at(force, pairs[:, 1], -(stiffness * extension)[:, None] * normals)
        relaxation = sparse_linalg.lsqr(
            hessian, force.ravel(), atol=1e-18, btol=1e-18, iter_lim=20000, conlim=1e12
        )[0].reshape(-1, 3)
        relaxed_energy = affine_energy - 0.5 * float((force * relaxation).sum()) / volume
        scale = float((stiffness * np.abs(extension) * length).sum() / len(pairs))
        results["modes"][mode] = {
            "affine_constant_Pa": 2.0 * affine_energy,
            "relaxed_constant_Pa": 2.0 * relaxed_energy,
            "ratio_relaxed_to_affine": relaxed_energy / affine_energy if affine_energy > 0 else float("nan"),
            "affine_force_residual_N": float(np.linalg.norm(force, axis=1).max()),
            "affine_force_scale_N": scale,
            "affine_force_relative": float(np.linalg.norm(force, axis=1).max() / scale)
            if scale > 0 else float("nan"),
            "internal_strain_max_m": float(np.linalg.norm(relaxation, axis=1).max()),
            "internal_strain_per_unit_strain": (
                relaxation / float(np.abs(strain).max())),
            "max_bond_extension_affine_m": float(np.abs(extension).max()),
        }
    return results


def free_surface_specimen(
    size: int,
    radius: float = 0.025,
    stiffness: float = 2000.0,
    mode: str = "shear",
    strain: float = 1e-5,
) -> dict[str, Any]:
    """Free-surface specimen stiffness, sparse version of the earlier audit.

    Reproduces the protocol of ``dem3d_lattice_elastic.relaxed_patch``: the
    driven faces follow the affine field, all other faces are free, and the
    constant comes from the *interior* (core) energy density so that affine and
    relaxed values are read with the same estimator. ``shear`` drives the y faces
    (``u_x = strain * y``) and ``xx`` drives the x faces.
    """
    if mode not in ("xx", "shear"):
        raise ValueError("mode must be 'xx' or 'shear'")
    size = int(size)
    if size < 3:
        raise ValueError("size must be >= 3")
    spacing = 2.0 * float(radius)
    positions = _supercell_positions((size, size, size), np.eye(3) * spacing)
    # Centred, exactly as the solver builds its box lattice.
    positions = positions - positions.mean(axis=0)
    index = {tuple(np.round(p / spacing).astype(int)): n for n, p in enumerate(positions)}
    coords = np.round(positions / spacing).astype(int)
    pairs: list[tuple[int, int]] = []
    for node, ijk in enumerate(coords):
        for offset in SOLVER_HALF_SPACE_OFFSETS:
            other = index.get(tuple(ijk + np.array(offset)))
            if other is not None:
                pairs.append((node, other))
    pairs = np.array(pairs, dtype=np.int64)
    vectors = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    length = np.linalg.norm(vectors, axis=1)
    normals = vectors / length[:, None]
    hessian = _hessian(len(positions), pairs, normals, np.full(len(pairs), float(stiffness)))

    strain_tensor = STRAIN_MODES[mode]
    affine = (positions - positions.min(axis=0)) @ strain_tensor.T
    # Protocol of the merged audit (``relaxed_patch(boundary="yz_faces")`` pins
    # axis 1 only, despite the name): shear drives the y faces, tension the x
    # faces. Driving both lateral faces instead gives a different, stiffer
    # specimen (0.847 vs 0.743 at n = 4) -- the merged name is a trap.
    driven_axes = (1,) if mode == "shear" else (0,)
    driven = np.zeros(len(positions), dtype=bool)
    for axis in driven_axes:
        axial = positions[:, axis]
        driven |= np.isclose(axial, axial.max()) | np.isclose(axial, axial.min())
    driven_nodes = np.flatnonzero(driven)
    driven_dofs = (3 * driven_nodes[:, None] + np.arange(3)).ravel()
    free = np.ones(3 * len(positions), dtype=bool)
    free[driven_dofs] = False
    free_dofs = np.flatnonzero(free)
    displacement = np.zeros(3 * len(positions))
    displacement[driven_dofs] = affine.ravel()[driven_dofs]
    rhs = -(hessian[free_dofs][:, driven_dofs] @ displacement[driven_dofs])
    free_block = hessian[free_dofs][:, free_dofs]
    # The free-surface boundary-value problem is singular for some sizes (the
    # 26-neighbour central-force network has zero-energy modes once a set of
    # faces is fully prescribed on an even-numbered lattice, see the audit
    # notes). The zero modes carry no energy, so the min-norm least-squares
    # solution gives the same relaxed energy; a plain factorisation may return
    # garbage there, hence the fallback.
    method = "spsolve"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", sparse_linalg.MatrixRankWarning)
            solution = sparse_linalg.spsolve(free_block, rhs)
        if not np.all(np.isfinite(solution)):
            raise ArithmeticError("non-finite direct solution")
    except Exception:  # singular free block -> minimum-norm solution
        method = "lsqr"
        solution = sparse_linalg.lsqr(free_block, rhs, atol=1e-16, btol=1e-16,
                                      iter_lim=200000, conlim=1e14)[0]
    displacement[free_dofs] = solution
    planar = displacement.reshape(-1, 3)
    residual = float(np.abs(hessian @ displacement)[free_dofs].max())

    low = positions.min(axis=0)
    high = positions.max(axis=0)
    tolerance = 1e-9 * spacing
    core = np.ones(len(positions), dtype=bool)
    for axis in range(3):
        core &= positions[:, axis] > low[axis] + spacing - tolerance
        core &= positions[:, axis] < high[axis] - spacing + tolerance
    count = int(core.sum())

    def core_density(field: np.ndarray) -> float:
        extension = np.einsum("ij,ij->i", field[pairs[:, 1]] - field[pairs[:, 0]], normals)
        energy = 0.5 * stiffness * extension**2
        share = 0.5 * np.where(core[pairs[:, 0]], energy, 0.0) \
            + 0.5 * np.where(core[pairs[:, 1]], energy, 0.0)
        return float(share.sum() / (count * spacing**3))

    relaxed_constant = 2.0 * core_density(planar)
    affine_constant = 2.0 * core_density(affine)
    # Penetration diagnostic: |u| of the deepest core shell versus the surface.
    distance = np.minimum.reduce([
        positions[:, axis] - low[axis] for axis in range(3)
    ] + [high[axis] - positions[:, axis] for axis in range(3)]) / spacing
    magnitude = np.linalg.norm(planar, axis=1)
    shell = np.round(distance).astype(int)
    per_shell = {
        int(k): float(np.sqrt((magnitude[shell == k] ** 2).mean()))
        for k in range(0, int(shell.max()) + 1)
    }
    return {
        "size": size,
        "mode": mode,
        "particles": int(len(positions)),
        "bonds": int(len(pairs)),
        "core_particles": count,
        "spacing_m": spacing,
        "strain": float(strain),
        "relaxed_constant_Pa": float(relaxed_constant),
        "affine_same_specimen_Pa": float(affine_constant),
        "ratio_relaxed_to_affine": float(relaxed_constant / affine_constant),
        "free_dof_residual_N": residual,
        "solve_method": method,
        "max_free_displacement_m": float(magnitude[free_dofs // 3].max()),
        "displacement_per_affine": float(
            magnitude[free_dofs // 3].max() / (strain * spacing * size)),
        "shell_rms_displacement_m": per_shell,
        "core_rms_displacement_m": float(np.sqrt((magnitude[core] ** 2).mean())),
        "surface_rms_displacement_m": float(np.sqrt((magnitude[~core] ** 2).mean())),
    }


def fit_surface_model(sizes: Sequence[int], ratios: Sequence[float],
                      orders: Sequence[int] = (1, 2, 3)) -> dict[str, Any]:
    """Least-squares fits of ``ratio(n) = 1 - sum_k c_k / n^k``.

    A free-surface specimen carries a constant surface energy deficit, so the
    apparent modulus behaves like ``1 - c1 / n + O(1/n^2)``. Fitting only
    ``1/n`` over a narrow size range extrapolates poorly; including higher orders
    exposes whether the data support the surface model at all.
    """
    sizes = np.asarray(sizes, dtype=float)
    ratios = np.asarray(ratios, dtype=float)
    fits: dict[str, Any] = {}
    for order in orders:
        design = np.column_stack([np.ones_like(sizes)] + [1.0 / sizes**k for k in range(1, order + 1)])
        if design.shape[1] > len(sizes):
            continue
        solution, *_ = np.linalg.lstsq(design, ratios, rcond=None)
        residual = design @ solution - ratios
        fits[f"order_{order}"] = {
            "intercept": float(solution[0]),
            "coefficients": [float(v) for v in solution[1:]],
            "max_abs_residual": float(np.abs(residual).max()),
            "rms_residual": float(np.sqrt((residual**2).mean())),
            "dof": int(len(sizes) - design.shape[1]),
        }
    return {"sizes": [int(v) for v in sizes], "fits": fits}