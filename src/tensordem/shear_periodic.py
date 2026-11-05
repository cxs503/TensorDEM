"""Lateral periodic closure of the DEM3D direct-shear specimen.

Why this module exists
----------------------
``docs/dem3d_shear_static_reference.md`` §5 left one open item on the road from a
bond stiffness ``k`` to the macroscopic shear modulus ``G``: the whole-layer
reaction of :func:`tensordem.shear_static.static_direct_shear` is reduced by a
free-surface boundary layer, so ``G_eff`` only approaches ``C44 = 7k/(3a)`` as the
cross-section grows (measured 0.687 at ``n = 5`` and 0.980 at ``n = 61`` for
``h/a = 2``).  That page named two ways out:

> either a cross-section much wider than the boundary layer ... or periodic
> boundaries.

``docs/dem3d-shear-cross-section-convergence/`` took the first route and fitted
``d(n) = c1/n + c2/n^2`` with an extrapolated bias ``c0 ~ 1e-3``.  This module takes
the second route and turns it into a **negative control**: on a laterally periodic
cell the affine field ``u_x = gamma z`` is an *exact* solution of the discrete
equilibrium, because every node then owns its full centrally symmetric 26-neighbour
set.  The predicted reaction is therefore

    R = sum_{bottom nodes} k sum_{upward bonds} gamma dz n_x^2 = n^2 * (7/3) k gamma a

and ``A_eff = n^2 a^2`` (the bond census of :func:`~tensordem.shear_static.affine_effective_area`
specialised to the fully coordinated layer), so ``G_eff / C44 = 1`` to machine
precision *at every size*, with no fit and no extrapolation.  Any deviation would be
an assembly or convention bug, not physics.

That is what makes it evidence rather than a tautology: the same code path, with the
wrap disabled, must reproduce the free-surface solver reference
(:func:`~tensordem.shear_static.static_direct_shear`) to solver accuracy -- the tests
assert exactly that -- and only then is the periodic result allowed to speak about
the free-surface deficit.

It also supplies the second, independent falsification of the fitted asymptote: a
cell that is periodic along ``x`` but free along ``y`` keeps *one* pair of free
surfaces, so its deficit must follow a one-dimensional ``c1/n_y + c2/n_y^2`` law with
``c0 = 0``.  Agreement with the two-dimensional fit's ``c0`` (1.0e-3, i.e. inside its
own RMS residual) confirms that the reported ``G`` of the direct-shear specimen has
no hidden asymptotic bias.

Conventions (identical to the solver and to :mod:`tensordem.shear_static`)
-------------------------------------------------------------------------
* sites on a simple-cubic lattice with spacing ``a = 2r``, ``nz`` layers,
  ``h = (nz - 1) a``;
* 26 neighbour offsets, central-force bond law ``f = k (n . du) n``, stiffness
  ``k n_a n_b`` per bond;
* bottom layer fixed in all three components, top layer with ``u_x = gamma h``
  prescribed only, all other DOF free;
* "periodic" means the neighbour image is wrapped in that lateral direction and the
  bond keeps its *unwrapped* offset (minimum-image bond vector).  ``z`` is never
  periodic: the top and bottom layers are real surfaces.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .dem3d import DEM3DConfig
from .shear_specimen import lattice_spacing, shear_modulus_closed_form
from .shear_static import AFFINE_WEIGHT_PER_NODE

__all__ = [
    "NEIGHBOUR_OFFSETS",
    "lattice_sites",
    "bond_list",
    "periodic_stiffness_matrix",
    "static_shear_periodic",
    "periodic_series",
    "height_series",
    "affine_residual",
]

#: The 26 neighbour offsets of the ``IceDEM3D`` bond set (integer lattice units).
NEIGHBOUR_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)

#: Relative tolerance under which a measurement counts as converged (platform gate).
ACCURACY_GATE = 0.03


def _config(config: DEM3DConfig | None) -> DEM3DConfig:
    base = config or DEM3DConfig()
    return base


def lattice_sites(nx: int, ny: int, nz: int, spacing: float) -> np.ndarray:
    """``(N, 3)`` site coordinates of an ``nx x ny x nz`` simple-cubic block.

    Node ``p`` sits at lattice index ``(p % nx, (p // nx) % ny, p // (nx*ny))`` --
    the same column-major convention :func:`bond_list` uses for its topology.
    """
    for name, value in (("nx", nx), ("ny", ny), ("nz", nz)):
        if int(value) < 2:
            raise ValueError(f"{name} must be >= 2")
    index = np.arange(int(nx) * int(ny) * int(nz))
    ix = index % int(nx)
    iy = (index // int(nx)) % int(ny)
    iz = index // (int(nx) * int(ny))
    return np.stack([ix, iy, iz], axis=1).astype(float) * spacing


def bond_list(
    nx: int,
    ny: int,
    nz: int,
    spacing: float,
    *,
    periodic_x: bool = True,
    periodic_y: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bond topology of the block with optional lateral periodic closure.

    Returns ``(i, j, delta)``: node indices (``i < j`` in the global numbering when
    the bond is in-plane, and always the lower layer for cross-layer bonds) and the
    *unwrapped* bond vector ``delta = d * spacing`` used by the central-force law.
    Bonds whose image would leave a non-periodic lateral face simply do not exist --
    that is the free-surface deficit in its discrete form.
    """
    nx, ny, nz = int(nx), int(ny), int(nz)
    n_layers = nx * ny

    def flat(ix: int, iy: int, iz: int) -> int:
        return ix + nx * iy + n_layers * iz

    i_idx: list[int] = []
    j_idx: list[int] = []
    deltas: list[tuple[float, float, float]] = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                node = flat(ix, iy, iz)
                for dx, dy, dz in NEIGHBOUR_OFFSETS:
                    if dz < 0:
                        continue  # covered by the dz > 0 partner
                    if dz == 0 and not (dx > 0 or (dx == 0 and dy > 0)):
                        continue  # count in-plane bonds once
                    jx, jy, jz = ix + dx, iy + dy, iz + dz
                    if jz >= nz:
                        continue  # free top surface
                    if jx < 0 or jx >= nx:
                        if not periodic_x:
                            continue
                        jx %= nx
                    if jy < 0 or jy >= ny:
                        if not periodic_y:
                            continue
                        jy %= ny
                    other = flat(jx, jy, jz)
                    if other == node:
                        continue  # self image of a cell shorter than the neighbour range
                    i_idx.append(node)
                    j_idx.append(other)
                    deltas.append((dx * spacing, dy * spacing, dz * spacing))
    return (np.asarray(i_idx, dtype=np.int64),
            np.asarray(j_idx, dtype=np.int64),
            np.asarray(deltas, dtype=float))


def periodic_stiffness_matrix(
    positions: np.ndarray,
    bonds: tuple[np.ndarray, np.ndarray, np.ndarray],
    stiffness: float,
) -> sparse.csr_matrix:
    """Assembled ``3N x 3N`` translational stiffness of the bond list.

    Each bond contributes ``k n_a n_b`` to the four ``(i, j)`` blocks; the geometric
    term vanishes at the undeformed state (same convention as
    :func:`tensordem.shear_static.shear_stiffness_matrix`).
    """
    i, j, delta = bonds
    length = np.linalg.norm(delta, axis=1)
    normal = delta / length[:, None]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for a in range(3):
        for b in range(3):
            block = stiffness * normal[:, a] * normal[:, b]
            rows.extend((3 * i + a, 3 * j + a, 3 * i + a, 3 * j + a))
            cols.extend((3 * i + b, 3 * j + b, 3 * j + b, 3 * i + b))
            vals.extend((block, block, -block, -block))
    return sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * len(positions), 3 * len(positions)),
    ).tocsr()


def _affine_field(n_nodes: int, z: np.ndarray, gamma: float) -> np.ndarray:
    """``u = gamma z e_x`` flattened to the solver's DOF ordering."""
    u = np.zeros(3 * n_nodes)
    u[0::3] = gamma * z
    return u


def _effective_area(bonds: tuple[np.ndarray, np.ndarray, np.ndarray], n_layers: int) -> float:
    """Affine bond-census area of the sheared plane (zero for missing bonds).

    ``weight = sum l^2 n_x^2 n_z^2`` over the bonds that leave the fixed bottom
    layer, divided by the per-node affine weight ``7 a^2 / 3`` -- the same identity
    as :func:`tensordem.shear_static.affine_effective_area`.  With the lateral faces
    periodic every bottom node owns its full 26-neighbour set, so the census reduces
    to the nominal ``nx a * ny a``.
    """
    i, _j, delta = bonds
    length = np.linalg.norm(delta, axis=1)
    normal = delta / length[:, None]
    leaves_bottom = (delta[:, 2] > 0.0) & ((i // n_layers) == 0)
    weight = float(np.sum(
        length[leaves_bottom] ** 2
        * normal[leaves_bottom, 0] ** 2
        * normal[leaves_bottom, 2] ** 2))
    return weight / AFFINE_WEIGHT_PER_NODE


def static_shear_periodic(
    nx: int,
    ny: int,
    nz: int = 3,
    *,
    periodic_x: bool = True,
    periodic_y: bool = True,
    config: DEM3DConfig | None = None,
    gamma: float = 1.0e-3,
    return_displacement: bool = False,
) -> dict[str, Any]:
    """Rate-free direct shear of the block under (optionally) periodic lateral faces.

    Solves ``K u = 0`` on the free DOF with the bottom layer fixed and
    ``u_x = gamma h`` on the top layer, then reports the bottom reaction normalised
    by the *nominal* area ``nx a * ny a`` and by the bond-census area
    :func:`~tensordem.shear_static.affine_effective_area`.  With both lateral
    directions periodic the affine field is an exact solution, so
    ``G_effective_over_C44`` must be ``1`` to round-off; that is the verification
    assertion of this module.

    ``return_displacement=True`` adds the nodal equilibrium field
    (``displacement_m``, ``positions_m``) so the deviation from the affine field can
    be inspected layer by layer; it is omitted by default to keep the payload
    JSON-friendly.
    """
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and positive")
    cfg = _config(config)
    a = lattice_spacing(cfg)
    k = cfg.bond_stiffness
    height = (int(nz) - 1) * a
    nx, ny, nz = int(nx), int(ny), int(nz)
    positions = lattice_sites(nx, ny, nz, a)
    bonds = bond_list(nx, ny, nz, a, periodic_x=periodic_x, periodic_y=periodic_y)
    K = periodic_stiffness_matrix(positions, bonds, k)
    n_nodes = len(positions)
    z = positions[:, 2]
    top = np.isclose(z, z.max())
    bottom = np.isclose(z, z.min())
    fixed = np.array([3 * p + c for p in np.where(bottom)[0] for c in range(3)], dtype=np.int64)
    prescribed = np.array([3 * p for p in np.where(top)[0]], dtype=np.int64)
    free = np.setdiff1d(np.arange(3 * n_nodes), np.concatenate([fixed, prescribed]))
    if free.size == 0:
        raise ValueError("specimen has no free degrees of freedom")

    u = np.zeros(3 * n_nodes)
    u[prescribed] = gamma * height
    u[free] = sparse_linalg.spsolve(
        K[free][:, free].tocsc(), -K[free][:, prescribed] @ u[prescribed])
    residual = K @ u
    reaction = float(residual[fixed[0::3]].sum())

    area_nominal = nx * a * ny * a
    area_effective = _effective_area(bonds, nx * ny)
    c44 = shear_modulus_closed_form(cfg)
    tau_nominal = abs(reaction) / area_nominal
    tau_effective = abs(reaction) / area_effective

    # Independent affine prediction: every fully coordinated interior node owns the
    # 26-bond set, whose upward half gives sum dz n_x^2 = 7/3 (in units of a).
    predicted_reaction = float(np.count_nonzero(bottom)) * (7.0 / 3.0) * k * gamma * a
    i, _j, delta = bonds
    length = np.linalg.norm(delta, axis=1)
    normal = delta / length[:, None]
    dx_bond = float(np.sum(k * gamma * delta[:, 2] * normal[:, 0] ** 2))
    del i
    bond_force_scale = k * gamma * a
    payload = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "periodic_x": bool(periodic_x),
        "periodic_y": bool(periodic_y),
        "gamma": float(gamma),
        "lattice_spacing_m": a,
        "height_m": height,
        "particle_count": int(n_nodes),
        "bond_count": int(len(bonds[0])),
        "C44_Pa": c44,
        "area_nominal_m2": area_nominal,
        "area_effective_m2": area_effective,
        "area_ratio": area_effective / area_nominal,
        "reaction_N": reaction,
        "reaction_from_bonds_N": dx_bond,
        "reaction_affine_prediction_N": predicted_reaction,
        "tau_nominal_Pa": tau_nominal,
        "tau_effective_Pa": tau_effective,
        "G_nominal_over_C44": tau_nominal / gamma / c44,
        "G_effective_over_C44": tau_effective / gamma / c44,
        "deficit": 1.0 - tau_effective / gamma / c44,
        "free_residual_over_bond_force": float(np.abs(residual[free]).max() / bond_force_scale),
        "affine_residual_over_bond_force": float(
            np.abs((K @ _affine_field(n_nodes, z, gamma))[free]).max() / bond_force_scale),
        "max_free_displacement_over_gamma_h": float(np.abs(u[free]).max() / (gamma * height)),
        "reaction_over_affine_prediction": abs(reaction) / predicted_reaction,
    }
    if return_displacement:
        payload["displacement_m"] = u.reshape(-1, 3)
        payload["positions_m"] = positions
    return payload


def periodic_series(
    sizes: Sequence[int],
    *,
    nz: int = 3,
    periodic_x: bool = True,
    periodic_y: bool = True,
    config: DEM3DConfig | None = None,
    gamma: float = 1.0e-3,
) -> dict[str, Any]:
    """Run :func:`static_shear_periodic` over a list of lateral sizes."""
    sizes = [int(n) for n in sizes]
    if not sizes or any(b <= a_ for a_, b in zip(sizes, sizes[1:])):
        raise ValueError("sizes must be non-empty and strictly increasing")
    rows = []
    for n in sizes:
        got = static_shear_periodic(
            n, n, nz, periodic_x=periodic_x, periodic_y=periodic_y,
            config=config, gamma=gamma)
        rows.append({
            "n": n,
            "nz": int(nz),
            "periodic_x": bool(periodic_x),
            "periodic_y": bool(periodic_y),
            "particle_count": got["particle_count"],
            "bond_count": got["bond_count"],
            "area_ratio": got["area_ratio"],
            "G_nominal_over_C44": got["G_nominal_over_C44"],
            "G_effective_over_C44": got["G_effective_over_C44"],
            "deficit": got["deficit"],
            "reaction_over_affine_prediction": got["reaction_over_affine_prediction"],
            "reaction_N": abs(got["reaction_N"]),
            "reaction_affine_prediction_N": got["reaction_affine_prediction_N"],
            "free_residual_over_bond_force": got["free_residual_over_bond_force"],
            "affine_residual_over_bond_force": got["affine_residual_over_bond_force"],
            "max_free_displacement_over_gamma_h": got["max_free_displacement_over_gamma_h"],
        })
    return {
        "sizes": sizes,
        "nz": int(nz),
        "periodic_x": bool(periodic_x),
        "periodic_y": bool(periodic_y),
        "gamma": float(gamma),
        "rows": rows,
        "max_abs_deficit": float(max(abs(r["deficit"]) for r in rows)),
    }


def affine_residual(
    nx: int,
    ny: int,
    nz: int = 3,
    *,
    periodic_x: bool = True,
    periodic_y: bool = True,
    config: DEM3DConfig | None = None,
    gamma: float = 1.0e-3,
) -> dict[str, float]:
    """Residual of the *undeformed-configuration* affine field, before solving.

    ``u_affine = gamma z e_x`` is inserted into ``K u`` and the largest force on the
    free degrees of freedom is reported in units of the bond-force scale
    ``k gamma a``.  This is the mechanism diagnostic: it is zero to round-off exactly
    when every free node's *missing-bond set* has a vanishing first moment in the
    sheared direction, which is the case under lateral periodic closure -- and then
    the affine field is the exact solution, so the measured ``G`` carries no
    free-surface deficit at all.
    """
    cfg = _config(config)
    a = lattice_spacing(cfg)
    k = cfg.bond_stiffness
    nx, ny, nz = int(nx), int(ny), int(nz)
    positions = lattice_sites(nx, ny, nz, a)
    bonds = bond_list(nx, ny, nz, a, periodic_x=periodic_x, periodic_y=periodic_y)
    K = periodic_stiffness_matrix(positions, bonds, k)
    n_nodes = len(positions)
    z = positions[:, 2]
    bottom = np.isclose(z, z.min())
    top = np.isclose(z, z.max())
    fixed = np.array([3 * p + c for p in np.where(bottom)[0] for c in range(3)], dtype=np.int64)
    prescribed = np.array([3 * p for p in np.where(top)[0]], dtype=np.int64)
    free = np.setdiff1d(np.arange(3 * n_nodes), np.concatenate([fixed, prescribed]))
    u = np.zeros(3 * n_nodes)
    u[0::3] = gamma * z
    residual = K @ u
    scale = k * gamma * a
    return {
        "max_free_residual_N": float(np.abs(residual[free]).max()) if free.size else 0.0,
        "affine_residual_over_bond_force": float(np.abs(residual[free]).max() / scale) if free.size else 0.0,
        "free_dofs": int(free.size),
    }


def height_series(
    nzs: Sequence[int],
    *,
    n: int = 9,
    periodic_x: bool = True,
    periodic_y: bool = True,
    config: DEM3DConfig | None = None,
    gamma: float = 1.0e-3,
) -> dict[str, Any]:
    """Hold the cross-section fixed and vary the height under lateral periodic closure.

    Used to show that ``C44`` is a material constant of the *bond law*, not a
    function of the specimen aspect ratio ``h/a``: under PBC every layer owns its full
    neighbour set, so ``G_eff / C44 = 1`` must hold for every ``nz``.
    """
    nzs = [int(v) for v in nzs]
    if not nzs or any(b <= a_ for a_, b in zip(nzs, nzs[1:])):
        raise ValueError("nzs must be non-empty and strictly increasing")
    rows = []
    for nz in nzs:
        got = static_shear_periodic(
            n, n, nz, periodic_x=periodic_x, periodic_y=periodic_y,
            config=config, gamma=gamma)
        rows.append({
            "n": int(n),
            "nz": nz,
            "height_over_spacing": float(nz - 1),
            "particle_count": got["particle_count"],
            "G_effective_over_C44": got["G_effective_over_C44"],
            "deficit": got["deficit"],
            "reaction_over_affine_prediction": got["reaction_over_affine_prediction"],
            "reaction_N": abs(got["reaction_N"]),
            "reaction_affine_prediction_N": got["reaction_affine_prediction_N"],
            "free_residual_over_bond_force": got["free_residual_over_bond_force"],
            "affine_residual_over_bond_force": got["affine_residual_over_bond_force"],
            "max_free_displacement_over_gamma_h": got["max_free_displacement_over_gamma_h"],
        })
    return {"n": int(n), "nzs": nzs, "rows": rows}
