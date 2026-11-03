"""Two-dimensional anisotropic continuum reference for diametral compression tests.

Why this module exists
----------------------
``docs/dem3d_brazilian_hondros_closure.md`` (mission m2543) left one item open:
the bonded lattice settles at a centre principal-stress ratio of ``-2.015``
against the isotropic point-load reference ``-3``, and the report attributed the
gap to the lattice's own elastic anisotropy (Zener 2.33) without having an
anisotropic benchmark to check that attribution against ("需要各向异性中心场解，
本单未做").  This module supplies that benchmark with a *different*
discretisation than the DEM engine: a bilinear-quadrilateral (Q4) plane-stress
finite-element model of a disc under two diametrically opposite loads, using the
lattice's own effective (Born / affine) stiffness tensor.

With it the lattice-vs-Hondros gap splits into three separately measured parts:

1. **window averaging** -- the DEM reports a virial stress averaged over a
   window of +-1.5 lattice spacings, not the continuum centre value.  That is
   measurable with the *isotropic* FE model (box average vs centre node).
2. **constitutive anisotropy** -- anisotropic FE minus isotropic FE for the same
   geometry, load and window.
3. **residual** -- whatever is left when the DEM result is compared with the
   anisotropic FE reference: lattice spacing / contact-arc / force-law effects.

The isotropic limit is checked against the exact Hondros (1959) / ASTM D3967
point-load field (load axis ``z``, so ``z`` is the *axial* direction and ``x``
the *transverse* one)::

    sigma_zz = -6 P / (pi D t)      (axial, compression)
    sigma_xx = +2 P / (pi D t)      (transverse, tension, = "split stress")

See ``tests/test_anisotropic_continuum.py``.

Conventions
-----------
* ``z`` is the load axis and ``x`` the transverse in-plane axis, matching the
  labels used by the DEM reports (``sigma_xx`` = split tension, ``sigma_zz`` =
  axial compression).  ``y`` is the thickness direction, plane stress in ``y``
  (free faces).
* SI units (m, Pa, N); tension positive, as in the DEM virial stress.
* Voigt order ``(xx, zz, xz)`` for stress and strain, with *engineering* shear
  strain ``gamma_xz = 2 eps_xz``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "stiffness_matrix_2d",
    "plane_stress_from_cubic",
    "zener_anisotropy",
    "isotropic_plane_stress",
    "Q4DiscMesh",
    "q4_disc_mesh",
    "DiscSolution",
    "solve_diametral_disc",
    "disc_stress_field",
    "HONDROS_SPLIT_FACTOR",
]

#: ``sigma_axial / sigma_transverse`` of the isotropic point-load solution.
HONDROS_SPLIT_FACTOR = -3.0


# ---------------------------------------------------------------------------
# constitutive helpers
# ---------------------------------------------------------------------------
def stiffness_matrix_2d(c11: float, c12: float, c66: float) -> np.ndarray:
    """In-plane stiffness of a sheet with 4-fold (square) in-plane symmetry.

    ``c11 = C_xxxx = C_zzzz``, ``c12 = C_xxzz``, ``c66 = C_xzxz`` (engineering
    shear).  A cubic crystal seen in a {100} plane and an orthotropic sheet with
    equal in-plane Young moduli both have this form.
    """
    c11 = float(c11)
    c12 = float(c12)
    c66 = float(c66)
    if min(c11, c66) <= 0.0:
        raise ValueError("c11 and c66 must be positive")
    if abs(c12) >= c11:
        raise ValueError("|c12| must be smaller than c11 for a stable sheet")
    return np.array([[c11, c12, 0.0], [c12, c11, 0.0], [0.0, 0.0, c66]], dtype=float)


def plane_stress_from_cubic(c11: float, c12: float, c44: float) -> np.ndarray:
    """Reduce a cubic 3-D stiffness to 2-D plane stress with the free axis ``y``.

    ``sigma_yy = 0`` gives ``eps_yy = -(c12/c11)(eps_xx + eps_zz)``, hence

        C~11 = c11 - c12^2 / c11,   C~12 = c12 - c12^2 / c11,   C~66 = c44.

    The in-plane shear entry is untouched: the free surfaces cannot carry
    ``tau_xy``/``tau_yz``, the only entries the reduction would otherwise mix.
    """
    red = float(c12) ** 2 / float(c11)
    return stiffness_matrix_2d(float(c11) - red, float(c12) - red, float(c44))


def zener_anisotropy(c11: float, c12: float, c44: float) -> float:
    """Zener ratio ``C44 / ((C11 - C12)/2)``; 1 for an isotropic crystal."""
    return float(c44) / (0.5 * (float(c11) - float(c12)))


def isotropic_plane_stress(young: float, poisson: float) -> np.ndarray:
    """Plane-stress stiffness of an isotropic sheet with Young modulus ``young``."""
    factor = young / (1.0 - poisson**2)
    shear = young / (2.0 * (1.0 + poisson))
    return stiffness_matrix_2d(factor, poisson * factor, shear)


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Q4DiscMesh:
    """Staircase Q4 mesh of a disc built on a square grid centred at the origin.

    Element inclusion is decided by the element *centroid* radius, which keeps
    the grid node at the load point attached to two elements.  The load point is
    therefore at distance ``R`` from the centre, so the loading diameter equals
    ``2R`` exactly -- required when the centre stress is compared with
    ``2P/(pi D t)``.
    """

    radius: float
    thickness: float
    h: float
    coords: np.ndarray  # (nnode, 2) columns are (x, z)
    elements: np.ndarray  # (nelem, 4)
    center_node: int
    elem_centroids: np.ndarray  # (nelem, 2)
    elements_per_node: np.ndarray  # (nnode,)

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.coords.shape[0])

    def load_nodes(
        self,
        mode: str = "point",
        patch_length: float | None = None,
        axis: str = "z",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nodes on the positive loading side and their share of the total load."""
        axial, transverse = _axis_columns(axis)
        if mode == "point":
            tip = np.zeros(2)
            tip[axial] = self.radius
            return np.array([_node_at(self, tip[0], tip[1])]), np.array([1.0])
        if mode != "patch":
            raise ValueError(f"unknown load mode {mode!r}")
        if patch_length is None or patch_length <= 0.0:
            raise ValueError("patch mode needs a positive patch_length")
        boundary = self.elements_per_node < 4
        cand = np.where(
            boundary
            & (np.abs(self.coords[:, transverse]) <= 0.5 * patch_length)
            & (self.coords[:, axial] > 0.0)
        )[0]
        if cand.size == 0:
            raise ValueError("no boundary nodes inside the requested patch")
        tip = self.coords[cand, axial].max()
        cand = cand[self.coords[cand, axial] >= tip - 1.0001 * self.h]
        return cand, np.full(cand.size, 1.0 / cand.size)


def _axis_columns(axis: str) -> tuple[int, int]:
    if axis == "z":
        return 1, 0
    if axis == "x":
        return 0, 1
    raise ValueError(f"unknown load axis {axis!r}")


def _node_at(mesh: Q4DiscMesh, x: float, z: float) -> int:
    d = np.hypot(mesh.coords[:, 0] - x, mesh.coords[:, 1] - z)
    return int(np.argmin(d))


def q4_disc_mesh(radius: float, n_radial: int, thickness: float = 1.0) -> Q4DiscMesh:
    """Build the staircase disc mesh with ``n_radial`` elements per radius."""
    radius = float(radius)
    n_radial = int(n_radial)
    if n_radial < 2:
        raise ValueError("n_radial must be >= 2")
    h = radius / n_radial
    m = 2 * n_radial + 1
    ii, jj = np.meshgrid(np.arange(m - 1), np.arange(m - 1), indexing="ij")
    cx = (ii + 0.5) * h - radius
    cz = (jj + 0.5) * h - radius
    inc = (cx**2 + cz**2) <= radius * radius

    used = np.zeros((m, m), dtype=bool)
    used[:-1, :-1] |= inc
    used[1:, :-1] |= inc
    used[1:, 1:] |= inc
    used[:-1, 1:] |= inc

    idmap = np.full((m, m), -1, dtype=np.int64)
    idmap[used] = np.arange(int(used.sum()))
    ix, iz = np.nonzero(used)
    coords = np.stack([ix * h - radius, iz * h - radius], axis=1).astype(float)

    elements = np.stack(
        [
            idmap[:-1, :-1][inc],
            idmap[1:, :-1][inc],
            idmap[1:, 1:][inc],
            idmap[:-1, 1:][inc],
        ],
        axis=1,
    ).astype(np.int64)

    elem_centroids = np.stack([cx[inc], cz[inc]], axis=1).astype(float)
    elements_per_node = np.bincount(elements.ravel(), minlength=coords.shape[0])

    center = int(idmap[n_radial, n_radial])
    if center < 0:  # pragma: no cover - mesh invariant
        raise RuntimeError("centre node missing from the mesh")
    return Q4DiscMesh(
        radius=radius,
        thickness=float(thickness),
        h=h,
        coords=coords,
        elements=elements,
        center_node=center,
        elem_centroids=elem_centroids,
        elements_per_node=elements_per_node,
    )


# ---------------------------------------------------------------------------
# Q4 element kernels
# ---------------------------------------------------------------------------
def _gradients(xy: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Shape-function gradients in physical coordinates plus the Jacobian.

    ``xy`` is the (4, 2) array of element node coordinates in the anticlockwise
    order bottom-left, bottom-right, top-right, top-left.
    """
    dxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
    deta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])
    j = np.array(
        [
            [dxi @ xy[:, 0], dxi @ xy[:, 1]],
            [deta @ xy[:, 0], deta @ xy[:, 1]],
        ]
    )
    det = j[0, 0] * j[1, 1] - j[0, 1] * j[1, 0]
    if det <= 0.0:
        raise ValueError("non-positive Jacobian: element ordering must be anticlockwise")
    inv = np.array([[j[1, 1], -j[0, 1]], [-j[1, 0], j[0, 0]]]) / det
    grads = inv @ np.stack([dxi, deta])
    return grads[0], grads[1], det


def _strain_displacement(gx: np.ndarray, gz: np.ndarray) -> np.ndarray:
    b = np.zeros((3, 8))
    b[0, 0::2] = gx
    b[1, 1::2] = gz
    b[2, 0::2] = gz
    b[2, 1::2] = gx
    return b


def _element_stiffness(xy: np.ndarray, d: np.ndarray, t: float) -> np.ndarray:
    a = 1.0 / np.sqrt(3.0)
    ke = np.zeros((8, 8))
    for xi, eta in ((-a, -a), (a, -a), (a, a), (-a, a)):
        gx, gz, det = _gradients(xy, xi, eta)
        b = _strain_displacement(gx, gz)
        ke += t * det * (b.T @ d @ b)
    return ke


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DiscSolution:
    """Centre field of a diametrally loaded disc, in the DEM promise window."""

    sigma_xx_center: float
    sigma_zz_center: float
    tau_xz_center: float
    sigma_xx_window: float
    sigma_zz_window: float
    split_stress: float
    ratio_center: float
    ratio_window: float
    window_half_width: float
    load: float
    diameter: float
    thickness: float
    n_radial: int
    n_dof: int
    axial_reaction: float
    transverse_reaction: float
    load_axis: str
    c_in_plane: tuple[float, float, float]

    def as_dict(self) -> dict:
        return {
            "sigma_xx_center": self.sigma_xx_center,
            "sigma_zz_center": self.sigma_zz_center,
            "tau_xz_center": self.tau_xz_center,
            "sigma_xx_window": self.sigma_xx_window,
            "sigma_zz_window": self.sigma_zz_window,
            "split_stress": self.split_stress,
            "ratio_center": self.ratio_center,
            "ratio_window": self.ratio_window,
            "window_half_width": self.window_half_width,
            "load": self.load,
            "diameter": self.diameter,
            "thickness": self.thickness,
            "n_radial": self.n_radial,
            "n_dof": self.n_dof,
            "axial_reaction": self.axial_reaction,
            "transverse_reaction": self.transverse_reaction,
            "load_axis": self.load_axis,
            "c_in_plane": list(self.c_in_plane),
        }


def _load_vector(mesh: Q4DiscMesh, load: float, mode: str, patch_length, axis: str) -> np.ndarray:
    axial, _ = _axis_columns(axis)
    f = np.zeros(2 * mesh.n_nodes)
    nodes, weights = mesh.load_nodes(mode, patch_length, axis)
    for node, w in zip(nodes, weights):
        tip = mesh.coords[node].copy()
        mirror = tip.copy()
        mirror[axial] *= -1.0
        other = _node_at(mesh, mirror[0], mirror[1])
        f[2 * node + axial] -= float(load) * float(w)
        f[2 * other + axial] += float(load) * float(w)
    return f


def _symmetry_dofs(mesh: Q4DiscMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tol = 1e-9 * mesh.radius
    on_x_axis = np.where(np.abs(mesh.coords[:, 1]) <= tol)[0]  # z = 0 row -> u_z = 0
    on_z_axis = np.where(np.abs(mesh.coords[:, 0]) <= tol)[0]  # x = 0 column -> u_x = 0
    return on_x_axis, on_z_axis, tol


def _solve(d: np.ndarray, load: float, diameter: float, thickness: float, n_radial: int,
           load_mode: str, patch_length, axis: str) -> tuple[Q4DiscMesh, np.ndarray, np.ndarray]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    radius = 0.5 * float(diameter)
    thickness = float(thickness)
    d = np.asarray(d, dtype=float)
    mesh = q4_disc_mesh(radius, n_radial, thickness)
    ndof = 2 * mesh.n_nodes

    rows, cols, vals = [], [], []
    for conn in mesh.elements:
        ke = _element_stiffness(mesh.coords[conn], d, thickness)
        dofs = np.empty(8, dtype=np.int64)
        dofs[0::2] = 2 * conn
        dofs[1::2] = 2 * conn + 1
        rows.append(np.repeat(dofs, 8))
        cols.append(np.tile(dofs, 8))
        vals.append(ke.ravel())
    k = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ndof, ndof),
    ).tocsr()

    f = _load_vector(mesh, load, load_mode, patch_length, axis)

    row_zero, col_zero, _ = _symmetry_dofs(mesh)
    fixed = np.zeros(ndof, dtype=bool)
    fixed[2 * row_zero + 1] = True  # u_z = 0 on the z = 0 row
    fixed[2 * col_zero] = True      # u_x = 0 on the x = 0 column
    free = np.where(~fixed)[0]
    u = np.zeros(ndof)
    u[free] = spsolve(k[free][:, free], f[free])

    stress = np.zeros((mesh.n_elements, 3))
    for e, conn in enumerate(mesh.elements):
        gx, gz, _ = _gradients(mesh.coords[conn], 0.0, 0.0)
        b = _strain_displacement(gx, gz)
        ue = np.empty(8)
        ue[0::2] = u[2 * conn]
        ue[1::2] = u[2 * conn + 1]
        stress[e] = d @ (b @ ue)
    return mesh, u, stress


def solve_diametral_disc(
    d: np.ndarray,
    load: float,
    diameter: float,
    thickness: float,
    n_radial: int = 40,
    window_half_width: float | None = None,
    load_mode: str = "point",
    patch_length: float | None = None,
    load_axis: str = "z",
) -> DiscSolution:
    """Solve the plane-stress disc under two opposite diametral loads.

    Symmetry boundary conditions (``u_x = 0`` on the ``x = 0`` column,
    ``u_z = 0`` on the ``z = 0`` row) are exact for a material with 4-fold
    in-plane symmetry under a symmetric load: they remove the rigid-body modes
    without adding spurious reactions, which is why the reported reaction sums
    must come out at zero (they are returned so the caller can check it).
    """
    radius = 0.5 * float(diameter)
    mesh, u, stress = _solve(d, load, diameter, thickness, n_radial, load_mode, patch_length, load_axis)
    centre = mesh.coords[mesh.center_node]

    core = np.all(np.abs(mesh.elem_centroids - centre) <= 0.51 * mesh.h, axis=1)
    if not core.any():  # pragma: no cover - mesh invariant
        raise RuntimeError("no element found at the centre node")
    stress_center = stress[core].mean(axis=0)

    if window_half_width is None:
        window_half_width = 1.5 * mesh.h
    window = np.all(np.abs(mesh.elem_centroids - centre) <= float(window_half_width), axis=1)
    # a zero-width window means "the centre node itself", i.e. the core average
    stress_window = stress[window].mean(axis=0) if window.any() else stress_center

    # reaction check: re-assemble k @ u - f on the constrained sets
    row_zero, col_zero, _ = _symmetry_dofs(mesh)
    from scipy.sparse import coo_matrix

    ndof = 2 * mesh.n_nodes
    d = np.asarray(d, dtype=float)
    rows, cols, vals = [], [], []
    for conn in mesh.elements:
        ke = _element_stiffness(mesh.coords[conn], d, thickness)
        dofs = np.empty(8, dtype=np.int64)
        dofs[0::2] = 2 * conn
        dofs[1::2] = 2 * conn + 1
        rows.append(np.repeat(dofs, 8))
        cols.append(np.tile(dofs, 8))
        vals.append(ke.ravel())
    k = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ndof, ndof),
    ).tocsr()
    f = _load_vector(mesh, load, load_mode, patch_length, load_axis)
    reaction = k @ u - f

    split = 2.0 * float(load) / (np.pi * float(diameter) * thickness)
    return DiscSolution(
        sigma_xx_center=float(stress_center[0]),
        sigma_zz_center=float(stress_center[1]),
        tau_xz_center=float(stress_center[2]),
        sigma_xx_window=float(stress_window[0]),
        sigma_zz_window=float(stress_window[1]),
        split_stress=split,
        ratio_center=float(stress_center[1] / stress_center[0]),
        ratio_window=float(stress_window[1] / stress_window[0]),
        window_half_width=float(window_half_width),
        load=float(load),
        diameter=float(diameter),
        thickness=float(thickness),
        n_radial=int(n_radial),
        n_dof=int(ndof),
        axial_reaction=float(reaction[2 * row_zero + 1].sum()),
        transverse_reaction=float(reaction[2 * col_zero].sum()),
        load_axis=load_axis,
        c_in_plane=(float(d[0, 0]), float(d[0, 1]), float(d[2, 2])),
    )


def disc_stress_field(
    d: np.ndarray,
    load: float,
    diameter: float,
    thickness: float,
    n_radial: int = 40,
    load_axis: str = "z",
) -> tuple[Q4DiscMesh, np.ndarray]:
    """Return the mesh and the per-element centroid stress (for field plots)."""
    mesh, _, stress = _solve(d, load, diameter, thickness, n_radial, "point", None, load_axis)
    return mesh, stress