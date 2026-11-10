"""2-D bonded-particle (spring-lattice) model with an LEFM bond failure law.

The model is the 2-D counterpart of the 3-D bonded-sphere ice model in
:mod:`tensordem.dem3d`: point particles on a simple square lattice, connected by
central (normal) springs to their 8 nearest neighbours.  A bond stores

    0.5 * k * (e * L) ** 2,      e = elongation / rest length

so the lattice is a central-force truss.  Two facts make it usable as a
*continuum* elasticity discretisation (derivation in
``docs/LEFM_ICE_CRACK.md``):

* with the diagonal stiffness set to half the axial stiffness
  (``k_diag = k_axial / 2``) the 8-neighbour square lattice is **isotropic**
  with ``nu = 1/4`` exactly, ``lambda = mu = k_axial / 2`` and
  ``E_2D = 1.25 * k_axial`` in the plane-stress sense (``E_2D = E * B``);
* the affine-deformation patch test is satisfied exactly, so the virial
  (bond-force) stress of an affine displacement field equals ``C : eps``.

Geometry is a rectangle ``ncols x nrows`` of lattice sites with spacing ``a``;
node ``(i, j)`` sits at ``(i*a, j*a)`` and has the flat index ``j*ncols + i``.
A crack is a *zero-width* slit: bonds crossing a lattice line are deleted, which
is what a sharp crack looks like at the resolution of the lattice.

Units are SI.  ``thickness`` (``B``) is the out-of-plane specimen thickness; all
spring stiffnesses already include it, so total energies are energies of the
whole specimen and stresses are true 3-D stresses.

Nothing in this module imports or mutates the dynamic DEM solvers; it is a
quasi-static lattice tool used for fracture-mechanics verification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["Lattice2DConfig", "BondedLattice2D", "ISOTROPIC_POISSON"]


#: Poisson ratio of the 8-neighbour central-force square lattice (k_diag = k/2).
ISOTROPIC_POISSON = 0.25


@dataclass(frozen=True)
class Lattice2DConfig:
    """Lattice/material parameters (SI)."""

    spacing: float
    axial_stiffness: float
    diagonal_stiffness: float
    thickness: float = 1.0
    young_modulus: float = 5.0e9
    poisson_ratio: float = ISOTROPIC_POISSON

    @classmethod
    def from_material(
        cls,
        young_modulus: float = 5.0e9,
        thickness: float = 0.05,
        spacing: float = 5.0e-3,
        poisson_ratio: float = ISOTROPIC_POISSON,
    ) -> "Lattice2DConfig":
        """Map an isotropic 3-D material onto the lattice.

        Plane stress, ``E_2D = E * B``; isotropy forces ``nu = 1/4`` and
        ``k_axial = E_2D / 1.25``, ``k_diagonal = k_axial / 2``.
        """
        if abs(poisson_ratio - ISOTROPIC_POISSON) > 1e-12:
            raise ValueError(
                "8-neighbour central-force square lattice is isotropic only at "
                f"nu = {ISOTROPIC_POISSON}; got nu = {poisson_ratio}"
            )
        k_axial = young_modulus * thickness / 1.25
        return cls(
            spacing=spacing,
            axial_stiffness=k_axial,
            diagonal_stiffness=0.5 * k_axial,
            thickness=thickness,
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
        )

    @property
    def lame_lambda(self) -> float:
        """2-D plane-stress Lame parameter (per unit thickness folded in)."""
        return 0.5 * self.axial_stiffness

    @property
    def shear_modulus(self) -> float:
        return 0.5 * self.axial_stiffness

    @property
    def modulus_2d(self) -> float:
        """Plane-stress modulus ``E*B`` of the lattice."""
        return 1.25 * self.axial_stiffness


class BondedLattice2D:
    """Square-lattice bonded-particle assembly with removable bonds."""

    #: neighbour offsets, stiffness multiplier, and bond family tag
    _NEIGHBOURS = (
        (1, 0, 0),   # axial x
        (0, 1, 0),   # axial y
        (1, 1, 1),   # diagonal /
        (-1, 1, 1),  # diagonal \
    )

    def __init__(self, config: Lattice2DConfig, ncols: int, nrows: int):
        if ncols < 3 or nrows < 3:
            raise ValueError("lattice must be at least 3x3 sites")
        self.cfg = config
        self.ncols = int(ncols)
        self.nrows = int(nrows)
        self.nnodes = self.ncols * self.nrows
        a = config.spacing
        ii = np.tile(np.arange(self.ncols), self.nrows)
        jj = np.repeat(np.arange(self.nrows), self.ncols)
        self.x = ii * a
        self.y = jj * a
        self.ii = ii
        self.jj = jj

        bi: list[int] = []
        bj: list[int] = []
        bkind: list[int] = []
        for di, dj, kind in self._NEIGHBOURS:
            i1 = ii + di
            j1 = jj + dj
            ok = (i1 >= 0) & (i1 < self.ncols) & (j1 >= 0) & (j1 < self.nrows)
            n0 = np.where(ok, jj * self.ncols + ii, 0)
            n1 = np.where(ok, j1 * self.ncols + i1, 0)
            bi.append(n0[ok])
            bj.append(n1[ok])
            bkind.append(np.full(int(ok.sum()), kind, dtype=int))
        self.bonds = np.stack(
            [np.concatenate(bi), np.concatenate(bj), np.concatenate(bkind)], axis=1
        )
        self.nbonds = self.bonds.shape[0]
        n0 = self.bonds[:, 0]
        n1 = self.bonds[:, 1]
        dx = self.x[n1] - self.x[n0]
        dy = self.y[n1] - self.y[n0]
        self.blen = np.hypot(dx, dy)
        self.bdir = np.stack([dx / self.blen, dy / self.blen], axis=1)
        kind = self.bonds[:, 2]
        self.bk = np.where(kind == 0, config.axial_stiffness, config.diagonal_stiffness)

        self.active = np.ones(self.nbonds, dtype=bool)
        self.u = np.zeros(2 * self.nnodes)
        self._fixed: dict[int, tuple[float, float]] = {}
        self._loads = np.zeros(2 * self.nnodes)
        self._cache: dict[str, object] = {}

    # ------------------------------------------------------------------ setup
    def clone_geometry(self) -> "BondedLattice2D":
        other = BondedLattice2D.__new__(BondedLattice2D)
        other.cfg = self.cfg
        other.ncols, other.nrows, other.nnodes = self.ncols, self.nrows, self.nnodes
        other.x, other.y, other.ii, other.jj = self.x, self.y, self.ii, self.jj
        other.bonds, other.nbonds = self.bonds.copy(), self.nbonds
        other.blen, other.bdir, other.bk = self.blen, self.bdir, self.bk
        other.active = np.ones(other.nbonds, dtype=bool)
        other.u = np.zeros(2 * other.nnodes)
        other._fixed = {}
        other._loads = np.zeros(2 * other.nnodes)
        other._cache = {}
        return other

    def set_fixed(self, nodes: Iterable[int], ux: float | None = 0.0,
                  uy: float | None = 0.0) -> None:
        """Prescribe nodal displacements; ``None`` leaves that component free."""
        for n in np.atleast_1d(np.asarray(list(nodes), dtype=int)):
            self._fixed[int(n)] = (
                None if ux is None else float(ux),
                None if uy is None else float(uy),
            )

    def clear_conditions(self) -> None:
        self._fixed.clear()
        self._loads[:] = 0.0
        self._cache.clear()

    def add_load(self, node: int, fx: float, fy: float) -> None:
        self._loads[2 * int(node)] += fx
        self._loads[2 * int(node) + 1] += fy

    def add_distributed_load(self, nodes: Sequence[int], fx: float, fy: float) -> None:
        nodes = np.asarray(list(nodes), dtype=int)
        self._loads[2 * nodes] += fx / nodes.size
        self._loads[2 * nodes + 1] += fy / nodes.size

    def node(self, i: int, j: int) -> int:
        return int(j) * self.ncols + int(i)

    def kill_bonds(self, mask_or_indices) -> int:
        idx = np.asarray(mask_or_indices)
        if idx.dtype == bool:
            n = int((self.active & idx).sum())
            self.active &= ~idx
            self._cache.clear()
            return n
        before = int(self.active[idx].sum())
        self.active[idx] = False
        self._cache.clear()
        return before

    # --------------------------------------------------------------- assembly
    def stiffness_matrix(self) -> sp.csr_matrix:
        act = self.active
        n0, n1 = self.bonds[act, 0], self.bonds[act, 1]
        k = self.bk[act]
        nx, ny = self.bdir[act, 0], self.bdir[act, 1]
        nn = {(0, 0): nx * nx, (0, 1): nx * ny, (1, 0): nx * ny, (1, 1): ny * ny}
        rows, cols, vals = [], [], []
        for (A, sA) in ((n0, 1.0), (n1, -1.0)):
            for (B, sB) in ((n0, 1.0), (n1, -1.0)):
                for a in (0, 1):
                    for b in (0, 1):
                        rows.append(2 * A + a)
                        cols.append(2 * B + b)
                        vals.append(k * (sA * sB) * nn[(a, b)])
        mat = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(2 * self.nnodes, 2 * self.nnodes),
        )
        return mat.tocsr()

    # ---------------------------------------------------------------- solving
    def solve(self) -> np.ndarray:
        """Solve the linear elastic equilibrium for the current bond set."""
        K = self.stiffness_matrix()
        fixed_dofs = []
        fixed_vals = []
        pairs = []
        for n, (ux, uy) in self._fixed.items():
            if ux is not None:
                pairs.append((2 * n, ux))
            if uy is not None:
                pairs.append((2 * n + 1, uy))
        pairs.sort()
        fixed_dofs = np.asarray([p[0] for p in pairs], dtype=int)
        fixed_vals = np.asarray([p[1] for p in pairs], dtype=float)
        all_dofs = np.arange(2 * self.nnodes)
        free = np.setdiff1d(all_dofs, fixed_dofs)
        u = np.zeros(2 * self.nnodes)
        u[fixed_dofs] = fixed_vals
        f = self._loads.copy()
        rhs = f[free] - K[free][:, fixed_dofs] @ u[fixed_dofs]
        Kff = K[free][:, free].tocsc()
        # tiny Tikhonov term: particles cut loose by crack growth have an empty
        # row (a mechanism); pinning them to zero displacement at 1e-9 relative
        # strength removes the singularity without touching the elastic solution
        diag_scale = float(Kff.diagonal().max()) if Kff.shape[0] else 1.0
        Kff = Kff + sp.identity(Kff.shape[0], format="csc") * (1e-9 * diag_scale)
        lu = spla.splu(Kff)
        u[free] = lu.solve(rhs)
        self.u = u
        self._cache = {}
        return u

    # -------------------------------------------------------------- kinematics
    def bond_state(self) -> dict[str, np.ndarray]:
        """Elongation, force (tension > 0) and stored energy of active bonds."""
        act = self.active
        n0, n1 = self.bonds[act, 0], self.bonds[act, 1]
        u = self.u
        du = np.stack([u[2 * n1] - u[2 * n0], u[2 * n1 + 1] - u[2 * n0 + 1]], axis=1)
        proj = np.einsum("ij,ij->i", du, self.bdir[act])
        elongation = proj
        strain = proj / self.blen[act]
        force = self.bk[act] * elongation
        energy = 0.5 * self.bk[act] * elongation**2
        return {
            "elongation": elongation,
            "strain": strain,
            "force": force,
            "energy": energy,
            "index": np.flatnonzero(act),
        }

    def energy(self) -> float:
        return float(self.bond_state()["energy"].sum())

    def external_work(self) -> float:
        return float(0.5 * self._loads @ self.u)

    def node_energy_density(self) -> np.ndarray:
        """Strain energy density per node [J/m**3] (sums to the total energy)."""
        st = self.bond_state()
        a, B = self.cfg.spacing, self.cfg.thickness
        V = a * a * B
        n0, n1 = self.bonds[st["index"], 0], self.bonds[st["index"], 1]
        out = np.zeros(self.nnodes)
        np.add.at(out, n0, 0.5 * st["energy"] / V)
        np.add.at(out, n1, 0.5 * st["energy"] / V)
        return out

    def node_stress(self) -> np.ndarray:
        """Virial (bond-force) stress per node, shape ``(nnodes, 2, 2)`` [Pa]."""
        st = self.bond_state()
        a, B = self.cfg.spacing, self.cfg.thickness
        V = a * a * B
        idx = st["index"]
        n0, n1 = self.bonds[idx, 0], self.bonds[idx, 1]
        f = st["force"]
        d = self.bdir[idx] * self.blen[idx][:, None]
        outer = f[:, None, None] * (d[:, :, None] * d[:, None, :]) / self.blen[idx][:, None, None]
        out = np.zeros((self.nnodes, 2, 2))
        np.add.at(out, n0, 0.5 * outer / V)
        np.add.at(out, n1, 0.5 * outer / V)
        return out

    def node_displacement_gradient(self) -> np.ndarray:
        """Least-squares displacement gradient from bonded neighbours.

        Only bonded neighbours enter the stencil, so the gradient is never taken
        across a crack face (where the field is discontinuous).
        """
        cached = self._cache.get("grad")
        if cached is not None:
            return cached
        st = self.bond_state()
        idx = st["index"]
        n0, n1 = self.bonds[idx, 0], self.bonds[idx, 1]
        a = self.cfg.spacing
        d = np.stack([self.x[n1] - self.x[n0], self.y[n1] - self.y[n0]], axis=1) / a
        du = np.stack([self.u[2 * n1] - self.u[2 * n0],
                       self.u[2 * n1 + 1] - self.u[2 * n0 + 1]], axis=1) / a
        S = np.zeros((self.nnodes, 2, 2))
        R = np.zeros((self.nnodes, 2, 2))
        np.add.at(S, n0, d[:, :, None] * d[:, None, :])
        np.add.at(S, n1, d[:, :, None] * d[:, None, :])
        np.add.at(R, n0, d[:, :, None] * du[:, None, :])
        np.add.at(R, n1, d[:, :, None] * du[:, None, :])
        # closed-form 2x2 inverse (batched np.linalg.solve is pathologically slow
        # for thousands of tiny systems in numpy >= 2.3)
        s00 = S[:, 0, 0] + 1e-6
        s11 = S[:, 1, 1] + 1e-6
        s01 = 0.5 * (S[:, 0, 1] + S[:, 1, 0])
        det = s00 * s11 - s01 * s01
        det = np.where(np.abs(det) < 1e-30, 1e-30, det)
        inv = np.empty_like(S)
        inv[:, 0, 0] = s11 / det
        inv[:, 1, 1] = s00 / det
        inv[:, 0, 1] = inv[:, 1, 0] = -s01 / det
        grad = np.einsum("nab,nbc->nac", inv, R)
        grad = np.transpose(grad, (0, 2, 1))  # [component, derivative direction]
        self._cache["grad"] = grad
        return grad

    # ------------------------------------------------------------------ crack
    def notch_bonds(self, i_col: int, height: float, drop_diagonals: bool = True):
        """Bond indices crossing the vertical lattice line ``x = i_col*a`` below ``height``."""
        n0, n1 = self.bonds[:, 0], self.bonds[:, 1]
        kind = self.bonds[:, 2]
        x0, x1 = self.x[n0], self.x[n1]
        ymid = 0.5 * (self.y[n0] + self.y[n1])
        crosses = np.minimum(x0, x1) < (i_col * self.cfg.spacing) + 1e-12
        crosses &= np.maximum(x0, x1) > (i_col * self.cfg.spacing) + 1e-12
        crosses &= ymid <= height + 1e-12
        if not drop_diagonals:
            crosses &= kind == 0
        return np.flatnonzero(crosses)

    def slit_bonds(self, j_row: int, extent: float, tol: float = 1e-9):
        """Bond indices of a zero-width slit on ``y = j_row*a`` from ``x=0`` to ``extent``.

        Two bond families are removed: those *crossing* the slit line, and the
        axial bonds lying *along* it (``|y - y_c| < tol`` with midpoint
        ``x <= extent``).  Without the second family the two faces stay stitched
        together by a bond chain and the "crack" is bridged, which corrupts every
        near-tip quantity downstream.
        """
        n0, n1 = self.bonds[:, 0], self.bonds[:, 1]
        y0, y1 = self.y[n0], self.y[n1]
        xmid = 0.5 * (self.x[n0] + self.x[n1])
        yc = j_row * self.cfg.spacing
        crossing = (np.minimum(y0, y1) < yc + tol) & (np.maximum(y0, y1) > yc + tol)
        crossing &= xmid <= extent + tol
        along = (np.abs(y0 - yc) < tol) & (np.abs(y1 - yc) < tol)
        along &= xmid <= extent + tol
        return np.flatnonzero(crossing | along)
