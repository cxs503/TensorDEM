"""Kirchhoff plate FEM on a structured rectangle mesh -- the continuous (FEM)
half of the FEM -> DEM fracture handoff.

Why this element and this benchmark set
---------------------------------------
Level-ice failure is a *plate bending* problem: the ice sheet floats on water, so
the buoyancy reaction is a Winkler foundation ``k = rho_w g``, and the flexural
rigidity ``D = E t**3 / (12 (1 - nu**2))`` sets the characteristic length
``l_c = (D / k)**(1/4)`` that the existing ice modules of this package already
use (:mod:`tensordem.ice_beam_bending`, :mod:`tensordem.ice_crack_pattern`).
The sheet is therefore solved as a foundation plate, not as a beam, and the
flexural stress that drives the fracture criterion is
``sigma = 6 |M| / t**2``, the extreme-fibre stress of a plate strip.

The element is the classical 12-dof Adini-Clough-Melosh (ACM) rectangle:
``(w, w,x, w,y)`` at each corner, ``w`` a 12-term polynomial that is complete to
third order plus the two "twist" terms ``x**3 y`` and ``x y**3``.  It is the
standard non-conforming choice for thin plates (Adini & Clough 1961; Melosh
1963; Zienkiewicz, *The Finite Element Method*, ch. on plates) and its curvature
field is exact for any bicubic-deflection patch, which is what the fracture
criterion needs: the moment resultants must be accurate, not only the
deflection.

Why not a Mindlin Q4: the first implementation of this module used a 4-node
Mindlin element with selective reduced (1-point) shear integration.  Measured on
this worktree it *did* reproduce the deflection (7.5 % on the simply supported
square, not improving with refinement) but the centre bending moment was 71 %
low at 40x40 cells and still climbing -- i.e. the 1-point shear field carried
the spurious hourglass modes the element is known for, and since the fracture
criterion is a *moment* criterion that failure mode is fatal.  The ACM element
has no transverse shear energy at all, so no locking and no hourglass; its price
is that it is a thin-plate (Kirchhoff) element, valid for ``a / t >> 1``, which
is exactly the ice-sheet regime (``t ~ 0.5 m`` on ``10 m``, ``a/t = 20``) but is
*not* valid for a thick floe.  The validity boundary is stated in the module
docstring of :mod:`tensordem.fem_dem_fracture` and checked by a test that the
thin-plate assumption holds for the production geometry.

Conventions
-----------
* ``DOF`` order per node: ``(w, theta_x, theta_y)`` with ``theta_x = w,x`` and
  ``theta_y = w,y`` (so a clamped edge is ``w = theta_x = theta_y = 0``).
* curvature ``kappa = (w,xx ; w,yy ; 2 w,xy)``,
  ``M = Db @ kappa`` with ``Db = D [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]``.
  Magnitudes only are used downstream (``sigma = 6 max|M_i| / t**2``), so the
  sign convention of the textbook moment is not needed.
* ``M_1, M_2`` are the principal values; the principal direction reported is the
  eigenvector of the moment of *larger magnitude*, and the crack tracer extends a
  crack perpendicular to it (a mode-I crack opens under the tensile stress whose
  axis is that eigenvector, so the crack runs along the perpendicular).

Verification (tests in ``tests/test_fem_dem_fracture_handoff.py``, numbers in
``docs/FEM_DEM_FRACTURE_HANDOFF.md``)
--------------------------------------
* cylindrical bending of a long simply supported strip -> ``w = 5 q a**4 /
  (384 D)``;
* simply supported square plate under uniform load -> Navier double series,
  ``w_max = 0.00406 q a**4 / D`` and ``M_x(centre) = 0.0479 q a**2``
  (Timoshenko & Woinowsky-Krieger, Table 5);
* clamped square plate -> ``w_max = 0.00126 q a**4 / D``;
* Winkler plate under a local load on a large domain -> Hetenyi's closed form
  ``w_max = P l_c**2 / (8 D)`` (Hetenyi 1946).  This is *the* ice benchmark: it
  contains both ``D`` and the buoyancy ``k``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

__all__ = [
    "WATER_DENSITY",
    "GRAVITY",
    "PlateMaterial",
    "PlateMesh",
    "PressurePatch",
    "PlateSolution",
    "solve_plate",
    "hetenyi_point_load_deflection",
    "navier_ssss_square_deflection",
    "navier_ssss_square_moment",
    "clamped_square_deflection",
    "cylindrical_bending_deflection",
    "cylindrical_bending_moment",
]

WATER_DENSITY = 1025.0        # kg/m**3, sea water
GRAVITY = 9.80665             # m/s**2

_GAUSS3 = (-math.sqrt(0.6), 0.0, math.sqrt(0.6))
_GAUSS_WEIGHTS3 = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)

_PLATE_EDGES = ("x0", "x1", "y0", "y1")


# --------------------------------------------------------------------------
# materials and geometry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PlateMaterial:
    """Elastic plate properties.

    Defaults are first-year sea ice: ``E = 5 GPa``, ``nu = 0.3``,
    ``rho = 917 kg/m**3``, flexural strength ``0.7 MPa`` (mid-range of the
    0.5--1.0 MPa band reported for first-year ice, Timco & Weeks 2010).
    """

    young_modulus: float = 5.0e9
    poisson_ratio: float = 0.3
    thickness: float = 0.5
    density: float = 917.0
    sigma_flexure: float = 0.7e6

    def __post_init__(self) -> None:
        if self.young_modulus <= 0.0:
            raise ValueError("young_modulus must be positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must lie in (-1, 0.5)")
        if self.thickness <= 0.0:
            raise ValueError("thickness must be positive")
        if self.sigma_flexure <= 0.0:
            raise ValueError("sigma_flexure must be positive")

    @property
    def flexural_rigidity(self) -> float:
        """``D = E t**3 / (12 (1 - nu**2))`` [N m]."""
        return (self.young_modulus * self.thickness ** 3
                / (12.0 * (1.0 - self.poisson_ratio ** 2)))

    @property
    def area_density(self) -> float:
        """Mass per unit plate area ``rho t`` [kg/m**2]."""
        return self.density * self.thickness

    def characteristic_length(self, foundation_modulus: float) -> float:
        """``l_c = (D / k)**(1/4)`` [m] (Winkler-foundation decay length)."""
        if foundation_modulus <= 0.0:
            raise ValueError("foundation_modulus must be positive")
        return (self.flexural_rigidity / foundation_modulus) ** 0.25

    def failure_moment(self) -> float:
        """Moment resultant at first flexural cracking ``sigma_f t**2 / 6``."""
        return self.sigma_flexure * self.thickness ** 2 / 6.0


@dataclass(frozen=True)
class PlateMesh:
    """Structured ``nx x ny`` quadrilateral mesh of a rectangular plate.

    Cell ``(i, j)`` spans ``[i hx, (i+1) hx] x [j hy, (j+1) hy]`` and its flat
    index is ``j nx + i``; node ``(i, j)`` is ``j (nx + 1) + i``.  The crack
    tracer of :mod:`tensordem.fem_dem_fracture.crack_network` uses the same
    numbering, so the FEM stress field and the crack lattice are two views of a
    single discretisation -- which is what makes the floe partition exact.
    """

    nx: int = 40
    ny: int = 40
    length_x: float = 10.0
    length_y: float = 10.0

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError("need at least 2x2 cells")
        if self.length_x <= 0.0 or self.length_y <= 0.0:
            raise ValueError("plate dimensions must be positive")

    @property
    def hx(self) -> float:
        return self.length_x / self.nx

    @property
    def hy(self) -> float:
        return self.length_y / self.ny

    @property
    def n_nodes(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    @property
    def cell_area(self) -> float:
        return self.hx * self.hy

    @property
    def plate_area(self) -> float:
        return self.length_x * self.length_y

    def node_id(self, i: int, j: int) -> int:
        return j * (self.nx + 1) + i

    def node_index(self, node: int) -> tuple[int, int]:
        return node % (self.nx + 1), node // (self.nx + 1)

    def node_xy(self) -> np.ndarray:
        """``(n_nodes, 2)`` node coordinates."""
        i = np.arange(self.nx + 1)
        j = np.arange(self.ny + 1)
        xx, yy = np.meshgrid(i * self.hx, j * self.hy, indexing="xy")
        return np.stack([xx.ravel(), yy.ravel()], axis=1)

    def cell_index(self, cell: int) -> tuple[int, int]:
        return cell % self.nx, cell // self.nx

    def cell_ids(self) -> np.ndarray:
        """``(n_cells, 4)`` node ids of every cell, counter-clockwise."""
        ii, jj = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="xy")
        ii = ii.ravel()
        jj = jj.ravel()
        return np.stack(
            [
                self.node_id(ii, jj),
                self.node_id(ii + 1, jj),
                self.node_id(ii + 1, jj + 1),
                self.node_id(ii, jj + 1),
            ],
            axis=1,
        )

    def edge_node_mask(self, edge: str) -> np.ndarray:
        """Boolean mask of the nodes lying on one plate edge."""
        i = np.arange(self.nx + 1)
        j = np.arange(self.ny + 1)
        ii, jj = np.meshgrid(i, j, indexing="xy")
        if edge == "x0":
            on = ii == 0
        elif edge == "x1":
            on = ii == self.nx
        elif edge == "y0":
            on = jj == 0
        elif edge == "y1":
            on = jj == self.ny
        else:
            raise ValueError(f"unknown edge {edge!r}; expected one of {_PLATE_EDGES}")
        return on.ravel()

    def boundary_node_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_nodes, dtype=bool)
        for edge in _PLATE_EDGES:
            mask |= self.edge_node_mask(edge)
        return mask


@dataclass(frozen=True)
class PressurePatch:
    """Uniform pressure ``p`` [Pa] on an axis-aligned rectangle.

    Used for contact loads (structure face, pressure-ridge / rubble-field
    contact, ice keel).  The loading of a level ice sheet by a structure is
    never a smooth global pressure but a set of contact zones, and the number
    and placement of those zones is what decides where the sheet cracks -- so
    the load model is part of the fracture model.
    """

    center: tuple[float, float]
    size: tuple[float, float]
    pressure: float

    def bounds(self) -> tuple[float, float, float, float]:
        cx, cy = self.center
        sx, sy = self.size
        if sx <= 0.0 or sy <= 0.0:
            raise ValueError("patch size must be positive")
        return cx - 0.5 * sx, cx + 0.5 * sx, cy - 0.5 * sy, cy + 0.5 * sy


def patch_pressure(patches, x, y):
    """Summed pressure of ``patches`` at the points ``(x, y)`` [Pa]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    total = np.zeros(np.broadcast(x, y).shape)
    for patch in patches:
        x0, x1, y0, y1 = patch.bounds()
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        total = total + np.where(inside, patch.pressure, 0.0)
    return total


# --------------------------------------------------------------------------
# ACM element: polynomial interpolation and its inverse map
# --------------------------------------------------------------------------
def _monomials(x: float, y: float) -> np.ndarray:
    return np.array(
        [1.0, x, y, x * x, x * y, y * y,
         x ** 3, x * x * y, x * y * y, y ** 3, x ** 3 * y, x * y ** 3]
    )


def _monomial_dx(x: float, y: float) -> np.ndarray:
    return np.array(
        [0.0, 1.0, 0.0, 2 * x, y, 0.0,
         3 * x * x, 2 * x * y, y * y, 0.0, 3 * x * x * y, y ** 3]
    )


def _monomial_dy(x: float, y: float) -> np.ndarray:
    return np.array(
        [0.0, 0.0, 1.0, 0.0, x, 2 * y,
         0.0, x * x, 2 * x * y, 3 * y * y, x ** 3, 3 * x * y * y]
    )


def _monomial_dxx(x: float, y: float) -> np.ndarray:
    return np.array(
        [0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
         6 * x, 2 * y, 0.0, 0.0, 6 * x * y, 0.0]
    )


def _monomial_dyy(x: float, y: float) -> np.ndarray:
    return np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0,
         0.0, 0.0, 2 * x, 6 * y, 0.0, 6 * x * y]
    )


def _monomial_dxy(x: float, y: float) -> np.ndarray:
    return np.array(
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
         0.0, 2 * x, 2 * y, 0.0, 3 * x * x, 3 * y * y]
    )


def acm_inverse(hx: float, hy: float) -> np.ndarray:
    """``C^{-1}`` mapping the 12 nodal dofs to the 12 monomial coefficients.

    Corners in the order ``(0,0), (hx,0), (hx,hy), (0,hy)``, each contributing
    ``(w, w,x, w,y)``.  The matrix is identical for every element because the
    mesh is structured, so it is built once per mesh and cached by the caller.
    """
    rows = []
    for (x, y) in ((0.0, 0.0), (hx, 0.0), (hx, hy), (0.0, hy)):
        rows.append(_monomials(x, y))
        rows.append(_monomial_dx(x, y))
        rows.append(_monomial_dy(x, y))
    return np.linalg.inv(np.array(rows))


def _db_matrix(material: PlateMaterial) -> np.ndarray:
    d = material.flexural_rigidity
    nu = material.poisson_ratio
    return d * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]])


def _clipped_gauss_rule(low: float, high: float, lo: float, hi: float):
    """3-point Gauss rule clipped to ``[max(low,lo), min(high,hi)]`` ([] if disjoint)."""
    a = max(low, lo)
    b = min(high, hi)
    if b <= a:
        return []
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    return [(mid + half * xi, half * wi) for xi, wi in zip(_GAUSS3, _GAUSS_WEIGHTS3)]


def _pressure_load_vector(mesh, patches, cinv, w_dofs, n_dof) -> np.ndarray:
    """Consistent nodal loads of a piecewise-constant pressure field.

    The pressure of every patch is integrated *exactly*: element and patch are
    both axis-aligned rectangles, so their intersection is a rectangle and a
    3-point Gauss rule per direction integrates the cubic ACM interpolant
    without error.  This matters -- the first implementation used a 3x3 rule
    over the whole element, which silently lost load whenever a contact patch
    was smaller than a cell (measured: a 0.5 m patch on a 0.52 m cell lost 40 %
    of its load and pushed the Hetenyi benchmark from 0.9 % to 64 % error).
    """
    loads = np.zeros(n_dof)
    if not patches:
        return loads
    hx, hy = mesh.hx, mesh.hy
    for cell in range(mesh.n_cells):
        i, j = mesh.cell_index(cell)
        ox, oy = i * hx, j * hy
        dofs = w_dofs[cell]
        for patch in patches:
            x0, x1, y0, y1 = patch.bounds()
            rx = _clipped_gauss_rule(ox, ox + hx, x0, x1)
            if not rx:
                continue
            ry = _clipped_gauss_rule(oy, oy + hy, y0, y1)
            if not ry:
                continue
            for (gx, wx) in rx:
                for (gy, wy) in ry:
                    n_w = (_monomials(gx - ox, gy - oy) @ cinv)[0::3]
                    loads[dofs] += wx * wy * patch.pressure * n_w
    return loads


# --------------------------------------------------------------------------
# solution container
# --------------------------------------------------------------------------
@dataclass
class PlateSolution:
    """FE solution plus per-cell moment / flexural stress post-processing.

    Per-cell arrays are indexed like :meth:`PlateMesh.cell_ids`
    (``j * nx + i``).
    """

    mesh: PlateMesh
    material: PlateMaterial
    foundation_modulus: float
    w: np.ndarray                 # (n_nodes,) deflection [m]
    theta_x: np.ndarray           # (n_nodes,) w,x
    theta_y: np.ndarray           # (n_nodes,) w,y
    moment_x: np.ndarray          # (n_cells,) M_x at the cell centroid
    moment_y: np.ndarray
    moment_xy: np.ndarray
    moment_principal_1: np.ndarray   # larger-|M| principal moment (signed)
    moment_principal_2: np.ndarray
    principal_angle: np.ndarray       # direction of principal moment 1 [rad]
    sigma_max: np.ndarray             # (n_cells,) 6 max(|M1|,|M2|) / t**2
    load_amplitude: float = 1.0
    extra: dict = field(default_factory=dict)

    @property
    def max_sigma(self) -> float:
        return float(np.max(self.sigma_max))

    @property
    def max_deflection(self) -> float:
        return float(np.max(np.abs(self.w)))

    def critical_cell(self) -> int:
        """Flat index of the most stressed cell."""
        return int(np.argmax(self.sigma_max))

    def node_sigma_max(self) -> np.ndarray:
        """``(n_nodes,)`` largest adjacent-cell stress (conservative nodal value)."""
        cells = self.mesh.cell_ids()
        out = np.zeros(self.mesh.n_nodes)
        for k in range(4):
            np.maximum.at(out, cells[:, k], self.sigma_max)
        return out

    def cell_center(self, cell: int) -> tuple[float, float]:
        return ((self.mesh.cell_index(cell)[0] + 0.5) * self.mesh.hx,
                (self.mesh.cell_index(cell)[1] + 0.5) * self.mesh.hy)

    def scale_to_failure(self) -> float:
        """Load amplitude at which ``max sigma`` reaches ``sigma_f``.

        The problem is linear elastic, so ``sigma`` is proportional to the load:
        the failure amplitude follows exactly from one solve,
        ``amplitude * sigma_f / sigma_max``, with no bisection tolerance.
        """
        if self.max_sigma <= 0.0:
            raise ValueError("zero stress field: cannot scale to failure")
        return self.load_amplitude * self.material.sigma_flexure / self.max_sigma

    def scaled(self, amplitude: float) -> "PlateSolution":
        """The same field re-labelled with a different load amplitude."""
        factor = amplitude / self.load_amplitude
        return PlateSolution(
            mesh=self.mesh,
            material=self.material,
            foundation_modulus=self.foundation_modulus,
            w=self.w * factor,
            theta_x=self.theta_x * factor,
            theta_y=self.theta_y * factor,
            moment_x=self.moment_x * factor,
            moment_y=self.moment_y * factor,
            moment_xy=self.moment_xy * factor,
            moment_principal_1=self.moment_principal_1 * factor,
            moment_principal_2=self.moment_principal_2 * factor,
            principal_angle=self.principal_angle.copy(),
            sigma_max=self.sigma_max * factor,
            load_amplitude=amplitude,
            extra=dict(self.extra),
        )


# --------------------------------------------------------------------------
# assembly / solve
# --------------------------------------------------------------------------
BC_KINDS = ("free", "simply_supported", "clamped")


def solve_plate(
    mesh: PlateMesh,
    material: PlateMaterial,
    patches=(),
    *,
    foundation_modulus: float = 0.0,
    load_amplitude: float = 1.0,
    boundary: str = "free",
    constrained_edges=_PLATE_EDGES,
) -> PlateSolution:
    """Assemble and solve the thin-plate bending problem.

    Parameters
    ----------
    patches:
        iterable of :class:`PressurePatch`; the total pressure is their sum.
    foundation_modulus:
        Winkler modulus ``k`` [N/m**3]; ``rho_w g = 10052`` for sea water.
        A free plate *needs* a foundation, otherwise the transverse translation
        is a mechanism (checked and rejected).
    boundary:
        ``"free"`` (natural BC -- correct for a floating floe supported by
        buoyancy), ``"simply_supported"`` (``w = 0`` on the constrained edges,
        rotations free so ``M_n = 0``) or ``"clamped"`` (``w = w,x = w,y = 0``).
    constrained_edges:
        which plate edges carry the ``boundary`` condition (default: all four).
    """
    if boundary not in BC_KINDS:
        raise ValueError(f"unknown boundary {boundary!r}")
    if foundation_modulus < 0.0:
        raise ValueError("foundation_modulus must be non-negative")
    if foundation_modulus == 0.0 and boundary == "free":
        raise ValueError(
            "a free plate without a Winkler foundation is a mechanism: "
            "add foundation stiffness or use a supported boundary condition"
        )
    for edge in constrained_edges:
        if edge not in _PLATE_EDGES:
            raise ValueError(f"unknown edge {edge!r}")

    hx, hy = mesh.hx, mesh.hy
    det_j = 0.25 * hx * hy
    cinv = acm_inverse(hx, hy)
    db = _db_matrix(material)
    cells = mesh.cell_ids()
    n_dof = 3 * mesh.n_nodes

    # Element dof layout (12,) = [(w, tx, ty) per corner].
    edof = np.empty((mesh.n_cells, 12), dtype=int)
    for k in range(4):
        edof[:, 3 * k + 0] = 3 * cells[:, k] + 0
        edof[:, 3 * k + 1] = 3 * cells[:, k] + 1
        edof[:, 3 * k + 2] = 3 * cells[:, k] + 2

    # 3x3 Gauss rule on the unit reference square, exact to degree 5 per
    # direction -- i.e. exact for the cubic ACM interpolation of w.
    gauss_local = [(_GAUSS3[a] * 0.5 * hx + 0.5 * hx,
                    _GAUSS3[b] * 0.5 * hy + 0.5 * hy,
                    _GAUSS_WEIGHTS3[a] * _GAUSS_WEIGHTS3[b] * det_j)
                   for a in range(3) for b in range(3)]

    # Bending and Winkler stiffness (identical for every element of a structured
    # mesh, so it is built once).
    ke = np.zeros((12, 12))
    foundation = np.zeros((12, 12))
    for (x, y, wgt) in gauss_local:
        bb = np.array([_monomial_dxx(x, y), _monomial_dyy(x, y),
                       2.0 * _monomial_dxy(x, y)]) @ cinv
        ke += wgt * (bb.T @ db @ bb)
        n_w = (_monomials(x, y) @ cinv)[0::3]      # only the w interpolants
        foundation[0::3, 0::3] += wgt * np.outer(n_w, n_w)
    if foundation_modulus > 0.0:
        ke = ke + foundation_modulus * foundation

    rows = np.repeat(edof, 12, axis=1).ravel()
    cols = np.tile(edof, (1, 12)).ravel()
    vals = np.tile(ke.ravel(), mesh.n_cells)
    stiffness = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()

    w_dofs = edof[:, 0::3]
    loads = _pressure_load_vector(mesh, patches, cinv, w_dofs, n_dof)

    # Dirichlet boundary conditions.
    constrained = np.zeros(n_dof, dtype=bool)
    if boundary != "free":
        edge_mask = np.zeros(mesh.n_nodes, dtype=bool)
        for edge in constrained_edges:
            edge_mask |= mesh.edge_node_mask(edge)
        node_ids = np.nonzero(edge_mask)[0]
        constrained[3 * node_ids] = True
        if boundary == "clamped":
            constrained[3 * node_ids + 1] = True
            constrained[3 * node_ids + 2] = True

    free = ~constrained
    solution = np.zeros(n_dof)
    solution[free] = spsolve(stiffness[free][:, free].tocsc(), loads[free] * load_amplitude)
    if np.any(~np.isfinite(solution)):
        raise RuntimeError("plate solve produced non-finite values")

    # Post-process the moment field at every cell centroid.
    kappa = np.empty((mesh.n_cells, 3))
    for cell in range(mesh.n_cells):
        local = solution[edof[cell]]
        x = 0.5 * hx
        y = 0.5 * hy
        bb = np.array([_monomial_dxx(x, y), _monomial_dyy(x, y),
                       2.0 * _monomial_dxy(x, y)]) @ cinv
        kappa[cell] = bb @ local
    moments = kappa @ db.T

    moment_x = moments[:, 0]
    moment_y = moments[:, 1]
    moment_xy = moments[:, 2]
    mean = 0.5 * (moment_x + moment_y)
    radius = np.hypot(0.5 * (moment_x - moment_y), moment_xy)
    m_plus = mean + radius
    m_minus = mean - radius
    angle_plus = 0.5 * np.arctan2(2.0 * moment_xy, moment_x - moment_y)
    swap = np.abs(m_plus) < np.abs(m_minus)
    m1 = np.where(swap, m_minus, m_plus)
    angle = np.where(swap, angle_plus + 0.5 * math.pi, angle_plus)
    m2 = np.where(swap, m_plus, m_minus)
    sigma = 6.0 * np.maximum(np.abs(m1), np.abs(m2)) / material.thickness ** 2

    return PlateSolution(
        mesh=mesh,
        material=material,
        foundation_modulus=float(foundation_modulus),
        w=solution[0::3].copy(),
        theta_x=solution[1::3].copy(),
        theta_y=solution[2::3].copy(),
        moment_x=moment_x,
        moment_y=moment_y,
        moment_xy=moment_xy,
        moment_principal_1=m1,
        moment_principal_2=m2,
        principal_angle=angle,
        sigma_max=sigma,
        load_amplitude=float(load_amplitude),
        extra={"boundary": boundary,
               "constrained_edges": tuple(constrained_edges),
               "n_dof": n_dof},
    )


# --------------------------------------------------------------------------
# closed-form benchmarks
# --------------------------------------------------------------------------
def cylindrical_bending_deflection(
    uniform_pressure: float, span: float, material: PlateMaterial
) -> float:
    """``w_max = 5 q a**4 / (384 D)`` -- simply supported strip, uniform load."""
    return 5.0 * uniform_pressure * span ** 4 / (384.0 * material.flexural_rigidity)


def cylindrical_bending_moment(
    uniform_pressure: float, span: float
) -> float:
    """``M_max = q a**2 / 8`` -- the same strip, mid-span moment per unit width."""
    return uniform_pressure * span ** 2 / 8.0


def navier_ssss_square_deflection(
    uniform_pressure: float, side: float, material: PlateMaterial, n_terms: int = 41
) -> float:
    """Navier solution for the centre deflection of a simply supported square plate.

    ``w_max = (16 q a**4 / (pi**6 D)) sum_{m,n odd} 1/(m n (m**2+n**2)**2)``,
    converging to ``0.00406 q a**4 / D`` (Timoshenko & Woinowsky-Krieger,
    *Theory of Plates and Shells*, Table 5).
    """
    d = material.flexural_rigidity
    total = 0.0
    for m in range(1, n_terms, 2):
        for n in range(1, n_terms, 2):
            # At the plate centre sin(m pi/2) sin(n pi/2) = (-1)**((m+n)/2 - 1),
            # i.e. the odd modes alternate in sign there; dropping the sign (the
            # first implementation did) inflates w_max by ~5 % and M_x by ~37 %.
            sign = (-1.0) ** ((m - 1) // 2 + (n - 1) // 2)
            total += sign / (m * n * (m * m + n * n) ** 2)
    return 16.0 * uniform_pressure * side ** 4 * total / (math.pi ** 6 * d)


def navier_ssss_square_moment(
    uniform_pressure: float, side: float, material: PlateMaterial, n_terms: int = 41
) -> float:
    """``M_x`` at the centre of a simply supported square plate (Navier series).

    ``M_x = (16 q a**2 / pi**4) sum_{m,n odd} (m**2 + nu n**2)/(m n (m**2+n**2)**2)
    -> 0.0479 q a**2`` at ``nu = 0.3``.
    """
    nu = material.poisson_ratio
    total = 0.0
    for m in range(1, n_terms, 2):
        for n in range(1, n_terms, 2):
            sign = (-1.0) ** ((m - 1) // 2 + (n - 1) // 2)
            total += sign * (m * m + nu * n * n) / (m * n * (m * m + n * n) ** 2)
    return 16.0 * uniform_pressure * side ** 2 * total / math.pi ** 4


def clamped_square_deflection(
    uniform_pressure: float, side: float, material: PlateMaterial
) -> float:
    """``w_max = 0.00126 q a**4 / D`` for the clamped square plate (Timoshenko)."""
    return 0.00126 * uniform_pressure * side ** 4 / material.flexural_rigidity


def hetényi_point_load_deflection(
    point_load: float, material: PlateMaterial, foundation_modulus: float
) -> float:
    """Hetenyi (1946): ``w_max = P l_c**2 / (8 D)`` for an infinite foundation
    plate under a point load, ``l_c = (D/k)**(1/4)``.

    This is the closed form the ice sheet is verified against: it contains both
    the plate stiffness ``D`` and the buoyancy ``k``, and it is the small-loaded-
    area limit of the contact-load case used for the fracture run.
    """
    lc = material.characteristic_length(foundation_modulus)
    return point_load * lc ** 2 / (8.0 * material.flexural_rigidity)


# ASCII alias, so callers that cannot use the accent can still import it.
hetenyi_point_load_deflection = hetényi_point_load_deflection
__all__.append("hetenyi_point_load_deflection")