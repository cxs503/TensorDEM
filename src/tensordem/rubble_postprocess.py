"""Post-processing of DEM rubble piles: porosity field, force-chain network and
:math:`\\mu(I)` rheology calibration.

Motivation
----------
A broken-ice (rubble) pile is a *discrete* medium, but ship-resistance or
ice-load models need *equivalent continuum* parameters.  This module converts a
DEM particle configuration plus its contact set into

1. a **conservative porosity field** :math:`\\phi(x,t)` on a grid of size
   :math:`h = 5\\,d_{50}`;
2. a **Voronoi / Delaunay** pore-scale characterisation (per-particle local
   porosity, Delaunay edge statistics) of the same configuration;
3. **force-chain network statistics**: strong/weak split by contact-force
   quantile, coordination number :math:`Z`, orientation distribution and the
   anisotropy :math:`A_c` of the force-carrying structure tensor;
4. the macroscopic rheology point :math:`(I, \\mu)` with
   :math:`I = \\dot\\gamma\\, d_{50}/\\sqrt{P/\\rho}` and :math:`\\mu = \\tau/P`,
   the quadratic law :math:`\\mu(I) = \\mu_0 + \\alpha I + \\beta I^2` fitted over
   the three scenario datasets, and the equivalent viscosity
   :math:`\\eta = \\mu P/\\dot\\gamma`.

Acceptance (e) -- "no contact, :math:`\\mu(I)` undefined" -- is enforced
*structurally*: :func:`contact_stress`, :func:`inertial_number`,
:func:`rheology_point` and :func:`fit_mu_i_quadratic` raise
:class:`RubbleNoContactsError` / :class:`RubbleRheologyError` instead of
returning a silent zero.

Conventions
-----------
* 2-D plane strain, SI units: lengths in m, forces in N, 2-D stress in N/m.
  The scalar 3-D-equivalent stress in Pa is ``sigma_2d / thickness``.
* Stress is **compression positive**:
  :math:`\\sigma_{ab} = -\\frac{1}{A t}\\sum_c f_a\\, b_b` with ``f`` the contact
  force on the partner and :math:`b = x_j - x_i` the branch vector.
* Pressure :math:`P = \\tfrac12 \\operatorname{tr}\\sigma` and maximum shear
  :math:`\\tau = \\tfrac12|\\sigma_1-\\sigma_2|`, so in a simple shear cell
  :math:`\\tau \\to |\\sigma_{xy}|` and :math:`\\mu = \\tau/P`.
* The scalar shear rate entering :math:`I` is the equivalent one,
  :math:`\\dot\\gamma = \\sqrt{2\\,D{:}D}`; it equals ``dvx/dy`` for simple shear
  and ``2|\\dot\\epsilon|`` for constant-area plane-strain compression.

Porosity conservation
---------------------
The field is built from *exact* circle/rectangle intersection areas, so
:math:`\\int \\phi\\,dA = A_\\text{domain} - \\sum_c A_\\text{solid}` holds to
round-off.  :func:`porosity_conservation_audit` re-derives that identity from
the field itself; the test suite also runs a negative control (dropping the
clipped part of boundary-crossing particles gives a residual of order 1e-3), so
the audit is demonstrably not vacuous.

Evidence grading
----------------
Everything computed here is ``REAL`` (closed-form geometry, contact-force
statistics, least squares).  The literature anchor
:data:`GUIMARAES_2022_VISCOSITY_RANGE` is an order-of-magnitude window for
ice-debris/rubble effective viscosity at :math:`\\dot\\gamma = 0.1\\,s^{-1}`; a
digitised per-point comparison is ``WITHHELD`` (see
``docs/rubble_rheology_postprocess.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

__all__ = [
    "RubblePostProcessError",
    "RubbleNoContactsError",
    "RubbleRheologyError",
    "ParticleSample",
    "ContactView",
    "contact_view",
    "circle_box_area",
    "grid_porosity_field",
    "porosity_conservation_audit",
    "voronoi_local_porosity",
    "contact_stress",
    "stress_invariants",
    "inertial_number",
    "strong_weak_split",
    "contact_partition_audit",
    "fabric_tensor",
    "force_chain_statistics",
    "mu_of_i",
    "fit_mu_i_quadratic",
    "effective_viscosity",
    "rheology_point",
    "calibrate_mu_i",
    "GUIMARAES_2022_VISCOSITY_RANGE",
]


class RubblePostProcessError(ValueError):
    """Base class for rubble post-processing errors."""


class RubbleNoContactsError(RubblePostProcessError):
    """Raised when a rheological observable is requested without any contact.

    A contact-free configuration has no pressure, hence no inertial number and
    no friction coefficient; returning ``0.0`` would silently fabricate a
    constitutive point (negative control of acceptance (e)).
    """


class RubbleRheologyError(RubblePostProcessError):
    """Raised for undefined rheological inputs (``P <= 0``, ``gamma_dot <= 0``)."""


#: Order-of-magnitude window for the effective viscosity of an ice-rubble /
#: ice-debris flow at a nominal shear rate of ``0.1 1/s`` (Guimaraes et al. 2022,
#: ice-debris flow rheology).  Used only as an acceptance window.
GUIMARAES_2022_VISCOSITY_RANGE = (1.0e3, 1.0e5)


# --------------------------------------------------------------------- sample
@dataclass(frozen=True)
class ParticleSample:
    """Snapshot of a 2-D plane-strain particle configuration.

    ``positions`` are disk centres (m) inside ``[0, Lx] x [0, Ly]``.  A
    ``periodic`` axis is wrapped, so particle area crossing the seam still
    belongs to the domain; on a non-periodic axis the part of a disk outside the
    domain is excluded from every area (the wall owns it).
    """

    positions: np.ndarray
    radii: np.ndarray
    domain: tuple[float, float]
    thickness: float = 0.03
    periodic: tuple[bool, bool] = (False, False)
    label: str = ""
    time: float = 0.0

    def __post_init__(self) -> None:
        pos = np.asarray(self.positions, dtype=np.float64)
        rad = np.asarray(self.radii, dtype=np.float64)
        lx, ly = float(self.domain[0]), float(self.domain[1])
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise RubblePostProcessError("positions must have shape (N, 2)")
        if rad.shape != (pos.shape[0],):
            raise RubblePostProcessError("radii must have shape (N,)")
        if not np.isfinite(pos).all() or not np.isfinite(rad).all():
            raise RubblePostProcessError("positions and radii must be finite")
        if np.any(rad <= 0.0):
            raise RubblePostProcessError("radii must be positive")
        if not (lx > 0.0 and ly > 0.0):
            raise RubblePostProcessError("domain edges must be positive")
        if not (self.thickness > 0.0):
            raise RubblePostProcessError("thickness must be positive")
        object.__setattr__(self, "positions", pos)
        object.__setattr__(self, "radii", rad)
        object.__setattr__(self, "domain", (lx, ly))

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])

    @property
    def diameters(self) -> np.ndarray:
        return 2.0 * self.radii

    @property
    def d50(self) -> float:
        return float(np.median(self.diameters))

    @property
    def area(self) -> float:
        return float(self.domain[0] * self.domain[1])

    def solid_area(self) -> float:
        """Disk area inside the domain (periodic-aware, boundary-clipped)."""
        lx, ly = self.domain
        cells = np.tile(np.array([0.0, lx, 0.0, ly]), (self.n, 1))
        return float(np.sum(_image_cell_areas(self, np.arange(self.n), cells)))

    def area_fraction(self) -> float:
        return self.solid_area() / self.area

    def porosity(self) -> float:
        return 1.0 - self.area_fraction()


@dataclass(frozen=True)
class ContactView:
    """Disk-disk contact arrays in a post-processing friendly layout."""

    i: np.ndarray
    j: np.ndarray
    normal_force: np.ndarray
    force_on_i: np.ndarray

    @property
    def count(self) -> int:
        return int(self.i.shape[0])

    @property
    def magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.force_on_i, axis=1)

    def require_contacts(self, what: str = "contact statistics") -> "ContactView":
        if self.count == 0:
            raise RubbleNoContactsError(
                f"{what} is undefined without disk-disk contacts "
                "(no contacts left after filtering)")
        return self


def contact_view(contacts: Mapping[str, np.ndarray], *,
                 particle_mask: np.ndarray | None = None) -> ContactView:
    """Extract the disk-disk contacts (``kind == 0``) of an engine contact dict.

    Wall / hull contacts (``kind != 0``) are boundary tractions and stay out of
    the bulk network.  ``particle_mask`` keeps only contacts with *both*
    endpoints in a bulk subset.  The output is canonically ordered (``i < j``,
    lexsorted) so a given configuration always yields the same network.
    """
    missing = [k for k in ("i", "j", "kind", "f_on_i", "fn") if k not in contacts]
    if missing:
        raise RubblePostProcessError(f"contact record missing keys: {missing}")
    i = np.asarray(contacts["i"], dtype=np.int64)
    j = np.asarray(contacts["j"], dtype=np.int64)
    kind = np.asarray(contacts["kind"])
    fn = np.asarray(contacts["fn"], dtype=np.float64)
    force = np.asarray(contacts["f_on_i"], dtype=np.float64).reshape(-1, 2)
    disk = (kind == 0) & (j >= 0)
    if particle_mask is not None:
        mask = np.asarray(particle_mask, dtype=bool)
        disk &= mask[np.clip(i, 0, len(mask) - 1)] & mask[np.clip(j, 0, len(mask) - 1)]
    i, j, fn, force = i[disk], j[disk], fn[disk], force[disk]
    swap = i > j
    if np.any(swap):
        i, j = np.where(swap, j, i), np.where(swap, i, j)
        force = np.where(swap[:, None], -force, force)
    order = np.lexsort((j, i))
    return ContactView(i=i[order], j=j[order], normal_force=fn[order],
                       force_on_i=force[order])


# -------------------------------------------------- exact circle/rectangle area
def _sqrt_antiderivative(u: np.ndarray, r: np.ndarray) -> np.ndarray:
    u = np.clip(u, -r, r)
    s = np.sqrt(np.maximum(r * r - u * u, 0.0))
    return 0.5 * (u * s + r * r * np.arcsin(np.clip(u / r, -1.0, 1.0)))


def _cumulative_disk_area(X: np.ndarray, Y: np.ndarray, r: np.ndarray) -> np.ndarray:
    """``Q(X, Y) = area({u <= X, v <= Y} & disk(r))`` -- exact, vectorised.

    The rectangle/disk intersection is obtained by inclusion-exclusion on this
    cumulative measure, which avoids any case analysis on rectangle corners.
    """
    X, Y, r = np.broadcast_arrays(np.asarray(X, dtype=np.float64),
                                  np.asarray(Y, dtype=np.float64),
                                  np.asarray(r, dtype=np.float64))
    out = np.zeros(X.shape, dtype=np.float64)
    valid = (X > -r) & (Y > -r)          # outside the support -> zero measure
    if not np.any(valid):
        return out
    Xc = np.clip(X, -r, r)
    Yc = np.clip(Y, -r, r)
    Sr = _sqrt_antiderivative(-r, r)
    upper = Y >= r                        # full chord over u <= Xc
    out = np.where(upper, 2.0 * (_sqrt_antiderivative(Xc, r) - Sr), out)
    mid = valid & (~upper)
    if np.any(mid):
        u0 = np.sqrt(np.maximum(r * r - Yc * Yc, 0.0))
        e1 = np.minimum(Xc, -u0)          # |Y| >= s(u): full chord or nothing
        p1 = np.where(Yc > 0.0, 2.0 * (_sqrt_antiderivative(e1, r) - Sr), 0.0)
        lo2, hi2 = -u0, np.minimum(Xc, u0)   # |Y| < s(u): clipped chord Yc + s
        p2 = np.where(hi2 > lo2,
                      Yc * (hi2 - lo2) + (_sqrt_antiderivative(hi2, r)
                                          - _sqrt_antiderivative(lo2, r)),
                      0.0)
        p3 = np.where((Xc > u0) & (Yc > 0.0),
                      2.0 * (_sqrt_antiderivative(Xc, r) - _sqrt_antiderivative(u0, r)),
                      0.0)
        out = np.where(mid, p1 + p2 + p3, out)
    return np.where(valid, out, 0.0)


def _image_pair_areas(sample: ParticleSample, particle_index: np.ndarray,
                      cells: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Circle/rectangle areas for explicitly given image centres."""
    r = sample.radii[particle_index]
    X = np.stack([cells[:, 1] - cx, cells[:, 0] - cx,
                  cells[:, 1] - cx, cells[:, 0] - cx], axis=1)
    Y = np.stack([cells[:, 3] - cy, cells[:, 3] - cy,
                  cells[:, 2] - cy, cells[:, 2] - cy], axis=1)
    Q = _cumulative_disk_area(X, Y, np.repeat(r[:, None], 4, axis=1))
    return Q @ np.array([1.0, -1.0, -1.0, 1.0])


def circle_box_area(cx: float, cy: float, r: float,
                    x0: float, x1: float, y0: float, y1: float) -> float:
    """Exact area of ``disk((cx, cy), r) & [x0, x1] x [y0, y1]``."""
    if x1 <= x0 or y1 <= y0:
        raise RubblePostProcessError("rectangle must have positive extent")
    X = np.array([x1 - cx, x0 - cx, x1 - cx, x0 - cx])
    Y = np.array([y1 - cy, y1 - cy, y0 - cy, y0 - cy])
    sign = np.array([1.0, -1.0, -1.0, 1.0])   # (+,-,-,+) inclusion-exclusion
    Q = _cumulative_disk_area(X, Y, np.full(4, float(r)))
    return float(np.dot(sign, Q))


# --------------------------------------------------------- image bookkeeping
def _image_offsets(sample: ParticleSample) -> list[tuple[float, float]]:
    lx, ly = sample.domain
    ox = (0.0, -lx, lx) if sample.periodic[0] else (0.0,)
    oy = (0.0, -ly, ly) if sample.periodic[1] else (0.0,)
    return [(dx, dy) for dx in ox for dy in oy]


def _image_cell_areas(sample: ParticleSample, particle_index: np.ndarray,
                      cells: np.ndarray) -> np.ndarray:
    """Solid area contributed by each listed ``(particle, cell)`` pair."""
    lx, ly = sample.domain
    out = np.zeros(len(particle_index), dtype=np.float64)
    if len(particle_index) == 0:
        return out
    px, py, rad = sample.positions[:, 0], sample.positions[:, 1], sample.radii
    signs = np.array([1.0, -1.0, -1.0, 1.0])
    for dx, dy in _image_offsets(sample):
        cx = px[particle_index] + dx
        cy = py[particle_index] + dy
        r = rad[particle_index]
        keep = ((cx + r > 0.0) & (cx - r < lx) & (cy + r > 0.0) & (cy - r < ly))
        sel = np.flatnonzero(keep)
        if not len(sel):
            continue
        X = np.stack([cells[sel, 1] - cx[sel], cells[sel, 0] - cx[sel],
                      cells[sel, 1] - cx[sel], cells[sel, 0] - cx[sel]], axis=1)
        Y = np.stack([cells[sel, 3] - cy[sel], cells[sel, 3] - cy[sel],
                      cells[sel, 2] - cy[sel], cells[sel, 2] - cy[sel]], axis=1)
        Q = _cumulative_disk_area(X, Y, np.repeat(r[sel, None], 4, axis=1))
        out[sel] += Q @ signs
    return out


# ------------------------------------------------------------ porosity field
def grid_porosity_field(sample: ParticleSample, *,
                        cell_size: float | None = None,
                        h_over_d50: float = 5.0) -> dict[str, Any]:
    """Conservative porosity field on a grid of size ``h = 5 d50``.

    Every disk's solid area is split **exactly** between the grid cells it
    overlaps (circle/rectangle intersection), so the area-weighted integral of
    the field reproduces the global porosity of the same domain to round-off.
    Cell sizes are ``Lx/nx``, ``Ly/ny`` with ``n = round(L/h)`` so the grid tiles
    the domain exactly (cell anisotropy stays below 1 %).
    """
    lx, ly = sample.domain
    d50 = sample.d50
    h_target = float(cell_size) if cell_size is not None else h_over_d50 * d50
    if not h_target > 0.0:
        raise RubblePostProcessError("cell_size must be positive")
    nx = max(1, int(round(lx / h_target)))
    ny = max(1, int(round(ly / h_target)))
    hx, hy = lx / nx, ly / ny

    solid = np.zeros(nx * ny, dtype=np.float64)
    px, py, rad = sample.positions[:, 0], sample.positions[:, 1], sample.radii
    # Unwrapped cell indices: a wrapped particle keeps its true geometry and the
    # index is folded back into the grid at accumulation time, so no periodic
    # image is generated here (that would double count the wrapped sliver).
    ix0 = np.floor((px - rad) / hx).astype(np.int64)
    ix1 = np.floor((px + rad) / hx).astype(np.int64)
    iy0 = np.floor((py - rad) / hy).astype(np.int64)
    iy1 = np.floor((py + rad) / hy).astype(np.int64)
    if not sample.periodic[0]:
        ix0, ix1 = np.clip(ix0, 0, nx - 1), np.clip(ix1, 0, nx - 1)
    if not sample.periodic[1]:
        iy0, iy1 = np.clip(iy0, 0, ny - 1), np.clip(iy1, 0, ny - 1)
    iw, ih = ix1 - ix0 + 1, iy1 - iy0 + 1
    kw, kh = int(iw.max()), int(ih.max())
    pid = np.repeat(np.arange(sample.n), kw * kh)
    local = np.tile(np.arange(kw * kh), sample.n)
    off_x, off_y = local % kw, local // kw
    ok = (off_x < iw[pid]) & (off_y < ih[pid])
    pid, off_x, off_y = pid[ok], off_x[ok], off_y[ok]
    ix, iy = ix0[pid] + off_x, iy0[pid] + off_y
    cells = np.stack([ix * hx, (ix + 1) * hx, iy * hy, (iy + 1) * hy], axis=1)
    areas = _image_pair_areas(sample, pid, cells, px[pid], py[pid])
    in_grid = np.ones(len(pid), dtype=bool)
    if sample.periodic[0]:
        ix = np.mod(ix, nx)
    else:
        in_grid &= (ix >= 0) & (ix < nx)
    if sample.periodic[1]:
        iy = np.mod(iy, ny)
    else:
        in_grid &= (iy >= 0) & (iy < ny)
    np.add.at(solid, iy[in_grid] * nx + ix[in_grid], areas[in_grid])
    solid = solid.reshape(ny, nx)
    cell_area = hx * hy
    phi = 1.0 - solid / cell_area

    solid_total = sample.solid_area()
    phi_global = 1.0 - solid_total / sample.area
    phi_integral = float(np.sum(phi) * cell_area / sample.area)
    return {
        "phi": phi,
        "solid_area": solid,
        "cell_size": (hx, hy),
        "grid_shape": (ny, nx),
        "h_over_d50": float(0.5 * (hx + hy) / d50),
        "solid_area_from_field": float(np.sum(solid)),
        "solid_area_total": float(solid_total),
        "domain_area": sample.area,
        "phi_global": float(phi_global),
        "phi_integral": float(phi_integral),
        "phi_mean": float(np.mean(phi)),
        "phi_std": float(np.std(phi)),
        "conservation_error": float(abs(phi_integral - phi_global)),
        "d50": d50,
    }


def porosity_conservation_audit(field: Mapping[str, Any], sample: ParticleSample,
                                *, tol: float = 1.0e-6) -> dict[str, Any]:
    """Acceptance (a): ``|int(phi) dA / A - phi_global| <= tol`` from the field.

    The audit is deliberately *field-driven*: it re-integrates the stored
    porosity array and compares it with an independently computed global
    porosity.  A field that loses boundary-clipped particle area fails it.
    """
    phi = np.asarray(field["phi"], dtype=np.float64)
    hx, hy = field["cell_size"]
    domain_area = float(np.prod(sample.domain))
    phi_integral = float(np.sum(phi) * hx * hy / domain_area)
    phi_global = sample.porosity()
    err = abs(phi_integral - phi_global)
    return {
        "phi_integral": phi_integral,
        "phi_global": phi_global,
        "absolute_error": float(err),
        "tolerance": float(tol),
        "passed": bool(err <= tol),
    }

# --------------------------------------------------- Voronoi pore structure
def _augmented_points(sample: ParticleSample) -> tuple[np.ndarray, int]:
    """Real points plus periodic images / mirror images on closed axes."""
    lx, ly = sample.domain
    px, py = sample.positions[:, 0], sample.positions[:, 1]

    def axis_copies(v, length, periodic):
        """Identity plus periodic images / mirror images on a closed axis."""
        if periodic:
            return (v, v - length, v + length)
        return (v, -v, 2.0 * length - v)

    xs = axis_copies(px, lx, sample.periodic[0])
    ys = axis_copies(py, ly, sample.periodic[1])
    pts = [np.stack([x, y], axis=1) for x in xs for y in ys]
    return np.vstack(pts), sample.n


def voronoi_local_porosity(sample: ParticleSample) -> dict[str, Any]:
    """Per-particle local porosity from a Voronoi tessellation of the centres.

    ``phi_i = 1 - A_disk_i / A_voronoi_i``.  For a polydisperse packing the
    Voronoi cell is not the material volume of the particle, so this estimator
    is a *pore-scale descriptor* (spatial variability, hot spots), not a
    conservative field; the conservative field is :func:`grid_porosity_field`.
    Closed (non-periodic) boundaries are handled by mirroring the points so that
    no real cell is unbounded.
    """
    pts, n = _augmented_points(sample)
    vor = Voronoi(pts)
    areas = np.full(n, np.nan)
    n_unbounded = 0
    for k in range(n):
        region = vor.regions[vor.point_region[k]]
        if len(region) < 3 or -1 in region:
            n_unbounded += 1
            continue
        areas[k] = float(ConvexHull(vor.vertices[region]).volume)   # 2-D area
    disk_area = math.pi * sample.radii ** 2
    phi = 1.0 - disk_area / areas
    good = np.isfinite(phi)

    tri = Delaunay(pts)
    edges = set()
    for simplex in tri.simplices:
        for a in range(3):
            for b in range(a + 1, 3):
                u, v = int(simplex[a]), int(simplex[b])
                if u < n and v < n:
                    edges.add((min(u, v), max(u, v)))
    edge_arr = np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)
    lengths = np.zeros(len(edge_arr))
    if len(edge_arr):
        d = sample.positions[edge_arr[:, 1]] - sample.positions[edge_arr[:, 0]]
        if sample.periodic[0]:
            d[:, 0] -= sample.domain[0] * np.round(d[:, 0] / sample.domain[0])
        if sample.periodic[1]:
            d[:, 1] -= sample.domain[1] * np.round(d[:, 1] / sample.domain[1])
        lengths = np.linalg.norm(d, axis=1)
    d50 = sample.d50
    return {
        "phi_local": phi,
        "voronoi_area": areas,
        "phi_mean": float(np.mean(phi[good])) if good.any() else float("nan"),
        "phi_std": float(np.std(phi[good])) if good.any() else float("nan"),
        "phi_min": float(np.min(phi[good])) if good.any() else float("nan"),
        "phi_max": float(np.max(phi[good])) if good.any() else float("nan"),
        "phi_p05": float(np.percentile(phi[good], 5)) if good.any() else float("nan"),
        "phi_p95": float(np.percentile(phi[good], 95)) if good.any() else float("nan"),
        "phi_voronoi_weighted": (float(np.sum(phi[good] * areas[good])
                                       / np.sum(areas[good])) if good.any() else float("nan")),
        "phi_grid_global": float(sample.porosity()),
        "n_unbounded_cells": int(n_unbounded),
        "delaunay_n_edges": int(len(edge_arr)),
        "delaunay_mean_length_over_d50": (float(np.mean(lengths) / d50) if len(lengths) else float("nan")),
        "delaunay_p90_length_over_d50": (float(np.percentile(lengths, 90) / d50) if len(lengths) else float("nan")),
    }


# ------------------------------------------------------------------ stresses
def _branch_vectors(sample: ParticleSample, view: ContactView) -> np.ndarray:
    b = sample.positions[view.j] - sample.positions[view.i]
    if sample.periodic[0]:
        b[:, 0] -= sample.domain[0] * np.round(b[:, 0] / sample.domain[0])
    if sample.periodic[1]:
        b[:, 1] -= sample.domain[1] * np.round(b[:, 1] / sample.domain[1])
    return b


def contact_stress(sample: ParticleSample, view: ContactView, *,
                   particle_mask: np.ndarray | None = None) -> dict[str, Any]:
    """Love-Weber (virial) stress of the bulk contact network.

    Compression positive, in Pa (``sigma_2d / thickness``).  Raises
    :class:`RubbleNoContactsError` when the (optionally masked) network is empty
    -- a stress tensor cannot be defined from zero contacts.
    """
    if particle_mask is not None and view.count:
        mask = np.asarray(particle_mask, dtype=bool)
        keep = mask[view.i] & mask[view.j]
        view = ContactView(view.i[keep], view.j[keep],
                           view.normal_force[keep], view.force_on_i[keep])
    view.require_contacts("contact stress")
    b = _branch_vectors(sample, view)
    volume = sample.area * sample.thickness
    sigma = -(view.force_on_i.T @ b) / volume
    return {"sigma": sigma, "n_contacts": view.count, "volume": volume}


def stress_invariants(sigma: np.ndarray) -> dict[str, float]:
    """Pressure, maximum shear, friction coefficient and principal axes."""
    s = np.asarray(sigma, dtype=np.float64).reshape(2, 2)
    s = 0.5 * (s + s.T)
    eig = np.linalg.eigvalsh(s)                 # ascending
    lam1, lam2 = float(eig[1]), float(eig[0])
    P = 0.5 * (lam1 + lam2)
    tau = 0.5 * (lam1 - lam2)
    mu = tau / P if P > 0.0 else float("nan")
    return {
        "sigma_xx": float(s[0, 0]), "sigma_yy": float(s[1, 1]),
        "sigma_xy": float(s[0, 1]),
        "principal_max": lam1, "principal_min": lam2,
        "pressure": float(P), "max_shear": float(tau), "mu": float(mu),
        "K0": float(s[0, 0] / s[1, 1]) if s[1, 1] != 0.0 else float("nan"),
    }


def inertial_number(gamma_dot: float, d50: float, pressure: float,
                    density: float) -> float:
    """``I = gamma_dot d50 / sqrt(P / rho)``; undefined for ``P <= 0``."""
    if not (gamma_dot >= 0.0) or not np.isfinite(gamma_dot):
        raise RubbleRheologyError(f"shear rate must be finite and >= 0, got {gamma_dot}")
    if not (pressure > 0.0) or not np.isfinite(pressure):
        raise RubbleRheologyError(
            f"inertial number undefined for pressure {pressure} Pa "
            "(no confining stress / no contacts)")
    if not (d50 > 0.0) or not (density > 0.0):
        raise RubbleRheologyError("d50 and density must be positive")
    return float(gamma_dot * d50 / math.sqrt(pressure / density))


# -------------------------------------------------------------- force chains
def strong_weak_split(magnitudes: Sequence[float], *, mode: str = "quantile",
                      quantile: float = 0.8, k: float = 1.0
                      ) -> tuple[np.ndarray, np.ndarray, float]:
    """Split contacts into a strong and a weak network.

    ``mode='quantile'`` (default) puts the top ``1 - quantile`` fraction of
    contacts by force magnitude into the strong network -- a *fixed mass
    fraction*, as requested by the battle node.  ``mode='mean_k'`` is the
    Radjai/Peters criterion ``f > mean + k std``.  The returned masks are exact
    complements, so strong + weak always equals the total contact count.
    """
    mag = np.asarray(magnitudes, dtype=np.float64).reshape(-1)
    if mag.size == 0:
        raise RubbleNoContactsError("force-chain split undefined without contacts")
    if mode == "quantile":
        if not 0.0 < quantile < 1.0:
            raise RubblePostProcessError("quantile must be in (0, 1)")
        n_strong = int(round((1.0 - quantile) * mag.size))
        strong = np.zeros(mag.size, dtype=bool)
        if n_strong > 0:
            order = np.argsort(-mag, kind="stable")
            strong[order[:n_strong]] = True
        threshold = float(np.min(mag[strong])) if n_strong else float(np.max(mag))
    elif mode == "mean_k":
        threshold = float(np.mean(mag) + float(k) * np.std(mag))
        strong = mag > threshold
    else:
        raise RubblePostProcessError(f"unknown split mode {mode!r}")
    return strong, ~strong, threshold


def contact_partition_audit(magnitudes: Sequence[float], strong: np.ndarray,
                            weak: np.ndarray, *,
                            tol: float = 1.0e-10) -> dict[str, Any]:
    """Acceptance (b): strong + weak = total contacts and force sums add up."""
    mag = np.asarray(magnitudes, dtype=np.float64).reshape(-1)
    strong = np.asarray(strong, dtype=bool)
    weak = np.asarray(weak, dtype=bool)
    total = int(mag.size)
    n_strong, n_weak = int(np.sum(strong)), int(np.sum(weak))
    n_overlap = int(np.sum(strong & weak))
    n_uncovered = int(np.sum(~(strong | weak)))
    f_total = float(np.sum(mag))
    f_res = abs(float(np.sum(mag[strong])) + float(np.sum(mag[weak])) - f_total)
    f_res_rel = f_res / f_total if f_total > 0.0 else float("nan")
    count_res = float(abs(n_strong + n_weak - total))
    return {
        "n_total": total, "n_strong": n_strong, "n_weak": n_weak,
        "n_overlap": n_overlap, "n_uncovered": n_uncovered,
        "count_residual": count_res,
        "force_sum_residual": f_res,
        "force_sum_residual_relative": float(f_res_rel),
        "tolerance": float(tol),
        "passed": bool(count_res <= tol and n_overlap == 0 and n_uncovered == 0
                       and (f_res_rel <= tol if np.isfinite(f_res_rel) else True)),
    }


def fabric_tensor(normals: np.ndarray, weights: np.ndarray | None = None
                  ) -> dict[str, Any]:
    """Second-order structure tensor ``R = <n n>`` with anisotropy ``A_c``.

    ``A_c = 2 (l1 - l2) / (l1 + l2)``: 0 for an isotropic network (in 2-D
    ``l1 + l2 = 1``) and 2 for a single direction.  ``theta_deg`` is the major
    principal direction measured from the ``+x`` axis, in ``[0, 180)``.
    """
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 2)
    if len(n) == 0:
        raise RubbleNoContactsError("fabric tensor undefined without contacts")
    w = np.ones(len(n)) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if np.any(w < 0.0) or np.sum(w) <= 0.0:
        raise RubblePostProcessError("weights must be nonnegative with positive sum")
    R = (n * w[:, None]).T @ n / np.sum(w)
    eig, vec = np.linalg.eigh(R)
    l1, l2 = float(eig[1]), float(eig[0])
    theta = math.degrees(math.atan2(vec[1, 1], vec[0, 1])) % 180.0
    A = 2.0 * (l1 - l2) / (l1 + l2) if (l1 + l2) > 0.0 else float("nan")
    return {"R": R, "lambda_max": l1, "lambda_min": l2,
            "anisotropy": float(A), "theta_deg": float(theta)}


def force_chain_statistics(sample: ParticleSample, view: ContactView, *,
                           mode: str = "quantile", quantile: float = 0.8,
                           k: float = 1.0, n_bins: int = 18,
                           percolation_fraction: float = 0.9) -> dict[str, Any]:
    """Strong/weak network statistics: ``Z``, orientation distribution, ``A_c``.

    Contact directions use the unit branch vector; two opposite directions are
    equivalent, so the histogram and every tensor are built on ``[0, pi)`` by
    flipping the normal.  The strong network's connected components are reported
    together with their span along each axis, which is the mechanical backbone
    behind the resistance of a rubble pile.
    """
    view.require_contacts("force-chain statistics")
    mag = view.magnitude
    strong, weak, threshold = strong_weak_split(mag, mode=mode, quantile=quantile, k=k)
    audit = contact_partition_audit(mag, strong, weak)

    b = _branch_vectors(sample, view)
    norm = np.linalg.norm(b, axis=1)
    n = b / np.maximum(norm, 1e-30)[:, None]
    flip = n[:, 0] < 0.0
    n = np.where(flip[:, None], -n, n)
    theta = np.degrees(np.arctan2(n[:, 1], n[:, 0])) % 180.0

    bins = np.linspace(0.0, 180.0, n_bins + 1)
    counts, _ = np.histogram(theta, bins=bins)
    force_counts, _ = np.histogram(theta, bins=bins, weights=mag)
    centres = 0.5 * (bins[:-1] + bins[1:])
    f_tot = float(np.sum(mag))

    fabric_all = fabric_tensor(n)
    fabric_strong = (fabric_tensor(n[strong], weights=mag[strong])
                     if np.any(strong) else None)

    n_comp, largest_fraction = 0, 0.0
    percolates = [False, False]
    if np.any(strong):
        ii, jj = view.i[strong], view.j[strong]
        mat = coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(sample.n, sample.n))
        mat = (mat + mat.T).tocsr()
        _, labels = connected_components(mat, directed=False)
        touched = np.unique(np.concatenate([ii, jj]))
        comp_ids = np.unique(labels[touched])
        n_comp = int(len(comp_ids))
        # contact-count share of the largest component
        lab_of_contact = labels[ii]
        counts_by_comp = np.bincount(lab_of_contact)
        largest_fraction = float(counts_by_comp.max() / max(len(ii), 1))
        for cid in comp_ids:
            members = touched[labels[touched] == cid]
            for axis in (0, 1):
                span = float(np.ptp(sample.positions[members, axis]))
                if span >= percolation_fraction * sample.domain[axis]:
                    percolates[axis] = True

    return {
        "n_contacts": view.count,
        "n_strong": int(np.sum(strong)),
        "n_weak": int(np.sum(weak)),
        "strong_threshold": threshold,
        "split_mode": mode,
        "split_quantile": float(quantile) if mode == "quantile" else float("nan"),
        "partition": audit,
        "coordination_z": float(2.0 * view.count / sample.n),
        "coordination_z_strong": float(2.0 * int(np.sum(strong)) / sample.n),
        "coordination_z_weak": float(2.0 * int(np.sum(weak)) / sample.n),
        "fabric": {"lambda_max": fabric_all["lambda_max"],
                   "lambda_min": fabric_all["lambda_min"],
                   "anisotropy": fabric_all["anisotropy"],
                   "theta_deg": fabric_all["theta_deg"]},
        "fabric_strong": (None if fabric_strong is None else
                          {"lambda_max": fabric_strong["lambda_max"],
                           "lambda_min": fabric_strong["lambda_min"],
                           "anisotropy": fabric_strong["anisotropy"],
                           "theta_deg": fabric_strong["theta_deg"]}),
        "anisotropy_a_c": (float(fabric_strong["anisotropy"])
                           if fabric_strong is not None else float("nan")),
        "anisotropy_a_c_all": float(fabric_all["anisotropy"]),
        "theta_c_deg": (float(fabric_strong["theta_deg"])
                        if fabric_strong is not None else float("nan")),
        "force_share_strong": float(np.sum(mag[strong]) / f_tot) if f_tot > 0.0 else float("nan"),
        "force_mean": float(np.mean(mag)),
        "force_std": float(np.std(mag)),
        "force_max": float(np.max(mag)),
        "angle_bins_deg": centres.tolist(),
        "angle_counts": counts.tolist(),
        "angle_count_fraction": (counts / max(view.count, 1)).tolist(),
        "angle_force_fraction": (force_counts / f_tot).tolist() if f_tot > 0.0 else [],
        "strong_components": n_comp,
        "strong_largest_fraction": largest_fraction,
        "percolates_x": bool(percolates[0]),
        "percolates_y": bool(percolates[1]),
    }


# ------------------------------------------------------------- mu(I) rheology
def mu_of_i(I: Sequence[float] | float, mu0: float, alpha: float,
            beta: float) -> np.ndarray:
    """Quadratic rheology law ``mu(I) = mu0 + alpha I + beta I**2``."""
    arr = np.asarray(I, dtype=np.float64)
    return mu0 + alpha * arr + beta * arr ** 2


def fit_mu_i_quadratic(I: Sequence[float], mu: Sequence[float]) -> dict[str, Any]:
    """Least-squares fit of ``mu = mu0 + alpha I + beta I**2``.

    Returns the coefficients, ``R^2``, the residual standard deviation (with the
    ``n - 3`` denominator), the per-point residuals and the parameter
    covariance.  Raises :class:`RubbleRheologyError` when the dataset cannot
    define the law (fewer than 4 points, or no variance in ``mu``), and
    :class:`RubbleNoContactsError` when it is empty -- an undefined rheology is
    never silently reported as a zero.
    """
    I_arr = np.asarray(I, dtype=np.float64).reshape(-1)
    mu_arr = np.asarray(mu, dtype=np.float64).reshape(-1)
    if I_arr.size == 0 or mu_arr.size == 0:
        raise RubbleNoContactsError(
            "mu(I) fit undefined: no rheology points (no contacts in the dataset)")
    if I_arr.size != mu_arr.size:
        raise RubbleRheologyError("I and mu must have the same length")
    if I_arr.size < 4:
        raise RubbleRheologyError(
            f"at least 4 rheology points are required to fit 3 coefficients, got {I_arr.size}")
    if not np.isfinite(I_arr).all() or not np.isfinite(mu_arr).all():
        raise RubbleRheologyError("I and mu must be finite")
    if float(np.ptp(mu_arr)) == 0.0:
        raise RubbleRheologyError("mu has no variance: the quadratic law is not identifiable")
    design = np.stack([np.ones_like(I_arr), I_arr, I_arr ** 2], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, mu_arr, rcond=None)
    pred = design @ coeffs
    resid = mu_arr - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((mu_arr - np.mean(mu_arr)) ** 2))
    n = int(I_arr.size)
    dof = max(n - 3, 1)
    sigma2 = ss_res / dof
    cov = sigma2 * np.linalg.inv(design.T @ design)
    return {
        "mu0": float(coeffs[0]), "alpha": float(coeffs[1]), "beta": float(coeffs[2]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        "residual_std": float(math.sqrt(sigma2)),
        "rmse": float(math.sqrt(ss_res / n)),
        "max_abs_residual": float(np.max(np.abs(resid))),
        "ss_res": ss_res, "ss_tot": ss_tot,
        "n_points": n,
        "i_range": [float(np.min(I_arr)), float(np.max(I_arr))],
        "mu_range": [float(np.min(mu_arr)), float(np.max(mu_arr))],
        "residuals": resid.tolist(),
        "parameter_stderr": {
            "mu0": float(math.sqrt(cov[0, 0])),
            "alpha": float(math.sqrt(cov[1, 1])),
            "beta": float(math.sqrt(cov[2, 2])),
        },
    }


def effective_viscosity(mu: float, pressure: float, shear_rate: float) -> float:
    """Equivalent Newtonian viscosity ``eta = mu P / gamma_dot`` [Pa s]."""
    if not (shear_rate > 0.0) or not np.isfinite(shear_rate):
        raise RubbleRheologyError(f"viscosity undefined for shear rate {shear_rate}")
    if not (pressure > 0.0) or not np.isfinite(pressure):
        raise RubbleRheologyError(f"viscosity undefined for pressure {pressure}")
    if not np.isfinite(mu):
        raise RubbleRheologyError("viscosity undefined for non-finite mu")
    return float(mu * pressure / shear_rate)


def rheology_point(sample: ParticleSample, view: ContactView, *,
                   shear_rate: float, density: float,
                   particle_mask: np.ndarray | None = None,
                   d50: float | None = None) -> dict[str, Any]:
    """Assemble one ``(I, mu)`` point from a configuration + contact set.

    ``shear_rate`` is the *equivalent* shear rate ``sqrt(2 D:D)`` measured for
    the configuration (see the module docstring), never the nominal drive.  A
    contact-free configuration raises :class:`RubbleNoContactsError`.
    """
    st = contact_stress(sample, view, particle_mask=particle_mask)
    inv = stress_invariants(st["sigma"])
    d = sample.d50 if d50 is None else float(d50)
    if not (inv["pressure"] > 0.0):
        raise RubbleNoContactsError(
            "mu(I) undefined: the masked bulk carries no compressive stress")
    I = inertial_number(shear_rate, d, inv["pressure"], density)
    mu = inv["mu"]
    return {
        "I": float(I),
        "mu": float(mu),
        "pressure": inv["pressure"],
        "max_shear": inv["max_shear"],
        "shear_rate": float(shear_rate),
        "d50": float(d),
        "density": float(density),
        "n_contacts": int(st["n_contacts"]),
        "sigma_xx": inv["sigma_xx"], "sigma_yy": inv["sigma_yy"],
        "sigma_xy": inv["sigma_xy"], "K0": inv["K0"],
    }


def calibrate_mu_i(points: Sequence[Mapping[str, Any]], *,
                   viscosity_reference: tuple[float, float] = (0.1, 2.0e4),
                   r2_min: float = 0.95,
                   residual_std_max: float = 0.05,
                   viscosity_range: tuple[float, float] = GUIMARAES_2022_VISCOSITY_RANGE
                   ) -> dict[str, Any]:
    """Fit the pooled quadratic law and evaluate the acceptance criteria.

    ``points`` are the per-run rheology points (dicts with at least ``I``, ``mu``
    and ``scenario``).  A per-scenario fit is reported alongside the pooled one
    as a *consistency check*; the acceptance criterion is declared in the
    pre-registration and is applied to the pooled dataset
    (``prereg/rubble_rheology_prereg.json``).
    """
    I = np.array([float(p["I"]) for p in points], dtype=np.float64)
    mu = np.array([float(p["mu"]) for p in points], dtype=np.float64)
    fit = fit_mu_i_quadratic(I, mu)
    scenarios = np.array([str(p.get("scenario", "unknown")) for p in points])
    per_scenario: dict[str, Any] = {}
    for name in sorted(set(scenarios.tolist())):
        sel = scenarios == name
        if int(np.sum(sel)) >= 4:
            try:
                per_scenario[name] = fit_mu_i_quadratic(I[sel], mu[sel])
            except RubbleRheologyError as exc:
                per_scenario[name] = {"error": str(exc), "n_points": int(np.sum(sel))}
        else:
            per_scenario[name] = {"error": "fewer than 4 points", "n_points": int(np.sum(sel))}

    gamma_ref, p_ref = viscosity_reference
    I_ref = inertial_number(gamma_ref, float(np.mean([p["d50"] for p in points]))
                            if "d50" in points[0] else 1.0,
                            p_ref, float(np.mean([p.get("density", 917.0) for p in points])))
    mu_ref = float(mu_of_i(I_ref, fit["mu0"], fit["alpha"], fit["beta"]))
    eta_ref = effective_viscosity(mu_ref, p_ref, gamma_ref)
    return {
        "fit": fit,
        "per_scenario": per_scenario,
        "viscosity_reference": {
            "shear_rate": float(gamma_ref), "pressure": float(p_ref),
            "I": float(I_ref), "mu": mu_ref, "eta": eta_ref,
            "range": list(viscosity_range),
            "inside_range": bool(viscosity_range[0] <= eta_ref <= viscosity_range[1]),
        },
        "acceptance": {
            "r2_min": float(r2_min), "r2": fit["r2"],
            "r2_passed": bool(fit["r2"] >= r2_min),
            "residual_std_max": float(residual_std_max),
            "residual_std": fit["residual_std"],
            "residual_passed": bool(fit["residual_std"] <= residual_std_max),
            "viscosity_passed": bool(viscosity_range[0] <= eta_ref <= viscosity_range[1]),
        },
        "n_points": int(I.size),
        "n_scenarios": int(len(set(scenarios.tolist()))),
    }
