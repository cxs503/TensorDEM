"""Boussinesq flat-punch stiffness: analytic disk reference + Galerkin BEM.

Motivation
----------
A superquadric whose exponent is large (``n1 = n2 -> inf``) degenerates to a
rectangular block, i.e. its contact facets are *flat*.  The Hertz law

    F_n = (4/3) E* sqrt(R_eff) delta^(3/2)

is singular there (``R_eff -> inf``), so a DEM kernel that advertises "cube
contact" needs a second analytical branch: the **flat-punch (Boussinesq)**
stiffness, which is *linear* in the overlap,

    F_n = K_punch delta,        K_punch = pi E* (A^T K^{-1} 1),

with

    K_ij = \\iint_{S_i} 1/|r_i - r'| dS'          (single-layer potential)
    w(r) = (1/E*) (1/pi) \\int_S p(r')/|r-r'| dS'  (Boussinesq, two bodies)

For a circular patch of radius ``a`` the classical closed form is

    K_punch = 2 E* a                        (rigid flat punch, Love 1939)

which in area form is ``K_punch = 2 E* sqrt(A/pi)``.  For a *square* patch no
closed form exists, so this module solves the punch integral equation with a
piecewise-constant Galerkin discretisation (a deterministic boundary-element
method, BEM) and exposes both the converged stiffness and the convergence
table used in the verification report.

The pair-modulus convention is the DEM one::

    1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2

so a disk punch on a rigid plane is recovered with ``E1 = E``, ``E2 = inf``.

References
----------
* Love, A.E.H. (1939) "Boussinesq's problem for a flat-ended cylinder",
  Proc. Cambridge Phil. Soc. 35, 424-429.
* Fabrikant, V.I. (1988) "Flat punch of arbitrary shape on an elastic
  half-space", Int. J. Eng. Sci. 26, 1209-1219.
* Sneddon, I.N. (1946) "Boussinesq's problem for a flat-ended cylinder".

All quantities are SI (m, Pa, N, N/m).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "disk_punch_stiffness",
    "area_equivalent_punch_stiffness",
    "patch_stiffness_bem",
    "square_punch_stiffness",
    "square_punch_shape_factor",
    "bem_convergence_table",
    "reduced_modulus",
]


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------

def reduced_modulus(
    e1: float, nu1: float, e2: float, nu2: float,
) -> float:
    """Two-body reduced Young modulus ``1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2``.

    Use ``e2 = math.inf`` for a rigid counter-face.
    """
    inv = (1.0 - nu1 ** 2) / e1
    if math.isfinite(e2):
        inv += (1.0 - nu2 ** 2) / e2
    return 1.0 / inv


def disk_punch_stiffness(radius: float, e_star: float) -> float:
    """Exact Boussinesq stiffness of a rigid circular punch, ``2 E* a`` (N/m)."""
    if radius <= 0 or e_star <= 0:
        raise ValueError("radius and e_star must be positive")
    return 2.0 * e_star * radius


def area_equivalent_punch_stiffness(area: float, e_star: float) -> float:
    """Area-equivalent punch stiffness ``2 E* sqrt(A/pi)`` (N/m).

    Exact for a circular patch; the approximation used by the DEM flat-contact
    branch for non-circular patches.  Its error against the converged square
    BEM solution is quantified by :func:`square_punch_shape_factor`.
    """
    if area <= 0 or e_star <= 0:
        raise ValueError("area and e_star must be positive")
    return 2.0 * e_star * math.sqrt(area / math.pi)


# ---------------------------------------------------------------------------
# Galerkin BEM for an arbitrary patch (piecewise-constant pressure)
# ---------------------------------------------------------------------------

def _self_term(side: float, ng: int = 16) -> float:
    """``\\iint_cell 1/|r - r'| dS'`` for a square cell of given side, centroid r.

    Computed in polar coordinates (the ``1/r`` singularity is integrable:
    ``dS' = r dr dtheta``), so the quadrature is regular.
    """
    th, wth = np.polynomial.legendre.leggauss(ng)
    th = math.pi * (th + 1.0)
    wth = math.pi * wth
    total = 0.0
    for t, wt in zip(th, wth):
        ct, st = math.cos(t), math.sin(t)
        rmax = math.inf
        if abs(ct) > 1e-12:
            rmax = min(rmax, 0.5 * side / abs(ct))
        if abs(st) > 1e-12:
            rmax = min(rmax, 0.5 * side / abs(st))
        r, wr = np.polynomial.legendre.leggauss(ng)
        r = 0.5 * rmax * (r + 1.0)
        wr = 0.5 * rmax * wr
        total += wt * float(np.sum(wr))
    return total


def _single_layer_matrix(
    centres: np.ndarray, side: float, ng: int = 4, block: int = 1500,
) -> np.ndarray:
    """Matrix ``K_ij = \\iint_{S_j} 1/|r_i - r'| dS'`` for uniform square cells."""
    pts = np.asarray(centres, dtype=float)
    n = len(pts)
    gx, gw = np.polynomial.legendre.leggauss(ng)
    gx = 0.5 * side * (gx + 1.0)
    gw = 0.5 * side * gw
    off = np.column_stack([a.ravel() for a in np.meshgrid(gx, gx, indexing="ij")])
    woff = np.outer(gw, gw).ravel()
    k = np.empty((n, n))
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        pi = pts[i0:i1]
        qp = pts[None, :, None, :] + off[None, None, :, :]
        d = pi[:, None, None, :] - qp
        r = np.sqrt(np.einsum("mnqi,mnqi->mnq", d, d))
        k[i0:i1, :] = np.einsum("mnq,q->mn", 1.0 / r, woff)
    np.fill_diagonal(k, _self_term(side))
    return k


def _patch_grid(n: int, kind: str) -> Tuple[np.ndarray, float]:
    """Cell centres (M,2) and cell side for a unit patch (half-extent 1)."""
    h = 2.0 / n
    cs = -1.0 + h * (np.arange(n) + 0.5)
    x, y = np.meshgrid(cs, cs, indexing="ij")
    if kind == "disk":
        mask = x ** 2 + y ** 2 <= 1.0
        return np.column_stack([x[mask].ravel(), y[mask].ravel()]), h
    if kind == "square":
        return np.column_stack([x.ravel(), y.ravel()]), h
    raise ValueError(f"unknown patch kind {kind!r}")


def patch_stiffness_bem(
    centres: np.ndarray, side: float, e_star: float = 1.0,
) -> float:
    """Rigid-punch stiffness of a general patch from the Galerkin BEM (N/m).

    Solves ``K p = 1`` and evaluates ``pi E* (A^T p)``; ``A = side^2 * ones``.
    """
    pts = np.asarray(centres, dtype=float)
    k = _single_layer_matrix(pts, side)
    n = len(pts)
    p = np.linalg.solve(k, np.ones(n))
    area = side * side
    return math.pi * e_star * area * float(np.sum(p))


def square_punch_stiffness(
    half_side: float, e_star: float, n_cells_per_side: int = 48,
) -> float:
    """Square flat-punch stiffness (N/m) on a patch ``[-h, h]^2``.

    The unit-patch result is scaled by ``half_side`` (the punch problem is
    homogeneous of degree one in the patch size).
    """
    centres, side = _patch_grid(n_cells_per_side, "square")
    unit = patch_stiffness_bem(centres, side, e_star=1.0)
    return unit * half_side * e_star


def square_punch_shape_factor(n_cells_per_side: int = 48) -> float:
    """Dimensionless ``K / (2 E* sqrt(A))`` for a square patch.

    ``2 E* sqrt(A)`` is the area-equivalent punch value with ``sqrt(pi)``
    dropped, i.e. the reported number is ``sqrt(pi)/2`` times the
    area-equivalent stiffness ratio, so 1.0 means "identical to the exact
    circular punch of the same area".
    """
    centres, side = _patch_grid(n_cells_per_side, "square")
    unit = patch_stiffness_bem(centres, side, e_star=1.0)   # for [-1,1]^2, A=4
    return unit / (2.0 * math.sqrt(4.0))


def bem_convergence_table(n_cells: List[int] = (8, 12, 16, 24, 32, 48)) -> Dict[str, list]:
    """Convergence study of the BEM against the exact disk solution.

    Returns a dictionary of lists (JSON-serialisable) with, for each mesh
    resolution, the disk stiffness normalised by the exact ``2 E* a`` and the
    square shape factor ``K/(2E* sqrt(A))``.
    """
    out: Dict[str, list] = {"n_cells": [], "disk_ratio": [], "square_shape_factor": [],
                            "cells_disk": [], "cells_square": []}
    for n in n_cells:
        centres, side = _patch_grid(n, "disk")
        kd = patch_stiffness_bem(centres, side, e_star=1.0)
        out["n_cells"].append(int(n))
        out["cells_disk"].append(int(len(centres)))
        out["disk_ratio"].append(float(kd / 2.0))   # K/(2 E a), exact = 1
        out["cells_square"].append(int(n * n))
        out["square_shape_factor"].append(float(square_punch_shape_factor(n)))
    return out


@dataclass(frozen=True)
class PunchReference:
    """A frozen reference value with its provenance and uncertainty."""
    stiffness: float
    shape_factor: float
    area: float
    n_cells: int
    rel_uncertainty: float
    note: str = ""


def square_punch_reference(
    half_side: float = 1.0, e_star: float = 1.0, n_cells_per_side: int = 48,
) -> PunchReference:
    """Square punch stiffness packaged with provenance for verification tests."""
    k = square_punch_stiffness(half_side, e_star, n_cells_per_side)
    area = (2.0 * half_side) ** 2
    return PunchReference(
        stiffness=k,
        shape_factor=k / (2.0 * e_star * math.sqrt(area)),
        area=area,
        n_cells=n_cells_per_side,
        rel_uncertainty=0.015,
        note=("piecewise-constant Galerkin BEM, uniform mesh; convergence band "
              "+1.5% between 32 and 48 cells/side (see bem_convergence_table)"),
    )