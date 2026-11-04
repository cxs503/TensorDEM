"""Anisotropic-continuum reference field for the Brazilian (diametral) disc.

Why this script exists
----------------------
`docs/dem3d_brazilian_hondros_closure.md` settled the *transient* question (the
static relaxed lattice, not a longer ramp, is what the DEM reaches) and left the
verdict ``FAIL``: the centre principal-stress ratio of the bonded-sphere specimen
is ``-2.015`` against the Hondros point-load value ``-3``, and the centre tensile
stress overshoots ``2 P / (pi D t)`` by 30.5 %.

That report also stated the open question it could not answer:

    "未做晶格分辨/接触弧加密扫描（recipe 已写进 §6），故"加密后能否进 15%"未知"

i.e. is the miss a *discretisation* artefact (lattice spacing, finite contact
arc) that a finer specimen would remove, or a *constitutive* property of the
network?  A refinement study alone cannot separate the two, because refining the
lattice refines both.  What separates them is an independent continuum model that
carries the **same elastic tensor as the bond network** and the same contact arc,
but has no lattice spacing at all:

* if the anisotropic continuum reproduces the DEM centre field, the miss is
  caused by the cubic (Cauchy-solid) elastic anisotropy of a central-force
  network, and no refinement will ever reach Hondros;
* if the anisotropic continuum is close to Hondros, the miss is caused by the
  lattice/contact-arc discretisation, and refinement is the remedy.

This module supplies that reference.  It is a plane-stress / plane-strain Q4
finite-element disc under diametral loading, with the constitutive matrix built
from either

* ``isotropic``  -> ``C11 = lambda + 2 mu``, ``C12 = lambda``, ``C44 = mu``, or
* ``cubic``      -> the Born (affine) constants ``C11, C12, C44`` of the bond
  network, taken from :func:`tensordem.lattice_elasticity3d.periodic_elastic_constants`.

The finite-element implementation is validated *before* it is used as a
reference: the isotropic branch must reproduce the Hondros point-load solution
(``sigma_xx(0,0) = 2P/(pi D t)``, ``sigma_zz/sigma_xx = -3``, and a *uniform*
``sigma_xx`` along the loaded diameter) as the contact arc is narrowed, and the
mesh must be converged.  ``tests/test_brazilian_aniso_continuum.py`` asserts all
of that, so the anisotropic result is trusted for the same reason the lattice
result is: an independent closed form backs it in a limit where the answer is
known.

Geometry and boundary conditions
--------------------------------
Quarter disc ``x, z >= 0`` (``y`` is the thickness direction of the specimen).
The square parameter domain ``(u, v) in [0,1]^2`` is mapped onto the quarter disc
by the standard "squircle" blending

    x = R u sqrt(1 - v^2/2),     z = R v sqrt(1 - u^2/2),

which puts the origin at the square's corner and keeps well-shaped Q4 elements
everywhere -- a polar mesh would produce sliver elements exactly at the centre
we are trying to measure.  Symmetry conditions ``u_x = 0`` on ``x = 0`` and
``u_z = 0`` on ``z = 0`` replace the other three quarter discs; the diametral
load is a uniform vertical traction over the contact patch ``|x| <= w/2`` at
``z = R``, which is how the frictionless platen of the DEM acts.

Scope: numerical verification of the apparatus.  Nothing here calibrates an ice
material or certifies a tensile strength.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# 2x2 Gauss quadrature and the Q4 natural node coordinates (counter-clockwise).
_GAUSS = 1.0 / math.sqrt(3.0)
GAUSS_POINTS: tuple[tuple[float, float], ...] = (
    (-_GAUSS, -_GAUSS), (_GAUSS, -_GAUSS), (_GAUSS, _GAUSS), (-_GAUSS, _GAUSS))


# --------------------------------------------------------------------------- #
# constitutive matrices
# --------------------------------------------------------------------------- #
def isotropic_constants(young_modulus: float, poisson_ratio: float) -> tuple[float, float, float]:
    """Cubic constants of an isotropic solid: ``C11 = lambda + 2 mu`` etc."""
    if not math.isfinite(young_modulus) or young_modulus <= 0:
        raise ValueError("young_modulus must be finite and positive")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must be in (-1, 0.5)")
    lam = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    return lam + 2.0 * mu, lam, mu


def plane_stress_matrix(c11: float, c12: float, c44: float) -> np.ndarray:
    """In-plane (x-z) stiffness with ``sigma_yy = 0``.

    For cubic symmetry with the crystal axes on ``x, y, z`` the out-of-plane
    strain is ``eps_yy = -(C12/C11) (eps_xx + eps_zz)``, hence

        D11 = C11 - C12^2/C11,   D12 = C12 - C12^2/C11,   D33 = C44.
    """
    for name, value in (("c11", c11), ("c12", c12), ("c44", c44)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if c11 <= 0:
        raise ValueError("c11 must be positive for a stable plane-stress reduction")
    d11 = c11 - c12 * c12 / c11
    d12 = c12 - c12 * c12 / c11
    return np.array([[d11, d12, 0.0],
                     [d12, d11, 0.0],
                     [0.0, 0.0, c44]], dtype=float)


def plane_strain_matrix(c11: float, c12: float, c44: float) -> np.ndarray:
    """In-plane (x-z) stiffness with ``eps_yy = 0`` (bracket for the 3-D body)."""
    return np.array([[c11, c12, 0.0],
                     [c12, c11, 0.0],
                     [0.0, 0.0, c44]], dtype=float)


def constitutive_matrix(mode: str, constants: tuple[float, float, float]) -> np.ndarray:
    c11, c12, c44 = constants
    if mode == "isotropic-plane-stress":
        return plane_stress_matrix(c11, c12, c44)
    if mode == "cubic-plane-stress":
        return plane_stress_matrix(c11, c12, c44)
    if mode == "cubic-plane-strain":
        return plane_strain_matrix(c11, c12, c44)
    raise ValueError(f"unknown constitutive mode {mode!r}")


# --------------------------------------------------------------------------- #
# mesh
# --------------------------------------------------------------------------- #
def quarter_disc_mesh(radius: float, rings: int, sectors: int,
                      grading: float = 1.0) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Polar-structured Q4 mesh of the quarter disc ``x, z >= 0``.

    Node 0 is the disc centre; ring ``i`` (1..rings) carries ``sectors + 1`` nodes
    at ``theta = (pi/2) j / sectors``.  The innermost ring is meshed with collapsed
    quads (a repeated centre node), the standard way to close a polar mesh without
    leaving a hole at the axis -- a hole would perturb exactly the centre field we
    are measuring.

    An earlier revision blended a square onto the quarter disc with the squircle
    map ``x = R u sqrt(1 - v^2/2), z = R v sqrt(1 - u^2/2)``.  That map is
    *degenerate at the arc corner* (``det J = 0`` at ``u = v = 1``), so the outer
    elements were folded and the whole solution was meaningless (centre
    ``sigma_xx`` came out ~1e-3 of the applied load scale).  It is recorded here
    so that nobody re-derives it: keep the polar mesh.
    """
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if rings < 2 or sectors < 3:
        raise ValueError("rings must be >= 2 and sectors >= 3")
    if not math.isfinite(grading) or grading <= 0:
        raise ValueError("grading must be finite and positive")
    theta = 0.5 * math.pi * np.arange(sectors + 1) / sectors
    nodes = [(0.0, 0.0)]
    ring_index: list[np.ndarray] = []
    for i in range(1, rings + 1):
        r = radius * (i / rings) ** grading
        start = len(nodes)
        nodes.extend((r * math.cos(t), r * math.sin(t)) for t in theta)
        ring_index.append(np.arange(start, start + sectors + 1))
    node_array = np.asarray(nodes, dtype=float)
    elements: list[list[int]] = []
    first = ring_index[0]
    for j in range(sectors):
        elements.append([0, int(first[j]), int(first[j + 1]), 0])   # collapsed quad
    for i in range(1, rings):
        inner, outer = ring_index[i - 1], ring_index[i]
        for j in range(sectors):
            elements.append([int(inner[j]), int(outer[j]), int(outer[j + 1]), int(inner[j + 1])])
    boundaries = {
        "axis_bottom": np.array([0] + [int(r[0]) for r in ring_index]),                        # z = 0
        "axis_left": np.array([0] + [int(r[-1]) for r in ring_index]),                        # x = 0
        "arc": ring_index[-1],                                                               # x^2+z^2 = R^2
    }
    return node_array, np.asarray(elements, dtype=np.int64), boundaries


def _shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    d_xi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    d_eta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return n, d_xi, d_eta


def _batch_kinematics(coords: np.ndarray, xi: float, eta: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``B`` matrices (n_elements, 3, 8) and ``det J`` at one Gauss point.

    Written with ``einsum`` and a closed-form 2x2 inverse on purpose: the element
    loop of a small hand-written FE code otherwise dispatches thousands of tiny
    BLAS calls, and on a busy many-core host the thread pool dominates the
    runtime (measured: 24 min CPU for 56 s of work on 6400 elements).
    """
    _, d_xi, d_eta = _shape(xi, eta)
    # jacobian of the isoparametric map: J[a, b] = sum_n dN_a/dxi_b * x_n
    jac = np.zeros((len(coords), 2, 2))
    x, z = coords[:, :, 0], coords[:, :, 1]
    jac[:, 0, 0] = np.einsum("nk,k->n", x, d_xi)
    jac[:, 0, 1] = np.einsum("nk,k->n", z, d_xi)
    jac[:, 1, 0] = np.einsum("nk,k->n", x, d_eta)
    jac[:, 1, 1] = np.einsum("nk,k->n", z, d_eta)
    det_j = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
    if np.any(det_j <= 0.0):
        raise ValueError("non-positive element Jacobian")
    inv = np.empty_like(jac)
    inv[:, 0, 0] = jac[:, 1, 1] / det_j
    inv[:, 0, 1] = -jac[:, 0, 1] / det_j
    inv[:, 1, 0] = -jac[:, 1, 0] / det_j
    inv[:, 1, 1] = jac[:, 0, 0] / det_j
    # d_phys = J^-1 * d_nat  ->  rows are dN/dx and dN/dz
    d_dx = inv[:, 0, 0, None] * d_xi + inv[:, 0, 1, None] * d_eta
    d_dz = inv[:, 1, 0, None] * d_xi + inv[:, 1, 1, None] * d_eta
    b = np.zeros((len(coords), 3, 8))
    b[:, 0, 0::2] = d_dx
    b[:, 1, 1::2] = d_dz
    b[:, 2, 0::2] = d_dz
    b[:, 2, 1::2] = d_dx
    return b, det_j


def _element_dofs(elements: np.ndarray) -> np.ndarray:
    """Local dof map, interleaved per node as ``(u_x0, u_z0, u_x1, u_z1, ...)``.

    The interleaving must match the column layout of ``B`` (x rows in the even
    columns, z rows in the odd columns).  A block ordering ``(u_x0..u_x3,
    u_z0..u_z3)`` silently assembles the element stiffness onto the wrong global
    dofs: the solve still returns a smooth displacement field *and* zero
    equilibrium residuals, so only the uniform-strain patch test exposes it
    (recovered ``sigma_xx`` was 38x the exact value).
    """
    return np.stack([2 * elements, 2 * elements + 1], axis=2).reshape(len(elements), 8)


# --------------------------------------------------------------------------- #
# assembly, boundary conditions, loading
# --------------------------------------------------------------------------- #
def _assemble(nodes: np.ndarray, elements: np.ndarray, d_matrix: np.ndarray,
              thickness: float) -> coo_matrix:
    """Sparse global stiffness by vectorised element matrices."""
    dofs = _element_dofs(elements)
    coords = nodes[elements]
    ke = np.zeros((len(elements), 8, 8))
    for xi, eta in GAUSS_POINTS:
        b, det_j = _batch_kinematics(coords, xi, eta)
        ke += np.einsum("eki,kl,elj->eij", b, d_matrix, b) * det_j[:, None, None] * thickness
    rows = np.repeat(dofs, 8, axis=1)
    cols = np.tile(dofs, (1, 8))
    size = 2 * len(nodes)
    return coo_matrix((ke.reshape(-1), (rows.reshape(-1), cols.reshape(-1))),
                      shape=(size, size)).tocsr()


def _top_patch_loads(nodes: np.ndarray, arc: np.ndarray, patch_width: float,
                     total_load: float, thickness: float) -> np.ndarray:
    """Consistent nodal vertical forces for a uniform traction on ``|x| <= w/2``.

    ``total_load`` is the vertical force carried by the *full* patch (both halves
    of the disc); the quarter model receives half of it.  ``arc`` lists the
    boundary nodes of the outer ring, which are sorted by ``x`` here.
    """
    if patch_width <= 0:
        raise ValueError("patch_width must be positive")
    force = np.zeros((len(nodes), 2))
    traction = total_load / (patch_width * thickness)      # Pa, compressive (downwards)
    order = np.asarray(arc)[np.argsort(nodes[np.asarray(arc), 0])]
    accumulated = 0.0
    for a, b in zip(order[:-1], order[1:]):
        xa, xb = float(nodes[a, 0]), float(nodes[b, 0])
        overlap = min(xb, 0.5 * patch_width) - xa
        if overlap <= 0.0:
            continue
        share = 0.5 * traction * overlap * thickness
        force[a, 1] -= share
        force[b, 1] -= share
        accumulated += 2.0 * share
    if accumulated <= 0.0:
        raise ValueError("contact patch does not overlap any arc node interval")
    return force


def _symmetry_dofs(boundaries: dict[str, np.ndarray]) -> np.ndarray:
    """Dirichlet dofs: ``u_x = 0`` on ``x = 0``, ``u_z = 0`` on ``z = 0``."""
    fixed = [2 * int(node) for node in boundaries["axis_left"]]
    fixed += [2 * int(node) + 1 for node in boundaries["axis_bottom"]]
    return np.asarray(sorted(set(fixed)), dtype=np.int64)


def solve_disc(radius: float, thickness: float, rings: int, sectors: int,
               patch_width: float, total_load: float,
               d_matrix: np.ndarray) -> dict[str, Any]:
    """Solve one diametral-compression problem and return displacement + reactions."""
    nodes, elements, boundaries = quarter_disc_mesh(radius, rings, sectors)
    stiffness = _assemble(nodes, elements, d_matrix, thickness)
    load = _top_patch_loads(nodes, boundaries["arc"], patch_width, total_load, thickness)
    fixed = _symmetry_dofs(boundaries)
    free = np.setdiff1d(np.arange(2 * len(nodes)), fixed)
    reduced = stiffness[free][:, free]
    rhs = load.reshape(-1)[free]
    displacement = np.zeros((len(nodes), 2))
    displacement.reshape(-1)[free] = spsolve(reduced.tocsc(), rhs)
    flat = displacement.reshape(-1)
    reaction = stiffness @ flat - load.reshape(-1)
    applied = float(-load[:, 1].sum())
    vertical_fixed = np.asarray([d for d in fixed if d % 2 == 1], dtype=np.int64)
    supported = float(reaction[vertical_fixed].sum())
    balance = float(abs(reaction.sum() + load.sum()))
    return {
        "nodes": nodes,
        "elements": elements,
        "displacement": displacement,
        "applied_vertical_force_N": applied,
        "supported_vertical_force_N": supported,
        "global_balance_rel": balance / max(applied, 1e-30),
        "support_to_load_rel_error": abs(supported - applied) / max(applied, 1e-30),
        "max_displacement_m": float(np.abs(displacement).max()),
    }


def gauss_stresses(nodes: np.ndarray, elements: np.ndarray,
                   displacement: np.ndarray, d_matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Stress components at every Gauss point of the mesh (vectorised)."""
    dofs = _element_dofs(elements)
    coords = nodes[elements]
    ue = displacement.reshape(-1)[dofs]
    x_out, z_out, sigma_out, weight_out = [], [], [], []
    for xi, eta in GAUSS_POINTS:
        b, det_j = _batch_kinematics(coords, xi, eta)
        n, _, _ = _shape(xi, eta)
        strain = np.einsum("eki,ei->ek", b, ue)
        sigma = np.einsum("ei,li->el", strain, d_matrix)
        x_out.append(np.einsum("k,nk->n", n, coords[:, :, 0]))
        z_out.append(np.einsum("k,nk->n", n, coords[:, :, 1]))
        sigma_out.append(sigma)
        weight_out.append(det_j)
    sigma_all = np.concatenate(sigma_out, axis=0)
    return {
        "x": np.concatenate(x_out),
        "z": np.concatenate(z_out),
        "sigma_xx": sigma_all[:, 0],
        "sigma_zz": sigma_all[:, 1],
        "sigma_xz": sigma_all[:, 2],
        "weight": np.concatenate(weight_out),
    }


def window_mean(field: dict[str, np.ndarray], diameter: float,
                window_fraction: float = 0.1) -> dict[str, float]:
    """Area-weighted mean stress in ``|x|, |z| <= window_fraction * D``."""
    limit = window_fraction * diameter
    mask = (np.abs(field["x"]) <= limit) & (np.abs(field["z"]) <= limit)
    if not mask.any():
        raise ValueError("no Gauss points inside the centre window")
    weight = field["weight"][mask]
    return {
        "centre_gauss_points": int(mask.sum()),
        "centre_sigma_xx_Pa": float(np.average(field["sigma_xx"][mask], weights=weight)),
        "centre_sigma_zz_Pa": float(np.average(field["sigma_zz"][mask], weights=weight)),
        "centre_sigma_xz_Pa": float(np.average(field["sigma_xz"][mask], weights=weight)),
    }


def axial_profile(field: dict[str, np.ndarray], diameter: float,
                  window_fraction: float = 0.1, bins: int = 8) -> list[dict[str, float]]:
    """Area-weighted ``sigma_xx`` in axial bins, ``|x| <= window_fraction * D``."""
    limit = window_fraction * diameter
    mask = np.abs(field["x"]) <= limit
    edges = np.linspace(-0.5 * diameter, 0.5 * diameter, bins + 1)
    profile = []
    for k in range(bins):
        inside = mask & (field["z"] >= edges[k]) & (field["z"] < edges[k + 1])
        if not inside.any():
            continue
        profile.append({
            "z_over_D": float(0.5 * (edges[k] + edges[k + 1]) / diameter),
            "sigma_xx_Pa": float(np.average(field["sigma_xx"][inside],
                                            weights=field["weight"][inside])),
            "gauss_points": int(inside.sum()),
        })
    return profile


def axial_uniformity(profile: list[dict[str, float]]) -> float:
    """``(max - min) / |mean|`` of the axial ``sigma_xx`` profile, as the DEM reports it."""
    values = np.array([row["sigma_xx_Pa"] for row in profile])
    mean = values.mean()
    if mean == 0.0:
        return float("inf")
    return float((values.max() - values.min()) / abs(mean))


def run_case(*, radius: float, thickness: float, rings: int, sectors: int, patch_ratio: float,
             total_load: float, constitutive: str,
             constants: tuple[float, float, float]) -> dict[str, Any]:
    """One diametral-compression solve with the normalised Hondros metrics.

    Quantities are reported both in pascals and normalised by ``P / (D t)`` so
    that the DEM lattice can be compared without matching load levels.
    """
    diameter = 2.0 * radius
    patch_width = patch_ratio * diameter
    d_matrix = constitutive_matrix(constitutive, constants)
    solution = solve_disc(radius, thickness, rings, sectors, patch_width, total_load, d_matrix)
    field = gauss_stresses(solution["nodes"], solution["elements"],
                           solution["displacement"], d_matrix)
    centroid = window_mean(field, diameter)
    profile = axial_profile(field, diameter)
    normalisation = total_load / (diameter * thickness)
    ratio = (centroid["centre_sigma_zz_Pa"] / centroid["centre_sigma_xx_Pa"]
             if centroid["centre_sigma_xx_Pa"] else float("nan"))
    return {
        "constitutive": constitutive,
        "rings": rings,
        "sectors": sectors,
        "elements": int(len(solution["elements"])),
        "dof": int(2 * len(solution["nodes"])),
        "patch_ratio": patch_ratio,
        "patch_width_m": patch_width,
        "constants_C11_C12_C44_Pa": list(constants),
        "global_balance_rel": solution["global_balance_rel"],
        "support_to_load_rel_error": solution["support_to_load_rel_error"],
        "max_displacement_m": solution["max_displacement_m"],
        "applied_vertical_force_N": solution["applied_vertical_force_N"],
        "centre_particle_count": centroid["centre_gauss_points"],
        "centre_sigma_xx_Pa": centroid["centre_sigma_xx_Pa"],
        "centre_sigma_zz_Pa": centroid["centre_sigma_zz_Pa"],
        "centre_sigma_xz_Pa": centroid["centre_sigma_xz_Pa"],
        "centre_sigma_xx_over_P_on_Dt": centroid["centre_sigma_xx_Pa"] / normalisation,
        "centre_principal_ratio": ratio,
        # Hondros point-load references: sigma_xx = 2P/(pi D t), sigma_zz = -3 sigma_xx.
        "hondros_split_stress_Pa": 2.0 * normalisation / math.pi,
        "centre_tension_rel_error": abs(centroid["centre_sigma_xx_Pa"]
                                        / (2.0 * normalisation / math.pi) - 1.0),
        "centre_principal_ratio_rel_error": abs(ratio + 3.0) / 3.0,
        "axial_uniformity_rel": axial_uniformity(profile),
        "axial_profile": profile,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=0.15,
                        help="disc radius in metres (default: the brazil_elastic_field specimen)")
    parser.add_argument("--thickness", type=float, default=0.06)
    parser.add_argument("--load", type=float, default=1.0,
                        help="total diametral load in newtons; results are also normalised")
    parser.add_argument("--rings", type=int, default=60)
    parser.add_argument("--sectors", type=int, default=60)
    parser.add_argument("--patch-ratio", type=float, default=0.0668693921916482,
                        help="contact patch width / diameter (default: measured DEM value)")
    parser.add_argument("--constants", type=float, nargs=3, default=(433333.3333333333,
                                                                   233333.3333333333,
                                                                   233333.3333333333),
                        help="C11 C12 C44 in Pa (default: Born constants of the 26-neighbour lattice)")
    parser.add_argument("--constitutive", default="cubic-plane-stress",
                        choices=["complete-displacement"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    constants = tuple(float(value) for value in args.constants)
    result = run_case(radius=args.radius, thickness=args.thickness,
                      rings=args.rings, sectors=args.sectors,
                      patch_ratio=args.patch_ratio, total_load=args.load,
                      constitutive="cubic-plane-stress", constants=constants)
    summary = {key: value for key, value in result.items() if key != "axial_profile"}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()