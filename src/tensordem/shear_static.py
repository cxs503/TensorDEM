"""Static (rate-independent) reference for the DEM3D direct-shear specimen.

Why this module exists
----------------------
``tensordem.shear_specimen.run_direct_shear`` measures an *apparent* shear modulus
by ramping the top layer of a bonded block at constant speed and fitting
``tau = R/area`` against ``gamma = disp/h``.  That measurement mixes three
different things:

1. the material constant ``C44 = 7 k / (3 a)`` of the infinite lattice,
2. a **cross-section definition bias**: the nominal area ``nx*a * ny*a`` counts
   boundary nodes whose lateral bonds do not exist, so the affine stress is
   underestimated by an ``O(1/nx + 1/ny)`` factor, and
3. a **free-surface / stubby-specimen relaxation** of the displacement field,
   which is genuinely rate independent.

Only after (2) and (3) are removed can (1) be estimated from a simulation, and
only then is a rate study meaningful.  ``docs/dem3d_shear_static_reference.md``
reports the numbers; this module supplies the two reference objects:

* :func:`affine_effective_area` -- the bond census that makes the *affine*
  traction come out exactly at ``C44 gamma`` instead of at ``0.787 C44 gamma``
  for the 9x5 default lattice.  Under affine shear this identity is exact and is
  asserted in the tests, so it is a closed-form reference, not a calibration.
* :func:`static_direct_shear` -- the equilibrium of the same bond network under
  exactly the solver's boundary conditions (bottom layer fixed, top layer x
  prescribed, ``x`` at the other faces free).  This is the ``rate -> 0`` limit of
  the driven test *inside the geometrically linear regime*, obtained by one
  sparse solve instead of a ramp.  It therefore separates effect (3) from any
  rate or inertia error.

The stiffness matrix is assembled from the solver's own bond list and validated
against a finite-difference Jacobian of ``IceDEM3D.forces`` in the tests, so no
force law is re-typed from memory.  The equilibrium is cross-checked against the
solver's own ``forces`` routine (free-DOF residual and bottom reaction), which is
an independent code path.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .dem3d import DEM3DConfig, IceDEM3D
from .shear_specimen import (
    direct_shear_specimen_area,
    direct_shear_specimen_height,
    lattice_spacing,
    shear_modulus_closed_form,
)

__all__ = [
    "affine_effective_area",
    "bond_census",
    "shear_stiffness_matrix",
    "static_direct_shear",
    "layer_profile",
]

#: Number of upward (dz = 1) bonds per interior node and the weight
#: ``sum l^2 n_x^2 n_z^2 = 7 a^2 / 3`` they carry under affine shear ``u_x = gamma z``.
AFFINE_WEIGHT_PER_NODE = 7.0 / 3.0


def _specimen_config(config: DEM3DConfig | None) -> DEM3DConfig:
    base = config or DEM3DConfig()
    return replace(base, fix_bottom=True, fix_x_edges=False, drag=0.0)


def _upward_bottom_bonds(sim: IceDEM3D) -> torch.Tensor:
    """Mask of bonds that start on the fixed bottom layer and rise above it."""
    i, j = sim.pairs.T
    p = sim.initial_positions
    z_bottom = p[:, 2].min()
    bottom = torch.isclose(p[:, 2], z_bottom)
    return bottom[i] & (p[j, 2] > z_bottom)


def bond_census(config: DEM3DConfig | None = None) -> dict[str, float]:
    """Count the bonds that actually transmit x-traction across the shear plane.

    ``weight = sum_bonds l^2 n_x^2 n_z^2`` over the bonds attached to the fixed
    layer.  For an interior node exploiting the full 26-neighbour set this is
    ``7 a^2 / 3``; boundary nodes own fewer bonds, which is exactly the deficit
    the nominal area ignores.
    """
    sim = IceDEM3D(_specimen_config(config))
    a = lattice_spacing(sim.config)
    mask = _upward_bottom_bonds(sim)
    d = (sim.initial_positions[sim.pairs[:, 1]] - sim.initial_positions[sim.pairs[:, 0]])[mask]
    length = torch.linalg.vector_norm(d, dim=1)
    normal = d / length[:, None]
    weight = float((length.square() * normal[:, 0].square() * normal[:, 2].square()).sum())
    nodes = float(len(torch.unique(sim.pairs[:, 0][mask])))
    return {
        "bond_count": float(int(mask.sum())),
        "node_count": nodes,
        "weight": weight,
        "weight_per_node": weight / max(nodes, 1.0),
        "lattice_spacing_m": a,
        "nominal_area_m2": direct_shear_specimen_area(sim.config),
    }


def affine_effective_area(config: DEM3DConfig | None = None) -> float:
    """Area that makes the affine reaction equal ``C44 * gamma`` exactly.

    ``A_eff = weight / (7 a^2 / 3) * a^2``; for a specimen whose fixed layer is
    entirely interior nodes it reduces to the nominal ``nx a * ny a``.
    """
    census = bond_census(config)
    a = census["lattice_spacing_m"]
    return census["weight"] / AFFINE_WEIGHT_PER_NODE


def shear_stiffness_matrix(sim: IceDEM3D) -> sparse.csr_matrix:
    """Assembled translational stiffness matrix ``K`` (3N x 3N) of the bond network.

    Each live axial bond contributes ``k n_a n_b`` to the four ``(i, j)`` blocks.
    The geometric term ``k * extension / |d| * (I - n n^T)`` vanishes at the
    undeformed state, which is the configuration this matrix is used at.
    """
    k = sim.config.bond_stiffness
    i, j = sim.pairs.T
    delta = sim.positions[j] - sim.positions[i]
    length = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    normal = (delta / length.clamp_min(torch.finfo(delta.dtype).eps)).cpu().numpy()
    alive = sim.alive.cpu().numpy().astype(bool)
    i = i.cpu().numpy()[alive]
    j = j.cpu().numpy()[alive]
    normal = normal[alive]
    n = len(sim.positions)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for a_ in range(3):
        for b_ in range(3):
            block = k * normal[:, a_] * normal[:, b_]
            rows.extend((3 * i + a_, 3 * j + a_, 3 * i + a_, 3 * j + a_))
            cols.extend((3 * i + b_, 3 * j + b_, 3 * j + b_, 3 * i + b_))
            vals.extend((block, block, -block, -block))
    return sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * n, 3 * n),
    ).tocsr()


def static_direct_shear(
    config: DEM3DConfig | None = None,
    *,
    gamma: float = 1.0e-3,
    return_displacement: bool = False,
    include_state: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], "IceDEM3D", np.ndarray]:
    """Linear elastic equilibrium of the direct-shear specimen (rate-free limit).

    Boundary conditions are the solver's: the bottom layer is a hard support
    (all three components), the top layer has ``u_x = gamma * h`` prescribed only,
    and the remaining degrees of freedom are free.  Returns tractions normalised
    by the nominal area and by :func:`affine_effective_area`, plus residuals that
    prove the discrete equilibrium was actually reached.
    """
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and positive")
    cfg = _specimen_config(config)
    sim = IceDEM3D(cfg)
    a = lattice_spacing(cfg)
    height = direct_shear_specimen_height(cfg)
    area_nominal = direct_shear_specimen_area(cfg)
    area_effective = affine_effective_area(cfg)
    c44 = shear_modulus_closed_form(cfg)

    positions = sim.initial_positions
    z = positions[:, 2].numpy()
    top = np.isclose(z, z.max())
    bottom = np.isclose(z, z.min())
    fixed = np.array([3 * p + c for p in np.where(bottom)[0] for c in range(3)])
    prescribed = 3 * np.where(top)[0]
    free = np.setdiff1d(np.arange(3 * len(positions)), np.concatenate([fixed, prescribed]))
    if free.size == 0:
        raise ValueError("specimen has no free degrees of freedom")

    K = shear_stiffness_matrix(sim)
    u = np.zeros(3 * len(positions))
    u[prescribed] = gamma * height
    u[free] = sparse_linalg.spsolve(
        K[free][:, free].tocsc(),
        -K[free][:, prescribed] @ u[prescribed],
    )
    residual = K @ u
    reaction = float(residual[fixed[0::3]].sum())
    tau_nominal = abs(reaction) / area_nominal
    tau_effective = abs(reaction) / area_effective

    # Independent check through the solver's own force routine.
    sim.positions = sim.initial_positions + torch.tensor(
        u.reshape(-1, 3), dtype=sim.initial_positions.dtype)
    forces, _ = sim.forces(update_fracture=False)
    solver_reaction = float(forces[sim.fixed].sum(dim=0)[0])
    free_mask = torch.ones(len(positions), dtype=torch.bool)
    free_mask[torch.tensor(np.where(bottom)[0])] = False
    free_mask[torch.tensor(np.where(top)[0])] = False
    free_residual = float(forces[free_mask].norm())
    bond_force_scale = cfg.bond_stiffness * gamma * a

    displacement = u.reshape(-1, 3)
    payload = {
        "gamma": gamma,
        "C44_Pa": c44,
        "height_m": height,
        "area_nominal_m2": area_nominal,
        "area_effective_m2": area_effective,
        "area_ratio": area_effective / area_nominal,
        "reaction_N": reaction,
        "tau_nominal_Pa": tau_nominal,
        "tau_effective_Pa": tau_effective,
        "G_nominal_over_C44": tau_nominal / gamma / c44,
        "G_effective_over_C44": tau_effective / gamma / c44,
        "affine_traction_over_C44": abs(reaction) / area_effective / (c44 * gamma),
        "max_free_displacement_over_gamma_h": float(np.abs(u[free]).max() / (gamma * height)),
        "solver_reaction_N": solver_reaction,
        "reaction_relative_error": abs(solver_reaction + reaction) / max(abs(reaction), 1e-30),
        "free_residual_over_bond_force": free_residual / max(bond_force_scale, 1e-30),
        "particle_count": int(len(positions)),
        "profile": layer_profile(displacement, z, gamma * height),
    }
    if return_displacement:
        payload["displacement_m"] = displacement
        payload["initial_positions_m"] = sim.initial_positions.cpu().numpy()
    if include_state:
        return payload, sim, displacement
    return payload


def layer_profile(displacement: np.ndarray, z: np.ndarray, gamma_h: float) -> list[dict[str, float]]:
    """Mean / peak axial displacement of each lattice layer, normalised by gamma*h."""
    rows = []
    for level in np.unique(z):
        mask = np.isclose(z, level)
        rows.append({
            "z_rel": float((level - z.min()) / (z.max() - z.min())),
            "mean_ux_over_gamma_h": float(displacement[mask, 0].mean() / gamma_h),
            "spread_ux_over_gamma_h": float(
                (displacement[mask, 0].max() - displacement[mask, 0].min()) / gamma_h),
            "mean_uz_over_gamma_h": float(displacement[mask, 2].mean() / gamma_h),
        })
    return rows

def _lattice_indices(sim: IceDEM3D) -> np.ndarray:
    """Integer ``(nx, ny, nz)`` lattice coordinates of every particle."""
    a = lattice_spacing(sim.config)
    p = sim.initial_positions.cpu().numpy()
    return np.rint((p - p.min(axis=0)) / a).astype(int)


def interior_traction(
    sim: IceDEM3D,
    displacement: np.ndarray,
    *,
    gamma: float,
    margin: int = 1,
    layer: int | None = None,
) -> dict[str, float]:
    """Shear traction carried by the *interior* bonds crossing the bottom plane.

    The whole-layer reaction of :func:`static_direct_shear` mixes the bulk
    response with the free-surface boundary layer, so it only approaches
    ``C44 * gamma`` as the cross-section grows.  This routine measures the same
    bond network on the sub-window whose nodes sit at least ``margin`` lattice
    spacings from every lateral face, i.e. outside the boundary layer.  Bond
    forces are recomputed from the displacement field with the bond law
    ``f = k * (n . (u_j - u_i)) * n`` (the same law ``IceDEM3D.forces`` uses),
    and the window area is censused with the affine identity of
    :func:`affine_effective_area` restricted to the same bonds, so numerator and
    denominator are consistent -- under affine shear the ratio is again
    ``C44 * gamma`` exactly.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    cfg = sim.config
    a = lattice_spacing(cfg)
    k = cfg.bond_stiffness
    idx = _lattice_indices(sim)
    pairs = sim.pairs.cpu().numpy()
    alive = sim.alive.cpu().numpy().astype(bool)
    p0 = sim.initial_positions.cpu().numpy()
    z = p0[:, 2]
    z_bottom = z.min()
    i, j = pairs[:, 0], pairs[:, 1]
    if layer is None:
        lower = z_bottom
    else:
        levels = np.unique(np.rint((z - z_bottom) / a))
        if layer < 0 or layer >= len(levels) - 1:
            raise ValueError("layer index out of range")
        lower = z_bottom + layer * a
    cross = np.isclose(idx[i, 2], int(round((lower - z_bottom) / a))) & (
        idx[j, 2] == int(round((lower - z_bottom) / a)) + 1) & alive
    # window: both endpoints at least `margin` sites away from every lateral face
    nx, ny, nz = idx.max(axis=0) + 1
    keep = np.ones(len(i), dtype=bool)
    for axis, n_axis in ((0, nx), (1, ny)):
        lo, hi = margin, n_axis - 1 - margin
        keep &= (idx[i, axis] >= lo) & (idx[i, axis] <= hi)
        keep &= (idx[j, axis] >= lo) & (idx[j, axis] <= hi)
    mask = cross & keep
    if not mask.any():
        raise ValueError("no bonds inside the requested interior window")
    iw, jw = i[mask], j[mask]
    d = p0[jw] - p0[iw]
    length = np.linalg.norm(d, axis=1)
    normal = d / length[:, None]
    du = displacement[jw] - displacement[iw]
    extension = np.einsum("bi,bi->b", normal, du)
    fx = k * extension * normal[:, 0]
    reaction = float(np.abs(fx.sum()))
    weight = float(np.sum(length ** 2 * normal[:, 0] ** 2 * normal[:, 2] ** 2))
    area_window = weight / AFFINE_WEIGHT_PER_NODE
    c44 = shear_modulus_closed_form(cfg)
    return {
        "bond_count": float(int(mask.sum())),
        "node_count": float(len(np.unique(np.concatenate([iw, jw])))),
        "area_window_m2": area_window,
        "tau_window_Pa": abs(reaction) / area_window,
        "G_window_over_C44": abs(reaction) / area_window / (c44 * gamma),
        "window_fraction_of_area": area_window / direct_shear_specimen_area(cfg),
        "margin_sites": float(margin),
        "layer": -1.0 if layer is None else float(layer),
        "gamma": gamma,
    }


def cross_section_series(
    sizes: "list[tuple[int, int, int]]",
    *,
    gamma: float = 1.0e-6,
    margin: int = 1,
) -> list[dict[str, Any]]:
    """Static direct-shear modulus of a family of specimens (one sparse solve each).

    Returns one row per ``(nx, ny, nz)`` holding the whole-layer modulus
    (``G_effective_over_C44``, as in :func:`static_direct_shear`) and the
    interior-window modulus of :func:`interior_traction`, both normalised by the
    infinite-lattice ``C44 = 7k/(3a)``.
    """
    rows: list[dict[str, Any]] = []
    for nx, ny, nz in sizes:
        cfg = replace(DEM3DConfig(), nx=nx, ny=ny, nz=nz)
        payload, sim, displacement = static_direct_shear(
            cfg, gamma=gamma, include_state=True)
        # a margin larger than the specimen would leave no interior bonds
        margin_used = min(margin, max(0, min(nx, ny) // 2 - 1))
        interior = interior_traction(sim, displacement, gamma=gamma,
                                     margin=margin_used)
        rows.append({
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "height_over_a": (nz - 1),
            "particle_count": payload["particle_count"],
            "reaction_N": payload["reaction_N"],
            "area_ratio": payload["area_ratio"],
            "G_nominal_over_C44": payload["G_nominal_over_C44"],
            "G_effective_over_C44": payload["G_effective_over_C44"],
            "G_window_over_C44": interior["G_window_over_C44"],
            "window_margin_used": margin_used,
            "window_fraction_of_area": interior["window_fraction_of_area"],
            "window_bond_count": interior["bond_count"],
            "reaction_relative_error": payload["reaction_relative_error"],
            "free_residual_over_bond_force": payload["free_residual_over_bond_force"],
            "linear_gamma_residual": payload["max_free_displacement_over_gamma_h"],
        })
    return rows


def fit_boundary_bias(
    rows: "list[dict[str, Any]]",
    *,
    key: str = "G_effective_over_C44",
    degree: int = 1,
) -> dict[str, float]:
    """Least-squares extrapolation of the free-surface bias to infinite cross-section.

    The measured deficit ``1 - G/G_inf`` is modelled as a polynomial in
    ``x = 1/nx + 1/ny`` (the fraction of the cross-section occupied by the
    free-surface boundary layer), so the intercept is a size-independent
    residual and the limit is ``1 - intercept``.  ``degree=2`` adds the
    curvature term for the Richardson-type estimate.
    """
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    x = np.array([1.0 / r["nx"] + 1.0 / r["ny"] for r in rows])
    y = np.array([1.0 - r[key] for r in rows])
    design = np.column_stack([np.ones_like(x)] + ([x, x ** 2] if degree == 2 else [x]))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "degree": float(degree),
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "curvature": float(coef[2]) if degree == 2 else 0.0,
        "limit_over_C44": float(1.0 - coef[0]),
        "rms_residual": float(np.sqrt(ss_res / len(x))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_points": float(len(x)),
    }

def fit_boundary_layer(
    margins: "list[int]",
    values: "list[float]",
    *,
    reference: float = 1.0,
) -> dict[str, float]:
    """Exponential decay length of the free-surface deficit ``reference - value``.

    ``deficit(m) ~ A exp(-m / lambda)`` is fitted by least squares on
    ``log(deficit)``; ``decay_length_sites`` is the fitted ``lambda`` in lattice
    spacings ``a``.  A finite ``lambda`` means the depression is a surface layer,
    not a bulk material effect.
    """
    m = np.asarray(margins, dtype=float)
    deficit = reference - np.asarray(values, dtype=float)
    keep = deficit > 0.0
    dropped = int((~keep).sum())
    m, deficit = m[keep], deficit[keep]
    if len(m) < 3:
        raise ValueError("need >=3 points with positive deficit (reference too low)")
    slope, intercept = np.polyfit(m, np.log(deficit), 1)
    pred = intercept + slope * m
    ss_res = float(np.sum((np.log(deficit) - pred) ** 2))
    ss_tot = float(np.sum((np.log(deficit) - np.log(deficit).mean()) ** 2))
    return {
        "decay_length_sites": float(-1.0 / slope),
        "amplitude": float(np.exp(intercept)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_points": float(len(m)),
        "dropped_points": float(dropped),
    }
