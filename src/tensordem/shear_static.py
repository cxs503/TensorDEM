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
) -> dict[str, Any]:
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
    return {
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