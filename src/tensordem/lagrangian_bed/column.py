"""One-dimensional fluid column coupled to Lagrangian particles.

The fluid is carried on a 1-D grid of ``n_cells`` slabs.  With a constant
column area, continuity for an incompressible fluid gives a *uniform
superficial* velocity ``U(t)`` (volume flux per bed area); the interstitial
velocity is ``u_f = U / eps``.

Two solution modes are provided:

``steady_flow``  -- particles frozen, prescribed superficial velocity ``U``.
    The cell momentum balance
    ``0 = -eps dp/dz - beta (u_f - v_p) + rho eps g - rho U du_f/dz``
    is integrated from the outlet (p = 0) downwards.  This is the "semi-coupled"
    configuration: the fluid sees the (static) particles through the drag
    closure and the drag is felt back by the particles, but the particle
    positions are held fixed for the pressure-drop measurement.

``fluid_at_rest`` -- fluid at rest (``U = 0``), one-way coupling: the particles
    feel drag, buoyancy and pressure-gradient forces computed from the local
    slip, and the pressure field that *would* be needed to hold the fluid at
    rest is reconstructed by integrating the momentum balance.  This is the
    configuration for batch (Richardson-Zaki) sedimentation.

``unsteady_flow`` -- a 0-D momentum balance for the whole column, used for the
    time-step convergence (GCI) study; the fluid velocity is a single scalar
    ``U(t)`` that responds to the volume-integrated drag and pressure
    difference (open ends, ``p = 0`` at both ends).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import closures as cl


@dataclass
class FluidProperties:
    rho: float = 1000.0
    mu: float = 1.0e-3
    gravity: float = cl.GRAVITY


@dataclass
class ColumnProfile:
    """Static (per-call) description of the fluid column.

    ``area`` is the column cross-section [m^2]; static quantities are reported
    per unit bed area, so the voidage partition only needs it to convert the
    grain volumes into a solid fraction.
    """

    z_lo: float
    z_hi: float
    n_cells: int
    area: float = 1.0

    @property
    def dz(self) -> float:
        return (self.z_hi - self.z_lo) / self.n_cells

    @property
    def length(self) -> float:
        return self.z_hi - self.z_lo

    def edges(self) -> np.ndarray:
        return np.linspace(self.z_lo, self.z_hi, self.n_cells + 1)

    def centres(self) -> np.ndarray:
        e = self.edges()
        return 0.5 * (e[:-1] + e[1:])


def column_state(profile: ColumnProfile, pos, vel, radius, diameter):
    """Cell voidage and volume-weighted particle velocity for a column."""
    edges = profile.edges()
    eps = np.ones(profile.n_cells)
    vp_cells = np.zeros((profile.n_cells, 3))
    solid_weights = np.zeros(profile.n_cells)
    area = float(profile.area)
    for c in range(profile.n_cells):
        frac = cl.sphere_slab_fraction(radius, pos[:, 2], edges[c], edges[c + 1])
        solid_volume = float(np.sum(frac * cl.sphere_volume(2.0 * radius)))
        eps[c] = np.clip(1.0 - solid_volume / (profile.dz * area), 1e-3, 1.0)
        w = frac * cl.sphere_volume(2.0 * radius)
        solid_weights[c] = float(w.sum())
        if solid_weights[c] > 0:
            vp_cells[c] = (w[:, None] * vel).sum(axis=0) / solid_weights[c]
    return {"eps": eps, "v_p": vp_cells, "solid_volume": solid_weights * 1.0}


def steady_flow(
    profile: ColumnProfile,
    eps,
    v_p_cells,
    u_superficial,
    diameter,
    fluid: FluidProperties,
    include_convection: bool = True,
):
    """Integrate the steady 1-D fluid momentum balance for a fixed bed."""
    dz = profile.dz
    u_f = u_superficial / eps
    if include_convection:
        duf_dz = np.gradient(u_f, dz, edge_order=1)
    else:
        duf_dz = np.zeros_like(u_f)
    slip = u_f - v_p_cells[:, 2]
    beta, dense = cl.gidaspow_beta(eps, diameter, fluid.rho, fluid.mu, slip)
    # fluid momentum: 0 = -eps dp/dz - beta slip + rho eps g - rho U du_f/dz
    dpdz = (-beta * slip + fluid.rho * eps * fluid.gravity - fluid.rho * u_superficial * duf_dz) / eps
    # integrate from the outlet (p = 0 at z_hi) downwards
    edges = profile.edges()
    p_at_edges = np.zeros(profile.n_cells + 1)
    for c in range(profile.n_cells - 1, -1, -1):
        p_at_edges[c] = p_at_edges[c + 1] + dpdz[c] * dz
    p_cells = 0.5 * (p_at_edges[:-1] + p_at_edges[1:])
    dp_total = float(p_at_edges[0] - p_at_edges[-1])
    hydrostatic = float(fluid.rho * fluid.gravity * np.sum(eps) * dz)
    drag_per_volume = beta * slip
    return {
        "eps": eps,
        "u_f": u_f,
        "slip": slip,
        "beta": beta,
        "dense_branch": dense,
        "dpdz_cells": dpdz,
        "p_edges": p_at_edges,
        "p_cells": p_cells,
        "dp_total": dp_total,
        "dp_hydrostatic": hydrostatic,
        "dp_drag": dp_total + hydrostatic,  # measured drop minus hydrostatic head
        "drag_per_volume": drag_per_volume,
        "duf_dz": duf_dz,
    }


def drag_pressure_drop_loss(profile: ColumnProfile, eps, fluid: FluidProperties, u_superficial, diameter, v_p_cells=None):
    """Ergun-consistent drag pressure drop from the cell drag integrals."""
    if v_p_cells is None:
        v_p_cells = np.zeros((profile.n_cells, 3))
    u_f = u_superficial / eps
    slip = u_f - v_p_cells[:, 2]
    beta, _ = cl.gidaspow_beta(eps, diameter, fluid.rho, fluid.mu, slip)
    return float(np.sum(beta * slip / eps) * profile.dz)


def at_rest_pressure(profile: ColumnProfile, eps, v_p_cells, diameter, fluid: FluidProperties):
    """Pressure field that holds a fluid at rest against the drag of moving particles."""
    dz = profile.dz
    slip = -v_p_cells[:, 2]  # u_f = 0
    beta, dense = cl.gidaspow_beta(eps, diameter, fluid.rho, fluid.mu, np.abs(slip))
    dpdz = (-beta * slip + fluid.rho * eps * fluid.gravity) / eps
    edges = profile.edges()
    p_edges = np.zeros(profile.n_cells + 1)
    for c in range(profile.n_cells - 1, -1, -1):
        p_edges[c] = p_edges[c + 1] + dpdz[c] * dz
    return {
        "eps": eps,
        "beta": beta,
        "dense_branch": dense,
        "dpdz_cells": dpdz,
        "p_edges": p_edges,
        "dp_total": float(p_edges[0] - p_edges[-1]),
    }


def momentum_ledger(profile, eps, v_p_cells, u_superficial, diameter, fluid, steady):
    """Momentum-flux ledger between the particle side and the fluid side.

    Returns the residuals that the acceptance criterion bounds by 2 %:

    ``drag_reaction``  particles' summed drag vs the cell drag integral
                       (independent local-voidage representations);
    ``momentum_flux``  particles' summed drag vs the fluid integral including
                       the convective momentum flux ``rho U (u_f,out - u_f,in)``.
    """
    dz = profile.dz
    slip = steady["slip"]
    drag_cells = float(np.sum(steady["drag_per_volume"] * dz))  # force per unit bed area
    u_f = steady["u_f"]
    conv = float(fluid.rho * u_superficial * (u_f[-1] - u_f[0]))
    fluid_side = float(steady["dp_drag"] - conv)
    denom = max(abs(fluid_side), 1e-30)
    return {
        "drag_force_cells_per_area": drag_cells,
        "momentum_flux_conv": conv,
        "dp_drag": float(steady["dp_drag"]),
        "fluid_side_total": fluid_side,
        "drag_reaction_rel": abs(drag_cells - fluid_side) / denom,
        "dpdz_max": float(np.max(np.abs(steady["dpdz_cells"]))),
    }


def drag_force_partitioned(profile, pos, radius, eps, diameter, fluid, u_superficial, v_p_cells=None):
    """Momentum-conserving (divided-volume) drag on the grains of a column.

    The drag of cell ``c`` is ``beta_c * slip_c`` per unit *bed* volume, i.e.
    ``beta_c * slip_c * V_c`` per cell; distributing it over the grains by their
    partial volume fraction ``V_i,c / V_c`` makes the grain-side sum equal to the
    fluid-side integral *exactly*, whatever the grid-to-grain ratio::

        sum_i F_i = sum_c beta_c slip_c V_c    (per unit bed area: * dz)

    This is the form used for the pressure-drop and momentum-ledger criteria;
    the time-accurate group settling uses the centre-cell voidage instead.
    """
    pos = np.asarray(pos, dtype=float)
    rad = np.broadcast_to(np.asarray(radius, dtype=float), (pos.shape[0],))
    vol = float(0.0) + 4.0 / 3.0 * np.pi * rad**3
    edges = profile.edges()
    dz = profile.dz
    force = np.zeros((pos.shape[0], 3))
    v_p_cells = np.zeros((profile.n_cells, 3)) if v_p_cells is None else v_p_cells
    cell_drag = np.zeros(profile.n_cells)
    for c in range(profile.n_cells):
        frac = cl.sphere_slab_fraction(rad, pos[:, 2], edges[c], edges[c + 1])
        slip_c = u_superficial / eps[c] - v_p_cells[c, 2]
        beta_c, _ = cl.gidaspow_beta(eps[c], diameter, fluid.rho, fluid.mu, abs(slip_c))
        cell_drag[c] = beta_c * slip_c * dz  # force per unit bed area
        share = frac * vol / (1.0 - eps[c])  # volume of suspension represented
        force[:, 2] += beta_c * slip_c * share
    return {"force": force, "cell_drag": cell_drag, "volume": vol}


def particle_drag_sum(eps_i, diameter, fluid, u_f_particle, v_particle):
    """Total vertical drag on the particles [N] from the per-particle closure."""
    force, beta, s_mag = cl.particle_drag_force(
        eps_i, diameter, fluid.rho, fluid.mu, u_f_particle, v_particle
    )
    return force, beta, s_mag