"""Disordered packing generator for silo/hopper DEM (no ordered-lattice arching).

The m6296 audit (``docs/silo_discharge_beverloo_verification.md`` §6) shows why
an ordered HCP start is fatal for a discharge benchmark: a 3--5 d orifice is
spanned by an arch almost immediately and the flow rate collapses after the
start-up burst.  This module therefore builds a *disordered* bed:

1. a simple-cubic lattice at spacing ``2 r (1 + gap)`` is cut to the container
   shape (a loose, crystalline seed);
2. every centre is jittered by up to ``jitter * r`` (destroys the lattice
   symmetry that lets arches lock in);
3. overlaps introduced by the jitter are removed by a Jacobi push-apart
   relaxation that also respects the analytic walls, so the contact solver never
   starts from a large interpenetration (the second divergence route reported in
   the audit).

The resulting packing fraction is ~0.45 in the loose state and compacts to
~0.58--0.62 once it settles under gravity, which is the range reported for
random sphere packings.
"""
from __future__ import annotations

import math

import numpy as np

from .kernel import Material, SiloGeometry, SiloFlow

__all__ = ["lattice_fill", "relax_overlaps", "deposit_bed", "build_bed"]


def lattice_fill(
    geometry: SiloGeometry,
    material: Material,
    fill_height: float,
    *,
    gap: float = 0.06,
    jitter: float = 0.30,
    seed: int = 0,
    z_floor: float | None = None,
) -> np.ndarray:
    """Jittered simple-cubic lattice inside the container, below ``fill_height``.

    ``gap`` is the lattice gap in units of the particle diameter, ``jitter`` the
    random displacement amplitude in units of the particle radius.
    """
    r = material.radius
    if not (0.0 <= gap < 0.5):
        raise ValueError("gap must lie in [0, 0.5)")
    if not (0.0 <= jitter < 1.0):
        raise ValueError("jitter must lie in [0, 1)")
    if fill_height <= 2.0 * r:
        raise ValueError("fill_height is too small for a particle layer")
    if fill_height > geometry.z_top - 0.5 * r:
        # a bed above the container rim would be free of any wall contact and
        # could drift back inside with a large penetration once the level drops
        raise ValueError(
            f"fill_height {fill_height:.4f} m exceeds the container rim "
            f"{geometry.z_top:.4f} m: increase cylinder_height"
        )
    rng = np.random.default_rng(seed)
    # face-centred-cubic seed: nearest-neighbour distance 2 r (1 + gap) -> the
    # seed packing fraction is 0.7405 / (1 + gap)^3 (0.64 for gap = 0.05)
    s = 2.0 * math.sqrt(2.0) * r * (1.0 + gap)
    r_max = float(np.max(geometry.radius_at(np.linspace(0.0, fill_height, 64)))) + r
    axis = np.arange(-r_max, r_max + s, s)
    z0 = r + (0.0 if z_floor is None else z_floor)
    zs = np.arange(z0, fill_height, s)
    gx, gy, gz = np.meshgrid(axis, axis, zs, indexing="ij")
    base = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]) * s
    pts = (base[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    pts = pts + rng.uniform(-jitter * r, jitter * r, size=pts.shape)
    rho = np.hypot(pts[:, 0], pts[:, 1])
    inside = (rho <= (geometry.radius_at(pts[:, 2]) - r)) & (pts[:, 2] >= r)
    return np.ascontiguousarray(pts[inside])


def relax_overlaps(flow: SiloFlow, *, iterations: int = 1200, relax: float = 0.25,
                   tolerance: float = 0.02) -> float:
    """Position-based overlap removal (also pushes particles off the walls).

    A Jacobi push-apart converges slowly where the cross-section is narrow (the
    cone of a hopper), so the displacement of every particle is the *mean* of the
    pushes it receives and the loop stops once the worst overlap is below
    ``tolerance * r`` (or after ``iterations``).  The residual overlap is
    returned in units of the particle radius; it matters because the stored
    elastic energy ``0.5 k_n d^2`` of a 0.1 r overlap is already enough to throw
    particles out of the container once the dynamics starts.
    """
    r = flow.mat.radius
    worst = 0.0
    for _ in range(iterations):
        i, j, delta, nvec = flow._pairs(force=True)
        worst = float(delta.max()) if len(delta) else 0.0
        if worst <= tolerance * r:
            break
        if len(i):
            n = flow.n
            push = (0.5 * relax * delta)[:, None] * nvec
            acc = np.stack([np.bincount(i, weights=push[:, c], minlength=n)
                            + np.bincount(j, weights=-push[:, c], minlength=n)
                            for c in range(3)], axis=1)
            flow.pos += acc
        for wc in flow._wall_contacts():
            idx, nv, dl = wc["idx"], wc["n"], wc["delta"]
            flow.pos[idx] += (relax * dl)[:, None] * nv
    return worst




def deposit_bed(
    geometry: SiloGeometry,
    material: Material,
    fill_height: float,
    *,
    wall_mu: float = 0.3,
    gap: float = 0.10,
    jitter: float = 0.35,
    seed: int = 0,
    settle_time: float = 0.06,
    insert_every: int = 150,
    dt: float | None = None,
    plug_outlet: bool = True,
    max_insert_wait: int = 900,
) -> "SiloFlow":
    """Pour the bed layer by layer from the bottom up ("rain deposition").

    This is the physics-clean route to a disordered packing: every lattice level
    is inserted only once the surface below it has settled, so particles never
    start deeply overlapped (the failure mode of a jittered dense lattice, see
    the m6296 audit) and no ordered crystal survives.  The deposited bed has a
    poured-packing fraction (~0.55-0.60) rather than a crystal one.
    """
    r = material.radius
    lattice = lattice_fill(geometry, material, fill_height, gap=gap, jitter=jitter, seed=seed)
    if len(lattice) < 8:
        raise ValueError("lattice too small: increase fill_height or the radius")
    order = np.argsort(lattice[:, 2], kind="stable")
    lattice = lattice[order]
    z = lattice[:, 2]
    s = 2.0 * math.sqrt(2.0) * r * (1.0 + gap)
    levels: list[np.ndarray] = []
    start = 0
    for k in range(1, len(z) + 1):
        if k == len(z) or z[k] - z[start] > 0.5 * s:
            levels.append(np.arange(start, k))
            start = k
    flow = SiloFlow(lattice[levels[0]], material, geometry, wall_mu=wall_mu, dt=dt,
                    plug_outlet=plug_outlet)
    for k, lev in enumerate(levels[1:], start=1):
        z_lev = float(lattice[lev, 2].min())
        waited = 0
        while waited < max_insert_wait and float(flow.pos[:, 2].max()) > z_lev - 1.2 * r:
            flow.step()
            waited += 1
        flow.add_particles(lattice[lev])
        for _ in range(insert_every):
            flow.step()
    for _ in range(int(round(settle_time / flow.dt))):
        flow.step()
    flow.vel *= 0.0
    flow.omega *= 0.0
    flow._E_before = flow.e_kin + flow.e_grav + flow.e_elast
    flow.deposited_levels = len(levels)
    return flow


def build_bed(
    geometry: SiloGeometry,
    material: Material,
    fill_height: float,
    *,
    wall_mu: float = 0.3,
    gap: float = 0.06,
    jitter: float = 0.30,
    seed: int = 0,
    settle_time: float = 0.10,
    plug_outlet: bool = True,
    dt: float | None = None,
) -> SiloFlow:
    """Lattice + jitter + overlap relaxation + gravity settling -> static bed."""
    pos = lattice_fill(geometry, material, fill_height, gap=gap, jitter=jitter, seed=seed)
    flow = SiloFlow(pos, material, geometry, wall_mu=wall_mu, dt=dt, plug_outlet=plug_outlet)
    flow.initial_overlap_over_radius = relax_overlaps(flow) / material.radius
    n_settle = int(round(settle_time / flow.dt))
    for k in range(n_settle):
        flow.step()
        if k < 1500:          # overdamped start: let the residual overlaps relax
            flow.vel *= 0.7
            flow.omega *= 0.7
    # quench the residual kinetic energy so the bed is a well-defined initial state
    flow.vel *= 0.0
    flow.omega *= 0.0
    flow._E_before = flow.e_kin + flow.e_grav + flow.e_elast
    return flow