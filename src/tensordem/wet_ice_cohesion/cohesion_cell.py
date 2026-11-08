"""Cohesive 2D DEM tension cell: measured cohesion of wet vs dry granular ice.

Why a tension cell rather than a shear box
------------------------------------------
The quantity the campaign needs from the many-body level is the **capillary
cohesion** ``c`` of a granular ice assembly, i.e. the tensile capacity that the
liquid-bridge network adds to a packing that has none when dry.  A direct
tension (split) test measures exactly that number and nothing else:

* the packing is built once per run at a fixed solid fraction (row deposition +
  position-based compression, no dynamics involved);
* it is relaxed with both walls fixed (overdamped, fixed walls);
* the top wall is then pulled up at a constant quasi-static velocity while the
  bottom wall stays fixed, and the wall traction is recorded;
* the peak traction is the cohesive (tensile) strength ``sigma_t = c``;
* ``gamma = 0`` gives the dry control, whose peak traction must be zero to
  within the noise of the measurement.

Attempting this with a constant-stress *shear* cell was tried in this node and
abandoned: the proportional wall-height servo coupled to the near-jamming
stiffness ``d sigma / dH ~ k_n N_c / L_x`` rings, and a fixed-height variant
dilates out of contact during shear.  That negative result is recorded in
``docs/wet_ice_cohesion_P35N2.md`` section 5.  The tension cell has no servo and
no free wall degree of freedom, so there is nothing to ring.

Honest limitations
------------------
* 2D disks; the coordination number of a 2D packing is lower than a 3D one.  The
  micro-mechanical reference used for the comparison is evaluated with the
  coordination *measured in the same 2D cell*, so the comparison is like-for-like
  but the absolute value is a 2D index, not a 3D rubble strength.
* particles do not rotate (friction without rolling), no breakage, no freeze
  bonds (see :mod:`tensordem.rolling_resistance` for rolling).
* one pendular bridge per contacting pair with a fixed ``V / R^3``.
* the rupture is *geometric* (the wall pulls until the bridges exceed their
  cutoff); no dynamic break-up of the bridges is modelled.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from .capillary import capillary_force_array, cutoff_separation

_DEBUG = bool(os.environ.get("WET_ICE_DEBUG"))

__all__ = ["CohesionCellResult", "run_cohesion_cell"]


@dataclass
class CohesionCellResult:
    cohesion: float                 # measured peak tensile traction
    coordination: float             # mean contacting neighbours per particle
    bridge_force_mean: float        # mean |bridge force| over active bridges
    bridge_count: float             # mean number of active bridges
    gamma: float
    solid_fraction: float
    peak_gap: float                 # wall separation at the peak traction
    midplane_traction: float = 0.0  # traction across the mid-plane at the same instant
    midplane_history: np.ndarray = field(default_factory=lambda: np.zeros(1))
    traction_history: np.ndarray = field(default_factory=lambda: np.zeros(1))
    separation_history: np.ndarray = field(default_factory=lambda: np.zeros(1))
    params: dict = field(default_factory=dict)


def _deposit(rng, radii, Lx, gap: float = 0.02):
    """Row deposition: particles are dealt into rows inside the periodic box."""
    n = radii.size
    order = list(rng.permutation(n))
    rows, cur, width = [], [], 0.0
    for k in order:
        r = radii[k]
        if not cur:
            cur, width = [k], 2.0 * r
            continue
        add = (radii[cur[-1]] + r + 2.0 * gap) + r
        if width + add <= Lx - radii[cur[0]]:
            cur.append(k)
            width += add
        else:
            rows.append(cur)
            cur, width = [k], 2.0 * r
    if cur:
        rows.append(cur)
    x = np.zeros(n)
    y = np.zeros(n)
    row_y = float(radii.max()) + gap
    for row in rows:
        for m, k in enumerate(row):
            x[k] = radii[k] + gap if m == 0 else x[row[m - 1]] + radii[row[m - 1]] + radii[k] + 2.0 * gap
            y[k] = row_y
        row_y = max(y[k] + radii[k] for k in row) + gap
    return x, y


def _compress(x, y, radii, Lx, Ly_target, n_iter: int = 6000, tol: float = 2e-3):
    """Position-based compression to ``Ly_target`` (kinematic, cannot explode)."""
    y = radii + (y - y.min()) * (Ly_target - 2.0 * radii.max()) / max(1e-9, y.max() - y.min())
    r_sum = radii[:, None] + radii[None, :]
    worst = 0.0
    for _ in range(n_iter):
        dx = x[:, None] - x[None, :]
        dx -= Lx * np.rint(dx / Lx)
        dy = y[:, None] - y[None, :]
        d = np.hypot(dx, dy)
        np.fill_diagonal(d, 1e9)
        overlap = np.maximum(r_sum - d, 0.0)
        worst = float(overlap.max())
        if worst < tol:
            break
        inv = 1.0 / np.maximum(d, 1e-9)
        push = 0.5 * overlap * inv
        x = (x + np.sum(push * dx, axis=1)) % Lx
        y = y + np.sum(push * dy, axis=1)
        y = np.clip(y, radii, Ly_target - radii)
    return x, y, worst


def run_cohesion_cell(
    n_particles: int = 250,
    gamma: float = 0.0,
    particle_radius: float = 1.0,
    bridge_volume_fraction: float = 0.02,
    contact_angle_deg: float = 0.0,
    friction: float = 0.5,
    kn: float = 2.0e3,
    kt_fraction: float = 0.5,
    area_fraction: float = 0.80,
    pull_velocity: float = 0.01,
    n_relax: int = 400,
    n_pull: int = 4000,
    dt: float = 0.004,
    seed: int = 11,
) -> CohesionCellResult:
    """Pull a cohesive disk packing apart and measure its tensile strength.

    Dimensionless DEM units (``R = rho = 1``).  ``gamma = 0`` gives the dry
    control (the cohesion array is exactly zero, so the code path is identical
    and the dry result is the *same run* with the term switched off).

    Returns a :class:`CohesionCellResult` whose ``cohesion`` is the peak wall
    traction, in DEM stress units.
    """
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    rng = np.random.default_rng(seed)
    R = particle_radius
    radii = R * rng.uniform(0.85, 1.15, n_particles)
    mass = math.pi * radii ** 2

    area = math.pi * float(np.sum(radii ** 2)) / area_fraction
    Lx = math.sqrt(area * 1.5)
    Ly = area / Lx
    x, y = _deposit(rng, radii, Lx)
    x, y, residual = _compress(x, y, radii, Lx, Ly)

    r_sum = radii[:, None] + radii[None, :]
    r_eff = 0.5 * r_sum
    bridge_volume = np.full((n_particles, n_particles), bridge_volume_fraction * R ** 3)
    bridge_cut = np.vectorize(lambda re, v: cutoff_separation(re, v, contact_angle_deg))(
        r_eff, bridge_volume
    )
    np.fill_diagonal(bridge_cut, 0.0)
    wall_vol = bridge_volume_fraction * R ** 3
    wall_cut = cutoff_separation(R, wall_vol, contact_angle_deg)

    cn = 2.0 * 0.04 * math.sqrt(kn * float(np.mean(mass)))
    kt = kt_fraction * kn
    Ft = np.zeros((n_particles, n_particles))
    vx = np.zeros(n_particles)
    vy = np.zeros(n_particles)
    H = float(Ly)

    def forces(x, y, vx, vy, H):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dx -= Lx * np.rint(dx / Lx)
        d = np.hypot(dx, dy)
        np.fill_diagonal(d, 1e9)
        gap = d - r_sum
        nx = dx / np.maximum(d, 1e-12)
        ny = dy / np.maximum(d, 1e-12)
        vn = (vx[:, None] - vx[None, :]) * nx + (vy[:, None] - vy[None, :]) * ny
        overlap = np.where(gap < 0.0, -gap, 0.0)
        fn_rep = np.maximum(kn * overlap - cn * vn, 0.0)
        engaged = gap < bridge_cut
        att = np.zeros_like(gap)
        if gamma > 0.0:
            raw = capillary_force_array(r_eff, gamma, bridge_volume,
                                        np.maximum(gap, 0.0), contact_angle_deg)
            att = np.where(engaged, raw, 0.0)
        fn = fn_rep - att
        vt = -(vx[:, None] - vx[None, :]) * ny + (vy[:, None] - vy[None, :]) * nx
        Ft_new = np.where(engaged, Ft - kt * vt * dt, 0.0)
        lim = friction * np.maximum(fn, 0.0)
        np.clip(Ft_new, -lim, lim, out=Ft_new)
        fy_mat = fn * ny + Ft_new * nx
        Fx = np.sum(fn * nx - Ft_new * ny, axis=1)
        Fy = np.sum(fy_mat, axis=1)

        gap_b = y - radii
        gap_t = H - (y + radii)
        fn_b = np.maximum(kn * np.maximum(-gap_b, 0.0), 0.0)
        fn_t = np.maximum(kn * np.maximum(-gap_t, 0.0), 0.0)
        att_b = np.zeros(n_particles)
        att_t = np.zeros(n_particles)
        if gamma > 0.0:
            att_b = np.where(gap_b < wall_cut, capillary_force_array(
                radii, gamma, wall_vol, np.maximum(gap_b, 0.0), contact_angle_deg), 0.0)
            att_t = np.where(gap_t < wall_cut, capillary_force_array(
                radii, gamma, wall_vol, np.maximum(gap_t, 0.0), contact_angle_deg), 0.0)
        fnb = fn_b - att_b
        fnt = fn_t - att_t
        Fy = Fy + fnb - fnt
        ftb = np.clip(-kt * vx * dt, -friction * np.maximum(fnb, 0.0), friction * np.maximum(fnb, 0.0))
        Fx = Fx + ftb
        # traction transmitted across the mid-plane (bulk measurement)
        y_mid = 0.5 * Ly
        cross = ((y[:, None] < y_mid) & (y[None, :] >= y_mid)) & (~np.eye(n_particles, dtype=bool))
        plane = float(np.sum(np.where(cross, fy_mat, 0.0))) / Lx
        return dict(Fx=Fx, Fy=Fy, Ft=Ft_new, fnb=fnb, fnt=fnt, att=att,
                    engaged=engaged, plane_traction=plane)

    def step(f, damping=1.0):
        nonlocal Ft, vx, vy, x, y
        Ft = f["Ft"]
        vx = damping * (vx + dt * f["Fx"] / mass)
        vy = damping * (vy + dt * f["Fy"] / mass)
        x = (x + dt * vx) % Lx
        y = y + dt * vy

    # ---------------- relax between fixed walls ------------------------------
    for _ in range(n_relax):
        step(forces(x, y, vx, vy, H), damping=0.4)
    vx[:] = 0.0
    vy[:] = 0.0
    Ft[:] = 0.0

    # ---------------- pull the top wall --------------------------------------
    traction = np.empty(n_pull)
    separation = np.empty(n_pull)
    plane_hist = []
    coord_acc = 0.0
    bridge_acc = 0.0
    count_acc = 0.0
    n_acc = 0
    iu = np.triu_indices(n_particles, 1)
    for k in range(n_pull):
        f = forces(x, y, vx, vy, H)
        step(f)
        y = np.clip(y, radii, H - radii)
        H += pull_velocity * dt
        traction[k] = -float(np.sum(f["fnt"])) / Lx     # > 0 means the wall is pulled down
        plane_hist.append(f["plane_traction"])
        separation[k] = H - Ly
        coord_acc += 2.0 * float(np.count_nonzero(f["engaged"][iu])) / n_particles
        bridge_acc += float(np.sum(np.abs(f["att"][iu])))
        count_acc += float(np.count_nonzero(f["att"][iu] > 0.0))
        n_acc += 1
        if _DEBUG and k % 250 == 0:
            print("pull", k, "dH", round(separation[k], 3), "traction",
                  round(float(traction[k]), 5), "vmax", round(float(np.max(np.abs(vy))), 4))

    peak_index = int(np.argmax(traction))
    plane_arr = np.asarray(plane_hist, dtype=float)
    return CohesionCellResult(
        cohesion=float(traction[peak_index]),
        midplane_traction=float(plane_arr[peak_index]),
        midplane_history=plane_arr,
        coordination=coord_acc / max(1, n_acc),
        bridge_force_mean=bridge_acc / max(1, n_acc),
        bridge_count=count_acc / max(1, n_acc),
        gamma=gamma,
        solid_fraction=float(area / (Lx * Ly)),
        peak_gap=float(separation[peak_index]),
        traction_history=traction,
        separation_history=separation,
        params=dict(
            n_particles=n_particles, gamma=gamma,
            bridge_volume_fraction=bridge_volume_fraction,
            contact_angle_deg=contact_angle_deg, friction=friction, kn=kn, kt=kt, cn=cn,
            dt=dt, n_pull=n_pull, n_relax=n_relax, pull_velocity=pull_velocity,
            box=(Lx, float(Ly)), seed=seed, packing_residual=residual,
            area_fraction=area_fraction,
        ),
    )