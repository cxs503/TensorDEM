"""Hydrodynamic closures for Lagrangian (point-particle) DEM-fluid coupling.

This module is the closure library of :mod:`tensordem.lagrangian_bed`.  It is
deliberately independent of :mod:`tensordem.bedcoupling` (semi-analytic closure
audit) and of :mod:`tensordem.cfd_dem_dense` (Eulerian column); the acceptance
study cross-checks the three implementations against each other.

Conventions
-----------
``U``      superficial fluid velocity (volume flux per unit bed area), +z up.
``u_f``    interstitial (pore) fluid velocity, ``u_f = U / eps``.
``v_p``    particle velocity.
``s``      interstitial slip velocity ``s = u_f - v_p``.
``beta``   Gidaspow (1994) momentum-exchange coefficient [kg m^-3 s^-1]: the
           drag force on the *particles* per unit *bed* volume is ``beta * s``,
           hence the force on one particle is ``beta * s * V_p / (1 - eps)``.

The dense branch of ``beta`` combined with the fluid momentum balance
``0 = -eps dp/dz - beta s`` reproduces the Ergun (1952) relation exactly::

    -dp/dz = 150 mu (1-eps)^2 U / (eps^3 d^2) + 1.75 rho (1-eps) U^2 / (eps^3 d)

which is checked analytically in ``tests/test_lagrangian_bed_fluid.py``.
"""
from __future__ import annotations

import numpy as np

GRAVITY = 9.81
GIDASPOW_SWITCH = 0.8
_CD_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def sphere_volume(diameter):
    """Volume of a sphere [m^3] from its diameter [m]."""
    d = np.asarray(diameter, dtype=float)
    return np.pi / 6.0 * d**3


def sphere_projected_area(diameter):
    """Projected (frontal) area of a sphere [m^2]."""
    d = np.asarray(diameter, dtype=float)
    return np.pi / 4.0 * d**2


def sphere_below_plane(radius, dz):
    """Volume of a sphere of ``radius`` lying below a plane at offset ``dz``.

    ``dz`` is measured from the sphere centre along +z (``dz = z_plane - z_c``).
    """
    r = np.asarray(radius, dtype=float)
    h = np.asarray(dz, dtype=float)
    full = 4.0 / 3.0 * np.pi * r**3
    return np.where(
        h <= -r,
        0.0,
        np.where(h >= r, full, np.pi / 3.0 * (r + h) ** 2 * (2.0 * r - h)),
    )


def sphere_slab_fraction(radius, z_center, z_lo, z_hi):
    """Fraction (0..1) of a sphere volume inside the slab ``[z_lo, z_hi]``."""
    r = np.asarray(radius, dtype=float)
    full = 4.0 / 3.0 * np.pi * r**3
    frac = (
        sphere_below_plane(r, z_hi - z_center) - sphere_below_plane(r, z_lo - z_center)
    ) / full
    return np.clip(frac, 0.0, 1.0)


def sphere_overlap_volume(r1, r2, distance):
    """Analytic volume shared by two spheres (0 if separated)."""
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    dist = np.asarray(distance, dtype=float)
    v_small = 4.0 / 3.0 * np.pi * np.minimum(r1, r2) ** 3
    safe = np.maximum(dist, 1e-30)
    lens = (
        np.pi
        * (r1 + r2 - dist) ** 2
        * (dist**2 + 2.0 * dist * (r1 + r2) - 3.0 * (r1 - r2) ** 2)
        / (12.0 * safe)
    )
    out = np.where(dist >= r1 + r2, 0.0, lens)
    out = np.where(dist <= np.abs(r1 - r2), v_small, out)
    return np.clip(out, 0.0, v_small)


# ---------------------------------------------------------------------------
# drag laws
# ---------------------------------------------------------------------------
def schiller_naumann_cd(reynolds):
    """Schiller-Naumann drag coefficient (0 < Re <~ 1000)."""
    re = np.asarray(reynolds, dtype=float)
    re_safe = np.maximum(re, _CD_FLOOR)
    return 24.0 / re_safe * (1.0 + 0.15 * re_safe**0.687)


def schiller_naumann_drag(u_slip, diameter, rho_fluid, mu):
    """Unhindered drag force [N] on a sphere, finite at Re -> 0."""
    u = np.asarray(u_slip, dtype=float)
    re = rho_fluid * np.abs(u) * diameter / mu
    return 3.0 * np.pi * mu * diameter * u * (1.0 + 0.15 * np.maximum(re, 0.0) ** 0.687)


def ergun_beta(eps, diameter, rho_fluid, mu, slip):
    """Dense (Ergun) branch of the Gidaspow exchange coefficient."""
    eps = np.clip(np.asarray(eps, dtype=float), 1e-6, 1.0)
    s = np.abs(np.asarray(slip, dtype=float))
    return (
        150.0 * mu * (1.0 - eps) ** 2 / (eps * diameter**2)
        + 1.75 * rho_fluid * (1.0 - eps) * s / diameter
    )


def wen_yu_beta(eps, diameter, rho_fluid, mu, slip):
    """Dilute (Wen-Yu) branch of the Gidaspow exchange coefficient."""
    eps = np.clip(np.asarray(eps, dtype=float), 1e-6, 1.0)
    s = np.abs(np.asarray(slip, dtype=float))
    cd = schiller_naumann_cd(rho_fluid * s * diameter / mu)
    return 0.75 * cd * rho_fluid * eps * (1.0 - eps) * s / diameter * eps ** (-2.65)


def gidaspow_beta(eps, diameter, rho_fluid, mu, slip, switch=GIDASPOW_SWITCH):
    """Gidaspow (1994) piecewise drag: Ergun below ``switch``, Wen-Yu above."""
    eps_arr = np.asarray(eps, dtype=float)
    dense = eps_arr <= switch
    beta = np.where(
        dense,
        ergun_beta(eps_arr, diameter, rho_fluid, mu, slip),
        wen_yu_beta(eps_arr, diameter, rho_fluid, mu, slip),
    )
    return beta, dense


def gidaspow_branch_jump(diameter, rho_fluid, mu, slip=1e-3, switch=GIDASPOW_SWITCH, tol=1e-6):
    """Relative discontinuity of ``beta`` across the Ergun/Wen-Yu switch."""
    lo = float(ergun_beta(switch - tol, diameter, rho_fluid, mu, slip))
    hi = float(wen_yu_beta(switch + tol, diameter, rho_fluid, mu, slip))
    return abs(hi - lo) / lo


def particle_drag_force(
    eps, diameter, rho_fluid, mu, u_fluid, v_particle, switch=GIDASPOW_SWITCH, volume=None
):
    """Drag force [N] on Lagrangian particles from local voidage and slip.

    ``u_fluid`` (N,3) interstitial fluid velocity, ``v_particle`` (N,3).
    ``volume`` optionally supplies the per-grain volume (polydisperse packs);
    the momentum ledger between the grain side and the cell side only closes
    when the *actual* grain volumes are used.

    Returns ``(force, beta, slip_magnitude)``.
    """
    u_f = np.atleast_2d(np.asarray(u_fluid, dtype=float))
    v_p = np.atleast_2d(np.asarray(v_particle, dtype=float))
    eps_arr = np.clip(np.asarray(eps, dtype=float), 1e-6, 1.0 - 1e-9)
    slip = u_f - v_p
    s_mag = np.linalg.norm(slip, axis=1)
    beta, _ = gidaspow_beta(eps_arr, diameter, rho_fluid, mu, s_mag, switch=switch)
    vp = sphere_volume(diameter) if volume is None else np.asarray(volume, dtype=float)
    vp = np.broadcast_to(vp, eps_arr.shape)
    scale = (beta * vp / (1.0 - eps_arr))[:, None]
    return scale * slip, beta, s_mag


def rz_hindrance_drag(eps, diameter, rho_fluid, mu, slip, n):
    """Drag force [N] from a Richardson-Zaki-consistent hindrance function.

    The closure is written on the *velocity* rather than on the force so that it
    reproduces ``s = u_t eps^(n-1)`` for any particle Reynolds number: the
    isolated Schiller-Naumann drag is evaluated at the hindered slip
    ``s * eps^(1-n)``.  The classical local closures (Wen-Yu ``eps^-2.65``,
    Ergun dense branch) imply a *constant* or weakly varying exponent that does
    not follow the RZ correlation; that deviation is measured and reported by
    the acceptance study.
    """
    eps = np.clip(np.asarray(eps, dtype=float), 1e-6, 1.0)
    n_arr = np.asarray(n, dtype=float)
    return schiller_naumann_drag(
        np.asarray(slip, dtype=float) * eps ** (1.0 - n_arr), diameter, rho_fluid, mu
    )


def rz_hindrance_beta(eps, diameter, rho_fluid, mu, slip, n):
    """Momentum-exchange coefficient equivalent to :func:`rz_hindrance_drag`."""
    eps = np.clip(np.asarray(eps, dtype=float), 1e-6, 1.0)
    force = rz_hindrance_drag(eps, diameter, rho_fluid, mu, slip, n)
    slip = np.asarray(slip, dtype=float)
    vp = sphere_volume(diameter)
    scale = np.where(np.abs(slip) > 0, force / np.where(np.abs(slip) > 0, slip, 1.0), 0.0)
    return scale * (1.0 - eps) / vp


def buoyancy_force(diameter, rho_fluid, gravity=GRAVITY):
    """Buoyancy force vector [N] on a particle (+z, opposing gravity)."""
    vp = float(sphere_volume(diameter))
    f = np.zeros(3)
    f[2] = rho_fluid * vp * gravity
    return f


def pressure_gradient_force(diameter, grad_p):
    """Force [N] on a particle from a (dynamic) pressure gradient."""
    vp = float(sphere_volume(diameter))
    return -vp * np.asarray(grad_p, dtype=float)


def submerged_weight(diameter, rho_particle, rho_fluid, gravity=GRAVITY):
    """Submerged weight [N]; negative = downward (z component)."""
    vp = float(sphere_volume(diameter))
    return -(rho_particle - rho_fluid) * vp * gravity


# ---------------------------------------------------------------------------
# terminal velocity / Richardson-Zaki
# ---------------------------------------------------------------------------
def terminal_velocity(
    diameter, rho_particle, rho_fluid, mu, gravity=GRAVITY, tol=1e-13, max_iter=300
):
    """Terminal velocity of an isolated sphere with Schiller-Naumann drag."""
    vp = float(sphere_volume(diameter))
    weight = (rho_particle - rho_fluid) * vp * gravity
    if weight <= 0.0:
        return {"u_terminal": 0.0, "re_terminal": 0.0, "weight": 0.0, "cd": None}
    lo, hi = 0.0, 1e-4
    while float(schiller_naumann_drag(hi, diameter, rho_fluid, mu)) < weight:
        hi *= 2.0
        if hi > 1e3:  # pragma: no cover - guard
            break
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if float(schiller_naumann_drag(mid, diameter, rho_fluid, mu)) < weight:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(hi, 1e-30):
            break
    u = 0.5 * (lo + hi)
    re_t = rho_fluid * u * diameter / mu
    return {
        "u_terminal": float(u),
        "re_terminal": float(re_t),
        "weight": float(weight),
        "cd": float(schiller_naumann_cd(re_t)),
    }


def rz_exponent(re_terminal, d_over_D=0.0):
    """Richardson-Zaki (1954) voidage exponent ``n`` for batch sedimentation.

    ===========  ==================================
    Re_t         n
    ===========  ==================================
    < 0.2        4.65 + 19.5 d/D
    0.2 - 1      (4.4 + 18 d/D) Re_t^-0.03
    1 - 200      (4.4 + 18 d/D) Re_t^-0.1
    200 - 500    4.4 Re_t^-0.1
    > 500        2.4
    ===========  ==================================

    ``d/D`` is particle-to-column diameter ratio (0 for an unbounded bed).
    """
    re = np.asarray(re_terminal, dtype=float)
    re_safe = np.maximum(re, 1e-12)
    return np.where(
        re < 0.2,
        4.65 + 19.5 * d_over_D,
        np.where(
            re < 1.0,
            (4.4 + 18.0 * d_over_D) * re_safe**-0.03,
            np.where(
                re < 200.0,
                (4.4 + 18.0 * d_over_D) * re_safe**-0.1,
                np.where(re < 500.0, 4.4 * re_safe**-0.1, 2.4),
            ),
        ),
    )


def rz_slip_velocity(u_terminal, eps, n):
    """Richardson-Zaki hindered slip velocity ``u_t * eps^(n-1)``."""
    eps = np.clip(np.asarray(eps, dtype=float), 1e-6, 1.0)
    return u_terminal * eps ** (np.asarray(n, dtype=float) - 1.0)


def fit_rz_exponent(eps, u_slip, u_terminal):
    """Least-squares exponent from ``u_slip = u_t eps^(n-1)`` (log-log fit)."""
    eps = np.asarray(eps, dtype=float)
    u = np.asarray(u_slip, dtype=float)
    mask = (eps > 0) & (eps < 1) & (u > 0)
    slope, intercept = np.polyfit(np.log(eps[mask]), np.log(u[mask] / u_terminal), 1)
    pred = np.exp(intercept) * eps[mask] ** slope * u_terminal
    rms = float(np.sqrt(np.mean((pred - u[mask]) ** 2)) / np.mean(u[mask]))
    return {"n": float(1.0 + slope), "log_intercept": float(intercept), "relative_rms": rms}


# ---------------------------------------------------------------------------
# local voidage
# ---------------------------------------------------------------------------
def local_voidage(pos, radius, probe_factor, min_eps=1e-3, period=None):
    """Per-particle local voidage from a spherical probe of radius probe_factor*d.

    ``eps_i = 1 - (sum of particle volumes inside the probe) / V_probe`` with
    analytic sphere-sphere overlap volumes.  ``period`` is the (optional) box
    length for a minimum-image sum in x and y, required for a laterally periodic
    column (otherwise probes near a periodic boundary see a vacuous half-space).
    Returns (eps_i, solid_fraction_i).
    """
    pos = np.asarray(pos, dtype=float)
    rad = np.broadcast_to(np.asarray(radius, dtype=float), (pos.shape[0],))
    probe = float(probe_factor) * 2.0 * float(np.mean(rad))
    v_probe = 4.0 / 3.0 * np.pi * probe**3
    solid = np.zeros(pos.shape[0])
    for i in range(pos.shape[0]):
        delta = pos - pos[i]
        if period:
            delta[:, 0] -= period * np.round(delta[:, 0] / period)
            delta[:, 1] -= period * np.round(delta[:, 1] / period)
        dist = np.linalg.norm(delta, axis=1)
        near = dist < probe + rad
        solid[i] = np.sum(
            sphere_overlap_volume(probe, rad[near], np.maximum(dist[near], 1e-12))
        ) / v_probe
    solid = np.clip(solid, 0.0, 1.0 - min_eps)
    return 1.0 - solid, solid


def voidage_at_point(pos, radius, point, probe_factor, period=None):
    """Voidage inside a spherical probe centred at ``point`` (radius probe_factor*d)."""
    pos = np.asarray(pos, dtype=float)
    rad = np.broadcast_to(np.asarray(radius, dtype=float), (pos.shape[0],))
    probe = float(probe_factor) * 2.0 * float(np.mean(rad))
    v_probe = 4.0 / 3.0 * np.pi * probe**3
    delta = pos - np.asarray(point, dtype=float)[None, :]
    if period:
        delta[:, 0] -= period * np.round(delta[:, 0] / period)
        delta[:, 1] -= period * np.round(delta[:, 1] / period)
    dist = np.linalg.norm(delta, axis=1)
    near = dist < probe + rad
    solid = float(np.sum(sphere_overlap_volume(probe, rad[near], np.maximum(dist[near], 1e-12))) / v_probe)
    return float(np.clip(1.0 - solid, 1e-3, 1.0))


def cell_voidage(pos, radius, z_lo, z_hi, n_cells):
    """Exact voidage per z-slab (analytic sphere-slab partition)."""
    pos = np.asarray(pos, dtype=float)
    rad = np.broadcast_to(np.asarray(radius, dtype=float), (pos.shape[0],))
    edges = np.linspace(z_lo, z_hi, int(n_cells) + 1)
    cell_vol = (z_hi - z_lo) / n_cells * np.pi * 0.0  # placeholder overwritten below
    solid_vol = np.zeros(int(n_cells))
    for c in range(int(n_cells)):
        frac = sphere_slab_fraction(rad, pos[:, 2], edges[c], edges[c + 1])
        solid_vol[c] = np.sum(frac * 4.0 / 3.0 * np.pi * rad**3)
    return edges, solid_vol, cell_vol


def cell_index(z, z_lo, z_hi, n_cells):
    """Index of the slab containing ``z`` (clipped to the column)."""
    z = np.asarray(z, dtype=float)
    dz = (z_hi - z_lo) / n_cells
    idx = np.floor((z - z_lo) / dz).astype(int)
    return np.clip(idx, 0, n_cells - 1)


def cell_particle_velocity(pos, vel, radius, z_lo, z_hi, n_cells):
    """Volume-weighted mean particle velocity per z-slab, (n_cells, 3)."""
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    rad = np.broadcast_to(np.asarray(radius, dtype=float), (pos.shape[0],))
    edges = np.linspace(z_lo, z_hi, int(n_cells) + 1)
    vp = 4.0 / 3.0 * np.pi * rad**3
    out = np.zeros((int(n_cells), 3))
    weight = np.zeros(int(n_cells))
    for c in range(int(n_cells)):
        frac = sphere_slab_fraction(rad, pos[:, 2], edges[c], edges[c + 1])
        w = frac * vp
        weight[c] = w.sum()
        if weight[c] > 0:
            out[c] = (w[:, None] * vel).sum(axis=0) / weight[c]
    return out, weight