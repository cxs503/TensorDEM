"""Bulk-cargo hold granular-dynamics verification surface (LW3-N06).

Physical background
-------------------
A bulk carrier stows a dry bulk cargo (ore / coal / grain) in a prismatic hold
and unloads it through bottom openings.  Three macroscopic quantities decide
whether that hold can be discharged, and all three are *emergent* properties of
the packing (no closed-form prediction exists without empirical constants):

1. **Angle of repose** ``theta_r`` -- the slope a poured heap can sustain.  It
   is what decides the residual cargo left on the tank top.
2. **Discharge rate** ``W(D)`` -- Beverloo's law
   ``W = C rho_b sqrt(g) (D - k d)^(5/2)`` (Beverloo, Leniger & van de Velde,
   1961; ``C ~ 0.58``, ``k ~ 1.4`` for near-spherical beads).  Its falsifiable
   content is the exponent ``5/2`` and the effective-aperture offset ``k d``.
3. **Wall pressure** ``p(z)`` in a filled hold -- Janssen's slab balance
   ``sigma_v(z) = p_inf (1 - exp(-z/lambda))``, ``lambda = R_h/(2 mu_w K)``,
   which saturates with depth instead of growing hydrostatically.  It sets the
   design load on the inner hull / hopper plating.

This module measures all three in one real 3-D discrete-element model of an
idealised prismatic cargo hold (rectangular plan form, four vertical walls,
flat bottom with a circular opening, plus a square feed funnel for the poured
heap), and tests the three laws against those measurements.

Numerics
--------
Contacts use :mod:`tensordem.granular3d_core` (linear spring-dashpot normal law,
Cundall-Strack tangential spring saturated at ``mu |F_n|``, isotropic rolling
resistance).  Boundaries are analytic (four walls, bottom plate with an opening
plus rim arc, feed funnel) and carry per-particle tangential state, so *static*
wall friction -- the mechanism behind Janssen saturation -- is modelled rather
than replaced by a velocity-dependent slider.  The time step is the contact
period criterion of ``Material3D.timestep`` (20 steps per period) and is
additionally compared with the Rayleigh estimate in :func:`numerical_checks`.

Honesty contract
----------------
Every reported number is measured from the trajectories / mass balance.  The
analytic laws only appear as the comparand, in separate ``*_theory`` fields, so
a measurement can never be relabelled as a prediction.  ``tests/test_bulk_cargo_discharge.py``
pins it (flow rate must equal the discharged-count mass balance; a plugged hold
must discharge nothing; ``mu = 0`` pile must flatten).

All quantities are SI (m, kg, s, Pa; reported angles in degrees).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .granular3d_core import (
    ContactState,
    Material3D,
    build_pairs,
    contact_forces,
    hcp_positions_in_box,
)

__all__ = [
    "BEVERLOO_C",
    "BEVERLOO_K",
    "REPOSE_REFERENCE_DEG",
    "BulkCargoError",
    "CargoHoldCase",
    "PourCase",
    "beverloo_prediction",
    "fit_power_law",
    "fit_janssen",
    "measure_repose_angle",
    "numerical_checks",
    "simulate_hold_discharge",
    "simulate_hold_static",
    "simulate_pour",
]

# Beverloo, Leniger & van de Velde (1961), glass-bead values.
BEVERLOO_C = 0.58
BEVERLOO_K = 1.4

# Literature band of *poured* repose angles for sub-rounded dry bulk cargoes
# (rounded sand / glass beads / ore fines at mu ~ 0.5): reported values span
# roughly 25-35 deg depending on the filling method; 30 deg is the central
# comparand used for the C1 criterion.
REPOSE_REFERENCE_DEG = 30.0
REPOSE_REFERENCE_BAND_DEG = (25.0, 35.0)


class BulkCargoError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# case definitions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CargoHoldCase:
    """Idealised prismatic bulk-cargo hold + monodisperse sphere cargo (SI)."""

    # cargo / contact law
    radius: float = 0.005
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.20
    restitution: float = 0.3
    kn: float = 1.0e4
    kt_ratio: float = 0.5
    gravity: float = 9.81
    # hold geometry
    length: float = 0.10
    breadth: float = 0.06
    fill_height: float = 0.10
    outlet_diameter: float = 0.030
    wall_mu: float = 0.4
    floor_mu: float = 0.5
    # numerics
    packing_gap: float = 0.03
    jitter: float = 0.12
    seed: int = 1
    settle_time: float = 0.25
    discharge_time: float = 1.00
    n_windows: int = 5

    def __post_init__(self) -> None:
        for name in ("radius", "density", "kn", "gravity", "length", "breadth",
                     "fill_height", "outlet_diameter"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0.0:
                raise BulkCargoError(f"{name} must be finite and positive")
        for name in ("mu", "mu_rolling", "wall_mu", "floor_mu", "packing_gap", "jitter"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v < 0.0:
                raise BulkCargoError(f"{name} must be finite and nonnegative")
        if not 0.0 <= self.restitution < 1.0:
            raise BulkCargoError("restitution must lie in [0, 1)")
        if self.outlet_diameter >= min(self.length, self.breadth):
            raise BulkCargoError("outlet must be smaller than the hold plan form")
        if self.fill_height <= 2.0 * self.radius:
            raise BulkCargoError("fill height must exceed two particle diameters")
        if self.n_windows < 2:
            raise BulkCargoError("need at least two flow-rate windows")

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def outlet_radius(self) -> float:
        return 0.5 * self.outlet_diameter

    @property
    def plan_area(self) -> float:
        return self.length * self.breadth

    @property
    def hydraulic_radius(self) -> float:
        """Janssen length scale ``R_h = A / P`` of the hold cross-section."""
        return self.plan_area / (2.0 * (self.length + self.breadth))

    @property
    def outlet_center(self) -> tuple[float, float]:
        return 0.5 * self.length, 0.5 * self.breadth

    @property
    def material(self) -> Material3D:
        return Material3D(
            radius=self.radius, density=self.density, mu=self.mu,
            mu_rolling=self.mu_rolling, restitution=self.restitution,
            kn=self.kn, kt_ratio=self.kt_ratio, gravity=self.gravity)

    @property
    def timestep(self) -> float:
        return self.material.timestep

    @property
    def particle_mass(self) -> float:
        return self.material.mass


@dataclass(frozen=True)
class PourCase:
    """Poured-heap (repose angle) case: square feed funnel above a flat floor."""

    radius: float = 0.005
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.20
    restitution: float = 0.3
    kn: float = 1.0e4
    kt_ratio: float = 0.5
    gravity: float = 9.81
    funnel_side: float = 0.030
    funnel_bottom: float = 0.060
    column_height: float = 0.050
    wall_mu: float = 0.0
    floor_mu: float = 0.5
    packing_gap: float = 0.03
    jitter: float = 0.10
    seed: int = 7
    settle_time: float = 0.40
    max_time: float = 6.0

    def __post_init__(self) -> None:
        for name in ("radius", "density", "kn", "gravity", "funnel_side",
                     "funnel_bottom", "column_height"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0.0:
                raise BulkCargoError(f"{name} must be finite and positive")
        if self.funnel_side <= 4.0 * self.radius:
            raise BulkCargoError("funnel mouth is too small to discharge")

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def material(self) -> Material3D:
        return Material3D(
            radius=self.radius, density=self.density, mu=self.mu,
            mu_rolling=self.mu_rolling, restitution=self.restitution,
            kn=self.kn, kt_ratio=self.kt_ratio, gravity=self.gravity)

    @property
    def timestep(self) -> float:
        return self.material.timestep


# --------------------------------------------------------------------------- #
# packings
# --------------------------------------------------------------------------- #
def _relax_overlaps(pos: np.ndarray, radius: float, length: float,
                    breadth: float, *, iters: int = 80) -> np.ndarray:
    """Jacobi push-apart so the solver never starts from a deep overlap.

    ``docs/silo_discharge_beverloo_verification.md`` §6 records that a lattice
    seeding with large initial interpenetration diverges; a jittered HCP start
    can be born overlapped, so overlaps are removed here, respecting the walls.
    """
    pos = np.asarray(pos, dtype=np.float64).copy()
    lo = radius * 1.001
    for _ in range(iters):
        iu, ju = build_pairs(pos, 2.0 * radius * 1.02)
        if iu.size == 0:
            break
        d = pos[ju] - pos[iu]
        dist = np.linalg.norm(d, axis=1)
        overlap = 2.0 * radius - dist
        bad = overlap > 1e-9
        if not bad.any():
            break
        nvec = d[bad] / np.maximum(dist[bad], 1e-30)[:, None]
        push = 0.5 * overlap[bad]
        np.add.at(pos, iu[bad], -nvec * push[:, None])
        np.add.at(pos, ju[bad], nvec * push[:, None])
        pos[:, 0] = np.clip(pos[:, 0], lo, length - lo)
        pos[:, 1] = np.clip(pos[:, 1], lo, breadth - lo)
        pos[:, 2] = np.maximum(pos[:, 2], lo)
    return pos


def _hold_bed(case: CargoHoldCase) -> np.ndarray:
    pos = hcp_positions_in_box(
        case.radius, case.length, case.breadth, case.fill_height,
        packing_gap=case.packing_gap, jitter=case.jitter, seed=case.seed)
    return _relax_overlaps(pos, case.radius, case.length, case.breadth)


def _funnel_bed(case: PourCase) -> np.ndarray:
    """Column of cargo standing inside the square feed funnel."""
    s = case.funnel_side
    pos = hcp_positions_in_box(
        case.radius, s, s, case.column_height, packing_gap=case.packing_gap,
        jitter=case.jitter, seed=case.seed)
    pos[:, 0] += 0.5 * s
    pos[:, 1] += 0.5 * s
    pos[:, 2] += case.funnel_bottom
    lo = case.radius * 1.001
    for _ in range(80):
        iu, ju = build_pairs(pos, 2.0 * case.radius * 1.02)
        if iu.size == 0:
            break
        d = pos[ju] - pos[iu]
        dist = np.linalg.norm(d, axis=1)
        overlap = 2.0 * case.radius - dist
        bad = overlap > 1e-9
        if not bad.any():
            break
        nvec = d[bad] / np.maximum(dist[bad], 1e-30)[:, None]
        push = 0.5 * overlap[bad]
        np.add.at(pos, iu[bad], -nvec * push[:, None])
        np.add.at(pos, ju[bad], nvec * push[:, None])
        pos[:, 0] = np.clip(pos[:, 0], 0.5 * s + lo, 1.5 * s - lo)
        pos[:, 1] = np.clip(pos[:, 1], 0.5 * s + lo, 1.5 * s - lo)
    return pos


# --------------------------------------------------------------------------- #
# analytic boundaries
# --------------------------------------------------------------------------- #
def _plane_contact(pos, vel, omega, mat, tang, mu_w, dt, normal, offset, *,
                   interior_sign=1.0, z_range=None, gid=None, roll=None,
                   mu_r_wall=None):
    """One analytic plane wall: normal spring-dashpot + Coulomb friction.

    ``interior_sign = +1`` puts the interior on the ``+normal`` side.  ``tang``
    is the per-particle tangential-displacement array of this wall and is
    updated in place (zeroed where the contact is lost).  Returns
    ``(force, torque, |F_n|)``.
    """
    n = len(pos)
    f = np.zeros((n, 3))
    tq = np.zeros((n, 3))
    fn_mag = np.zeros(n)
    if n == 0:
        return f, tq, fn_mag
    if gid is None:
        gid = np.arange(n)
    d = pos @ normal - offset
    pen = mat.radius - interior_sign * d
    active = pen > 0.0
    if z_range is not None:
        active &= (pos[:, 2] >= z_range[0]) & (pos[:, 2] <= z_range[1])
    tang[gid[~active]] = 0.0
    if roll is not None:
        roll[gid[~active]] = 0.0
    if not active.any():
        return f, tq, fn_mag
    idx = np.nonzero(active)[0]
    gidx = gid[idx]
    vn = vel[idx] @ normal
    f_n = np.maximum(mat.kn * pen[idx] - mat.gamma_wall * vn, 0.0)
    fn_mag[idx] = f_n
    n_exp = np.broadcast_to(normal, (idx.size, 3))
    v_t = vel[idx] - vn[:, None] * n_exp
    if mu_w > 0.0:
        surf = np.cross(omega[idx], -n_exp * mat.radius)
        t_new = tang[gidx] + (v_t + surf) * dt
        tnorm = np.linalg.norm(t_new, axis=1)
        cap = mu_w * f_n / mat.kt
        scale = np.where(tnorm > cap, cap / np.maximum(tnorm, 1e-30), 1.0)
        t_new = t_new * scale[:, None]
        tang[gidx] = t_new
        f_t = -mat.kt * t_new
    else:
        f_t = np.zeros((idx.size, 3))
    f[idx] += f_n[:, None] * n_exp + f_t
    tq[idx] += np.cross(-n_exp * mat.radius, f_t)
    # rolling resistance at the rigid wall (Ai et al. style rolling spring).
    # Without it an ideal sphere rolls on the floor without dissipation, which
    # is exactly the failure mode that flattens a poured heap.
    if roll is not None and mu_r_wall is not None and mu_r_wall > 0.0:
        roll_new = roll[gidx] + omega[idx] * dt
        rnorm = np.linalg.norm(roll_new, axis=1)
        cap_r = mu_r_wall * mat.radius * f_n / mat.kr
        scale_r = np.where(rnorm > cap_r, cap_r / np.maximum(rnorm, 1e-30), 1.0)
        roll_new = roll_new * scale_r[:, None]
        roll[gidx] = roll_new
        tq[idx] += -mat.kr * roll_new
    return f, tq, fn_mag


def _hold_forces(pos, vel, omega, mat, case: CargoHoldCase, tang, dt, *,
                 floor_open: bool, record: bool = False, gid=None, roll=None):
    """All analytic boundaries of the hold (four walls + bottom plate).

    ``floor_open=False`` is the loaded / plugged hold; ``True`` is the
    discharging hold (plate with a circular opening plus rim-arc contact).
    """
    n = len(pos)
    if gid is None:
        gid = np.arange(n)
    f = np.zeros((n, 3))
    tq = np.zeros((n, 3))
    diag = {"fn_wall": np.zeros(n), "fric_wall": np.zeros((n, 3)),
            "fn_floor": np.zeros(n)}
    r = mat.radius

    for key, normal, offset in (
        ("x0", np.array([1.0, 0.0, 0.0]), 0.0),
        ("xL", np.array([-1.0, 0.0, 0.0]), -case.length),
        ("y0", np.array([0.0, 1.0, 0.0]), 0.0),
        ("yB", np.array([0.0, -1.0, 0.0]), -case.breadth),
    ):
        fw, tw, fnm = _plane_contact(
            pos, vel, omega, mat, tang[key], case.wall_mu, dt, normal, offset,
            gid=gid, roll=None if roll is None else roll[key],
            mu_r_wall=mat.mu_rolling)
        f += fw
        tq += tw
        diag["fn_wall"] += fnm
        if record:
            diag["fric_wall"] += fw - fnm[:, None] * normal[None, :]

    radial = np.hypot(pos[:, 0] - case.outlet_center[0],
                      pos[:, 1] - case.outlet_center[1])
    plate = np.ones(n, dtype=bool) if not floor_open else radial > case.outlet_radius
    pen = r - pos[:, 2]
    active = (pen > 0.0) & plate
    tang_floor = tang["floor"]
    tang_floor[gid[~active]] = 0.0
    if roll is not None:
        roll["floor"][gid[~active]] = 0.0
    if active.any():
        idx = np.nonzero(active)[0]
        gidx = gid[idx]
        vn = vel[idx, 2]
        f_n = np.maximum(mat.kn * pen[idx] - mat.gamma_wall * vn, 0.0)
        diag["fn_floor"][idx] = f_n
        e_z = np.array([0.0, 0.0, 1.0])
        if case.floor_mu > 0.0:
            surf = np.cross(omega[idx], -e_z[None, :] * r)
            v_t = vel[idx] - vn[:, None] * e_z[None, :]
            t_new = tang_floor[gidx] + (v_t + surf) * dt
            tnorm = np.linalg.norm(t_new, axis=1)
            cap = case.floor_mu * f_n / mat.kt
            scale = np.where(tnorm > cap, cap / np.maximum(tnorm, 1e-30), 1.0)
            t_new = t_new * scale[:, None]
            tang_floor[gidx] = t_new
            f_t = -mat.kt * t_new
        else:
            f_t = np.zeros((idx.size, 3))
        f[idx, 2] += f_n
        f[idx] += f_t
        tq[idx] += np.cross(-e_z[None, :] * r, f_t)

    if floor_open:                                  # rim arc of the opening
        a = case.outlet_radius
        near = (np.abs(pos[:, 2]) < r) & (np.abs(radial - a) < r)
        if near.any():
            idx = np.nonzero(near)[0]
            rho = np.maximum(radial[idx], 1e-30)
            uxy = np.zeros((idx.size, 3))
            uxy[:, 0] = (pos[idx, 0] - case.outlet_center[0]) / rho
            uxy[:, 1] = (pos[idx, 1] - case.outlet_center[1]) / rho
            q = np.zeros((idx.size, 3))
            q[:, 0] = case.outlet_center[0] + a * uxy[:, 0]
            q[:, 1] = case.outlet_center[1] + a * uxy[:, 1]
            dvec = pos[idx] - q
            dist = np.linalg.norm(dvec, axis=1)
            pen_r = r - dist
            ok = pen_r > 0.0
            if ok.any():
                idx = idx[ok]
                nvec = dvec[ok] / np.maximum(dist[ok], 1e-30)[:, None]
                vn = (vel[idx] * nvec).sum(axis=1)
                f_n = np.maximum(mat.kn * pen_r[ok] - mat.gamma_wall * vn, 0.0)
                ft = np.zeros_like(nvec)
                if case.floor_mu > 0.0:
                    v_t = vel[idx] - vn[:, None] * nvec
                    v_t = v_t + np.cross(omega[idx], -nvec * r)
                    tnorm = np.linalg.norm(v_t, axis=1)
                    mag = np.minimum(case.floor_mu * f_n,
                                     mat.kt * tnorm * dt)
                    ft = np.where(tnorm[:, None] > 1e-30,
                                  -v_t / np.maximum(tnorm, 1e-30)[:, None], 0.0
                                  ) * mag[:, None]
                f[idx] += f_n[:, None] * nvec + ft
                tq[idx] += np.cross(-nvec * r, ft)
    return f, tq, diag


def _funnel_forces(pos, vel, omega, mat, case: PourCase, tang, dt, gid=None,
                   roll=None):
    """Feed-funnel walls (square tube spanning ``z >= funnel_bottom``) + floor."""
    n = len(pos)
    if gid is None:
        gid = np.arange(n)
    f = np.zeros((n, 3))
    tq = np.zeros((n, 3))
    s = case.funnel_side
    zr = (case.funnel_bottom, 1e9)
    for key, normal, offset in (
        ("fx0", np.array([1.0, 0.0, 0.0]), 0.5 * s),
        ("fxS", np.array([-1.0, 0.0, 0.0]), -1.5 * s),
        ("fy0", np.array([0.0, 1.0, 0.0]), 0.5 * s),
        ("fyS", np.array([0.0, -1.0, 0.0]), -1.5 * s),
    ):
        fw, tw, _ = _plane_contact(
            pos, vel, omega, mat, tang[key], case.wall_mu, dt, normal, offset,
            z_range=zr, gid=gid, roll=None if roll is None else roll[key],
            mu_r_wall=mat.mu_rolling)
        f += fw
        tq += tw
    fw, tw, _ = _plane_contact(
        pos, vel, omega, mat, tang["floor"], case.floor_mu, dt,
        np.array([0.0, 0.0, 1.0]), 0.0, gid=gid,
        roll=None if roll is None else roll["floor"], mu_r_wall=mat.mu_rolling)
    f += fw
    tq += tw
    return f, tq


# --------------------------------------------------------------------------- #
# measurement helpers
# --------------------------------------------------------------------------- #
def beverloo_prediction(D, d, rho_b, *, C: float = BEVERLOO_C,
                        k: float = BEVERLOO_K, g: float = 9.81):
    """``W = C rho_b sqrt(g) (D - k d)^(5/2)``; 0 when ``D <= k d``."""
    D = np.asarray(D, dtype=np.float64)
    eff = np.maximum(D - k * d, 0.0)
    return C * rho_b * math.sqrt(g) * eff ** 2.5


def fit_power_law(D, W, d, *, k: float = BEVERLOO_K) -> dict:
    """Fit ``log W = log A + p log(D - k d)``; the law claims ``p = 2.5``."""
    D = np.asarray(D, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    x = D - k * d
    ok = (W > 0.0) & (x > 0.0)
    if ok.sum() < 2:
        return {"exponent": float("nan"), "r2": float("nan"),
                "prefactor": float("nan"), "n_points": int(ok.sum())}
    lx, ly = np.log(x[ok]), np.log(W[ok])
    A = np.vstack([np.ones_like(lx), lx]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return {"exponent": float(coef[1]),
            "r2": float(1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0),
            "prefactor": float(math.exp(coef[0])),
            "n_points": int(ok.sum())}


def fit_janssen(depth, sigma) -> dict:
    """Least-squares fit of ``sigma = p_inf (1 - exp(-z/lambda))``."""
    from scipy.optimize import curve_fit

    z = np.asarray(depth, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    keep = np.isfinite(z) & np.isfinite(s) & (z > 0.0)
    z, s = z[keep], s[keep]
    if z.size < 4:
        return {"p_inf": float("nan"), "lambda": float("nan"),
                "rms_rel": float("nan"), "n_points": int(z.size)}
    try:
        popt, _ = curve_fit(lambda zz, pi, lam: pi * (1.0 - np.exp(-zz / lam)),
                            z, s, p0=[max(float(s.max()), 1e-6), max(float(z.mean()), 1e-6)],
                            maxfev=20000, bounds=([0.0, 1e-6], [np.inf, np.inf]))
    except Exception:                                    # pragma: no cover
        return {"p_inf": float("nan"), "lambda": float("nan"),
                "rms_rel": float("nan"), "n_points": int(z.size)}
    model = popt[0] * (1.0 - np.exp(-z / popt[1]))
    rel = (s - model) / np.maximum(np.abs(model), 1e-12)
    return {"p_inf": float(popt[0]), "lambda": float(popt[1]),
            "rms_rel": float(np.sqrt(np.mean(rel ** 2))),
            "max_abs_rel": float(np.max(np.abs(rel))),
            "n_points": int(z.size)}


def measure_repose_angle(pos: np.ndarray, *, center_xy=None,
                         bin_width: float | None = None, min_count: int = 3,
                         surface_quantile: float = 90.0,
                         radius: float | None = None,
                         min_fit_radius: float | None = None,
                         max_fit_fraction: float = 0.85) -> dict:
    """Conical fit of a poured heap's free surface -> angle of repose.

    The surface is the ``surface_quantile`` of the centre height inside each
    radial annulus (robust "top layer": one stray particle cannot move it; the
    quantile sits a *constant* offset below the free surface, so the slope -- and
    therefore the angle -- is unbiased) and
    a line ``z = z0 - tan(theta) r`` is fitted over the mid-slope range
    ``[0.2, 0.85] r_pile`` (apex and thin outer fringe excluded).  Two 2-D
    sections along +x and +y give an independent check.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) < 12:
        raise BulkCargoError("need at least 12 particle centres to fit a heap")
    if center_xy is None:
        center_xy = (float(np.median(pos[:, 0])), float(np.median(pos[:, 1])))
    rad = np.hypot(pos[:, 0] - center_xy[0], pos[:, 1] - center_xy[1])
    z = pos[:, 2]
    d = 2.0 * radius if radius is not None else 0.01
    bw = bin_width if bin_width is not None else d
    r_pile = float(np.quantile(rad, 0.98))
    if r_pile <= 0.0:
        raise BulkCargoError("degenerate heap (no radial extent)")
    edges = np.arange(0.0, r_pile + 2.0 * bw, bw)
    idx = np.digitize(rad, edges) - 1
    r_mid, z_surf, counts = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        c = int(m.sum())
        if c >= min_count:
            r_mid.append(0.5 * (edges[b] + edges[b + 1]))
            z_surf.append(float(np.quantile(z[m], surface_quantile / 100.0)))
            counts.append(c)
    r_mid = np.asarray(r_mid)
    z_surf = np.asarray(z_surf)
    if r_mid.size < 5:
        raise BulkCargoError(f"heap surface has only {r_mid.size} usable annuli")
    r_lo = 0.20 * r_pile
    if min_fit_radius is not None:
        r_lo = max(r_lo, float(min_fit_radius))
    r_hi = max_fit_fraction * r_pile
    fit = (r_mid >= r_lo) & (r_mid <= r_hi)
    if fit.sum() < 4:
        raise BulkCargoError(
            f"only {int(fit.sum())} annuli in the fit range "
            f"(r_pile={r_pile:.4f} m)")
    A = np.vstack([np.ones(int(fit.sum())), r_mid[fit]]).T
    coef, *_ = np.linalg.lstsq(A, z_surf[fit], rcond=None)
    slope = float(coef[1])
    pred = A @ coef
    ss_res = float(((z_surf[fit] - pred) ** 2).sum())
    ss_tot = float(((z_surf[fit] - z_surf[fit].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    theta = math.degrees(math.atan(max(-slope, 0.0)))

    sections = {}
    for name, axis in (("x", 0), ("y", 1)):
        coord = pos[:, axis] - center_xy[axis]
        other = pos[:, 1 - axis] - center_xy[1 - axis]
        near = np.abs(other) < 3.0 * bw
        if near.sum() < 8:
            continue
        rr, zz = coord[near], z[near]
        i2 = np.digitize(rr, edges) - 1
        rs, zs = [], []
        for b in range(len(edges) - 1):
            m = (i2 == b) & (rr > 0.0)
            if m.sum() >= 2:
                rs.append(0.5 * (edges[b] + edges[b + 1]))
                zs.append(float(np.max(zz[m])))
        if len(rs) >= 4:
            rs, zs = np.asarray(rs), np.asarray(zs)
            keep = (rs >= r_lo) & (rs <= r_hi)
            if keep.sum() >= 3:
                As = np.vstack([np.ones(int(keep.sum())), rs[keep]]).T
                cfs, *_ = np.linalg.lstsq(As, zs[keep], rcond=None)
                sections[name] = math.degrees(math.atan(max(-float(cfs[1]), 0.0)))

    return {
        "theta_deg": theta,
        "tan_theta": float(-slope),
        "r2": float(r2),
        "n_bins": int(fit.sum()),
        "r_pile_m": r_pile,
        "fit_range_m": [float(r_lo), float(r_hi)],
        "center_xy": [float(center_xy[0]), float(center_xy[1])],
        "section_theta_deg": sections,
        "surface": {"r": r_mid.tolist(), "z": z_surf.tolist(), "count": counts},
        "n_particles": int(len(pos)),
        "apex_height_m": float(np.max(z)),
    }


def numerical_checks(case) -> dict:
    """Time-step / resolution audit: reported, never silently relaxed."""
    mat = case.material
    dt = mat.timestep
    G = mat.kn / (2.0 * mat.radius)
    dt_rayleigh = math.pi * mat.radius * math.sqrt(mat.density / G)
    out = {
        "dt": dt,
        "dt_rayleigh": dt_rayleigh,
        "dt_over_rayleigh": dt / dt_rayleigh,
        "contact_period_steps": 20.0,
        "particle_diameter": mat.diameter,
    }
    if hasattr(case, "outlet_diameter"):
        out["outlet_over_d"] = case.outlet_diameter / mat.diameter
        out["hold_plan_particles"] = [case.length / mat.diameter,
                                      case.breadth / mat.diameter]
        out["fill_height_particles"] = case.fill_height / mat.diameter
    if hasattr(case, "funnel_side"):
        out["funnel_over_d"] = case.funnel_side / mat.diameter
    return out


# --------------------------------------------------------------------------- #
# simulations
# --------------------------------------------------------------------------- #
def simulate_hold_discharge(case: CargoHoldCase, *, return_history: bool = False,
                            keep_positions: bool = False) -> dict:
    """Load -> settle (plugged) -> open the outlet -> measure ``W`` by mass balance.

    ``W`` is ``dm_discharged / window`` over ``n_windows`` windows; the Beverloo
    value is *not* used anywhere in the measurement path.
    """
    mat = case.material
    r, mass, dt = mat.radius, mat.mass, case.timestep
    pos = _hold_bed(case)
    n0 = len(pos)
    vel = np.zeros((n0, 3))
    omega = np.zeros((n0, 3))
    state = ContactState()
    tang = {k: np.zeros((n0, 3)) for k in ("x0", "xL", "y0", "yB", "floor")}
    roll = {k: np.zeros((n0, 3)) for k in tang}
    alive = np.ones(n0, dtype=bool)

    n_settle = max(1, int(round(case.settle_time / dt)))
    n_dis = max(1, int(round(case.discharge_time / dt)))
    settle_ke = []
    for step in range(n_settle):
        idx_a = np.nonzero(alive)[0]
        pa, va, oa = pos[idx_a], vel[idx_a], omega[idx_a]
        iu, ju = build_pairs(pa, r)
        gi, gj = idx_a[iu], idx_a[ju]
        f, tq = contact_forces(pa, va, oa, iu, ju, state, mat, dt,
                               state_keys=gi * n0 + gj)
        fw, tw, _ = _hold_forces(pa, va, oa, mat, case, tang, dt,
                                 floor_open=False, gid=idx_a, roll=roll)
        f += fw
        tq += tw
        f[:, 2] -= mass * case.gravity
        va = va + f * (dt / mass)
        oa = oa + tq * (dt / mat.inertia)
        pa = pa + va * dt
        pos[idx_a], vel[idx_a], omega[idx_a] = pa, va, oa
        if step % 250 == 0:
            settle_ke.append(float(0.5 * mass * float((va ** 2).sum())))

    bed_top = float(np.quantile(pos[alive, 2], 0.995))
    rho_b = mass * n0 / (case.plan_area * bed_top)
    settle_rms_v = float(np.sqrt((vel[alive] ** 2).sum(axis=1).mean()))

    discharged = 0.0
    win = case.discharge_time / case.n_windows
    win_mass = [0.0]
    win_steps = [int(round((i + 1) * win / dt)) for i in range(case.n_windows)]
    step_of_win = 0
    for step in range(n_dis):
        idx_a = np.nonzero(alive)[0]
        pa, va, oa = pos[idx_a], vel[idx_a], omega[idx_a]
        iu, ju = build_pairs(pa, r)
        gi, gj = idx_a[iu], idx_a[ju]
        f, tq = contact_forces(pa, va, oa, iu, ju, state, mat, dt,
                               state_keys=gi * n0 + gj)
        fw, tw, _ = _hold_forces(pa, va, oa, mat, case, tang, dt,
                                 floor_open=True, gid=idx_a, roll=roll)
        f += fw
        tq += tw
        f[:, 2] -= mass * case.gravity
        va = va + f * (dt / mass)
        oa = oa + tq * (dt / mat.inertia)
        pa = pa + va * dt
        pos[idx_a], vel[idx_a], omega[idx_a] = pa, va, oa

        below = alive & (pos[:, 2] < -r)
        if below.any():
            discharged += mass * int(below.sum())
            alive[below] = False
        if step_of_win < len(win_steps) and step >= win_steps[step_of_win]:
            win_mass.append(discharged)
            step_of_win += 1
    # flush any windows the loop did not reach (short runs)
    while len(win_mass) < case.n_windows + 1:
        win_mass.append(discharged)

    rates = np.array([(win_mass[i + 1] - win_mass[i]) / win
                      for i in range(case.n_windows)])
    W_mean = float(rates.mean())
    W_std = float(rates.std(ddof=1)) if rates.size > 1 else 0.0
    t_edges = np.arange(1, case.n_windows + 1) * win
    r2 = 1.0
    if case.n_windows > 2 and np.ptp(np.array(win_mass[1:])) > 0:
        A = np.vstack([np.ones(case.n_windows), t_edges]).T
        coef, *_ = np.linalg.lstsq(A, np.array(win_mass[1:]), rcond=None)
        pred = A @ coef
        ss_res = float(((np.array(win_mass[1:]) - pred) ** 2).sum())
        ss_tot = float(((np.array(win_mass[1:]) - np.mean(win_mass[1:])) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    out = {
        "kind": "hold_discharge",
        "n_initial": int(n0),
        "n_discharged": int((~alive).sum()),
        "discharged_mass_kg": float(discharged),
        "flow_rate_kg_s": W_mean,
        "flow_rate_std_kg_s": W_std,
        "flow_rate_windows_kg_s": [float(x) for x in rates],
        "flow_steadiness_r2": float(r2),
        "bulk_density_kg_m3": float(rho_b),
        "packing_fraction": float(rho_b / case.density),
        "bed_top_m": bed_top,
        "settle_rms_velocity": settle_rms_v,
        "settle_kinetic_energy": settle_ke,
        "outlet_diameter": case.outlet_diameter,
        "particle_diameter": mat.diameter,
        "outlet_over_d": case.outlet_diameter / mat.diameter,
        "mu": case.mu,
        "mu_rolling": case.mu_rolling,
        "wall_mu": case.wall_mu,
        "dt": dt,
        "steps": n_settle + n_dis,
        "numerics": numerical_checks(case),
    }
    if keep_positions or return_history:
        out["positions"] = pos[alive]
        out["alive"] = alive
    return out


def simulate_hold_static(case: CargoHoldCase, *, n_slabs: int = 10) -> dict:
    """Settled, plugged hold -> wall normal stress and vertical stress profiles.

    ``sigma_h(z)`` = wall normal force per unit wall area of a horizontal slab;
    ``sigma_v(z)`` = force balance of the slice above ``z``,
    ``(W_above - F_wall_friction_above) / A`` -- no stress-tensor assumption.
    """
    mat = case.material
    r, mass, dt = mat.radius, mat.mass, case.timestep
    pos = _hold_bed(case)
    n0 = len(pos)
    vel = np.zeros((n0, 3))
    omega = np.zeros((n0, 3))
    state = ContactState()
    tang = {k: np.zeros((n0, 3)) for k in ("x0", "xL", "y0", "yB", "floor")}
    roll = {k: np.zeros((n0, 3)) for k in tang}
    n_steps = max(1, int(round(case.settle_time / dt)))
    sample_from = int(0.6 * n_steps)

    fn_acc = np.zeros(n0)
    fric_acc = np.zeros((n0, 3))
    n_samples = 0
    for step in range(n_steps):
        pa, va, oa = pos, vel, omega
        iu, ju = build_pairs(pa, r)
        f, tq = contact_forces(pa, va, oa, iu, ju, state, mat, dt,
                               state_keys=iu * n0 + ju)
        fw, tw, diag = _hold_forces(pa, va, oa, mat, case, tang, dt,
                                    floor_open=False, record=True, roll=roll)
        f += fw
        tq += tw
        f[:, 2] -= mass * case.gravity
        va = va + f * (dt / mass)
        oa = oa + tq * (dt / mat.inertia)
        pa = pa + va * dt
        pos, vel, omega = pa, va, oa
        if step >= sample_from:
            fn_acc += diag["fn_wall"]
            fric_acc += diag["fric_wall"]
            n_samples += 1
    if n_samples == 0:                                   # pragma: no cover
        n_samples = 1
    fn = fn_acc / n_samples
    fric = fric_acc / n_samples

    z = pos[:, 2]
    bed_top = float(np.quantile(z, 0.995))
    rho_b = mass * n0 / (case.plan_area * bed_top)
    edges = np.linspace(0.0, bed_top, n_slabs + 1)
    perimeter = 2.0 * (case.length + case.breadth)
    rows = []
    for i in range(n_slabs):
        lo, hi = edges[i], edges[i + 1]
        m = (z >= lo) & (z < hi)
        dz = hi - lo
        if m.sum() < 3:
            continue
        sigma_h = float(fn[m].sum() / (perimeter * dz))
        weight_above = float(mass * case.gravity * np.count_nonzero(z >= hi))
        fric_above = float(np.sum(fric[z >= hi, 2]))
        sigma_v = float((weight_above - fric_above) / case.plan_area)
        rows.append({"z_mid": float(0.5 * (lo + hi)), "z_lo": float(lo),
                     "z_hi": float(hi), "n_particles": int(m.sum()),
                     "sigma_h_Pa": max(sigma_h, 0.0),
                     "sigma_v_Pa": max(sigma_v, 0.0)})

    depth = np.array([row["z_mid"] for row in rows])
    sig_h = np.array([row["sigma_h_Pa"] for row in rows])
    sig_v = np.array([row["sigma_v_Pa"] for row in rows])
    # depth measured from the free surface (Janssen's z)
    depth_below_surface = bed_top - depth
    fit = fit_janssen(depth_below_surface, sig_v)
    fit_h = fit_janssen(depth_below_surface, sig_h)
    k_meas = float(np.median(sig_h / np.maximum(sig_v, 1e-30))) if sig_v.size else float("nan")
    hydro = rho_b * case.gravity * depth_below_surface
    rel_hydro = (sig_v - hydro) / np.maximum(hydro, 1e-30)
    return {
        "kind": "hold_static",
        "n_particles": int(n0),
        "bed_top_m": bed_top,
        "bulk_density_kg_m3": float(rho_b),
        "packing_fraction": float(rho_b / case.density),
        "hydraulic_radius_m": case.hydraulic_radius,
        "profile": rows,
        "janssen_fit_sigma_v": fit,
        "janssen_fit_sigma_h": fit_h,
        "K_measured": k_meas,
        "hydrostatic_reference_Pa": hydro.tolist(),
        "rel_dev_from_hydrostatic": rel_hydro.tolist(),
        "identity_p_inf_over_lambda_Pa_per_m": (
            fit["p_inf"] / fit["lambda"] if fit["lambda"] == fit["lambda"]
            and fit["lambda"] > 0 else float("nan")),
        # Janssen slab balance: d(sigma_v)/dz = rho_b g - mu_w K sigma_v / R_h
        # => lambda = R_h/(mu_w K), p_inf = rho_b g R_h/(mu_w K), hence the
        # geometry-free identity p_inf / lambda = rho_b g (parameter free!).
        "identity_theory_Pa_per_m": rho_b * case.gravity,
        "wall_mu": case.wall_mu,
        "settle_rms_velocity": float(np.sqrt((vel ** 2).sum(axis=1).mean())),
        "dt": dt,
        "steps": n_steps,
        "numerics": numerical_checks(case),
    }


def simulate_pour(case: PourCase) -> dict:
    """Pour cargo from a feed funnel onto a flat floor -> measured repose angle."""
    mat = case.material
    r, mass, dt = mat.radius, mat.mass, case.timestep
    pos = _funnel_bed(case)
    n0 = len(pos)
    vel = np.zeros((n0, 3))
    omega = np.zeros((n0, 3))
    state = ContactState()
    tang = {k: np.zeros((n0, 3)) for k in ("fx0", "fxS", "fy0", "fyS", "floor")}
    roll = {k: np.zeros((n0, 3)) for k in tang}
    n_steps = max(1, int(round(case.max_time / dt)))
    n_settle = max(1, int(round(case.settle_time / dt)))
    funnel_done_step = None
    move_hist = []
    ke_hist = []
    for step in range(n_steps):
        iu, ju = build_pairs(pos, r)
        f, tq = contact_forces(pos, vel, omega, iu, ju, state, mat, dt,
                               state_keys=iu * n0 + ju)
        fw, tw = _funnel_forces(pos, vel, omega, mat, case, tang, dt, roll=roll)
        f += fw
        tq += tw
        f[:, 2] -= mass * case.gravity
        vel = vel + f * (dt / mass)
        omega = omega + tq * (dt / mat.inertia)
        pos = pos + vel * dt
        if step % 500 == 0:
            ke = float(0.5 * mass * float((vel ** 2).sum()))
            ke_hist.append(ke)
            move_hist.append(float(np.abs(vel).max()))
        if funnel_done_step is None:
            above = pos[:, 2] > case.funnel_bottom + 1.5 * r
            if above.sum() < 3:
                funnel_done_step = step
        elif step - funnel_done_step > n_settle:
            break
    heap = measure_repose_angle(
        pos, radius=r, bin_width=0.5 * r,
        min_fit_radius=0.5 * case.funnel_side + r, surface_quantile=90.0)
    out = {
        "kind": "pour_heap",
        "repose": heap,
        "n_particles": int(n0),
        "mu": case.mu,
        "mu_rolling": case.mu_rolling,
        "floor_mu": case.floor_mu,
        "funnel_side": case.funnel_side,
        "funnel_bottom": case.funnel_bottom,
        "pour_complete_step": funnel_done_step,
        "max_velocity_hist": move_hist,
        "kinetic_energy_hist": ke_hist,
        "dt": case.timestep,
        "steps_run": step + 1,
        "numerics": numerical_checks(case),
        "positions": pos,
    }
    return out