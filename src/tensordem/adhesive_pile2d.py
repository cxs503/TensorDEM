"""Two-dimensional cohesive granular pile: repose angle vs adhesion (Bond number).

This module drives the adhesion laws of :mod:`tensordem.adhesion` in a small,
self-contained 2D DEM column-collapse experiment.  It exists to answer one
physical question with a *measurable* number:

    does turning on the JKR pull-off force make a granular pile steeper?

Model
-----
Disks of radius R and mass m in a rectangular box (floor + vertical walls), unit
gravity, monodisperse.  The contact laws are the standard "linear spring + dashpot
+ Coulomb friction + rolling resistance" DEM laws, augmented with a **JKR-anchored
cohesive branch**:

* compression ``delta > 0``::

      F_n = k_n delta - c_n v_n - F_pull            (clamped at F_n >= -F_pull)

* tension ``-delta_sep < delta <= 0`` (cohesive zone)::

      F_n = -F_pull (1 + delta/delta_sep)

* separated ``delta <= -delta_sep``: ``F_n = 0``

with the *anchored* pull-off force

    ``F_pull = jkr_pull_off_force(gamma_solid, R_eff)``  (``= 1.5 pi gamma R_eff``)

so ``F_pull = 0`` reproduces exactly the classical cohesionless Cundall-Strack
contact (the negative control).  The cohesive zone width ``delta_sep`` is a
numerical regularisation of the (physical) sub-nanometre JKR detachment
displacement; its value is reported and swept, and the heap result is checked to
be insensitive to it (see ``docs/adhesion_jkr_capillary.md`` §4.3).

All forces are expressed in reduced units (m = 1, R = 1/2, g = 1), so the only
dimensionless group that controls the heap is the **Bond number**

    Bo = F_pull / (m g)

plus the friction coefficients.  The mapping back to a physical powder
(R = 5 mm sphere, rho = 2500 kg/m^3) is given in the documentation.

Repose-angle measurement
------------------------
The free-surface height is sampled in vertical bins of one diameter; the flank is
fitted by least squares over the middle 75 % of the shoulder-to-toe span with a
one-sided 2-sigma clipping pass (same estimator as
:func:`tensordem.repose_calibration.repose_angle`, but in 2D).

Limitations (stated in the docs): 2D disks, monodisperse, no polydispersity, no
liquid bridges inside the heap (dry adhesion only), Coulomb rolling friction
without an elastic rolling state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .adhesion import jkr_pull_off_force

DTYPE = torch.float64

__all__ = [
    "PileCase",
    "initial_packing",
    "simulate_pile",
    "repose_angle_2d",
    "bond_number_sweep",
]


@dataclass(frozen=True)
class PileCase:
    """Column-collapse case (unit mass, unit gravity, monodisperse disks)."""

    mu: float = 0.40
    mu_wall: float = 0.40
    mu_rolling: float = 0.50
    bond_number: float = 0.0
    kn: float = 2.0e3
    kt_ratio: float = 0.5
    restitution: float = 0.2
    radius: float = 0.5
    mass: float = 1.0
    gravity: float = 1.0
    box_width_in_d: float = 52.0
    cohesion_range: float = 0.02  # delta_sep / radius
    settle_time: float = 6.0
    wedge_slope_deg: float = 45.0
    wedge_half_base_in_d: float = 6.0
    dt_factor: float = 0.02
    seed: int = 20261106

    # -- derived ------------------------------------------------------------
    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def inertia(self) -> float:
        return 0.5 * self.mass * self.radius ** 2

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    @property
    def box_width(self) -> float:
        return self.box_width_in_d * self.diameter

    @property
    def pull_off(self) -> float:
        """Cohesive pull-off force ``Bo * m g`` (in reduced units)."""
        return self.bond_number * self.mass * self.gravity

    @property
    def delta_sep(self) -> float:
        return self.cohesion_range * self.radius

    @property
    def dt(self) -> float:
        period = 2.0 * math.pi * math.sqrt(0.5 * self.mass / self.kn)
        return self.dt_factor * period

    @property
    def steps(self) -> int:
        return int(math.ceil(self.settle_time / self.dt))

    @property
    def damping(self) -> float:
        """Pair normal dashpot for the requested restitution."""
        e = self.restitution
        zeta = -math.log(e) / math.sqrt(math.pi ** 2 + math.log(e) ** 2) if 0.0 < e < 1.0 else 0.0
        return 2.0 * zeta * math.sqrt(self.kn * 0.5 * self.mass)

    def __post_init__(self) -> None:
        if not (0.0 < self.restitution < 1.0):
            raise ValueError("restitution must lie in (0, 1)")
        if self.bond_number < 0.0:
            raise ValueError("bond_number must be nonnegative")
        if not (0.0 < self.dt_factor < 0.2):
            raise ValueError("dt_factor must lie in (0, 0.2)")


def initial_packing(case: PileCase) -> torch.Tensor:
    """Hexagonally packed column at the left wall (small deterministic jitter)."""
    r = case.radius
    d = case.diameter
    dx = d * 1.02
    dy = d * math.sqrt(3.0) / 2.0 * 1.02
    g = torch.Generator().manual_seed(case.seed)
    pts = []
    for row in range(case.rows):
        for col in range(case.columns):
            x = r + col * dx + (0.5 * dx if row % 2 else 0.0)
            y = r + row * dy
            jitter = (torch.rand(2, generator=g, dtype=DTYPE) - 0.5) * 0.02 * d
            pts.append(torch.stack([torch.tensor(x, dtype=DTYPE) + jitter[0],
                                    torch.tensor(y, dtype=DTYPE) + jitter[1]]))
    return torch.stack(pts)


def initial_wedge(case: PileCase) -> torch.Tensor:
    """Triangular heap steeper than the repose angle (wedge relaxation test).

    The pile is released from a 45 deg wedge and avalanches down to its repose
    state; the relaxation involves only short rearrangements, so the heap settles
    in a few seconds (unlike a column collapse, whose long run-out keeps flowing).
    """
    r = case.radius
    d = case.diameter
    dx = d * 1.02
    dy = d * math.sqrt(3.0) / 2.0 * 1.02
    slope = math.tan(math.radians(case.wedge_slope_deg))
    half_base = case.wedge_half_base_in_d * d
    height = slope * half_base
    x_c = r + 2.0 * d + half_base
    g = torch.Generator().manual_seed(case.seed)
    pts = []
    row = 0
    y = r
    while y <= height:
        offset = 0.5 * dx if row % 2 else 0.0
        x = r + offset
        while x <= x_c + half_base:
            if y + r <= height - abs(x - x_c) * slope and x >= r:
                jitter = (torch.rand(2, generator=g, dtype=DTYPE) - 0.5) * 0.01 * d
                pts.append(torch.stack([torch.tensor(x, dtype=DTYPE) + jitter[0],
                                        torch.tensor(y, dtype=DTYPE) + jitter[1]]))
            x += dx
        y += dy
        row += 1
    if len(pts) < 20:
        raise ValueError("wedge is too small: increase wedge_half_base_in_d")
    return torch.stack(pts)


def _pair_index(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    iu, ju = torch.triu_indices(n, n, offset=1)
    return iu, ju


def simulate_pile(case: PileCase, *, return_particles: bool = False) -> dict[str, object]:
    """Run one cohesive column collapse and measure the repose angle.

    Returns a dict with the repose angle, the fit diagnostics, the Bond number,
    the pull-off force, the residual kinetic energy (settling check) and the
    timestep.
    """
    # small tensors: intra-op threading costs more than it saves (10x on 702-node
    # loong64), so the integration runs single-threaded and restores the setting.
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _simulate_pile_locked(case, return_particles=return_particles)
    finally:
        torch.set_num_threads(n_threads)


def _simulate_pile_locked(case: PileCase, *, return_particles: bool = False) -> dict[str, object]:
    """Integration loop of :func:`simulate_pile` (single-threaded)."""
    pos = initial_wedge(case)
    n = len(pos)
    r = case.radius
    m = case.mass
    inertia = case.inertia
    dt = case.dt
    kn = case.kn
    kt = case.kt
    c_n = case.damping
    f_pull = case.pull_off
    delta_sep = case.delta_sep
    mu = case.mu
    mu_wall = case.mu_wall
    mu_roll_r = case.mu_rolling * r

    vel = torch.zeros_like(pos)
    omega = torch.zeros(n, dtype=DTYPE)

    iu, ju = _pair_index(n)
    n_pairs = len(iu)
    tang = torch.zeros((n_pairs, 2), dtype=DTYPE)
    roll = torch.zeros(n_pairs, dtype=DTYPE)
    kr = kt * r * r            # rolling spring stiffness (N m / rad)

    box = case.box_width
    steps = case.steps

    for _ in range(steps):
        force = torch.zeros_like(pos)
        torque = torch.zeros(n, dtype=DTYPE)

        # ---- particle-particle contacts --------------------------------------
        dvec = pos[ju] - pos[iu]
        dist = torch.linalg.vector_norm(dvec, dim=1).clamp_min(1e-12)
        normal = dvec / dist[:, None]
        overlap = 2.0 * r - dist
        active = overlap > -delta_sep
        if bool(active.any()):
            idx = torch.nonzero(active, as_tuple=False).flatten()
            ov = overlap[idx]
            nrm = normal[idx]
            rel_v = vel[ju[idx]] - vel[iu[idx]]
            v_n = (rel_v * nrm).sum(1)

            # normal: compression (adhesive shift) or cohesive zone in tension
            f_comp = kn * ov - c_n * torch.clamp(v_n, max=0.0) - f_pull
            f_comp = torch.clamp(f_comp, min=-f_pull)
            f_coh = -f_pull * (1.0 + ov / delta_sep)
            f_n = torch.where(ov > 0.0, f_comp, f_coh)

            # tangential: Cundall-Strack spring with Coulomb cone |f_t| <= mu |f_n|
            # relative surface velocity at the contact: v_rel + r (w_i + w_j) x n
            # (Cundall-Strack; the two contact arms are opposite, hence the sum)
            surf_v = rel_v - v_n[:, None] * nrm
            surf_v = surf_v + torch.stack(
                [nrm[:, 1], -nrm[:, 0]], dim=1) * r * (
                    omega[ju[idx]] + omega[iu[idx]])[:, None]
            tang_new = tang[idx] + surf_v * dt
            tan_norm = torch.linalg.vector_norm(tang_new, dim=1)
            cap = mu * torch.abs(f_n) / kt
            scale = torch.where(tan_norm > cap, cap / tan_norm.clamp_min(1e-30),
                                torch.ones_like(tan_norm))
            tang_new = tang_new * scale[:, None]
            tang[idx] = tang_new
            if bool((~active).any()):
                tang[~active] = 0.0
            f_t = -kt * tang_new

            # rolling resistance: elastic-plastic rolling spring with the Coulomb
            # cap mu_roll * r * |F_n| (same regularisation as
            # tensordem.repose_calibration, which avoids the chatter of an
            # instantaneous Coulomb torque)
            omega_rel = omega[iu[idx]] - omega[ju[idx]]
            roll_new = roll[idx] + omega_rel * dt
            cap_roll = mu_roll_r * torch.abs(f_n) / kr
            roll_norm = roll_new.abs()
            scale_roll = torch.where(roll_norm > cap_roll,
                                     cap_roll / roll_norm.clamp_min(1e-30),
                                     torch.ones_like(roll_norm))
            roll_new = roll_new * scale_roll
            roll[idx] = roll_new
            if bool((~active).any()):
                roll[~active] = 0.0
            tau = -kr * roll_new

            f_pair = f_n[:, None] * nrm + f_t
            force.index_add_(0, ju[idx], f_pair)
            force.index_add_(0, iu[idx], -f_pair)

            # torque from the tangential force: the two contact arms are +/- r n
            # and the forces are opposite, so both disks spin the same way
            # (r_i x (-F) = r_j x F = -r n x F); the rolling torque opposes the
            # *relative* spin and therefore has opposite signs on the pair.
            tq = -r * (nrm[:, 0] * f_t[:, 1] - nrm[:, 1] * f_t[:, 0])
            torque.index_add_(0, iu[idx], tq + tau)
            torque.index_add_(0, ju[idx], tq - tau)

        # ---- container: floor, left wall, right wall (friction, no adhesion) --
        for axis, wall, sign in ((1, 0.0, +1.0), (0, 0.0, +1.0), (0, box, -1.0)):
            if sign > 0:
                pen = r - (pos[:, axis] - wall)
            else:
                pen = (pos[:, axis] - wall) + r
            hit = torch.nonzero(pen > 0.0, as_tuple=False).flatten()
            if len(hit) == 0:
                continue
            n_wall = torch.zeros((len(hit), 2), dtype=DTYPE)
            n_wall[:, axis] = -sign
            v_n = (vel[hit] * n_wall).sum(1)
            f = torch.clamp(kn * pen[hit] - c_n * torch.clamp(v_n, max=0.0), min=0.0)
            other = 1 - axis
            v_t = vel[hit, other].clone()
            f_t = -torch.clamp(kt * dt * v_t, min=-mu_wall * f, max=mu_wall * f)
            f_w = torch.zeros((len(hit), 2), dtype=DTYPE)
            f_w[:, axis] = f
            f_w[:, other] = f_t
            force.index_add_(0, hit, f_w)
            # wall friction acts at the contact point -r n_wall from the centre
            torque.index_add_(0, hit, -r * (n_wall[:, 0] * f_w[:, 1]
                                            - n_wall[:, 1] * f_w[:, 0]))

        force[:, 1] -= m * case.gravity

        vel = vel + force * (dt / m)
        omega = omega + torque * (dt / inertia)
        pos = pos + vel * dt

    speed = torch.linalg.vector_norm(vel, dim=1)
    result: dict[str, object] = {
        "n_particles": n,
        "dt": dt,
        "steps": steps,
        "bond_number": case.bond_number,
        "pull_off": f_pull,
        "cohesion_range_over_R": case.cohesion_range,
        "mu": case.mu,
        "mu_rolling": case.mu_rolling,
        "kn": kn,
        "max_speed": float(speed.max()),
        "mean_speed": float(speed.mean()),
        "kinetic_energy": float(0.5 * m * speed.square().sum()),
    }
    result.update(repose_angle_2d(pos, r))
    if return_particles:
        result["positions"] = pos
        result["velocities"] = vel
    return result


def repose_angle_2d(positions: torch.Tensor, radius: float, *,
                    min_count: int = 2,
                    band: tuple[float, float] = (0.15, 0.85)) -> dict[str, float]:
    """Free-surface slope of a 2D heap (degrees) by a height-band flank fit.

    The x axis is split into bins of one radius and the free-surface height of a
    bin is the highest particle in it.  Bins are restricted to the flank beyond
    the apex and to the height band ``band`` x apex height, which removes both
    the flat toe (scattered run-out grains) and the rounded crest from the fit.
    A least-squares line is fitted with one-sided 2-sigma clipping.

    Using a height band rather than a radial span makes the estimator insensitive
    to the long shallow run-out tail that a collapsed cohesive heap develops.
    """
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must be an (n, 2) tensor")
    if len(positions) < 6:
        raise ValueError("at least 6 particles are needed")

    x = positions[:, 0]
    y = positions[:, 1]
    width = radius
    n_bins = int(math.floor(float(x.max()) / width)) + 1
    bin_index = (x / width).floor().long().clamp_(0, n_bins - 1)

    heights: list[float] = []
    centres: list[float] = []
    for b in range(n_bins):
        mask = bin_index == b
        if int(mask.sum()) < min_count:
            continue
        heights.append(float(y[mask].max()))
        centres.append((b + 0.5) * width)
    if len(heights) < 6:
        raise ValueError("not enough bins on the free surface")

    h = torch.tensor(heights, dtype=DTYPE)
    c = torch.tensor(centres, dtype=DTYPE)
    apex = int(torch.argmax(h))
    apex_y = float(h[apex])
    lo_y = band[0] * apex_y
    hi_y = band[1] * apex_y
    window = (c > float(c[apex])) & (h >= lo_y) & (h <= hi_y)
    if int(window.sum()) < 3:
        raise ValueError("height band contains fewer than 3 bins")

    xx = c[window]
    yy = h[window]
    index = torch.arange(len(xx))
    slope = intercept = r2 = 0.0
    for _ in range(3):
        if len(index) < 3:
            break
        xk, yk = xx[index], yy[index]
        xm, ym = xk.mean(), yk.mean()
        sxx = ((xk - xm) ** 2).sum()
        sxy = ((xk - xm) * (yk - ym)).sum()
        slope = float(sxy / sxx)
        intercept = float(ym - slope * xm)
        resid = yk - (slope * xk + intercept)
        std = float(resid.std()) if len(xk) > 2 else 0.0
        ss_tot = ((yk - ym) ** 2).sum()
        r2 = 1.0 - float((resid ** 2).sum() / ss_tot) if float(ss_tot) > 0 else 0.0
        keep = resid < 2.0 * std if std > 0 else torch.ones_like(resid, dtype=torch.bool)
        if int(keep.sum()) < 3 or bool(keep.all()):
            break
        index = index[keep]
    return {
        "theta_r_deg": math.degrees(math.atan(-slope)),
        "slope": slope,
        "r2": r2,
        "shoulder_x": float(c[apex]),
        "toe_x": float(c[-1]),
        "apex_y": apex_y,
        "n_bins": len(heights),
        "n_fit_bins": int(len(index)),
    }


def bond_number_sweep(
    bond_numbers: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
    *,
    base: Optional[PileCase] = None,
    **kwargs,
) -> list[dict[str, object]]:
    """Run the heap for several Bond numbers and return one result row each."""
    base = base or PileCase(**kwargs)
    rows: list[dict[str, object]] = []
    for bo in bond_numbers:
        case = PileCase(**{**base.__dict__, "bond_number": bo})
        row = simulate_pile(case)
        row["theta_r_deg"] = row["theta_r_deg"]
        rows.append(row)
    return rows