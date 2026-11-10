"""Friction-angle -> angle-of-repose DEM calibration surface (TensorDEM m4970).

Motivation
----------
Granular DEM inputs are *microscopic* parameters (Coulomb friction coefficient
``mu``, rolling-resistance coefficient ``mu_r``), while the quantity an engineer
can measure in a hold, a hopper or an ice-rubble pile is *macroscopic*: the
angle of repose ``theta_r``.  Up to this module TensorDEM had no automatic
calibration bridge for that pair: every existing calibration tool
(``material_calibration``, ``calibrate_dem3d_restitution``,
``calibrate_dem3d_ucs``) targets stiffness, restitution or UCS strength, never a
free-surface angle.  This module adds the minimal surface that closes the loop:

    forward  : mu  ->  DEM column-collapse simulation  ->  theta_r
    build    : sorted (mu, theta_r) nodes  ->  monotone PCHIP calibration table
    inverse  : theta_r  ->  mu   (interpolated, recovery error reported)
    guard    : queries outside the tabulated bracket are REJECTED, never
               silently extrapolated.

Physics of the forward model
----------------------------
A hollow cylinder of spheres (hexagonal packing) rests on a frictional floor,
confined by a smooth cylindrical wall.  The wall is removed at
``t = hold_time`` and the material collapses into an axisymmetric heap; the
free-surface angle of that heap is the repose angle.  Contact laws are the
TensorDEM ones:

* normal: linear spring-dashpot, ``k_n`` and a damping ratio derived from the
  coefficient of restitution (same relation as
  ``tensordem.contact_tangential.restitution_to_damping``);
* tangential: Cundall-Strack incremental spring, saturated at ``mu |F_n|``;
* rolling: isotropic rolling spring (Ai et al. style) saturated at
  ``mu_r R_eff |F_n|`` -- without it a pile of ideal spheres cannot sustain any
  slope, which is exactly what the ``mu_r = 0`` negative control demonstrates.

Two falsifiable anchors ship with the module and are exercised by the tests:

A. *Irrotational slider*: a sphere whose rotation is frozen on a plane tilted at
   ``theta`` starts to slide exactly at ``theta = arctan(mu)`` (verifies the
   tangential law through time stepping inside this driver).
B. *Kinematic bound*: for cohesionless spheres the heap surface is stable only
   while ``tan(theta_r) <= mu``; every simulated pile is checked against it.

Limitations (see ``docs/repose_angle_calibration.md``): monodisperse spheres,
fixed ``mu_r`` for the primary table (joint ``(mu, mu_r)`` inversion is declared
as roadmap only), smooth wall, no cohesion, no particle-size distribution.  The
values are a *numerical* calibration of this solver, not laboratory values for a
specific bulk material.

All quantities are SI (m, kg, s, rad; angles reported in degrees).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

from .rolling_resistance import RollingResistanceConfig, rolling_resistance_torque

DTYPE = torch.float64


class ReposeCalibrationError(ValueError):
    """Base class for calibration-surface errors."""


class ReposeExtrapolationError(ReposeCalibrationError):
    """Raised when a query falls outside the tabulated calibration bracket."""


class NonMonotoneTableError(ReposeCalibrationError):
    """Raised when a calibration table is not strictly monotone increasing."""


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReposeCase:
    """Column-collapse repose-angle case (monodisperse spheres)."""

    mu: float
    mu_rolling: float = 0.05
    radius: float = 0.005
    density: float = 2500.0
    kn: float = 1.0e5          # N/m, pair normal stiffness
    kt_ratio: float = 0.5      # k_t / k_n
    restitution: float = 0.4
    mu_wall: float | None = None            # None -> mu
    mu_rolling_wall: float | None = None    # None -> mu_rolling
    gravity: float = 9.81
    cylinder_radius: float = 0.05
    fill_height: float = 0.08
    packing_gap: float = 0.02   # relative gap between neighbours, 2 R (1 + gap)
    jitter: float = 0.02        # deterministic positional jitter, x radius
    hold_time: float = 0.05     # wall fully down: pre-settle
    lift_time: float = 0.15     # wall bottom rises at fill_height/lift_time
    settle_time: float = 0.30   # free: deposit and relax to repose
    dt: float | None = None     # None -> stability-derived
    allow_rotation: bool = True
    seed: int = 4970

    # -- derived quantities -------------------------------------------------
    @property
    def particle_mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.particle_mass * self.radius ** 2

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    @property
    def wall_mu(self) -> float:
        return self.mu if self.mu_wall is None else self.mu_wall

    @property
    def wall_mu_rolling(self) -> float:
        return self.mu_rolling if self.mu_rolling_wall is None else self.mu_rolling_wall

    @property
    def timestep(self) -> float:
        """``dt`` resolving the stiffest linear spring by 20 steps per period.

        Contact period ``T = 2 pi sqrt(m_eff / k)`` with ``m_eff = m / 2`` for a
        pair.  The rolling spring has ``k_r = k_t R^2`` against ``I = 0.4 m R^2``,
        i.e. the same period up to the ``kt_ratio`` factor, so one bound covers
        both (``kt_ratio <= 1`` assumed).
        """
        if self.dt is not None:
            return self.dt
        period = 2.0 * math.pi * math.sqrt((0.5 * self.particle_mass) / self.kn)
        return period / 20.0

    def __post_init__(self) -> None:
        for name in ("mu", "mu_rolling"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0.0):
                raise ReposeCalibrationError(f"{name} must be finite and nonnegative")
        if self.mu <= 0.0:
            raise ReposeCalibrationError("mu must be strictly positive for a table")
        for name in ("radius", "density", "kn", "kt_ratio", "gravity",
                     "cylinder_radius", "fill_height", "hold_time", "lift_time",
                     "settle_time"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value > 0.0):
                raise ReposeCalibrationError(f"{name} must be finite and positive")
        if self.cylinder_radius <= self.radius:
            raise ReposeCalibrationError("cylinder_radius must exceed radius")
        if self.fill_height <= 2.0 * self.radius:
            raise ReposeCalibrationError("fill_height must exceed one particle layer")
        if not 0.0 <= self.restitution < 1.0:
            raise ReposeCalibrationError("restitution must lie in [0, 1)")
        if self.dt is not None and not (math.isfinite(self.dt) and self.dt > 0.0):
            raise ReposeCalibrationError("dt must be finite and positive")
        if not 0.0 <= self.jitter < 0.5:
            raise ReposeCalibrationError("jitter must lie in [0, 0.5)")
        if not 0.0 < self.packing_gap < 1.0:
            raise ReposeCalibrationError("packing_gap must lie in (0, 1)")


# ---------------------------------------------------------------------------
# Initial packing
# ---------------------------------------------------------------------------

def initial_packing(case: ReposeCase) -> torch.Tensor:
    """Deterministic hexagonal-close-packed column inside the cylinder wall."""
    r = case.radius
    spacing = 2.0 * r * (1.0 + case.packing_gap)
    layer_dz = spacing * math.sqrt(2.0 / 3.0)
    n_layers = max(1, int(math.floor((case.fill_height - r) / layer_dz)) + 1)
    r_max = case.cylinder_radius - r * 1.02
    generator = torch.Generator().manual_seed(case.seed)

    row_pitch = spacing * math.sqrt(3.0) / 2.0
    # Odd layers sit above the hollow sites of the layer below: the in-plane
    # shift is one lattice half-vector (spacing/2, spacing/(2 sqrt 3)) and the
    # vertical pitch is spacing sqrt(2/3), so every inter-layer contact is a
    # next-nearest distance of exactly `spacing` (hexagonal close packing).
    shift_x = 0.5 * spacing
    shift_y = spacing / (2.0 * math.sqrt(3.0))
    points: list[tuple[float, float, float]] = []
    for layer in range(n_layers):
        z = r + layer * layer_dz
        odd = layer % 2 == 1
        span = int(math.ceil((case.cylinder_radius + spacing) / spacing))
        for ix in range(-span, span + 1):
            for iy in range(-span, span + 1):
                x = ix * spacing + 0.5 * spacing * (iy % 2)
                y = iy * row_pitch
                if odd:
                    x += shift_x
                    y += shift_y
                if math.hypot(x, y) <= r_max:
                    points.append((x, y, z))
    positions = torch.tensor(points, dtype=DTYPE)
    if len(positions) < 8:
        raise ReposeCalibrationError(
            "packing produced fewer than 8 particles; enlarge the cylinder or radius")
    if case.jitter:
        noise = torch.randn(positions.shape, dtype=DTYPE, generator=generator)
        positions = positions + noise * (case.jitter * r)
    positions[:, 2].clamp_(min=case.radius * 1.001)
    return positions


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _pair_index(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    iu, ju = torch.triu_indices(n, n, offset=1)
    return iu, ju


def _damping_ratio(restitution: float) -> float:
    """zeta = |ln e| / sqrt(pi^2 + ln^2 e) (dashpot from restitution)."""
    e = max(min(restitution, 1.0 - 1e-12), 1e-6)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)



def _epsd_rolling_torque(
    spin: torch.Tensor,
    normal_force_mag: torch.Tensor,
    mu_r: float,
    k_r: float,
    radius: float,
    roll_state: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """EPSD rolling moment through the single shared implementation.

    Thin adapter around :func:`tensordem.rolling_resistance.rolling_resistance_torque`
    (Ai et al. 2011 ``"epsd"`` branch).  The caller stores the elastic *spin*
    history ``roll_state`` (rad); the library stores the elastic *moment*
    ``-k_r * roll_state`` (N*m), so the two are converted at the boundary.

    Returns ``(torque_on_j, roll_state_next)`` with ``|torque_on_j| <= mu_r R F_n``.
    """
    config = RollingResistanceConfig(model="epsd", mu_r=float(mu_r), k_r=float(k_r))
    moment, moment_next = rolling_resistance_torque(
        spin,
        normal_force_mag,
        torch.full_like(normal_force_mag, float(radius)),
        config,
        dt=dt,
        torque_state=-float(k_r) * roll_state,
    )
    return moment_next, -moment_next / float(k_r)

def simulate(case: ReposeCase, *, return_particles: bool = False
             ) -> dict[str, object]:
    """Run one column-collapse case and measure the repose angle.

    Returns the angle in degrees together with the flank-fit diagnostics, the
    particle count, the timestep, and the residual kinetic energy (settling
    check: a heap that is still flowing yields a meaningless angle).
    """
    positions = initial_packing(case)
    n = len(positions)
    r = case.radius
    mass = case.particle_mass
    inertia = case.inertia
    dt = case.timestep
    kn, kt = case.kn, case.kt
    kr = kt * r * r                      # rolling spring stiffness (N m/rad)
    zeta = _damping_ratio(case.restitution)
    gamma = 2.0 * zeta * math.sqrt(kn * 0.5 * mass)   # pair dashpot
    gamma_wall = 2.0 * zeta * math.sqrt(kn * mass)    # rigid-counterpart dashpot

    velocities = torch.zeros_like(positions)
    omega = torch.zeros_like(positions)

    iu, ju = _pair_index(n)
    n_pairs = len(iu)
    tang = torch.zeros((n_pairs, 3), dtype=DTYPE)
    roll = torch.zeros((n_pairs, 3), dtype=DTYPE)
    floor_tang = torch.zeros((n, 3), dtype=DTYPE)
    floor_roll = torch.zeros((n, 3), dtype=DTYPE)
    floor_n = torch.zeros((n, 3), dtype=DTYPE)
    wall_tang = torch.zeros((n, 3), dtype=DTYPE)

    total = case.hold_time + case.lift_time + case.settle_time
    steps = int(math.ceil(total / dt))

    for step in range(steps):
        t_now = step * dt
        if t_now <= case.hold_time:
            wall_bottom = 0.0
        else:
            wall_bottom = case.fill_height * min(
                1.0, (t_now - case.hold_time) / case.lift_time)

        force = torch.zeros_like(positions)
        torque = torch.zeros_like(positions)

        # ---- particle-particle contacts ---------------------------------
        dvec = positions[ju] - positions[iu]
        dist = torch.linalg.vector_norm(dvec, dim=1)
        overlap = 2.0 * r - dist
        active = overlap > 0.0
        if bool(active.any()):
            idx = torch.nonzero(active, as_tuple=False).flatten()
            normal = dvec[idx] / dist[idx].clamp_min(1e-30)[:, None]
            rel_v = velocities[ju[idx]] - velocities[iu[idx]]
            v_n = (rel_v * normal).sum(1)
            f_n = torch.clamp(kn * overlap[idx] - gamma * v_n, min=0.0)
            surf = rel_v
            if case.allow_rotation:
                surf = (surf + torch.linalg.cross(omega[ju[idx]], normal * (-r))
                        - torch.linalg.cross(omega[iu[idx]], normal * r))
            tang_new = tang[idx] + (surf - (surf * normal).sum(1)[:, None]) * dt
            tang_norm = torch.linalg.vector_norm(tang_new, dim=1)
            cap = case.mu * f_n / kt
            scale = torch.where(tang_norm > cap,
                                cap / tang_norm.clamp_min(1e-30),
                                torch.ones_like(tang_norm))
            tang_new = tang_new * scale[:, None]
            tang[idx] = tang_new
            f_t = -kt * tang_new
            # `normal` points from i to j: the contact force on j is
            # +f_n n + f_t and the reaction on i is its negative.
            pair_force = f_n[:, None] * normal + f_t
            force.index_add_(0, iu[idx], -pair_force)
            force.index_add_(0, ju[idx], pair_force)
            if case.allow_rotation:
                # Contact arm is r n for i and -r n for j; the tangential force
                # on i is -f_t, so both contact torques equal -r (n x f_t).
                contact_torque = -torch.linalg.cross(normal * r, f_t)
                torque.index_add_(0, iu[idx], contact_torque)
                torque.index_add_(0, ju[idx], contact_torque)
                spin = omega[ju[idx]] - omega[iu[idx]]
                # Internal couple opposing the relative spin (equal and
                # opposite, so it removes relative spin energy).  Single EPSD
                # implementation: tensordem.rolling_resistance.
                torque_r, roll_new = _epsd_rolling_torque(
                    spin, f_n, case.mu_rolling, kr, r, roll[idx], dt)
                roll[idx] = roll_new
                torque.index_add_(0, iu[idx], -torque_r)
                torque.index_add_(0, ju[idx], torque_r)
            inactive = ~active
            if bool(inactive.any()):
                tang[inactive] = 0.0
                roll[inactive] = 0.0

        # ---- floor plane at z = 0 ---------------------------------------
        pen_floor = r - positions[:, 2]
        touching = pen_floor > 0.0
        if bool(touching.any()):
            idx = torch.nonzero(touching, as_tuple=False).flatten()
            fn_floor = torch.clamp(
                kn * pen_floor[idx] - gamma_wall * velocities[idx, 2], min=0.0)
            radial_v = velocities[idx].clone()
            radial_v[:, 2] = 0.0
            tang_new = floor_tang[idx] + radial_v * dt
            tang_norm = torch.linalg.vector_norm(tang_new, dim=1)
            cap = case.wall_mu * fn_floor / kt
            scale = torch.where(tang_norm > cap,
                                cap / tang_norm.clamp_min(1e-30),
                                torch.ones_like(tang_norm))
            tang_new = tang_new * scale[:, None]
            floor_tang[idx] = tang_new
            ft_floor = -kt * tang_new
            force_col = torch.zeros((len(idx), 3), dtype=DTYPE)
            force_col[:, 2] = fn_floor
            force_col += ft_floor
            force.index_add_(0, idx, force_col)
            floor_n[idx] = fn_floor[:, None] * force_col.new_tensor([0.0, 0.0, 1.0])
            if case.allow_rotation:
                contact_arm = torch.tensor([[0.0, 0.0, -r]], dtype=DTYPE).expand(
                    len(idx), 3)
                torque.index_add_(0, idx, torch.linalg.cross(
                    contact_arm, ft_floor))
                torque_r, roll_new = _epsd_rolling_torque(
                    omega[idx], fn_floor, case.wall_mu_rolling, kr, r,
                    floor_roll[idx], dt)
                floor_roll[idx] = roll_new
                torque.index_add_(0, idx, torque_r)
        floor_tang[~touching] = 0.0
        floor_roll[~touching] = 0.0
        floor_n[~touching] = 0.0

        # ---- lifting cylindrical wall ------------------------------------
        # The wall bottom rises at fill_height/lift_time; only material above
        # the current bottom edge is confined, so the column drains from below
        # and deposits a heap instead of being released in one splash.
        if wall_bottom < case.fill_height:
            radial = positions[:, :2]
            rad = torch.linalg.vector_norm(radial, dim=1)
            pen = rad + r - case.cylinder_radius
            hit = (pen > 0.0) & (positions[:, 2] > wall_bottom)
            if bool(hit.any()):
                idx = torch.nonzero(hit, as_tuple=False).flatten()
                normal = torch.zeros((len(idx), 3), dtype=DTYPE)
                normal[:, :2] = -radial[idx] / rad[idx].clamp_min(1e-30)[:, None]
                v_n = (velocities[idx] * normal).sum(1)
                f = torch.clamp(kn * pen[idx] - gamma_wall * v_n, min=0.0)
                # Rough wall: Cundall-Strack tangential spring against the
                # static container wall, saturated at mu_wall |F_n|.
                surf_w = velocities[idx] - v_n[:, None] * normal
                if case.allow_rotation:
                    surf_w = surf_w + torch.linalg.cross(omega[idx], normal * (-r))
                tang_w = wall_tang[idx] + surf_w * dt
                tw_norm = torch.linalg.vector_norm(tang_w, dim=1)
                cap_w = case.wall_mu * f / kt
                scale_w = torch.where(tw_norm > cap_w,
                                      cap_w / tw_norm.clamp_min(1e-30),
                                      torch.ones_like(tw_norm))
                tang_w = tang_w * scale_w[:, None]
                wall_tang[idx] = tang_w
                ft_w = -kt * tang_w
                force.index_add_(0, idx, (f[:, None] * normal) + ft_w)
                if case.allow_rotation:
                    torque.index_add_(0, idx, torch.linalg.cross(
                        (normal * (-r)), ft_w))

        # ---- gravity ------------------------------------------------------
        force[:, 2] -= mass * case.gravity

        # ---- semi-implicit Euler -----------------------------------------
        velocities = velocities + force * (dt / mass)
        if case.allow_rotation:
            omega = omega + torque * (dt / inertia)
        positions = positions + velocities * dt

    speed = torch.linalg.vector_norm(velocities, dim=1)
    result: dict[str, object] = {
        "n_particles": n,
        "dt": dt,
        "steps": steps,
        "mu": case.mu,
        "mu_rolling": case.mu_rolling,
        "allow_rotation": case.allow_rotation,
        "restitution": case.restitution,
        "max_speed": float(speed.max()),
        "mean_speed": float(speed.mean()),
        "kinetic_energy_J": float(0.5 * mass * speed.square().sum()),
        "radius": r,
        "density": case.density,
        "kn": kn,
        "kt": kt,
        "cylinder_radius": case.cylinder_radius,
        "fill_height": case.fill_height,
    }
    result.update(repose_angle(positions, r))
    if return_particles:
        result["positions"] = positions
        result["velocities"] = velocities
    return result


# ---------------------------------------------------------------------------
# Repose-angle measurement
# ---------------------------------------------------------------------------

def repose_angle(positions: torch.Tensor, radius: float, *,
                 sectors: int = 8, min_count: int = 2) -> dict[str, float]:
    """Slope of the outer free surface of an axisymmetric heap, in degrees.

    The heap is described in cylindrical coordinates about the centroid of the
    deposit.  Each radial annulus of width ``2 radius`` is split into ``sectors``
    azimuthal sectors; the *maximum* elevation inside a sector is the
    free-surface height there, and the annulus height is the mean over sectors
    (this suppresses single-particle outliers).  A straight line is least-squares
    fitted over the flank between 20 % and 85 % of the shoulder-to-toe radial
    span, and ``theta_r = atan(-slope)``.

    Returns ``theta_r_deg`` plus the diagnostics ``slope``, ``r2``,
    ``shoulder_radius``, ``toe_radius``, ``apex_z``, ``n_bins``.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ReposeCalibrationError("positions must be an (n, 3) tensor")
    if len(positions) < 4:
        raise ReposeCalibrationError("at least 4 particles are needed")

    centre = positions[:, :2].mean(dim=0)
    rel = positions[:, :2] - centre
    radial = torch.linalg.vector_norm(rel, dim=1)
    theta = torch.atan2(rel[:, 1], rel[:, 0])
    z = positions[:, 2]

    width = radius
    n_bins = int(math.floor(float(radial.max()) / width)) + 1
    sector_index = ((theta + math.pi) / (2.0 * math.pi) * sectors).floor().long()
    sector_index.clamp_(0, sectors - 1)
    bin_index = (radial / width).floor().long().clamp_(0, n_bins - 1)

    heights: list[float] = []
    centres: list[float] = []
    for b in range(n_bins):
        in_bin = bin_index == b
        if int(in_bin.sum()) < min_count:
            continue
        sector_heights = [float(z[in_bin & (sector_index == s)].max())
                          for s in range(sectors)
                          if bool((in_bin & (sector_index == s)).any())]
        heights.append(sum(sector_heights) / len(sector_heights))
        centres.append((b + 0.5) * width)

    if len(heights) < 4:
        raise ReposeCalibrationError("not enough radial bins to fit a flank")

    heights_t = torch.tensor(heights, dtype=DTYPE)
    centres_t = torch.tensor(centres, dtype=DTYPE)
    apex = int(torch.argmax(heights_t))
    shoulder = float(centres_t[apex])
    toe = float(centres_t[-1])
    span = toe - shoulder
    if span <= 3.0 * width:
        raise ReposeCalibrationError(
            "heap has no measurable flank (shoulder-to-toe span too short)")
    lo = shoulder + 0.15 * span
    hi = shoulder + 0.90 * span
    window = (centres_t >= lo) & (centres_t <= hi)
    if int(window.sum()) < 3:
        raise ReposeCalibrationError("flank window has fewer than 3 bins")

    x = centres_t[window]
    y = heights_t[window]
    # One-sided 2-sigma clipping pass: a few grains ejected above the surface
    # must not drag the fitted flank slope.
    index = torch.arange(len(x))
    slope = intercept = 0.0
    r2 = 0.0
    for _ in range(3):
        if len(index) < 3:
            break
        xk, yk = x[index], y[index]
        x_mean, y_mean = xk.mean(), yk.mean()
        sxx = ((xk - x_mean) ** 2).sum()
        sxy = ((xk - x_mean) * (yk - y_mean)).sum()
        slope = float(sxy / sxx)
        intercept = float(y_mean - slope * x_mean)
        residual = yk - (slope * xk + intercept)
        std = float(residual.std()) if len(xk) > 2 else 0.0
        ss_tot = ((yk - y_mean) ** 2).sum()
        r2 = 1.0 - float((residual ** 2).sum() / ss_tot) if float(ss_tot) > 0 else 0.0
        keep_mask = residual < 2.0 * std
        if int(keep_mask.sum()) < 3 or bool(keep_mask.all()):
            break
        index = index[keep_mask]
    n_fit_bins = int(len(index))
    return {
        "theta_r_deg": math.degrees(math.atan(-slope)),
        "slope": slope,
        "r2": r2,
        "shoulder_radius": shoulder,
        "toe_radius": toe,
        "apex_z": float(heights_t.max()),
        "n_bins": len(heights),
        "n_fit_bins": n_fit_bins,
    }


# ---------------------------------------------------------------------------
# Anchor A: irrotational slider on a tilted plane
# ---------------------------------------------------------------------------

def sliding_displacement(mu: float, tilt_deg: float, *, radius: float = 0.005,
                         density: float = 2500.0, kn: float = 1.0e5,
                         restitution: float = 0.4, duration: float = 0.05,
                         dt: float | None = None, allow_rotation: bool = False,
                         mu_rolling: float = 0.0) -> dict[str, float]:
    """Down-slope displacement of one sphere on a plane tilted by ``tilt_deg``.

    The plane stays horizontal (``z = 0``) and gravity is rotated instead:
    ``g = g (sin theta, 0, -cos theta)``, so the test exercises exactly the
    contact laws of :func:`simulate` (Cundall-Strack tangential spring saturated
    at ``mu F_n``, rolling spring saturated at ``mu_r R F_n``; the wall friction
    coefficient is reused as the plane coefficient).

    Two closed-form anchors follow from the equilibrium of a single sphere:

    * ``allow_rotation=False``: it slides when ``tan theta > mu``;
    * ``allow_rotation=True``: it moves when ``tan theta > min(mu, mu_r)``
      (sliding needs ``tan theta <= mu``, rolling needs ``tan theta <= mu_r``).

    The steady result is either a static sphere (displacement ~ 0) or a sphere
    translating down-slope at ``a = g (sin theta - mu_eff cos theta)``.
    """
    if mu <= 0.0:
        raise ReposeCalibrationError("mu must be positive")
    if mu_rolling < 0.0:
        raise ReposeCalibrationError("mu_rolling must be nonnegative")
    if not 0.0 < tilt_deg < 90.0:
        raise ReposeCalibrationError("tilt_deg must lie in (0, 90)")
    r = radius
    mass = density * 4.0 / 3.0 * math.pi * r ** 3
    inertia = 0.4 * mass * r * r
    if dt is None:
        dt = 2.0 * math.pi * math.sqrt(0.5 * mass / kn) / 20.0
    kt = 0.5 * kn
    kr = kt * r * r
    zeta = _damping_ratio(restitution)
    gamma_wall = 2.0 * zeta * math.sqrt(kn * mass)
    tilt = math.radians(tilt_deg)
    g = 9.81
    g_vec = torch.tensor([g * math.sin(tilt), 0.0, -g * math.cos(tilt)], dtype=DTYPE)
    arm = torch.tensor([0.0, 0.0, -r], dtype=DTYPE)

    pos = torch.tensor([0.0, 0.0, r], dtype=DTYPE)
    vel = torch.zeros(3, dtype=DTYPE)
    omega = torch.zeros(3, dtype=DTYPE)
    tang = torch.zeros(3, dtype=DTYPE)
    roll = torch.zeros(3, dtype=DTYPE)
    steps = int(math.ceil(duration / dt))
    for _ in range(steps):
        pen = max(r - float(pos[2]), 0.0)
        f_n = max(kn * pen - gamma_wall * float(vel[2]), 0.0)
        surf = vel
        if allow_rotation:
            surf = vel + torch.linalg.cross(omega, arm)
        t_surf = surf.clone()
        t_surf[2] = 0.0
        tang = tang + t_surf * dt
        tnorm = float(torch.linalg.vector_norm(tang))
        cap = mu * f_n / kt
        if tnorm > cap:
            tang = tang * (cap / tnorm)
        f_t = -kt * tang
        force = torch.tensor([0.0, 0.0, f_n], dtype=DTYPE) + f_t + mass * g_vec
        torque = torch.zeros(3, dtype=DTYPE)
        if allow_rotation:
            torque = torch.linalg.cross(arm, f_t)
            roll = roll + omega * dt
            rnorm = float(torch.linalg.vector_norm(roll))
            cap_r = mu_rolling * r * f_n / kr
            if rnorm > cap_r:
                roll = roll * (cap_r / rnorm) if rnorm > 0.0 else roll
            torque = torque - kr * roll
        vel = vel + force * (dt / mass)
        if allow_rotation:
            omega = omega + torque * (dt / inertia)
        pos = pos + vel * dt
    return {
        "displacement": float(pos[0]),
        "speed": float(torch.linalg.vector_norm(vel)),
        "spin": float(torch.linalg.vector_norm(omega)),
        "tilt_deg": tilt_deg,
        "mu": mu,
        "mu_rolling": mu_rolling,
        "dt": dt,
        "duration": duration,
    }


def sliding_onset_angle(mu: float, *, tolerance_deg: float = 0.25,
                        slide_fraction: float = 0.25, duration: float = 0.15,
                        **_case_kwargs) -> dict[str, float]:
    """Critical tilt angle of the irrotational slider, by bisection.

    A sphere is held until ``mu * m g cos(theta) >= m g sin(theta)``; the onset is
    therefore ``theta* = arctan(mu)``.  The bisection brackets
    ``[0.5, 1.5] arctan(mu)`` and uses a displacement threshold of
    ``slide_fraction * radius`` over ``duration``.
    """
    predicted = math.degrees(math.atan(mu))
    lo, hi = 0.5 * predicted, min(1.5 * predicted, 89.0)
    radius = float(_case_kwargs.get("radius", 0.005))

    def slides(tilt: float) -> bool:
        out = sliding_displacement(mu, tilt, duration=duration, **_case_kwargs)
        return out["displacement"] > slide_fraction * radius

    if not slides(hi):
        raise ReposeCalibrationError(
            "slider did not slide at 1.5 arctan(mu); tangential law inactive")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if slides(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < tolerance_deg * 1e-3:
            break
    measured = 0.5 * (lo + hi)
    return {
        "mu": mu,
        "measured_deg": measured,
        "predicted_deg": predicted,
        "error_deg": measured - predicted,
        "relative_error": (measured - predicted) / predicted,
    }


# ---------------------------------------------------------------------------
# Monotone PCHIP calibration table + inverse lookup
# ---------------------------------------------------------------------------

def _pchip_slopes(x: Sequence[float], y: Sequence[float]) -> list[float]:
    """Fritsch-Carlson monotone piecewise-cubic slopes (no overshoot)."""
    n = len(x)
    if n < 2:
        raise ReposeCalibrationError("need at least two table nodes")
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    if any(hi <= 0.0 for hi in h):
        raise ReposeCalibrationError("table abscissae must be strictly increasing")
    delta = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    if n == 2:
        m[0] = m[1] = delta[0]
        return m
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    # shape-preserving one-sided end conditions (Fritsch-Carlson, 1980)
    m[0] = _end_slope(h[0], h[1], delta[0], delta[1])
    m[-1] = _end_slope(h[-1], h[-2], delta[-1], delta[-2])
    return m


def _end_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    """One-sided three-point endpoint slope, limited to preserve monotonicity."""
    m = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if m * d0 <= 0.0:
        return 0.0
    if d0 * d1 < 0.0 and abs(m) > abs(3.0 * d0):
        return 3.0 * d0
    return m


@dataclass(frozen=True)
class ReposeCalibrationTable:
    """Monotone ``mu -> theta_r`` calibration table with guarded inverse lookup.

    The table stores the DEM-measured pairs; interpolation is monotone PCHIP, so
    the interpolant preserves the table's monotonicity and the inverse is
    uniquely defined on the tabulated bracket.  Queries outside the bracket raise
    :class:`ReposeExtrapolationError` -- silent extrapolation of a calibrated
    surface is not allowed (the physical map saturates and is not measured
    there).
    """

    mu: tuple[float, ...]
    theta_deg: tuple[float, ...]
    mu_rolling: float = 0.05
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.mu) != len(self.theta_deg):
            raise ReposeCalibrationError("mu and theta_deg must have equal length")
        if len(self.mu) < 2:
            raise ReposeCalibrationError("table needs at least two nodes")
        for i in range(len(self.mu) - 1):
            if not self.mu[i + 1] > self.mu[i]:
                raise ReposeCalibrationError("mu nodes must be strictly increasing")
        for i in range(len(self.theta_deg) - 1):
            if not self.theta_deg[i + 1] > self.theta_deg[i]:
                raise NonMonotoneTableError(
                    "theta_r must increase strictly with mu; non-monotone "
                    f"interval at index {i} "
                    f"({self.theta_deg[i]:.4f} -> {self.theta_deg[i + 1]:.4f} deg)")
        object.__setattr__(self, "_slopes", tuple(_pchip_slopes(self.mu, self.theta_deg)))

    # -- forward ----------------------------------------------------------
    def theta_from_mu(self, mu: float, *, atol: float = 1e-12) -> float:
        mu_min, mu_max = self.mu[0], self.mu[-1]
        if mu < mu_min - atol or mu > mu_max + atol:
            raise ReposeExtrapolationError(
                f"mu={mu!r} is outside the calibrated bracket "
                f"[{mu_min}, {mu_max}]; extrapolation refused")
        x = min(max(mu, mu_min), mu_max)
        return self._evaluate(x)

    def _evaluate(self, x: float) -> float:
        xs, ys, ms = self.mu, self.theta_deg, self._slopes
        n = len(xs)
        if x >= xs[-1]:
            return ys[-1]
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= x:
                lo = mid
            else:
                hi = mid
        h = xs[hi] - xs[lo]
        t = (x - xs[lo]) / h
        t2, t3 = t * t, t * t * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        return h00 * ys[lo] + h10 * h * ms[lo] + h01 * ys[hi] + h11 * h * ms[hi]

    # -- inverse -----------------------------------------------------------
    def mu_from_theta(self, theta_deg: float, *, tol: float = 1e-12,
                      max_iter: int = 200) -> float:
        lo_y, hi_y = self.theta_deg[0], self.theta_deg[-1]
        if theta_deg < lo_y - 1e-12 or theta_deg > hi_y + 1e-12:
            raise ReposeExtrapolationError(
                f"theta_r={theta_deg!r} deg is outside the calibrated bracket "
                f"[{lo_y}, {hi_y}] deg; extrapolation refused")
        target = min(max(theta_deg, lo_y), hi_y)
        lo, hi = self.mu[0], self.mu[-1]
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            value = self._evaluate(mid)
            if abs(value - target) < tol or hi - lo < tol:
                return mid
            if value < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # -- diagnostics -------------------------------------------------------
    def bracket(self) -> dict[str, float]:
        return {
            "mu_min": self.mu[0], "mu_max": self.mu[-1],
            "theta_min_deg": self.theta_deg[0], "theta_max_deg": self.theta_deg[-1],
        }

    def to_json(self) -> dict:
        return {
            "kind": "tensordem-repose-calibration-table-v1",
            "mu_rolling": self.mu_rolling,
            "mu": list(self.mu),
            "theta_r_deg": list(self.theta_deg),
            "bracket": self.bracket(),
            "pchip_slopes": list(self._slopes),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_rows(cls, rows: Iterable[dict], *, mu_rolling: float = 0.05,
                  provenance: dict | None = None) -> "ReposeCalibrationTable":
        rows = sorted(rows, key=lambda row: float(row["mu"]))
        return cls(mu=tuple(float(row["mu"]) for row in rows),
                   theta_deg=tuple(float(row["theta_r_deg"]) for row in rows),
                   mu_rolling=mu_rolling,
                   provenance=dict(provenance or {}))

    @classmethod
    def from_json(cls, payload: dict) -> "ReposeCalibrationTable":
        return cls(mu=tuple(float(v) for v in payload["mu"]),
                   theta_deg=tuple(float(v) for v in payload["theta_r_deg"]),
                   mu_rolling=float(payload.get("mu_rolling", 0.05)),
                   provenance=dict(payload.get("provenance", {})))


# ---------------------------------------------------------------------------
# Table construction + inverse recovery verification
# ---------------------------------------------------------------------------

def build_calibration_table(mu_nodes: Sequence[float], *,
                            case_kwargs: dict | None = None,
                            mu_rolling: float | None = None
                            ) -> tuple[ReposeCalibrationTable, list[dict]]:
    """Run the forward model at every ``mu`` node and assemble the table.

    Returns ``(table, runs)``; ``runs`` keeps the full per-node record (angle,
    fit diagnostics, settling check) for the audit artifact.
    """
    kwargs = dict(case_kwargs or {})
    mu_rolling = (kwargs.get("mu_rolling", 0.05) if mu_rolling is None
                  else mu_rolling)
    runs: list[dict] = []
    for mu in mu_nodes:
        case = ReposeCase(mu=float(mu), mu_rolling=mu_rolling, **kwargs)
        runs.append(simulate(case))
    table = ReposeCalibrationTable.from_rows(
        runs, mu_rolling=mu_rolling,
        provenance={"method": "tensordem column-collapse DEM",
                    "case": {k: v for k, v in kwargs.items()}})
    return table, runs


def inverse_recovery(table: ReposeCalibrationTable,
                     check_mu: Sequence[float]) -> list[dict]:
    """Closed-loop check: ``mu -> theta_r -> mu_hat`` at off-node values.

    ``check_mu`` must lie strictly inside the tabulated bracket; each entry is
    simulated with the forward model, inverted through the table and compared.
    The relative error on ``mu`` is the quantity the acceptance criterion bounds.
    """
    results: list[dict] = []
    case_kwargs = dict(table.provenance.get("case", {}))
    for mu in check_mu:
        mu = float(mu)
        case = ReposeCase(mu=mu, mu_rolling=table.mu_rolling, **case_kwargs)
        forward = simulate(case)
        theta = float(forward["theta_r_deg"])
        mu_hat = table.mu_from_theta(theta)
        results.append({
            "mu_true": mu,
            "theta_r_deg": theta,
            "mu_recovered": mu_hat,
            "abs_error": mu_hat - mu,
            "rel_error": (mu_hat - mu) / mu,
            "table_interpolated_theta_deg": table.theta_from_mu(mu),
            "n_particles": forward["n_particles"],
            "max_speed": forward["max_speed"],
        })
    return results


def kinematic_bound_margin(theta_r_deg: float, mu: float) -> dict[str, float]:
    """Check ``tan theta_r <= mu`` (cohesionless sliding-friction bound)."""
    tan_theta = math.tan(math.radians(theta_r_deg))
    return {"tan_theta_r": tan_theta, "mu": mu,
            "margin": mu - tan_theta, "satisfied": tan_theta <= mu + 1e-9}


__all__ = [
    "ReposeCalibrationError",
    "ReposeExtrapolationError",
    "NonMonotoneTableError",
    "ReposeCase",
    "ReposeCalibrationTable",
    "initial_packing",
    "simulate",
    "repose_angle",
    "sliding_displacement",
    "sliding_onset_angle",
    "build_calibration_table",
    "inverse_recovery",
    "kinematic_bound_margin",
]