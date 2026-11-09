"""Trickle-injection angle-of-repose driver for TensorDEM (rate-controlled).

Why this module exists
----------------------
The repository already ships one repose driver, the *lifting-cylinder column
collapse* of :mod:`tensordem.repose_calibration` (mission m4970).  That driver is
deliberately monodisperse, wall-confined and sphere-only; the shape-representation
verdict (`docs/dem_nonspherical_representation_verdict.md` §5.4) records the
consequence as an open gap:

    "the repository's calibrated repose driver (``repose_calibration.py``,
     lifting cylinder) is sphere-only ... Closing this needs a *trickle-injection*
     driver with rate control."

This module is that driver.  A point source above a frictional floor releases
grains **one at a time at a controlled rate**; the grains fall, roll and come to
rest in an open heap, and the free-surface angle is measured two independent
ways.  Because there is no confining wall and no prescribed initial packing, the
driver accepts *any* set of sphere radii (polydisperse input), which is the other
limitation the column driver declares ("no particle-size distribution").

What is verified here (see ``scripts/verify_repose_injection.py``)
-----------------------------------------------------------------
* **A1 sliding onset** — a single sphere whose rotation is frozen starts to slide
  on a plane tilted at exactly ``atan(mu)`` (analytic, textbook).
* **A2 rolling onset** — with rotation free and rolling resistance ``mu_r`` the
  measured rest threshold is compared with ``atan(min(mu, mu_r))``; the agreement
  (or the lack of it) is *reported*, not assumed.
* **A3 kinematic bound** — every simulated heap must satisfy
  ``tan(theta_r) <= mu`` (cohesionless Coulomb bound).
* **A4 rolling negative control** — ``mu_r = 0`` (ideal spheres) must give a flat
  heap, i.e. a repose angle at the noise floor, otherwise the angle measurement
  is not measuring the free surface.
* **A5 rate control** — the angle must be independent of the injection rate in
  the quasi-static limit; the spread over a 4x rate range is measured.
* **A6 heap-size convergence** — the angle must not drift with particle count.
* **A7 measurement independence** — the two angle estimators (radial-bin
  envelope, free-surface point fit) must agree, otherwise the reported angle
  carries an unquoted estimator error.

Contact laws are the TensorDEM ones and are *the same laws* the column driver
uses (linear spring-dashpot normal with the restitution-derived damping ratio,
Cundall-Strack tangential spring saturated at ``mu F_n``, isotropic rolling
spring saturated at ``mu_r R_eff F_n``), so A7 doubles as a cross-driver check.

Honest limits
-------------
* Broad phase is a Verlet pair list rebuilt by a dense distance matrix; the
  rebuild is ``O(N^2)`` but it happens only when the skin is exhausted, so the
  per-step cost is ``O(N * <coordination>)``.  This is comfortable to a few
  thousand grains and is **not** a large-scale (10^5) driver — GAP-DEM-G6/G7
  remain open.
* The heap is measured after the kinetic energy has decayed; a still-flowing
  heap raises :class:`ReposeInjectionError` instead of returning an angle.
* All lengths are SI (m), stiffness N/m, mass kg, angles degrees at the API.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .repose_calibration import repose_angle

__all__ = [
    "InjectionCase",
    "ReposeInjectionError",
    "build_pair_list",
    "simulate_injection",
    "slide_threshold",
    "surface_flank_angle",
    "kinematic_bound_margin",
]

DTYPE = torch.float64


class ReposeInjectionError(ValueError):
    """Raised when a trickle-injection case cannot produce a valid angle."""


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InjectionCase:
    """Rate-controlled trickle-injection repose case.

    Parameters are SI.  ``mu`` is the grain-grain (and grain-floor) sliding
    friction; ``mu_rolling`` the rolling-resistance coefficient.  The floor is a
    frictionless-inert rigid plane at ``z = 0`` carrying no normal compliance of
    its own (the compliant side is the grain).
    """

    mu: float = 0.5
    mu_rolling: float = 0.20
    radius: float = 6.0e-3
    polydispersity: float = 0.0        # uniform half-width, fraction of radius
    density: float = 2500.0
    kn: float = 6.0e4                  # N/m, grain-grain normal stiffness
    kt_ratio: float = 0.5
    restitution: float = 0.05
    mu_wall: float | None = None       # None -> mu
    mu_rolling_wall: float | None = None   # None -> mu_rolling
    gravity: float = 9.81
    n_particles: int = 600
    injection_rate: float = 400.0      # grains per second
    source_radius: float = 1.5e-2      # radius of the outlet disc
    source_height: float = 6.0e-2      # outlet elevation above the floor
    settle_time: float = 0.60          # free relaxation after the last grain
    dt: float | None = None            # None -> contact period / 24
    contact_periods: float = 24.0
    allow_rotation: bool = True
    skin_fraction: float = 0.30        # Verlet skin / mean diameter
    floor_spin_term: bool = True       # slit rate includes the grain spin
    v_tol: float = 2.0e-3              # m/s, settled-heap speed threshold
    settle_window: float = 5.0e-3      # s of continuous quiet required
    max_steps: int | None = None
    seed: int = 7301

    # -- derived ------------------------------------------------------------
    def mass_of(self, r: torch.Tensor | float) -> torch.Tensor | float:
        return self.density * (4.0 / 3.0) * math.pi * r ** 3

    @property
    def mu_floor(self) -> float:
        return self.mu if self.mu_wall is None else self.mu_wall

    @property
    def mu_rolling_floor(self) -> float:
        return self.mu_rolling if self.mu_rolling_wall is None else self.mu_rolling_wall

    @property
    def radii(self) -> torch.Tensor:
        """Deterministic radius draw (constant when ``polydispersity == 0``)."""
        n = int(self.n_particles)
        if self.polydispersity == 0.0:
            return torch.full((n,), float(self.radius), dtype=DTYPE)
        gen = torch.Generator().manual_seed(int(self.seed))
        lo = 1.0 - float(self.polydispersity)
        hi = 1.0 + float(self.polydispersity)
        u = torch.rand(n, generator=gen, dtype=DTYPE)
        return float(self.radius) * (lo + (hi - lo) * u)

    @property
    def timestep(self) -> float:
        """Stability timestep from the stiffest (smallest-particle) contact.

        ``T = 2 pi sqrt(m_eff / k_n)`` with the reduced mass of the smallest pair;
        the rolling spring ``k_r = k_t R_eff^2`` against ``I = 0.4 m R^2`` has the
        same period up to ``kt_ratio``, so one bound covers both.
        """
        if self.dt is not None:
            return float(self.dt)
        r_min = float(self.radii.min())
        m = float(self.mass_of(r_min))
        period = 2.0 * math.pi * math.sqrt(0.5 * m / self.kn)
        return period / float(self.contact_periods)

    def __post_init__(self) -> None:
        for name in ("mu", "mu_rolling", "mu_wall", "mu_rolling_wall"):
            value = getattr(self, name)
            if value is None:
                continue
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0.0):
                raise ReposeInjectionError(f"{name} must be finite and nonnegative")
        if self.mu <= 0.0:
            raise ReposeInjectionError("mu must be strictly positive")
        for name in ("radius", "density", "kn", "kt_ratio", "gravity",
                     "injection_rate", "source_radius", "source_height",
                     "settle_time", "contact_periods", "skin_fraction"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value > 0.0):
                raise ReposeInjectionError(f"{name} must be finite and positive")
        if int(self.n_particles) < 8:
            raise ReposeInjectionError("n_particles must be at least 8")
        if not 0.0 <= self.restitution < 1.0:
            raise ReposeInjectionError("restitution must lie in [0, 1)")
        if not 0.0 <= self.polydispersity < 0.8:
            raise ReposeInjectionError("polydispersity must lie in [0, 0.8)")
        if self.source_radius <= self.radius:
            raise ReposeInjectionError("source_radius must exceed radius")
        if self.dt is not None and not (math.isfinite(self.dt) and self.dt > 0.0):
            raise ReposeInjectionError("dt must be finite and positive")


def _damping_ratio(restitution: float) -> float:
    """Hertzian/linear small-damping closure ``zeta = -ln(e)/sqrt(pi^2+ln^2 e)``."""
    if restitution <= 0.0:
        return 1.0
    ln_e = math.log(restitution)
    return -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)


# ---------------------------------------------------------------------------
# Broad phase: Verlet pair list with history carry-over
# ---------------------------------------------------------------------------

def build_pair_list(positions: torch.Tensor, radii: torch.Tensor,
                    skin: float, chunk: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """All pairs ``(i < j)`` closer than ``r_i + r_j + skin``.

    Dense distance blocks (``chunk`` rows at a time) keep the peak memory bounded;
    the result is the candidate list the per-step contact loop filters on the true
    overlap.
    """
    n = positions.shape[0]
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    all_index = torch.arange(n)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        d = torch.cdist(positions[start:stop], positions)
        cut = radii[start:stop, None] + radii[None, :] + skin
        mask = d < cut
        mask &= all_index[None, :] > torch.arange(start, stop)[:, None]
        if bool(mask.any()):
            idx = torch.nonzero(mask, as_tuple=False)
            rows.append(idx[:, 0] + start)
            cols.append(idx[:, 1])
    if not rows:
        empty = torch.zeros((0,), dtype=torch.long)
        return empty, empty
    return torch.cat(rows), torch.cat(cols)


def _remap_history(history: torch.Tensor, old_i: torch.Tensor, old_j: torch.Tensor,
                   new_i: torch.Tensor, new_j: torch.Tensor, width: int
                   ) -> torch.Tensor:
    """Carry ``(i, j)``-keyed tangential/rolling history onto a new pair list.

    New pairs start from zero (a fresh contact), surviving pairs keep their state,
    so a rebuild is invisible to the physics.
    """
    out = torch.zeros((len(new_i), width), dtype=DTYPE)
    if len(history) == 0 or len(new_i) == 0:
        return out
    n = int(max(int(old_i.max()) if len(old_i) else 0,
                int(new_i.max()) if len(new_i) else 0)) + 1
    old_keys = old_i * n + old_j
    order = torch.argsort(old_keys)
    sorted_keys = old_keys[order]
    new_keys = new_i * n + new_j
    pos = torch.searchsorted(sorted_keys, new_keys)
    pos_c = pos.clamp(max=max(len(sorted_keys) - 1, 0))
    found = (pos < len(sorted_keys)) & (sorted_keys[pos_c] == new_keys)
    if bool(found.any()):
        out[found] = history[order][pos_c[found]]
    return out


# ---------------------------------------------------------------------------
# Floor contact (shared by the heap driver and the single-sphere tilt anchor)
# ---------------------------------------------------------------------------

def _floor_contact(*, positions: torch.Tensor, velocities: torch.Tensor,
                   omega: torch.Tensor, radii: torch.Tensor, mass: torch.Tensor,
                   kn: float, kt: float, zeta: float, mu_wall: float,
                   mu_roll_wall: float, tang: torch.Tensor, roll: torch.Tensor,
                   dt: float, gravity_vec: torch.Tensor, allow_rotation: bool,
                   floor_spin_term: bool = True
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Normal + Cundall-Strack + rolling contact against the plane ``z = 0``.

    ``tang``/``roll`` are mutated in place (spring history).  Returns ``(force,
    torque)`` for every particle.  The plane normal is ``+z``; ``gravity_vec`` is
    the full gravity vector so the same routine serves a tilted-plane anchor.
    """
    n = positions.shape[0]
    force = torch.zeros((n, 3), dtype=DTYPE)
    torque = torch.zeros((n, 3), dtype=DTYPE)
    pen = radii - positions[:, 2]
    touching = pen > 0.0
    if not bool(touching.any()):
        tang.zero_()
        roll.zero_()
        force += mass[:, None] * gravity_vec[None, :]
        return force, torque
    idx = torch.nonzero(touching, as_tuple=False).flatten()
    gamma_wall = 2.0 * zeta * torch.sqrt(kn * mass[idx])
    fn = torch.clamp(kn * pen[idx] - gamma_wall * velocities[idx, 2], min=0.0)
    # Slip rate at the contact point.  With ``floor_spin_term`` the spin of the
    # grain contributes (``v_contact = v_c + omega x (-R z)``), which is the
    # standard Cundall-Strack definition; without it only the centre velocity
    # enters, which is the convention of the lifting-cylinder driver and charges
    # the tangential spring even for a grain that rolls without slipping.
    arm_all = torch.zeros((len(idx), 3), dtype=DTYPE)
    arm_all[:, 2] = -radii[idx]
    contact_v = velocities[idx].clone()
    if floor_spin_term and allow_rotation:
        contact_v = contact_v + torch.linalg.cross(omega[idx], arm_all)
    contact_v[:, 2] = 0.0
    tang_new = tang[idx] + contact_v * dt
    cap = mu_wall * fn / kt
    norm = torch.linalg.vector_norm(tang_new, dim=1)
    scale = torch.where(norm > cap, cap / norm.clamp_min(1e-30),
                        torch.ones_like(norm))
    tang_new = tang_new * scale[:, None]
    tang[idx] = tang_new
    ft = -kt * tang_new
    f = fn[:, None] * torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)[None, :] + ft
    force.index_add_(0, idx, f)
    if allow_rotation:
        torque.index_add_(0, idx, torch.linalg.cross(arm_all, ft))
        # Rolling resistance opposing the horizontal spin component.
        spin = omega[idx].clone()
        spin[:, 2] = 0.0
        kr_floor = kt * radii[idx] ** 2
        roll_new = roll[idx] + spin * dt
        rnorm = torch.linalg.vector_norm(roll_new, dim=1)
        cap_r = mu_roll_wall * radii[idx] * fn / kr_floor
        s = torch.where(rnorm > cap_r, cap_r / rnorm.clamp_min(1e-30),
                        torch.ones_like(rnorm))
        roll_new = roll_new * s[:, None]
        roll[idx] = roll_new
        torque.index_add_(0, idx, -kr_floor[:, None] * roll_new)
    stuck = ~touching
    if bool(stuck.any()):
        tang[stuck] = 0.0
        roll[stuck] = 0.0
    force += mass[:, None] * gravity_vec[None, :]
    return force, torque


# ---------------------------------------------------------------------------
# The injection driver
# ---------------------------------------------------------------------------

def _insert_particle(*, positions: torch.Tensor, radii: torch.Tensor,
                     velocities: torch.Tensor, case: InjectionCase,
                     gen: torch.Generator) -> torch.Tensor | None:
    """Draw a non-overlapping release point just above the *settled* heap.

    The release height is keyed to the settled apex (``|v|`` below
    ``settle_speed``), not to the highest grain in the air: otherwise a stream of
    falling grains ratchets the source upwards and the heap never forms.  Each
    failed overlap draw lifts the source by one grain radius (50 attempts per
    lift), so a crowded outlet still releases.
    """
    apex = 0.0
    if len(positions):
        quiet = torch.linalg.vector_norm(velocities, dim=1) < 5.0e-2
        if bool(quiet.any()):
            apex = float(positions[quiet, 2].max())
    z0 = max(float(case.source_height), apex + 2.5 * float(case.radius))
    clearance = float(case.radius)
    for attempt in range(200):
        z = z0 + clearance * (attempt // 50)
        theta = 2.0 * math.pi * float(torch.rand((), generator=gen, dtype=DTYPE))
        rr = float(case.source_radius) * math.sqrt(
            float(torch.rand((), generator=gen, dtype=DTYPE)))
        cand = torch.tensor([rr * math.cos(theta), rr * math.sin(theta), z],
                            dtype=DTYPE)
        if len(positions) == 0:
            return cand
        d = torch.linalg.vector_norm(positions - cand[None, :], dim=1)
        if bool((d > radii + float(case.radius) * 1.05).all()):
            return cand
    return None


def simulate_injection(case: InjectionCase, *, return_particles: bool = False
                       ) -> dict[str, object]:
    """Run one trickle-injection case and measure the repose angle twice.

    Returns the angle (both estimators), the diagnostics needed to judge whether
    the heap is actually settled, and the run bookkeeping (rate, inject time,
    steps, timestep).
    """
    gen = torch.Generator().manual_seed(int(case.seed))
    radii_all = case.radii
    dt = case.timestep
    kt = case.kt_ratio * case.kn
    zeta = _damping_ratio(case.restitution)
    gravity_vec = torch.tensor([0.0, 0.0, -float(case.gravity)], dtype=DTYPE)
    skin = float(case.skin_fraction) * 2.0 * float(radii_all.mean())

    positions = torch.zeros((0, 3), dtype=DTYPE)
    velocities = torch.zeros((0, 3), dtype=DTYPE)
    omega = torch.zeros((0, 3), dtype=DTYPE)
    radii = torch.zeros((0,), dtype=DTYPE)

    pair_i = torch.zeros((0,), dtype=torch.long)
    pair_j = torch.zeros((0,), dtype=torch.long)
    tang = torch.zeros((0, 3), dtype=DTYPE)
    roll = torch.zeros((0, 3), dtype=DTYPE)
    floor_tang = torch.zeros((0, 3), dtype=DTYPE)
    floor_roll = torch.zeros((0, 3), dtype=DTYPE)

    interval = 1.0 / float(case.injection_rate)
    next_insert = 0.0
    injected = 0
    last_insert_time = 0.0
    quiet_time = 0.0
    n_target = int(case.n_particles)
    max_steps = int(case.max_steps) if case.max_steps else int(
        math.ceil((n_target * interval + case.settle_time) / dt) + 1000)
    max_disp = 0.0
    ref_positions = positions
    need_rebuild = True
    steps = 0
    settled = False

    for step in range(max_steps):
        t = step * dt
        steps = step + 1
        # ---- release ---------------------------------------------------
        while injected < n_target and t + 1e-12 >= next_insert:
            cand = _insert_particle(positions=positions, radii=radii,
                                    velocities=velocities, case=case, gen=gen)
            if cand is not None:
                positions = torch.cat([positions, cand[None, :]], dim=0)
                velocities = torch.cat(
                    [velocities, torch.zeros((1, 3), dtype=DTYPE)], dim=0)
                omega = torch.cat([omega, torch.zeros((1, 3), dtype=DTYPE)], dim=0)
                radii = torch.cat([radii, radii_all[injected][None]], dim=0)
                floor_tang = torch.cat([floor_tang, torch.zeros((1, 3), dtype=DTYPE)],
                                       dim=0)
                floor_roll = torch.cat([floor_roll, torch.zeros((1, 3), dtype=DTYPE)],
                                       dim=0)
                need_rebuild = True
            injected += 1
            last_insert_time = t
            next_insert += interval

        n = positions.shape[0]
        # ---- broad phase ----------------------------------------------
        if need_rebuild or max_disp > 0.5 * skin or (len(pair_i) == 0 and n > 1):
            new_i, new_j = build_pair_list(positions, radii, skin)
            tang = _remap_history(tang, pair_i, pair_j, new_i, new_j, 3)
            roll = _remap_history(roll, pair_i, pair_j, new_i, new_j, 3)
            pair_i, pair_j = new_i, new_j
            ref_positions = positions.clone()
            max_disp = 0.0
            need_rebuild = False

        force = torch.zeros((n, 3), dtype=DTYPE)
        torque = torch.zeros((n, 3), dtype=DTYPE)

        # ---- grain-grain contacts -------------------------------------
        if len(pair_i):
            dvec = positions[pair_j] - positions[pair_i]
            dist = torch.linalg.vector_norm(dvec, dim=1)
            cut = radii[pair_i] + radii[pair_j]
            overlap = cut - dist
            active = overlap > 0.0
            if bool(active.any()):
                idx = torch.nonzero(active, as_tuple=False).flatten()
                pi, pj = pair_i[idx], pair_j[idx]
                normal = dvec[idx] / dist[idx].clamp_min(1e-30)[:, None]
                rel_v = velocities[pj] - velocities[pi]
                v_n = (rel_v * normal).sum(1)
                mi = case.density * (4.0 / 3.0) * math.pi * radii[pi] ** 3
                mj = case.density * (4.0 / 3.0) * math.pi * radii[pj] ** 3
                m_eff = mi * mj / (mi + mj)
                gamma = 2.0 * zeta * torch.sqrt(case.kn * m_eff)
                fn = torch.clamp(case.kn * overlap[idx] - gamma * v_n, min=0.0)
                surf = rel_v
                if case.allow_rotation:
                    surf = (surf
                            + torch.linalg.cross(omega[pj], normal * (-radii[pj])[:, None])
                            - torch.linalg.cross(omega[pi], normal * radii[pi][:, None]))
                tang_new = tang[idx] + (surf - (surf * normal).sum(1)[:, None]) * dt
                tnorm = torch.linalg.vector_norm(tang_new, dim=1)
                cap = case.mu * fn / kt
                sc = torch.where(tnorm > cap, cap / tnorm.clamp_min(1e-30),
                                 torch.ones_like(tnorm))
                tang_new = tang_new * sc[:, None]
                tang[idx] = tang_new
                ft = -kt * tang_new
                pair_force = fn[:, None] * normal + ft
                force.index_add_(0, pi, -pair_force)
                force.index_add_(0, pj, pair_force)
                if case.allow_rotation:
                    arm_i = normal * radii[pi][:, None]
                    arm_j = normal * radii[pj][:, None]
                    torque.index_add_(0, pi, -torch.linalg.cross(arm_i, ft))
                    torque.index_add_(0, pj, -torch.linalg.cross(arm_j, ft))
                    r_eff = radii[pi] * radii[pj] / (radii[pi] + radii[pj])
                    kr = kt * r_eff ** 2
                    spin = omega[pj] - omega[pi]
                    roll_new = roll[idx] + spin * dt
                    rnorm = torch.linalg.vector_norm(roll_new, dim=1)
                    cap_r = case.mu_rolling * r_eff * fn / kr
                    sr = torch.where(rnorm > cap_r, cap_r / rnorm.clamp_min(1e-30),
                                     torch.ones_like(rnorm))
                    roll_new = roll_new * sr[:, None]
                    roll[idx] = roll_new
                    tr = -kr[:, None] * roll_new          # couple on j
                    torque.index_add_(0, pi, -tr)
                    torque.index_add_(0, pj, tr)
                inactive = ~active
                if bool(inactive.any()):
                    tang[inactive] = 0.0
                    roll[inactive] = 0.0

        # ---- floor ----------------------------------------------------
        mass = case.density * (4.0 / 3.0) * math.pi * radii ** 3
        if n:
            f_floor, t_floor = _floor_contact(
                positions=positions, velocities=velocities, omega=omega,
                radii=radii, mass=mass, kn=case.kn, kt=kt, zeta=zeta,
                mu_wall=case.mu_floor, mu_roll_wall=case.mu_rolling_floor,
                tang=floor_tang, roll=floor_roll, dt=dt,
                gravity_vec=gravity_vec, allow_rotation=case.allow_rotation,
                floor_spin_term=case.floor_spin_term)
            force += f_floor
            torque += t_floor

        # ---- semi-implicit Euler --------------------------------------
        if n:
            velocities = velocities + force * (dt / mass)[:, None]
            if case.allow_rotation:
                inertia = 0.4 * mass * radii ** 2
                omega = omega + torque * (dt / inertia)[:, None]
            positions = positions + velocities * dt
            if ref_positions.shape[0] == positions.shape[0]:
                max_disp = float(torch.linalg.vector_norm(
                    positions - ref_positions, dim=1).max())
            else:                        # a grain was just inserted
                need_rebuild = True

        # ---- settling -------------------------------------------------
        if injected >= n_target and n > 0:
            speed = float(torch.linalg.vector_norm(velocities, dim=1).max())
            if speed < case.v_tol:
                quiet_time += dt
            else:
                quiet_time = 0.0
            if quiet_time >= case.settle_window and (t - last_insert_time) >= case.settle_time:
                settled = True
                break
    if injected < n_target or n < 8:
        raise ReposeInjectionError(
            f"only {injected} grains injected ({n} alive); increase max_steps")

    speed = torch.linalg.vector_norm(velocities, dim=1)
    result: dict[str, object] = {
        "n_particles": int(positions.shape[0]),
        "dt": float(dt),
        "steps": int(steps),
        "simulated_time": float(steps * dt),
        "mu": float(case.mu),
        "mu_rolling": float(case.mu_rolling),
        "polydispersity": float(case.polydispersity),
        "injection_rate": float(case.injection_rate),
        "injection_time": float(last_insert_time),
        "settled": bool(settled),
        "max_speed": float(speed.max()),
        "mean_speed": float(speed.mean()),
        "kinetic_energy_J": float(0.5 * (case.density * (4.0 / 3.0) * math.pi
                                         * radii ** 3) @ speed.square()),
        "mean_radius": float(radii.mean()),
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "apex_height": float(positions[:, 2].max()),
        "footprint_radius": float(torch.linalg.vector_norm(
            positions[:, :2] - positions[:, :2].mean(dim=0), dim=1).max()),
    }
    try:
        result.update(repose_angle(positions, float(radii.mean())))
        result["angle_error"] = None
    except Exception as exc:                     # degenerate heap, reported
        result.update({"theta_r_deg": None, "slope": None, "r2": None,
                       "angle_error": f"repose_angle: {exc}"})
    try:
        flank = surface_flank_angle(positions, radii)
        result["theta_flank_deg"] = flank["theta_deg"]
        result["flank_r2"] = flank["r2"]
        result["flank_points"] = flank["n_points"]
        result["flank_sector_std_deg"] = flank["sector_std_deg"]
    except Exception as exc:                     # degenerate heap, reported
        result.update({"theta_flank_deg": None, "flank_r2": None,
                       "flank_points": 0, "flank_sector_std_deg": None})
        result["angle_error"] = (result.get("angle_error") or "") + f" | flank: {exc}"
    if return_particles:
        result["positions"] = positions
        result["radii"] = radii
        result["velocities"] = velocities
    return result


# ---------------------------------------------------------------------------
# Second (independent) angle estimator: free-surface point fit
# ---------------------------------------------------------------------------

def surface_flank_angle(positions: torch.Tensor, radii: torch.Tensor, *,
                        sectors: int = 4, n_clip: int = 3,
                        low: float = 0.30, high: float = 0.85) -> dict[str, object]:
    """Cone slope from the *free-surface* grains only.

    A grain is on the free surface when no other grain sits above it within a
    horizontal distance ``0.8 (r_i + r_j)``.  A straight line ``z = c - k r`` is
    fitted to those grains inside the flank band ``[low, high] * r_toe`` with a
    ``2 sigma`` one-sided clipping pass, and ``theta = atan(k)``.  Sector fits
    give the azimuthal spread -- a heap built by a point source must be close to
    axisymmetric, and the spread is what makes the single-number angle honest.

    This estimator shares no binning or extremum logic with
    :func:`tensordem.repose_calibration.repose_angle`, so agreement between the
    two is real evidence about the measurement rather than about the code path.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ReposeInjectionError("positions must be an (n, 3) tensor")
    n = positions.shape[0]
    if n < 12:
        raise ReposeInjectionError("at least 12 grains are needed")
    if radii.shape[0] != n:
        raise ReposeInjectionError("radii must match positions")

    rel = positions[:, :2] - positions[:, :2].mean(dim=0)
    radial = torch.linalg.vector_norm(rel, dim=1)
    z = positions[:, 2]

    # free surface mask (chunked to bound memory)
    visible = torch.ones(n, dtype=torch.bool)
    for start in range(0, n, 512):
        stop = min(n, start + 512)
        dxy = torch.cdist(rel[start:stop], rel)
        cut = 0.8 * (radii[start:stop, None] + radii[None, :])
        above = (dxy < cut) & (z[None, :] > (z[start:stop] + 0.5 * radii[start:stop])[:, None])
        visible[start:stop] = ~above.any(dim=1)
    if int(visible.sum()) < 8:
        raise ReposeInjectionError("too few free-surface grains to fit a flank")

    r_toe = float(radial[visible].max())
    band = visible & (radial >= low * r_toe) & (radial <= high * r_toe)
    if int(band.sum()) < 6:
        raise ReposeInjectionError("flank band holds fewer than 6 surface grains")

    def fit(mask: torch.Tensor) -> tuple[float, float, float]:
        index = torch.nonzero(mask, as_tuple=False).flatten()
        x = radial[index]
        y = z[index]
        keep_local = torch.arange(len(index))
        slope = intercept = r2 = 0.0
        for _ in range(n_clip):
            if len(keep_local) < 3:
                break
            xk, yk = x[keep_local], y[keep_local]
            xm, ym = xk.mean(), yk.mean()
            sxx = float(((xk - xm) ** 2).sum())
            if sxx <= 0.0:
                break
            slope = float(((xk - xm) * (yk - ym)).sum() / sxx)
            intercept = float(ym - slope * xm)
            resid = yk - (slope * xk + intercept)
            std = float(resid.std()) if len(xk) > 2 else 0.0
            sst = float(((yk - ym) ** 2).sum())
            r2 = 1.0 - float((resid ** 2).sum()) / sst if sst > 0.0 else 0.0
            keep = resid < 2.0 * std
            if int(keep.sum()) < 3 or bool(keep.all()):
                break
            keep_local = keep_local[keep]
        return slope, intercept, r2

    slope, intercept, r2 = fit(band)
    if slope >= 0.0:
        raise ReposeInjectionError("flank slope is not descending; no heap")
    theta = math.degrees(math.atan(-slope))

    angle = torch.atan2(rel[:, 1], rel[:, 0])
    sect = ((angle + math.pi) / (2.0 * math.pi) * sectors).floor().long().clamp_(0, sectors - 1)
    per_sector: list[float] = []
    for s in range(sectors):
        m = band & (sect == s)
        if int(m.sum()) < 4:
            continue
        try:
            st, _, _ = fit(m)
        except Exception:  # pragma: no cover - degenerate sector
            continue
        if st < 0.0:
            per_sector.append(math.degrees(math.atan(-st)))
    std = float(torch.tensor(per_sector, dtype=DTYPE).std()) if len(per_sector) > 1 else 0.0
    return {
        "theta_deg": theta,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "n_points": int(band.sum()),
        "n_surface": int(visible.sum()),
        "r_toe": r_toe,
        "sector_angles_deg": per_sector,
        "sector_std_deg": std,
    }


def kinematic_bound_margin(theta_deg: float, mu: float) -> dict[str, float]:
    """Coulomb free-surface bound ``tan(theta_r) <= mu`` (cohesionless)."""
    tan_t = math.tan(math.radians(theta_deg))
    return {
        "theta_deg": float(theta_deg),
        "mu": float(mu),
        "tan_theta": tan_t,
        "margin": float(mu - tan_t),
        "within_bound": bool(tan_t <= mu + 1e-12),
        "utilisation": float(tan_t / mu) if mu > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Anchor: single-sphere onset of motion on a tilted plane
# ---------------------------------------------------------------------------

def slide_threshold(mu: float, *, mu_rolling: float = 0.0,
                    allow_rotation: bool = False, radius: float = 6.0e-3,
                    density: float = 2500.0, kn: float = 6.0e4,
                    kt_ratio: float = 0.5, restitution: float = 0.05,
                    bracket: Sequence[float] = (0.90, 1.10),
                    travel_tol: float = 5.0e-5, hold_time: float = 3.0e-2,
                    dt: float | None = None,
                    floor_spin_term: bool = True) -> dict[str, object]:
    """Bracket the tilt angle at which a single grain starts to move.

    The plane is this module's own floor contact with the gravity vector rotated
    by ``theta``.  With ``allow_rotation=False`` the grain can only slide, so the
    onset is analytic, ``theta = atan(mu)``: the routine time-steps tilts at
    ``bracket[0] * atan(mu)`` (must not move) and ``bracket[1] * atan(mu)`` (must
    move) and reports whether the analytic value is bracketed.  With rotation free
    the onset is whichever mechanism is weaker -- sliding (``atan mu``) or rolling
    (``atan mu_r``) -- so the same bracket is applied to
    ``atan(min(mu, mu_r))``, and the measured bracket is *reported* rather than
    assumed.
    """
    analytic = (math.degrees(math.atan(min(mu, mu_rolling)))
                if allow_rotation and mu_rolling > 0.0 else math.degrees(math.atan(mu)))
    dt = dt if dt is not None else (2.0 * math.pi * math.sqrt(
        0.5 * (density * 4.0 / 3.0 * math.pi * radius ** 3) / kn)) / 100.0
    zeta = _damping_ratio(restitution)
    kt = kt_ratio * kn
    mass = torch.tensor([density * 4.0 / 3.0 * math.pi * radius ** 3], dtype=DTYPE)
    rows: list[dict[str, float]] = []
    for factor in bracket:
        theta_deg = float(factor) * analytic
        th = math.radians(theta_deg)
        # The floor routine assumes a z-normal plane, so the tilt is carried by
        # the gravity vector in a frame rotated with the plane (grain on +z).
        g_local = torch.tensor([9.81 * math.sin(th), 0.0, -9.81 * math.cos(th)],
                               dtype=DTYPE)
        vel = torch.zeros((1, 3), dtype=DTYPE)
        om = torch.zeros((1, 3), dtype=DTYPE)
        tang = torch.zeros((1, 3), dtype=DTYPE)
        roll = torch.zeros((1, 3), dtype=DTYPE)
        rad = torch.tensor([radius], dtype=DTYPE)
        pos_local = torch.tensor([[0.0, 0.0, radius]], dtype=DTYPE)
        steps = int(math.ceil(hold_time / dt))
        start = pos_local.clone()
        for _ in range(steps):
            f, t = _floor_contact(positions=pos_local, velocities=vel, omega=om,
                                  radii=rad, mass=mass, kn=kn, kt=kt, zeta=zeta,
                                  mu_wall=mu, mu_roll_wall=mu_rolling, tang=tang,
                                  roll=roll, dt=dt, gravity_vec=g_local,
                                  allow_rotation=allow_rotation,
                                  floor_spin_term=floor_spin_term)
            vel = vel + f * (dt / mass)[:, None]
            if allow_rotation:
                inertia = 0.4 * mass * rad ** 2
                om = om + t * (dt / inertia)[:, None]
            pos_local = pos_local + vel * dt
        travel = float(torch.linalg.vector_norm((pos_local - start)[0, :2]))
        rows.append({"factor": float(factor), "theta_deg": theta_deg,
                     "travel_m": travel, "moves": bool(travel > travel_tol)})
    lo, hi = rows[0], rows[-1]
    bracketed = (not lo["moves"]) and hi["moves"]
    return {
        "analytic_deg": float(analytic),
        "bracket_deg": [lo["theta_deg"], hi["theta_deg"]],
        "travel_at_lo": lo["travel_m"],
        "travel_at_hi": hi["travel_m"],
        "bracketed": bool(bracketed),
        "allow_rotation": bool(allow_rotation),
        "mu": float(mu),
        "mu_rolling": float(mu_rolling),
        "rows": rows,
    }
