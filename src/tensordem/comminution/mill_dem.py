"""2-D discrete element ball mill: polydisperse charge, lifter bars, power ledger.

Why this module exists
----------------------
:mod:`tensordem.drum_flow` already provides a rotating drum with a **smooth**
wall and a power ledger, and it is the right tool for drum regime maps.  A ball
mill is different in two ways that matter for the energy surface:

* it carries **lifter bars** -- without lifters a smooth-wall mill relies on
  wall friction alone to lift a 35% ball charge, which under-predicts the torque
  of a real (always liftered) mill by a large factor;
* it carries a **polydisperse** charge (ball size distribution) and its product
  of interest is the *collision energy spectrum* that the population balance
  consumes, not only the torque.

So this is a self-contained kernel (it does not modify ``drum_flow``) with

* particles: frictional disks with radius ``r_i``, mass ``m_i = rho pi r_i^2 t``
  and ``I_i = m_i r_i^2/2``, ``t`` the out-of-plane slice thickness
  (``t = 2 <r>``: the simulation is a one-ball-thick slice of the mill; absolute
  power is scaled to the full length by :attr:`MillConfig.layers`, and
  *mass-specific* quantities are layer-independent);
* normal contact: ``F_n = max(k_n delta - gamma v_n, 0)`` with
  ``gamma = 2 zeta sqrt(k_n m_eff)``;
* tangential contact: Cundall-Strack incremental spring (per pair, history kept
  across neighbour rebuilds) saturated at ``mu F_n``;
* rolling resistance: viscous-regularised Coulomb couple;
* boundaries: the rotating circular shell **plus** ``n_lifter`` radial bars that
  rotate with the shell (each bar is a rectangle in the shell frame, the
  particle-bar contact is the point-to-rectangle distance).

Power ledger
------------
Wall/lifter contacts are split out of the mechanical-energy balance:

.. math:: \\dot K + \\dot U = P_{in} - \\Phi_{diss},\\quad
          P_{in} = \\sum_{shell,lifters} F\\cdot v_{material},

with ``P_in`` accumulated from the contact forces and ``v_material`` of the
moving boundary, and ``Phi_diss = Phi_n + Phi_t + Phi_roll`` accumulated from
the three dissipative laws -- two independent code paths whose agreement is the
closure check.

A second, independent check is the **angular-momentum balance**: in a steady
state the shell torque must equal the gravitational torque of the charge about
the mill axis, ``tau = M_b g x_cm`` (``x_cm`` the horizontal centre-of-mass
offset of the charge).  ``tau`` comes from contact forces, ``x_cm`` from
positions, so the comparison is a genuine momentum ledger and is what
:func:`run_mill` reports as ``moment_closure``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..drum_flow.kernel import cross_z, damping_ratio

__all__ = [
    "MillConfig",
    "MillState",
    "make_mill_state",
    "measure_centrifuging_onset_rpm",
    "run_mill",
]


def _positive(value, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return v


@dataclass(frozen=True)
class MillConfig:
    """Mill geometry, charge specification and contact parameters (SI)."""

    diameter: float = 0.30
    length: float = 0.15
    lifter_count: int = 8
    lifter_height: float = 0.006
    lifter_width: float = 0.006
    ball_radii: tuple = (0.0125, 0.009, 0.006)
    ball_counts: tuple = (18, 27, 40)
    density: float = 7800.0
    kn: float = 2.0e5
    kt_ratio: float = 0.5
    restitution: float = 0.5
    mu: float = 0.5
    mu_wall: float = 0.5
    mu_rolling: float = 0.05
    c_rolling: float = 0.02
    gravity: float = 9.80665
    dt: float = 1.0e-5
    skin: float = 0.002
    rebuild_every: int = 8

    def __post_init__(self) -> None:
        _positive(self.diameter, "diameter")
        _positive(self.length, "length")
        _positive(self.density, "density")
        _positive(self.kn, "kn")
        _positive(self.dt, "dt")
        if len(self.ball_radii) != len(self.ball_counts):
            raise ValueError("ball_radii and ball_counts must have equal length")
        if not self.ball_radii:
            raise ValueError("at least one ball size is required")
        for r in self.ball_radii:
            if _positive(r, "ball radius") >= 0.5 * self.diameter:
                raise ValueError("ball radius must be smaller than the mill radius")
        if int(self.lifter_count) > 0:
            _positive(self.lifter_height, "lifter_height")
            _positive(self.lifter_width, "lifter_width")
        if not 0.0 < float(self.restitution) <= 1.0:
            raise ValueError("restitution must be in (0, 1]")
        for name in ("mu", "mu_wall", "mu_rolling", "c_rolling", "skin", "kt_ratio"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    # ------------------------------------------------------------- geometry
    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def slice_thickness(self) -> float:
        """Out-of-plane thickness of the simulated slice [m] (one ball thick)."""
        return 2.0 * float(np.mean(self.ball_radii))

    @property
    def layers(self) -> float:
        """Number of identical slices that make up the mill length."""
        return self.length / self.slice_thickness

    @property
    def zeta(self) -> float:
        return damping_ratio(self.restitution)

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    def mass(self, radius) -> np.ndarray:
        r = np.asarray(radius, dtype=np.float64)
        return self.density * math.pi * r * r * self.slice_thickness

    def inertia(self, radius) -> np.ndarray:
        r = np.asarray(radius, dtype=np.float64)
        return 0.5 * self.mass(r) * r * r

    def gamma(self, m_eff) -> np.ndarray:
        return 2.0 * self.zeta * np.sqrt(self.kn * np.asarray(m_eff, dtype=np.float64))

    def volumes(self) -> dict:
        """Characteristic volumes [m^3]: mill interior and 3-D charge volume."""
        v_mill = math.pi * 0.25 * self.diameter ** 2 * self.length
        return {"mill": v_mill}


@dataclass
class MillState:
    """Particle state plus the rotating shell speed."""

    pos: np.ndarray
    vel: np.ndarray
    spin: np.ndarray
    radii: np.ndarray
    omega: float
    config: MillConfig
    fill: float

    @property
    def n(self) -> int:
        return int(self.pos.shape[0])

    @property
    def charge_mass_2d(self) -> float:
        return float(np.sum(self.config.mass(self.radii)))

    @property
    def charge_mass(self) -> float:
        """3-D charge mass [kg] (slice mass x number of layers)."""
        return self.charge_mass_2d * self.config.layers


def _place_charge(cfg: MillConfig, fill: float, seed: int = 0):
    """Random sequential addition of the ball charge into the mill cross-section.

    The charge is seeded in the lower half of the shell (where gravity puts it)
    and then settled by the caller, so the exact seed layout does not bias the
    steady state.
    """
    if not 0.0 < fill < 0.6:
        raise ValueError("fill must be in (0, 0.6)")
    rng = np.random.default_rng(seed)
    R = cfg.radius
    target_area = fill * math.pi * R * R
    areas = np.array([math.pi * r * r for r in cfg.ball_radii])
    weights = np.asarray(cfg.ball_counts, dtype=float) * areas
    counts = np.zeros(len(cfg.ball_radii), dtype=int)
    counts[:] = np.maximum(1, np.round(np.asarray(cfg.ball_counts, float)
                                       * (target_area / weights.sum())).astype(int))
    radii = np.repeat(np.asarray(cfg.ball_radii, dtype=float), counts)
    rng.shuffle(radii)
    pos = np.zeros((radii.size, 2))
    placed = 0
    for r in radii:
        for _ in range(20000):
            # seed in the lower 2/3 of the shell; allow the whole shell as a
            # fallback so the target area is always reachable
            ymax = R - r if placed > 0.75 * radii.size else 0.35 * R
            x = rng.uniform(-(R - r), R - r)
            ylo = -math.sqrt(max((R - r) ** 2 - x * x, 0.0))
            yhi = min(ymax, math.sqrt(max((R - r) ** 2 - x * x, 0.0)))
            if yhi <= ylo:
                continue
            y = rng.uniform(ylo, yhi)
            if placed:
                d = np.hypot(pos[:placed, 0] - x, pos[:placed, 1] - y)
                if np.any(d < radii[:placed] + r + 1e-9):
                    continue
            pos[placed] = (x, y)
            placed += 1
            break
        else:  # pragma: no cover - defensive
            raise RuntimeError("could not place the ball charge; reduce fill")
    return pos[:placed], radii[:placed]


def make_mill_state(cfg: MillConfig, fill: float, rpm: float, seed: int = 0) -> MillState:
    """Initialise a settled-geometry mill charge at ``rpm`` (charge not yet settled)."""
    omega = 2.0 * math.pi * float(rpm) / 60.0
    pos, radii = _place_charge(cfg, fill, seed=seed)
    n = pos.shape[0]
    return MillState(pos=pos, vel=np.zeros((n, 2)), spin=np.zeros(n), radii=radii,
                     omega=omega, config=cfg, fill=float(fill))


def _lifter_geometry(cfg: MillConfig):
    """Lifter base angles and the local-frame rectangle of one bar."""
    if int(cfg.lifter_count) <= 0:
        return np.zeros(0), 0.0, 0.0
    phi = 2.0 * math.pi * np.arange(int(cfg.lifter_count)) / int(cfg.lifter_count)
    u_lo = cfg.radius - 0.25 * cfg.lifter_height
    u_hi = cfg.radius + cfg.lifter_height
    return phi, u_lo, u_hi


def _stored_energy(cfg: MillConfig, pos, radii, pair_i, pair_j, xi_pp, xi_w,
                   delta_w, pen_b) -> float:
    """Stored normal + tangential spring energy of all current contacts [J]."""
    kn = cfg.kn
    kt = cfg.kt
    u = 0.5 * kn * float(np.sum(pen_b ** 2)) if pen_b.size else 0.0
    if pair_i.size:
        dvec = pos[pair_i] - pos[pair_j]
        dist = np.sqrt(dvec[:, 0] ** 2 + dvec[:, 1] ** 2)
        delta = radii[pair_i] + radii[pair_j] - dist
        act = delta > 0.0
        if act.any():
            u += 0.5 * kn * float(np.sum(delta[act] ** 2))
            xi = xi_pp[act]
            u += 0.5 * kt * float(np.sum(xi[:, 0] ** 2 + xi[:, 1] ** 2))
    wall_act = delta_w > 0.0
    if wall_act.any():
        u += 0.5 * kn * float(np.sum(delta_w[wall_act] ** 2))
        xw = xi_w[wall_act]
        u += 0.5 * kt * float(np.sum(xw[:, 0] ** 2 + xw[:, 1] ** 2))
    return float(u)


def run_mill(state: MillState, revolutions: float = 3.0, measure_last: float = 1.5,
             sample_every: int = 200, settle_time: float = 0.4,
             record_collisions: bool = True, max_collisions: int = 400000,
             keep_snapshot: bool = True) -> dict:
    """Integrate the mill and return observables, ledgers and the collision spectrum.

    Parameters
    ----------
    revolutions : revolutions integrated after the settling phase.
    measure_last : number of those revolutions used for time averages and the
        power/momentum ledgers (the last ``measure_last`` revolutions).
    sample_every : sampling stride for the per-particle diagnostics.
    settle_time : gravity settling time with a static shell before rotation.
    record_collisions : record the impact spectrum (approach speed and the two
        radii of every fresh particle-particle impact) in the measurement window.

    Ledgers returned in ``result["ledger"]``
    ----------------------------------------
    ``p_in_w``          mechanical power delivered through the shell + lifters
    ``phi_n_w``         normal dashpot dissipation (real)
    ``phi_t_w``         tangential Coulomb slip dissipation (real)
    ``phi_roll_w``      rolling-resistance dissipation (real)
    ``phi_release_w``   tangential spring energy discarded when a contact is
                        lost (a *model artefact*, not a physical sink; reported
                        separately so the closure is not flattered by it)
    ``closure_rel``     ``|P_in - (phi_n + phi_t + phi_roll)| / P_in``
    ``full_closure_rel``  the complete bookkeeping identity including the
                        release artefact and the change of mechanical + elastic
                        energy over the window
    ``moment_closure_rel``  ``|tau_shell - M g x_cm| / (M g x_cm)`` -- the
                        steady-state angular-momentum balance, ``tau`` from
                        contact forces and ``x_cm`` from positions.
    """
    cfg = state.config
    pos = state.pos.copy()
    vel = state.vel.copy()
    spin = state.spin.copy()
    radii = np.asarray(state.radii, dtype=np.float64)
    n = pos.shape[0]
    mass = cfg.mass(radii)
    inertia = cfg.inertia(radii)
    R = cfg.radius
    dt = cfg.dt
    omega_abs = abs(state.omega)
    if omega_abs < 1e-12:
        raise ValueError("run_mill needs a non-zero shell speed")

    # hoisted scalars: the inner loop is the hot spot of the whole package
    kn = cfg.kn
    kt = cfg.kt
    kt_inv = 1.0 / kt
    gfac = 2.0 * cfg.zeta * math.sqrt(kn)
    mu = cfg.mu
    mu_wall = cfg.mu_wall
    mu_roll = cfg.mu_rolling
    c_roll_half = 0.5 * cfg.c_rolling
    gravity = cfg.gravity
    inv_mass = 1.0 / mass
    inv_inertia = 1.0 / inertia
    mass_total = float(np.sum(mass))

    rev_time = 2.0 * math.pi / omega_abs
    settle_steps = int(round(settle_time / dt))
    spin_steps = int(round(revolutions * rev_time / dt))
    total_steps = settle_steps + spin_steps
    measure_from = max(0, total_steps - int(round(measure_last * rev_time / dt)))

    phi_lift, u_lo, u_hi = _lifter_geometry(cfg)
    half_w = 0.5 * cfg.lifter_width
    e_r = np.stack([np.cos(phi_lift), np.sin(phi_lift)], axis=1)
    e_t = np.stack([-np.sin(phi_lift), np.cos(phi_lift)], axis=1)

    r_max = float(radii.max())
    pair_i = np.empty(0, dtype=np.int64)
    pair_j = np.empty(0, dtype=np.int64)
    xi_pp = np.zeros((0, 2))
    keys_pp = np.empty(0, dtype=np.int64)
    imp_pp = np.zeros(0, dtype=bool)
    xi_w = np.zeros((n, 2))
    nn_contacts = np.zeros(n)
    delta_w = np.zeros(n)
    pen_b = np.zeros(0)

    win = {"p_in": 0.0, "phi_n": 0.0, "phi_t": 0.0, "phi_r": 0.0, "phi_rel": 0.0,
           "t": 0.0, "x_cm": 0.0, "e_start": None, "e_end": None}
    samples = []
    impacts = []
    n_recorded = 0
    max_delta = 0.0

    step = 0
    while step < total_steps:
        omega_now = state.omega if step >= settle_steps else 0.0
        rotating = omega_now != 0.0

        # ------------------------------------------------ neighbour list
        if step % cfg.rebuild_every == 0:
            tree = cKDTree(pos)
            pairs = np.asarray(tree.query_pairs(2.0 * r_max + cfg.skin,
                                                output_type="ndarray"),
                               dtype=np.int64).reshape(-1, 2)
            if pairs.size:
                pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
                keys_new = pairs[:, 0] * n + pairs[:, 1]
                xi_new = np.zeros((keys_new.size, 2))
                imp_new = np.zeros(keys_new.size, dtype=bool)
                if keys_pp.size:
                    idx = np.clip(np.searchsorted(keys_pp, keys_new), 0, keys_pp.size - 1)
                    hit = keys_pp[idx] == keys_new
                    if hit.any():
                        xi_new[hit] = xi_pp[idx[hit]]
                        imp_new[hit] = imp_pp[idx[hit]]
                xi_pp, keys_pp, imp_pp = xi_new, keys_new, imp_new
                pair_i, pair_j = pairs[:, 0].copy(), pairs[:, 1].copy()
            else:
                pair_i = np.empty(0, dtype=np.int64)
                pair_j = np.empty(0, dtype=np.int64)
                xi_pp = np.zeros((0, 2))
                keys_pp = np.empty(0, dtype=np.int64)
                imp_pp = np.zeros(0, dtype=bool)

        force = np.zeros((n, 2))
        torque = np.zeros(n)
        phi_n_step = phi_t_step = phi_r_step = phi_rel_step = 0.0
        p_in_step = 0.0
        nn_contacts[:] = 0.0

        # ------------------------------------------------ particle-particle
        if pair_i.size:
            ia_all, ja_all = pair_i, pair_j
            dx = pos[ia_all, 0] - pos[ja_all, 0]
            dy = pos[ia_all, 1] - pos[ja_all, 1]
            dist = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-12)
            nx, ny = dx / dist, dy / dist
            delta = radii[ia_all] + radii[ja_all] - dist
            active = delta > 0.0
            dead = ~active
            if dead.any():
                xd = xi_pp[dead]
                phi_rel_step += 0.5 * kt * float(np.sum(xd[:, 0] ** 2 + xd[:, 1] ** 2)) / dt
                xi_pp[dead] = 0.0
                imp_pp[dead] = False
            if active.any():
                ia, ja = ia_all[active], ja_all[active]
                na_x, na_y = nx[active], ny[active]
                delta_a = delta[active]
                nn_contacts += np.bincount(ia, minlength=n)
                nn_contacts += np.bincount(ja, minlength=n)
                ri = radii[ia]
                rj = radii[ja]
                spin_sum = ri * spin[ia] + rj * spin[ja]
                # surface relative velocity (drum_flow convention)
                vrx = vel[ia, 0] - vel[ja, 0] + spin_sum * na_y
                vry = vel[ia, 1] - vel[ja, 1] - spin_sum * na_x
                vn = vrx * na_x + vry * na_y
                mi, mj = mass[ia], mass[ja]
                m_eff = mi * mj / (mi + mj)
                gamma_ij = gfac * np.sqrt(m_eff)
                fn = kn * delta_a - gamma_ij * vn
                np.maximum(fn, 0.0, out=fn)
                utx = vrx - vn * na_x
                uty = vry - vn * na_y
                ut_mag = np.sqrt(utx * utx + uty * uty)
                idx = np.flatnonzero(active)
                xi = xi_pp[idx]
                xi[:, 0] += utx * dt
                xi[:, 1] += uty * dt
                xi_proj = xi[:, 0] * na_x + xi[:, 1] * na_y
                xi[:, 0] -= xi_proj * na_x
                xi[:, 1] -= xi_proj * na_y
                ftx = -kt * xi[:, 0]
                fty = -kt * xi[:, 1]
                ft_mag = np.sqrt(ftx * ftx + fty * fty)
                ft_cap = mu * fn
                over = ft_mag > ft_cap
                if over.any():
                    ratio = np.where(over, ft_cap / np.maximum(ft_mag, 1e-300), 1.0)
                    ftx *= ratio
                    fty *= ratio
                    xi[:, 0] = np.where(over, -ftx * kt_inv, xi[:, 0])
                    xi[:, 1] = np.where(over, -fty * kt_inv, xi[:, 1])
                fx = fn * na_x + ftx
                fy = fn * na_y + fty
                force[:, 0] += np.bincount(ia, weights=fx, minlength=n)
                force[:, 0] -= np.bincount(ja, weights=fx, minlength=n)
                force[:, 1] += np.bincount(ia, weights=fy, minlength=n)
                force[:, 1] -= np.bincount(ja, weights=fy, minlength=n)
                arm_x = -ri * na_x
                arm_y = -ri * na_y
                tq = arm_x * fy - arm_y * fx
                torque += np.bincount(ia, weights=tq, minlength=n)
                torque += np.bincount(ja, weights=tq, minlength=n)
                dw = spin[ia] - spin[ja]
                rmin = np.minimum(ri, rj)
                mr_cap = mu_roll * rmin * fn
                mr = np.clip(-c_roll_half * rmin * fn * dw, -mr_cap, mr_cap)
                torque += np.bincount(ia, weights=mr, minlength=n)
                torque -= np.bincount(ja, weights=mr, minlength=n)
                phi_n_step += float(np.sum(gamma_ij * vn * vn * (fn > 0.0)))
                if over.any():
                    phi_t_step += float(np.sum(ft_cap[over] * ut_mag[over]))
                phi_r_step += float(np.sum(np.abs(mr * dw)))
                if float(delta_a.max()) > max_delta:
                    max_delta = float(delta_a.max())
                if record_collisions and step >= measure_from and n_recorded < max_collisions:
                    fresh = (~imp_pp[idx]) & (vn < 0.0)
                    if fresh.any():
                        sel = np.flatnonzero(fresh)[: max_collisions - n_recorded]
                        impacts.append(np.stack([-vn[sel], ri[sel], rj[sel]], axis=1))
                        n_recorded += int(sel.size)
                    imp_pp[idx] = imp_pp[idx] | (vn < 0.0)

        # ------------------------------------------------ shell (circle)
        dist_c = np.maximum(np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2), 1e-12)
        no_x, no_y = pos[:, 0] / dist_c, pos[:, 1] / dist_c
        delta_w[:] = radii + dist_c - R
        left = delta_w <= 0.0
        if left.any():
            xw = xi_w[left]
            phi_rel_step += 0.5 * kt * float(np.sum(xw[:, 0] ** 2 + xw[:, 1] ** 2)) / dt
            xi_w[left] = 0.0
        wact = np.flatnonzero(delta_w > 0.0)
        if wact.size:
            nx_w, ny_w = no_x[wact], no_y[wact]
            r_c = radii[wact]
            nn_contacts[wact] += 1.0
            v_wall_x = -omega_now * R * ny_w
            v_wall_y = omega_now * R * nx_w
            sw = spin[wact]
            vpx = vel[wact, 0] + r_c * sw * (-ny_w)
            vpy = vel[wact, 1] + r_c * sw * (nx_w)
            vrx = vpx - v_wall_x
            vry = vpy - v_wall_y
            vn = -(vrx * nx_w + vry * ny_w)          # inward normal
            gamma_w = gfac * np.sqrt(mass[wact])
            fn = kn * delta_w[wact] - gamma_w * vn
            np.maximum(fn, 0.0, out=fn)
            utx = vrx + vn * nx_w
            uty = vry + vn * ny_w
            ut_mag = np.sqrt(utx * utx + uty * uty)
            xi = xi_w[wact]
            xi[:, 0] += utx * dt
            xi[:, 1] += uty * dt
            xi_proj = -(xi[:, 0] * nx_w + xi[:, 1] * ny_w)
            xi[:, 0] += xi_proj * nx_w
            xi[:, 1] += xi_proj * ny_w
            ftx = -kt * xi[:, 0]
            fty = -kt * xi[:, 1]
            ft_mag = np.sqrt(ftx * ftx + fty * fty)
            ft_cap = mu_wall * fn
            over = ft_mag > ft_cap
            if over.any():
                ratio = np.where(over, ft_cap / np.maximum(ft_mag, 1e-300), 1.0)
                ftx *= ratio
                fty *= ratio
                xi[:, 0] = np.where(over, -ftx * kt_inv, xi[:, 0])
                xi[:, 1] = np.where(over, -fty * kt_inv, xi[:, 1])
            fx = -fn * nx_w + ftx
            fy = -fn * ny_w + fty
            force[wact, 0] += fx
            force[wact, 1] += fy
            arm_x = r_c * nx_w
            arm_y = r_c * ny_w
            tq = arm_x * fy - arm_y * fx
            mr_cap = mu_roll * r_c * fn
            dw_w = sw - omega_now
            mr = np.clip(-c_roll_half * r_c * fn * dw_w, -mr_cap, mr_cap)
            torque[wact] += tq + mr
            p_in_step += float(np.sum(fx * v_wall_x + fy * v_wall_y))
            phi_n_step += float(np.sum(gamma_w * vn * vn * (fn > 0.0)))
            if over.any():
                phi_t_step += float(np.sum(ft_cap[over] * ut_mag[over]))
            phi_r_step += float(np.sum(np.abs(mr * dw_w)))

        # ------------------------------------------------ lifter bars
        pen_b = np.zeros(0)
        if phi_lift.size and rotating:
            u = pos @ e_r.T
            v = pos @ e_t.T
            u_c = np.clip(u, u_lo, u_hi)
            v_c = np.clip(v, -half_w, half_w)
            du = u - u_c
            dv = v - v_c
            dist_b = np.sqrt(du * du + dv * dv)
            hit = (dist_b < radii[:, None]) & (dist_b > 1e-12)
            if hit.any():
                pi_idx, li = np.nonzero(hit)
                r_p = radii[pi_idx]
                dloc = np.stack([du[pi_idx, li], dv[pi_idx, li]], axis=1) \
                    / dist_b[pi_idx, li][:, None]
                nb_x = dloc[:, 0] * e_r[li, 0] + dloc[:, 1] * e_t[li, 0]
                nb_y = dloc[:, 0] * e_r[li, 1] + dloc[:, 1] * e_t[li, 1]
                pen = r_p - dist_b[pi_idx, li]
                pen_b = pen
                cpt_x = u_c[pi_idx, li] * e_r[li, 0] + v_c[pi_idx, li] * e_t[li, 0]
                cpt_y = u_c[pi_idx, li] * e_r[li, 1] + v_c[pi_idx, li] * e_t[li, 1]
                v_wall_x = -omega_now * cpt_y
                v_wall_y = omega_now * cpt_x
                nn_contacts[pi_idx] += 1.0
                sp = spin[pi_idx]
                vpx = vel[pi_idx, 0] + r_p * sp * nb_y
                vpy = vel[pi_idx, 1] - r_p * sp * nb_x
                vrx = vpx - v_wall_x
                vry = vpy - v_wall_y
                vn = vrx * nb_x + vry * nb_y            # approaching -> negative
                gamma_b = gfac * np.sqrt(mass[pi_idx])
                fn = kn * pen - gamma_b * vn
                np.maximum(fn, 0.0, out=fn)
                utx = vrx - vn * nb_x
                uty = vry - vn * nb_y
                ut_mag = np.sqrt(utx * utx + uty * uty)
                xi = xi_w[pi_idx]
                xi[:, 0] += utx * dt
                xi[:, 1] += uty * dt
                xi_proj = xi[:, 0] * nb_x + xi[:, 1] * nb_y
                xi[:, 0] -= xi_proj * nb_x
                xi[:, 1] -= xi_proj * nb_y
                ftx = -kt * xi[:, 0]
                fty = -kt * xi[:, 1]
                ft_mag = np.sqrt(ftx * ftx + fty * fty)
                ft_cap = mu_wall * fn
                over = ft_mag > ft_cap
                if over.any():
                    ratio = np.where(over, ft_cap / np.maximum(ft_mag, 1e-300), 1.0)
                    ftx *= ratio
                    fty *= ratio
                    xi[:, 0] = np.where(over, -ftx * kt_inv, xi[:, 0])
                    xi[:, 1] = np.where(over, -fty * kt_inv, xi[:, 1])
                fx = fn * nb_x + ftx
                fy = fn * nb_y + fty
                force[:, 0] += np.bincount(pi_idx, weights=fx, minlength=n)
                force[:, 1] += np.bincount(pi_idx, weights=fy, minlength=n)
                arm_x = -r_p * nb_x
                arm_y = -r_p * nb_y
                tq = arm_x * fy - arm_y * fx
                mr_cap = mu_roll * r_p * fn
                dw_b = sp - omega_now
                mr = np.clip(-c_roll_half * r_p * fn * dw_b, -mr_cap, mr_cap)
                torque += np.bincount(pi_idx, weights=tq + mr, minlength=n)
                p_in_step += float(np.sum(fx * v_wall_x + fy * v_wall_y))
                phi_n_step += float(np.sum(gamma_b * vn * vn * (fn > 0.0)))
                if over.any():
                    phi_t_step += float(np.sum(ft_cap[over] * ut_mag[over]))
                phi_r_step += float(np.sum(np.abs(mr * dw_b)))

        # ------------------------------------------------ integrate
        vel[:, 0] += force[:, 0] * inv_mass * dt
        vel[:, 1] += (force[:, 1] * inv_mass - gravity) * dt
        pos += vel * dt
        spin += torque * inv_inertia * dt

        if step == measure_from:
            win["e_start"] = (0.5 * float(np.sum(mass * (vel[:, 0] ** 2 + vel[:, 1] ** 2)))
                              + 0.5 * float(np.sum(inertia * spin * spin))
                              + float(np.sum(mass * gravity * pos[:, 1]))
                              + _stored_energy(cfg, pos, radii, pair_i, pair_j, xi_pp,
                                               xi_w, delta_w, pen_b))
        if step >= measure_from:
            win["p_in"] += p_in_step * dt
            win["phi_n"] += phi_n_step * dt
            win["phi_t"] += phi_t_step * dt
            win["phi_r"] += phi_r_step * dt
            win["phi_rel"] += phi_rel_step * dt
            win["t"] += dt
            win["x_cm"] += float(np.sum(mass * pos[:, 0])) * dt
            if step % sample_every == 0:
                r_norm = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2) / R
                loaded = nn_contacts > 0.0
                rad2 = np.maximum(pos[:, 0] ** 2 + pos[:, 1] ** 2, 1e-12)
                ang = (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]) / rad2
                bed_sel = (r_norm < 0.85) & loaded
                sq = np.sqrt(rad2)
                et_x, et_y = -pos[:, 1] / sq, pos[:, 0] / sq
                vt = vel[:, 0] * et_x + vel[:, 1] * et_y
                wall_layer = (r_norm > 0.85) & loaded
                surf = (r_norm > 0.5) & loaded & (pos[:, 1] > 0.0)
                samples.append({
                    "t": (step - settle_steps) * dt,
                    "omega_bed": float(np.median(ang[bed_sel])) if bed_sel.any() else 0.0,
                    "airborne_fraction": float(np.mean(nn_contacts == 0.0)),
                    "r_p05_over_R": float(np.percentile(r_norm, 5)),
                    "wall_velocity_ratio": (float(np.mean(vt[wall_layer]))
                                            / (omega_abs * R)) if wall_layer.any() else 0.0,
                    "surface_speed_ratio": (float(np.mean(vt[surf]))
                                            / (omega_abs * R)) if surf.any() else 0.0,
                    "x_com_over_R": float(np.sum(mass * pos[:, 0]) / mass_total / R),
                    "y_com_over_R": float(np.sum(mass * pos[:, 1]) / mass_total / R),
                    "ke": 0.5 * float(np.sum(mass * (vel[:, 0] ** 2 + vel[:, 1] ** 2))),
                    "max_v": float(np.max(np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2))),
                    "n_contacts_mean": float(np.mean(nn_contacts)),
                })
        step += 1

    win["e_end"] = (0.5 * float(np.sum(mass * (vel[:, 0] ** 2 + vel[:, 1] ** 2)))
                    + 0.5 * float(np.sum(inertia * spin * spin))
                    + float(np.sum(mass * gravity * pos[:, 1]))
                    + _stored_energy(cfg, pos, radii, pair_i, pair_j, xi_pp, xi_w,
                                     delta_w, pen_b))

    t_win = max(win["t"], dt)
    p_in = win["p_in"] / t_win
    phi_n = win["phi_n"] / t_win
    phi_t = win["phi_t"] / t_win
    phi_r = win["phi_r"] / t_win
    phi_rel = win["phi_rel"] / t_win
    x_com = (win["x_cm"] / t_win) / mass_total
    tau_on_charge = p_in / state.omega
    tau_gravity = mass_total * gravity * x_com
    d_e = (win["e_end"] - win["e_start"]) / t_win if win["e_start"] is not None else float("nan")
    residual = p_in - (phi_n + phi_t + phi_r + phi_rel + d_e)
    ledger = {
        "p_in_w_2d": p_in,
        "p_in_w": p_in * cfg.layers,
        "tau_on_charge_nm_2d": tau_on_charge,
        "tau_on_charge_nm": tau_on_charge * cfg.layers,
        "tau_gravity_nm_2d": tau_gravity,
        "tau_gravity_nm": tau_gravity * cfg.layers,
        "phi_n_w_2d": phi_n,
        "phi_t_w_2d": phi_t,
        "phi_roll_w_2d": phi_r,
        "phi_release_w_2d": phi_rel,
        "phi_diss_w_2d": phi_n + phi_t + phi_r,
        "dE_mech_w_2d": d_e,
        "residual_w_2d": residual,
        "closure_rel": abs(p_in - (phi_n + phi_t + phi_r)) / p_in if p_in > 0 else float("nan"),
        "full_closure_rel": abs(residual) / p_in if p_in > 0 else float("nan"),
        "release_fraction": phi_rel / p_in if p_in > 0 else float("nan"),
        "moment_closure_rel": (abs(tau_on_charge - tau_gravity) / abs(tau_gravity)
                               if tau_gravity != 0 else float("nan")),
        "window_s": t_win,
        "x_com_m": x_com,
        "charge_mass_kg_2d": mass_total,
        "charge_mass_kg": mass_total * cfg.layers,
        "layers": cfg.layers,
    }
    mean = {}
    if samples:
        for key in samples[0]:
            vals = np.array([s[key] for s in samples], dtype=float)
            mean[key] = float(np.mean(vals))
    spectrum = None
    if impacts:
        arr = np.concatenate(impacts, axis=0)
        spectrum = {
            "approach_speed": arr[:, 0],
            "radius_a": arr[:, 1],
            "radius_b": arr[:, 2],
            "window_s": t_win,
            "n_events": int(arr.shape[0]),
        }
    return {
        "n_particles": n,
        "omega": state.omega,
        "rpm": state.omega * 60.0 / (2.0 * math.pi),
        "fill_realised": float(np.sum(math.pi * radii ** 2) / (math.pi * R * R)),
        "steps": total_steps,
        "samples": samples,
        "mean": mean,
        "ledger": ledger,
        "collisions": spectrum,
        "max_overlap_rel": max_delta / float(radii.max()) if n else 0.0,
        "snapshot": ({"pos": pos, "vel": vel, "spin": spin, "radii": radii,
                      "nn_contacts": nn_contacts} if keep_snapshot else None),
    }


def _tracer_revolution(omega: float, R: float, rt: float, m: float, inertia: float,
                       gamma: float, kn: float, mu_wall: float, g: float,
                       dt: float, lock_spin: bool = True) -> dict:
    """One revolution of a single ball on the rotating shell (reduced kernel).

    Same contact law as :func:`run_mill` (normal spring-dashpot + Coulomb
    friction) but specialised to one particle, so the critical-speed detector is
    cheap enough to bisect.  Returns the shell-contact fraction, the net angular
    advance in turns and the extreme radii, which is what the detector needs.
    """
    px, py = 0.0, -(R - rt)
    vx, vy = -omega * (R - rt), 0.0
    w = omega
    n_steps = max(int(round(2.0 * math.pi / omega / dt)), 1)
    contact = 0
    theta_prev = math.atan2(py, px)
    advance = 0.0
    r_min = math.inf
    r_max = 0.0
    for _ in range(n_steps):
        dist_c = math.sqrt(px * px + py * py) or 1e-12
        if dist_c < r_min:
            r_min = dist_c
        if dist_c > r_max:
            r_max = dist_c
        no_x, no_y = px / dist_c, py / dist_c
        delta = rt + dist_c - R
        fx = fy = 0.0
        if delta > 0.0:
            contact += 1
            cx = px + rt * no_x
            cy = py + rt * no_y
            vwx, vwy = -omega * cy, omega * cx
            vpx = vx + (0.0 if lock_spin else rt * w * (-no_y))
            vpy = vy + (0.0 if lock_spin else rt * w * (no_x))
            vrx, vry = vpx - vwx, vpy - vwy
            vn = -(vrx * no_x + vry * no_y)
            fn = max(kn * delta - gamma * vn, 0.0)
            ftx = -(vrx + vn * no_x)
            fty = -(vry + vn * no_y)
            ft_mag = math.sqrt(ftx * ftx + fty * fty)
            cap = mu_wall * fn
            if ft_mag > cap and ft_mag > 0.0:
                s = cap / ft_mag
                ftx *= s
                fty *= s
            fx = -fn * no_x + ftx
            fy = -fn * no_y + fty
            if not lock_spin:
                w += (rt * no_x * fy - rt * no_y * fx) / inertia * dt
        fy -= m * g
        vx += fx / m * dt
        vy += fy / m * dt
        px += vx * dt
        py += vy * dt
        theta_now = math.atan2(py, px)
        advance += (theta_now - theta_prev + math.pi) % (2.0 * math.pi) - math.pi
        theta_prev = theta_now
    return {
        "contact_fraction": contact / n_steps,
        "turns": advance / (2.0 * math.pi),
        "r_min_over_R": r_min / R,
        "r_max_over_R": r_max / R,
        "steps": n_steps,
    }


def _tracer_setup(diameter: float, tracer_radius: float, kn: float, dt: float | None):
    d = _positive(diameter, "diameter")
    rt = _positive(tracer_radius, "tracer_radius")
    if 2.0 * rt >= 0.5 * d:
        raise ValueError("tracer radius too large for the mill")
    cfg = MillConfig(diameter=d, length=max(4.0 * rt, 1e-3), lifter_count=0,
                     ball_radii=(rt,), ball_counts=(1,), kn=kn)
    m = float(cfg.mass(np.array([rt]))[0])
    inertia = float(cfg.inertia(np.array([rt]))[0])
    gamma = float(cfg.gamma(np.array([m]))[0])
    if dt is None:
        dt = math.pi * math.sqrt(m / kn) / 20.0
    return cfg, rt, m, inertia, gamma, float(dt)


def tracer_response(diameter: float, tracer_radius: float, rpm, *,
                    mu_wall: float = 5.0, kn: float = 2.0e5,
                    dt: float | None = None, lock_spin: bool = True) -> list:
    """Detector response curve: contact fraction and net advance vs speed.

    A single frictional ball on the shell is carried up by wall friction.  Below
    the critical speed the normal force vanishes before the apex and the ball
    leaves the shell (contact fraction < 1, and it never completes the turn);
    above it, the ball is pinned for the whole revolution (contact fraction ~ 1,
    net advance 1 turn).  This function exists so the *detector* itself is
    auditable rather than a black box: it returns one dict per requested speed.
    """
    cfg, rt, m, inertia, gamma, dt_eff = _tracer_setup(diameter, tracer_radius, kn, dt)
    from .critical_speed import critical_speed_rpm, critical_speed_rpm_charge_centre

    nc = critical_speed_rpm(diameter)
    out = []
    for speed in np.atleast_1d(np.asarray(rpm, dtype=float)):
        omega = 2.0 * math.pi * float(speed) / 60.0
        row = _tracer_revolution(omega, cfg.radius, rt, m, inertia, gamma, kn,
                                 mu_wall, cfg.gravity, dt_eff, lock_spin)
        row.update(rpm=float(speed), fraction_of_critical=float(speed) / nc,
                   mu_wall=mu_wall, dt=dt_eff,
                   critical_speed_rpm_charge_centre=critical_speed_rpm_charge_centre(
                       diameter, 2.0 * rt))
        out.append(row)
    return out


def measure_centrifuging_onset_rpm(diameter: float, tracer_radius: float,
                                   dt: float | None = None, kn: float = 2.0e5,
                                   mu_wall: float = 5.0,
                                   tolerance: float = 2.0e-3,
                                   max_iterations: int = 12,
                                   low_fraction: float = 0.75,
                                   high_fraction: float = 1.25,
                                   lock_spin: bool = True) -> dict:
    """Detect the centrifuging onset of a tracer ball in a smooth mill by DEM.

    Definition used (the textbook one): the speed above which the ball is
    *pinned* to the shell for a whole revolution, i.e. the speed at which the
    normal force at the apex stops vanishing.  A tracer is released at 6 o'clock
    co-rotating with the shell and integrated for one revolution with the same
    soft-sphere contact law as :func:`run_mill`; the run counts as *stuck* when
    the ball stays in shell contact for >= 99% of the steps **and** completes at
    least 0.9 of a turn.

    Two caveats that make this a real test rather than a tautology:

    * the wall friction must be high enough to carry the ball (``mu_wall = 5``,
      the idealisation of the lifter pocket that carries a real charge).  With a
      realistic ``mu_wall = 0.5`` a ball slips back down the rising side even
      *above* the critical speed (the equatorial sticking condition is
      ``omega^2 r >= g/mu``), so contact fraction alone cannot detect anything;
    * the detected speed is compared with both the textbook ``42.3/sqrt(D)`` and
      the ball-centre reference ``60/(2 pi) sqrt(2g/(D-d))``, because a DEM
      contact is defined centre-to-shell and the two differ by
      ``sqrt(D/(D-d))``.
    """
    from .critical_speed import critical_speed_rpm, critical_speed_rpm_charge_centre

    cfg, rt, m, inertia, gamma, dt_eff = _tracer_setup(diameter, tracer_radius, kn, dt)
    R = cfg.radius
    g = cfg.gravity

    def stuck(rpm: float) -> bool:
        row = _tracer_revolution(2.0 * math.pi * rpm / 60.0, R, rt, m, inertia,
                                 gamma, kn, mu_wall, g, dt_eff, lock_spin)
        return row["contact_fraction"] >= 0.99 and row["turns"] >= 0.9

    nc = critical_speed_rpm(diameter)
    nc_cc = critical_speed_rpm_charge_centre(diameter, 2.0 * rt)
    lo, hi = low_fraction * nc, high_fraction * nc
    if stuck(lo):
        raise RuntimeError("tracer already stuck below 0.75 Nc: contact parameters too soft")
    if not stuck(hi):
        raise RuntimeError("tracer never sticks above 1.25 Nc: contact parameters too stiff")
    n_iter = 0
    for _ in range(int(max_iterations)):
        mid = 0.5 * (lo + hi)
        if stuck(mid):
            hi = mid
        else:
            lo = mid
        n_iter += 1
        if abs(hi - lo) / hi <= tolerance:
            break
    rpm_onset = 0.5 * (lo + hi)
    return {
        "rpm": rpm_onset,
        "bracket": (lo, hi),
        "critical_speed_rpm_nominal": nc,
        "critical_speed_rpm_charge_centre": nc_cc,
        "error_vs_nominal": rpm_onset / nc - 1.0,
        "error_vs_charge_centre": rpm_onset / nc_cc - 1.0,
        "tracer_radius": rt,
        "diameter": diameter,
        "dt": dt_eff,
        "kn": kn,
        "mu_wall": mu_wall,
        "iterations": n_iter,
    }