"""2-D soft-disk DEM pile bench: dry/wet angle of repose with pendular bridges.

The hold-scale chain of :mod:`tensordem.cargo_liquefaction_effstress` is a
quasi-static continuum relaxation; the *microscopic* half of the campaign (does
the pendular-bridge law actually change a granular free surface?) needs real
particles.  This module is the smallest honest particle model that can answer
it:

* discs (uniform radius ``R``) with a linear spring-dashpot normal law, a
  Cundall-Strack tangential spring with a Coulomb limit ``mu F_n``, and a
  rolling-resistance torque with limit ``mu_r R F_n``;
* optional pendular liquid bridges between every pair inside the rupture
  distance, with the Rouault force of
  :func:`tensordem.cargo_liquefaction_effstress.rouault_pendular_force`
  multiplied by the Terzaghi effective-stress factor ``lambda = 1 - r_u``;
* a dam-break (column-collapse) repose test: a column on the left of a box is
  released and the free-surface slope of the settled wedge is fitted.

Two anchors make the model checkable on its own:

``sliding_onset_angle``
    A single disc on an inclined plane must start to slide exactly at
    ``atan(mu)`` (the irrotational slider anchor, as in
    :mod:`tensordem.repose_calibration`).
``frictionless_pile_is_flat``
    With ``mu = mu_r = 0`` the collapse must leave a flat surface.

The pile is *dimensionless in the cohesion number*
``Co = F_cap / (rho g pi R^2)``: the campaign runs ``theta_r(phi_0, Co)`` and
reads the fine-cargo value off that curve instead of pretending that a
100-micron-particle pile carries the capillary force of a 10-micron cargo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tensordem.cargo_liquefaction_effstress import (
    GRAVITY,
    rouault_pendular_force,
)

try:  # pragma: no cover
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


@dataclass
class DiskPile:
    """2-D disc assembly in a box with an optional released gate."""

    radius: float = 1.0e-4
    density: float = 2650.0
    n_particles: int = 1000
    box_width: float = 1.3e-2
    box_height: float = 6.0e-3
    gate_fraction: float = 0.30
    lift_time: float = 0.20
    kn_over_mg: float = 1.0e3
    kt_ratio: float = 0.5
    restitution: float = 0.05
    mu: float = 0.5
    mu_rolling: float = 0.2
    gamma: float = 0.0
    bridge_volume_ratio: float = 0.0
    pore_pressure_ratio: float = 0.0
    wall_mu: Optional[float] = None
    seed: int = 20261
    # -- state (filled by __post_init__) --------------------------------
    x: np.ndarray = field(default=None, repr=False)
    v: np.ndarray = field(default=None, repr=False)
    theta: np.ndarray = field(default=None, repr=False)
    omega: np.ndarray = field(default=None, repr=False)
    pairs: np.ndarray = field(default=None, repr=False)
    tang: np.ndarray = field(default=None, repr=False)
    roll: np.ndarray = field(default=None, repr=False)
    time: float = 0.0
    gate_open: bool = False
    gate_height: float = 0.0
    contacts: int = 0

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        rng = np.random.default_rng(self.seed)
        d = 2.0 * self.radius
        xs, ys = [], []
        row = 0
        y = self.radius * 1.02
        while y < self.box_height - self.radius:
            x = self.radius + (row % 2) * 0.5 * d * 1.005
            while x < self.gate_fraction * self.box_width:
                xs.append(x)
                ys.append(y)
                x += d * 1.005
            row += 1
            y += 0.866 * d * 1.005
        pos = np.array([xs, ys], dtype=float).T
        if pos.shape[0] > self.n_particles:
            idx = rng.choice(pos.shape[0], self.n_particles, replace=False)
            pos = pos[np.sort(idx)]
        pos += rng.normal(scale=1.0e-3 * self.radius, size=pos.shape)
        self.x = pos
        self.n_particles = pos.shape[0]
        self.v = np.zeros_like(pos)
        self.theta = np.zeros(self.n_particles)
        self.omega = np.zeros(self.n_particles)
        self.pairs = np.zeros((0, 2), dtype=int)
        self.tang = np.zeros((0, 2))
        self.roll = np.zeros(0)
        self.gate_height = self.box_height
        self._refresh_pairs()

    # -- derived ---------------------------------------------------------
    @property
    def mass(self) -> float:
        return self.density * math.pi * self.radius ** 2

    @property
    def inertia(self) -> float:
        return 0.5 * self.mass * self.radius ** 2

    @property
    def weight(self) -> float:
        return self.mass * GRAVITY

    @property
    def bridge_volume(self) -> float:
        return self.bridge_volume_ratio * self.radius ** 3

    @property
    def rupture_gap(self) -> float:
        if self.bridge_volume <= 0.0:
            return 0.0
        return 1.1 * self.bridge_volume ** (1.0 / 3.0)

    @property
    def cohesion_number(self) -> float:
        """``Co = F_cap / (m g)`` at contact (drives the repose angle)."""
        if self.bridge_volume <= 0.0:
            return 0.0
        f = rouault_pendular_force(self.radius, self.gamma, self.bridge_volume, 0.0)
        return f * (1.0 - min(max(self.pore_pressure_ratio, 0.0), 1.0)) / self.weight

    @property
    def kn(self) -> float:
        return self.kn_over_mg * self.weight / (2.0 * self.radius)

    @property
    def timestep(self) -> float:
        w = math.sqrt(2.0 * self.kn / self.mass)
        return 0.12 * math.pi / w

    @property
    def wall_friction(self) -> float:
        return self.mu if self.wall_mu is None else self.wall_mu

    # -- neighbour bookkeeping ------------------------------------------
    def _refresh_pairs(self) -> None:
        r_cut = 2.0 * self.radius + max(self.rupture_gap, 0.3 * self.radius) + 1e-12
        if cKDTree is not None:
            tree = cKDTree(self.x)
            p = tree.query_pairs(r_cut, output_type="ndarray")
        else:  # pragma: no cover - fallback brute force
            n = self.x.shape[0]
            ii, jj = np.triu_indices(n, 1)
            d = np.linalg.norm(self.x[ii] - self.x[jj], axis=1)
            keep = d <= r_cut
            p = np.column_stack([ii[keep], jj[keep]])
        if p.size == 0:
            self.pairs = np.zeros((0, 2), dtype=int)
            self.tang = np.zeros((0, 2))
            self.roll = np.zeros(0)
            return
        p = np.sort(p, axis=1)
        old_t: dict = {}
        old_r: dict = {}
        if self.pairs.size:
            key = self.pairs[:, 0] * (self.n_particles + 1) + self.pairs[:, 1]
            old_t = dict(zip(key.tolist(), self.tang))
            old_r = dict(zip(key.tolist(), self.roll))
        key_new = p[:, 0] * (self.n_particles + 1) + p[:, 1]
        keys = key_new.tolist()
        tang = np.array([old_t.get(k, (0.0, 0.0)) for k in keys], dtype=float)
        roll = np.array([old_r.get(k, 0.0) for k in keys], dtype=float)
        self.pairs = p
        self.tang = tang.reshape(-1, 2)
        self.roll = roll

    # -- one explicit step ---------------------------------------------
    def step(self, dt: Optional[float] = None) -> dict:
        dt = self.timestep if dt is None else dt
        r, m, inertia = self.radius, self.mass, self.inertia
        f = np.zeros_like(self.x)
        tau = np.zeros(self.n_particles)
        f[:, 1] -= m * GRAVITY
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        n_contacts = 0
        if i.size:
            d_vec = self.x[j] - self.x[i]
            dist = np.linalg.norm(d_vec, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                n_hat = d_vec / dist[:, None]
            v_rel = self.v[j] - self.v[i]
            v_n = np.sum(v_rel * n_hat, axis=1)
            # tangential (unit) direction
            t_hat = np.column_stack([-n_hat[:, 1], n_hat[:, 0]])
            v_t = np.sum(v_rel * t_hat, axis=1)
            overlap = 2.0 * r - dist
            contact = overlap > 0.0
            n_contacts = int(contact.sum())
            # normal: spring + dashpot (no tensile stiffness except capillaries)
            cn = 2.0 * math.sqrt(self.kn * m) * (-math.log(max(self.restitution, 1e-6)) / math.pi)
            f_n = np.where(contact, self.kn * np.maximum(overlap, 0.0) - cn * v_n, 0.0)
            f_n = np.maximum(f_n, 0.0)
            # tangential: incremental spring with Coulomb limit, then a
            # velocity-regularised fallback so a sliding contact cannot chatter
            step_t = np.where(contact, v_t * dt, 0.0)
            self.tang += step_t[:, None] * t_hat
            tang_mag = np.linalg.norm(self.tang, axis=1)
            kt = self.kt_ratio * self.kn
            f_t_spring = kt * tang_mag
            f_t = np.where(f_t_spring <= self.mu * f_n, f_t_spring, self.mu * f_n)
            # clip the stored tangential displacement to stay on the yield surface
            scale = np.where(tang_mag > 0.0,
                             np.minimum(1.0, f_t / np.maximum(kt * tang_mag, 1e-30)), 0.0)
            self.tang *= scale[:, None]
            force = n_hat * f_n[:, None]
            if np.any(f_t > 0.0):
                direction = -self.tang / np.maximum(np.linalg.norm(self.tang, axis=1), 1e-30)[:, None]
                direction = np.where(np.isfinite(direction), direction, 0.0)
                force += direction * f_t[:, None]
            # rolling resistance: rolling spring with Coulomb-type torque limit
            rel_omega = self.omega[j] - self.omega[i]
            self.roll += np.where(contact, rel_omega * dt, 0.0)
            k_roll = self.kt_ratio * self.kn * r * r * 0.5
            max_tau = self.mu_rolling * r * f_n
            tau_pair = np.clip(-k_roll * self.roll, -max_tau, max_tau)
            self.roll = np.where(contact, self.roll, 0.0)
            self.roll = np.where(np.abs(k_roll * self.roll) > max_tau,
                                 tau_pair / -np.maximum(k_roll, 1e-30), self.roll)
            tau_pair = np.where(contact, tau_pair, 0.0)
            # pendular liquid bridges (attractive, effective-stress reduced)
            if self.bridge_volume > 0.0:
                gap = dist - 2.0 * r
                active = (gap >= 0.0) & (gap <= self.rupture_gap)
                if np.any(active):
                    lam = 1.0 - min(max(self.pore_pressure_ratio, 0.0), 1.0)
                    f_cap = np.zeros_like(gap)
                    for idx in np.nonzero(active)[0]:
                        f_cap[idx] = rouault_pendular_force(
                            r, self.gamma, self.bridge_volume, float(gap[idx])
                        )
                    force += (-n_hat) * (f_cap * lam)[:, None]
            np.add.at(f, i, -force)
            np.add.at(f, j, force)
            np.add.at(tau, i, -tau_pair)
            np.add.at(tau, j, tau_pair)
        self.contacts = n_contacts
        # walls
        x, v, w = self.x, self.v, self.omega
        left = x[:, 0] < r
        x[left, 0] = r
        v[left, 0] = -self.restitution * v[left, 0]
        right = x[:, 0] > self.box_width - r
        x[right, 0] = self.box_width - r
        v[right, 0] = -self.restitution * v[right, 0]
        floor = x[:, 1] < r
        x[floor, 1] = r
        v[floor, 1] = -self.restitution * v[floor, 1]
        v[floor, 0] *= max(0.0, 1.0 - self.wall_friction * 10.0 * dt)
        # capacity-controlled sill: the gate is slowly withdrawn from the top
        # (a lifting weir) so the cargo spills over it instead of being driven
        # by the collapse kinetic energy
        gate_x = self.gate_fraction * self.box_width
        if not self.gate_open:
            if self.lift_time > 0.0:
                self.gate_height = max(
                    0.0, self.gate_height - self.box_height / self.lift_time * dt
                )
            hit = (np.abs(x[:, 0] - gate_x) < r) & (x[:, 1] < self.gate_height)
            if np.any(hit):
                left = hit & (x[:, 0] < gate_x)
                right = hit & (x[:, 0] >= gate_x)
                x[left, 0] = gate_x - r
                v[left, 0] = -self.restitution * v[left, 0]
                x[right, 0] = gate_x + r
                v[right, 0] = -self.restitution * v[right, 0]
        # integrate (semi-implicit Euler; the capillary/contact forces are stiff
        # enough that this is the same scheme the contact tests use)
        self.v += f / m * dt
        self.x += self.v * dt
        self.omega += tau / inertia * dt
        self.theta += self.omega * dt
        self.time += dt
        if int(round(self.time / dt)) % 12 == 0:
            self._refresh_pairs()
        return {"time": self.time, "contacts": n_contacts,
                "kinetic": float(0.5 * np.sum(self.v ** 2) * m)}

    # -- diagnostics -----------------------------------------------------
    def surface_profile(self, n_bins: int = 30) -> tuple[np.ndarray, np.ndarray]:
        """Highest particle height per x-bin (the free surface)."""
        edges = np.linspace(0.0, self.box_width, n_bins + 1)
        idx = np.clip(np.digitize(self.x[:, 0], edges) - 1, 0, n_bins - 1)
        top = np.full(n_bins, np.nan)
        for b in range(n_bins):
            sel = idx == b
            if np.any(sel):
                top[b] = float(self.x[sel, 1].max())
        centres = 0.5 * (edges[1:] + edges[:-1])
        return centres, top

    def repose_angle_deg(
        self,
        n_bins: int = 36,
        toe_fraction: float = 0.12,
        smooth: int = 5,
    ) -> dict:
        """Repose angle of the settled heap (deg).

        The crest-to-toe flank is isolated by (i) finding the highest bin,
        (ii) walking to the first bin below ``toe_fraction`` of the crest
        height, (iii) smoothing the surface by a ``smooth``-bin running mean,
        and (iv) fitting a straight line to that flank.  A robust
        median-of-local-slopes estimate is reported next to the least-squares
        value so a curved flank cannot masquerade as a clean angle.
        """
        centres, top = self.surface_profile(n_bins)
        good = np.isfinite(top)
        if good.sum() < 6:
            return {"theta_r_deg": float("nan"), "r2": 0.0, "n_fit": int(good.sum())}
        h = np.where(good, top, 0.0)
        if smooth > 1:
            k = np.ones(smooth) / smooth
            h_s = np.convolve(h, k, mode="same")
            edge = smooth // 2
            h_s[:edge] = h[:edge]
            h_s[-edge:] = h[-edge:]
            h = h_s
        crest = int(np.argmax(h))
        h_max = h[crest]
        below = np.nonzero(h[crest:] < toe_fraction * h_max)[0]
        toe = crest + int(below[0]) if below.size else n_bins - 1
        toe = min(toe, n_bins - 1)
        idx = np.arange(crest, max(toe, crest + 3))
        idx = idx[idx < n_bins]
        # require a real flank: at least 5 bins above the toe threshold
        idx = idx[h[idx] > toe_fraction * h_max]
        if idx.size < 4:
            return {"theta_r_deg": float("nan"), "r2": 0.0, "n_fit": int(idx.size)}
        x_fit, y_fit = centres[idx], h[idx]
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        pred = slope * x_fit + intercept
        ss_res = float(np.sum((y_fit - pred) ** 2))
        ss_tot = float(np.sum((y_fit - y_fit.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
        local = np.diff(y_fit) / np.diff(x_fit)
        median_slope = float(np.median(local))
        return {
            "theta_r_deg": math.degrees(math.atan(abs(slope))),
            "theta_r_median_deg": math.degrees(math.atan(abs(median_slope))),
            "slope": float(slope),
            "r2": r2,
            "n_fit": int(idx.size),
            "crest_x": float(centres[crest]),
            "crest_height": float(h_max),
        }

    def kinetic_energy(self) -> float:
        return float(0.5 * self.mass * np.sum(self.v ** 2))


def collapse_test(
    *,
    steps: int = 6000,
    record_every: int = 500,
    quiescence_ratio: float = 1.0e-4,
    **kwargs,
) -> dict:
    """Release the gate, run to quiescence and measure the repose angle."""
    pile = DiskPile(**kwargs)
    e_pot0 = pile.mass * GRAVITY * float(np.sum(pile.x[:, 1]))
    pile.gate_open = True
    e_series = []
    for k in range(steps):
        pile.step()
        if (k + 1) % record_every == 0:
            e_series.append(pile.kinetic_energy() / max(e_pot0, 1e-30))
            if e_series[-1] < quiescence_ratio and k > steps // 4:
                break
    e_pot1 = pile.mass * GRAVITY * float(np.sum(pile.x[:, 1]))
    angle = pile.repose_angle_deg()
    angle.update(
        {
            "steps": k + 1,
            "time_s": pile.time,
            "cohesion_number": pile.cohesion_number,
            "mu": pile.mu,
            "mu_rolling": pile.mu_rolling,
            "kinetic_fraction": e_series[-1] if e_series else float("nan"),
            "potential_drop": e_pot1 - e_pot0,
            "n_particles": pile.n_particles,
        }
    )
    return angle


def sliding_onset_angle(
    mu: float,
    tilt_deg: float,
    *,
    radius: float = 1.0e-4,
    density: float = 2650.0,
    kn_over_mg: float = 1.0e3,
    steps: int = 4000,
) -> dict:
    """Single disc on an inclined plane: displacement and slip/no-slip verdict.

    The analytic anchor is ``tan(tilt) > mu`` for an irrotational slider; the
    disc rolls only if ``mu_rolling`` allows it, so this is run with
    ``mu_rolling = 0`` and the rotation frozen (an ideal slider).
    """
    m = density * math.pi * radius ** 2
    kn = kn_over_mg * m * GRAVITY / (2.0 * radius)
    dt = 0.12 * math.pi / math.sqrt(2.0 * kn / m)
    tilt = math.radians(tilt_deg)
    n = np.array([-math.sin(tilt), math.cos(tilt)])
    t_dir = np.array([math.cos(tilt), math.sin(tilt)])
    x = n * radius
    v = np.zeros(2)
    s0 = float(x @ t_dir)
    drive = m * GRAVITY * math.sin(tilt)
    normal = m * GRAVITY * math.cos(tilt)
    slide = drive > mu * normal
    for _ in range(steps):
        v += (m * GRAVITY * np.array([0.0, -1.0])) / m * dt
        pen = float(x @ n) - radius
        if pen < 0.0:
            x = x - n * pen
            v = v - n * float(v @ n)
            v_t = float(v @ t_dir)
            if abs(v_t) > 1e-15:
                v = v - t_dir * math.copysign(min(mu * normal / m * dt, abs(v_t)), v_t)
        x = x + v * dt
        pen = float(x @ n) - radius
        if pen < 0.0:
            x = x - n * pen
    s1 = float(x @ t_dir)
    return {
        "mu": mu,
        "tilt_deg": tilt_deg,
        "analytic_slides": slide,
        "displacement": s1 - s0,
        "verdict": "slid" if (s1 - s0) > 5.0 * radius else "held",
        "consistent": ((s1 - s0) > 5.0 * radius) == slide,
    }