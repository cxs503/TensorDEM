"""2-D pile test bed for shape representation (P40-N10): ``sphere`` channel.

Campaign node P40-N10 (TensorDEM).  The module was dropped by the merge gate while
its acceptance tests survived; restored here to the contract those tests pin:

* explicit soft-particle DEM, Hertz normal spring with restitution-based critical
  damping, Coulomb tangential spring with per-pair history, rolling resistance;
* a ``sphere`` channel (2-D disks) with one contact-law family, so that a repose
  comparison across shape channels differs only by the representation;
* :func:`repose_angle_from_pile` -- the free-surface envelope estimator, validated
  against a synthetic pile of known flank angle;
* :func:`run_pile` -- the study driver used by ``scripts/p40_n10_shape_verdict.py``.

Scope of the restoration
------------------------
Only the ``sphere`` channel is restored.  ``clump`` (rigid disk clusters) and
``superellipse`` (support-function contact) channels were dropped with the module
and are *not* re-derived here: they raise :class:`NotImplementedError` so that no
caller can silently mistake a disk for a clump (see
``docs/SHAPE_REPRESENTATION_MODULE_RESTORATION.md``).

Units: SI, 2-D with unit out-of-plane thickness.  A disk of radius ``r`` and
density ``rho`` has mass ``rho pi r^2`` and inertia ``m r^2 / 2``.  Contact
stiffness is the 3-D Hertz value for equivalent spheres, which keeps the
stiffness-to-weight ratio of the reference study.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "PileConfig2D",
    "PileResult",
    "repose_angle_from_pile",
    "run_pile",
]


@dataclass(frozen=True)
class PileConfig2D:
    """Configuration of the 2-D pile test bed."""

    channel: str = "sphere"          # "sphere" | "clump" | "superellipse"
    n_particles: int = 100
    radius: float = 0.02
    dt: float = 1.5e-4
    n_steps: int = 6000
    n_settle: int = 4500
    mu_slide: float = 0.55
    mu_roll: float = 0.06
    restitution: float = 0.1
    viscous_damp: float = 0.99
    record_every: int = 1000
    n_sub: int = 8                   # spheres per clump (clump channel only)
    aspect: float = 1.0              # b/a of the body (clump / superellipse)
    shape_factor: float = 2.0        # superellipse exponent
    density: float = 2500.0          # kg/m3
    young: float = 5.0e7             # Pa
    poisson: float = 0.25
    gravity: float = 9.81
    box_half: float = 0.35           # half width of the container (m)
    gate_half: float = 0.10          # half width of the released block (m)
    spacing_factor: float = 2.4      # initial centre spacing / radius
    seed: int = 0

    def __post_init__(self) -> None:
        if self.channel not in ("sphere", "clump", "superellipse"):
            raise ValueError(f"unknown channel {self.channel!r}")
        if self.radius <= 0.0 or self.dt <= 0.0:
            raise ValueError("radius and dt must be positive")
        if not 0.0 <= self.mu_slide and 0.0 <= self.mu_roll:
            raise ValueError("friction coefficients must be non-negative")
        if not 0.0 < self.restitution <= 1.0:
            raise ValueError("restitution must be in (0, 1]")
        if not 0.0 < self.viscous_damp <= 1.0:
            raise ValueError("viscous_damp must be in (0, 1]")


def _critical_beta(restitution: float) -> float:
    """Damping ratio that reproduces the requested coefficient of restitution."""
    e = min(max(float(restitution), 1e-6), 1.0)
    ln_e = math.log(e)
    return -ln_e / math.sqrt(ln_e * ln_e + math.pi ** 2)


class _Pile:
    """A 2-D pile of disks on a frictional floor inside a rectangular box."""

    def __init__(self, cfg: PileConfig2D) -> None:
        if cfg.channel != "sphere":
            raise NotImplementedError(
                f"channel {cfg.channel!r} was dropped with the pre-gate module and is "
                "not restored; only 'sphere' is available (see "
                "docs/SHAPE_REPRESENTATION_MODULE_RESTORATION.md)"
            )
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        r = float(cfg.radius)
        n = int(cfg.n_particles)
        self.radius = r
        self.mass = cfg.density * math.pi * r * r
        self.inertia = 0.5 * self.mass * r * r

        # loose column inside the gate, layer by layer, no initial overlap
        spacing = cfg.spacing_factor * r
        per_row = max(int(math.floor(2.0 * cfg.gate_half / spacing)), 1)
        self.pos = np.zeros((n, 2))
        for k in range(n):
            row, col = divmod(k, per_row)
            x = -0.5 * (per_row - 1) * spacing + col * spacing
            y = r + row * spacing
            self.pos[k] = (x, y)
        self.pos += 0.02 * r * self.rng.standard_normal((n, 2))
        self.vel = np.zeros((n, 2))
        self.theta = np.zeros(n)
        self.omega = np.zeros(n)
        self.t = 0
        self.contacts = 0

        # per-pair tangential history (only the upper triangle is used)
        self._xi = np.zeros((n, n, 2))
        self._xi_wall = np.zeros((n, 4, 2))       # floor, x-, x+, (unused)
        nu = cfg.poisson
        self._e_pair = cfg.young / (2.0 * (1.0 - nu * nu))
        self._e_wall = cfg.young / (1.0 - nu * nu)
        self._k_pair = (4.0 / 3.0) * self._e_pair * math.sqrt(0.5 * r)
        self._k_wall = (4.0 / 3.0) * self._e_wall * math.sqrt(r)
        self._beta = _critical_beta(cfg.restitution)

    # -- geometry helpers ---------------------------------------------------
    @property
    def n(self) -> int:
        return self.pos.shape[0]

    def _paired(self) -> Tuple[np.ndarray, np.ndarray]:
        ii, jj = np.triu_indices(self.n, k=1)
        return ii, jj

    # -- force model --------------------------------------------------------
    def _pair_forces(self) -> Tuple[np.ndarray, np.ndarray, int]:
        cfg = self.cfg
        ii, jj = self._paired()
        d = self.pos[jj] - self.pos[ii]
        dist = np.linalg.norm(d, axis=1)
        active = dist < 2.0 * self.radius
        if not np.any(active):
            self._xi[ii, jj] = 0.0
            return np.zeros_like(self.pos), np.zeros(self.n), 0
        ii, jj, d, dist = ii[active], jj[active], d[active], dist[active]
        n_hat = d / dist[:, None]
        delta = 2.0 * self.radius - dist
        f_n = self._k_pair * delta ** 1.5
        k_eff = 1.5 * self._k_pair * np.sqrt(delta)
        m_eff = 0.5 * self.mass
        c_n = 2.0 * self._beta * np.sqrt(k_eff * m_eff)

        v_rel = self.vel[jj] - self.vel[ii]
        v_n = np.sum(v_rel * n_hat, axis=1)
        f_n = np.maximum(f_n - c_n * v_n, 0.0)

        # tangential: velocity of the contact point (includes spin)
        r_vec = -self.radius * n_hat
        v_ci = self.vel[ii] + np.column_stack([-self.omega[ii] * r_vec[:, 1],
                                               self.omega[ii] * r_vec[:, 0]])
        v_cj = self.vel[jj] - np.column_stack([-self.omega[jj] * r_vec[:, 1],
                                               self.omega[jj] * r_vec[:, 0]])
        t_hat = np.column_stack([-n_hat[:, 1], n_hat[:, 0]])
        v_t = np.sum((v_cj - v_ci) * t_hat, axis=1)

        k_t = 0.4 * k_eff
        self._xi[ii, jj] += (v_t * cfg.dt)[:, None] * t_hat
        f_t = -k_t * np.sum(self._xi[ii, jj] * t_hat, axis=1) - 0.2 * c_n * v_t
        f_max = cfg.mu_slide * f_n
        slip = np.abs(f_t) > f_max
        f_t = np.where(slip, np.sign(f_t) * f_max, f_t)
        # reset the stored history on the sliding branch (project to the limit)
        if np.any(slip):
            proj = (np.sign(f_t) * f_max / np.maximum(k_t, 1e-30))[:, None] * t_hat
            self._xi[ii[slip], jj[slip]] = -proj[slip]

        force = np.zeros_like(self.pos)
        f_pair = f_n[:, None] * n_hat + f_t[:, None] * t_hat
        np.add.at(force, ii, -f_pair)
        np.add.at(force, jj, f_pair)

        # rolling resistance torque: M = -mu_r F_n R sign(omega_i - omega_j)
        torque = np.zeros(self.n)
        d_omega = self.omega[ii] - self.omega[jj]
        m_roll = cfg.mu_roll * f_n * 0.5 * self.radius * np.sign(d_omega)
        np.add.at(torque, ii, -m_roll)
        np.add.at(torque, jj, -m_roll)
        return force, torque, len(ii)

    def _wall_forces(self) -> Tuple[np.ndarray, np.ndarray, int]:
        cfg = self.cfg
        force = np.zeros_like(self.pos)
        torque = np.zeros(self.n)
        count = 0
        walls = (
            (np.array([0.0, 1.0]), 0.0, 0),                  # floor y >= r
            (np.array([1.0, 0.0]), -cfg.box_half, 1),        # left wall x >= -box+r
            (np.array([-1.0, 0.0]), -cfg.box_half, 2),       # right wall x <= box-r
        )
        for n_hat, offset, idx in walls:
            if idx == 0:
                pen = self.radius - self.pos[:, 1]
            elif idx == 1:
                pen = self.radius - (self.pos[:, 0] + cfg.box_half)
            else:
                pen = self.radius - (cfg.box_half - self.pos[:, 0])
            active = pen > 0.0
            if not np.any(active):
                continue
            count += int(active.sum())
            delta = pen[active]
            k_eff = 1.5 * self._k_wall * np.sqrt(delta)
            c_n = 2.0 * self._beta * np.sqrt(k_eff * self.mass)
            v_n = np.sum(self.vel[active] * n_hat, axis=1)
            f_n = np.maximum(self._k_wall * delta ** 1.5 - c_n * v_n, 0.0)
            force[active] += f_n[:, None] * n_hat[None, :]

            t_hat = np.array([-n_hat[1], n_hat[0]])
            r_vec = -self.radius * n_hat
            v_c = self.vel[active] + np.column_stack([-self.omega[active] * r_vec[1],
                                                      self.omega[active] * r_vec[0]])
            v_t = v_c @ t_hat
            k_t = 0.4 * k_eff
            self._xi_wall[active, idx] += (v_t * cfg.dt)[:, None] * t_hat[None, :]
            f_t = -k_t * (self._xi_wall[active, idx] @ t_hat) - 0.2 * c_n * v_t
            f_max = cfg.mu_slide * f_n
            slip = np.abs(f_t) > f_max
            f_t = np.where(slip, np.sign(f_t) * f_max, f_t)
            if np.any(slip):
                self._xi_wall[active, idx][slip] = (
                    -np.sign(f_t[slip]) * f_max[slip] / np.maximum(k_t[slip], 1e-30)
                )[:, None] * t_hat[None, :]
            force[active] += f_t[:, None] * t_hat[None, :]
            torque[active] += -cfg.mu_roll * f_n * 0.5 * self.radius * np.sign(self.omega[active])
        return force, torque, count

    # -- integration --------------------------------------------------------
    def step(self) -> None:
        cfg = self.cfg
        force, torque, n_contacts = self._pair_forces()
        f_wall, t_wall, n_wall = self._wall_forces()
        force += f_wall
        torque += t_wall
        self.contacts = n_contacts + n_wall

        force[:, 1] -= self.mass * cfg.gravity
        self.vel += (force / self.mass) * cfg.dt
        self.omega += (torque / self.inertia) * cfg.dt
        self.vel *= cfg.viscous_damp
        self.omega *= cfg.viscous_damp
        self.pos += self.vel * cfg.dt
        self.theta += self.omega * cfg.dt
        self.t += 1

    def energy(self) -> Tuple[float, float]:
        """``(kinetic, potential)`` energy; potential is measured from the floor."""
        ke = float(np.sum(0.5 * self.mass * np.sum(self.vel ** 2, axis=1)
                          + 0.5 * self.inertia * self.omega ** 2))
        pe = float(np.sum(self.mass * self.cfg.gravity * self.pos[:, 1]))
        return ke, pe

    def total_energy(self) -> float:
        ke, pe = self.energy()
        return ke + pe

    def repose_angle(self, bin_width: Optional[float] = None) -> Tuple[float, float]:
        bw = 0.5 * self.radius if bin_width is None else bin_width
        return repose_angle_from_pile(self.pos, bw)


def repose_angle_from_pile(points: np.ndarray, bin_width: float) -> Tuple[float, float]:
    """Angle of repose from the free-surface envelope of a 2-D point cloud.

    The pile is binned along ``x`` (width ``bin_width``); the per-bin maximum ``y``
    is the envelope.  The flank on each side of the apex is fitted by a straight
    line over the bins below ``0.9 *`` the apex height, and the returned angle is
    the mean of the two flank angles; the spread is half their difference (so an
    asymmetric pile reports a wide bar instead of a biased angle).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.shape[0] < 4:
        return float("nan"), float("nan")
    x, y = pts[:, 0], pts[:, 1]
    edges = np.arange(x.min(), x.max() + bin_width, bin_width)
    if len(edges) < 3:
        return float("nan"), float("nan")
    idx = np.digitize(x, edges) - 1
    env = np.full(len(edges) - 1, np.nan)
    for b in range(len(env)):
        sel = idx == b
        if np.any(sel):
            env[b] = float(y[sel].max())
    xs = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(env)
    if not np.any(ok):
        return float("nan"), float("nan")
    xs, env = xs[ok], env[ok]
    if len(xs) < 3:
        return float("nan"), float("nan")
    peak = int(np.argmax(env))
    apex = env[peak]

    def flank(side: str) -> float:
        if side == "left":
            sl = slice(0, peak)
            if peak < 1:
                return float("nan")
            dx = xs[peak] - (xs[sl] + 0.5 * bin_width)
            dy = apex - env[sl]
        else:
            sl = slice(peak + 1, len(xs))
            if peak >= len(xs) - 1:
                return float("nan")
            dx = (xs[sl] - 0.5 * bin_width) - xs[peak]
            dy = apex - env[sl]
        keep = dy > 0.15 * apex        # ignore the flat toe and the apex bin
        if np.count_nonzero(keep) < 2:
            return float("nan")
        slope = float(np.sum(dx[keep] * dy[keep]) / np.sum(dx[keep] ** 2))
        return math.degrees(math.atan(slope))

    angles = [a for a in (flank("left"), flank("right")) if math.isfinite(a)]
    if not angles:
        return float("nan"), float("nan")
    if len(angles) == 1:
        return angles[0], 0.0
    return 0.5 * (angles[0] + angles[1]), 0.5 * abs(angles[0] - angles[1])


@dataclass
class PileResult:
    """Outcome of one pile run (fields consumed by the P40-N10 verdict script)."""

    channel: str
    n_particles: int
    n_sub: int
    shape_factor: float
    aspect: float
    repose_deg: float
    repose_deg_std: float
    pseudo_dissipation: float
    kinetic_energy_final: float
    potential_energy_final: float
    contacts_final: int
    energy_history: List[Dict[str, float]] = field(default_factory=list)
    mass_kg: float = 0.0
    wall_time_s: float = 0.0


def run_pile(cfg: PileConfig2D) -> PileResult:
    """Collapse a loose block into a pile and measure repose and spurious dissipation."""
    import time

    t0 = time.time()
    pile = _Pile(cfg)
    history: List[Dict[str, float]] = []
    prev = pile.total_energy()
    ratios: List[float] = []
    for step in range(cfg.n_steps):
        pile.step()
        if step >= cfg.n_settle:
            e = pile.total_energy()
            ratios.append(abs(e - prev) / max(abs(e), 1e-30))
            prev = e
        if cfg.record_every and (step + 1) % cfg.record_every == 0:
            ke, pe = pile.energy()
            history.append({"step": step + 1, "kinetic": ke, "potential": pe,
                            "contacts": pile.contacts})
    phi, spread = pile.repose_angle()
    ke, pe = pile.energy()
    return PileResult(
        channel=cfg.channel, n_particles=cfg.n_particles, n_sub=cfg.n_sub,
        shape_factor=cfg.shape_factor, aspect=cfg.aspect,
        repose_deg=phi, repose_deg_std=spread,
        pseudo_dissipation=float(np.mean(ratios)) if ratios else float("nan"),
        kinetic_energy_final=ke, potential_energy_final=pe,
        contacts_final=pile.contacts, energy_history=history,
        mass_kg=pile.mass, wall_time_s=time.time() - t0,
    )