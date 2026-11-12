"""Two-dimensional soft-sphere DEM kernel for a rotating circular drum.

The kernel is deliberately small and explicit so that every force term can be
audited:

* normal contact   : linear spring-dashpot, ``F_n = k_n d - gamma v_n`` (clamped >= 0)
* tangential       : penalised Coulomb friction, ``|F_t| <= mu F_n`` applied at the
                     contact point, so it also generates torque (spin DOF is tracked)
* rolling friction : torque ``mu_r F_n r`` opposing spin
* wall             : circular drum wall moving with surface velocity ``omega x r``
* integration      : semi-implicit Euler at fixed ``dt``

The drum rotates about its centre with angular velocity ``omega``; gravity acts
along ``-y``.  Particles are disks (per-unit-thickness mass) so the model is a
quasi-2D slice of a long cylinder, which is the geometry of the classical
Brazil-nut / core-segregation drum experiments.

Public API
----------
``DrumConfig``            frozen parameter record
``make_bed``              well-mixed bidisperse RSA bed inside the drum
``simulate``              run settle + rotation, returning snapshots and diagnostics
``DrumRun``               result record (positions, species, time series)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial import cKDTree

from . import metrics as _m

__all__ = ["DrumConfig", "DrumRun", "make_bed", "simulate", "coulomb_damping_ratio"]


@dataclass(frozen=True)
class DrumConfig:
    """Parameter record for one drum segregation case."""

    # --- geometry -------------------------------------------------------
    radius: float = 0.150          # drum inner radius [m]
    fill: float = 0.30             # bed cross-section / drum cross-section
    d_small: float = 0.010         # small particle diameter [m]
    d_large: float = 0.020         # large particle diameter [m]
    large_area_fraction: float = 0.30   # area fraction of the LARGE species in the bed
    thickness: float = 0.010       # out-of-plane thickness (per-unit-thickness mass) [m]

    # --- material -------------------------------------------------------
    rho_small: float = 2500.0      # [kg/m^3]
    rho_large: float = 2500.0      # [kg/m^3]
    kn: float = 5.0e4              # normal stiffness [N/m]
    kt_ratio: float = 0.5          # tangential / normal stiffness
    restitution: float = 0.30      # normal restitution coefficient
    mu: float = 0.40               # sliding friction coefficient
    mu_rolling: float = 0.10       # rolling friction coefficient (rough grains)

    # --- driving --------------------------------------------------------
    omega: float = 2.0             # drum angular velocity [rad/s]
    dt: float = 1.0e-4             # time step [s]
    settle_steps: int = 6000       # static settling before rotation starts
    settle_speed: float = 5.0e-3   # rms particle speed [m/s] that defines "bed at rest"
    settle_damping: float = 0.80   # per-step velocity factor of the quasistatic bed prep
    settle_max_quench_rounds: int = 20   # cap on kinetic-energy quench rounds
    revs: float = 6.0              # number of drum revolutions simulated
    sample_revs: float = 0.25      # snapshot interval in revolutions

    # --- misc -----------------------------------------------------------
    n_bins: int = 10               # radial bins for the concentration profile
    bed_region_factor: float = 2.6  # RSA region / bed area (looseness of the initial bed)
    skin: float = 0.25             # neighbour-list skin (fraction of max diameter)
    rebuild_steps: int = 8         # neighbour list rebuild interval
    seed: int = 0
    gravity: float = 9.81

    # --- derived --------------------------------------------------------
    @property
    def r_small(self) -> float:
        return 0.5 * self.d_small

    @property
    def r_large(self) -> float:
        return 0.5 * self.d_large

    @property
    def size_ratio(self) -> float:
        return self.d_large / self.d_small

    @property
    def density_ratio(self) -> float:
        return self.rho_large / self.rho_small

    @property
    def n_steps(self) -> int:
        if self.omega <= 0.0:
            # static controls still integrate for the same physical time as a
            # reference rotation so that the comparison is time-matched
            return int(round(self.revs * 2.0 * math.pi / 2.0 / self.dt))
        return int(round(self.revs * 2.0 * math.pi / self.omega / self.dt))

    @property
    def froude(self) -> float:
        """Froude number of the drum wall, Fr = omega^2 R / g."""
        return self.omega ** 2 * self.radius / self.gravity

    def with_(self, **kw) -> "DrumConfig":
        return replace(self, **kw)


@dataclass
class DrumRun:
    """Result of one drum simulation."""

    config: DrumConfig
    pos: np.ndarray                 # (N,2) final positions
    radii: np.ndarray               # (N,)
    is_large: np.ndarray            # (N,) bool
    masses: np.ndarray              # (N,)
    actual_fill: float
    actual_large_area_fraction: float
    n_large: int
    n_small: int
    vel: np.ndarray = None          # (N,2) final velocities
    spin: np.ndarray = None         # (N,) final spins
    snapshots: list = field(default_factory=list)   # list of dict(pos, rev)
    series: dict = field(default_factory=dict)      # time series of scalars
    diagnostics: dict = field(default_factory=dict)


def coulomb_damping_ratio(restitution: float) -> float:
    """Viscous damping ratio matching a target coefficient of restitution."""
    e = float(min(max(restitution, 1e-3), 0.999))
    return -math.log(e) / math.sqrt(math.pi ** 2 + math.log(e) ** 2)


# ---------------------------------------------------------------------------
# bed generation
# ---------------------------------------------------------------------------
def make_bed(cfg: DrumConfig, rng: np.random.Generator | None = None) -> tuple:
    """Build a randomly mixed (well mixed) bidisperse bed in the drum.

    Particles are placed by random sequential addition inside the lower part of
    the drum and then relaxed.  Species labels are drawn at random so that the
    *initial* state carries no segregation: any segregation measured later is
    produced by the dynamics.

    Returns ``(pos, radii, is_large)``.
    """
    rng = np.random.default_rng(cfg.seed if rng is None else rng)
    rs, rl = cfg.r_small, cfg.r_large
    a_s, a_l = math.pi * rs ** 2, math.pi * rl ** 2

    bed_area = cfg.fill * math.pi * cfg.radius ** 2
    area_large = cfg.large_area_fraction * bed_area
    area_small = bed_area - area_large
    n_large = max(1, int(round(area_large / a_l)))
    n_small = max(1, int(round(area_small / a_s)))
    n = n_large + n_small

    radii = np.concatenate([np.full(n_large, rl), np.full(n_small, rs)])
    is_large = np.concatenate([np.ones(n_large, bool), np.zeros(n_small, bool)])
    perm = rng.permutation(n)
    radii, is_large = radii[perm], is_large[perm]

    # RSA inside a circular-segment region below y = y_top.  The region is sized
    # so that random sequential addition can reach the target bed area (RSA
    # saturates near 0.55-0.60 area fraction), then the bed is settled under
    # gravity which compacts it to the requested fill.
    gap = 1.01
    pos, radii_out = _rsa_place(cfg, radii, bed_area, rng, gap,
                                factor=float(cfg.bed_region_factor))
    pos = np.array(pos)

    # relax overlaps and let the bed fall to the bottom of the drum
    pos = _relax(pos, radii, cfg.radius, n_iter=120)
    return pos, radii, is_large



def _u_for_segment_area(target_fraction: float) -> float:
    """Chord height u (y_top/R) such that the area below it is target_fraction
    of the whole disk (0 < target < 1).  Solved by bisection on the closed form."""
    target_fraction = float(min(max(target_fraction, 1e-3), 0.995))

    def below(u: float) -> float:
        return 1.0 - (math.acos(u) - u * math.sqrt(max(0.0, 1.0 - u * u))) / math.pi

    lo, hi = -1.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if below(mid) < target_fraction:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)



def _rsa_place(cfg, radii, bed_area, rng, gap=1.01, max_tries=6, factor=2.6):
    """Random sequential addition, largest particle first.

    Two practical points:

    * the region (a circular segment below ``y_top``) is widened until the bed
      fits; random packing saturates near 0.55-0.60 area fraction, so a fixed
      region fails for dense beds, and for ``fill -> 1`` the region grows to
      (almost) the whole drum, which is exactly the physical limit being probed;
    * particles are inserted **in descending radius order**.  Inserting big
      particles last jams RSA (they no longer find holes) and made the size-ratio
      3 and fill 0.5 cases unbuildable.  Insertion order does not bias the radial
      distribution (the region is homogeneous) and the initial size-position
      correlation is checked in the tests.
    * the tolerated gap shrinks on retry (down to 0.5), i.e. the last resort is a
      loose bed with overlaps which ``_relax`` and the settling phase remove.
    """
    n = len(radii)
    order = np.argsort(-radii)
    gaps = [gap, 1.005, 1.0, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    placed = 0
    for attempt_no in range(len(gaps) + max_tries):
        tol = gaps[min(attempt_no, len(gaps) - 1)]
        target_region = min(factor * bed_area, 0.985 * math.pi * cfg.radius ** 2)
        u_top = _u_for_segment_area(target_region / (math.pi * cfg.radius ** 2))
        y_top = u_top * cfg.radius
        pos = np.zeros((n, 2))
        filled = np.zeros(n, bool)
        pts = np.zeros((0, 2))
        done = np.zeros(n, bool)
        attempts = 0
        max_attempts = 50 * n
        while not done.all() and attempts < max_attempts:
            idx = int(order[np.argmax(~done)]) if False else None
            # next unplaced particle (largest first among the unplaced)
            remaining = order[~done[order]]
            i = int(remaining[0])
            attempts += 1
            r_try = cfg.radius - 1.02 * radii[i] - 1e-9
            ang = rng.uniform(0.0, 2.0 * math.pi)
            rad = r_try * math.sqrt(rng.uniform(0.0, 1.0))
            p = np.array([rad * math.cos(ang), rad * math.sin(ang)])
            if p[1] > y_top:
                continue
            if pts.shape[0]:
                d = np.linalg.norm(pts - p, axis=1)
                if np.any(d < tol * (radii[i] + radii[filled])):
                    continue
            pos[i] = p
            filled[i] = True
            done[i] = True
            pts = np.vstack([pts, p[None, :]])
        placed = int(done.sum())
        if placed == n:
            return pos, radii
        factor *= 1.15
    raise RuntimeError(
        f"RSA could only place {placed}/{n} particles; reduce fill or particle count")


def _relax(pos: np.ndarray, radii: np.ndarray, drum_r: float, n_iter: int = 100) -> np.ndarray:
    pos = pos.copy()
    for _ in range(n_iter):
        tree = cKDTree(pos)
        pairs = tree.query_pairs(1.02 * 2.0 * radii.max(), output_type="ndarray")
        if len(pairs) == 0:
            break
        i, j = pairs[:, 0], pairs[:, 1]
        d = pos[j] - pos[i]
        dist = np.linalg.norm(d, axis=1)
        overlap = radii[i] + radii[j] - dist
        act = (overlap > 0) & (dist > 1e-12)
        if not act.any():
            break
        i, j, dist, overlap = i[act], j[act], dist[act], overlap[act]
        n = d[act] / dist[:, None]
        push = 0.5 * overlap[:, None] * n
        delta = np.zeros_like(pos)
        np.add.at(delta, i, -push)
        np.add.at(delta, j, push)
        pos += delta
        r_pos = np.linalg.norm(pos, axis=1)
        outside = r_pos > drum_r - 1.02 * radii
        if outside.any():
            scale = (drum_r - 1.02 * radii[outside]) / r_pos[outside]
            pos[outside] *= scale[:, None]
    return pos


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------
def _contact_forces(pos, vel, spin, radii, masses, inertia, cfg, dt, zeta):
    """Return ``(force, torque, n_pairs, max_overlap_ratio, n_contacts)``.

    ``n_contacts`` counts, per particle, the pair contacts *and* the wall
    contact carrying a non-zero normal force at this instant; it is the
    observable behind the ``airborne_fraction`` diagnostic (a particle with
    ``n_contacts == 0`` is in free flight).
    """
    n = len(pos)
    force = np.zeros((n, 2))
    torque = np.zeros(n)
    n_contacts = np.zeros(n, dtype=np.int64)

    kn, mu = cfg.kn, cfg.mu
    kt = cfg.kt_ratio * kn

    # ---- particle-particle ------------------------------------------------
    reach = 2.0 * cfg.r_large * (1.0 + cfg.skin)
    tree = cKDTree(pos)
    pairs = tree.query_pairs(reach, output_type="ndarray")
    max_ov = 0.0
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        d = pos[j] - pos[i]
        dist = np.linalg.norm(d, axis=1)
        dist_safe = np.maximum(dist, 1e-12)
        delta = radii[i] + radii[j] - dist
        act = delta > 0.0
        i, j, delta = i[act], j[act], delta[act]
        d = d[act]
        dist_safe = dist_safe[act]
        if len(i):
            nvec = d / dist_safe[:, None]
            max_ov = float(np.max(delta / (radii[i] + radii[j])))
            tvec = np.stack([-nvec[:, 1], nvec[:, 0]], axis=1)
            v_rel = vel[j] - vel[i]
            vn = np.sum(v_rel * nvec, axis=1)
            m_eff = masses[i] * masses[j] / (masses[i] + masses[j])
            gamma = 2.0 * zeta * np.sqrt(kn * m_eff)
            fn = np.maximum(kn * delta - gamma * vn, 0.0)
            vt = (np.sum(v_rel * tvec, axis=1)
                  - (radii[i] * spin[i] + radii[j] * spin[j]))
            # tangential force from a one-step spring displacement (no memory ->
            # viscous Coulomb friction); magnitude capped by mu*F_n
            ft = -np.clip(kt * vt * dt, -mu * fn, mu * fn)
            fvec = fn[:, None] * nvec + ft[:, None] * tvec
            np.add.at(force, i, -fvec)
            np.add.at(force, j, fvec)
            np.add.at(n_contacts, i, 1)
            np.add.at(n_contacts, j, 1)
            np.add.at(torque, i, -radii[i] * ft)
            np.add.at(torque, j, -radii[j] * ft)
            np.add.at(torque, i, -cfg.mu_rolling * fn * radii[i] * np.tanh(spin[i] / 0.5))
            np.add.at(torque, j, -cfg.mu_rolling * fn * radii[j] * np.tanh(spin[j] / 0.5))

    # ---- wall -------------------------------------------------------------
    r_pos = np.linalg.norm(pos, axis=1)
    near = r_pos > cfg.radius - 1.15 * radii
    if near.any():
        idx = np.nonzero(near)[0]
        p = pos[idx]
        rp = r_pos[idx]
        nvec = -p / rp[:, None]                       # wall -> particle (inward)
        delta = rp + radii[idx] - cfg.radius      # penetration depth
        act = delta > 0.0
        if act.any():
            idx = idx[act]
            nvec = nvec[act]
            delta = delta[act]
            p = p[act]
            max_ov = max(max_ov, float(np.max(delta / radii[idx])))
            tvec = np.stack([-nvec[:, 1], nvec[:, 0]], axis=1)
            u_wall = cfg.omega * np.stack([-p[:, 1], p[:, 0]], axis=1)
            v_contact = vel[idx] + (spin[idx] * radii[idx])[:, None] * tvec
            v_rel = v_contact - u_wall
            vn = np.sum(v_rel * nvec, axis=1)
            gamma = 2.0 * zeta * np.sqrt(kn * masses[idx])
            fn = np.maximum(kn * delta - gamma * vn, 0.0)
            vt = np.sum(v_rel * tvec, axis=1)
            # tangential force from a one-step spring displacement (no memory ->
            # viscous Coulomb friction); magnitude capped by mu*F_n
            ft = -np.clip(kt * vt * dt, -mu * fn, mu * fn)
            np.add.at(force, idx, fn[:, None] * nvec + ft[:, None] * tvec)
            np.add.at(torque, idx, -radii[idx] * ft)
            np.add.at(n_contacts, idx, (fn > 0.0).astype(np.int64))
            np.add.at(torque, idx, -cfg.mu_rolling * fn * radii[idx] * np.tanh(spin[idx] / 0.5))
    return (force, torque, int(len(pairs)) if len(pairs) else 0, max_ov,
            n_contacts)


def simulate(cfg: DrumConfig, progress: bool = False) -> DrumRun:
    """Run one drum case: static settling followed by rotation."""
    rng = np.random.default_rng(cfg.seed)
    pos, radii, is_large = make_bed(cfg, rng)
    n = len(pos)

    area = math.pi * radii ** 2
    masses = np.where(is_large, cfg.rho_large, cfg.rho_small) * area * cfg.thickness
    inertia = 0.5 * masses * radii ** 2

    vel = np.zeros((n, 2))
    spin = np.zeros(n)
    zeta = coulomb_damping_ratio(cfg.restitution)

    dt = cfg.dt
    g_acc = np.array([0.0, -cfg.gravity])

    # ---- static settling (drum at rest) ------------------------------------
    # The RSA bed is loose; it is settled at omega = 0 with a kinetic-energy
    # quench until the rms particle speed drops below ``settle_speed`` (checked
    # every 500 steps, capped at 4x the nominal budget).  The number of steps
    # actually used is recorded so the report can prove the bed was at rest
    # before the rotation started.
    cfg_settle = cfg.with_(omega=0.0)
    settle_speed = float(cfg.settle_speed)
    max_settle = 3 * cfg.settle_steps
    settle_steps_used = 0
    rms_speed = float("inf")
    while settle_steps_used < max_settle:
        f, t, _, _, _ = _contact_forces(pos, vel, spin, radii, masses, inertia,
                                        cfg_settle, dt, zeta)
        vel += (f / masses[:, None] + g_acc) * dt
        # quasistatic bed preparation: strong local (viscous) damping, NOT
        # applied during the rotated phase.  With dt = 1e-4 s the residual
        # free-fall speed of an overdamped particle is g*dt/(1-damping) ~ 5e-3
        # m/s, i.e. the rest threshold is reachable only for a supported bed.
        vel *= cfg.settle_damping
        spin *= cfg.settle_damping
        pos += vel * dt
        spin += (t / inertia) * dt
        settle_steps_used += 1
        if settle_steps_used % 500 == 0:
            rms_speed = float(np.sqrt(np.mean(np.sum(vel ** 2, axis=1))))
            if rms_speed < settle_speed:
                break
    # Kinetic-energy quench rounds: zero the velocities and relax the contact
    # network quasistatically.  This removes the residual "jitter" of the
    # collapse.  The loop is converged (bounded by ``settle_max_quench_rounds``)
    # and the *measured* residual rms speed is reported, so a bed that never
    # reaches the rest criterion shows up as such instead of being asserted.
    quench_rounds = 0
    while quench_rounds < int(cfg.settle_max_quench_rounds):
        rms_speed = float(np.sqrt(np.mean(np.sum(vel ** 2, axis=1))))
        if rms_speed < settle_speed:
            break
        vel[:] = 0.0
        spin[:] = 0.0
        for _ in range(1000):
            f, t, _, _, _ = _contact_forces(pos, vel, spin, radii, masses,
                                            inertia, cfg_settle, dt, zeta)
            vel += (f / masses[:, None] + g_acc) * dt
            vel *= cfg.settle_damping
            spin *= cfg.settle_damping
            pos += vel * dt
            spin += (t / inertia) * dt
        settle_steps_used += 1000
        quench_rounds += 1
        rms_speed = float(np.sqrt(np.mean(np.sum(vel ** 2, axis=1))))
    initial_ke = float(0.5 * np.sum(masses * np.sum(vel ** 2, axis=1)))

    # ---- measurement series ------------------------------------------------
    n_samples = max(2, int(round(cfg.revs / cfg.sample_revs)) + 1)
    sample_every = max(1, cfg.n_steps // n_samples)
    rev_per_step = (cfg.omega / (2.0 * math.pi)) * dt if cfg.omega > 0 else 0.0

    series = {
        "rev": [], "segregation": [], "core_shell_excess": [], "size_pos_corr": [],
        "phi_large_outer": [], "phi_large_inner": [],
        "mean_radius_large": [], "mean_radius_small": [],
        "kinetic_energy": [], "max_overlap_ratio": [],
        # measured kinematics (m15649): the *observables* behind any regime
        # claim, sampled along with the segregation metrics
        "wall_velocity_ratio": [],       # <v_theta>_wall / (omega R)
        "wall_contact_arc_upper_deg": [],  # upper end of the bed/wall contact arc
        "airborne_fraction": [],         # fraction of particles with n_contacts == 0
        "upper_flight_fraction": [],     # ... and above the drum centre (ballistic)
        "r_p05_over_R": [],              # 5th percentile of r/R (annulus indicator)
        "bed_omega_over_wall": [],       # load angular velocity / wall angular velocity
    }
    snapshots = []

    t_sim = 0.0
    n_contacts = np.zeros(n, dtype=np.int64)
    for step in range(cfg.n_steps):
        f, t, npairs, max_ov, n_contacts = _contact_forces(
            pos, vel, spin, radii, masses, inertia, cfg, dt, zeta)
        acc = f / masses[:, None] + g_acc
        vel += acc * dt
        pos += vel * dt
        spin += (t / inertia) * dt

        if step % sample_every == 0 or step == cfg.n_steps - 1:
            r_pos = np.linalg.norm(pos, axis=1)
            prof = _m.radial_profile(pos, radii, is_large, cfg.radius, cfg.n_bins)
            s = _m.concentration_segregation(prof)
            excess = _m.core_shell_excess(prof)
            series["rev"].append((step * dt) * (cfg.omega / (2.0 * math.pi)))
            series["segregation"].append(s)
            series["core_shell_excess"].append(excess)
            series["size_pos_corr"].append(
                _m.size_position_correlation(radii, r_pos / cfg.radius))
            series["phi_large_outer"].append(_mean(prof["phi"][prof["centres"] > 0.5]))
            series["phi_large_inner"].append(_mean(prof["phi"][prof["centres"] < 0.5]))
            series["mean_radius_large"].append(float(r_pos[is_large].mean() / cfg.radius))
            series["mean_radius_small"].append(float(r_pos[~is_large].mean() / cfg.radius))
            ke = 0.5 * float(np.sum(masses * np.sum(vel ** 2, axis=1)))
            series["kinetic_energy"].append(ke)
            series["max_overlap_ratio"].append(max_ov)
            # ---- measured kinematics -------------------------------------
            r_norm = r_pos / cfg.radius
            r_hat = pos / np.maximum(r_pos, 1e-12)[:, None]
            t_hat = np.stack([-r_hat[:, 1], r_hat[:, 0]], axis=1)
            v_theta = np.sum(vel * t_hat, axis=1)
            wall_layer = r_pos > cfg.radius - 2.5 * radii
            if wall_layer.any():
                if cfg.omega > 0.0:
                    series["wall_velocity_ratio"].append(
                        float(np.mean(v_theta[wall_layer])
                              / (cfg.omega * cfg.radius)))
                else:
                    series["wall_velocity_ratio"].append(float("nan"))
                theta_wall = np.degrees(
                    np.arctan2(pos[wall_layer, 0], -pos[wall_layer, 1]))
                series["wall_contact_arc_upper_deg"].append(
                    float(np.percentile(theta_wall, 95.0)))
            else:
                series["wall_velocity_ratio"].append(float("nan"))
                series["wall_contact_arc_upper_deg"].append(float("nan"))
            free = n_contacts == 0
            series["airborne_fraction"].append(float(np.mean(free)))
            series["upper_flight_fraction"].append(
                float(np.mean(free & (pos[:, 1] > 0.15 * cfg.radius))))
            series["r_p05_over_R"].append(
                float(np.percentile(r_norm, 5.0)))
            i_bed = float(np.sum(masses * r_pos ** 2))
            if i_bed > 0.0 and cfg.omega > 0.0:
                series["bed_omega_over_wall"].append(
                    float(np.sum(masses * v_theta * r_pos) / i_bed / cfg.omega))
            else:
                series["bed_omega_over_wall"].append(float("nan"))
            snapshots.append({"rev": float(series["rev"][-1]),
                              "pos": pos.copy(), "phi": prof["phi"].copy()})
        t_sim += dt

    final_r = np.linalg.norm(pos, axis=1) / cfg.radius
    actual_fill = float(area.sum() / (math.pi * cfg.radius ** 2))
    actual_large = float(area[is_large].sum() / area.sum())

    final_prof = _m.radial_profile(pos, radii, is_large, cfg.radius, cfg.n_bins)
    diag = {
        "profile": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in final_prof.items()},
        "froude": cfg.froude,
        "n_pairs_last": npairs,
        "max_overlap_ratio": max(float(np.max(series["max_overlap_ratio"]))
                                 if series["max_overlap_ratio"] else 0.0, 0.0),
        "ke_final": float(series["kinetic_energy"][-1]) if series["kinetic_energy"] else 0.0,
        "settle_steps_used": settle_steps_used,
        "quench_rounds": quench_rounds,
        "settle_rms_speed": rms_speed,
        "settle_speed_threshold": float(cfg.settle_speed),
        "resting_ok": bool(rms_speed < float(cfg.settle_speed)),
        "initial_ke": initial_ke,
        "dt": cfg.dt,
        "steps": cfg.n_steps,
        "physical_time": cfg.n_steps * cfg.dt,
        "mean_radius_large_final": float(final_r[is_large].mean()),
        "mean_radius_small_final": float(final_r[~is_large].mean()),
        # time-averaged measured kinematics over the whole rotation window
        "wall_velocity_ratio_mean": _mean(series["wall_velocity_ratio"]),
        "wall_contact_arc_upper_deg_mean": _mean(series["wall_contact_arc_upper_deg"]),
        "airborne_fraction_mean": _mean(series["airborne_fraction"]),
        "upper_flight_fraction_mean": _mean(series["upper_flight_fraction"]),
        "r_p05_over_R_mean": _mean(series["r_p05_over_R"]),
        "bed_omega_over_wall_mean": _mean(series["bed_omega_over_wall"]),
    }

    # free-surface (dynamic repose) slope: topmost particle per x-column
    x_edges = np.linspace(-cfg.radius, cfg.radius, 9)
    xs, ys = [], []
    for a, b in zip(x_edges[:-1], x_edges[1:]):
        m = (pos[:, 0] >= a) & (pos[:, 0] < b)
        if m.sum() >= 3:
            top = pos[m, 1].max()
            xs.append(0.5 * (a + b))
            ys.append(top)
    slope_deg = float("nan")
    if len(xs) >= 3:
        coef = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
        slope_deg = float(math.degrees(math.atan(coef[0])))
    r_hat = np.stack([-pos[:, 1], pos[:, 0]], axis=1) / np.maximum(
        np.linalg.norm(pos, axis=1), 1e-12)[:, None]
    v_theta = np.sum(vel * r_hat, axis=1)
    diag["surface_slope_deg"] = slope_deg
    diag["mean_azimuthal_speed"] = float(np.mean(v_theta))
    diag["rms_azimuthal_speed"] = float(np.sqrt(np.mean(v_theta ** 2)))

    return DrumRun(
        config=cfg, pos=pos, radii=radii, is_large=is_large,
        masses=masses, vel=vel, spin=spin, actual_fill=actual_fill,
        actual_large_area_fraction=actual_large,
        n_large=int(is_large.sum()), n_small=int((~is_large).sum()),
        snapshots=snapshots, series={k: np.asarray(v, float) for k, v in series.items()},
        diagnostics=diag,
    )


# ---------------------------------------------------------------------------
# small internal helpers (also used by tests)
# ---------------------------------------------------------------------------
def _mean(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _radial_phi(r_norm, area, is_large, bins):
    """Large-species area fraction per radial bin (NaN for empty bins)."""
    phi = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        m = (r_norm >= bins[b]) & (r_norm < bins[b + 1])
        if m.any():
            tot = area[m].sum()
            if tot > 0:
                phi[b] = area[m & is_large].sum() / tot
    return phi


def _segregation(phi, bins, phi0):
    """Area-weighted mean absolute deviation S and core-shell excess."""
    valid = np.isfinite(phi)
    if not valid.any() or phi0 <= 0:
        return 0.0, 0.0
    # weight by the geometric area of the annulus (outer bins carry more material)
    r_out, r_in = bins[1:], bins[:-1]
    w = (r_out ** 2 - r_in ** 2)[valid]
    w = w / w.sum()
    s = float(np.sum(w * np.abs(phi[valid] - phi0)) / phi0)
    inner = valid & (bins[:-1] < 0.5)
    outer = valid & (bins[1:] > 0.5)
    excess = 0.0
    if inner.any() and outer.any():
        excess = float(np.mean(phi[outer]) - np.mean(phi[inner]))
    return s, excess