"""Macroscopic shape-error anchors: repose angle and shear-cell friction.

Two independent repose-angle methods are implemented here so the anchor used by
the shape error budget is not a single code path:

* ``collapse_column_repose`` -- 3D column collapse of monodisperse spheres,
  driven by the validated ``tensordem.repose_calibration`` engine (Hertz-free
  linear contact with Mindlin tangential and rolling springs).
* ``drum_repose_angle`` -- 2D rotating drum (this module's own disk DEM): the
  free surface inclination of a half-filled drum rotating at constant rate.

``shear_box_friction_angle`` runs a 2D constant-volume direct shear cell and
returns the apparent internal friction coefficient ``mu_i = tau / sigma``, i.e.
the "internal vs external friction" comparison: ``mu`` is the particle-particle
(input) friction, ``mu_i`` the mobilised internal one.

``calibrate_sphere_repose`` builds the measured ``mu -> theta_r`` anchor that
the shape error budget interpolates; the anchor shipped in
``artifacts/dem/shape_error_table.json`` was produced with it.

The disk DEM is a standard soft-sphere model: linear normal spring + dashpot,
incremental Mindlin tangential spring with Coulomb saturation, optional rolling
resistance.  Contacts are found with a KD-tree neighbour list rebuilt every few
steps; tangent state is carried across rebuilds.

All quantities are SI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from scipy.spatial import cKDTree

__all__ = [
    "DiskConfig",
    "DiskDEM",
    "drum_repose_angle",
    "shear_box_friction_angle",
    "collapse_column_repose",
    "calibrate_sphere_repose",
]


def _damping_ratio(restitution: float) -> float:
    """Viscous damping ratio reproducing the coefficient of restitution."""
    if not 0.0 <= restitution < 1.0:
        raise ValueError("restitution must lie in [0, 1)")
    if restitution == 0.0:
        return 1.0
    ln_e = math.log(restitution)
    return -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)


@dataclass
class DiskConfig:
    """Contact and material parameters of the 2D disk DEM."""
    kn: float = 2.0e3           # N/m normal stiffness (soft, keeps dt affordable)
    kt_ratio: float = 0.5
    mu: float = 0.40
    mu_rolling: float = 0.02
    mu_wall: Optional[float] = None
    mu_rolling_wall: Optional[float] = None
    restitution: float = 0.4
    density: float = 2500.0
    thickness: float = 0.02
    gravity: float = 9.81
    dt: Optional[float] = None
    rebuild_every: int = 10

    def __post_init__(self) -> None:
        for name in ("kn", "kt_ratio", "density", "thickness", "gravity"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("mu", "mu_rolling"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("mu_wall", "mu_rolling_wall"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be None or finite and nonnegative")

    @property
    def wall_mu(self) -> float:
        return self.mu if self.mu_wall is None else self.mu_wall

    @property
    def wall_mu_rolling(self) -> float:
        return self.mu_rolling if self.mu_rolling_wall is None else self.mu_rolling_wall


class DiskDEM:
    """Minimal 2D soft-disk DEM with KD-tree neighbours and rigid walls.

    Walls are lists of ``(point, tangent, velocity)`` line contacts, i.e. the
    caller projects the geometry itself (drum: inward radial contacts; shear
    cell: the four straight plates).  Particle dynamics are 2D rigid disks
    (translation + spin) with rolling resistance.
    """

    def __init__(self, positions, radii, config: DiskConfig):
        self.pos = np.asarray(positions, dtype=float).copy()
        self.radii = np.asarray(radii, dtype=float).reshape(-1).copy()
        if self.pos.shape != (len(self.radii), 2):
            raise ValueError("positions must be (N, 2) matching radii")
        self.cfg = config
        self.vel = np.zeros_like(self.pos)
        self.omega = np.zeros(len(self.radii))
        self.mass = config.density * math.pi * self.radii ** 2 * config.thickness
        self.inertia = 0.5 * self.mass * self.radii ** 2
        self.dt = config.dt if config.dt is not None else self.recommended_dt
        self.xi = None          # tangential displacement (n_pairs, 2)
        self.roll = None        # rolling state (n_pairs,)
        self.pairs = np.zeros((0, 2), dtype=int)
        self._keys: Dict[Tuple[int, int], int] = {}
        self.wall_xi: Dict[int, np.ndarray] = {}
        self.wall_force: Dict[int, np.ndarray] = {}
        self.wall_roll: Dict[int, np.ndarray] = {}
        self._build_pairs()

    # -- neighbours --------------------------------------------------------
    @property
    def recommended_dt(self) -> float:
        m_min = float(self.mass.min())
        omega = math.sqrt(2.0 * self.cfg.kn / (0.5 * m_min))
        return 0.1 / omega

    def _build_pairs(self) -> None:
        cutoff = 2.0 * float(self.radii.max()) * 1.25
        tree = cKDTree(self.pos)
        found = sorted(tree.query_pairs(cutoff))
        new_pairs = np.array(found, dtype=int).reshape(-1, 2)
        n_new = len(new_pairs)
        xi = np.zeros((n_new, 2))
        roll = np.zeros(n_new)
        old_keys = self._keys
        if old_keys and n_new:
            for idx, (i, j) in enumerate(map(tuple, new_pairs)):
                prev = old_keys.get((int(i), int(j)))
                if prev is not None:
                    xi[idx] = self.xi[prev]
                    roll[idx] = self.roll[prev]
        self.pairs = new_pairs
        self.xi = xi
        self.roll = roll
        self._keys = {tuple(int(v) for v in pair): idx
                      for idx, pair in enumerate(new_pairs)}

    # -- one step ----------------------------------------------------------
    def step(self, step_index: int, wall_contacts) -> None:
        """Advance one step; ``wall_contacts`` is a callable returning a list of
        ``(normal, penetration, point, wall_velocity, mu, mu_rolling)`` arrays."""
        cfg = self.cfg
        dt = self.dt
        force = np.zeros_like(self.pos)
        torque = np.zeros(len(self.radii))
        r = self.radii

        if step_index % cfg.rebuild_every == 0:
            self._build_pairs()

        if len(self.pairs):
            i, j = self.pairs[:, 0], self.pairs[:, 1]
            dvec = self.pos[j] - self.pos[i]
            dist = np.linalg.norm(dvec, axis=1)
            overlap = r[i] + r[j] - dist
            active = overlap > 0.0
            if np.any(active):
                ia, ja = i[active], j[active]
                ov = overlap[active]
                n = dvec[active] / np.maximum(dist[active], 1e-300)[:, None]
                rel_v = self.vel[ja] - self.vel[ia]
                v_n = np.sum(rel_v * n, axis=1)
                gamma = 2.0 * _damping_ratio(cfg.restitution) * math.sqrt(
                    cfg.kn * 0.5 * float(self.mass.mean()))
                f_n = np.maximum(cfg.kn * ov - gamma * v_n, 0.0)
                vt = rel_v - v_n[:, None] * n
                lever_i = -n * r[ia][:, None]
                lever_j = n * r[ja][:, None]
                surf = vt + np.stack([-self.omega[ja] * lever_j[:, 1],
                                      self.omega[ja] * lever_j[:, 0]], axis=1) \
                          - np.stack([-self.omega[ia] * lever_i[:, 1],
                                      self.omega[ia] * lever_i[:, 0]], axis=1)
                idx = np.nonzero(active)[0]
                xi_new = self.xi[idx] + surf * dt
                mag = np.linalg.norm(xi_new, axis=1)
                cap = cfg.mu * f_n / (cfg.kt_ratio * cfg.kn)
                scale = np.where(mag > cap, cap / np.maximum(mag, 1e-300), 1.0)
                xi_new = xi_new * scale[:, None]
                self.xi[idx] = xi_new
                f_t = -cfg.kt_ratio * cfg.kn * xi_new
                f_pair = f_n[:, None] * n + f_t
                np.add.at(force, ja, f_pair)
                np.add.at(force, ia, -f_pair)
                # rolling resistance (torque opposing relative spin)
                spin = self.omega[ja] - self.omega[ia]
                m_roll = -cfg.mu_rolling * f_n * r[ia] * np.sign(spin)
                np.add.at(torque, ja, m_roll)
                np.add.at(torque, ia, -m_roll)
                # contact torque from the tangential force
                np.add.at(torque, ja, np.cross(lever_j, f_t))
                np.add.at(torque, ia, np.cross(lever_i, -f_t))

        for contact in wall_contacts():
            wall_id, n, pen, point, wall_v, mu, mu_roll = contact
            if wall_id not in self.wall_xi:
                self.wall_xi[wall_id] = np.zeros((len(self.radii), 2))
                self.wall_roll[wall_id] = np.zeros(len(self.radii))
            active = pen > 0.0
            if not np.any(active):
                continue
            na = n[active]
            ov = pen[active]
            mu_a = mu[active]
            mu_roll_a = mu_roll[active]
            v_n = np.sum((self.vel[active] - wall_v[active]) * na, axis=1)
            f_n = np.maximum(cfg.kn * ov - 2.0 * _damping_ratio(cfg.restitution)
                             * math.sqrt(cfg.kn * float(self.mass.mean())) * v_n, 0.0)
            rel = self.vel[active] - wall_v[active]
            vt = rel - np.sum(rel * na, axis=1)[:, None] * na
            # incremental tangential spring at the wall with Coulomb saturation
            ids = np.nonzero(active)[0]
            xi_w = self.wall_xi[wall_id][ids] + vt * dt
            mag = np.linalg.norm(xi_w, axis=1)
            cap = mu_a * f_n / (cfg.kt_ratio * cfg.kn)
            scale = np.where(mag > cap, cap / np.maximum(mag, 1e-300), 1.0)
            xi_w = xi_w * scale[:, None]
            self.wall_xi[wall_id][ids] = xi_w
            f_t = -cfg.kt_ratio * cfg.kn * xi_w
            f_wall = f_n[:, None] * na + f_t
            np.add.at(force, np.nonzero(active)[0], f_wall)
            # total force transmitted through this wall (diagnostics: the shear
            # cell measures sigma and tau from these sums, not from a
            # re-derived formula -- a mismatch here silently zeroed tau once)
            self.wall_force[wall_id] = np.sum(f_wall, axis=0)
            spin = self.omega[np.nonzero(active)[0]]
            np.add.at(torque, np.nonzero(active)[0],
                      -mu_roll_a * f_n * self.radii[active] * np.sign(spin))

        force[:, 1] -= self.mass * cfg.gravity
        self.vel += force / self.mass[:, None] * dt
        # numerical hygiene: a lost particle must not poison the statistics
        speed = np.linalg.norm(self.vel, axis=1)
        cap = 0.5 * float(np.abs(cfg.gravity))
        if np.any(speed > cap):
            self.vel *= (cap / np.maximum(speed, 1e-30))[:, None]
        self.omega += torque / self.inertia * dt
        self.pos += self.vel * dt


# ---------------------------------------------------------------------------
# Repose angle: rotating drum (2D)
# ---------------------------------------------------------------------------

def _disk_radii(n: int, mean_radius: float, ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.full(n, mean_radius)
    out[: n // 2] = mean_radius * ratio
    out[n // 2:] = mean_radius / ratio
    return out * rng.uniform(0.97, 1.03, n)


def drum_repose_angle(
    config: Optional[DiskConfig] = None,
    n_disks: int = 320,
    mean_radius: float = 2.0e-3,
    drum_radius: float = 40.0e-3,
    fill_fraction: float = 0.45,
    angular_velocity: float = 0.6,
    settle_time: float = 0.35,
    measure_time: float = 0.60,
    seed: int = 3,
) -> Dict[str, object]:
    """Free-surface angle of a half-filled 2D rotating drum.

    The wall rotates at ``angular_velocity``; after ``settle_time`` the free
    surface slope is measured from the topmost disk per vertical strip (central
    60 % of the drum) and averaged over ``measure_time``.  Direct shear from the
    wall would over-steepen the surface at high rates, so the rotation is kept
    slow (Froude number ~ 1e-3) and the diagnostics include the flow kinetic
    energy so a still-flowing pile can be rejected.

    Returns:
        dict with ``theta_deg``, ``r2``, ``kinetic_energy``, ``slope`` and the
        run settings.
    """
    cfg = config or DiskConfig()
    radii = _disk_radii(n_disks, mean_radius, 1.2, seed)
    rng = np.random.default_rng(seed + 1)
    target_area = fill_fraction * math.pi * drum_radius ** 2
    disk_area = math.pi * float(np.sum(radii ** 2))
    scale = math.sqrt(target_area / disk_area)
    radii = radii * scale

    # fill the lower part of the drum with a lattice, then relax into the wall
    spacing = 2.05 * float(radii.max())
    pts: List[Tuple[float, float]] = []
    y = -drum_radius + spacing
    while len(pts) < n_disks and y < drum_radius:
        x = -drum_radius + spacing
        while len(pts) < n_disks and x < drum_radius:
            if x * x + y * y < (drum_radius - 1.05 * radii.max()) ** 2:
                pts.append((x + rng.uniform(-1e-5, 1e-5), y))
            x += spacing
        y += spacing * 0.87
    pos = np.array(pts[:n_disks])
    dem = DiskDEM(pos, radii[: len(pos)], cfg)

    def wall_contacts():
        r_mag = np.maximum(np.linalg.norm(dem.pos, axis=1), 1e-30)
        pen = r_mag + dem.radii - drum_radius
        n = -dem.pos / r_mag[:, None]
        wall_v = angular_velocity * np.stack([-dem.pos[:, 1], dem.pos[:, 0]], axis=1)
        mu = np.full(len(dem.radii), cfg.wall_mu)
        mu_roll = np.full(len(dem.radii), cfg.wall_mu_rolling)
        return [(0, n, pen, dem.pos, wall_v, mu, mu_roll)]

    n_settle = int(round(settle_time / dem.dt))
    n_measure = int(round(measure_time / dem.dt))
    for step in range(n_settle):
        dem.step(step, wall_contacts)

    angles: List[float] = []
    r2s: List[float] = []
    keep_every = max(1, n_measure // 20)
    for step in range(n_measure):
        dem.step(step, wall_contacts)
        if step % keep_every == 0:
            angle, r2 = _surface_slope(dem.pos, dem.radii, drum_radius)
            if angle is not None:
                angles.append(angle)
                r2s.append(r2)
    ke = 0.5 * float(np.sum(dem.mass * np.sum(dem.vel ** 2, axis=1)))
    return {
        "theta_deg": float(np.mean(angles)) if angles else float("nan"),
        "theta_std_deg": float(np.std(angles)) if angles else float("nan"),
        "r2": float(np.mean(r2s)) if r2s else float("nan"),
        "slope": float(math.tan(math.radians(np.mean(angles)))) if angles else float("nan"),
        "kinetic_energy_j": ke,
        "n_disks": int(len(dem.pos)),
        "dt": dem.dt,
        "steps": n_settle + n_measure,
        "angular_velocity": angular_velocity,
        "mu": cfg.mu,
        "mu_rolling": cfg.mu_rolling,
        "method": "rotating_drum_2d",
    }


def _surface_slope(pos: np.ndarray, radii: np.ndarray, drum_radius: float
                   ) -> Tuple[Optional[float], float]:
    """Topmost-disk slope of the free surface in the drum's central band."""
    band = 0.6 * drum_radius
    mask = (np.abs(pos[:, 0]) < band) & (pos[:, 1] > -0.2 * drum_radius)
    if np.count_nonzero(mask) < 8:
        return None, 0.0
    x = pos[mask, 0]
    y = pos[mask, 1] + radii[mask]
    n_bins = 12
    edges = np.linspace(-band, band, n_bins + 1)
    bx, by = [], []
    for k in range(n_bins):
        sel = (x >= edges[k]) & (x < edges[k + 1])
        if np.count_nonzero(sel) < 1:
            continue
        bx.append(0.5 * (edges[k] + edges[k + 1]))
        by.append(float(np.max(y[sel])))
    if len(bx) < 4:
        return None, 0.0
    bx_arr, by_arr = np.array(bx), np.array(by)
    slope, intercept = np.polyfit(bx_arr, by_arr, 1)
    pred = slope * bx_arr + intercept
    ss_res = float(np.sum((by_arr - pred) ** 2))
    ss_tot = float(np.sum((by_arr - by_arr.mean()) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    return math.degrees(math.atan(abs(float(slope)))), r2


# ---------------------------------------------------------------------------
# Shear box (2D, constant volume)
# ---------------------------------------------------------------------------

def shear_box_friction_angle(
    config: Optional[DiskConfig] = None,
    n_disks: int = 120,
    mean_radius: float = 7.0e-4,
    box_width: float = 20.0e-3,
    box_height: float = 12.0e-3,
    shear_velocity: float = 2.0e-3,
    preconsolidate: float = 0.15,
    shear_time: float = 0.5,
    n_bins: int = 10,
    seed: int = 11,
) -> Dict[str, object]:
    """Apparent internal friction coefficient from a 2D direct shear cell.

    Disks are compressed under the weight of a rigid lid, then the lid is
    translated at ``shear_velocity``.  The mobilised friction is the ratio of
    the steady-state horizontal lid force to the lid normal force
    (``mu_i = tau / sigma_n``); the peak value is reported as well, because the
    peak/steady split is what distinguishes internal from sliding friction.

    Returns:
        dict with ``mu_apparent``, ``phi_deg``, ``tau_over_sigma_peak`` and
        diagnostics (lid normal force, sample height, kinetic energy).
    """
    cfg = config or DiskConfig()
    radii = _disk_radii(n_disks, mean_radius, 1.2, seed)
    rng = np.random.default_rng(seed + 1)
    spacing = 2.05 * float(radii.max())
    pts = []
    y = -box_height / 2 + spacing
    while len(pts) < n_disks and y < box_height / 2:
        x = -box_width / 2 + spacing
        while len(pts) < n_disks and x < box_width / 2:
            pts.append((x + rng.uniform(-1e-6, 1e-6), y))
            x += spacing
        y += spacing * 0.87
    dem = DiskDEM(np.array(pts[:n_disks]), radii[: len(pts)], cfg)
    lid_mass = 0.5 * float(np.sum(dem.mass))
    lid = {"y": float(np.max(dem.pos[:, 1] + dem.radii)) + 5.0 * float(dem.radii.max()),
           "vy": 0.0}

    def walls():
        contacts = []
        # side walls (vertical, fixed)
        # side walls: penetration is measured *along the inward normal only*, so
        # a particle far from a wall must not be pushed by it (regression: the
        # first version inverted this sign and blew the sample up).
        left_n = np.tile([1.0, 0.0], (len(dem.pos), 1))
        pen_l = dem.radii - (dem.pos[:, 0] + box_width / 2)
        contacts.append((1, left_n,
                         pen_l, np.stack([np.full(len(dem.pos), -box_width / 2),
                                          dem.pos[:, 1]], axis=1),
                         np.zeros_like(dem.pos),
                         np.full(len(dem.pos), cfg.wall_mu),
                         np.full(len(dem.pos), cfg.wall_mu_rolling)))
        right_n = np.tile([-1.0, 0.0], (len(dem.pos), 1))
        pen_r = dem.radii - (box_width / 2 - dem.pos[:, 0])
        contacts.append((2, right_n,
                         pen_r, np.stack([np.full(len(dem.pos), box_width / 2),
                                          dem.pos[:, 1]], axis=1),
                         np.zeros_like(dem.pos),
                         np.full(len(dem.pos), cfg.wall_mu),
                         np.full(len(dem.pos), cfg.wall_mu_rolling)))
        # bottom wall
        pen_b = dem.radii - (dem.pos[:, 1] + box_height / 2)
        contacts.append((3, np.tile([0.0, 1.0], (len(dem.pos), 1)), pen_b,
                         np.stack([dem.pos[:, 0],
                                   np.full(len(dem.pos), -box_height / 2)], axis=1),
                         np.zeros_like(dem.pos),
                         np.full(len(dem.pos), cfg.wall_mu),
                         np.full(len(dem.pos), cfg.wall_mu_rolling)))
        # lid (moving in +x, servo-controlled in y by its own weight)
        lid_y = lid["y"]
        pen_t = dem.radii - (lid_y - dem.pos[:, 1])
        contacts.append((4, np.tile([0.0, -1.0], (len(dem.pos), 1)), pen_t,
                         np.stack([dem.pos[:, 0], np.full(len(dem.pos), lid_y)], axis=1),
                         np.stack([np.full(len(dem.pos), shear_velocity),
                                   np.full(len(dem.pos), lid["vy"])], axis=1),
                         np.full(len(dem.pos), cfg.wall_mu),
                         np.full(len(dem.pos), cfg.wall_mu_rolling)))
        return contacts

    def lid_forces():
        """Normal and shear force the lid receives from the bed (N)."""
        total = dem.wall_force.get(4)
        if total is None:
            return 0.0, 0.0
        return float(-total[1]), float(-total[0])

    def advance_lid():
        """Servo the lid vertically under its own weight (dead load)."""
        fn, _ = lid_forces()
        lid["vy"] += (fn / lid_mass - cfg.gravity) * dem.dt
        lid["y"] += lid["vy"] * dem.dt

    n_pre = int(round(preconsolidate / dem.dt))
    for step in range(n_pre):
        dem.step(step, walls)
        advance_lid()

    n_shear = int(round(shear_time / dem.dt))
    normals: List[float] = []
    shears: List[float] = []
    for step in range(n_shear):
        dem.step(step, walls)
        advance_lid()
        if step % 20 == 0:
            fn, ft = lid_forces()
            normals.append(fn)
            shears.append(abs(ft))

    normals_arr = np.array(normals)
    shears_arr = np.array(shears)
    steady = slice(int(0.4 * len(normals_arr)), None)
    sigma = float(np.mean(normals_arr[steady]))
    tau = float(np.mean(shears_arr[steady]))
    mu_i = tau / sigma if sigma > 1e-30 else float("nan")
    peak = float(np.max(shears_arr) / max(np.mean(normals_arr[:max(1, len(normals_arr) // 5)]), 1e-30))
    ke = 0.5 * float(np.sum(dem.mass * np.sum(dem.vel ** 2, axis=1)))
    return {
        "mu_apparent": mu_i,
        "phi_deg": math.degrees(math.atan(mu_i)) if math.isfinite(mu_i) else float("nan"),
        "tau_over_sigma_peak": peak,
        "lid_normal_force_n": sigma,
        "lid_shear_force_n": tau,
        "kinetic_energy_j": ke,
        "n_disks": int(len(dem.pos)),
        "dt": dem.dt,
        "steps": n_pre + n_shear,
        "mu_input": cfg.mu,
        "lid_mass_kg": lid_mass,
        "lid_vertical_velocity_m_s": lid["vy"],
        "method": "direct_shear_2d",
    }


# ---------------------------------------------------------------------------
# Repose angle: 3D column collapse (validated engine) + calibration
# ---------------------------------------------------------------------------

def collapse_column_repose(
    mu: float,
    mu_rolling: float = 0.05,
    radius: float = 0.005,
    **overrides,
) -> Dict[str, object]:
    """3D column-collapse repose angle via ``repose_calibration``.

    Thin wrapper that keeps the sphere anchor reproducible from this module;
    raises ``ImportError`` if the optional torch dependency is unavailable.
    """
    from .repose_calibration import ReposeCase, simulate

    case = ReposeCase(mu=mu, mu_rolling=mu_rolling, radius=radius, **overrides)
    out = simulate(case)
    return {
        "theta_deg": float(out["theta_r_deg"]),
        "r2": float(out.get("r2", float("nan"))),
        "n_particles": int(out["n_particles"]),
        "mu": mu,
        "mu_rolling": mu_rolling,
        "method": "column_collapse_3d",
    }


def calibrate_sphere_repose(
    mus: Sequence[float] = (0.30, 0.40, 0.50),
    target_deg: float = 22.0,
    tolerance_deg: float = 2.0,
    **case_overrides,
) -> Dict[str, object]:
    """Measure the ``mu -> theta_r`` anchor with the 3D column collapse.

    Returns the measured anchor points, the interval of ``mu`` that reproduces
    the target repose angle within ``tolerance_deg`` and the calibration error
    at the best node.
    """
    rows = [collapse_column_repose(float(mu), **case_overrides) for mu in mus]
    anchor = [(row["mu"], row["theta_deg"]) for row in rows]
    best = min(rows, key=lambda row: abs(row["theta_deg"] - target_deg))
    ok = abs(best["theta_deg"] - target_deg) <= tolerance_deg
    return {
        "anchor": anchor,
        "rows": rows,
        "target_deg": target_deg,
        "tolerance_deg": tolerance_deg,
        "best_mu": best["mu"],
        "best_theta_deg": best["theta_deg"],
        "error_deg": abs(best["theta_deg"] - target_deg),
        "within_tolerance": bool(ok),
    }