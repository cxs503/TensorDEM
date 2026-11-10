"""Vectorised soft-sphere DEM of ice-rubble rearrangement in a ridge keel.

Why this module exists
----------------------
The pre-refactor :mod:`.consolidation` integrated the contact loop in pure
Python over ``query_pairs``; at the documented default scale (1000 blocks,
500 s at dt = 1 ms = 5e5 steps) that is hours of wall clock, so the documented
"production" case never ran -- only 15-60 particle toy cases in the unit tests.
This module is a same-physics, array-based re-implementation so the documented
keel case is actually computable, plus a porosity estimator that can be
validated against a packing whose porosity is known by construction.

Physics (granular skeleton of angular ice blocks, modelled as rigid spheres)
---------------------------------------------------------------------------
* normal contact: linear spring-dashpot ``F_n = max(k_n*delta - gamma*v_n, 0)``
  with ``gamma`` from the restitution coefficient and the pair's reduced mass;
* tangential contact: regularised Coulomb
  ``F_t = -min(mu*F_n, k_t*|v_t|*dt) * v_t/|v_t|``.  The incremental
  Cundall-Strack spring history is *not* tracked (documented limitation); the
  invariant that holds exactly is the friction cone ``|F_t| <= mu*F_n``;
* rolling resistance ``tau_r = -mu_r * r_eff * |F_n| * omega_rel/|omega_rel|``
  acting on the rotational degree of freedom (a sphere needs no orientation
  state, only spin: its contact geometry is rotation invariant);
* rigid container = the keel frustum of :class:`..geometry.KeelGeometry`
  (side walls, floor at ``depth``, ceiling at the waterline) as penalty
  springs of the same stiffness, plus Coulomb wall friction;
* body force: *net effective* gravity along +z (z points downward, 0 at the
  waterline).  ``gravity`` is the net (submerged) acceleration of the ice
  skeleton, see :func:`net_effective_gravity`;
* cyclic wave loading: horizontal shaking ``a_x = A sin(omega t)``,
  ``a_y = 0.3 A cos(0.7 omega t)`` -- a rigid-body shaking of the parcel that
  represents the wave orbital motion (documented idealisation: no depth
  attenuation, no pore-pressure coupling).

Porosity estimator and why two are reported
-------------------------------------------
Rigid equal spheres cannot pack below ``n = 0.2595`` (fcc) and random close
packing sits near ``n ~ 0.36``, so a keel cannot reach ``n = 0.2`` by
rearrangement of rigid spheres alone.  Hence both definitions below are
reported, and :mod:`.pore_ice` closes the remaining gap thermally.

``porosity_slab``
    void ratio of the bed *as a layer in the container*:
    ``n = 1 - V_solid / V_container(z_surface .. depth)`` with ``z_surface``
    the area-weighted mean free surface (oedometer / void-ratio definition).
``porosity_point``
    unbiased point (Monte-Carlo) estimator: fraction of container points
    between ``z_surface`` and ``depth`` not covered by any block.  The two
    agree to ``|dn| < 0.02`` for the initial packing, whose porosity is fixed
    by construction -- that agreement is the estimator's calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from .geometry import KeelGeometry

__all__ = [
    "DEMRubbleConfig",
    "DEMRubbleResult",
    "RubbleDEM",
    "bed_porosity",
    "net_effective_gravity",
    "container_volume",
    "ICE_DENSITY_KG_M3",
    "SEA_WATER_DENSITY_KG_M3",
]

ICE_DENSITY_KG_M3 = 917.0
SEA_WATER_DENSITY_KG_M3 = 1025.0


def net_effective_gravity(dry_gravity: float = 9.81,
                          rho_ice: float = ICE_DENSITY_KG_M3,
                          rho_water: float = SEA_WATER_DENSITY_KG_M3,
                          submerged: bool = True) -> float:
    """Net (submerged) acceleration acting on an ice block in the keel.

    ``g_eff = g * (1 - rho_water/rho_ice)`` for a fully submerged block.  For
    sea ice (917 kg/m3 in 1025 kg/m3 water) this is **negative** (~ -1.15
    m/s2, i.e. upward): a submerged ice block is buoyant, so the self-weight
    mechanism of a dry granular pile is not directly available to a keel
    skeleton unless the surrounding ridge structure supplies confining load.
    Kept as a first-class function because it is the physical reason a thermal
    (pore-ice) closure is needed on top of the mechanical one.
    """
    if not submerged:
        return dry_gravity
    return dry_gravity * (1.0 - rho_water / rho_ice)


@dataclass(frozen=True)
class DEMRubbleConfig:
    """Integration / contact parameters for the keel rubble DEM.

    All values are SI.  ``dt`` must satisfy the linear stability bound
    ``dt < 0.2*sqrt(m_min/k_n)``; :class:`RubbleDEM` checks it on construction.
    """

    mode: str = "cell"                # "cell" (periodic box) | "keel" (frustum walls)
    box_size: tuple[float, float] = (4.0, 4.0)   # periodic cell footprint (m)
    floor_depth: float = 2.75         # z of the rigid floor (m below the parcel top)
    dt: float = 1.0e-3
    kn: float = 2.0e4                 # N/m, normal stiffness (penalty,
                                      # ~3% overlap at the bed's own weight)
    kt_ratio: float = 0.5             # k_t / k_n
    friction: float = 0.70            # effective mu, rotation-locked angular blocks
    rotation: bool = False            # sphere spin DOF (off: see module note)
    rolling_friction: float = 0.05    # mu_r (only used when rotation=True)
    restitution: float = 0.25
    wall_friction: float = 0.30
    wall_restitution: float = 0.10
    damping_ratio: float = 0.10       # pore-water viscous drag, fraction of critical
    gravity: float = 9.81             # net effective gravity (+z = downward)
    wave_amplitude: float = 0.5       # m/s2, horizontal shaking
    wave_period: float = 10.0         # s
    wave_enabled: bool = True
    wave_start_time: float = 0.0       # s; waves switch on after the deposition stage
    total_time: float = 600.0         # s
    record_interval: float = 5.0      # s
    pair_refresh: int = 10            # steps between neighbour-list rebuilds
    porosity_points: int = 40000      # Monte-Carlo points per porosity record
    porosity_surface_bins: int = 24   # lateral bins for the free-surface estimate
    spin_damping: float = 1.0         # multiplier on the viscous spin drag
    seed: int = 42

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(self.kn) or self.kn <= 0:
            raise ValueError("kn must be finite and positive")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must be in [0, 1]")
        if self.total_time <= 0:
            raise ValueError("total_time must be positive")
        if self.record_interval <= 0:
            raise ValueError("record_interval must be positive")
        if self.mode not in ("cell", "keel"):
            raise ValueError("mode must be 'cell' or 'keel'")
        if self.mode == "cell":
            if min(self.box_size) <= 1.05 * 1.0:
                raise ValueError("box_size must be positive")
            if self.floor_depth <= 0:
                raise ValueError("floor_depth must be positive")


@dataclass
class DEMRubbleResult:
    """Time series and final state of a keel-rubble DEM run."""

    time: np.ndarray
    porosity_slab: np.ndarray
    porosity_point: np.ndarray
    surface_depth: np.ndarray
    kinetic_energy: np.ndarray
    elastic_energy: np.ndarray
    coordination: np.ndarray
    mean_spin: np.ndarray
    positions: np.ndarray
    radii: np.ndarray
    velocity: np.ndarray
    config: DEMRubbleConfig
    energy_ledger: dict

    @property
    def porosity(self) -> np.ndarray:
        """Primary record: oedometer void ratio (porosity_slab)."""
        return self.porosity_slab


def _container_half_widths(geom: KeelGeometry, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zc = np.clip(z, 0.0, geom.depth)
    frac = zc / geom.depth
    half_l = 0.5 * (geom.L_top - (geom.L_top - geom.L_bot) * frac)
    half_w = 0.5 * (geom.W_top - (geom.W_top - geom.W_bot) * frac)
    return half_l, half_w


def container_volume(geom: KeelGeometry, z1: float, z2: float) -> float:
    """Exact volume of the keel frustum between depths ``z1 < z2`` (analytic)."""
    z1 = max(0.0, float(z1))
    z2 = min(geom.depth, float(z2))
    if z2 <= z1:
        return 0.0
    a = (geom.L_top - geom.L_bot) / geom.depth
    b = (geom.W_top - geom.W_bot) / geom.depth

    def _primitive(z: float) -> float:
        return (geom.L_top * geom.W_top * z
                - 0.5 * (geom.L_top * b + geom.W_top * a) * z ** 2
                + a * b * z ** 3 / 3.0)

    return _primitive(z2) - _primitive(z1)


def _surface_depth(pos: np.ndarray, radii: np.ndarray, geom: KeelGeometry,
                   n_bins: int = 24, threshold: float = 0.25) -> float:
    """Area-weighted mean free-surface depth (top of the bed).

    The lateral domain is binned; in each bin the ``threshold`` quantile of the
    shallowest block surfaces (``z - r``) is taken as the column level, and the
    columns are averaged with equal weight (the columns have equal area).  The
    quantile makes the level insensitive to one block sitting on the pile.
    """
    if len(pos) == 0:
        return 0.0
    top = pos[:, 2] - radii
    # only the upper half of the bed can define the free surface
    cut = float(np.quantile(top, 0.6))
    sel = top <= cut
    xs, ys, ts = pos[sel, 0], pos[sel, 1], top[sel]
    edges_x = np.linspace(-geom.L_top / 2, geom.L_top / 2, n_bins + 1)
    edges_y = np.linspace(-geom.W_top / 2, geom.W_top / 2, n_bins + 1)
    ix = np.clip(np.digitize(xs, edges_x) - 1, 0, n_bins - 1)
    iy = np.clip(np.digitize(ys, edges_y) - 1, 0, n_bins - 1)
    flat = ix * n_bins + iy
    levels = []
    for cell in np.unique(flat):
        ts_cell = ts[flat == cell]
        levels.append(float(np.quantile(ts_cell, threshold)))
    levels = np.asarray(levels)
    med = float(np.median(levels))
    keep = np.abs(levels - med) < 0.75
    if keep.sum() < 3:            # tiny packings / very few filled columns
        keep = np.ones_like(levels, dtype=bool)
    return float(np.clip(levels[keep].mean(), 0.0, geom.depth))


def bed_porosity(pos: np.ndarray, radii: np.ndarray, geom: KeelGeometry,
                 n_points: int = 40000, rng: np.random.Generator | None = None,
                 surface_depth: float | None = None,
                 bins: int = 24) -> tuple[float, float, float]:
    """Return ``(porosity_slab, porosity_point, surface_depth)``.

    ``porosity_slab = 1 - V_solid / V_container(z_surface .. depth)``.
    ``porosity_point`` samples ``n_points`` uniform container points in the
    same depth range and counts those covered by at least one block (the four
    nearest centres are tested, so a small block next to a big one is not
    missed).
    """
    if rng is None:
        rng = np.random.default_rng(12345)
    solid = float(np.sum(4.0 / 3.0 * math.pi * radii ** 3))
    if surface_depth is None:
        surface_depth = _surface_depth(pos, radii, geom, bins)
    v_container = container_volume(geom, surface_depth, geom.depth)
    n_slab = 1.0 - solid / v_container if v_container > 0 else 1.0

    n_point = 1.0
    if len(pos) > 0 and v_container > 0:
        pts = np.empty((n_points, 3))
        filled = 0
        while filled < n_points:
            m = max(2048, (n_points - filled) * 2)
            cand = np.column_stack((
                rng.uniform(-geom.L_top / 2, geom.L_top / 2, m),
                rng.uniform(-geom.W_top / 2, geom.W_top / 2, m),
                rng.uniform(surface_depth, geom.depth, m),
            ))
            half_l, half_w = _container_half_widths(geom, cand[:, 2])
            inside = (np.abs(cand[:, 0]) <= half_l) & (np.abs(cand[:, 1]) <= half_w)
            cand = cand[inside]
            if len(cand) == 0:
                continue
            take = min(len(cand), n_points - filled)
            pts[filled:filled + take] = cand[:take]
            filled += take
        tree = cKDTree(pos)
        k = min(4, len(pos))
        dist, idx = tree.query(pts, k=k)
        dist = np.asarray(dist).reshape(len(pts), k)
        idx = np.asarray(idx).reshape(len(pts), k)
        covered = np.any(dist <= radii[idx], axis=1)
        n_point = 1.0 - float(np.mean(covered))
    return float(n_slab), float(n_point), float(surface_depth)


def exposed_surface_level(pos: np.ndarray, radii: np.ndarray) -> float:
    """Area-weighted mean top surface (``z - r``) of the *exposed* blocks.

    A block is exposed when no other block centre lies above it within the sum
    of radii, i.e. nothing covers it.  The bed level is then
    ``sum(A_i (z_i - r_i)) / sum(A_i)`` with the projected area ``A_i = pi r_i^2``
    as weight.  This is unbiased for an ordered packing (all top-layer blocks
    exposed) and, unlike a fixed quantile, does not sink into the bed when the
    packing is layered -- a quantile level systematically underestimates the bed
    height by ~0.3 of a layer spacing, which biases the void ratio low.
    """
    if len(pos) == 0:
        return 0.0
    r = np.asarray(radii, dtype=float)
    tree = cKDTree(pos[:, :2])
    r_max = float(r.max())
    exposed = np.ones(len(pos), dtype=bool)
    for i in range(len(pos)):
        for j in tree.query_ball_point(pos[i, :2], r[i] + r_max):
            if j == i or pos[j, 2] >= pos[i, 2] - 1e-12:
                continue
            dxy = pos[i, :2] - pos[j, :2]
            if float(np.hypot(dxy[0], dxy[1])) < r[i] + r[j]:
                exposed[i] = False
                break
    if not exposed.any():
        exposed[:] = True
    top = pos[exposed, 2] - r[exposed]
    weight = np.pi * r[exposed] ** 2
    return float(np.sum(weight * top) / np.sum(weight))


def cell_surface_depth(pos: np.ndarray, radii: np.ndarray,
                       n_bins: int = 16, threshold: float = 0.15) -> float:
    """Free-surface depth of a bed: area-weighted level of the exposed blocks."""
    return exposed_surface_level(pos, radii)


def cell_porosity(pos: np.ndarray, radii: np.ndarray, box_size: tuple[float, float],
                  n_points: int = 40000, rng: np.random.Generator | None = None,
                  surface_depth: float | None = None) -> tuple[float, float, float]:
    """Void ratio of a bed in a periodic cell: ``(n_slab, n_point, z_surface)``.

    ``n_slab = 1 - V_solid / (A * h_bed)`` with ``h_bed = floor - z_surface``
    (the floor is taken from the deepest block surface, i.e. the container
    bottom is at the bottom of the bed); ``n_point`` is the independent
    Monte-Carlo point estimator over the same slab.
    """
    if rng is None:
        rng = np.random.default_rng(999)
    radii = np.asarray(radii, dtype=float)
    solid = float(np.sum(4.0 / 3.0 * math.pi * radii ** 3))
    if surface_depth is None:
        surface_depth = cell_surface_depth(pos, radii)
    floor = float(np.max(pos[:, 2] + radii)) if len(pos) else surface_depth + 1e-9
    lx, ly = float(box_size[0]), float(box_size[1])
    h = max(floor - surface_depth, 1e-6)
    n_slab = 1.0 - solid / (lx * ly * h)
    n_point = n_slab
    if len(pos) > 0:
        pts = np.column_stack((
            rng.uniform(0.0, lx, n_points),
            rng.uniform(0.0, ly, n_points),
            rng.uniform(surface_depth, floor, n_points),
        ))
        # minimum-image query against the wrapped block centres
        k = min(4, len(pos))
        covered = np.zeros(len(pts), dtype=bool)
        base = np.floor(pos / np.array([lx, ly, 1e9]))
        for shift_x in (-1, 0, 1):
            for shift_y in (-1, 0, 1):
                centres = pos + np.array([shift_x * lx, shift_y * ly, 0.0])
                tree = cKDTree(centres)
                dist, idx = tree.query(pts, k=k)
                dist = np.asarray(dist).reshape(len(pts), k)
                idx = np.asarray(idx).reshape(len(pts), k)
                covered |= np.any(dist <= radii[idx], axis=1)
        n_point = 1.0 - float(np.mean(covered))
    return float(n_slab), float(n_point), float(surface_depth)


class RubbleDEM:
    """Explicit-dynamics DEM of a polydisperse sphere assembly in a keel.

    The assembly is defined by ``positions`` (N, 3) and ``radii`` (N,).
    ``z`` is depth below the waterline (0 at the top, ``geom.depth`` at the
    keel tip); ``x``/``y`` are horizontal.
    """

    def __init__(self, positions: np.ndarray, radii: np.ndarray,
                 geom: KeelGeometry, config: DEMRubbleConfig | None = None) -> None:
        self.config = config or DEMRubbleConfig()
        self.geom = geom
        self.mode = self.config.mode
        self.box = (float(self.config.box_size[0]), float(self.config.box_size[1]))
        self.pos = np.array(positions, dtype=float, copy=True)
        if self.mode == "cell":
            # wrap the dilute configuration into the periodic cell
            self.pos[:, 0] = np.mod(self.pos[:, 0], self.box[0])
            self.pos[:, 1] = np.mod(self.pos[:, 1], self.box[1])
        self.radii = np.array(radii, dtype=float, copy=True)
        if self.pos.ndim != 2 or self.pos.shape != (len(self.radii), 3):
            raise ValueError("positions must be (N, 3) and match radii")
        self.mass = ICE_DENSITY_KG_M3 * 4.0 / 3.0 * math.pi * self.radii ** 3
        self.inertia = 0.4 * self.mass * self.radii ** 2
        self.vel = np.zeros_like(self.pos)
        self.spin = np.zeros_like(self.pos)
        self.time = 0.0
        self.steps = 0
        dt_max = 0.2 * math.sqrt(float(self.mass.min()) / self.config.kn)
        self.dt_max_stable = dt_max
        if self.config.dt > dt_max:
            raise ValueError(
                f"dt={self.config.dt:g} s exceeds the linear stability bound "
                f"{dt_max:g} s (0.2*sqrt(m_min/kn)); reduce dt or kn")
        e = min(max(self.config.restitution, 1e-3), 0.999)
        ln_e = math.log(e)
        zeta = abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)
        self.gamma_n = 2.0 * zeta * np.sqrt(self.config.kn * 0.5 * self.mass)
        self.drag = self.config.damping_ratio * math.sqrt(
            self.config.kn / float(self.mass.mean()))
        self._pairs = np.empty((0, 2), dtype=np.int64)
        self._tree_step = -10 ** 9
        self.work_dissipated = 0.0
        self.work_wall = 0.0
        self.work_body = 0.0
        self._last_record: dict | None = None

    # -- geometry helpers ---------------------------------------------------
    def _wall_penetrations(self) -> tuple[np.ndarray, np.ndarray]:
        """Penetration depth into each of the 6 container walls.

        Returns ``(pens, inward)`` with ``pens`` of shape (6, N) and
        ``inward`` the unit normal of each wall pointing *into* the container.
        The keel frustum is z-downward: 0 at the waterline, ``depth`` at the
        tip, so the ceiling's inward normal is +z and the floor's is -z.
        """
        g = self.geom
        x, y, z = self.pos[:, 0], self.pos[:, 1], self.pos[:, 2]
        r = self.radii
        if self.mode == "cell":
            pens = np.maximum(z + r - self.config.floor_depth, 0.0)[None, :]
            return pens, np.array([[0.0, 0.0, -1.0]])
        half_l, half_w = _container_half_widths(g, z)
        pens = np.vstack([
            np.maximum(r - half_l - x, 0.0),      # -x wall (x = -half_l)
            np.maximum(x + r - half_l, 0.0),      # +x wall (x = +half_l)
            np.maximum(r - half_w - y, 0.0),      # -y wall
            np.maximum(y + r - half_w, 0.0),      # +y wall
            np.maximum(r - z, 0.0),               # ceiling  (waterline, z = 0)
            np.maximum(z + r - g.depth, 0.0),     # floor    (keel tip, z = depth)
        ])
        inward = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                           [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        return pens, inward

    # -- neighbour list -----------------------------------------------------
    def _rebuild_pairs(self) -> None:
        if len(self.pos) < 2:
            self._pairs = np.empty((0, 2), dtype=np.int64)
            return
        r_max = float(self.radii.max())
        v_max = float(np.linalg.norm(self.vel, axis=1).max()) if len(self.vel) else 0.0
        skin = 2.0 * v_max * self.config.dt * self.config.pair_refresh + 0.03 * r_max
        cutoff = 2.0 * r_max + skin
        if self.mode == "cell":
            # scipy supports periodic neighbour search directly via boxsize,
            # which is ~10x faster than the 9-image Python set union.
            tree = cKDTree(self.pos, boxsize=[self.box[0], self.box[1], 1e12])
            pairs = tree.query_pairs(cutoff, output_type='ndarray')
            pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
            keep = pairs[:, 0] != pairs[:, 1]
            self._pairs = pairs[keep]
        else:
            pairs = cKDTree(self.pos).query_pairs(cutoff, output_type='ndarray')
            self._pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self._tree_step = self.steps

    def _contacts(self) -> dict:
        if len(self._pairs) == 0:
            e = np.empty(0)
            return dict(i=np.empty(0, dtype=np.int64), j=np.empty(0, dtype=np.int64),
                        n=np.empty((0, 3)), overlap=e, vn=e, vt=np.empty((0, 3)))
        i, j = self._pairs[:, 0], self._pairs[:, 1]
        d = self.pos[j] - self.pos[i]
        if self.mode == "cell":
            box = np.array([self.box[0], self.box[1], 1e9])
            d = d - box * np.round(d / box)
        dist = np.linalg.norm(d, axis=1)
        overlap = (self.radii[i] + self.radii[j]) - dist
        mask = (overlap > 0.0) & (dist > 1e-12)
        i, j, d, dist, overlap = i[mask], j[mask], d[mask], dist[mask], overlap[mask]
        n = d / np.maximum(dist, 1e-12)[:, None]
        # contact-point velocity of both bodies, spin included.  Omitting the
        # spin term (as an earlier version did) lets the tangential force spin
        # particles up for free: an energy source that the rolling resistance
        # then "dissipates", breaking the energy ledger.
        lever_i = (self.radii[i] - 0.5 * overlap)[:, None] * n
        lever_j = (self.radii[j] - 0.5 * overlap)[:, None] * n
        if self.config.rotation:
            v_i = self.vel[i] + np.cross(self.spin[i], lever_i)
            v_j = self.vel[j] - np.cross(self.spin[j], lever_j)
            v_rel = v_j - v_i
        else:
            v_rel = self.vel[j] - self.vel[i]
        vn = np.sum(v_rel * n, axis=1)
        vt = v_rel - vn[:, None] * n
        return dict(i=i, j=j, n=n, overlap=overlap, vn=vn, vt=vt,
                    lever_i=lever_i, lever_j=lever_j)

    # -- one step -----------------------------------------------------------
    def step(self) -> None:
        cfg = self.config
        if self.steps - self._tree_step >= cfg.pair_refresh:
            self._rebuild_pairs()
        c = self._contacts()
        i, j = c["i"], c["j"]
        force = np.zeros_like(self.pos)
        torque = np.zeros_like(self.pos)

        # ---- body forces ---------------------------------------------------
        acc = np.zeros_like(self.pos)
        acc[:, 2] += cfg.gravity
        if cfg.wave_enabled and self.time >= cfg.wave_start_time:
            omega = 2.0 * math.pi / cfg.wave_period
            acc[:, 0] += cfg.wave_amplitude * math.sin(omega * self.time)
            acc[:, 1] += 0.3 * cfg.wave_amplitude * math.cos(0.7 * omega * self.time)
        body = self.mass[:, None] * acc
        force += body
        self.work_body += float(np.sum(body * self.vel) * cfg.dt)

        # ---- particle contacts ---------------------------------------------
        if len(i):
            kt = cfg.kt_ratio * cfg.kn
            gamma = 0.5 * (self.gamma_n[i] + self.gamma_n[j])
            fn = np.maximum(cfg.kn * c["overlap"] - gamma * c["vn"], 0.0)
            fvec = fn[:, None] * c["n"]
            vt = c["vt"]
            vt_mag = np.linalg.norm(vt, axis=1)
            ft_mag = np.minimum(cfg.friction * fn, kt * vt_mag * cfg.dt)
            ft = np.zeros_like(vt)
            moving = vt_mag > 1e-12
            ft[moving] = -ft_mag[moving, None] * vt[moving] / vt_mag[moving, None]
            # force on j is +F_n*n (j is pushed away from i) plus the friction
            # force -ft; force on i is the opposite (Newton's third law).
            np.add.at(force, j, fvec - ft)
            np.add.at(force, i, ft - fvec)

            np.add.at(torque, i, np.cross(c["lever_i"], ft))
            np.add.at(torque, j, np.cross(c["lever_j"], ft))

            w_rel = self.spin[j] - self.spin[i]
            w_mag = np.linalg.norm(w_rel, axis=1)
            tau_mag = (cfg.rolling_friction
                       * 0.5 * (self.radii[i] + self.radii[j]) * fn)
            tau_r = np.zeros_like(w_rel)
            rolling = w_mag > 1e-12
            tau_r[rolling] = -tau_mag[rolling, None] * w_rel[rolling] / w_mag[rolling, None]
            np.add.at(torque, i, tau_r)
            np.add.at(torque, j, -tau_r)

            # the dashpot only dissipates while the contact is in compression
            # (F_n > 0); when the clip engages the pair law carries no force
            p_dash = gamma * c["vn"] ** 2 * (fn > 0.0)
            p_slide = -np.sum(ft * vt, axis=1)         # >= 0 (dissipative)
            p_roll = -np.sum(tau_r * w_rel, axis=1)    # >= 0 (dissipative)
            self.work_dissipated += float(np.sum(p_dash + p_slide + p_roll) * cfg.dt)

        # ---- container walls --------------------------------------------------
        if self.mode == "cell":
            # Flat rigid floor as a hard constraint.  A penalty spring here would
            # either need a very small dt (stiff) or allow centimetre-deep
            # penetration during the deposition impacts; the hard projection is
            # both cheaper and conservative for the energy ledger.
            pen = self.pos[:, 2] + self.radii - cfg.floor_depth
            mask = pen > 0.0
            if mask.any():
                idx = np.nonzero(mask)[0]
                self.pos[idx, 2] = cfg.floor_depth - self.radii[idx]
                v = self.vel[idx]
                approaching = v[:, 2] > 0.0
                if approaching.any():
                    v_new = v.copy()
                    dv_n = -(1.0 + cfg.wall_restitution) * v[approaching, 2]
                    v_new[approaching, 2] = -cfg.wall_restitution * v[approaching, 2]
                    vt = v_new[:, :2].copy()
                    vt_mag = np.linalg.norm(vt, axis=1)
                    cap = cfg.wall_friction * np.abs(dv_n)
                    scale = np.ones_like(vt_mag)
                    nz = vt_mag > 1e-12
                    scale[nz] = np.maximum(0.0, 1.0 - cap[nz] / vt_mag[nz])
                    v_new[:, :2] = vt * scale[:, None]
                    self.work_wall += float(np.sum(
                        0.5 * self.mass[idx] * (np.sum(v_new ** 2, axis=1)
                                                - np.sum(v ** 2, axis=1))))
                    self.vel[idx] = v_new
                self.wall_penetration_max = max(
                    getattr(self, "wall_penetration_max", 0.0), float(pen.max()))
        else:
          # ---- rigid frustum walls (penalty springs + Coulomb friction) ------
            pens, inward = self._wall_penetrations()
            for w in range(pens.shape[0]):
                mask = pens[w] > 0.0
                if not mask.any():
                    continue
                n_in = inward[w]
                idx = np.nonzero(mask)[0]
                v = self.vel[idx]
                vn = v @ n_in                       # <0 : moving into the wall
                fn = np.maximum(cfg.kn * pens[w][mask] + self.gamma_n[idx] * vn, 0.0)
                v_t = v - vn[:, None] * n_in
                vt_mag = np.linalg.norm(v_t, axis=1)
                ft_mag = np.minimum(cfg.wall_friction * fn,
                                cfg.kt_ratio * cfg.kn * vt_mag * cfg.dt)
                ft = np.zeros_like(v_t)
                moving = vt_mag > 1e-12
                ft[moving] = -ft_mag[moving, None] * v_t[moving] / vt_mag[moving, None]
                back = fn[:, None] * n_in + ft
                np.add.at(force, idx, back)
                self.work_wall += float(np.sum(back * v) * cfg.dt)
                self.wall_penetration_max = max(
                getattr(self, "wall_penetration_max", 0.0), float(pens[w].max()))

        # ---- pore-water viscous drag (translational + spin) ----------------
        drag = -self.drag * self.mass[:, None] * self.vel
        force += drag
        self.work_dissipated -= float(np.sum(drag * self.vel) * cfg.dt)  # drag power <= 0
        if cfg.rotation:
            spin_drag = -self.drag * cfg.spin_damping * self.inertia[:, None] * self.spin
            torque += spin_drag

        # ---- explicit Euler-Cromer -----------------------------------------
        self.vel += (force / self.mass[:, None]) * cfg.dt
        self.pos += self.vel * cfg.dt
        if cfg.rotation:
            self.spin += (torque / self.inertia[:, None]) * cfg.dt
        if self.mode == "cell":
            self.pos[:, 0] = np.mod(self.pos[:, 0], self.box[0])
            self.pos[:, 1] = np.mod(self.pos[:, 1], self.box[1])
        self.steps += 1
        self.time = self.steps * cfg.dt
        if not (np.isfinite(self.pos).all() and np.isfinite(self.vel).all()):
            raise FloatingPointError("non-finite DEM state; reduce dt or kn")

    # -- diagnostics ---------------------------------------------------------
    def kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.mass * np.sum(self.vel ** 2, axis=1)))

    def gravitational_energy(self) -> float:
        return float(-np.sum(self.mass * self.config.gravity * self.pos[:, 2]))

    def elastic_energy(self) -> float:
        c = self._contacts()
        e_contact = 0.5 * self.config.kn * float(np.sum(c["overlap"] ** 2)) if len(c["i"]) else 0.0
        if self.mode == "cell":
            return e_contact          # hard floor stores no elastic energy
        pens, _ = self._wall_penetrations()
        e_wall = 0.5 * self.config.kn * float(np.sum(pens ** 2))
        return e_contact + e_wall

    def coordination(self) -> float:
        c = self._contacts()
        if len(c["i"]) == 0 or len(self.pos) == 0:
            return 0.0
        counts = (np.bincount(c["i"], minlength=len(self.pos))
                  + np.bincount(c["j"], minlength=len(self.pos)))
        return float(counts.mean())

    def record(self) -> dict:
        if self.mode == "cell":
            n_slab, n_point, surf = cell_porosity(
                self.pos, self.radii, self.box,
                n_points=self.config.porosity_points)
        else:
            n_slab, n_point, surf = bed_porosity(
                self.pos, self.radii, self.geom,
                n_points=self.config.porosity_points,
                bins=self.config.porosity_surface_bins)
        pens, _ = self._wall_penetrations()
        rec = dict(time=self.time, porosity_slab=n_slab, porosity_point=n_point,
                   wall_penetration=float(pens.max()) if pens.size else 0.0,
                   surface_depth=surf, kinetic_energy=self.kinetic_energy(),
                   elastic_energy=self.elastic_energy(),
                   coordination=self.coordination(),
                   mean_spin=float(np.mean(np.linalg.norm(self.spin, axis=1))))
        self._last_record = rec
        return rec

    def run(self) -> DEMRubbleResult:
        cfg = self.config
        n_steps = int(round(cfg.total_time / cfg.dt))
        record_every = max(1, int(round(cfg.record_interval / cfg.dt)))
        recs = [self.record()]
        shake_start = None
        for k in range(1, n_steps + 1):
            if (shake_start is None and cfg.wave_enabled
                    and self.time >= cfg.wave_start_time):
                shake_start = self.energy_ledger()
            self.step()
            if k % record_every == 0 or k == n_steps:
                recs.append(self.record())
        times = np.array([r["time"] for r in recs])
        result = DEMRubbleResult(
            time=times,
            porosity_slab=np.array([r["porosity_slab"] for r in recs]),
            porosity_point=np.array([r["porosity_point"] for r in recs]),
            surface_depth=np.array([r["surface_depth"] for r in recs]),
            kinetic_energy=np.array([r["kinetic_energy"] for r in recs]),
            elastic_energy=np.array([r["elastic_energy"] for r in recs]),
            coordination=np.array([r["coordination"] for r in recs]),
            mean_spin=np.array([r["mean_spin"] for r in recs]),
            positions=self.pos.copy(),
            radii=self.radii.copy(),
            velocity=self.vel.copy(),
            config=cfg,
            energy_ledger=self.energy_ledger(),
        )
        if shake_start is not None:
            end = result.energy_ledger
            delta = {k: end[k] - shake_start.get(k, 0.0) for k in end}
            delta["closure_residual"] = (
                delta["work_body"] + delta["work_wall"] - delta["work_dissipated"]
                - (delta["kinetic_energy"] + delta["elastic_energy"]
                   + delta["gravitational_energy"]))
            result.energy_ledger["shake_phase"] = delta
        return result

    def energy_ledger(self) -> dict:
        """Energy-balance snapshot (J).

        ``work_body`` is the work done by gravity + wave shaking (negative of
        the change in potential energy for the gravity part), ``work_wall`` the
        work of the container, ``work_dissipated`` the positive work removed by
        dashpots, sliding friction, rolling resistance and pore drag.  Closure
        residual = ``work_body + work_wall - work_dissipated - (KE + E_el + PE)``
        should be small relative to the total work performed.
        """
        ke = self.kinetic_energy()
        pe = self.gravitational_energy()
        e_el = self.elastic_energy()
        residual = (self.work_body + self.work_wall - self.work_dissipated
                    - (ke + e_el + pe))
        return dict(work_body=self.work_body, work_wall=self.work_wall,
                    work_dissipated=self.work_dissipated, kinetic_energy=ke,
                    gravitational_energy=pe, elastic_energy=e_el,
                    closure_residual=residual)
