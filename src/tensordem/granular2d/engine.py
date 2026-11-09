"""Rigid frictional disks: Hertz-Mindlin normal/tangential + rolling resistance.

The H1 ice-channel model in :mod:`tensordem.ice_channel` is a bonded lattice and
carries no particle rotations, so it cannot represent a *granular* broken-ice
field where momentum is transferred through a frictional contact network.  This
module adds the missing degree of freedom (disk rotation) and the missing contact
physics (Coulomb tangential plasticity and rolling resistance) while re-using the
same SI conventions and the polyline-boundary idea of the H1 hull contact.

Everything is float64 NumPy and deterministic: neighbour lists come from a
deterministic ``scipy.spatial.cKDTree`` query, contacts are ordered by a stable
integer key and all force accumulations use explicit ``np.add.at`` summation so a
given seed gives a bit-identical trajectory and digest.

Force convention
----------------
For a contact ``(i, j)`` the unit normal ``n`` points from ``j`` towards ``i``.
The pair force on ``i`` is ``+f`` and on ``j`` is ``-f`` (exactly equal and
opposite), with

.. math:: f_n = \\max(0, k_n \\delta + c_n \\, n\\cdot v_{rel})
.. math:: f_t = -k_t \\xi_t, \\qquad |k_t \\xi_t| \\le \\mu f_n
.. math:: M_r = -k_r \\theta_{rel}, \\qquad |M_r| \\le \\mu_r r_{eff} f_n

Periodicity
-----------
``periodic_x`` and/or ``periodic_y`` wrap the cell.  For a doubly periodic cell
both the neighbour search (eight mirror images chained through the four
boundary-adjacent index sets) and the minimum-image convention are applied on
each axis, so wall layers disappear and ``domain_area()`` returns ``Lx * Ly``.
Periodic contact stress carries a wrap term: see
:func:`tensordem.granular2d.stress.stress_balance`.

Boundary segments (channel banks, ship hull) enter as an infinitely massive
static body: ``j = -1`` and the reaction ``-f`` is accumulated on the segment
owner (wall or hull) so boundary tractions are never lost.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["DiskConfig", "DiskSystem", "box_walls", "polyline_segments", "ship_segments"]


@dataclass(frozen=True)
class DiskConfig:
    """Material and contact parameters (SI)."""

    kn: float = 3.0e4
    kt_ratio: float = 0.5
    zeta: float = 0.2
    mu: float = 0.5
    mu_rolling: float = 0.05
    density: float = 917.0
    thickness: float = 0.03
    gravity: tuple[float, float] = (0.0, 0.0)
    drift_velocity: tuple[float, float] = (0.0, 0.0)
    drag_time: float = 0.0
    damping_time: float = 0.0
    dt: float | None = None
    skin: float = 0.0035

    def __post_init__(self) -> None:
        for name in ("kn", "kt_ratio", "density", "thickness"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("zeta", "mu", "mu_rolling", "drag_time", "damping_time", "skin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.dt is not None and (not math.isfinite(self.dt) or self.dt <= 0.0):
            raise ValueError("dt must be finite and positive")
        if len(tuple(self.gravity)) != 2 or len(tuple(self.drift_velocity)) != 2:
            raise ValueError("gravity and drift_velocity must have two components")
        if not all(math.isfinite(v) for v in tuple(self.gravity) + tuple(self.drift_velocity)):
            raise ValueError("gravity and drift_velocity must be finite")


def polyline_segments(points) -> np.ndarray:
    """Return ``(S, 2, 2)`` segments of a closed polyline through ``points``."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError("points must have shape (M, 2) with M >= 3")
    if not np.isfinite(pts).all():
        raise ValueError("points must be finite")
    return np.stack([pts, np.roll(pts, -1, axis=0)], axis=1)


def box_walls(width: float, height: float, *, left: bool = True, right: bool = True,
              bottom: bool = True, top: bool = True, margin: float = 0.0) -> np.ndarray:
    """Axis-aligned rectangular walls as ``(S, 2, 2)`` segments.

    The closest-point projection is exact for any straight segment, so a wall of
    finite length is enough; ``margin`` extends the segment past the corners so a
    disk near a corner still sees both walls.
    """
    for name, value in (("width", width), ("height", height)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    segs = []
    if bottom:
        segs.append([[-margin, 0.0], [width + margin, 0.0]])
    if top:
        segs.append([[-margin, height], [width + margin, height]])
    if left:
        segs.append([[0.0, -margin], [0.0, height + margin]])
    if right:
        segs.append([[width, -margin], [width, height + margin]])
    return np.asarray(segs, dtype=np.float64).reshape(-1, 2, 2)


def ship_segments(length: float, beam: float, origin=(0.0, 0.0), *, bow: float = 0.0,
                  stern: float = 0.0, n_bow: int = 6) -> np.ndarray:
    """Closed polyline of a ship hull in the x-y plane (bow towards +x)."""
    for name, value in (("length", length), ("beam", beam)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not (0.0 <= bow < length) or not (0.0 <= stern < length):
        raise ValueError("bow/stern must be smaller than the hull length")
    x0, y0 = float(origin[0]), float(origin[1])
    half = 0.5 * beam
    pts = [(x0, y0), (x0 + max(stern, 0.0), y0 - half),
           (x0 + length - bow, y0 - half)]
    if bow > 0.0:
        for k in range(1, n_bow):
            s = k / n_bow
            pts.append((x0 + length - bow + s * bow, y0 - half * (1.0 - s)))
    pts.append((x0 + length, y0))
    if bow > 0.0:
        for k in range(n_bow - 1, 0, -1):
            s = k / n_bow
            pts.append((x0 + length - bow + s * bow, y0 + half * (1.0 - s)))
    pts.extend([(x0 + length - bow, y0 + half), (x0 + max(stern, 0.0), y0 + half)])
    return polyline_segments(np.asarray(pts, dtype=np.float64))


class DiskSystem:
    """A deterministic 2D rigid-disk assembly with optional static boundaries."""

    def __init__(self, positions, radii, config: DiskConfig | None = None, *,
                 walls=None, hull=None, periodic_x: float | None = None,
                 periodic_y: float | None = None, wall_velocity=None) -> None:
        self.config = config or DiskConfig()
        pos = np.array(positions, dtype=np.float64)
        rad = np.array(radii, dtype=np.float64).reshape(-1)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")
        if pos.shape[0] != rad.shape[0]:
            raise ValueError("positions and radii must have the same length")
        if pos.shape[0] < 1:
            raise ValueError("at least one disk is required")
        if not np.isfinite(pos).all() or not np.isfinite(rad).all():
            raise ValueError("positions and radii must be finite")
        if (rad <= 0.0).any():
            raise ValueError("radii must be positive")
        self.pos = pos
        self.radii = rad
        self.vel = np.zeros_like(pos)
        self.theta = np.zeros(len(rad))
        self.omega = np.zeros(len(rad))
        cfg = self.config
        self.mass = cfg.density * math.pi * rad**2 * cfg.thickness
        self.inertia = 0.5 * self.mass * rad**2
        self.walls = self._as_segments(walls, "walls")
        self.hull = self._as_segments(hull, "hull")
        self.wall_velocity = self._as_wall_velocity(wall_velocity)
        self.periodic_x = None if periodic_x is None else float(periodic_x)
        self.periodic_y = None if periodic_y is None else float(periodic_y)
        for name, value in (("periodic_x", self.periodic_x), ("periodic_y", self.periodic_y)):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be a positive period or None")
        self.time = 0.0
        self.steps = 0
        self.dt = cfg.dt if cfg.dt is not None else self.recommended_dt
        if self.dt > self.recommended_dt * (1.0 + 1e-12):
            raise ValueError(f"dt exceeds the stability limit {self.recommended_dt:.6g} s")
        self._tree = None
        self._tree_pos = None
        self._state_keys = np.zeros(0, dtype=np.int64)
        self._state_xi = np.zeros((0, 2))
        self._state_roll = np.zeros(0)
        self.contacts: dict[str, np.ndarray] = {}
        self.hull_force = np.zeros(2)
        self.wall_force = np.zeros(2)
        self.hull_force_history: list[tuple[float, np.ndarray]] = []
        self.contact_history: list[dict[str, float]] = []
        self._rebuild_neighbors()

    # ------------------------------------------------------ moving-wall drive
    def _as_wall_velocity(self, value) -> np.ndarray | None:
        """Per-wall-segment translational velocity ``(S, 2)`` or ``None``.

        A rough wall that *slides* can only be represented if its velocity enters
        the tangential contact law; without it a moving wall would act as a
        brake.  ``None`` keeps the historical static-wall behaviour exactly
        (the velocity offset is then omitted from the relative velocity).
        """
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 1:
            arr = np.broadcast_to(arr.reshape(1, 2), (len(self.walls), 2)).copy()
        if arr.shape != (len(self.walls), 2):
            raise ValueError("wall_velocity must have shape (S, 2) matching walls")
        if not np.isfinite(arr).all():
            raise ValueError("wall_velocity must be finite")
        return arr

    def set_wall_velocity(self, value) -> None:
        """Set (or clear, with ``None``) the sliding velocity of every wall."""
        self.wall_velocity = self._as_wall_velocity(value)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _as_segments(value, name: str) -> np.ndarray:
        if value is None:
            return np.zeros((0, 2, 2))
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[1:] != (2, 2):
            raise ValueError(f"{name} must have shape (S, 2, 2)")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite")
        return arr

    @property
    def n(self) -> int:
        return len(self.pos)

    @property
    def mean_radius(self) -> float:
        return float(self.radii.mean())

    @property
    def recommended_dt(self) -> float:
        m_min = float(self.mass.min())
        omega = math.sqrt(2.0 * self.config.kn / (0.5 * m_min))
        return 0.1 / omega

    def area_fraction(self, area: float | None = None) -> float:
        disk_area = float(np.sum(math.pi * self.radii**2))
        if area is None:
            area = self.domain_area()
        if area is None or area <= 0.0:
            raise ValueError("an explicit positive area is required")
        return disk_area / area

    @property
    def periodic(self) -> tuple[float | None, float | None]:
        """Periods ``(Lx, Ly)``; ``None`` marks a non-periodic axis."""
        return (self.periodic_x, self.periodic_y)

    @property
    def n_periodic_axes(self) -> int:
        return int(self.periodic_x is not None) + int(self.periodic_y is not None)

    def _minimum_image(self, d: np.ndarray) -> np.ndarray:
        """Apply the minimum-image convention on every periodic axis."""
        if self.periodic_x is not None:
            d[..., 0] -= self.periodic_x * np.round(d[..., 0] / self.periodic_x)
        if self.periodic_y is not None:
            d[..., 1] -= self.periodic_y * np.round(d[..., 1] / self.periodic_y)
        return d

    def domain_area(self) -> float | None:
        """Cell area: the periodic box when both axes are periodic.

        With a doubly periodic cell the volume is unambiguous and no wall layer
        exists; with only ``periodic_x`` the area still comes from the walls.
        """
        if self.periodic_x is not None and self.periodic_y is not None:
            return float(self.periodic_x * self.periodic_y)
        if len(self.walls) == 0:
            return None
        pts = self.walls.reshape(-1, 2)
        return float(np.prod(pts.max(axis=0) - pts.min(axis=0)))

    def set_uniform_velocity(self, velocity) -> None:
        self.vel[:] = np.asarray(velocity, dtype=np.float64).reshape(1, 2)

    def zero_velocities(self) -> None:
        self.vel[:] = 0.0
        self.omega[:] = 0.0

    def momentum(self) -> np.ndarray:
        return (self.mass[:, None] * self.vel).sum(axis=0)

    def kinetic_energy(self) -> float:
        return float(np.sum(0.5 * self.mass * np.sum(self.vel**2, axis=1))
                     + np.sum(0.5 * self.inertia * self.omega**2))

    def digest(self) -> str:
        """Bit-level digest of the state (reproducibility evidence)."""
        h = hashlib.sha256()
        for arr in (self.pos, self.vel, self.theta, self.omega):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        h.update(np.float64(self.time).tobytes())
        return h.hexdigest()

    # ------------------------------------------------------------- neighbours
    def _candidate_pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(i, j, l)`` with ``i < j`` and ``l = x_j - x_i`` (minimum image)."""
        reach = 2.0 * float(self.radii.max()) + self.config.skin
        if (self._tree is None or self._tree_pos is None
                or float(np.max(np.linalg.norm(self.pos - self._tree_pos, axis=1)))
                > 0.5 * self.config.skin):
            self._tree = cKDTree(self.pos)
            self._tree_pos = self.pos.copy()
        pairs = self._tree.query_pairs(reach, output_type="ndarray").reshape(-1, 2)
        extra = []
        lx, ly = self.periodic_x, self.periodic_y
        # boundary-adjacent index sets: an image pair can only cross the border
        # through one of these
        near = {}
        if lx is not None:
            near["x0"] = self.pos[:, 0] < reach
            near["x1"] = self.pos[:, 0] > lx - reach
        if ly is not None:
            near["y0"] = self.pos[:, 1] < reach
            near["y1"] = self.pos[:, 1] > ly - reach
        combos = []
        if lx is not None:
            combos.append(((lx, 0.0), ("x0",), ("x1",)))
        if ly is not None:
            combos.append(((0.0, ly), ("y0",), ("y1",)))
        if lx is not None and ly is not None:
            combos.append(((lx, ly), ("x0", "y0"), ("x1", "y1")))
            combos.append(((lx, -ly), ("x0", "y1"), ("x1", "y0")))
        for shift, a_keys, b_keys in combos:
            a_mask = np.ones(len(self.pos), dtype=bool)
            b_mask = np.ones(len(self.pos), dtype=bool)
            for key in a_keys:
                a_mask &= near[key]
            for key in b_keys:
                b_mask &= near[key]
            a = np.flatnonzero(a_mask)
            b = np.flatnonzero(b_mask)
            if not len(a) or not len(b):
                continue
            d = self.pos[b][None, :, :] - (self.pos[a][:, None, :] + np.asarray(shift))
            d = self._minimum_image(d)
            ii, jj = np.nonzero(np.linalg.norm(d, axis=2) < reach)
            if len(ii):
                extra.append(np.stack([a[ii], b[jj]], axis=1))
        if extra:
            pairs = np.vstack([pairs] + extra)
        if not len(pairs):
            return (np.zeros(0, dtype=np.int64),) * 2 + (np.zeros((0, 2)),)
        pairs = np.unique(np.sort(pairs.astype(np.int64), axis=1), axis=0)
        i, j = pairs[:, 0], pairs[:, 1]
        l = self.pos[j] - self.pos[i]
        l = self._minimum_image(l)
        return i, j, l

    def _segment_contacts(self, segments: np.ndarray):
        """Disk/segment overlaps: ``(disk, point, n_out, overlap, segment)``."""
        if len(segments) == 0:
            empty_i = np.zeros(0, dtype=np.int64)
            return (empty_i, np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0),
                    empty_i)
        a = segments[:, 0, :]
        ab = segments[:, 1, :] - a
        ap = self.pos[:, None, :] - a[None, :, :]
        denom = np.maximum(np.sum(ab * ab, axis=1), 1e-30)
        t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / denom[None, :], 0.0, 1.0)
        cp = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        d = self.pos[:, None, :] - cp
        dist = np.linalg.norm(d, axis=2)
        overlap = self.radii[:, None] - dist
        ok = (overlap > 0.0) & (dist > 1e-12)
        disk, seg = np.nonzero(ok)
        if len(disk) == 0:
            return (np.zeros(0, dtype=np.int64), np.zeros((0, 2)),
                    np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=np.int64))
        point = cp[disk, seg]
        n_out = d[disk, seg] / dist[disk, seg][:, None]
        return (disk.astype(np.int64), point, n_out, overlap[disk, seg],
                seg.astype(np.int64))

    @staticmethod
    def _perp(vec: np.ndarray) -> np.ndarray:
        out = np.empty_like(vec)
        out[..., 0] = -vec[..., 1]
        out[..., 1] = vec[..., 0]
        return out

    @staticmethod
    def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def _load_state(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xi = np.zeros((len(keys), 2))
        roll = np.zeros(len(keys))
        if len(self._state_keys) and len(keys):
            idx = np.clip(np.searchsorted(self._state_keys, keys), 0, len(self._state_keys) - 1)
            found = self._state_keys[idx] == keys
            xi[found] = self._state_xi[idx[found]]
            roll[found] = self._state_roll[idx[found]]
        return xi, roll

    # ---------------------------------------------------------------- forces
    def _evaluate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(force, torque, external_force)`` acting on the disks.

        Refreshes ``self.contacts`` (per-contact records consumed by the stress
        and force-chain analysis) and the hull/wall reaction accumulators.
        """
        cfg = self.config
        n = self.n
        force = np.zeros((n, 2))
        torque = np.zeros(n)
        ext = np.zeros((n, 2))

        i, j, l = self._candidate_pairs()
        dist = np.linalg.norm(l, axis=1)
        n_ij = l / np.maximum(dist, 1e-30)[:, None]
        overlap = self.radii[i] + self.radii[j] - dist
        keep = overlap > 0.0
        i, j, l, dist, n_ij, overlap = i[keep], j[keep], l[keep], dist[keep], n_ij[keep], overlap[keep]
        n_pair = len(i)

        wd, wp, wn, wo, wseg = self._segment_contacts(self.walls)
        hd, hp, hn, ho, _hseg = self._segment_contacts(self.hull)
        n_w, n_h = len(wd), len(hd)

        disk_idx = np.concatenate([i, wd, hd]).astype(np.int64)
        # normal convention: n points from j towards i (so the force on i is +f n)
        n_all = np.concatenate([-n_ij, wn, hn])
        ov_all = np.concatenate([overlap, wo, ho])
        m_all = len(disk_idx)
        if m_all == 0:
            g = np.asarray(cfg.gravity, dtype=np.float64)
            if np.any(g != 0.0):
                force += self.mass[:, None] * g[None, :]
            if cfg.drag_time > 0.0:
                v_drift = np.asarray(cfg.drift_velocity, dtype=np.float64)
                force += self.mass[:, None] * (v_drift[None, :] - self.vel) / cfg.drag_time
            self.contacts = {k: np.zeros(0) for k in
                             ("i", "j", "kind", "fn", "ft", "overlap")}
            self.contacts.update({"n": np.zeros((0, 2)), "lever": np.zeros((0, 2)),
                                  "f": np.zeros((0, 2)), "point": np.zeros((0, 2)),
                                  "f_on_i": np.zeros((0, 2)), "f_on_j": np.zeros((0, 2))})
            self.hull_force = np.zeros(2)
            self.wall_force = np.zeros(2)
            self._contact_force, self._contact_torque = force, torque
            self._external_force = ext
            self._state_keys = np.zeros(0, dtype=np.int64)
            self._state_xi = np.zeros((0, 2))
            self._state_roll = np.zeros(0)
            return force, torque, ext

        # stable ascending contact keys --------------------------------
        keys = np.empty(m_all, dtype=np.int64)
        keys[:n_pair] = i * n + j
        keys[n_pair:] = n * n + np.arange(n_w + n_h, dtype=np.int64)
        xi, roll = self._load_state(keys)

        m_eff = np.empty(m_all)
        r_eff = np.empty(m_all)
        m_eff[:n_pair] = (self.mass[i] * self.mass[j]) / (self.mass[i] + self.mass[j])
        r_eff[:n_pair] = self.radii[i] * self.radii[j] / (self.radii[i] + self.radii[j])
        m_eff[n_pair:] = self.mass[disk_idx[n_pair:]]
        r_eff[n_pair:] = self.radii[disk_idx[n_pair:]]

        rad_other = np.zeros(m_all)
        rad_other[:n_pair] = self.radii[j]
        lever_i = -self.radii[disk_idx][:, None] * n_all
        lever_j = rad_other[:, None] * n_all

        omega_j = np.zeros(m_all)
        omega_j[:n_pair] = self.omega[j]
        v_i = self.vel[disk_idx] + self._perp(self.omega[disk_idx][:, None] * lever_i)
        v_j = np.zeros_like(v_i)
        v_j[:n_pair] = self.vel[j] + self._perp(omega_j[:n_pair][:, None] * lever_j[:n_pair])
        if self.wall_velocity is not None and n_w:
            v_j[n_pair:n_pair + n_w] = self.wall_velocity[wseg]
        v_rel = v_i - v_j
        v_n = np.sum(v_rel * n_all, axis=1)

        c_n = 2.0 * cfg.zeta * np.sqrt(cfg.kn * m_eff)
        f_n = np.maximum(0.0, cfg.kn * ov_all - c_n * v_n)

        v_t = v_rel - v_n[:, None] * n_all
        xi = xi + v_t * self.dt
        xi = xi - np.sum(xi * n_all, axis=1)[:, None] * n_all
        k_t = cfg.kt_ratio * cfg.kn
        f_t_vec = -k_t * xi
        f_t_mag = np.linalg.norm(f_t_vec, axis=1)
        cap = cfg.mu * f_n
        scale = np.where(f_t_mag > cap, cap / np.maximum(f_t_mag, 1e-30), 1.0)
        f_t_vec *= scale[:, None]
        xi *= scale[:, None]
        f_t_mag *= scale

        total = f_n[:, None] * n_all + f_t_vec
        np.add.at(force, disk_idx[:n_pair], total[:n_pair])
        np.add.at(force, j, -total[:n_pair])
        np.add.at(torque, disk_idx[:n_pair], self._cross(lever_i[:n_pair], total[:n_pair]))
        np.add.at(torque, j, self._cross(lever_j[:n_pair], -total[:n_pair]))

        wsel = slice(n_pair, n_pair + n_w)
        hsel = slice(n_pair + n_w, m_all)
        if n_w:
            np.add.at(force, disk_idx[wsel], total[wsel])
            np.add.at(torque, disk_idx[wsel], self._cross(lever_i[wsel], total[wsel]))
            np.add.at(ext, disk_idx[wsel], total[wsel])
            self.wall_force = -total[wsel].sum(axis=0)
        else:
            self.wall_force = np.zeros(2)
        if n_h:
            np.add.at(force, disk_idx[hsel], total[hsel])
            np.add.at(torque, disk_idx[hsel], self._cross(lever_i[hsel], total[hsel]))
            np.add.at(ext, disk_idx[hsel], total[hsel])
            self.hull_force = -total[hsel].sum(axis=0)
        else:
            self.hull_force = np.zeros(2)

        if cfg.mu_rolling > 0.0:
            roll = roll + (self.omega[disk_idx] - omega_j) * self.dt
            k_r = 0.25 * k_t * r_eff**2
            m_r = k_r * roll
            cap_r = cfg.mu_rolling * r_eff * f_n
            scale_r = np.where(np.abs(m_r) > cap_r, cap_r / np.maximum(np.abs(m_r), 1e-30), 1.0)
            m_r *= scale_r
            roll *= scale_r
            np.add.at(torque, disk_idx[:n_pair], -m_r[:n_pair])
            np.add.at(torque, j, m_r[:n_pair])
            np.add.at(torque, disk_idx[n_pair:], -m_r[n_pair:])

        g = np.asarray(cfg.gravity, dtype=np.float64)
        if np.any(g != 0.0):
            force = force + self.mass[:, None] * g[None, :]
        if cfg.drag_time > 0.0:
            v_drift = np.asarray(cfg.drift_velocity, dtype=np.float64)
            force = force + self.mass[:, None] * (v_drift[None, :] - self.vel) / cfg.drag_time

        # per-contact records ----------------------------------------------
        lever = np.zeros((m_all, 2))
        lever[:n_pair] = l
        if n_w:
            lever[n_pair:n_pair + n_w] = wp - self.pos[wd]
        if n_h:
            lever[n_pair + n_w:] = hp - self.pos[hd]
        point = np.zeros((m_all, 2))
        point[:n_pair] = self.pos[i] + self.radii[i][:, None] * n_ij
        if n_w:
            point[n_pair:n_pair + n_w] = wp
        if n_h:
            point[n_pair + n_w:] = hp
        j_all = np.concatenate([j, -np.ones(n_w + n_h, dtype=np.int64)])
        self.contacts = {
            "i": disk_idx,
            "j": j_all,
            "kind": np.concatenate([np.zeros(n_pair, dtype=np.int8),
                                    np.ones(n_w, dtype=np.int8),
                                    np.full(n_h, 2, dtype=np.int8)]),
            "n": n_all,
            "lever": lever,
            "f": total,
            "f_on_i": total,
            "f_on_j": -total,
            "fn": f_n,
            "ft": f_t_mag,
            "overlap": ov_all,
            "point": point,
        }
        self._state_keys = keys
        self._state_xi = xi
        self._state_roll = roll
        self._contact_force = force
        self._contact_torque = torque
        self._external_force = ext
        return force, torque, ext

    # ------------------------------------------------------------------- API
    def step(self, n: int = 1, *, record: bool = False) -> None:
        """Advance ``n`` steps with semi-implicit (symplectic) Euler."""
        if n < 0:
            raise ValueError("n must be nonnegative")
        for _ in range(n):
            force, torque, _ = self._evaluate()
            self.vel += self.dt * (force / self.mass[:, None])
            self.omega += self.dt * (torque / self.inertia)
            if self.config.damping_time > 0.0:
                decay = math.exp(-self.dt / self.config.damping_time)
                self.vel *= decay
                self.omega *= decay
            self.pos += self.dt * self.vel
            self.theta += self.dt * self.omega
            if self.periodic_x is not None:
                self.pos[:, 0] = np.mod(self.pos[:, 0], self.periodic_x)
            if self.periodic_y is not None:
                self.pos[:, 1] = np.mod(self.pos[:, 1], self.periodic_y)
            self.time += self.dt
            self.steps += 1
            if record:
                self.contact_history.append({
                    "time": self.time,
                    "hull_fx": float(self.hull_force[0]),
                    "hull_fy": float(self.hull_force[1]),
                    "wall_fx": float(self.wall_force[0]),
                    "wall_fy": float(self.wall_force[1]),
                    "kinetic_energy": self.kinetic_energy(),
                })

    def contact_forces(self) -> dict[str, np.ndarray]:
        """Evaluate forces and return the contact records without integrating."""
        self._evaluate()
        return self.contacts

    def mean_speed(self) -> float:
        return float(np.mean(np.linalg.norm(self.vel, axis=1)))

    def _rebuild_neighbors(self) -> None:
        self._tree = cKDTree(self.pos)
        self._tree_pos = self.pos.copy()