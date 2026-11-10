"""Bidisperse granular bed: vertically vibrated channel and gravity pour (quasi-2-D).

The solver is the validated frictional Hertz-Mindlin disk engine
:class:`tensordem.granular2d.engine.DiskSystem` (linear spring-dashpot normal
law, Cundall-Strack tangential plasticity, rolling resistance, rigid segment
boundaries).  Two additions are needed for a segregation study and they are the
only differences from the base engine:

``kinematic base drive``
    the bottom boundary segment is translated as ``y(t) = y0 + A sin(w t)`` and
    the contact damping term sees the true boundary velocity through the
    engine's ``wall_velocity_field`` hook.  The dimensionless drive is
    ``Gamma = A w^2 / g``.

``fast force path``
    :meth:`BidisperseBed._evaluate` reproduces the base engine's force kernel
    *bit-for-bit in the physics* but skips the per-contact record bookkeeping
    and uses ``np.bincount`` instead of ``np.add.at`` for the scatter
    accumulations.  ``tests/test_segregation_bed.py`` pins the fast path against
    ``DiskSystem._evaluate`` to 1e-12 relative, so the optimisation cannot drift
    away from the validated kernel.

Nothing in this module knows about ships: the bed is a generic rectangular
container with a driven floor.  The ship-side interpretation lives in
:mod:`tensordem.segregation.transfer`.
"""

from __future__ import annotations

import math

import numpy as np

from scipy.spatial import cKDTree

from ..granular2d.engine import DiskConfig, DiskSystem
from .metrics import particle_measure

__all__ = ["BidisperseBed", "channel_walls", "BoxTooSmall"]


class BoxTooSmall(ValueError):
    """Raised when a particle leaves the container (a numerical failure)."""


def channel_walls(width: float, height: float, *, margin: float = 0.0,
                  gate_x: float | None = None) -> tuple[np.ndarray, int, int | None]:
    """Walls of a rectangular trough with an open top.

    Returns ``(segments, base_index, gate_index)``.  Segment 0 is the base: it is
    the only horizontal segment at the bottom and is the one the vibration drive
    moves.  When ``gate_x`` is given a vertical gate segment is appended; moving
    it far away releases the material (the gravity-pour driver).
    """
    if not math.isfinite(width) or width <= 0.0 or not math.isfinite(height) or height <= 0.0:
        raise ValueError("width and height must be finite and positive")
    segs = [
        [[-margin, 0.0], [width + margin, 0.0]],          # 0: base (driven)
        [[0.0, -margin], [0.0, height + margin]],         # 1: left wall
        [[width, -margin], [width, height + margin]],     # 2: right wall
    ]
    gate = None
    if gate_x is not None:
        if not 0.0 < gate_x < width:
            raise ValueError("gate_x must lie strictly inside the trough")
        gate = len(segs)
        segs.append([[gate_x, 0.0], [gate_x, height]])  # 3: gate
    return np.asarray(segs, dtype=np.float64), 0, gate


class BidisperseBed(DiskSystem):
    """A two-component disk bed in a trough with a kinematically driven floor."""

    def __init__(
        self,
        positions: np.ndarray,
        radii: np.ndarray,
        is_large: np.ndarray,
        config: DiskConfig,
        *,
        walls: np.ndarray,
        base_index: int = 0,
        gate_index: int | None = None,
    ) -> None:
        super().__init__(positions, radii, config, walls=walls)
        self.is_large = np.asarray(is_large, dtype=bool).copy()
        if self.is_large.size != self.n:
            raise ValueError("is_large must have one entry per particle")
        self.measure = particle_measure(self.radii, 2)
        self.base_index = int(base_index)
        self.gate_index = gate_index
        base = self.walls[self.base_index]
        if abs(base[0, 1] - base[1, 1]) > 1e-12:
            raise ValueError("the driven base segment must be horizontal")
        self.base_y0 = float(base[0, 1])
        self.amplitude = 0.0
        self.frequency = 0.0
        self.phase = 0.0
        self.gate_open = False
        # last-step base contact records (kinematics of the load path)
        self.base_contact_disk = np.zeros(0, dtype=np.int64)
        self.base_contact_fn = np.zeros(0)
        self.base_contact_ft = np.zeros(0)
        self.base_contact_x = np.zeros(0)
        self._pair_i = np.zeros(0, dtype=np.int64)
        self._pair_j = np.zeros(0, dtype=np.int64)
        # ``DiskSystem.__init__`` builds a tree but no pair list; force the
        # first ``_candidate_pairs`` call to query it.
        self._tree = None
        self._tree_pos = None
        self.wall_velocity_field = self._base_velocity_field()

    # ------------------------------------------------------------------ drive
    def _base_velocity_field(self):
        bed = self

        def field(points: np.ndarray) -> np.ndarray:
            pts = np.asarray(points, dtype=np.float64)
            out = np.zeros_like(pts)
            if bed.amplitude == 0.0:
                return out
            v = bed.base_speed()
            near = np.abs(pts[:, 1] - bed.base_position()) < 4.0 * float(bed.radii.min())
            out[near, 1] = v
            return out

        return field

    def base_position(self) -> float:
        return self.base_y0 + self.amplitude * math.sin(
            self.frequency * self.time + self.phase
        )

    def base_speed(self) -> float:
        return (
            self.amplitude
            * self.frequency
            * math.cos(self.frequency * self.time + self.phase)
        )

    def set_vibration(self, *, amplitude: float, frequency: float, phase: float = 0.0) -> None:
        """Prescribe the base motion ``A sin(w t + phase)``."""
        for name, value in (("amplitude", amplitude), ("frequency", frequency)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.phase = float(phase)
        self._move_base(self.time)

    def set_vibration_gamma(self, gamma: float, frequency: float, *, gravity: float = 9.81) -> float:
        """Prescribe ``Gamma = A w^2 / g`` at a fixed frequency; returns ``A``."""
        if not math.isfinite(gamma) or gamma < 0.0:
            raise ValueError("gamma must be finite and nonnegative")
        omega = 2.0 * math.pi * float(frequency)
        amplitude = gamma * gravity / omega**2
        self.set_vibration(amplitude=amplitude, frequency=omega)
        return amplitude

    def stop_vibration(self) -> None:
        self.amplitude = 0.0
        self.frequency = 0.0
        self._move_base(self.time)

    def open_gate(self) -> None:
        """Move the gate segment far below the container (gravity pour)."""
        if self.gate_index is None:
            raise ValueError("this bed has no gate segment")
        self.gate_open = True
        self.walls[self.gate_index, :, 1] = -10.0

    def _move_base(self, t: float) -> None:
        y = self.base_y0 + self.amplitude * math.sin(self.frequency * t + self.phase)
        self.walls[self.base_index, :, 1] = y

    # ------------------------------------------------------------ neighbours
    def _candidate_pairs(self):
        """Skinned candidate pairs, ordered by pair key and cached between rebuilds.

        Two deliberate differences from the base engine:

        * the base engine deduplicates with ``np.unique(..., axis=0)``, which
          sorts a void view of the pair matrix and costs ~70 ms per step for a
          6000-disk bed -- an order of magnitude more than the force kernel.
          ``query_pairs`` already returns each pair once (the bed has no
          periodic images), so only the key ordering is established here;
        * the pair list is **cached** until the k-d tree is rebuilt.  The skin
          guarantees that no contact can be missed while the cached list is
          valid, so re-querying the tree every step (which the base engine
          does) is pure waste.
        """
        reach = 2.0 * float(self.radii.max()) + self.config.skin
        rebuild = (
            self._tree is None
            or self._tree_pos is None
            or float(np.max(np.linalg.norm(self.pos - self._tree_pos, axis=1)))
            > 0.5 * self.config.skin
        )
        if rebuild:
            self._tree = cKDTree(self.pos)
            self._tree_pos = self.pos.copy()
            pairs = self._tree.query_pairs(reach, output_type="ndarray").reshape(-1, 2)
            if not len(pairs):
                self._pair_i = np.zeros(0, dtype=np.int64)
                self._pair_j = np.zeros(0, dtype=np.int64)
            else:
                i = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
                j = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
                order = np.argsort(i * self.n + j, kind="stable")
                self._pair_i = i[order]
                self._pair_j = j[order]
        i, j = self._pair_i, self._pair_j
        return i, j, self.pos[j] - self.pos[i]

    # ----------------------------------------------------------------- forces
    def _wall_contacts(self):
        """Disk/segment overlaps including the owning segment index.

        Only disks whose centre lies inside a segment's bounding box grown by
        ``r_max`` can touch it, so the ``(N, S)`` projection is evaluated on a
        small candidate subset instead of the whole bed (the base engine
        evaluates it for every disk and every segment).
        """
        segs = self.walls
        r_max = float(self.radii.max())
        disks: list[np.ndarray] = []
        seg_ids: list[np.ndarray] = []
        for s_i in range(len(segs)):
            a = segs[s_i, 0, :]
            b = segs[s_i, 1, :]
            lo = np.minimum(a, b) - r_max
            hi = np.maximum(a, b) + r_max
            cand = np.flatnonzero(
                (self.pos[:, 0] >= lo[0]) & (self.pos[:, 0] <= hi[0])
                & (self.pos[:, 1] >= lo[1]) & (self.pos[:, 1] <= hi[1])
            )
            if not len(cand):
                continue
            ab = b - a
            denom = max(float(ab @ ab), 1e-30)
            ap = self.pos[cand] - a[None, :]
            t = np.clip((ap @ ab) / denom, 0.0, 1.0)
            cp = a[None, :] + t[:, None] * ab[None, :]
            d = self.pos[cand] - cp
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            overlap = self.radii[cand] - dist
            hit = (overlap > 0.0) & (dist > 1e-12)
            if hit.any():
                disks.append(cand[hit])
                seg_ids.append(np.full(int(hit.sum()), s_i, dtype=np.int64))
        if not disks:
            empty_i = np.zeros(0, dtype=np.int64)
            return (empty_i, np.zeros((0, 2)), np.zeros((0, 2)),
                    np.zeros(0), np.zeros(0, dtype=np.int64))
        # keep the global ordering used by the base engine: disk-major, then seg
        order = np.lexsort((np.concatenate(seg_ids), np.concatenate(disks)))
        disk = np.concatenate(disks)[order]
        seg = np.concatenate(seg_ids)[order]
        a = segs[seg, 0, :]
        ab = segs[seg, 1, :] - a
        denom = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-30)
        ap = self.pos[disk] - a
        t = np.clip(np.einsum("ij,ij->i", ap, ab) / denom, 0.0, 1.0)
        point = a + t[:, None] * ab
        d = self.pos[disk] - point
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        overlap = self.radii[disk] - dist
        n_out = d / dist[:, None]
        return disk.astype(np.int64), point, n_out, overlap, seg.astype(np.int64)

    @staticmethod
    def _scatter2(idx: np.ndarray, vec: np.ndarray, n: int) -> np.ndarray:
        # ``np.bincount`` on an *empty* weight array returns an integer array,
        # so the dtype is pinned explicitly here.
        out = np.zeros((n, 2))
        out[:, 0] = np.bincount(idx, weights=vec[:, 0], minlength=n)
        out[:, 1] = np.bincount(idx, weights=vec[:, 1], minlength=n)
        return out

    @staticmethod
    def _scatter1(idx: np.ndarray, vec: np.ndarray, n: int) -> np.ndarray:
        out = np.zeros(n)
        out += np.bincount(idx, weights=vec, minlength=n)
        return out

    def _evaluate(self):
        """Force/torque kernel (physics identical to ``DiskSystem._evaluate``)."""
        cfg = self.config
        n = self.n

        i, j, l = self._candidate_pairs()
        dist = np.sqrt(np.einsum("ij,ij->i", l, l))
        n_ij = l / np.maximum(dist, 1e-30)[:, None]
        overlap = self.radii[i] + self.radii[j] - dist
        keep = overlap > 0.0
        if not keep.all():
            i, j, l, dist, n_ij, overlap = (
                i[keep], j[keep], l[keep], dist[keep], n_ij[keep], overlap[keep]
            )
        n_pair = len(i)

        wd, wp, wn, wo, wseg = self._wall_contacts()
        n_w = len(wd)

        disk_idx = np.concatenate([i, wd]).astype(np.int64)
        n_all = np.concatenate([-n_ij, wn])
        ov_all = np.concatenate([overlap, wo])
        m_all = len(disk_idx)
        if m_all == 0:
            self._state_keys = np.zeros(0, dtype=np.int64)
            self._state_xi = np.zeros((0, 2))
            self._state_roll = np.zeros(0)
            force = self.mass[:, None] * np.asarray(cfg.gravity, dtype=np.float64)[None, :]
            self.base_contact_disk = np.zeros(0, dtype=np.int64)
            self.base_contact_fn = np.zeros(0)
            self.base_contact_ft = np.zeros(0)
            self.base_contact_x = np.zeros(0)
            return force, np.zeros(n), np.zeros((n, 2))

        keys = np.empty(m_all, dtype=np.int64)
        keys[:n_pair] = i * n + j
        keys[n_pair:] = n * n + np.arange(n_w, dtype=np.int64)
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
        wsel = slice(n_pair, m_all)
        if n_w and self.wall_velocity_field is not None:
            v_j[wsel] = self.wall_velocity_field(wp)
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
        self._state_keys = keys
        self._state_xi = xi
        self._state_roll = roll

        total = f_n[:, None] * n_all + f_t_vec
        idx_pairs = np.concatenate([i, j])
        tot_pairs = np.concatenate([total[:n_pair], -total[:n_pair]])
        force = self._scatter2(idx_pairs, tot_pairs, n)
        tq = self._scatter1(
            idx_pairs,
            np.concatenate([
                self._cross(lever_i[:n_pair], total[:n_pair]),
                self._cross(lever_j[:n_pair], -total[:n_pair]),
            ]),
            n,
        )
        ext = np.zeros((n, 2))
        if n_w:
            wsel_slice = slice(n_pair, m_all)
            force += self._scatter2(disk_idx[wsel_slice], total[wsel_slice], n)
            tq += self._scatter1(
                disk_idx[wsel_slice],
                self._cross(lever_i[wsel_slice], total[wsel_slice]),
                n,
            )
            ext += self._scatter2(disk_idx[wsel_slice], total[wsel_slice], n)
            self.wall_force = -total[wsel_slice].sum(axis=0)
            on_base = wseg == self.base_index
            if on_base.any():
                self.base_contact_disk = disk_idx[wsel_slice][on_base]
                self.base_contact_fn = f_n[wsel_slice][on_base]
                self.base_contact_ft = f_t_mag[wsel_slice][on_base]
                self.base_contact_x = wp[on_base, 0]
            else:
                self.base_contact_disk = np.zeros(0, dtype=np.int64)
                self.base_contact_fn = np.zeros(0)
                self.base_contact_ft = np.zeros(0)
                self.base_contact_x = np.zeros(0)
        else:
            self.wall_force = np.zeros(2)
            self.base_contact_disk = np.zeros(0, dtype=np.int64)
            self.base_contact_fn = np.zeros(0)
            self.base_contact_ft = np.zeros(0)
            self.base_contact_x = np.zeros(0)

        if cfg.mu_rolling > 0.0:
            roll = roll + (self.omega[disk_idx] - omega_j) * self.dt
            k_r = 0.25 * k_t * r_eff**2
            m_r = k_r * roll
            cap_r = cfg.mu_rolling * r_eff * f_n
            scale_r = np.where(np.abs(m_r) > cap_r, cap_r / np.maximum(np.abs(m_r), 1e-30), 1.0)
            m_r *= scale_r
            roll *= scale_r
            self._state_roll = roll
            pair_idx = np.concatenate([disk_idx[:n_pair], j, disk_idx[n_pair:]])
            pair_tq = np.concatenate([-m_r[:n_pair], m_r[:n_pair], -m_r[n_pair:]])
            tq += self._scatter1(pair_idx, pair_tq, n)

        g = np.asarray(cfg.gravity, dtype=np.float64)
        if np.any(g != 0.0):
            force += self.mass[:, None] * g[None, :]
        if cfg.drag_time > 0.0:
            v_drift = np.asarray(cfg.drift_velocity, dtype=np.float64)
            force += self.mass[:, None] * (v_drift[None, :] - self.vel) / cfg.drag_time
        return force, tq, ext

    # ------------------------------------------------------------------- time
    def step(self, n: int = 1, *, check: bool = True) -> None:
        """Advance ``n`` semi-implicit Euler steps with the base drive active."""
        if n < 0:
            raise ValueError("n must be nonnegative")
        cfg = self.config
        inv_m = 1.0 / self.mass
        inv_i = 1.0 / self.inertia
        damping_time = cfg.damping_time
        for _ in range(n):
            if self.amplitude != 0.0 or self.frequency != 0.0:
                self._move_base(self.time)
            force, torque, _ = self._evaluate()
            self.vel += self.dt * (force * inv_m[:, None])
            self.omega += self.dt * (torque * inv_i)
            if damping_time > 0.0:
                decay = math.exp(-self.dt / damping_time)
                self.vel *= decay
                self.omega *= decay
            self.pos += self.dt * self.vel
            self.theta += self.dt * self.omega
            self.time += self.dt
            self.steps += 1
        if check:
            self.check_containment()

    def check_containment(self, tol: float = 0.0) -> None:
        """Fail loudly if any particle has left the trough (numerical blow-up)."""
        x_lo = float(self.walls[1, 0, 0])
        x_hi = float(self.walls[2, 0, 0])
        r = self.radii
        bad = (
            (self.pos[:, 0] < x_lo - r - tol)
            | (self.pos[:, 0] > x_hi + r + tol)
            | (self.pos[:, 1] < -r - tol)
            | ~np.isfinite(self.pos).all(axis=1)
            | ~np.isfinite(self.vel).all(axis=1)
        )
        if bad.any():
            raise BoxTooSmall(
                f"{int(bad.sum())} particle(s) left the container or became non-finite "
                f"(first index {int(np.flatnonzero(bad)[0])})"
            )

    # ------------------------------------------------------------ diagnostics
    def area_fraction(self) -> float:
        """Solid area fraction of the current bed."""
        return float(np.sum(self.measure) / (self.domain_area() or 1.0))

    def kinetic_energy(self) -> float:
        return float(
            np.sum(0.5 * self.mass * np.sum(self.vel**2, axis=1))
            + np.sum(0.5 * self.inertia * self.omega**2)
        )

    def granular_temperature(self) -> float:
        """Fluctuation kinetic energy per particle (m^2/s^2)."""
        v = self.vel - np.average(self.vel, axis=0, weights=self.mass)
        return float(np.mean(np.sum(v**2, axis=1)))

    def species_mean_height(self) -> tuple[float, float]:
        return (
            float(np.average(self.pos[self.is_large, 1])),
            float(np.average(self.pos[~self.is_large, 1])),
        )