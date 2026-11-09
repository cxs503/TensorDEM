"""Silo / hopper DEM kernel with *rough* walls and a closed energy ledger.

Why a new kernel instead of :mod:`tensordem.silo_discharge`?
-----------------------------------------------------------
``silo_discharge`` (m5481) is a working flat-bottom DEM, but its own audit
(``docs/silo_discharge_beverloo_verification.md``) lists two limitations that
block the verification chain requested here:

* the wall/plate friction is *kinetic only* (``mu * |Fn| * vt/|vt|``) with no
  Cundall--Strack static branch, so a silo can not hold a Janssen-type
  frictional wall stress, and the mass/funnel flow transition can not be
  represented;
* the only packing generator is a near-perfect HCP lattice, which arches over
  a 3--5 d orifice and stops the discharge (the measured flow rate collapsed
  to zero after the start-up burst).

Both are supplied here: Cundall--Strack wall friction with a per-surface
tangential history, a disordered packing generator
(:mod:`tensordem.silo_flow.packing`), and a per-timestep energy ledger so the
integration itself can be audited (stage 4 of the verification chain).

Contact laws (all SI: m, kg, s, N, rad)
---------------------------------------
* normal: linear spring--dashpot ``f_n = max(k_n d - gamma v_n, 0)`` with
  ``gamma = 2 zeta sqrt(k_n m_eff)``, ``zeta = |ln e| / sqrt(pi^2 + ln^2 e)``;
* tangential: Cundall--Strack incremental spring, saturated at ``mu f_n``
  (the saturated part is booked as dissipated energy, so the ledger is exact);
* rolling: Coulomb torque ``|M| <= mu_r R_eff f_n`` opposing the relative spin
  (purely dissipative -> no history, exact dissipation bookkeeping);
* walls: analytic nearest-feature contacts for a cylinder, a flat circular
  plate with an orifice (top face + rim circle) and a conical hopper wall,
  each with its own Cundall--Strack tangential history, so wall friction can
  hold the material (``mu_wall`` is a first-class parameter).

The geometry is axisymmetric in ``z``.  For a *flat-bottom* silo the plate lies
at ``z = 0`` with a hole of radius ``outlet_radius``; for a *conical hopper* the
cone runs from the outlet (``z = 0``, radius ``outlet_radius``) up to
``z = h_cone`` (radius ``cylinder_radius``) with half angle
``hopper_half_angle_deg`` measured from the vertical axis, and a cylindrical
section of height ``cylinder_height`` sits on top.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "Material",
    "SiloGeometry",
    "SiloFlow",
    "damping_ratio",
    "contact_time",
]


def damping_ratio(restitution: float) -> float:
    """``zeta = |ln e| / sqrt(pi^2 + ln^2 e)`` (dashpot from restitution)."""
    e = min(max(float(restitution), 1e-6), 1.0 - 1e-12)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)


def contact_time(mass_eff: float, kn: float) -> float:
    """Linear-spring contact period ``T = 2 pi sqrt(m_eff / k_n)``."""
    return 2.0 * math.pi * math.sqrt(mass_eff / kn)


@dataclass(frozen=True)
class Material:
    """Monodisperse sphere material + contact parameters (SI)."""

    radius: float = 0.002
    density: float = 2500.0
    mu: float = 0.4
    mu_rolling: float = 0.05
    restitution: float = 0.3
    kn: float = 1.0e4
    kt_ratio: float = 0.5
    gravity: float = 9.81

    def __post_init__(self) -> None:
        for name in ("radius", "density", "mu", "restitution", "kn", "kt_ratio", "gravity"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not (0.0 < self.restitution < 1.0):
            raise ValueError("restitution must lie in (0, 1)")
        if self.mu_rolling < 0.0 or not math.isfinite(self.mu_rolling):
            raise ValueError("mu_rolling must be finite and nonnegative")

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    @property
    def mass_eff(self) -> float:
        return 0.5 * self.mass

    @property
    def gamma_n(self) -> float:
        return 2.0 * damping_ratio(self.restitution) * math.sqrt(self.kn * self.mass_eff)

    @property
    def gamma_t(self) -> float:
        """Tangential dashpot (same damping ratio, tangential stiffness)."""
        return 2.0 * damping_ratio(self.restitution) * math.sqrt(self.kt * self.mass_eff)

    @property
    def dt(self) -> float:
        """20 steps per contact period (linear-spring stability limit)."""
        return 0.05 * contact_time(self.mass_eff, self.kn)


@dataclass(frozen=True)
class SiloGeometry:
    """Axisymmetric silo: outlet + (cone or flat plate) + cylindrical section."""

    cylinder_radius: float = 0.045
    cylinder_height: float = 0.11
    outlet_radius: float = 0.008
    hopper_half_angle_deg: float | None = None  # None -> flat bottom plate

    def __post_init__(self) -> None:
        for name in ("cylinder_radius", "cylinder_height", "outlet_radius"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.outlet_radius >= self.cylinder_radius:
            raise ValueError("outlet_radius must be smaller than cylinder_radius")
        if self.hopper_half_angle_deg is not None:
            ang = float(self.hopper_half_angle_deg)
            if not (0.0 < ang < 89.0):
                raise ValueError("hopper_half_angle_deg must lie in (0, 89)")

    @property
    def is_hopper(self) -> bool:
        return self.hopper_half_angle_deg is not None

    @property
    def _tan(self) -> float:
        return math.tan(math.radians(float(self.hopper_half_angle_deg)))

    @property
    def cone_height(self) -> float:
        """Height of the conical section measured from the outlet plane."""
        if not self.is_hopper:
            return 0.0
        return (self.cylinder_radius - self.outlet_radius) / self._tan

    @property
    def cone_apex_z(self) -> float:
        """``z`` of the virtual apex (below the outlet plane)."""
        if not self.is_hopper:
            return 0.0
        return -self.outlet_radius / self._tan

    @property
    def z_top(self) -> float:
        return self.cone_height + self.cylinder_height

    def radius_at(self, z) -> np.ndarray:
        """Inner wall radius at height ``z`` (scalar or array)."""
        z_arr = np.asarray(z, dtype=np.float64)
        if not self.is_hopper:
            return np.where(z_arr > 0.0, self.cylinder_radius, self.outlet_radius)
        # the cone wall is  rho = (z - z_apex) tan(theta);  at z = 0 this is
        # exactly the outlet radius, so the outlet radius must NOT be added a
        # second time (doing so made radius_at() wider than the wall and the
        # lattice generator placed particles outside the hopper)
        cone = np.maximum(z_arr - self.cone_apex_z, 0.0) * self._tan
        return np.where(z_arr >= self.cone_height, self.cylinder_radius,
                        np.minimum(cone, self.cylinder_radius))

    def volume_below(self, z: float) -> float:
        """Nominal container volume below height ``z`` (for the bulk density)."""
        if z <= 0.0:
            return 0.0
        if not self.is_hopper:
            return math.pi * self.cylinder_radius ** 2 * z
        zc = self.cone_height
        if z <= zc:
            r0 = self.outlet_radius
            r1 = float(self.radius_at(z))
            return math.pi * z / 3.0 * (r0 * r0 + r0 * r1 + r1 * r1)
        r0 = self.outlet_radius
        r1 = self.cylinder_radius
        vol_cone = math.pi * zc / 3.0 * (r0 * r0 + r0 * r1 + r1 * r1)
        return vol_cone + math.pi * self.cylinder_radius ** 2 * (z - zc)


def _accumulate(dst: np.ndarray, idx: np.ndarray, vals: np.ndarray, n: int) -> None:
    """``dst[idx] += vals`` via ``bincount`` (much faster than ``np.add.at``)."""
    if idx.size == 0:
        return
    for c in range(3):
        dst[:, c] += np.bincount(idx, weights=vals[:, c], minlength=n)


class SiloFlow:
    """3-D DEM silo/hopper: state, contact forces, energy ledger."""

    SURF_CONE = 1
    SURF_PLATE = 2
    SURF_RIM = 3
    SURF_CYL = 4

    def __init__(
        self,
        pos,
        material: Material,
        geometry: SiloGeometry,
        *,
        wall_mu: float = 0.3,
        dt: float | None = None,
        vel=None,
        plug_outlet: bool = False,
    ) -> None:
        pos = np.ascontiguousarray(np.asarray(pos, dtype=np.float64))
        if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) == 0:
            raise ValueError("pos must have shape (N, 3) with N >= 1")
        if not np.isfinite(pos).all():
            raise ValueError("pos must be finite")
        if wall_mu < 0.0 or not math.isfinite(wall_mu):
            raise ValueError("wall_mu must be finite and nonnegative")
        self.mat = material
        self.geom = geometry
        self.wall_mu = float(wall_mu)
        self.plugged = bool(plug_outlet)
        self.dt = float(dt) if dt is not None else material.dt
        if self.dt <= 0.0 or self.dt > 0.25 * contact_time(material.mass_eff, material.kn):
            raise ValueError("dt exceeds the linear-spring stability limit")

        n = len(pos)
        self.pos = pos
        self.vel = (np.zeros((n, 3)) if vel is None
                    else np.ascontiguousarray(np.asarray(vel, dtype=np.float64)).copy())
        self.omega = np.zeros((n, 3))
        self.radii = np.full(n, material.radius)
        self.mass = np.full(n, material.mass)
        self.inertia = np.full(n, material.inertia)

        self.time = 0.0
        self.steps = 0
        self.n_initial = n
        self._key_base = n
        self.n_exited = 0
        self.mass_exited = 0.0

        # pair tangential history (sorted key = i * _key_base + j)
        self._keys = np.zeros(0, dtype=np.int64)
        self._xi = np.zeros((0, 3))
        self.nbr_interval = 20
        self.nbr_skin = 0.6
        self._cand = None
        self._cand_step = -10 ** 9
        self._wall_xi = {tag: np.zeros((n, 3)) for tag in (1, 2, 3, 4)}
        self._wall_active = {tag: np.zeros(n, dtype=bool) for tag in (1, 2, 3, 4)}

        # energy ledger
        self.e_kin = 0.0
        self.e_grav = 0.0
        self.e_elast = 0.0
        self.dissipated = 0.0
        self.e_exit = 0.0
        self.e_ref = 1e-30
        self.e_initial = 0.0
        self.residual_max = 0.0
        self.residual_series: list[float] = []
        self._E_before = 0.0
        self._compute_kinetic_potential()
        self._E_before = self.e_kin + self.e_grav
        self.e_initial = self._E_before
        self.e_ref = max(abs(self._E_before), 1e-30)

        # diagnostics
        self.wall_normal_force = np.zeros(n)
        self.plate_normal_force = np.zeros(n)
        self.wall_z_force = np.zeros(n)
        self.wall_force_by_tag: dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

    # ------------------------------------------------------------------ props
    @property
    def n(self) -> int:
        return len(self.pos)

    @property
    def bed_mass(self) -> float:
        return float(np.sum(self.mass))

    def bulk_density(self, fill_height: float | None = None) -> float:
        """Geometric bulk density ``mass / container volume below the fill line``."""
        if fill_height is None:
            fill_height = float(np.max(self.pos[:, 2]))
        vol = self.geom.volume_below(fill_height)
        if vol <= 0.0:
            raise ValueError("fill volume must be positive")
        return self.bed_mass / vol

    # ---------------------------------------------------------------- helpers
    def _expand(self, n: int) -> None:
        pad = n - self.n
        if pad <= 0:
            return
        self.vel = np.vstack([self.vel, np.zeros((pad, 3))])
        self.omega = np.vstack([self.omega, np.zeros((pad, 3))])
        self.radii = np.concatenate([self.radii, np.full(pad, self.mat.radius)])
        self.mass = np.concatenate([self.mass, np.full(pad, self.mat.mass)])
        self.inertia = np.concatenate([self.inertia, np.full(pad, self.mat.inertia)])
        for tag in self._wall_xi:
            self._wall_xi[tag] = np.vstack([self._wall_xi[tag], np.zeros((pad, 3))])
            self._wall_active[tag] = np.concatenate(
                [self._wall_active[tag], np.zeros(pad, dtype=bool)]
            )
        self.wall_normal_force = np.concatenate([self.wall_normal_force, np.zeros(pad)])
        self.plate_normal_force = np.concatenate([self.plate_normal_force, np.zeros(pad)])
        self.wall_z_force = np.concatenate([self.wall_z_force, np.zeros(pad)])

    def add_particles(self, pos) -> None:
        """Append particles (used by the layer-wise filling of the Janssen test)."""
        pos = np.ascontiguousarray(np.asarray(pos, dtype=np.float64))
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("pos must have shape (M, 3)")
        n0 = self.n
        self._expand(n0 + len(pos))
        self.pos = np.vstack([self.pos, pos])
        self.vel[n0:] = 0.0
        self.omega[n0:] = 0.0
        self._reset_history()

    def _reset_history(self) -> None:
        n = self.n
        self._cand = None
        self._cand_step = -10 ** 9
        self._key_base = n
        self._keys = np.zeros(0, dtype=np.int64)
        self._xi = np.zeros((0, 3))
        for tag in self._wall_xi:
            self._wall_xi[tag] = np.zeros((n, 3))
            self._wall_active[tag] = np.zeros(n, dtype=bool)

    def _remove(self, mask_remove: np.ndarray) -> None:
        keep = ~mask_remove
        if keep.all():
            return
        self.pos = self.pos[keep]
        self.vel = self.vel[keep]
        self.omega = self.omega[keep]
        self.radii = self.radii[keep]
        self.mass = self.mass[keep]
        self.inertia = self.inertia[keep]
        self.wall_normal_force = self.wall_normal_force[keep]
        self.plate_normal_force = self.plate_normal_force[keep]
        self.wall_z_force = self.wall_z_force[keep]
        self._reset_history()

    # ------------------------------------------------------------ pair search
    def _pairs(self, force: bool = False):
        n = self.n
        if n < 2:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 3))
        r = self.radii
        rr = float(r.max())
        if force or self._cand is None or (self.steps - self._cand_step) >= self.nbr_interval:
            tree = cKDTree(self.pos)
            self._cand = tree.query_pairs(2.0 * rr * (1.0 + self.nbr_skin), output_type="ndarray")
            self._cand_step = self.steps
        pairs = self._cand
        if pairs.size == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 3))
        i = pairs[:, 0].astype(np.int64)
        j = pairs[:, 1].astype(np.int64)
        nvec = self.pos[i] - self.pos[j]
        dist = np.linalg.norm(nvec, axis=1)
        nvec = nvec / np.maximum(dist, 1e-30)[:, None]
        delta = r[i] + r[j] - dist
        keep = delta > 0.0
        return i[keep], j[keep], delta[keep], nvec[keep]

    def _carry_pair_history(self, keys: np.ndarray) -> np.ndarray:
        if self._keys.size == 0 or keys.size == 0:
            return np.zeros((len(keys), 3))
        idx = np.searchsorted(self._keys, keys)
        idx_c = np.clip(idx, 0, self._keys.size - 1)
        found = self._keys[idx_c] == keys
        return np.where(found[:, None], self._xi[idx_c], 0.0)

    # ------------------------------------------------------------- wall model
    def _wall_contacts(self):
        """Per-surface wall contact candidates: ``tag``, ``idx``, ``n``, ``delta``."""
        out = []
        if self.n == 0:
            return out
        p = self.pos
        r = self.radii
        g = self.geom
        rho = np.hypot(p[:, 0], p[:, 1])
        rho_safe = np.maximum(rho, 1e-30)
        e_rho = np.stack([p[:, 0] / rho_safe, p[:, 1] / rho_safe, np.zeros(self.n)], axis=1)
        zc = g.cone_height

        if g.is_hopper:
            theta = math.radians(float(g.hopper_half_angle_deg))
            sin_t, cos_t = math.sin(theta), math.cos(theta)
            phi = (p[:, 2] - g.cone_apex_z) * sin_t - rho * cos_t
            delta = r - phi                     # distance from the wall = phi
            # the cone exists only above the outlet plane: below z = 0 the
            # 'virtual' extension of the cone must not act (it would put a
            # several-radius penetration on any particle leaving the outlet)
            mask = (delta > 0.0) & (p[:, 2] < zc) & (p[:, 2] >= 0.0)
            if np.any(mask):
                idx = np.flatnonzero(mask)
                out.append({
                    "tag": self.SURF_CONE, "idx": idx,
                    "n": -cos_t * e_rho[idx] + sin_t * np.array([0.0, 0.0, 1.0]),
                    "delta": delta[idx],
                })
            if self.plugged:
                delta_plug = r - p[:, 2]
                mask_plug = (delta_plug > 0.0) & (rho <= g.outlet_radius) & (p[:, 2] >= -r)
                if np.any(mask_plug):
                    idx = np.flatnonzero(mask_plug)
                    out.append({
                        "tag": self.SURF_PLATE, "idx": idx,
                        "n": np.broadcast_to(np.array([0.0, 0.0, 1.0]), (len(idx), 3)),
                        "delta": delta_plug[idx],
                    })
        else:
            a = g.outlet_radius
            delta_top = r - p[:, 2]
            if self.plugged:
                mask_top = (delta_top > 0.0) & (p[:, 2] >= -r)
                if np.any(mask_top):
                    idx = np.flatnonzero(mask_top)
                    out.append({
                        "tag": self.SURF_PLATE, "idx": idx,
                        "n": np.broadcast_to(np.array([0.0, 0.0, 1.0]), (len(idx), 3)),
                        "delta": delta_top[idx],
                    })
                z_lo = 0.0
                delta_cyl = rho + r - g.cylinder_radius
                mask_cyl = (delta_cyl > 0.0) & (p[:, 2] > z_lo)
                if np.any(mask_cyl):
                    idx = np.flatnonzero(mask_cyl)
                    out.append({"tag": self.SURF_CYL, "idx": idx,
                                "n": -e_rho[idx], "delta": delta_cyl[idx]})
                return out
            mask_top = (delta_top > 0.0) & (rho >= a) & (p[:, 2] >= -r)
            dr = rho - a
            rim_dist = np.hypot(dr, p[:, 2])
            delta_rim = r - rim_dist
            mask_rim = (delta_rim > 0.0) & (rim_dist > 1e-12)
            take_rim = mask_rim & (~mask_top | (delta_rim > delta_top))
            take_top = mask_top & ~take_rim
            if np.any(take_top):
                idx = np.flatnonzero(take_top)
                out.append({
                    "tag": self.SURF_PLATE, "idx": idx,
                    "n": np.broadcast_to(np.array([0.0, 0.0, 1.0]), (len(idx), 3)),
                    "delta": delta_top[idx],
                })
            if np.any(take_rim):
                idx = np.flatnonzero(take_rim)
                d = np.maximum(rim_dist[idx], 1e-30)
                out.append({
                    "tag": self.SURF_RIM, "idx": idx,
                    "n": np.stack([dr[idx] / d * e_rho[idx, 0],
                                   dr[idx] / d * e_rho[idx, 1],
                                   p[idx, 2] / d], axis=1),
                    "delta": delta_rim[idx],
                })

        z_lo = zc if g.is_hopper else 0.0
        delta_cyl = rho + r - g.cylinder_radius        # r - (R - rho)
        mask_cyl = (delta_cyl > 0.0) & (p[:, 2] > z_lo) & (p[:, 2] < g.z_top + 4.0 * self.mat.radius)
        if np.any(mask_cyl):
            idx = np.flatnonzero(mask_cyl)
            out.append({
                "tag": self.SURF_CYL, "idx": idx,
                "n": -e_rho[idx], "delta": delta_cyl[idx],
            })
        return out

    # ------------------------------------------------------------------- step
    def step(self, *, measure: bool = True) -> None:
        mat = self.mat
        n = self.n
        if n == 0:
            return
        dt = self.dt
        kn, kt, gamma_n, gamma_t = mat.kn, mat.kt, mat.gamma_n, mat.gamma_t
        mu, mu_r = mat.mu, mat.mu_rolling

        i, j, delta, nvec = self._pairs()
        keys = i * np.int64(self._key_base) + j
        xi = self._carry_pair_history(keys)
        if len(i):
            xi = xi - np.sum(xi * nvec, axis=1)[:, None] * nvec

        force = np.zeros((n, 3))
        torque = np.zeros((n, 3))
        diss = 0.0
        e_elast = 0.0

        # relative velocity of i w.r.t. j at the contact point
        v_rel = self.vel[i] - self.vel[j]
        spin_cross = np.cross(self.omega[i] + self.omega[j], nvec)
        v_rel = v_rel - 0.5 * (self.radii[i] + self.radii[j])[:, None] * spin_cross
        v_n = np.sum(v_rel * nvec, axis=1)
        f_n = kn * delta - gamma_n * v_n
        contact = f_n > 0.0
        f_n = np.where(contact, f_n, 0.0)
        diss += float(np.sum(gamma_n * v_n[contact] ** 2) * dt)

        v_t = v_rel - v_n[:, None] * nvec
        xi = xi + v_t * dt
        f_t = -kt * xi - gamma_t * v_t
        diss += float(np.sum(gamma_t * np.sum(v_t ** 2, axis=1)[contact]) * dt)
        f_t_mag = np.linalg.norm(f_t, axis=1)
        cap = mu * f_n
        sliding = f_t_mag > cap
        if np.any(sliding):
            scale = np.where(sliding, cap / np.maximum(f_t_mag, 1e-30), 1.0)
            xi_pre_t = f_t_mag / kt
            xi_post_t = f_t_mag * scale / kt
            f_t = f_t * scale[:, None]
            xi = xi * scale[:, None]
            diss += float(np.sum(0.5 * kt * (xi_pre_t ** 2 - xi_post_t ** 2)))
        e_elast += float(np.sum(0.5 * kn * delta ** 2) + np.sum(0.5 * kt * np.sum(xi ** 2, axis=1)))

        f_pair = f_n[:, None] * nvec + f_t
        if len(i):
            _accumulate(force, i, f_pair, n)
            _accumulate(force, j, -f_pair, n)
            # both particles see the *same* spin torque: the lever arm and the
            # force both flip sign at the other body, so tau_j = tau_i.  Using
            # -tq (the textbook sign error) adds angular momentum at every
            # contact and spins a static bed up to thousands of rad/s.
            r_c = 0.5 * (self.radii[i] + self.radii[j])
            tq = np.cross(-r_c[:, None] * nvec, f_t)
            _accumulate(torque, i, tq, n)
            _accumulate(torque, j, tq, n)

            w_rel = self.omega[i] - self.omega[j]
            w_mag = np.linalg.norm(w_rel, axis=1)
            r_eff = self.radii[i] * self.radii[j] / np.maximum(self.radii[i] + self.radii[j], 1e-30)
            m_roll = mu_r * r_eff * f_n
            active = (w_mag > 1e-14) & (m_roll > 0.0)
            if np.any(active):
                tq_r = np.zeros_like(w_rel)
                tq_r[active] = (m_roll[active] / w_mag[active])[:, None] * w_rel[active]
                _accumulate(torque, i, -tq_r, n)
                _accumulate(torque, j, tq_r, n)
                diss += float(np.sum(m_roll[active] * w_mag[active]) * dt)
            order = np.argsort(keys, kind="stable")
            self._keys = keys[order]
            self._xi = xi[order]
        else:
            self._keys = np.zeros(0, dtype=np.int64)
            self._xi = np.zeros((0, 3))

        # ---- walls
        wall_norm = np.zeros(n)
        plate_norm = np.zeros(n)
        wall_z_tot = np.zeros(n)
        wall_by_tag = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for wc in self._wall_contacts():
            tag, idx, nv, dl = wc["tag"], wc["idx"], wc["n"], wc["delta"]
            hist = self._wall_xi[tag]
            act = self._wall_active[tag]
            xi_w = np.where(act[idx][:, None], hist[idx], 0.0)
            xi_w = xi_w - np.sum(xi_w * nv, axis=1)[:, None] * nv
            # surface velocity at the wall contact point: v - r (omega x n)
            vr = self.vel[idx] - self.radii[idx][:, None] * np.cross(self.omega[idx], nv)
            v_nw = np.sum(vr * nv, axis=1)
            f_nw = kn * dl - gamma_n * v_nw
            cont = f_nw > 0.0
            f_nw = np.where(cont, f_nw, 0.0)
            diss += float(np.sum(gamma_n * v_nw[cont] ** 2) * dt)
            v_tw = vr - v_nw[:, None] * nv
            xi_w = xi_w + v_tw * dt
            f_tw = -kt * xi_w - gamma_t * v_tw
            diss += float(np.sum(gamma_t * np.sum(v_tw ** 2, axis=1)[cont]) * dt)
            f_mag = np.linalg.norm(f_tw, axis=1)
            cap_w = self.wall_mu * f_nw
            slide = f_mag > cap_w
            if np.any(slide):
                scale = np.where(slide, cap_w / np.maximum(f_mag, 1e-30), 1.0)
                pre = f_mag / kt
                post = f_mag * scale / kt
                f_tw = f_tw * scale[:, None]
                xi_w = xi_w * scale[:, None]
                diss += float(np.sum(0.5 * kt * (pre ** 2 - post ** 2)))
            f_wall = f_nw[:, None] * nv + f_tw
            _accumulate(force, idx, f_wall, n)
            r_p = self.radii[idx]
            _accumulate(torque, idx, np.cross(-r_p[:, None] * nv, f_tw), n)
            wall_norm[idx] += f_nw
            wall_z_tot[idx] += f_wall[:, 2]
            if tag == self.SURF_PLATE:
                plate_norm[idx] += f_nw * nv[:, 2]
            hist[idx] = xi_w
            new_active = np.zeros(n, dtype=bool)
            new_active[idx] = cont
            self._wall_active[tag] = new_active
            wall_by_tag[tag] += float(np.sum(f_nw))
            e_elast += float(np.sum(0.5 * kn * dl ** 2) + np.sum(0.5 * kt * np.sum(xi_w ** 2, axis=1)))

        self.wall_normal_force = wall_norm
        self.plate_normal_force = plate_norm
        self.wall_z_force = wall_z_tot
        self.wall_force_by_tag = wall_by_tag

        # ---- integrate (semi-implicit Euler)
        force[:, 2] -= self.mass * mat.gravity
        self.vel = self.vel + (force / self.mass[:, None]) * dt
        self.omega = self.omega + (torque / self.inertia[:, None]) * dt
        self.pos = self.pos + self.vel * dt
        self.time += dt
        self.steps += 1
        self.dissipated += diss
        self.e_elast = e_elast

        # ---- outlet: particles that have left the domain
        gone = self.pos[:, 2] < -3.0 * mat.radius
        if np.any(gone):
            e_exit = float(
                np.sum(0.5 * self.mass[gone] * np.sum(self.vel[gone] ** 2, axis=1))
                + np.sum(0.5 * self.inertia[gone] * np.sum(self.omega[gone] ** 2, axis=1))
                + np.sum(self.mass[gone] * mat.gravity * self.pos[gone, 2])
            )
            self.e_exit += e_exit
            self.mass_exited += float(np.sum(self.mass[gone]))
            self.n_exited += int(np.sum(gone))
            self._remove(gone)

        # ---- energy ledger (per timestep)
        e_before = self._E_before
        self._compute_kinetic_potential()
        e_now = self.e_kin + self.e_grav + self.e_elast
        if measure:
            residual = (e_now - e_before + diss) / self.e_ref
            self.residual_series.append(residual)
            self.residual_max = max(self.residual_max, abs(residual))
        self._E_before = e_now

    # ---------------------------------------------------------------- energy
    def _compute_kinetic_potential(self) -> None:
        if self.n == 0:
            self.e_kin = self.e_grav = 0.0
            return
        self.e_kin = float(
            np.sum(0.5 * self.mass * np.sum(self.vel ** 2, axis=1))
            + np.sum(0.5 * self.inertia * np.sum(self.omega ** 2, axis=1))
        )
        self.e_grav = float(np.sum(self.mass * self.mat.gravity * self.pos[:, 2]))

    def elastic_energy(self) -> float:
        """Spring energy of the current contact set (normal + tangential)."""
        kn, kt = self.mat.kn, self.mat.kt
        i, j, delta, nvec = self._pairs()
        total = float(np.sum(0.5 * kn * delta ** 2)) if len(i) else 0.0
        if len(i):
            xi = self._carry_pair_history(i * np.int64(self._key_base) + j)
            total += float(np.sum(0.5 * kt * np.sum(xi ** 2, axis=1)))
        for wc in self._wall_contacts():
            tag, idx, dl = wc["tag"], wc["idx"], wc["delta"]
            total += float(np.sum(0.5 * kn * dl ** 2))
            act = self._wall_active[tag][idx]
            if np.any(act):
                total += float(np.sum(0.5 * kt * np.sum(self._wall_xi[tag][idx][act] ** 2, axis=1)))
        return total

    def balance(self) -> dict:
        """Current energy-balance statement (SI, J)."""
        total = self.e_kin + self.e_grav + self.e_elast + self.dissipated + self.e_exit
        return {
            "time": self.time,
            "steps": self.steps,
            "n": self.n,
            "e_kin": self.e_kin,
            "e_grav": self.e_grav,
            "e_elastic": self.e_elast,
            "dissipated": self.dissipated,
            "e_exit": self.e_exit,
            "e_initial": self.e_initial,
            "closure_rel": (total - self.e_initial) / max(abs(self.e_initial), 1e-30),
            "residual_max_per_step": self.residual_max,
        }

    # ---------------------------------------------------------------- probes
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.vel, axis=1)

    def radial_coord(self) -> np.ndarray:
        return np.hypot(self.pos[:, 0], self.pos[:, 1])

    def wall_stress_profile(self, z_edges) -> dict:
        """Wall normal stress per horizontal layer from the wall contact forces.

        ``sigma_w(z) = sum|f_n| / (2 pi R(z) dz)``: the horizontal stress the bed
        exerts on the wall at that depth (Janssen's ``sigma_h``).
        """
        z_edges = np.asarray(z_edges, dtype=np.float64)
        nlay = len(z_edges) - 1
        idx = np.digitize(self.pos[:, 2], z_edges) - 1
        area = np.zeros(nlay)
        for k in range(nlay):
            dz = z_edges[k + 1] - z_edges[k]
            r_w = float(self.geom.radius_at(0.5 * (z_edges[k] + z_edges[k + 1])))
            area[k] = 2.0 * math.pi * r_w * dz
        force = np.zeros(nlay)
        count = np.zeros(nlay)
        for k in range(nlay):
            sel = idx == k
            force[k] = float(np.sum(self.wall_normal_force[sel]))
            count[k] = float(np.sum(sel))
        return {
            "z_edges": z_edges,
            "z_centers": 0.5 * (z_edges[:-1] + z_edges[1:]),
            "sigma_w": force / np.maximum(area, 1e-30),
            "force": force,
            "count": count,
            "area": area,
        }


    def vertical_stress_profile(self, z_edges) -> dict:
        """Vertical stress per layer from the *force balance* of the slice above.

        For a static bed the material above the plane ``z`` has weight
        ``W(z) = sum_{p: z_p > z} m g`` which is carried by the plane itself and
        by the vertical wall traction ``sum f_wall,z`` (upward positive).  Hence

            sigma_zz(z) = [ W(z) - sum_{z_p > z} f_wall,z ] / A(z)

        with ``A(z) = pi R(z)^2``.  This is an independent measurement of
        Janssen's ``sigma_v``: it uses only a force balance, no stress tensor
        assumption.
        """
        z_edges = np.asarray(z_edges, dtype=np.float64)
        nlay = len(z_edges) - 1
        p = self.pos
        z = p[:, 2]
        wz = self.mass * self.mat.gravity
        sigma = np.zeros(nlay)
        weight = np.zeros(nlay)
        wall = np.zeros(nlay)
        for k in range(nlay):
            above = z > z_edges[k + 1]
            weight[k] = float(np.sum(wz[above]))
            wall[k] = float(np.sum(self.wall_z_force[above]))
            r_w = float(self.geom.radius_at(z_edges[k + 1]))
            area = math.pi * r_w ** 2
            sigma[k] = (weight[k] - wall[k]) / max(area, 1e-30)
        free_surface = float(np.max(z)) if p.size else 0.0
        return {
            "z_edges": z_edges,
            "z_upper": z_edges[1:],
            "sigma_v": sigma,
            "weight": weight,
            "wall_z": wall,
            "free_surface": free_surface,
            "depth": free_surface - z_edges[1:],
        }

    def base_pressure(self) -> float:
        """Mean normal pressure on the flat bottom plate outside the orifice."""
        if self.geom.is_hopper:
            raise ValueError("base_pressure is defined for a flat bottom only")
        a = self.geom.outlet_radius
        area = math.pi * (self.geom.cylinder_radius ** 2 - a ** 2)
        if area <= 0.0:
            raise ValueError("plate area must be positive")
        return float(np.sum(self.plate_normal_force)) / area

    def hopper_wall_force(self) -> float:
        return float(self.wall_force_by_tag.get(self.SURF_CONE, 0.0))