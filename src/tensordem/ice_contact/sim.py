"""Ice-layer versus rigid-structure crushing DEM with a wall pressure field.

The experiment
--------------
A rigid **structure plane** sits at ``x = 0`` with outward normal ``+x`` into the
ice.  An ice layer occupies ``x in [0, Lx]``, ``y in [0, B]``, ``z in [0, H]``
(``z`` vertical, ``H = h_ice``) and a platen at ``x = Lx`` is driven towards the
structure at the indentation speed ``v``.  The layer therefore crushes against
the structure and extrudes sideways/upwards, which is the loading that
produces the local ice pressure on a hull or a jacket leg.

Contact laws (identical in the two shape models unless stated)
--------------------------------------------------------------
*Normal.*  Elastic spring with a **crushing cap**,

    ``F_n = clip(k_n * delta + c * (-v_n), 0, sigma_c * A)``,

i.e. the contact pressure ``F_n / A`` can never exceed the ice crushing strength
``sigma_c``.  ``A`` is the tributary facet area of the contacting vertex (a
polyhedron) or the projected area ``pi R^2`` (a sphere); ``delta`` is the
penetration.  The cap is what bounds the pressure field: without it the peak
pressure would be a discretisation artefact.  The elastic stiffness is a
soft-particle DEM stiffness (chosen so that the cap is reached at a few percent
of a radius), which is standard practice and is *stated*, not hidden.
*Tangential.*  Regularised Coulomb friction ``F_t = -mu F_n tanh(|v_t| / v_s)``
with ``v_s = 0.02 m/s``; no history variable, which is appropriate for the
continuously slipping crushing flow (and keeps the pair list stateless).
*Rolling.*  A sphere proxy carries the rolling-resistance couple
``M = -mu_r R F_n omega_hat_rel`` (the ``mu_r R F_n`` law used elsewhere in this
repository, see ``docs/dem3d_rolling_resistance_verification.md``).  A faceted
fragment carries ``mu_r = 0``: its resistance to rolling is *emergent* from
multi-point vertex contact and clump interlocking.

Particle-particle contacts use the multi-sphere clump of
:class:`~tensordem.ice_contact.shapes.FragmentShape` (one sphere per fragment for
the spherical proxy, ``K`` spheres for a polyhedron), so the two models differ
both at the structure interface (facet bearing versus a single sphere) and in
the interior (interlocking versus a single contact point).

Gravity is deliberately absent: the ice band is treated as neutrally buoyant
(the hydrostatic support of the surrounding water is not resolved), which is the
standard idealisation for a submerged contact band and keeps the initial state
free of a settling transient.

The output that matters is the **pressure field on the structure**
``p(y, z, t)``: contact normal forces are bilinearly deposited on a
``dz x dy`` grid covering the structure band ``z in [0, H]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .shapes import FragmentShape

__all__ = ["CrushParams", "IceWallCrush", "pack_slab"]


@dataclass
class CrushParams:
    """Contact and material parameters (SI units)."""

    k_n: float = 5.0e6          # elastic contact stiffness [N/m]
    zeta: float = 0.30          # normal damping ratio [-]
    mu_pp: float = 0.35         # fragment-fragment friction [-]
    mu_wall: float = 0.15       # ice-structure (steel) friction [-]
    mu_side: float = 0.20       # lateral artificial boundary friction [-]
    mu_floor: float = 0.30      # bottom boundary friction [-]
    mu_r: float = 0.0           # rolling-resistance coefficient of sphere proxy [-]
    sigma_c: float = 8.0e6      # crushing strength / contact pressure cap [Pa]
    rho: float = 917.0          # ice density [kg/m^3]
    v_smooth: float = 0.02      # friction regularisation velocity [m/s]
    skin: float = 0.03          # neighbour-list skin [m]
    rebuild: int = 5            # neighbour rebuild interval [steps]
    omega_smooth: float = 1.0   # rolling-resistance regularisation [rad/s]
    bulk_damping: float = 12.0  # velocity damping [1/s]; keeps the crushing quasi-static
    mass_scale: float = 1.0     # mass scaling; 1 = physical inertia (default)
    gravity: float = 9.81       # gravity [m/s^2]; keeps extruded debris from flying
    velocity_ceiling: float = 0.0  # |v| ceiling [m/s]; 0 disables (safety net)
    crush_hardening: float = 0.05  # residual stiffness above the crushing cap [-]

    def validate(self) -> None:
        for name in ("k_n", "sigma_c", "rho", "mass_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("zeta", "mu_pp", "mu_wall", "mu_side", "mu_floor", "mu_r",
                     "bulk_damping", "velocity_ceiling", "gravity"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
        if not 0.0 <= self.crush_hardening < 1.0:
            raise ValueError("crush_hardening must be in [0, 1)")
        if self.zeta > 1.0:
            raise ValueError("zeta > 1 is over-damped and not supported")


def _skew(w: np.ndarray) -> np.ndarray:
    k = np.zeros((w.shape[0], 3, 3))
    k[:, 0, 1] = -w[:, 2]
    k[:, 0, 2] = w[:, 1]
    k[:, 1, 0] = w[:, 2]
    k[:, 1, 2] = -w[:, 0]
    k[:, 2, 0] = -w[:, 1]
    k[:, 2, 1] = w[:, 0]
    return k


def _rotation_increment(w_dt: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map, vectorised over a batch of rotation vectors."""
    theta = np.linalg.norm(w_dt, axis=1)
    small = theta < 1e-12
    safe = np.where(small, 1.0, theta)
    k = _skew(w_dt)
    a = np.where(small, 1.0, np.sin(safe) / safe)
    b = np.where(small, 0.5, (1.0 - np.cos(safe)) / safe**2)
    return np.eye(3)[None] + a[:, None, None] * k + b[:, None, None] * (k @ k)


def _crush_law(force: np.ndarray, cap: np.ndarray, hardening: float) -> np.ndarray:
    """Elastic contact force with a crushing-pressure cap.

    Below the cap the contact is linear elastic.  Above it the force grows with
    the *residual* stiffness ``hardening * k`` instead of being clipped: a hard
    clip leaves a saturated contact with no restoring stiffness, so a fragment
    pinched between two saturated contacts is accelerated without bound (that
    pathology was observed and is the reason for the hardening -- see the report).
    Physically the residual stiffness stands for the (small) hardening of ice
    during crushing; numerically it guarantees a bounded, energy-consistent
    contact.
    """
    positive = np.maximum(force, 0.0)
    excess = np.maximum(positive - cap, 0.0)
    return np.where(positive <= cap, positive, cap + hardening * excess)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pack_slab(
    rng: np.random.Generator,
    shapes: list[FragmentShape],
    box: tuple[float, float, float],
    *,
    max_tries: int = 60,
) -> np.ndarray:
    """Random-sequential placement of the fragments inside the box.

    Placement uses the *bounding* radius, so the polyhedral and the spherical
    realisations of the same assembly start from bit-identical positions and the
    comparison is paired.  ``max_tries`` is a soft limit: a fragment that cannot
    find a free slot is placed at its last candidate (the DEM resolves the
    residual overlap), which keeps the initial solid fraction close to target.
    """
    lx, by, hz = (float(v) for v in box)
    n = len(shapes)
    radii = np.array([s.r_circ for s in shapes])
    pos = np.zeros((n, 3))
    for i in range(n):
        r = radii[i]
        candidate = np.array([r, r, r])
        if i:
            lo = radii[:i]
            sep = 1.0 * (r + lo)[:, None]
            for _ in range(max_tries):
                candidate = np.array(
                    [
                        rng.uniform(min(r, 0.5 * lx), max(lx - r, 0.5 * lx + 1e-9)),
                        rng.uniform(min(r, 0.5 * by), max(by - r, 0.5 * by + 1e-9)),
                        rng.uniform(min(r, 0.5 * hz), max(hz - r, 0.5 * hz + 1e-9)),
                    ]
                )
                delta = pos[:i] - candidate
                if np.all(np.linalg.norm(delta, axis=1) >= sep[:, 0]):
                    break
        pos[i] = candidate
    return pos


class IceWallCrush:
    """DEM of an ice layer crushing against a rigid structure plane.

    Parameters
    ----------
    shapes
        Fragments of one assembly (all polyhedra, or all their sphere proxies).
    box
        ``(Lx, B, h_ice)`` extent of the ice layer [m].
    drive_speed
        Indentation speed ``v`` of the platen [m/s].
    grid
        ``(dy, dz)`` resolution of the pressure field on the structure [m].
    stroke, dt, save_every
        Platen stroke [m], time step [s] and sampling interval [steps].  The
        number of steps is ``stroke / (v * dt)``, so the indentation *strain*
        history is the same at every speed while the strain *rate* differs.
    """

    def __init__(
        self,
        shapes: list[FragmentShape],
        *,
        box: tuple[float, float, float],
        drive_speed: float,
        params: CrushParams | None = None,
        seed: int = 0,
        grid: tuple[float, float] = (0.1, 0.1),
        positions: np.ndarray | None = None,
        stroke: float = 0.24,
        dt: float = 5.0e-5,
        save_every: int = 20,
        settle_steps: int = 200,
        preload: float = 0.02,
    ) -> None:
        if not shapes:
            raise ValueError("at least one fragment is required")
        if drive_speed <= 0.0:
            raise ValueError("drive_speed must be positive")
        self.shapes = list(shapes)
        self.params = params or CrushParams()
        self.params.validate()
        self.n = len(self.shapes)
        self.Lx, self.B, self.H = (float(v) for v in box)
        self.v = float(drive_speed)
        self.dy, self.dz = (float(v) for v in grid)
        self.stroke = float(stroke)
        self.dt = float(dt)
        self.save_every = int(save_every)
        self.settle_steps = int(settle_steps)
        self.preload = float(preload)

        rng = np.random.default_rng(seed)
        if positions is None:
            self.pos = pack_slab(rng, self.shapes, (self.Lx, self.B, self.H))
        else:
            self.pos = np.array(positions, dtype=float).reshape(self.n, 3)
        self.rot = np.repeat(np.eye(3)[None], self.n, axis=0)
        for i, s in enumerate(self.shapes):
            if not s.is_sphere:
                self.rot[i] = _random_rotation(rng)
        self.vel = np.zeros((self.n, 3))
        self.omega = np.zeros((self.n, 3))
        self.force = np.zeros((self.n, 3))
        self.torque = np.zeros((self.n, 3))

        # Quasi-static mass scaling: gravity is absent and the crushing process is
        # force-balanced, so inertia only enters through the transient.  Scaling
        # the mass up keeps the contact-driven accelerations resolvable at the
        # DEM time step; the insensitivity of the results to the scale factor is
        # verified in the test suite (see docs/ICE_CONTACT_PRESSURE_ROLLING_PROXY.md).
        self.mass = np.array(
            [self.params.rho * self.params.mass_scale * s.volume for s in self.shapes]
        )
        self.inertia = np.array([s.inertia * m for s, m in zip(self.shapes, self.mass)])
        self.r_particle = np.array([s.r_eq for s in self.shapes])
        self.r_bound = np.array([s.r_circ for s in self.shapes])
        self.mu_r_particle = np.array(
            [self.params.mu_r if s.is_sphere else 0.0 for s in self.shapes]
        )

        # multi-sphere clump flattened for the pair contacts
        sp_pid, sp_cen, sp_rad = [], [], []
        for i, s in enumerate(self.shapes):
            for k in range(s.n_sub):
                sp_pid.append(i)
                sp_cen.append(s.sub_centers[k])
                sp_rad.append(s.sub_radii[k])
        self.sp_pid = np.asarray(sp_pid, dtype=int)
        self.sp_cen = np.asarray(sp_cen, dtype=float)
        self.sp_rad = np.asarray(sp_rad, dtype=float)
        self.rmax = float(self.r_bound.max())

        # structure (and other plane) contact points
        cp_pid, cp_body, cp_rad, cp_area, cp_ref, cp_proj = [], [], [], [], [], []
        for i, s in enumerate(self.shapes):
            if s.is_sphere:
                cp_pid.append(i)
                cp_body.append(np.zeros(3))
                cp_rad.append(s.r_eq)
                cp_area.append(s.projected_area)
                cp_ref.append(s.projected_area)
                cp_proj.append(s.projected_area)
            else:
                ref = float(s.facet_area.mean())
                for v, a in zip(s.vertices, s.facet_area):
                    cp_pid.append(i)
                    cp_body.append(v)
                    cp_rad.append(0.0)
                    cp_area.append(a)
                    cp_ref.append(ref)
                    cp_proj.append(s.projected_area)
        self.cp_pid = np.asarray(cp_pid, dtype=int)
        self.cp_body = np.asarray(cp_body, dtype=float)
        self.cp_rad = np.asarray(cp_rad, dtype=float)
        self.cp_area = np.asarray(cp_area, dtype=float)
        self.cp_ref = np.asarray(cp_ref, dtype=float)
        self.cp_proj = np.asarray(cp_proj, dtype=float)
        self.proj_particle = np.array([s.projected_area for s in self.shapes])

        # pressure-field grid on the structure
        self.ny = max(int(round(self.B / self.dy)), 1)
        self.nz = max(int(round(self.H / self.dz)), 1)
        self.cell_area = (self.B / self.ny) * (self.H / self.nz)
        self._p_acc = np.zeros((self.nz, self.ny))

        self._pairs: np.ndarray | None = None
        self._pair_age = 0
        self._ft_acc = np.zeros(3)
        self.tangential: list[np.ndarray] = []
        self.pressure: list[np.ndarray] = []
        self.force_history: list[np.ndarray] = []
        self.ke_history: list[float] = []
        self.time_history: list[float] = []
        self.platen_force_history: list[float] = []
        self.steps = int(round(self.stroke / (self.v * self.dt)))

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    def _sub_world(self) -> np.ndarray:
        return self.pos[self.sp_pid] + np.einsum("mij,mj->mi", self.rot[self.sp_pid], self.sp_cen)

    def _contact_world(self) -> np.ndarray:
        return self.pos[self.cp_pid] + np.einsum("mij,mj->mi", self.rot[self.cp_pid], self.cp_body)

    @staticmethod
    def _scatter3(target: np.ndarray, index: np.ndarray, values: np.ndarray) -> None:
        """``target[index] += values`` for a 3-component field, via bincount.

        ``np.add.at`` is ~10x slower per element and dominated the runtime of
        this kernel; bincount accumulation is the same operation.
        """
        n = target.shape[0]
        for d in range(3):
            target[:, d] += np.bincount(index, weights=values[:, d], minlength=n)

    def _deposit(self, y: np.ndarray, z: np.ndarray, weight: np.ndarray) -> None:
        """Bilinear deposit of a normal force [N] on the structure grid."""
        if weight.size == 0:
            return
        fy = y / self.dy - 0.5
        fz = z / self.dz - 0.5
        iy = np.floor(fy).astype(int)
        iz = np.floor(fz).astype(int)
        ty = fy - iy
        tz = fz - iz
        for oy, oz, wgt in (
            (0, 0, (1 - ty) * (1 - tz)),
            (1, 0, ty * (1 - tz)),
            (0, 1, (1 - ty) * tz),
            (1, 1, ty * tz),
        ):
            jj = iy + oy
            kk = iz + oz
            ok = (jj >= 0) & (jj < self.ny) & (kk >= 0) & (kk < self.nz)
            if ok.any():
                flat = kk[ok] * self.ny + jj[ok]
                self._p_acc += np.bincount(
                    flat, weights=(weight * wgt)[ok], minlength=self.nz * self.ny
                ).reshape(self.nz, self.ny)

    # ------------------------------------------------------------------
    # contact models
    # ------------------------------------------------------------------
    def _pair_contacts(self) -> None:
        world = self._sub_world()
        if self._pairs is None or self._pair_age <= 0:
            cutoff = 2.0 * self.rmax + self.params.skin
            pairs = cKDTree(world).query_pairs(cutoff, output_type="ndarray")
            if len(pairs):
                i, j = pairs[:, 0], pairs[:, 1]
                keep = self.sp_pid[i] != self.sp_pid[j]
                keep &= np.linalg.norm(world[j] - world[i], axis=1) < (
                    self.sp_rad[i] + self.sp_rad[j] + self.params.skin
                )
                pairs = pairs[keep]
            self._pairs = pairs
            self._pair_age = self.params.rebuild
        self._pair_age -= 1
        pairs = self._pairs
        if pairs is None or len(pairs) == 0:
            return

        i, j = pairs[:, 0], pairs[:, 1]
        dvec = world[j] - world[i]
        dist = np.maximum(np.linalg.norm(dvec, axis=1), 1e-12)
        delta = self.sp_rad[i] + self.sp_rad[j] - dist
        active = delta > 0.0
        if not active.any():
            return
        i, j, dist, delta = i[active], j[active], dist[active], delta[active]
        dvec = dvec[active]
        normal = dvec / dist[:, None]
        pi, pj = self.sp_pid[i], self.sp_pid[j]

        contact = world[i] + normal * (self.sp_rad[i] - 0.5 * delta)[:, None]
        ri = contact - self.pos[pi]
        rj = contact - self.pos[pj]
        v_rel = (
            self.vel[pj] + np.cross(self.omega[pj], rj)
            - self.vel[pi] - np.cross(self.omega[pi], ri)
        )
        vn = np.einsum("ij,ij->i", v_rel, normal)
        m_eff = self.mass[pi] * self.mass[pj] / (self.mass[pi] + self.mass[pj])
        damp = 2.0 * self.params.zeta * np.sqrt(self.params.k_n * m_eff)
        cap = self.params.sigma_c * np.pi * np.minimum(self.sp_rad[i], self.sp_rad[j]) ** 2
        f_n = _crush_law(self.params.k_n * delta - damp * vn, cap, self.params.crush_hardening)
        v_t = v_rel - vn[:, None] * normal
        v_t_mag = np.linalg.norm(v_t, axis=1)
        f_t = (
            -self.params.mu_pp
            * f_n[:, None]
            * np.tanh(v_t_mag / self.params.v_smooth)[:, None]
            * (v_t / np.maximum(v_t_mag, 1e-12)[:, None])
        )
        f = f_n[:, None] * normal + f_t

        self._scatter3(self.force, pj, f)
        self._scatter3(self.force, pi, -f)
        self._scatter3(self.torque, pj, np.cross(rj, f))
        self._scatter3(self.torque, pi, np.cross(ri, -f))

        weight = self.mu_r_particle[pj]
        if np.any(weight > 0.0):
            omega_rel = self.omega[pj] - self.omega[pi]
            mag = np.linalg.norm(omega_rel, axis=1)
            r_ref = 0.5 * (self.r_particle[pi] + self.r_particle[pj])
            strength = (
                weight * self.params.mu_r * r_ref * f_n
                * np.tanh(mag / self.params.omega_smooth) / np.maximum(mag, 1e-12)
            )
            couple = -strength[:, None] * omega_rel
            self._scatter3(self.torque, pj, couple)
            self._scatter3(self.torque, pi, -couple)

    def _plane_contact(
        self,
        normal: np.ndarray,
        offset: float,
        plane_velocity: np.ndarray,
        mu: float,
        *,
        structure: bool = False,
        world: np.ndarray | None = None,
    ) -> float:
        """Contacts against one rigid plane; returns the total normal force [N]."""
        if world is None:
            world = self._contact_world()
        signed = world @ normal - offset
        delta = self.cp_rad - signed
        active = delta > 0.0
        if not active.any():
            return 0.0
        idx = np.nonzero(active)[0]
        pid = self.cp_pid[idx]
        delta = delta[idx]
        area = self.cp_area[idx]
        share = area / np.maximum(self.cp_ref[idx], 1e-30)
        point = world[idx] - signed[idx][:, None] * normal
        lever = point - self.pos[pid]
        v_rel = (
            self.vel[pid] + np.cross(self.omega[pid], lever) - plane_velocity
        )
        vn = v_rel @ normal
        damp = 2.0 * self.params.zeta * np.sqrt(self.params.k_n * self.mass[pid] * share)
        f_n = _crush_law(
            self.params.k_n * delta * share - damp * vn,
            self.params.sigma_c * area,
            self.params.crush_hardening,
        )
        v_t = v_rel - vn[:, None] * normal
        v_t_mag = np.linalg.norm(v_t, axis=1)
        f_t = (
            -mu * f_n[:, None]
            * np.tanh(v_t_mag / self.params.v_smooth)[:, None]
            * (v_t / np.maximum(v_t_mag, 1e-12)[:, None])
        )
        f = f_n[:, None] * normal + f_t
        self._scatter3(self.force, pid, f)
        self._scatter3(self.torque, pid, np.cross(lever, f))

        # rolling resistance of the spherical proxy against the plane
        weight = self.mu_r_particle[pid]
        if np.any(weight > 0.0):
            spin = self.omega[pid]
            mag = np.linalg.norm(spin, axis=1)
            strength = (
                weight * self.params.mu_r * self.r_particle[pid] * f_n
                * np.tanh(mag / self.params.omega_smooth) / np.maximum(mag, 1e-12)
            )
            self._scatter3(self.torque, pid, -strength[:, None] * spin)

        if structure:
            self._ft_acc += f_t.sum(axis=0)
            band = point[:, 2]
            keep = (band >= 0.0) & (band <= self.H)
            self._deposit(point[keep, 1], band[keep], f_n[keep])
        return float(f_n.sum())

    # ------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------
    def _integrate(self, dt: float) -> None:
        self.vel += (self.force / self.mass[:, None]) * dt
        if self.params.gravity:
            self.vel[:, 2] -= self.params.gravity * dt
        if self.params.bulk_damping > 0.0:
            self.vel *= np.exp(-self.params.bulk_damping * dt)
        if self.params.velocity_ceiling > 0.0:
            speed = np.linalg.norm(self.vel, axis=1)
            over = speed > self.params.velocity_ceiling
            if np.any(over):
                self.vel[over] *= (self.params.velocity_ceiling / speed[over])[:, None]
        self.pos += self.vel * dt

        # gyroscopic term: alpha = I_w^-1 (tau - omega x (I_w omega))
        omega_body = np.einsum("nji,nj->ni", self.rot, self.omega)
        inertia_omega_body = omega_body * self.inertia
        gyro_body = np.cross(omega_body, inertia_omega_body)
        effective_body = (
            np.einsum("nji,nj->ni", self.rot, self.torque) - gyro_body
        )
        alpha_body = effective_body / self.inertia
        alpha = np.einsum("nij,nj->ni", self.rot, alpha_body)
        self.omega += alpha * dt
        self.rot = np.einsum("nij,njk->nik", _rotation_increment(self.omega * dt), self.rot)

    def _orthonormalise(self) -> None:
        u, _, vt = np.linalg.svd(self.rot)
        self.rot = u @ vt

    def kinetic_energy(self) -> float:
        trans = 0.5 * float(np.sum(self.mass[:, None] * self.vel**2))
        omega_body = np.einsum("nji,nj->ni", self.rot, self.omega)
        rot = 0.5 * float(np.sum(self.inertia * omega_body**2))
        return trans + rot

    # ------------------------------------------------------------------
    def run(self) -> "IceWallCrush":
        """Advance the platen by ``stroke`` and record the structure pressure."""
        dt = self.dt
        ex = np.array([1.0, 0.0, 0.0])
        ey = np.array([0.0, 1.0, 0.0])
        ez = np.array([0.0, 0.0, 1.0])
        zero = np.zeros(3)
        platen_x = self.Lx
        # preload: bring the layer into contact with the structure at the
        # indentation speed (no recording), so that every condition starts from a
        # loaded wall instead of a gap.
        preload_steps = int(round(self.preload / (self.v * dt)))
        for step in range(preload_steps):
            self.force[:] = 0.0
            self.torque[:] = 0.0
            self._pair_contacts()
            world = self._contact_world()
            self._plane_contact(ex, 0.0, zero, self.params.mu_wall, structure=True, world=world)
            self._plane_contact(-ex, -platen_x, -self.v * ex, self.params.mu_wall, world=world)
            self._plane_contact(ez, 0.0, zero, self.params.mu_floor, world=world)
            self._plane_contact(ey, 0.0, zero, self.params.mu_side, world=world)
            self._plane_contact(-ey, -self.B, zero, self.params.mu_side, world=world)
            self._integrate(dt)
            platen_x -= self.v * dt
            self._p_acc[:] = 0.0
            self._ft_acc[:] = 0.0
            if step % 100 == 0:
                self._orthonormalise()
        # settle phase: relax the initial packing with the platen at rest
        for step in range(self.settle_steps):
            self.force[:] = 0.0
            self.torque[:] = 0.0
            self._pair_contacts()
            world = self._contact_world()
            self._plane_contact(ex, 0.0, zero, self.params.mu_wall, structure=True, world=world)
            self._plane_contact(-ex, -platen_x, zero, self.params.mu_wall, world=world)
            self._plane_contact(ez, 0.0, zero, self.params.mu_floor, world=world)
            self._plane_contact(ey, 0.0, zero, self.params.mu_side, world=world)
            self._plane_contact(-ey, -self.B, zero, self.params.mu_side, world=world)
            self._integrate(dt)
            self._p_acc[:] = 0.0
            self._ft_acc[:] = 0.0
            if step % 100 == 0:
                self._orthonormalise()
        self.vel *= 0.0
        self.omega *= 0.0
        for step in range(self.steps):
            self.force[:] = 0.0
            self.torque[:] = 0.0
            self._pair_contacts()
            wall_normal = self._plane_contact(ex, 0.0, zero, self.params.mu_wall, structure=True)
            platen_normal = self._plane_contact(
                -ex, -platen_x, -self.v * ex, self.params.mu_wall
            )
            self._plane_contact(ez, 0.0, zero, self.params.mu_floor)
            self._plane_contact(ey, 0.0, zero, self.params.mu_side)
            self._plane_contact(-ey, -self.B, zero, self.params.mu_side)

            self._integrate(dt)
            platen_x -= self.v * dt

            if step % self.save_every == 0:
                self._record(step, wall_normal, platen_normal)
            if step % 200 == 0:
                self._orthonormalise()
        self._record(self.steps, 0.0, 0.0)
        self.pressure_array = np.asarray(self.pressure)
        self.force_array = np.asarray(self.force_history)
        self.tangential_array = np.asarray(self.tangential)
        self.time_array = np.asarray(self.time_history)
        return self

    def _record(self, step: int, wall_normal: float, platen_normal: float) -> None:
        self.pressure.append((self._p_acc / self.cell_area).copy())
        self._p_acc[:] = 0.0
        self.force_history.append(float(wall_normal))
        self.tangential.append(self._ft_acc.copy())
        self._ft_acc[:] = 0.0
        self.platen_force_history.append(float(platen_normal))
        self.ke_history.append(self.kinetic_energy())
        self.time_history.append(step * self.dt)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Lightweight record for the study JSON."""
        return {
            "n_fragments": self.n,
            "kind": "sphere" if all(s.is_sphere for s in self.shapes) else "polyhedron",
            "n_subspheres": int(self.sp_pid.size),
            "box": [self.Lx, self.B, self.H],
            "drive_speed": self.v,
            "dt": self.dt,
            "steps": self.steps,
            "stroke": self.stroke,
            "settle_steps": self.settle_steps,
            "preload": self.preload,
            "t_end": self.steps * self.dt,
            "grid": [self.dy, self.dz],
            "cell_area": self.cell_area,
            "mu_r": self.params.mu_r,
            "mu_pp": self.params.mu_pp,
            "mu_wall": self.params.mu_wall,
            "sigma_c": self.params.sigma_c,
            "k_n": self.params.k_n,
            "mean_fragment_diameter": float(
                2.0 * np.cbrt(3.0 * np.mean([s.volume for s in self.shapes]) / (4.0 * np.pi))
            ),
            "max_kinetic_energy": float(np.max(self.ke_history)) if self.ke_history else 0.0,
        }