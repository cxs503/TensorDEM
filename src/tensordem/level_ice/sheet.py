"""Two-dimensional bonded-particle (parallel-bond) level-ice sheet.

Model statement
---------------
A vertical section of a level-ice sheet is discretised into a square lattice of
particles of spacing ``d`` (``n_z = round(h/d)`` layers through the thickness,
``n_x`` columns along the track).  Neighbouring particles are joined by
**parallel bonds**: two-node frame elements carrying axial force, shear and
bending moment, i.e. the 2-D bonded-particle model *with* rotational degrees of
freedom.  For a strut of in-plane thickness ``d`` and out-of-plane width ``b``

    A = b d,     I = b d**3 / 12,     W = b d**2 / 6,
    EA = E A,    EI = E I.

A square frame lattice of such struts has axial modulus ``E`` exactly (one strut
of area ``b d`` per cell of volume ``b d * d``) and slab bending stiffness
``E b h**3/12 + O((d/h)**2)``, so it represents the ice as an elastic continuum
whose discretisation error is the usual lattice ``(d/h)**2`` refinement error.
That is what the mesh-convergence study measures.

Buoyancy is a Winkler foundation of modulus ``rho_w * g`` per unit waterplane
area, linearised about the free-floating equilibrium and **compression only**
(water does not pull).  It is distributed across the thickness so that one
particle column carries ``rho_w * g * b * d``, i.e. the correct modulus per unit
sheet length.  Water drag is quadratic on the projected particle area.

Bond failure uses a maximum-fibre stress criterion

    sigma_t = N/A + |M|/W        limit sigma_f   (tension, upper fibre)
    sigma_c = |M|/W - N/A        limit sigma_c   (compression, lower fibre)
    tau     = |V|/A              limit tau_f     (shear)

so the material strengths enter with no hidden calibration factor.  A broken
bond becomes a repulsive Coulomb pair contact, so broken floes still push on
each other.

Hull
----
The ship is rigid.  Its bow is idealised as a **finite inclined face** whose
outward normal into the ice is ``n = (sin(beta), -cos(beta))`` with ``beta`` the
face inclination from the horizontal: the face pushes the sheet forward and
*down*, the level-ice bow mechanism that bends the sheet and fails it in tension
at the upper surface.  The face runs from above the ice top to below the ice
bottom, so ice driven below the face is released and free to float.  Contact is a
penalty normal law with a dashpot, a Coulomb tangential law and an optional
**crushing cap** ``N <= p_crush * (2 r) * b`` representing local crushing of the
ice edge at its unconfined compressive strength.

Resistance bookkeeping
----------------------
The horizontal hull reaction is integrated in three disjoint, exactly additive
buckets, so the reported split sums to the measured total with no residual:

``crushing_n_per_m``  normal contact force at contacts saturated by the
                      crushing cap (plastic indentation of the ice edge);
``flexure_n_per_m``   normal contact force at contacts below the cap (elastic
                      load path, i.e. force transmitted through the bent sheet);
``friction_n_per_m``  tangential Coulomb force along the hull face.

All forces are line loads in N/m (the sheet is one metre of out-of-plane
width).

What this module is not
-----------------------
No lateral escape or spalling, no floe rotation out of the section plane, no
hydrodynamic added mass or wave radiation (only a quadratic drag) and no
fracture energy (strength-only bonds).  Those omissions are why the model is
compared with the engineering anchors as a *reduced* model; they are quantified
in ``docs/LEVEL_ICE_RESISTANCE_DEM.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .formula import GRAVITY, WATER_DENSITY


@dataclass(frozen=True)
class SheetConfig:
    """Discretisation, material, water and hull parameters (all SI)."""

    thickness_m: float = 1.0
    particle_size_m: float = 0.125
    length_m: float = 40.0
    out_of_plane_width_m: float = 1.0
    elastic_modulus_pa: float = 5.0e9
    flexural_strength_pa: float = 0.5e6
    compressive_strength_pa: float = 1.5e6
    shear_strength_pa: float = 1.0e6
    density_kg_m3: float = 917.0
    water_density_kg_m3: float = WATER_DENSITY
    gravity_m_s2: float = GRAVITY
    buoyancy: bool = True
    drag_coefficient: float = 1.0
    inter_particle_friction: float = 0.1
    bow_angle_deg: float = 30.0
    hull_speed_m_s: float = 3.0
    hull_friction: float = 0.05
    crush_cap_pa: float | None = 1.5e6
    contact_stiffness_scale: float = 1.0
    contact_damping_ratio: float = 0.02
    strength_scale: float = 1.0
    mass_scale: float = 1.0
    dt_safety: float = 0.35
    dt: float | None = None

    def __post_init__(self) -> None:
        for name in ("thickness_m", "particle_size_m", "length_m",
                     "out_of_plane_width_m", "elastic_modulus_pa",
                     "flexural_strength_pa", "compressive_strength_pa",
                     "shear_strength_pa", "density_kg_m3",
                     "water_density_kg_m3", "gravity_m_s2",
                     "contact_stiffness_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.hull_speed_m_s) or self.hull_speed_m_s < 0.0:
            raise ValueError("hull_speed_m_s must be finite and nonnegative")
        if not math.isfinite(self.strength_scale) or self.strength_scale <= 0.0:
            raise ValueError("strength_scale must be finite and positive")
        if not math.isfinite(self.mass_scale) or self.mass_scale <= 0.0:
            raise ValueError("mass_scale must be finite and positive")
        if not 0.0 < self.bow_angle_deg < 89.0:
            raise ValueError("bow_angle_deg must be in (0, 89) degrees")
        for name in ("drag_coefficient", "hull_friction", "inter_particle_friction"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.crush_cap_pa is not None and (
                not math.isfinite(self.crush_cap_pa) or self.crush_cap_pa <= 0.0):
            raise ValueError("crush_cap_pa must be finite and positive when given")
        if self.dt is not None and (not math.isfinite(self.dt) or self.dt <= 0.0):
            raise ValueError("dt must be finite and positive")

    @property
    def n_z(self) -> int:
        return max(2, int(round(self.thickness_m / self.particle_size_m)))

    @property
    def n_x(self) -> int:
        return max(4, int(round(self.length_m / self.particle_size_m)))

    @property
    def spacing_m(self) -> float:
        return self.thickness_m / self.n_z

    @property
    def particle_radius_m(self) -> float:
        return 0.5 * self.spacing_m

    @property
    def particle_mass_kg(self) -> float:
        d = self.spacing_m
        return self.density_kg_m3 * self.out_of_plane_width_m * d * d * self.mass_scale

    @property
    def axial_stiffness_n_per_m(self) -> float:
        """``EA/d`` of one axis bond [N/m] (independent of the spacing)."""
        return self.elastic_modulus_pa * self.out_of_plane_width_m

    @property
    def recommended_dt_s(self) -> float:
        """Wave-CFL limit on the stiffest local mode (bond + hull contact)."""
        m = self.particle_mass_kg
        k = self.axial_stiffness_n_per_m * (1.0 + self.contact_stiffness_scale)
        return self.dt_safety * 2.0 * math.sqrt(m / k)

    def as_dict(self) -> dict[str, Any]:
        return {
            "thickness_m": self.thickness_m,
            "particle_size_m": self.particle_size_m,
            "n_x": self.n_x,
            "n_z": self.n_z,
            "spacing_m": self.spacing_m,
            "length_m": self.length_m,
            "out_of_plane_width_m": self.out_of_plane_width_m,
            "elastic_modulus_pa": self.elastic_modulus_pa,
            "flexural_strength_pa": self.flexural_strength_pa,
            "compressive_strength_pa": self.compressive_strength_pa,
            "shear_strength_pa": self.shear_strength_pa,
            "bow_angle_deg": self.bow_angle_deg,
            "hull_speed_m_s": self.hull_speed_m_s,
            "hull_friction": self.hull_friction,
            "crush_cap_pa": self.crush_cap_pa,
            "strength_scale": self.strength_scale,
            "buoyancy": self.buoyancy,
            "mass_scale": self.mass_scale,
            "n_particles": self.n_x * self.n_z,
            "n_bonds": 2 * self.n_x * self.n_z - self.n_x - self.n_z,
            "dt_s": self.dt if self.dt is not None else self.recommended_dt_s,
        }


@dataclass
class RunResult:
    """Measured resistance history of one sheet run."""

    config: dict[str, Any]
    dt_s: float
    time_s: np.ndarray
    advance_m: np.ndarray
    total_n_per_m: np.ndarray
    crushing_n_per_m: np.ndarray
    flexure_n_per_m: np.ndarray
    friction_n_per_m: np.ndarray
    broken_fraction: np.ndarray
    window: tuple[float, float]
    summary: dict[str, Any] = field(default_factory=dict)

    def _windowed(self, series: np.ndarray) -> np.ndarray:
        lo, hi = self.window
        mask = (self.advance_m >= lo) & (self.advance_m <= hi)
        if not np.any(mask):
            raise ValueError("empty measurement window")
        return series[mask]

    def steady_mean(self, key: str) -> float:
        return float(self._windowed(getattr(self, key)).mean())

    def steady_std(self, key: str) -> float:
        return float(self._windowed(getattr(self, key)).std())

    def steady_resistance(self) -> dict[str, float]:
        return {
            "total_n_per_m": self.steady_mean("total_n_per_m"),
            "crushing_n_per_m": self.steady_mean("crushing_n_per_m"),
            "flexure_n_per_m": self.steady_mean("flexure_n_per_m"),
            "friction_n_per_m": self.steady_mean("friction_n_per_m"),
            "total_std_n_per_m": self.steady_std("total_n_per_m"),
            "crushing_fraction": self.steady_mean("crushing_n_per_m")
            / max(self.steady_mean("total_n_per_m"), 1e-12),
            "flexure_fraction": self.steady_mean("flexure_n_per_m")
            / max(self.steady_mean("total_n_per_m"), 1e-12),
        }

class LevelIceSheet:
    """Bonded-particle level-ice sheet driven by a rigid inclined hull face."""

    def __init__(self, config: SheetConfig = SheetConfig()) -> None:
        self.config = cfg = config
        d = cfg.spacing_m
        self.dt = 0.0
        self.nx, self.nz = cfg.n_x, cfg.n_z
        idx = np.arange(self.nx * self.nz).reshape(self.nz, self.nx)
        self.grid_x = np.tile(np.arange(self.nx), self.nz)
        self.grid_z = np.repeat(np.arange(self.nz), self.nx)
        self.n = self.nx * self.nz
        self.x = (self.grid_x + 0.5) * d
        self.z = (self.grid_z + 0.5) * d
        self.u = np.zeros((self.n, 3))       # ux, uz, theta
        self.v = np.zeros((self.n, 3))
        self.mass = cfg.particle_mass_kg
        self.inertia = self.mass * d * d / 6.0
        right = np.stack((idx[:, :-1].ravel(), idx[:, 1:].ravel()), axis=1)
        top = np.stack((idx[:-1, :].ravel(), idx[1:, :].ravel()), axis=1)
        pairs = np.concatenate((right, top), axis=0)
        self.bi = pairs[:, 0]
        self.bj = pairs[:, 1]
        self.n_bonds = len(self.bi)
        dx = self.x[self.bj] - self.x[self.bi]
        dz = self.z[self.bj] - self.z[self.bi]
        self.blen = np.hypot(dx, dz)
        self.bc = dx / self.blen
        self.bs = dz / self.blen
        self.A = cfg.out_of_plane_width_m * d
        self.I = cfg.out_of_plane_width_m * d**3 / 12.0
        self.W = cfg.out_of_plane_width_m * d * d / 6.0
        self.EA = cfg.elastic_modulus_pa * self.A
        self.EI = cfg.elastic_modulus_pa * self.I
        self.strength_t = cfg.flexural_strength_pa * cfg.strength_scale
        self.strength_c = cfg.compressive_strength_pa * cfg.strength_scale
        self.strength_s = cfg.shear_strength_pa * cfg.strength_scale
        self.alive = np.ones(self.n_bonds, dtype=bool)
        self.break_time = np.full(self.n_bonds, -1.0)
        self.break_x = np.full(self.n_bonds, np.nan)
        self.break_mode = np.zeros(self.n_bonds, dtype=np.int8)
        self.k_foundation = (
            cfg.water_density_kg_m3 * cfg.gravity_m_s2
            * cfg.out_of_plane_width_m * d / self.nz
        )
        self.drag_area = cfg.out_of_plane_width_m * d / self.nz
        self.drag_coeff = 0.5 * cfg.water_density_kg_m3 * cfg.drag_coefficient
        beta = math.radians(cfg.bow_angle_deg)
        self.normal = np.array([math.sin(beta), -math.cos(beta)])
        self.tangent = np.array([math.cos(beta), math.sin(beta)])
        self.z_top = cfg.thickness_m
        self.z_bot = -0.9 * cfg.thickness_m
        self.face_len = 1.5 * cfg.thickness_m / abs(self.normal[1])
        self.face_dx = self.face_len * abs(self.normal[0])
        self.hull_x0 = -self.face_dx - 0.5 * d
        self.hull_v = cfg.hull_speed_m_s
        self.k_contact = (cfg.contact_stiffness_scale * cfg.elastic_modulus_pa
                          * cfg.out_of_plane_width_m)
        self.c_contact = 2.0 * cfg.contact_damping_ratio * math.sqrt(
            self.k_contact * self.mass)
        self.crush_cap_force = (
            None if cfg.crush_cap_pa is None
            else cfg.crush_cap_pa * cfg.strength_scale * 2.0 * cfg.particle_radius_m
            * cfg.out_of_plane_width_m
        )
        self.fixed = np.zeros(self.n, dtype=bool)
        self.active = np.ones(self.n, dtype=bool)
        self.time = 0.0
        self.steps = 0
        self.accounting = dict(fracture_release_J=0.0, hull_work_J=0.0,
                               drag_dissipation_J=0.0, foundation_work_J=0.0)
        self.Kg = self.element_matrices()
        self.dt = cfg.dt if cfg.dt is not None else self.stable_dt()
        if self.dt > cfg.recommended_dt_s * 1.000001:
            raise ValueError(
                f"dt {self.dt:.3e} s exceeds the stability limit "
                f"{cfg.recommended_dt_s:.3e} s"
            )

    # -- hull geometry ------------------------------------------------------
    def hull_position(self) -> float:
        return self.hull_x0 + self.time * self.hull_v

    def hull_vertices(self) -> tuple[np.ndarray, np.ndarray]:
        top = np.array([self.hull_position(), self.z_top])
        low = top - self.face_len * np.array(
            [abs(self.normal[0]), abs(self.normal[1])])
        return top, low

    def clamp_column(self, column: int) -> None:
        """Clamp every particle of a grid column (all three degrees of freedom)."""
        self.fixed[self.grid_x == column] = True
        self.u[self.fixed] = 0.0
        self.v[self.fixed] = 0.0

    def positions(self) -> np.ndarray:
        return np.stack((self.x + self.u[:, 0], self.z + self.u[:, 1]), axis=1)

    def hull_signed_distance(self) -> tuple[np.ndarray, np.ndarray]:
        """Signed distance of each particle centre from the hull face."""
        top, low = self.hull_vertices()
        p = self.positions()
        seg = low - top
        seg_len2 = float(seg @ seg)
        t = np.clip(((p - top) @ seg) / seg_len2, 0.0, 1.0)
        closest = top + t[:, None] * seg
        delta = p - closest
        on_face = (t > 0.0) & (t < 1.0)
        s_face = delta @ self.normal
        s_end = np.linalg.norm(delta, axis=1)
        s = np.where(on_face, s_face, s_end)
        normal = np.where(on_face[:, None], self.normal[None, :],
                          delta / np.maximum(s_end, 1e-12)[:, None])
        return s, normal

    # -- internal forces ----------------------------------------------------
    def bond_forces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Elastic bond forces (already negated for the momentum equation)."""
        c, s = self.bc, self.bs
        ux_i = self.u[self.bi, 0]
        uz_i = self.u[self.bi, 1]
        th_i = self.u[self.bi, 2]
        ux_j = self.u[self.bj, 0]
        uz_j = self.u[self.bj, 1]
        th_j = self.u[self.bj, 2]
        u1 = c * ux_i + s * uz_i
        w1 = -s * ux_i + c * uz_i
        u2 = c * ux_j + s * uz_j
        w2 = -s * ux_j + c * uz_j
        L = self.blen
        a = self.EA / L
        b1 = 12.0 * self.EI / L**3
        b2 = 6.0 * self.EI / L**2
        b3 = 4.0 * self.EI / L
        b4 = 2.0 * self.EI / L
        axial = a * (u2 - u1)                    # tension positive
        shear = b1 * (w1 - w2) + b2 * (th_i + th_j)
        m_i = b2 * (w1 - w2) + b3 * th_i + b4 * th_j
        m_j = b2 * (w1 - w2) + b4 * th_i + b3 * th_j
        if self.strength_t > 0.0:
            abs_m = np.maximum(np.abs(m_i), np.abs(m_j))
            sigma_t = axial / self.A + abs_m / self.W
            sigma_c = abs_m / self.W - axial / self.A
            tau = np.abs(shear) / self.A
            failure = self.alive & ((sigma_t > self.strength_t)
                                    | (sigma_c > self.strength_c)
                                    | (tau > self.strength_s))
            if np.any(failure):
                self.alive &= ~failure
                self.break_time[failure] = self.time
                self.break_x[failure] = 0.5 * (self.x[self.bi[failure]]
                                               + self.x[self.bj[failure]])
                mode = np.zeros(int(failure.sum()), dtype=np.int8)
                mode[sigma_t[failure] > self.strength_t] = 1
                mode[(sigma_c[failure] > self.strength_c) & (mode == 0)] = 2
                mode[(tau[failure] > self.strength_s) & (mode == 0)] = 3
                self.break_mode[failure] = mode
                released = 0.5 * axial[failure] ** 2 / a[failure]
                self.accounting["fracture_release_J"] += float(released.sum())
        active = self.alive
        u_elem = np.stack((ux_i, uz_i, th_i, ux_j, uz_j, th_j), axis=1)
        f_elem = -np.einsum("bij,bj->bi", self.Kg, u_elem)
        f_elem[~active] = 0.0
        nb = self.n
        fx = (np.bincount(self.bi, f_elem[:, 0], minlength=nb)
              + np.bincount(self.bj, f_elem[:, 3], minlength=nb))
        fz = (np.bincount(self.bi, f_elem[:, 1], minlength=nb)
              + np.bincount(self.bj, f_elem[:, 4], minlength=nb))
        mz = (np.bincount(self.bi, f_elem[:, 2], minlength=nb)
              + np.bincount(self.bj, f_elem[:, 5], minlength=nb))
        self.last_bond_stress = (axial, shear, m_i, m_j)
        return fx, fz, mz

    def pair_contact_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """Repulsive Coulomb contact across broken bonds."""
        broken = ~self.alive
        fx = np.zeros(self.n)
        fz = np.zeros(self.n)
        if not np.any(broken):
            return fx, fz
        pair_ok = broken & self.active[self.bi] & self.active[self.bj]
        i = self.bi[pair_ok]
        j = self.bj[pair_ok]
        if i.size == 0:
            return fx, fz
        p = self.positions()
        vel = self.v[:, :2]
        delta = p[j] - p[i]
        dist = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(dist, 1e-12)[:, None]
        overlap = np.maximum(0.0, 2.0 * self.config.particle_radius_m - dist)
        rel = vel[j] - vel[i]
        v_n = np.einsum("ij,ij->i", rel, normal)
        mag = self.k_contact * overlap - self.c_contact * v_n
        mag = np.where(overlap > 0.0, np.maximum(mag, 0.0), 0.0)
        tang = np.stack((-normal[:, 1], normal[:, 0]), axis=1)
        v_t = np.einsum("ij,ij->i", rel, tang)
        mu = self.config.inter_particle_friction
        f_t = -mu * mag * np.tanh(v_t / 0.01)
        force = mag[:, None] * normal + f_t[:, None] * tang
        fx = (np.bincount(i, force[:, 0], minlength=self.n)
              - np.bincount(j, force[:, 0], minlength=self.n))
        fz = (np.bincount(i, force[:, 1], minlength=self.n)
              - np.bincount(j, force[:, 1], minlength=self.n))
        return fx, fz

    def foundation_forces(self) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        fz = np.zeros(self.n)
        fx = np.zeros(self.n)
        if cfg.buoyancy:
            fz -= self.k_foundation * np.minimum(self.u[:, 1], 0.0) * self.active
        if cfg.drag_coefficient > 0.0:
            speed = np.linalg.norm(self.v[:, :2], axis=1)
            fx = -self.drag_coeff * self.drag_area * speed * self.v[:, 0] * self.active
            fz -= self.drag_coeff * self.drag_area * speed * self.v[:, 1] * self.active
        return fx, fz

    def hull_forces(self) -> dict[str, Any]:
        """Hull contact forces on the ice plus the additive resistance ledger."""
        s, normal = self.hull_signed_distance()
        r = self.config.particle_radius_m
        contact = s < r
        out: dict[str, Any] = {
            "total": 0.0, "crushing": 0.0, "flexure": 0.0, "friction": 0.0,
            "n_contacts": int(contact.sum()), "n_capped": 0, "removed": 0,
            "max_penetration": float(max(0.0, r - float(s.min()))),
        }
        if not np.any(contact):
            out["fx"] = np.zeros(self.n)
            out["fz"] = np.zeros(self.n)
            return out
        idx = np.flatnonzero(contact)
        n_dir = normal[idx]
        pen = r - s[idx]
        hull_v = np.array([self.hull_v, 0.0])
        rel = self.v[idx, :2] - hull_v
        v_n = np.einsum("ij,ij->i", rel, n_dir)
        mag = np.maximum(self.k_contact * pen - self.c_contact * v_n, 0.0)
        capped = np.zeros(len(idx), dtype=bool)
        if self.config.crush_cap_pa is not None:
            # Plastic indentation: the crushed notch widens with penetration, so
            # the load the edge can carry is p_crush * chord(penetration) * b.
            chord = 2.0 * np.sqrt(np.clip(pen ** 2 - (pen - r) ** 2, 0.0, None))
            chord = np.clip(chord, 0.05 * self.config.spacing_m, 2.0 * r)
            limit = (self.config.crush_cap_pa * self.config.strength_scale
                     * chord * self.config.out_of_plane_width_m)
            capped = mag > limit
            mag = np.minimum(mag, limit)
        t_dir = np.broadcast_to(self.tangent, (len(idx), 2))
        v_t = np.einsum("ij,ij->i", rel, t_dir)
        f_t = -self.config.hull_friction * mag * np.tanh(v_t / 0.01)
        force = mag[:, None] * n_dir + f_t[:, None] * t_dir
        fx = np.bincount(idx, force[:, 0], minlength=self.n)
        fz = np.bincount(idx, force[:, 1], minlength=self.n)
        horizontal_normal = mag * n_dir[:, 0]
        horizontal_friction = f_t * t_dir[:, 0]
        out["crushing"] = float(horizontal_normal[capped].sum())
        out["flexure"] = float(horizontal_normal[~capped].sum())
        out["friction"] = float(horizontal_friction.sum())
        out["total"] = out["crushing"] + out["flexure"] + out["friction"]
        out["n_capped"] = int(capped.sum())
        out["fx"] = fx
        out["fz"] = fz
        # Crushed ice: a particle whose centre has been driven through the hull
        # face is ground ice, not structure.  It is deactivated (element
        # deletion) so that its momentum cannot be fed back into the intact
        # sheet - the 3-D escape route that a plane section cannot resolve.
        beneath = self.active & (self.positions()[:, 0] < self.hull_position() - 0.05 * r) \
            & (self.positions()[:, 1] < 0.5 * self.config.thickness_m)
        crushed = (pen >= 0.6 * r)
        if np.any(beneath):
            idx_b = np.flatnonzero(beneath)
            self.active[idx_b] = False
            self.u[idx_b] = 0.0
            self.v[idx_b] = 0.0
            touching_b = (np.isin(self.bi, idx_b) | np.isin(self.bj, idx_b)) & self.alive
            self.alive[touching_b] = False
            self.break_mode[touching_b] = 4
            self.break_time[touching_b] = self.time
        if np.any(crushed):
            idx_c = idx[crushed]
            self.active[idx_c] = False
            self.u[idx_c] = 0.0
            self.v[idx_c] = 0.0
            touching = (np.isin(self.bi, idx_c) | np.isin(self.bj, idx_c)) & self.alive
            self.alive[touching] = False
            self.break_mode[touching] = 4
            self.break_time[touching] = self.time
            out["removed"] = int(idx_c.size)
        else:
            out["removed"] = 0
        out["fx"][~self.active] = 0.0
        out["fz"][~self.active] = 0.0
        return out

    # -- integration --------------------------------------------------------
    def step(self) -> dict[str, Any]:
        bfx, bfz, bmz = self.bond_forces()
        pfx, pfz = self.pair_contact_forces()
        ffx, ffz = self.foundation_forces()
        hull = self.hull_forces()
        fx = bfx + pfx + ffx + hull["fx"]
        fz = bfz + pfz + ffz + hull["fz"]
        mz = bmz
        free = self.active & ~self.fixed
        self.v[free, 0] += self.dt * fx[free] / self.mass
        self.v[free, 1] += self.dt * fz[free] / self.mass
        self.v[free, 2] += self.dt * mz[free] / self.inertia
        self.u[free] += self.dt * self.v[free]
        self.time += self.dt
        self.steps += 1
        if not (np.isfinite(self.u).all() and np.isfinite(self.v).all()):
            raise FloatingPointError("non-finite sheet state; reduce dt")
        return hull

    def run(self, *, distance_m: float = 10.0,
            record_stride: int = 20,
            window_fraction: tuple[float, float] = (0.45, 1.0),
            max_steps: int | None = None) -> RunResult:
        """Advance the hull by ``distance_m`` and record the resistance history."""
        if distance_m <= 0.0:
            raise ValueError("distance_m must be positive")
        if not 0.0 <= window_fraction[0] < window_fraction[1] <= 1.0:
            raise ValueError("window_fraction must be an increasing pair in [0, 1]")
        speed = max(self.hull_v, 1e-9)
        n_steps = int(math.ceil(distance_m / (speed * self.dt)))
        if max_steps is not None:
            n_steps = min(n_steps, int(max_steps))
        if n_steps < 1:
            raise ValueError("distance_m is shorter than one time step")
        times = [0.0]
        advances = [0.0]
        totals = [0.0]
        crushing = [0.0]
        flexure = [0.0]
        friction = [0.0]
        broken = [float(np.mean(~self.alive))]
        contacts = 0
        capped = 0
        removed = 0
        max_pen = 0.0
        for k in range(n_steps):
            hull = self.step()
            contacts = max(contacts, hull["n_contacts"])
            capped = max(capped, hull["n_capped"])
            removed = max(removed, hull.get("removed", 0))
            max_pen = max(max_pen, hull["max_penetration"])
            if (k + 1) % record_stride == 0 or k == n_steps - 1:
                times.append(self.time)
                advances.append(self.time * self.hull_v)
                totals.append(hull["total"])
                crushing.append(hull["crushing"])
                flexure.append(hull["flexure"])
                friction.append(hull["friction"])
                broken.append(float(np.mean(~self.alive)))
        advance = np.array(advances)
        lo, hi = window_fraction
        window = (lo * float(advance[-1]), hi * float(advance[-1]))
        summary = {
            "time_s": float(self.time),
            "advance_m": float(advance[-1]),
            "n_steps": self.steps,
            "dt_s": self.dt,
            "max_contacts": contacts,
            "max_capped_contacts": capped,
            "removed_particles": int(np.sum(~self.active)),
            "max_penetration_m": max_pen,
            "broken_bonds": int(np.sum(~self.alive)),
            "broken_fraction": float(np.mean(~self.alive)),
            "mean_break_advance_m": float(np.nanmean(self.break_x))
            if np.any(~self.alive) else float("nan"),
            "first_break_advance_m": float(self.break_time[self.break_time >= 0].min()
                                            * self.hull_v)
            if np.any(self.break_time >= 0) else float("nan"),
            "kinetic_energy_J_per_m": float(
                0.5 * self.mass * np.sum(self.v[:, :2] ** 2)
                + 0.5 * self.inertia * np.sum(self.v[:, 2] ** 2)),
            "accounting": dict(self.accounting),
            "resistance": {
                "total_n_per_m": float(np.mean(totals)),
                "crushing_n_per_m": float(np.mean(crushing)),
                "flexure_n_per_m": float(np.mean(flexure)),
                "friction_n_per_m": float(np.mean(friction)),
            },
            "config": self.config.as_dict(),
        }
        return RunResult(
            config=self.config.as_dict(), dt_s=self.dt,
            time_s=np.array(times), advance_m=advance,
            total_n_per_m=np.array(totals),
            crushing_n_per_m=np.array(crushing),
            flexure_n_per_m=np.array(flexure),
            friction_n_per_m=np.array(friction),
            broken_fraction=np.array(broken),
            window=window, summary=summary,
        )

    # -- linear static solve (verification against beam theory) --------------
    def element_matrices(self) -> np.ndarray:
        """Global-frame 6x6 stiffness of every bond (consistent with K assembly)."""
        nb = self.n_bonds
        c, s = self.bc, self.bs
        L = self.blen
        a = self.EA / L
        b1 = 12.0 * self.EI / L**3
        b2 = 6.0 * self.EI / L**2
        b3 = 4.0 * self.EI / L
        b4 = 2.0 * self.EI / L
        K = np.zeros((nb, 6, 6))
        K[:, 0, 0] = a
        K[:, 0, 3] = -a
        K[:, 3, 0] = -a
        K[:, 3, 3] = a
        K[:, 1, 1] = b1
        K[:, 1, 2] = b2
        K[:, 1, 4] = -b1
        K[:, 1, 5] = b2
        K[:, 2, 1] = b2
        K[:, 2, 2] = b3
        K[:, 2, 4] = -b2
        K[:, 2, 5] = b4
        K[:, 4, 1] = -b1
        K[:, 4, 2] = -b2
        K[:, 4, 4] = b1
        K[:, 4, 5] = -b2
        K[:, 5, 1] = b2
        K[:, 5, 2] = b4
        K[:, 5, 4] = -b2
        K[:, 5, 5] = b3
        T = np.zeros((nb, 6, 6))
        for blk in (0, 3):
            T[:, blk, blk] = c
            T[:, blk, blk + 1] = s
            T[:, blk + 1, blk] = -s
            T[:, blk + 1, blk + 1] = c
            T[:, blk + 2, blk + 2] = 1.0
        return np.einsum("bji,bjk,bkl->bil", T, K, T)

    def stable_dt(self, safety: float = 0.35) -> float:
        """Explicit-integration step from the largest eigenfrequency of K M^-1."""
        from scipy.sparse.linalg import eigsh

        K = self.stiffness_matrix().tocsr()
        Minv = np.zeros(3 * self.n)
        Minv[0::3] = 1.0 / self.mass
        Minv[1::3] = 1.0 / self.mass
        Minv[2::3] = 1.0 / self.inertia
        scaled = K.multiply(Minv[:, None]).tocsc()
        try:
            lam = float(eigsh(scaled, k=1, which="LA",
                              return_eigenvectors=False)[0])
        except Exception:  # pragma: no cover - fall back to the CFL estimate
            return self.config.recommended_dt_s
        omega = math.sqrt(max(lam, 0.0))
        if not math.isfinite(omega) or omega <= 0.0:  # pragma: no cover
            return self.config.recommended_dt_s
        return safety * 2.0 / omega

    def stiffness_matrix(self):
        """Assemble the linear elastic global stiffness matrix (sparse, 3N x 3N)."""
        from scipy import sparse

        nb = self.n_bonds
        c, s = self.bc, self.bs
        L = self.blen
        a = self.EA / L
        b1 = 12.0 * self.EI / L**3
        b2 = 6.0 * self.EI / L**2
        b3 = 4.0 * self.EI / L
        b4 = 2.0 * self.EI / L
        K = np.zeros((nb, 6, 6))
        K[:, 0, 0] = a
        K[:, 0, 3] = -a
        K[:, 3, 0] = -a
        K[:, 3, 3] = a
        K[:, 1, 1] = b1
        K[:, 1, 2] = b2
        K[:, 1, 4] = -b1
        K[:, 1, 5] = b2
        K[:, 2, 1] = b2
        K[:, 2, 2] = b3
        K[:, 2, 4] = -b2
        K[:, 2, 5] = b4
        K[:, 4, 1] = -b1
        K[:, 4, 2] = -b2
        K[:, 4, 4] = b1
        K[:, 4, 5] = -b2
        K[:, 5, 1] = b2
        K[:, 5, 2] = b4
        K[:, 5, 4] = -b2
        K[:, 5, 5] = b3
        # local -> global rotation applied as T^T K T per block
        T = np.zeros((nb, 6, 6))
        for blk in (0, 3):
            T[:, blk, blk] = c
            T[:, blk, blk + 1] = s
            T[:, blk + 1, blk] = -s
            T[:, blk + 1, blk + 1] = c
            T[:, blk + 2, blk + 2] = 1.0
        Kg = self.Kg if hasattr(self, "Kg") else np.einsum("bji,bjk,bkl->bil", T, K, T)
        dofs = np.zeros((nb, 6), dtype=int)
        dofs[:, 0] = 3 * self.bi
        dofs[:, 1] = 3 * self.bi + 1
        dofs[:, 2] = 3 * self.bi + 2
        dofs[:, 3] = 3 * self.bj
        dofs[:, 4] = 3 * self.bj + 1
        dofs[:, 5] = 3 * self.bj + 2
        rows = np.repeat(dofs, 6, axis=1).ravel()
        cols = np.tile(dofs, (1, 6)).ravel()
        data = Kg.reshape(-1)
        stiffness = sparse.coo_matrix((data, (rows, cols)),
                                      shape=(3 * self.n, 3 * self.n)).tocsr()
        if self.config.buoyancy:
            k_f = self.k_foundation
            vert = 3 * np.arange(self.n) + 1
            stiffness = stiffness + sparse.coo_matrix(
                (np.full(self.n, k_f), (vert, vert)),
                shape=(3 * self.n, 3 * self.n)).tocsr()
        return stiffness

    def static_displacements(self, loads: np.ndarray) -> np.ndarray:
        """Exact linear elastic static solution under nodal ``loads`` (N, 2)."""
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        loads = np.asarray(loads, dtype=float)
        if loads.shape != (self.n, 2):
            raise ValueError("loads must have shape (n_particles, 2)")
        stiffness = self.stiffness_matrix()
        rhs = np.zeros(3 * self.n)
        rhs[0::3] = loads[:, 0]
        rhs[1::3] = loads[:, 1]
        fixed_dofs = np.repeat(self.fixed, 3)
        fixed_dofs[2::3] |= self.fixed
        free = ~fixed_dofs
        solution = np.zeros(3 * self.n)
        K_ff = stiffness[free][:, free].tocsc()
        solution[free] = spsolve(K_ff, rhs[free])
        self.u = solution.reshape(self.n, 3)
        return self.u[:, :2].copy()
