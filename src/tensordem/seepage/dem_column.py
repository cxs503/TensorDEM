"""Wall-bounded DEM seepage column: *measured* effective stress and piping onset.

Why this module exists
----------------------
:mod:`tensordem.seepage.effective_stress` and
:mod:`tensordem.seepage.critical_gradient` provide the **analytic** Terzaghi
yardsticks.  A yardstick is not a measurement: an error can only be reported
against something that was actually simulated.  This module supplies that
something -- a soft-sphere DEM assembly in a wall-bounded column whose contact
network is brought to static equilibrium, from which

* the vertical **effective stress profile** is read off the inter-particle
  contact forces (plane-cut / Love--Weber sum, *not* an assumed formula);
* the **critical hydraulic gradient** is located as the gradient above which no
  static equilibrium exists (the base reaction vanishes and the bed heaves);
* the **suffusion** (fine-particle loss) rate is measured by counting the fines
  that are actually transported out of the bed.

Physics and bookkeeping
-----------------------
``z`` points **up**; the base wall is at ``z = 0``.  Each particle feels

    m dv/dt = F_contact + m g (-zhat) + rho_w g V_p (+zhat) + F_seep (+zhat)

The contact law is a linear spring--dashpot in the normal direction plus a
viscous tangential damper; there is no Coulomb cap and no contact spin, so the
assembly is *frictionless in the static limit*.  That is a deliberate,
conservative idealisation for piping: real soil carries part of the load in
friction, so a frictionless DEM column is expected to heave at or slightly
below Terzaghi's gradient, never far above.

Static states are reached with an **overdamped (dynamic relaxation)** descent
``v = F/(m rate)`` instead of inertial integration: the question the piping
criterion asks is whether a *static* equilibrium exists at a given gradient, and
the overdamped scheme answers exactly that, with ``|v| -> 0`` as its own
convergence certificate.

The seepage force per particle comes from the interstitial flow, not from
synonymising gradient and force: the superficial velocity is obtained by
inverting Ergun at the *measured* bed porosity, the excess pressure gradient is
evaluated slab-by-slab with the *measured* local porosity, and the particle
force is the corresponding pressure drop distributed over the slab volume.  The
measured porosity heterogeneity therefore enters the pore-pressure profile,
which is what the hydrostatic comparison actually measures.

All quantities SI (m, kg, s, Pa).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..bedcoupling.closures import ergun_pressure_gradient, sphere_volume

__all__ = [
    "ColumnDEMConfig",
    "SeepageColumn",
    "WATER_DENSITY",
    "GRAVITY",
]

WATER_DENSITY = 1000.0
GRAVITY = 9.81


@dataclass(frozen=True)
class ColumnDEMConfig:
    """Configuration of a (mono- or bi-disperse) DEM column."""

    n_particles: int = 1600
    diameter: float = 1.0e-3
    density: float = 2650.0
    rho_w: float = WATER_DENSITY
    mu_w: float = 1.0e-3
    gravity: float = GRAVITY
    width_diameters: float = 10.0
    kn: float = 200.0
    normal_damping: float = 0.6
    tangential_damping: float = 0.5
    dt_fraction: float = 0.08
    seed: int = 20261111
    fine_ratio: float = 0.0
    fine_solid_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.n_particles < 8:
            raise ValueError("n_particles must be >= 8")
        for name in ("diameter", "density", "kn", "width_diameters"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.normal_damping < 1.0:
            raise ValueError("normal_damping must be in [0, 1)")
        if not 0.0 <= self.fine_ratio < 1.0:
            raise ValueError("fine_ratio must be in [0, 1)")
        if not 0.0 <= self.fine_solid_fraction < 0.9:
            raise ValueError("fine_solid_fraction must be in [0, 0.9)")
        if (self.fine_ratio > 0.0) != (self.fine_solid_fraction > 0.0):
            raise ValueError("fine_ratio and fine_solid_fraction must be set together")

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def box_width(self) -> float:
        return self.width_diameters * self.diameter

    @property
    def area(self) -> float:
        return self.box_width ** 2

    @property
    def particle_volume(self) -> float:
        return sphere_volume(self.diameter)

    @property
    def particle_mass(self) -> float:
        return self.density * self.particle_volume

    @property
    def dt(self) -> float:
        return self.dt_fraction * math.sqrt(self.particle_mass / self.kn)

    @property
    def fine_count(self) -> int:
        if self.fine_ratio <= 0.0:
            return 0
        coarse_solid = self.n_particles * self.particle_volume
        fine_volume = (self.fine_ratio * self.diameter) ** 3 * math.pi / 6.0
        return max(1, int(round(self.fine_solid_fraction / (1.0 - self.fine_solid_fraction)
                                * coarse_solid / fine_volume)))


class SeepageColumn:
    """Soft-sphere DEM column with buoyancy and seepage body forces."""

    def __init__(self, config: ColumnDEMConfig | None = None) -> None:
        cfg = config or ColumnDEMConfig()
        self._setup(cfg)
        rng = np.random.default_rng(cfg.seed)
        self.positions = self._initial_lattice(rng, self.radius, cfg)
        self._after_setup()

    # ------------------------------------------------------------------ setup
    def _setup(self, cfg: ColumnDEMConfig) -> None:
        self.cfg = cfg
        self.n = cfg.n_particles
        self.radius = np.full(self.n, cfg.radius)
        self.mass = np.full(self.n, cfg.particle_mass)
        self.is_fine = np.zeros(self.n, dtype=bool)
        self._box_width = cfg.box_width
        self.cutoff = 2.0 * cfg.radius * 1.02
        self.dt = cfg.dt
        self.velocities = np.zeros((self.n, 3))
        self.gradient = 0.0
        self.seepage_force = np.zeros((self.n, 3))
        self.time = 0.0
        self.steps_done = 0
        self._m_eff = 0.5 * cfg.particle_mass
        self._cn = 2.0 * cfg.normal_damping * math.sqrt(cfg.kn * self._m_eff)
        self._ct = cfg.tangential_damping * self._cn
        self._pairs = None
        self._pair_cache = None
        self._pair_anchor = None
        self._pair_age = 0
        self._skin = 0.15 * cfg.diameter
        self._bed_top_reference = None

    def _after_setup(self) -> None:
        pass

    @property
    def box_width(self) -> float:
        return self._box_width

    @property
    def area(self) -> float:
        return self._box_width ** 2

    def _initial_lattice(self, rng: np.random.Generator, radii: np.ndarray,
                         cfg: ColumnDEMConfig) -> np.ndarray:
        r_max = float(np.max(radii))
        spacing = 2.40 * r_max
        per_side = max(2, int(math.floor(cfg.box_width / spacing)))
        per_layer = per_side * per_side
        pts = []
        idx = 0
        while idx < self.n:
            layer = idx // per_layer
            for ix in range(per_side):
                for iy in range(per_side):
                    if idx >= self.n:
                        break
                    pts.append(((ix + 0.5) * spacing + rng.normal(0.0, 0.05 * spacing),
                                (iy + 0.5) * spacing + rng.normal(0.0, 0.05 * spacing),
                                (layer + 0.5) * spacing))
                    idx += 1
        pos = np.array(pts[: self.n], dtype=float)
        lo = radii + 0.1 * spacing
        hi = cfg.box_width - radii - 0.1 * spacing
        pos[:, 0] = np.clip(pos[:, 0], lo, hi)
        pos[:, 1] = np.clip(pos[:, 1], lo, hi)
        pos[:, 2] = np.maximum(pos[:, 2], radii * 1.5)
        return pos

    @classmethod
    def from_random_packing(cls, config: ColumnDEMConfig | None = None,
                            target_porosity: float = 0.42, layers: int = 2,
                            seed: int = 20261111,
                            precompress: float = 2.0e-3) -> "SeepageColumn":
        """Build a dense column from a quasi-statically prepared random packing.

        The packing comes from
        :func:`tensordem.bedcoupling.packing.make_random_close_packing` (a
        periodic cell relaxed until the maximum overlap is zero), stacked
        ``layers`` times in ``z`` so the column is taller than it is wide.
        Stacking a *periodic* cell keeps the contact network continuous across
        the stacking plane; only the free surface and the base are new.  Because
        a zero-overlap packing carries no load, it is preloaded by a small
        uniform compaction so that the contacts are born in compression.
        """
        from ..bedcoupling.packing import make_random_close_packing

        cfg = config or ColumnDEMConfig()
        per_layer = max(8, cfg.n_particles // max(1, layers))
        bed = make_random_close_packing(per_layer, cfg.diameter, target_porosity,
                                       seed=seed)
        pos = bed.positions.copy()
        stitched = [pos + np.array([0.0, 0.0, k * bed.box_length]) for k in range(max(1, layers))]
        positions = np.vstack(stitched)
        obj = cls.__new__(cls)
        obj._setup(cfg)
        obj.n = positions.shape[0]
        obj.radius = np.full(obj.n, cfg.radius)
        obj.mass = np.full(obj.n, cfg.particle_mass)
        obj.is_fine = np.zeros(obj.n, dtype=bool)
        obj._box_width = bed.box_length
        dxy = 0.5 * (obj._box_width - bed.box_length)
        centred = positions + np.array([dxy, dxy, 0.0])
        # seat the bottom layer exactly on the base wall (no artificial gap)
        centred[:, 2] += cfg.radius - float(centred[:, 2].min())
        if precompress > 0.0:
            centre = np.array([0.5 * obj._box_width, 0.5 * obj._box_width, 0.0])
            centred = centre + (centred - centre) * (1.0 - precompress)
        obj.positions = centred
        obj.velocities = np.zeros((obj.n, 3))
        obj.seepage_force = np.zeros((obj.n, 3))
        return obj

    # ------------------------------------------------------------- mechanics
    def _neighbour_pairs(self) -> np.ndarray:
        """Candidate pairs with a Verlet skin, rebuilt when the skin is exhausted."""
        if self._pair_cache is not None and self._pair_anchor is not None:
            moved = float(np.max(np.linalg.norm(self.positions - self._pair_anchor, axis=1)))
            if moved < self._skin * 0.5 and self._pair_age < 500:
                self._pair_age += 1
                return self._pair_cache
        tree = cKDTree(self.positions)
        self._pair_cache = tree.query_pairs(self.cutoff + 2.0 * self._skin,
                                            output_type="ndarray")
        self._pair_anchor = self.positions.copy()
        self._pair_age = 0
        return self._pair_cache

    def pair_forces(self, stiffness_scale: float = 1.0,
                    damping_scale: float = 1.0):
        """Overlapping pairs with compressive normal force.

        Returns ``(i, j, overlap, fn, n_ij)``; ``n_ij`` is the unit vector from
        ``i`` to ``j`` and ``fn`` the normal force magnitude [N].
        """
        pairs = self._neighbour_pairs()
        if pairs.size == 0:
            return (np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0),
                    np.zeros((0, 3)))
        i, j = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
        d = self.positions[j] - self.positions[i]
        dist = np.linalg.norm(d, axis=1)
        ok = dist > 1e-15
        i, j, d, dist = i[ok], j[ok], d[ok], dist[ok]
        delta = (self.radius[i] + self.radius[j]) - dist
        active = delta > 0.0
        i, j, d, dist, delta = i[active], j[active], d[active], dist[active], delta[active]
        n_ij = d / dist[:, None]
        vrel = self.velocities[j] - self.velocities[i]
        vn = np.einsum("ij,ij->i", vrel, n_ij)
        fn = np.maximum(self.cfg.kn * stiffness_scale * delta
                        - damping_scale * self._cn * vn, 0.0)
        return i, j, delta, fn, n_ij

    def assembly_forces(self, stiffness_scale: float = 1.0,
                        damping_scale: float = 1.0) -> np.ndarray:
        """Total force on every particle [N] for the current configuration."""
        cfg = self.cfg
        kn = cfg.kn * stiffness_scale
        forces = np.zeros((self.n, 3))
        i, j, delta, fn, n_ij = self.pair_forces(stiffness_scale, damping_scale)
        if i.size:
            vrel = self.velocities[j] - self.velocities[i]
            vn = np.einsum("ij,ij->i", vrel, n_ij)
            vt = vrel - vn[:, None] * n_ij
            fvec = -fn[:, None] * n_ij - damping_scale * self._ct * vt
            for k in range(3):
                forces[:, k] += np.bincount(i, weights=fvec[:, k], minlength=self.n)
                forces[:, k] -= np.bincount(j, weights=fvec[:, k], minlength=self.n)
        self._pairs = (i, j, delta, fn, n_ij)

        r = self.radius
        z = self.positions[:, 2]
        base = z < r
        if np.any(base):
            pen = r[base] - z[base]
            vz = self.velocities[base, 2]
            fz = kn * pen + damping_scale * self._cn * np.maximum(-vz, 0.0)
            forces[base, 2] += np.maximum(fz, 0.0)
        for axis, high in ((0, False), (0, True), (1, False), (1, True)):
            coord = self.positions[:, axis]
            pen = (coord + r - self._box_width) if high else (r - coord)
            hit = pen > 0.0
            if np.any(hit):
                vn = self.velocities[hit, axis]
                closing = np.maximum(vn if high else -vn, 0.0)
                f = np.maximum(kn * pen[hit] + damping_scale * self._cn * closing, 0.0)
                forces[hit, axis] += (-1.0 if high else 1.0) * f

        forces[:, 2] -= self.mass * cfg.gravity
        forces[:, 2] += cfg.rho_w * cfg.gravity * 4.0 / 3.0 * math.pi * r ** 3
        forces += self.seepage_force
        return forces

    def step(self, stiffness_scale: float = 1.0) -> None:
        """One explicit (symplectic Euler) dynamics step."""
        forces = self.assembly_forces(stiffness_scale)
        self.velocities += forces / self.mass[:, None] * self.dt
        self.positions += self.velocities * self.dt
        self.time += self.dt
        self.steps_done += 1

    def potential_energy(self) -> float:
        """Total potential energy [J] of the conservative part of the model."""
        kn = self.cfg.kn
        i, j, delta, fn, n_ij = self.pair_forces(damping_scale=0.0)
        energy = 0.5 * float(np.sum(kn * delta ** 2)) if delta.size else 0.0
        r = self.radius
        z = self.positions[:, 2]
        pen = np.maximum(r - z, 0.0)
        energy += 0.5 * kn * float(np.sum(pen ** 2))
        for axis, high in ((0, False), (0, True), (1, False), (1, True)):
            coord = self.positions[:, axis]
            p = (coord + r - self._box_width) if high else (r - coord)
            energy += 0.5 * kn * float(np.sum(np.maximum(p, 0.0) ** 2))
        weight = (self.mass * self.cfg.gravity
                  - self.cfg.rho_w * self.cfg.gravity * 4.0 / 3.0 * math.pi * r ** 3)
        energy += float(np.sum((weight[:, None] * np.array([0.0, 0.0, 1.0]))[:, 2] * z))
        energy -= float(np.sum(self.seepage_force[:, 2] * z))
        return energy

    def solve_static(self, max_iter: int = 4000, gtol: float = 1e-4) -> dict:
        """Static equilibrium by bounded-memory energy minimisation (L-BFGS).

        The conservative energy (:meth:`potential_energy`) is minimised with the
        analytic gradient ``-F`` (:meth:`assembly_forces` with damping off).  A
        converged result is a mechanically admissible static state; if the
        applied gradient exceeds the piping threshold no minimum exists and the
        optimiser drives the assembly upward instead of converging.
        """
        from scipy.optimize import minimize

        x0 = self.positions.ravel().copy()
        weight = max(self.submerged_weight_stress() * self.area, 1e-12)

        def fun(x: np.ndarray):
            self.positions = x.reshape(-1, 3)
            u = self.potential_energy()
            f = self.assembly_forces(damping_scale=0.0)
            return u, -f.ravel()

        res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": max_iter, "maxfun": 4 * max_iter,
                                "ftol": 1e-14, "gtol": gtol * weight,
                                "maxls": 50})
        self.positions = res.x.reshape(-1, 3)
        forces = self.assembly_forces(damping_scale=0.0)
        residual = float(np.max(np.linalg.norm(forces, axis=1)) / weight)
        return {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "iterations": int(res.nit),
            "energy_j": float(res.fun),
            "force_residual": residual,
            "base_effective_stress_pa": self.base_effective_stress(),
            "bed_top_m": self.solid_top(),
            "displacement_max_m": float(np.max(np.linalg.norm(
                self.positions - x0.reshape(-1, 3), axis=1))),
        }

    def static_relax(self, steps: int = 2000, rate: float = 3000.0) -> float:
        """Overdamped (dynamic relaxation) descent to a static equilibrium.

        ``v = F / (m rate)`` removes inertia instead of integrating it, so a
        mechanically admissible state is reached with ``|v| -> 0``; the final
        residual speed is returned as the convergence certificate.  This is the
        granular analogue of a dynamic-relaxation static solver and is what the
        *static* question "is there an equilibrium at this gradient?" needs.
        """
        speed = 0.0
        for _ in range(int(steps)):
            forces = self.assembly_forces(damping_scale=0.0)
            self.velocities = forces / (self.mass[:, None] * rate)
            self.positions += self.velocities * self.dt
            self.time += self.dt
            self.steps_done += 1
            speed = self.max_speed()
        return speed

    def settle(self, steps: int = 2000, rate: float = 3000.0) -> "SeepageColumn":
        """Overdamped preparation: settle the assembly to a static state."""
        self.static_relax(steps, rate)
        if self._bed_top_reference is None:
            self._bed_top_reference = self.solid_top()
        return self

    def kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.mass[:, None] * self.velocities ** 2))

    def max_speed(self) -> float:
        return float(np.max(np.linalg.norm(self.velocities, axis=1)))

    # --------------------------------------------------------- measurement API
    def bed_top(self, quantile: float = 0.99) -> float:
        return float(np.quantile(self.positions[:, 2], quantile))

    def solid_top(self) -> float:
        return self.bed_top() + float(self.radius.max())

    def porosity(self) -> float:
        volume = self.area * self.solid_top()
        solid = float(np.sum(4.0 / 3.0 * math.pi * self.radius ** 3))
        return 1.0 - solid / volume

    def porosity_profile(self, cuts: np.ndarray) -> np.ndarray:
        cuts = np.asarray(cuts, dtype=float)
        vol_p = 4.0 / 3.0 * math.pi * self.radius ** 3
        solid = np.histogram(self.positions[:, 2], bins=cuts, weights=vol_p)[0]
        slab_vol = self.area * np.diff(cuts)
        return np.clip(1.0 - solid / slab_vol, 0.25, 0.98)

    def stress_profile(self, cuts: np.ndarray) -> np.ndarray:
        """Vertical effective stress [Pa] on each horizontal cut plane.

        ``sigma'(z_c) = (1/A) sum_{contacts crossing z_c} f_n |n_z|``
        (plane-cut / Love--Weber construction); compression positive.
        """
        i, j, delta, fn, n_ij = self._pairs if self._pairs is not None else self.pair_forces()
        cuts = np.atleast_1d(np.asarray(cuts, dtype=float))
        if i.size == 0:
            return np.zeros_like(cuts)
        zi = self.positions[i, 2]
        zj = self.positions[j, 2]
        lo, hi = np.minimum(zi, zj), np.maximum(zi, zj)
        f_proj = fn * np.abs(n_ij[:, 2])
        out = np.empty_like(cuts)
        for k, zc in enumerate(cuts):
            crossing = (lo < zc) & (hi >= zc)
            out[k] = float(np.sum(f_proj[crossing])) / self.area
        return out

    def base_effective_stress(self) -> float:
        """Base reaction per unit area from the wall contacts [Pa]."""
        r = self.radius
        z = self.positions[:, 2]
        hit = z < r
        if not np.any(hit):
            return 0.0
        vz = self.velocities[hit, 2]
        fz = self.cfg.kn * (r[hit] - z[hit]) + self._cn * np.maximum(-vz, 0.0)
        return float(np.sum(np.maximum(fz, 0.0))) / self.area

    def submerged_weight_stress(self) -> float:
        solid = float(np.sum(4.0 / 3.0 * math.pi * self.radius ** 3))
        return (self.cfg.density - self.cfg.rho_w) * self.cfg.gravity * solid / self.area

    def mean_contact_force(self) -> float:
        i, j, delta, fn, n_ij = self.pair_forces()
        return float(np.mean(fn)) if fn.size else 0.0

    # ------------------------------------------------------- fluid + coupling
    def fluid_profile(self, gradient: float, n_slabs: int | None = None) -> dict:
        """Solve the 1-D interstitial flow and return the pore-pressure profile.

        The slab thickness is kept at or above two particle diameters; a thinner
        slab makes the centre-count porosity (and therefore the Ergun drop)
        meaningless at the particle scale.
        """
        if gradient < 0.0:
            raise ValueError("gradient must be non-negative")
        top = self.solid_top()
        if n_slabs is None:
            n_slabs = max(4, min(40, int(top / (2.0 * self.cfg.diameter))))
        cuts = np.linspace(0.0, top, n_slabs + 1)
        eps_slab = self.porosity_profile(cuts)
        eps_mean = self.porosity()
        target = gradient * self.cfg.rho_w * self.cfg.gravity
        u_s = self.invert_ergun(target, eps_mean)
        drops = np.array([
            ergun_pressure_gradient(u_s, diameter=self.cfg.diameter, porosity=e,
                                    rho_fluid=self.cfg.rho_w, mu=self.cfg.mu_w).total
            for e in eps_slab
        ])
        dz = np.diff(cuts)
        z_mid = 0.5 * (cuts[:-1] + cuts[1:])
        # the slab-wise Ergun drop of a heterogeneous bed does not average to the
        # bed-mean drop (Ergun is convex in porosity): rescale the local field so
        # the applied *bulk* gradient is exactly the requested one, and report
        # the rescale factor as the measured heterogeneity effect.
        total = float(np.sum(drops * dz))
        homogenisation_factor = 1.0
        if total > 0.0 and target > 0.0:
            homogenisation_factor = (target * top) / total
            drops = drops * homogenisation_factor
        excess_above = np.cumsum((drops * dz)[::-1])[::-1]
        p_mid = excess_above + self.cfg.rho_w * self.cfg.gravity * (top - z_mid)
        return {
            "z": z_mid,
            "pore_pressure": p_mid,
            "excess_gradient": drops / (self.cfg.rho_w * self.cfg.gravity),
            "porosity_slab": eps_slab,
            "superficial_velocity": u_s,
            "bulk_gradient": float(gradient),
            "measured_gradient": float(np.sum(drops * dz)
                                       / (top * self.cfg.rho_w * self.cfg.gravity)),
            "homogenisation_factor": float(homogenisation_factor),
            "bed_top": top,
        }

    def invert_ergun(self, target_drop: float, eps: float) -> float:
        if target_drop <= 0.0:
            return 0.0
        lo, hi = 0.0, 10.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            drop = ergun_pressure_gradient(
                mid, diameter=self.cfg.diameter, porosity=eps,
                rho_fluid=self.cfg.rho_w, mu=self.cfg.mu_w).total
            if drop < target_drop:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def set_gradient(self, gradient: float, n_slabs: int | None = None) -> dict:
        """Apply upward seepage corresponding to the bulk hydraulic gradient."""
        if gradient < 0.0:
            raise ValueError("gradient must be non-negative")
        self.gradient = float(gradient)
        prof = self.fluid_profile(gradient, n_slabs)
        if gradient == 0.0:
            self.seepage_force[:] = 0.0
            return prof
        z = prof["z"]
        edges = np.concatenate([[0.0], 0.5 * (z[1:] + z[:-1]), [np.inf]])
        idx = np.clip(np.searchsorted(edges, self.positions[:, 2], side="right") - 1,
                      0, len(z) - 1)
        excess = prof["excess_gradient"][idx] * self.cfg.rho_w * self.cfg.gravity
        eps_local = prof["porosity_slab"][idx]
        vol_p = 4.0 / 3.0 * math.pi * self.radius ** 3
        self.seepage_force[:] = 0.0
        self.seepage_force[:, 2] = excess * vol_p / (1.0 - eps_local)
        return prof

    def fluid_momentum_residual(self, gradient: float, n_slabs: int | None = None) -> dict:
        """Residual of the fluid momentum balance implied by the particle forces."""
        prof = self.fluid_profile(gradient, n_slabs)
        top = prof["bed_top"]
        z = prof["z"]
        edges = np.concatenate([[0.0], 0.5 * (z[1:] + z[:-1]), [top]])
        slab_idx = np.clip(np.searchsorted(edges, self.positions[:, 2], side="right") - 1,
                           0, len(z) - 1)
        applied = np.zeros(len(z))
        for k in range(len(z)):
            sel = slab_idx == k
            if np.any(sel):
                applied[k] = np.sum(self.seepage_force[sel, 2]) / (
                    self.area * (edges[k + 1] - edges[k]))
        drop = prof["excess_gradient"] * self.cfg.rho_w * self.cfg.gravity
        scale = max(float(np.max(np.abs(drop))), 1e-12)
        return {
            "slab_drag_density_pa_per_m": applied,
            "slab_excess_gradient_pa_per_m": drop,
            "relative_residual": float(np.max(np.abs(applied - drop)) / scale),
            "slab_porosity": prof["porosity_slab"],
        }

    def mass_ledger(self) -> dict:
        solid = float(np.sum(self.mass))
        vol_p = float(np.sum(4.0 / 3.0 * math.pi * self.radius ** 3))
        top = self.solid_top()
        pore = self.area * top - vol_p
        return {
            "solid_mass_kg": solid,
            "pore_volume_m3": pore,
            "fluid_mass_kg": self.cfg.rho_w * max(pore, 0.0),
            "bed_height_m": top,
            "porosity": 1.0 - vol_p / (self.area * top),
            "area_m2": self.area,
        }

    def fine_loss(self) -> dict:
        """Measured fine-particle loss relative to the prepared bed top."""
        if not np.any(self.is_fine):
            return {"fine_mass_initial_kg": 0.0, "fine_mass_lost_kg": 0.0,
                    "fine_loss_fraction": 0.0, "escaped_count": 0, "fine_count": 0}
        top = self._bed_top_reference if self._bed_top_reference is not None else self.solid_top()
        initial = float(np.sum(self.mass[self.is_fine]))
        escaped = self.is_fine & (self.positions[:, 2] > top)
        lost = float(np.sum(self.mass[escaped]))
        return {
            "fine_mass_initial_kg": initial,
            "fine_mass_lost_kg": lost,
            "fine_loss_fraction": lost / initial if initial > 0 else 0.0,
            "escaped_count": int(np.count_nonzero(escaped)),
            "fine_count": int(np.count_nonzero(self.is_fine)),
        }

    def snapshot(self) -> dict:
        return {
            "time_s": self.time,
            "steps": self.steps_done,
            "gradient": self.gradient,
            "bed_top_m": self.solid_top(),
            "porosity_measured": self.porosity(),
            "base_effective_stress_pa": self.base_effective_stress(),
            "submerged_weight_stress_pa": self.submerged_weight_stress(),
            "max_speed_m_s": self.max_speed(),
            "kinetic_energy_j": self.kinetic_energy(),
        }