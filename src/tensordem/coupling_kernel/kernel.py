"""Backend-agnostic DEM kernel used as the coupling-chain DEM leg.

Physics
-------
Monodisperse spheres, translational degrees of freedom only (no spin):

* normal contact: Hertz law ``F_n = K δ^{3/2}`` with ``K = (4/3) E* sqrt(R*)``
  plus a linear dashpot ``F_d = -gamma * v_n`` (Cundall–Strack style).  The
  dashpot coefficient is derived from a target coefficient of restitution
  ``e`` and a reference contact stiffness (``zeta = -ln e / sqrt(pi^2 + ln^2 e)``,
  ``gamma = 2 zeta sqrt(m_eff k_ref)``) — a calibrated damping closure, not a
  tuned constant.
* walls: box faces are analytic half-spaces with the same normal law.
* gravity: net (buoyant) weight ``(rho_p - rho_f) V g``.
* fluid coupling: linear Stokes drag ``-c (v - u_f)`` or Schiller–Naumann
  ``-c (1 + 0.15 Re^0.687)(v - u_f)``.  This is the momentum the DEM leg hands
  to the flow leg and is booked as ``drag_dissipated_J``.

Discretisation: kick–drift–kick velocity-Verlet with the dissipative force
evaluated at the half-step velocity (standard DEM practice).  Dissipation is
integrated with the trapezoidal rule across the two force evaluations of a
step, which keeps the ledger closure second order.

Backends
--------
``backend="numpy"`` (reference, default) or ``backend="torch"`` (optional
accelerator).  Contact detection always runs in numpy on the host
(:mod:`tensordem.coupling_kernel.celllist`) so both backends see identical pair
lists; pair forces, integration and the energy reductions then run through
:class:`~tensordem.coupling_kernel.backend.ArrayBackend`.  Boundary closures
(walls, gravity, drag) are evaluated on the host in numpy, which keeps them
exactly identical across backends.

The exponent ``δ^{3/2}`` is evaluated as ``δ * sqrt(δ)`` (never ``pow``) so that
numpy and torch agree bit-for-bit on the Hertz term.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from .backend import ArrayBackend, get_backend
from .celllist import candidate_pairs
from .ledger import EnergyLedger

__all__ = ["DEMKernelConfig", "DEMKernel", "hertz_prefactor", "damping_from_restitution"]

_DRAG_MODELS = ("none", "stokes", "schiller_naumann")


def hertz_prefactor(young_modulus: float, poisson_ratio: float, radius_star: float) -> float:
    """``K`` in ``F_n = K δ^{3/2}`` for two identical spheres."""

    e_star = young_modulus / (2.0 * (1.0 - poisson_ratio**2))
    return (4.0 / 3.0) * e_star * math.sqrt(radius_star)


def damping_from_restitution(restitution: float, m_eff: float, stiffness: float) -> float:
    """Linear dashpot coefficient for a target restitution and contact stiffness."""

    e = min(max(float(restitution), 1e-6), 1.0)
    if e >= 1.0:
        return 0.0
    ln_e = math.log(e)
    zeta = -ln_e / math.sqrt(math.pi**2 + ln_e**2)
    return 2.0 * zeta * math.sqrt(m_eff * stiffness)


@dataclass(frozen=True)
class DEMKernelConfig:
    """Physical + numerical settings of the DEM leg."""

    radius: float
    density: float
    dt: float = 1.0e-6
    young_modulus: float = 1.0e7
    poisson_ratio: float = 0.25
    restitution: float = 0.1
    damping_override: float | None = None
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    fluid_density: float = 0.0
    fluid_viscosity: float = 0.0
    fluid_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    drag_model: str = "none"
    box_lo: tuple[float, float, float] = (0.0, 0.0, 0.0)
    box_hi: tuple[float, float, float] = (1e-3, 1e-3, 1e-3)
    wall_axes: tuple[bool, bool, bool] = (True, True, True)
    damping_reference_overlap_fraction: float = 1.0e-3

    def __post_init__(self) -> None:
        for name in ("radius", "density", "young_modulus", "dt"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not (0.0 <= self.poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must be in [0, 0.5)")
        if not (0.0 < self.restitution <= 1.0):
            raise ValueError("restitution must be in (0, 1]")
        if self.drag_model not in _DRAG_MODELS:
            raise ValueError(f"drag_model must be one of {_DRAG_MODELS}")
        lo = np.asarray(self.box_lo, dtype=float)
        hi = np.asarray(self.box_hi, dtype=float)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("box_lo/box_hi must have three components")
        if np.any(hi - lo <= 2.0 * self.radius):
            raise ValueError("box must be wider than one particle diameter")

    # -- derived quantities ----------------------------------------------
    @property
    def volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius**3

    @property
    def mass(self) -> float:
        return self.density * self.volume

    @property
    def fluid_mass(self) -> float:
        return self.fluid_density * self.volume

    @property
    def net_mass(self) -> float:
        return self.mass - self.fluid_mass

    @property
    def gravity_vector(self) -> np.ndarray:
        return np.asarray(self.gravity, dtype=float)

    @property
    def weight(self) -> np.ndarray:
        return self.net_mass * self.gravity_vector

    @property
    def contact_prefactor(self) -> float:
        return hertz_prefactor(self.young_modulus, self.poisson_ratio, 0.5 * self.radius)

    @property
    def reference_stiffness(self) -> float:
        delta_ref = self.damping_reference_overlap_fraction * self.radius
        return 1.5 * self.contact_prefactor * math.sqrt(delta_ref)

    @property
    def stokes_coefficient(self) -> float:
        return 6.0 * math.pi * self.fluid_viscosity * self.radius

    @property
    def damping_coefficient(self) -> float:
        """Dashpot coefficient actually used (override wins over restitution)."""

        if self.damping_override is not None:
            return float(self.damping_override)
        if self.restitution >= 1.0:
            return 0.0
        return damping_from_restitution(
            self.restitution, 0.5 * self.mass, self.reference_stiffness
        )

    def as_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out.update(
            mass_kg=self.mass,
            net_mass_kg=self.net_mass,
            volume_m3=self.volume,
            contact_prefactor_N_m15=self.contact_prefactor,
            damping_coefficient_N_s_m=self.damping_coefficient,
            stokes_coefficient_N_s_m=self.stokes_coefficient,
        )
        return out


class DEMKernel:
    """Translational soft-sphere DEM with a switchable array backend."""

    def __init__(
        self,
        config: DEMKernelConfig,
        positions,
        velocities=None,
        backend: str | ArrayBackend = "numpy",
        record_energy_every: int = 0,
        name: str = "",
    ) -> None:
        self.config = config
        self.be: ArrayBackend = (
            backend if isinstance(backend, ArrayBackend) else get_backend(backend)
        )
        self.name = name
        pos = np.asarray(positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must have shape (n, 3)")
        if pos.shape[0] == 0:
            raise ValueError("at least one particle is required")
        lo = np.asarray(config.box_lo)
        hi = np.asarray(config.box_hi)
        if np.any(pos < lo - 1e-30) or np.any(pos > hi + 1e-30):
            raise ValueError("initial positions must lie inside the box")
        vel = np.zeros_like(pos) if velocities is None else np.asarray(velocities, dtype=float)
        if vel.shape != pos.shape:
            raise ValueError("velocities must have the same shape as positions")

        self.n = pos.shape[0]
        self.pos = self.be.asarray(pos)
        self.vel = self.be.asarray(vel)
        self.time = 0.0
        self.step_count = 0
        self.cutoff = 2.0 * config.radius * (1.0 + 1e-12)
        self._gamma = config.damping_coefficient
        self._pair_i_np = np.zeros(0, dtype=np.int64)
        self._pair_j_np = np.zeros(0, dtype=np.int64)
        self._pair_i_be = None
        self._pair_j_be = None
        self._pair_stamp = -1

        self.dissipation = {
            "dem_damping_dissipated_J": 0.0,
            "wall_damping_dissipated_J": 0.0,
            "drag_dissipated_J": 0.0,
        }
        self._elastic_J = 0.0
        self.ledger = EnergyLedger()
        self.record_energy_every = int(record_energy_every)
        # prime contact diagnostics / elastic energy at t=0
        self._update_contact_state()
        self.ledger.record(0, 0.0, **self._energy_terms())

    # -- state ------------------------------------------------------------
    @property
    def positions(self) -> np.ndarray:
        return np.array(self.be.to_numpy(self.pos), dtype=float)

    @property
    def velocities(self) -> np.ndarray:
        return np.array(self.be.to_numpy(self.vel), dtype=float)

    def state(self) -> dict:
        return {
            "positions": self.positions,
            "velocities": self.velocities,
            "time_s": self.time,
            "step": self.step_count,
        }

    def kinetic_energy(self) -> float:
        be = self.be
        return 0.5 * self.config.mass * float(be.scalar(be.sum(be.rowsquare_norm(self.vel))))

    # -- contact detection (numpy host core) -----------------------------
    def _pairs(self):
        if self._pair_stamp == self.step_count and self._pair_i_be is not None:
            return self._pair_i_be, self._pair_j_be
        i, j = candidate_pairs(
            self.positions, self.cutoff, self.config.box_lo, self.config.box_hi
        )
        self._pair_i_np, self._pair_j_np = i, j
        self._pair_i_be = self.be.index(i)
        self._pair_j_be = self.be.index(j)
        self._pair_stamp = self.step_count
        return self._pair_i_be, self._pair_j_be

    # -- force evaluation -------------------------------------------------
    def _pair_forces(self, pos, vel):
        """Backend-side Hertz pair forces -> (flat force, elastic J, diss rate W)."""

        be = self.be
        cfg = self.config
        gamma = self._gamma
        i_idx, j_idx = self._pairs()
        if len(self._pair_i_np) == 0:
            return be.asarray(np.zeros((self.n, 3))), 0.0, 0.0
        d = be.gather(pos, j_idx) - be.gather(pos, i_idx)
        dist = be.sqrt(be.rowsquare_norm(d))
        overlap = be.maximum(2.0 * cfg.radius - dist, 0.0)
        safe = be.maximum(dist, 1e-300)
        inv = be.where(dist > 0.0, 1.0 / safe, 0.0)
        nx = d[:, 0] * inv
        ny = d[:, 1] * inv
        nz = d[:, 2] * inv
        dv = be.gather(vel, j_idx) - be.gather(vel, i_idx)
        vn = dv[:, 0] * nx + dv[:, 1] * ny + dv[:, 2] * nz
        # δ^{3/2} as δ * sqrt(δ): bit-identical across backends
        fn = cfg.contact_prefactor * (overlap * be.sqrt(overlap))
        fmag = be.maximum(fn - gamma * vn, 0.0)
        fx, fy, fz = fmag * nx, fmag * ny, fmag * nz
        cols = []
        for comp in (fx, fy, fz):
            cols.append(be.accumulate(self.n, j_idx, comp) - be.accumulate(self.n, i_idx, comp))
        force = be.assemble(cols)
        elastic = float(be.scalar(be.sum(0.4 * fn * overlap)))
        rate = float(be.scalar(be.sum(gamma * vn * vn)))
        return force, elastic, rate

    def _host_forces(self, vel_np):
        """Walls + buoyant weight + fluid drag, evaluated in numpy (host)."""

        cfg = self.config
        pos_np = self.positions
        be = self.be
        radius = cfg.radius
        K = cfg.contact_prefactor
        gamma = self._gamma
        lo = np.asarray(cfg.box_lo, dtype=float)
        hi = np.asarray(cfg.box_hi, dtype=float)
        force = np.zeros_like(pos_np)
        elastic = 0.0
        wall_rate = 0.0
        for axis, enabled in enumerate(cfg.wall_axes):
            if not enabled:
                continue
            for side, plane in ((+1.0, lo[axis]), (-1.0, hi[axis])):
                pen = radius - side * (pos_np[:, axis] - plane)
                active = pen > 0.0
                if not np.any(active):
                    continue
                pen_a = pen[active]
                vn = side * vel_np[active, axis]
                fn = K * pen_a * np.sqrt(pen_a)
                fmag = np.maximum(fn - gamma * vn, 0.0)
                force[active, axis] += side * fmag
                elastic += float(np.sum(0.4 * fn * pen_a))
                wall_rate += float(np.sum(gamma * vn * vn))
        force += np.tile(cfg.weight, (self.n, 1))
        drag_rate = 0.0
        if cfg.drag_model != "none":
            u = np.asarray(cfg.fluid_velocity, dtype=float)
            rel = vel_np - u
            speed = np.sqrt(np.sum(rel * rel, axis=1))
            c = cfg.stokes_coefficient
            if cfg.drag_model == "schiller_naumann":
                re = cfg.fluid_density * 2.0 * radius * speed / max(cfg.fluid_viscosity, 1e-300)
                c_eff = c * (1.0 + 0.15 * np.power(re, 0.687))
            else:
                c_eff = np.full(self.n, c)
            force -= c_eff[:, None] * rel
            drag_rate = float(np.sum(c_eff * speed * speed))
        rates = {
            "dem_damping_dissipated_J": 0.0,  # filled by caller
            "wall_damping_dissipated_J": wall_rate,
            "drag_dissipated_J": drag_rate,
        }
        return be.asarray(force), elastic, rates

    def _update_contact_state(self):
        """Recompute pair forces / elastic energy at the current state."""

        force_pair, elastic_pair, dem_rate = self._pair_forces(self.pos, self.vel)
        force_host, elastic_host, rates = self._host_forces(self.velocities)
        rates["dem_damping_dissipated_J"] = dem_rate
        self._elastic_J = elastic_pair + elastic_host
        self._rates = rates
        return force_pair + force_host

    # -- energy bookkeeping ------------------------------------------------
    def _energy_terms(self) -> dict:
        cfg = self.config
        pos_np = self.positions
        gravity_potential = float(
            -cfg.net_mass * np.sum(pos_np @ cfg.gravity_vector)
        )
        return {
            "kinetic_J": self.kinetic_energy(),
            "gravity_potential_J": gravity_potential,
            "elastic_J": float(self._elastic_J),
            **{k: v for k, v in self.dissipation.items()},
        }

    # -- integration -------------------------------------------------------
    def step(self, n_steps: int = 1) -> None:
        cfg = self.config
        dt = cfg.dt
        half_over_m = 0.5 * dt / cfg.mass
        be = self.be
        for _ in range(int(n_steps)):
            force = self._update_contact_state()
            rates0 = self._rates
            self.vel = self.vel + force * half_over_m
            self.pos = self.pos + self.vel * dt
            self.step_count += 1
            self.time += dt
            force = self._update_contact_state()
            rates1 = self._rates
            self.vel = self.vel + force * half_over_m
            for key in self.dissipation:
                self.dissipation[key] += 0.5 * (rates0[key] + rates1[key]) * dt
            if self.record_energy_every and self.step_count % self.record_energy_every == 0:
                self.record()

    def record(self) -> dict:
        return self.ledger.record(self.step_count, self.time, **self._energy_terms())

    # -- diagnostics -------------------------------------------------------
    def contact_diagnostics(self) -> dict:
        """Contact statistics at the current state (numpy host evaluation)."""

        self._pair_stamp = -1  # force a re-detect
        self._pairs()
        pos = self.positions
        radius = self.config.radius
        if len(self._pair_i_np) == 0:
            return {
                "contacts": 0,
                "mean_coordination": 0.0,
                "max_overlap_over_radius": 0.0,
                "max_pair_force_N": 0.0,
            }
        d = pos[self._pair_j_np] - pos[self._pair_i_np]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        overlap = np.maximum(2.0 * radius - dist, 0.0)
        active = overlap > 0.0
        fn = self.config.contact_prefactor * overlap[active] * np.sqrt(overlap[active])
        return {
            "contacts": int(np.count_nonzero(active)),
            "mean_coordination": float(2.0 * np.count_nonzero(active) / self.n),
            "max_overlap_over_radius": float(overlap.max() / radius),
            "max_pair_force_N": float(fn.max()) if fn.size else 0.0,
        }

    def summary(self) -> dict:
        return {
            "name": self.name,
            "backend": self.be.name,
            "particles": self.n,
            "steps": self.step_count,
            "time_s": self.time,
            "config": self.config.as_dict(),
            "ledger": self.ledger.summary(),
        }