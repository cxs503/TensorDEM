"""DEM oedometer: micro-mechanical ``E_oed``, porosity and permeability.

This is the particle-level link of the consolidation chain.  A saturated
granular bed is built as a deterministic HCP lattice with a seeded jitter in a
rigid oedometer cell (frictionless rigid side walls, rigid base, dead-weight
piston lid) and loaded in two stress increments.  The cell reuses the
repository's 3-D granular contact kernel
(:mod:`tensordem.granular3d_core`: linear spring-dashpot normal law,
Cundall-Strack tangential spring saturated at ``mu F_n``, rolling-resistance
couple), so the measured stiffness is produced by the same contact law as every
other granular benchmark in this repository.

What the DEM delivers (all *measured*, none assumed):

``E_oed``
    secant oedometer constrained modulus ``d_sigma / d_eps_v`` between the two
    load stages, ``d_eps_v = (h1 - h2)/h1``; the axial strain is read off the
    piston travel, so this is a genuine K0 compression of the packing;
``n``
    porosity at each stage from the (fixed) solid volume over the cell volume;
``Z``
    mean coordination number of the packing at each stage;
``k_hyd``
    hydraulic conductivity from the Carman-Kozeny closure evaluated with the
    *measured* porosity and the grain diameter,
    ``k_int = n^3 d^2 / (180 (1-n)^2)``, ``K = k_int gamma_w / mu_w``
    (closed-form literature closure, flagged as such);
``Re_p``
    pore Reynolds number ``rho_w v_s d / mu_w`` with ``v_s = K i`` and the
    mean hydraulic gradient ``i = d_sigma/(gamma_w H_dr)`` -- the witness the
    applicability gate consumes.

The consolidation curves themselves are *not* produced here: ``E_oed`` and
``k_hyd`` are handed to :class:`tensordem.consolidation.poroelastic.PoroelasticColumn`,
which assembles ``C_v = k E_oed / gamma_w`` from them.  Keeping that boundary
is what makes the ``C_v`` inversion a test rather than a tautology.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from tensordem.granular3d_core import (  # noqa: F401  (public re-use)
    ContactState,
    Material3D,
    build_pairs,
    contact_forces,
    hcp_positions_in_box,
)

__all__ = [
    "OedometerConfig",
    "OedometerCell",
    "carman_kozeny_conductivity",
    "pore_reynolds",
    "run_oedometer",
]

WATER_DENSITY = 1000.0
WATER_VISCOSITY = 1.0e-3
WATER_UNIT_WEIGHT = 9810.0


@dataclass(frozen=True)
class OedometerConfig:
    """Geometry, material and load protocol of the DEM oedometer (SI)."""

    radius: float = 1.0e-4          # grain radius (m): silt / fine sand
    density: float = 2650.0         # grain density (kg/m^3)
    n_row: int = 4                  # particles per row in x and y
    n_layer: int = 5                # initial layers along z
    kn: float = 2.0e3               # pair normal stiffness (N/m)
    friction: float = 0.40
    mu_rolling: float = 0.02
    restitution: float = 0.20
    gravity: float = 9.81
    initial_solid_fraction: float = 0.25   # RSA deposition density (n = 0.75)
    rsa_min_gap: float = 0.005             # RSA min separation (fraction of d)
    unjam_steps: int = 200                 # gravity-free de-overlap pass
    compaction_porosity: float | None = None  # optional pre-consolidation target
    lid_speed_fast: float = 3.0e-1          # roughing piston speed (m/s)
    compaction_speed: float = 5.0e-2        # displacement-control speed (m/s)
    seed: int = 20261108
    stress_levels: tuple[float, float] = (5.0e3, 15.0e3)  # Pa, two increments
    bulk_damping_rate: float = 2.0e2    # viscous bulk damping (1/s, quasi-static)
    lid_velocity_limit: float = 5.0e-3   # quasi-static piston speed (m/s)
    settle_speed_tol: float = 1.0e-4  # m/s
    max_steps_per_stage: int = 400_000
    check_every: int = 200
    stress_tolerance: float = 0.02  # 2 % of the target lid stress
    hold_checks: int = 3

    @property
    def material(self) -> Material3D:
        return Material3D(
            radius=self.radius,
            density=self.density,
            mu=self.friction,
            mu_rolling=self.mu_rolling,
            restitution=self.restitution,
            kn=self.kn,
            gravity=self.gravity,
        )

    @property
    def spacing(self) -> float:
        return 2.0 * self.radius * (1.0 + self.rsa_min_gap)

    @property
    def box_width(self) -> float:
        """Cell width holding ``n_row`` lattice sites (walls just outside)."""
        return self.n_row * self.spacing

    @property
    def particle_count(self) -> int:
        return self.n_row * self.n_row * self.n_layer

    @property
    def timestep(self) -> float:
        """10 steps per contact period at the stiffest (smallest) mass."""
        period = 2.0 * math.pi * math.sqrt(0.5 * self.material.mass / self.kn)
        return 0.05 * period


class OedometerCell:
    """Rigid-walled, dead-weight-piston oedometer cell (numpy DEM)."""

    def __init__(self, config: OedometerConfig | None = None) -> None:
        self.cfg = cfg = config or OedometerConfig()
        self.mat = cfg.material
        width = cfg.box_width
        self.width = width
        self.pos = self._initial_packing(cfg)
        self.vel = np.zeros_like(self.pos)
        self.omega = np.zeros_like(self.pos)
        self.state = ContactState()
        self.n = len(self.pos)
        self.mass = np.full(self.n, cfg.material.mass)
        # piston starts just in contact with the top layer (no free flight)
        self.lid_z = float(self.pos[:, 2].max() + cfg.radius)
        self.preload = 0.0
        self.lid_mass = float(self.n * cfg.material.mass)
        self.lid_velocity = 0.0
        self.dt = cfg.timestep
        self.steps = 0
        self.gravity_on = False  # enabled after the de-overlap pass
        self._manual_lid = False  # displacement-controlled piston
        self.lid_pressure_history: list[float] = []
        self.dz = 1.0

    # -- geometry --------------------------------------------------------
    @staticmethod
    def _initial_packing(cfg: OedometerConfig) -> np.ndarray:
        """Seeded random sequential deposition at a loose solid fraction.

        A lattice start is *not* used: a simple-cubic column keeps its vertical
        contact chains (``Z ~ 1.6``), which is a 1-D chain stiffness rather than
        a constrained modulus.  Random deposition followed by piston
        compaction gives an isotropic random packing -- the configuration the
        Carman-Kozeny porosity window ``0.30 <= n <= 0.55`` is calibrated for.

        The helper is written locally because ``hcp_positions_in_box`` carries
        an absolute ``1e-3 m`` wall margin, which is larger than the whole cell
        at silt grain sizes.
        """
        r = cfg.radius
        w = cfg.box_width
        n = cfg.particle_count
        solid = n * 4.0 / 3.0 * math.pi * r ** 3
        height = solid / (w * w * cfg.initial_solid_fraction)
        min_dist = 2.0 * r * (1.0 + cfg.rsa_min_gap)
        rng = np.random.default_rng(cfg.seed)
        placed = np.empty((n, 3))
        count = 0
        attempts = 0
        max_attempts = 2_000_000
        while count < n:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "random deposition failed: lower initial_solid_fraction "
                    f"({cfg.initial_solid_fraction}) or enlarge the cell"
                )
            cand = np.array([
                rng.uniform(r * 1.001, w - r * 1.001),
                rng.uniform(r * 1.001, w - r * 1.001),
                rng.uniform(r * 1.001, height - r * 1.001),
            ])
            if count:
                d = np.linalg.norm(placed[:count] - cand, axis=1).min()
                if d < min_dist:
                    continue
            placed[count] = cand
            count += 1
        self_height = float(placed[:, 2].max() + r)
        placed[:, 2] = np.minimum(placed[:, 2], self_height)
        return placed

    @property
    def area(self) -> float:
        return self.width ** 2

    @property
    def cell_height(self) -> float:
        return self.lid_z

    def solid_volume(self) -> float:
        return self.n * 4.0 / 3.0 * math.pi * self.cfg.radius ** 3

    def porosity(self) -> float:
        return 1.0 - self.solid_volume() / (self.area * self.cell_height)

    # -- forces ----------------------------------------------------------
    def pair_contacts(self) -> tuple[np.ndarray, np.ndarray]:
        iu, ju = build_pairs(self.pos, self.cfg.radius)
        if len(iu) == 0:
            return iu, ju
        dist = np.linalg.norm(self.pos[ju] - self.pos[iu], axis=1)
        keep = dist < 2.0 * self.cfg.radius
        return iu[keep], ju[keep]

    def contact_count(self) -> np.ndarray:
        """Number of active contacts per particle (coordination number)."""
        iu, ju = self.pair_contacts()
        counts = np.zeros(self.n, dtype=np.int64)
        if len(iu):
            np.add.at(counts, iu, 1)
            np.add.at(counts, ju, 1)
        return counts

    def forces(self) -> np.ndarray:
        iu, ju = self.pair_contacts()
        force, _torque = contact_forces(
            self.pos, self.vel, self.omega, iu, ju, self.state, self.mat, self.dt,
        )
        r = self.cfg.radius
        gw = self.mat.gamma_wall
        m = self.mat.mass

        def wall(mask: np.ndarray, overlap: np.ndarray, normal: np.ndarray) -> None:
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                return
            vn = (self.vel[idx] * normal).sum(axis=1)
            fn = np.maximum(self.cfg.kn * overlap[idx] - gw * vn, 0.0)
            force[idx] += fn[:, None] * normal

        # floor (normal +z), lid (normal -z), four rigid side walls
        wall(self.pos[:, 2] < r, r - self.pos[:, 2], np.array([0.0, 0.0, 1.0]))
        wall(self.pos[:, 2] > self.lid_z - r, r - (self.lid_z - self.pos[:, 2]),
             np.array([0.0, 0.0, -1.0]))
        wall(self.pos[:, 0] < r, r - self.pos[:, 0], np.array([1.0, 0.0, 0.0]))
        wall(self.pos[:, 0] > self.width - r, r - (self.width - self.pos[:, 0]),
             np.array([-1.0, 0.0, 0.0]))
        wall(self.pos[:, 1] < r, r - self.pos[:, 1], np.array([0.0, 1.0, 0.0]))
        wall(self.pos[:, 1] > self.width - r, r - (self.width - self.pos[:, 1]),
             np.array([0.0, -1.0, 0.0]))
        if self.gravity_on:
            force[:, 2] -= m * self.cfg.gravity
        return force

    def lid_force(self) -> float:
        """Total upward (-z) force the particles exert on the piston (N)."""
        r = self.cfg.radius
        mask = self.pos[:, 2] > self.lid_z - r
        if not np.any(mask):
            return 0.0
        overlap = r - (self.lid_z - self.pos[mask, 2])
        vn = -self.vel[mask, 2]
        fn = np.maximum(self.cfg.kn * overlap - self.mat.gamma_wall * vn, 0.0)
        return float(np.sum(fn))

    def lid_pressure(self) -> float:
        return self.lid_force() / self.area

    def max_speed(self) -> float:
        return float(np.max(np.linalg.norm(self.vel, axis=1)))

    # -- integration -----------------------------------------------------
    def step(self) -> None:
        force = self.forces()
        self.vel += force / self.mass[:, None] * self.dt
        self.vel *= 1.0 / (1.0 + self.cfg.bulk_damping_rate * self.dt)
        self.pos += self.vel * self.dt
        # piston: rate-limited dead-weight servo.  It descends at a fixed
        # quasi-static speed while the lid force is below the preload and stops
        # inside the dead band; no inertial overshoot is possible.
        error = self.preload - self.lid_force()
        dead = self.cfg.stress_tolerance * self.preload if self.preload > 0.0 else 0.0
        if self._manual_lid:
            pass  # velocity set by the caller (displacement control)
        elif abs(error) <= dead:
            self.lid_velocity = 0.0
        else:
            # roughing speed far from equilibrium, quasi-static speed in the
            # last 20 % of the preload so the final approach cannot overshoot
            fast = abs(error) > 0.2 * self.preload
            speed = self.cfg.lid_speed_fast if fast else self.cfg.lid_velocity_limit
            self.lid_velocity = speed * math.copysign(1.0, error)
        self.lid_z -= self.lid_velocity * self.dt
        self.steps += 1
        # leave the cell: report, do not silently explode
        if not np.all(np.isfinite(self.pos)):
            raise RuntimeError("DEM oedometer diverged (non-finite positions)")
        # contact-velocity scale: 10 % of the grain's own elastic velocity
        v_scale = 0.5 * self.cfg.radius * math.sqrt(self.cfg.kn / self.mat.mass)
        if self.max_speed() > v_scale:
            raise RuntimeError("DEM oedometer diverged (contact instability)")

    def compact_to_porosity(self, target: float, *, max_steps: int = 400_000) -> dict:
        """Displacement-control pre-consolidation to a target porosity.

        A reproducible *starting* state is what makes the two stress increments
        comparable across particle counts: without it the roughing history (and
        therefore the coordination number) would differ between runs.
        """
        self.preload = 0.0
        start = self.cell_height
        self._manual_lid = True
        self.lid_velocity = self.cfg.compaction_speed
        steps = 0
        try:
            for _ in range(int(max_steps)):
                if self.porosity() <= target:
                    return {"steps": steps, "porosity": self.porosity(),
                            "travel": start - self.cell_height, "reached": True}
                self.step()
                steps += 1
        finally:
            self.lid_velocity = 0.0
            self._manual_lid = False
        return {"steps": steps, "porosity": self.porosity(),
                "travel": start - self.cell_height, "reached": False}

    def unjam(self, steps: int) -> dict:
        """Gravity-free relaxation pass that removes deposition overlaps."""
        for _ in range(int(steps)):
            self.step()
        return {"steps": int(steps), "max_speed": self.max_speed()}

    def relax(self, *, max_steps: int | None = None, speed_tol: float | None = None) -> dict:
        cfg = self.cfg
        max_steps = int(max_steps if max_steps is not None else cfg.max_steps_per_stage)
        speed_tol = float(speed_tol if speed_tol is not None else cfg.settle_speed_tol)
        for _ in range(max_steps):
            self.step()
            if self.steps % cfg.check_every == 0 and self.max_speed() < speed_tol:
                return {"steps": self.steps, "converged": True, "max_speed": self.max_speed()}
        return {"steps": self.steps, "converged": False, "max_speed": self.max_speed()}

    def load_to(self, stress: float) -> dict:
        """Apply a dead-weight piston stress and relax to static equilibrium."""
        cfg = self.cfg
        self.preload = float(stress) * self.area
        holds = 0
        start_steps = self.steps
        for _ in range(int(cfg.max_steps_per_stage)):
            self.step()
            if self.steps % cfg.check_every == 0:
                p = self.lid_pressure()
                slow = self.max_speed() < cfg.settle_speed_tol and abs(self.lid_velocity) < 1e-4
                balanced = abs(p - stress) <= cfg.stress_tolerance * stress
                holds = holds + 1 if (slow and balanced) else 0
                if holds >= cfg.hold_checks:
                    return {
                        "target_stress": float(stress),
                        "lid_pressure": p,
                        "reached": True,
                        "steps": self.steps - start_steps,
                        "max_speed": self.max_speed(),
                        "height": self.cell_height,
                        "porosity": self.porosity(),
                    }
        return {
            "target_stress": float(stress),
            "lid_pressure": self.lid_pressure(),
            "reached": False,
            "steps": self.steps - start_steps,
            "max_speed": self.max_speed(),
            "height": self.cell_height,
            "porosity": self.porosity(),
        }

    def coordination(self) -> float:
        return float(np.mean(self.contact_count()))


def carman_kozeny_conductivity(porosity: float, grain_diameter: float,
                               *, gamma_w: float = WATER_UNIT_WEIGHT,
                               mu_w: float = WATER_VISCOSITY) -> float:
    """Hydraulic conductivity ``K`` (m/s) from the Carman-Kozeny closure.

    ``k_int = n^3 d^2 / (180 (1-n)^2)`` (intrinsic permeability, m^2) for
    monodisperse spheres, ``K = k_int gamma_w / mu_w``.
    """
    if not 0.0 < porosity < 1.0:
        raise ValueError("porosity must lie in (0, 1)")
    k_int = porosity ** 3 * grain_diameter ** 2 / (180.0 * (1.0 - porosity) ** 2)
    return k_int * gamma_w / mu_w


def pore_reynolds(conductivity: float, gradient: float, grain_diameter: float,
                  *, rho_w: float = WATER_DENSITY,
                  mu_w: float = WATER_VISCOSITY) -> float:
    """``Re_p = rho_w v_s d / mu_w`` with ``v_s = K i`` (Darcy superficial)."""
    return rho_w * abs(conductivity * gradient) * grain_diameter / mu_w


def run_oedometer(
    config: OedometerConfig | None = None,
    *,
    stress_levels: Iterable[float] | None = None,
    progress=None,
) -> dict:
    """Full two-increment oedometer test; returns the measured properties."""
    cfg = config or OedometerConfig()
    levels = tuple(stress_levels) if stress_levels is not None else cfg.stress_levels
    if len(levels) != 2:
        raise ValueError("two stress increments are required")
    cell = OedometerCell(cfg)
    if progress:
        progress(f"DEM oedometer: {cell.n} grains, d={2*cfg.radius:.3g} m, dt={cell.dt:.3g} s")
    unjam = cell.unjam(cfg.unjam_steps)
    cell.gravity_on = True
    compaction = (
        cell.compact_to_porosity(cfg.compaction_porosity)
        if cfg.compaction_porosity is not None
        else {"note": "no pre-consolidation: the first load stage consolidates the bed"}
    )
    if progress:
        progress(f"  de-overlap pass: {unjam['steps']} steps, max speed {unjam['max_speed']:.3g} m/s, "
                 f"n={cell.porosity():.4f}, Z={cell.coordination():.2f}")
        if "porosity" in compaction:
            progress(f"  pre-consolidation: n={compaction['porosity']:.4f} "
                     f"travel={compaction['travel']*1e3:.4f} mm reached={compaction['reached']}")
    stages = []
    stage_reports = []
    for level in levels:
        report = cell.load_to(level)
        stages.append({
            "stress": float(level),
            "height": cell.cell_height,
            "porosity": cell.porosity(),
            "coordination": cell.coordination(),
            "reached": report["reached"],
            "lid_pressure": report["lid_pressure"],
        })
        stage_reports.append(report)
        if progress:
            progress(
                f"  stress {level/1e3:.1f} kPa -> p={report['lid_pressure']/1e3:.2f} kPa, "
                f"h={cell.cell_height*1e3:.4f} mm, n={stages[-1]['porosity']:.4f}, "
                f"Z={stages[-1]['coordination']:.2f}, reached={report['reached']}"
            )

    p1, p2 = stages[0]["stress"], stages[1]["stress"]
    h1, h2 = stages[0]["height"], stages[1]["height"]
    eps = (h1 - h2) / h1
    d_sigma = p2 - p1
    E_oed = d_sigma / eps if eps > 0.0 else float("inf")
    eps_ref = eps  # reference strain for the alt (mean-height) modulus
    E_oed_mid = d_sigma / ((h1 - h2) / (0.5 * (h1 + h2))) if eps > 0.0 else float("inf")

    n_mean = 0.5 * (stages[0]["porosity"] + stages[1]["porosity"])
    diameter = 2.0 * cfg.radius
    K = carman_kozeny_conductivity(n_mean, diameter)
    k_int = K * WATER_VISCOSITY / WATER_UNIT_WEIGHT
    h_dr = 0.5 * h1  # two-way drainage of the DEM cell height
    gradient = d_sigma / (WATER_UNIT_WEIGHT * h_dr)
    Re_p = pore_reynolds(K, gradient, diameter)

    return {
        "config": {
            "radius": cfg.radius,
            "diameter": diameter,
            "density": cfg.density,
            "n_row": cfg.n_row,
            "n_layer": cfg.n_layer,
            "kn": cfg.kn,
            "friction": cfg.friction,
            "mu_rolling": cfg.mu_rolling,
            "restitution": cfg.restitution,
            "gravity": cfg.gravity,
            "seed": cfg.seed,
            "dt": cell.dt,
            "particle_count": cell.n,
            "box_width": cell.width,
            "stress_levels": list(levels),
        },
        "unjam": unjam,
        "pre_consolidation": compaction,
        "settle": {"note": "bed compacted from the deposited packing directly to the first load stage"},
        "stages": stages,
        "stage_reports": stage_reports,
        "total_steps": cell.steps,
        "E_oed": E_oed,
        "E_oed_mid": E_oed_mid,
        "strain_increment": eps_ref,
        "porosity": n_mean,
        "coordination": 0.5 * (stages[0]["coordination"] + stages[1]["coordination"]),
        "k_int": k_int,
        "k_hyd": K,
        "gradient": gradient,
        "Re_p": Re_p,
        "drainage_height": h_dr,
        "saturated": True,
    }