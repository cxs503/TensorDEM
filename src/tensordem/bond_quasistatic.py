"""Quasi-static single-bond / bond-chain tension driver for the rate-dependent bond.

Why this module exists
----------------------
The specimen campaign of mission m6464 (``scripts/verify_ice_bond_specimen_rate.py``,
``docs/ice_bond_specimen_rate_dependence.md``) wired the Maxwell-Wiechert bond
and the Kachanov creep clock into ``IceDEM3D`` and measured a fitted strength
exponent of 0.244 at 0.25-4 1/s against the analytic 1/(m+1) = 1/3 -- but the
*negative control* (pure linear bond) was not flat either (slope +0.117) and the
strength came out 7-9x the rigid-plate bound, so the specimen was inertia
dominated and the exponent was recorded as an honest FAIL.  The gap ledger names
the next node explicitly: *isolate the constitutive response with a single-bond
or single-chain driver*.

This module is that driver.  A bonded chain of particles is loaded in **tension**
by a *prescribed displacement* (kinematic boundary condition on the top
particle), not by force or platen servo:

* ``n_particles = 2`` -> one bond and **no free particle at all**; the bond
  extension history is exactly ``eps(t) = edot * t`` and inertia cannot enter.
  What is being exercised is therefore the engine's bond call site
  (``IceDEM3D._bond_force`` + ``_commit_fracture`` + ``_commit_creep_damage``),
  not just the constitutive helpers.
* ``n_particles > 2`` -> interior particles are free, so the chain redistributes
  the imposed displacement and inertia *does* enter; the driver measures the
  inertia index ``KE_free / SE`` and the strain non-uniformity so the deviation
  from the quasi-static closed form can be attributed rather than guessed.
  Scaling ``density`` down (standard DEM mass scaling) increases the chain's
  wave speed, i.e. it is the quasi-static servo dial.

Tension (rather than compression) is deliberate: separating spheres carry the
load through the bond law alone, so the particle-particle contact branch (which
would add ``k_contact * overlap`` in parallel with the bond) provably cannot
contaminate the measurement.

Closed-form references (no free parameters)
-------------------------------------------
With a linear elastic bond ``F = k_u eps L0`` and the Kachanov clock
``dD/dt = (|F|/F_ref)**m / t_ref``, ``F_ref = k_u eps_b L0`` and a constant
strain rate::

    D(t) = (edot/eps_b)**m * t**(m+1) / ((m+1) * t_ref)  = 1  at failure

so the failure strain is exactly::

    eps_f / eps_b = [ (m+1) t_ref edot / eps_b ] ** (1/(m+1))

and the ductile branch exists while ``eps_f < eps_b``, i.e. for
``edot < eps_b / ((m+1) t_ref)``; above that the brittle strain criterion wins
and the strength is rate independent.

For the Maxwell-Wiechert bond the uniaxial force under a strain ramp is also
closed form (Prony relaxation, exact per-branch advance)::

    F(t) = edot L0 [ k_e t + sum_a k_a tau_a (1 - exp(-t/tau_a)) ]

which is integrated here to high resolution for the damage reference.  The
reference is therefore independent of the engine's stepping.

Usage
-----
    cfg = QuasiStaticBondConfig(bond_creep_exponent=2, bond_creep_time_s=5e-3)
    run = ChainTensionDriver(cfg).run(strain_rate=0.1)
    ref = elastic_creep_reference(cfg, strain_rate=0.1)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .dem3d import IceDEM3D, DEM3DConfig
from .ice_bond_viscoelastic import MaxwellWiechertBond

__all__ = [
    "QuasiStaticBondConfig",
    "ChainTensionDriver",
    "bond_count",
    "gauge_length",
    "brittle_force_scale",
    "ductile_failure_strain",
    "crossover_strain_rate",
    "maxwell_wiechert_force",
    "elastic_creep_reference",
    "maxwell_wiechert_reference",
    "affine_network_factor",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuasiStaticBondConfig:
    """Chain geometry, bond law and numerics for one quasi-static pull test.

    ``dt = None`` uses the engine's conservative ``recommended_dt``; a positive
    value is passed straight through (the engine still enforces its stability
    bound).  ``sample_every`` decimates the recorded histories only -- the
    integration always runs at ``dt``.
    """

    # Geometry: a 2 x 2 bonded column lattice (nx = ny = 2, nz = n_layers).
    # ``n_layers = 2`` means the bottom layer is a hard support and the top
    # layer is driven kinematically -> *no free particle at all*, so the only
    # constitutive response left is the bond law + damage clock.
    n_layers: int = 2
    radius: float = 0.025
    density: float = 917.0
    bond_stiffness: float = 2_000.0
    contact_stiffness: float = 5_000.0
    breaking_strain: float = 0.015
    shear_breaking_strain: float = 0.03
    contact_damping: float = 5.0
    drag: float = 2.0
    # --- bond constitutive model (mirrors DEM3DConfig) ---
    bond_model: str = "linear"                     # "linear" | "maxwell_wiechert"
    bond_equilibrium_fraction: float = 0.35
    bond_delayed_elastic_fraction: tuple[float, ...] = (0.45, 0.20)
    bond_relaxation_time_s: tuple[float, ...] = (5.0e-3, 5.0e-2)
    bond_creep_exponent: int = 0                   # m; 0 = brittle only
    bond_creep_time_s: float = 5.0e-3              # t_ref
    bond_creep_force_fraction: float = 1.0         # F_ref / (k_u eps_b L0)
    # --- numerics ---
    dt: float | None = None
    max_strain: float = 0.06                       # 4 * breaking_strain
    sample_every: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.n_layers, bool) or not isinstance(self.n_layers, int):
            raise TypeError("n_layers must be an int")
        if self.n_layers < 2:
            raise ValueError("n_layers must be >= 2 (a bonded specimen needs one interface)")
        for name in ("radius", "density", "bond_stiffness", "contact_stiffness",
                     "breaking_strain", "shear_breaking_strain", "drag",
                     "contact_damping", "max_strain"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if self.bond_model not in ("linear", "maxwell_wiechert"):
            raise ValueError("bond_model must be 'linear' or 'maxwell_wiechert'")
        if isinstance(self.bond_creep_exponent, bool) or not isinstance(
                self.bond_creep_exponent, int) or self.bond_creep_exponent < 0:
            raise ValueError("bond_creep_exponent must be a nonnegative integer")
        if not (math.isfinite(self.bond_creep_time_s) and self.bond_creep_time_s > 0.0):
            raise ValueError("bond_creep_time_s must be finite and positive")
        if not (math.isfinite(self.bond_creep_force_fraction)
                and self.bond_creep_force_fraction > 0.0):
            raise ValueError("bond_creep_force_fraction must be finite and positive")
        if self.bond_model == "maxwell_wiechert" and not self.bond_relaxation_time_s:
            raise ValueError("bond_model='maxwell_wiechert' needs at least one relaxation time")
        if self.dt is not None and not (math.isfinite(self.dt) and self.dt > 0.0):
            raise ValueError("dt must be finite and positive when given")
        if isinstance(self.sample_every, bool) or not isinstance(self.sample_every, int) \
                or self.sample_every < 1:
            raise ValueError("sample_every must be a positive int")

    @property
    def n_particles(self) -> int:
        """Total particles: ``4 * n_layers`` (2 x 2 columns)."""
        return 4 * self.n_layers

    @property
    def n_columns(self) -> int:
        """Particles per layer (the load path is this many parallel bonds)."""
        return 4

    # -- derived geometry / law helpers -----------------------------------
    @property
    def bond_rest_length(self) -> float:
        """L0 = 2 r (simple-cubic spacing of the chain)."""
        return 2.0 * self.radius

    @property
    def bond_law(self) -> MaxwellWiechertBond | None:
        if self.bond_model != "maxwell_wiechert":
            return None
        return MaxwellWiechertBond.from_unrelaxed_stiffness(
            self.bond_stiffness,
            equilibrium_fraction=self.bond_equilibrium_fraction,
            delayed_elastic_fraction=self.bond_delayed_elastic_fraction,
            relaxation_time=self.bond_relaxation_time_s,
        )


def bond_count(cfg: QuasiStaticBondConfig) -> int:
    """Bonds in the axial load path: one per column per layer interface."""
    return cfg.n_columns * (cfg.n_layers - 1)


def gauge_length(cfg: QuasiStaticBondConfig) -> float:
    """Total bonded length (m); the imposed displacement is divided by this."""
    return (cfg.n_layers - 1) * cfg.bond_rest_length


def brittle_force_scale(cfg: QuasiStaticBondConfig) -> float:
    """Force at which the brittle strain criterion fires: ``k_u eps_b L0`` (N)."""
    law = cfg.bond_law
    k_u = cfg.bond_stiffness if law is None else float(law.unrelaxed_stiffness)
    return k_u * cfg.breaking_strain * cfg.bond_rest_length


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------
def ductile_failure_strain(
    strain_rate: float,
    *,
    creep_time_s: float,
    exponent: int,
    breaking_strain: float,
) -> float:
    """Exact Kachanov failure strain for a linear spring under a strain ramp.

    ``eps_f / eps_b = [(m+1) t_ref edot / eps_b]**(1/(m+1))``.  Returns
    ``breaking_strain`` (brittle branch) when the ductile root exceeds it, so the
    caller always gets ``min(ductile, brittle)``.
    """
    if not (math.isfinite(strain_rate) and strain_rate > 0.0):
        raise ValueError("strain_rate must be finite and positive")
    if exponent <= 0:
        return breaking_strain
    if not (math.isfinite(creep_time_s) and creep_time_s > 0.0):
        raise ValueError("creep_time_s must be finite and positive")
    if not (math.isfinite(breaking_strain) and breaking_strain > 0.0):
        raise ValueError("breaking_strain must be finite and positive")
    ratio = ((exponent + 1.0) * creep_time_s * strain_rate / breaking_strain) ** (
        1.0 / (exponent + 1.0))
    return min(breaking_strain, breaking_strain * ratio)


def crossover_strain_rate(
    *,
    creep_time_s: float,
    exponent: int,
    breaking_strain: float,
) -> float:
    """Strain rate at which ductile creep and brittle cleavage coincide.

    ``edot_c = eps_b / ((m+1) t_ref)``: a pure *time* comparison (the damage
    clock runs at ``1/t_ref`` only at ``F = F_ref``), not a tuned stress.
    """
    if exponent <= 0:
        return math.inf
    return breaking_strain / ((exponent + 1.0) * creep_time_s)


def _damage_rate(force: float, reference_force: float, creep_time_s: float,
                 exponent: float) -> float:
    return (abs(force) / reference_force) ** exponent / creep_time_s


def _composite_simpson(f, a: float, b: float, n: int) -> float:
    """Composite Simpson integral of ``f`` on ``[a, b]`` with ``n`` intervals."""
    if n % 2:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4.0 if i % 2 else 2.0) * f(a + i * h)
    return total * h / 3.0


def maxwell_wiechert_force(strain_rate: float, time: float,
                           bond_rest_length: float,
                           law: MaxwellWiechertBond) -> float:
    """Exact uniaxial bond force under ``eps(t) = strain_rate * t`` (N).

    ``F = edot L0 [k_e t + sum_a k_a tau_a (1 - exp(-t/tau_a))]`` -- the Prony
    branches integrated in closed form, i.e. the same trajectory
    ``advance_branches`` produces exactly, written without stepping.
    """
    if time < 0.0:
        raise ValueError("time must be nonnegative")
    total = float(law.equilibrium_stiffness) * time
    for k_a, tau_a in zip(law.branch_stiffness, law.relaxation_time):
        tau = float(tau_a)
        total += float(k_a) * tau * (1.0 - math.exp(-time / tau))
    return strain_rate * bond_rest_length * total


def elastic_creep_reference(cfg: QuasiStaticBondConfig, strain_rate: float,
                            *, integral_samples: int = 40001) -> dict[str, float]:
    """Quasi-static reference for a linear elastic bond + Kachanov clock.

    ``integral_samples`` is only used for the numerical cross-check of the
    closed form (``damage_at_failure``); the reported strain is analytic.
    """
    eps_f = ductile_failure_strain(
        strain_rate,
        creep_time_s=cfg.bond_creep_time_s,
        exponent=cfg.bond_creep_exponent,
        breaking_strain=cfg.breaking_strain,
    )
    t_f = eps_f / strain_rate
    reference_force = (cfg.bond_creep_force_fraction * cfg.bond_stiffness
                       * cfg.breaking_strain * cfg.bond_rest_length)
    # Independent numerical damage integral at the analytic failure time: must
    # be 1 for the closed form to be self-consistent (no engine involved).
    damage = 0.0
    if cfg.bond_creep_exponent > 0 and t_f > 0.0:
        m = float(cfg.bond_creep_exponent)

        def rate(t: float) -> float:
            force = cfg.bond_stiffness * strain_rate * cfg.bond_rest_length * t
            return _damage_rate(force, reference_force, cfg.bond_creep_time_s, m)

        damage = _composite_simpson(rate, 0.0, t_f, integral_samples)
    return {
        "strain_rate": strain_rate,
        "family": "linear+creep",
        "failure_strain": eps_f,
        "failure_time_s": t_f,
        "failure_force_N": (cfg.bond_stiffness * eps_f * cfg.bond_rest_length),
        "brittle_strain_cap": cfg.breaking_strain,
        "regime": "ductile" if eps_f < cfg.breaking_strain else "brittle",
        "damage_at_failure": damage,
    }


def maxwell_wiechert_reference(cfg: QuasiStaticBondConfig, strain_rate: float,
                               *, samples_per_decade: int = 4000,
                               max_iterations: int = 60) -> dict[str, float]:
    """Quasi-static reference for the Maxwell-Wiechert bond + Kachanov clock.

    The force history is the closed form above; the damage integral
    ``D(t) = int_0^t (F/F_ref)**m dt / t_ref`` is evaluated by composite Simpson
    on a grid refined until the bisection bracket is machine-tight, so the
    reference is independent of the engine's time stepping.  The brittle cap is
    applied afterwards, exactly as ``_commit_fracture`` does.
    """
    law = cfg.bond_law
    if law is None:
        raise ValueError("maxwell_wiechert_reference needs bond_model='maxwell_wiechert'")
    L0 = cfg.bond_rest_length
    reference_force = (cfg.bond_creep_force_fraction * float(law.unrelaxed_stiffness)
                       * cfg.breaking_strain * L0)

    def force(t: float) -> float:
        return maxwell_wiechert_force(strain_rate, t, L0, law)

    def damage(t: float) -> float:
        if t <= 0.0:
            return 0.0
        if cfg.bond_creep_exponent <= 0:
            return 0.0
        m = float(cfg.bond_creep_exponent)
        n = max(64, int(samples_per_decade * math.log10(max(t, 1e-12) / 1e-9)))
        n = min(n, 200_000)
        return _composite_simpson(
            lambda s: _damage_rate(force(s), reference_force, cfg.bond_creep_time_s, m),
            0.0, t, n)

    # Upper bracket: the brittle cleavage time, or a generous multiple of t_ref.
    t_brittle = cfg.breaking_strain / strain_rate
    lo, hi = 0.0, t_brittle
    regime = "brittle"
    if cfg.bond_creep_exponent > 0 and damage(t_brittle) >= 1.0:
        regime = "ductile"
        lo, hi = 0.0, t_brittle
        for _ in range(max_iterations):
            mid = 0.5 * (lo + hi)
            if damage(mid) >= 1.0:
                hi = mid
            else:
                lo = mid
        t_f = hi
    else:
        t_f = t_brittle
    eps_f = strain_rate * t_f
    return {
        "strain_rate": strain_rate,
        "family": "maxwell_wiechert+creep",
        "failure_strain": eps_f,
        "failure_time_s": t_f,
        "failure_force_N": force(t_f),
        "brittle_strain_cap": cfg.breaking_strain,
        "regime": regime,
        "damage_at_failure": damage(t_f) if cfg.bond_creep_exponent > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Engine driver
# ---------------------------------------------------------------------------
def affine_network_factor(cfg: QuasiStaticBondConfig) -> dict[str, float]:
    """Analytic load amplification of the 2 x 2 column lattice (pure geometry).

    Under an affine axial stretch ``u_z = eps z`` the strain of a bond with
    rest vector ``v`` is ``eps cos^2(theta)`` with ``cos(theta) = (v . z)/|v|``,
    so the z force on the supported layer is ``sum_b k_u |v_b| eps cos^3(theta)``.
    For the 2 x 2 lattice with fully connected neighbouring layers this gives
    ``4`` axial + ``8`` face-diagonal + ``4`` body-diagonal bonds, i.e. a net
    amplification per axial bond force of ``4 + 8*(1/2) + 4*(1/3) = 28/3``.
    Returned as the *ratio to a single axial bond force* (dimensionless).
    """
    L = cfg.bond_rest_length
    axial = cfg.n_columns * (cfg.n_layers - 1) * 1.0
    per_interface_face = 8 * 0.5
    per_interface_body = 4 * (1.0 / 3.0)
    factor = axial + (cfg.n_layers - 1) * (per_interface_face + per_interface_body)
    # Axial bond force itself is k_u * L * eps, so the ratio is dimensionless.
    return {
        "net_per_axial_bond": factor,
        "n_axial": float(axial),
        "n_face_diagonal": float((cfg.n_layers - 1) * 8),
        "n_body_diagonal": float((cfg.n_layers - 1) * 4),
        "axial_rest_length_m": L,
    }


class ChainTensionDriver:
    """Prescribed-displacement tension test on a bonded column inside ``IceDEM3D``.

    Geometry is the smallest lattice ``DEM3DConfig`` allows (``nx = ny = 2``,
    ``nz = n_layers``).  The bottom layer is a hard support and the top layer is
    driven kinematically (position and velocity written after every ``step``).
    With ``n_layers = 2`` there is **no free particle**, so the response is the
    bond network's constitutive answer with zero inertia; with ``n_layers > 2``
    the interior layers are free and inertia enters, which is what the
    ``inertia_index`` / ``max_nonuniformity`` fields measure.

    Tension (not compression) is used so the particle-particle contact branch --
    which would add ``k_contact * overlap`` in parallel with the bond on the same
    pair -- provably cannot contribute: separating spheres have zero overlap.

    Loads are read from an explicit bond-force probe (``IceDEM3D._bond_force``
    documents ``update_history=False`` as probe safe), summed over the bonds
    incident to the supported layer with their ``z`` direction cosine.  The
    support layer is immobile, so no drag term can leak into the reading.

    A driver is single-use state; ``run`` rebuilds the solver so repeated calls
    (rate sweeps, dt convergence) start from a virgin lattice.
    """

    def __init__(self, cfg: QuasiStaticBondConfig) -> None:
        self.cfg = cfg
        self.gauge = gauge_length(cfg)
        # Same discipline as TriaxialRunner: tiny lattices are pure overhead in
        # the intra-op thread pool, and one thread keeps runs bit-reproducible.
        torch.set_num_threads(1)
        self._base = DEM3DConfig(
            nx=2,
            ny=2,
            nz=cfg.n_layers,
            radius=cfg.radius,
            density=cfg.density,
            bond_stiffness=cfg.bond_stiffness,
            contact_stiffness=cfg.contact_stiffness,
            breaking_strain=cfg.breaking_strain,
            shear_breaking_strain=cfg.shear_breaking_strain,
            contact_damping=cfg.contact_damping,
            drag=cfg.drag,
            friction_coefficient=0.0,
            rolling_resistance_coefficient=0.0,
            # nx = 2 means fix_x_edges would pin *every* particle, so the
            # specimen is held only at its bottom layer.
            fix_x_edges=False,
            fix_bottom=True,
            geometry="box",
            bond_model=cfg.bond_model,
            bond_equilibrium_fraction=cfg.bond_equilibrium_fraction,
            bond_delayed_elastic_fraction=cfg.bond_delayed_elastic_fraction,
            bond_relaxation_time_s=cfg.bond_relaxation_time_s,
            bond_creep_exponent=cfg.bond_creep_exponent,
            bond_creep_time_s=cfg.bond_creep_time_s,
            bond_creep_force_fraction=cfg.bond_creep_force_fraction,
        )
        self.sim = self._build_sim()

    # -- construction ------------------------------------------------------
    def _build_sim(self) -> IceDEM3D:
        cfg = self.cfg
        dt = self._base.recommended_dt if cfg.dt is None else cfg.dt
        sim = IceDEM3D(replace(self._base, dt=dt))
        # Isolate the bond: park the spherical indenter far away and freeze it.
        sim.tool_start[:] = torch.tensor([1.0e6, 1.0e6, 1.0e6], dtype=sim.dtype)
        sim.tool_velocity.zero_()
        z = sim.initial_positions[:, 2]
        z_top, z_bottom = float(z.max()), float(z.min())
        self.top = torch.nonzero(z >= z_top - 1e-12, as_tuple=False).flatten()
        self.bottom = torch.nonzero(z <= z_bottom + 1e-12, as_tuple=False).flatten()
        self.drive_z0 = z_top
        self._top_xy = sim.initial_positions[self.top, :2].clone()
        i, j = sim.pairs.T
        in_bottom = torch.zeros(len(sim.positions), dtype=torch.bool)
        in_bottom[self.bottom] = True
        self._bond_sign = (in_bottom[i].to(sim.dtype) - in_bottom[j].to(sim.dtype))
        self._bottom_bonds = self._bond_sign != 0
        delta = sim.initial_positions[j] - sim.initial_positions[i]
        cosine_z = delta[:, 2] / torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-30)
        # Bonds parallel to the loading axis: the load path whose constitutive
        # answer the closed form predicts.  The oblique bonds carry a reduced
        # share (cos^2 of the angle) and the within-layer bonds carry nothing.
        self._axial_bonds = cosine_z.abs() > 1.0 - 1e-12
        self.sim = sim
        return sim

    # -- state helpers -----------------------------------------------------
    def _axial_extensions(self) -> torch.Tensor:
        i, j = self.sim.pairs.T
        delta = self.sim.positions[j] - self.sim.positions[i]
        return torch.linalg.vector_norm(delta, dim=1) - self.sim.rest_lengths

    def _load(self) -> tuple[float, float]:
        """``(net z force on the supported layer, mean axial-bond force)`` in N."""
        sim = self.sim
        extension = self._axial_extensions()
        magnitude = sim._bond_force(extension, update_history=False)   # probe safe
        i, j = sim.pairs.T
        delta = sim.positions[j] - sim.positions[i]
        distance = torch.linalg.vector_norm(delta, dim=1)
        cosine_z = (delta[:, 2] / distance.clamp_min(torch.finfo(sim.dtype).eps))
        contribution = (magnitude * cosine_z * self._bond_sign)[self._bottom_bonds]
        # Deliberately *not* masked by ``alive``: the force a bond carried at
        # rupture is the quantity of interest and ``_bond_force`` returns the
        # constitutive value for a deleted bond as well.
        axial = magnitude[self._axial_bonds]
        axial_force = float(axial.mean()) if axial.numel() else 0.0
        return float(contribution.sum()), axial_force

    def _inertia(self) -> tuple[float, float, float, float]:
        """``(KE_free, SE_bonds, strain_nonuniformity, bond_force_spread)``."""
        sim = self.sim
        live = sim.velocities.clone()
        live[self.bottom] = 0.0
        live[self.top] = 0.0
        kinetic = 0.5 * sim.config.mass * float(live.square().sum())
        extension = self._axial_extensions()
        alive = sim.alive
        if not bool(alive.any()):
            return kinetic, 0.0, 0.0, 0.0
        stored = float(sim.bond_stored_energy(extension)[alive].sum())
        path = self._axial_bonds & alive
        strain = (extension / sim.rest_lengths)[path]
        mean = float(strain.mean()) if strain.numel() else 0.0
        nonuniform = ((float(strain.max()) - float(strain.min())) / abs(mean)
                      if mean else 0.0)
        magnitude = sim._bond_force(extension, update_history=False).abs()[path]
        mean_force = float(magnitude.mean())
        spread = ((float(magnitude.max()) - float(magnitude.min())) / mean_force
                  if mean_force > 0.0 else 0.0)
        return kinetic, stored, nonuniform, spread

    # -- run ---------------------------------------------------------------
    def run(self, strain_rate: float, *, max_strain: float | None = None,
            stop_at_strain: float | None = None) -> dict:
        """Pull the column at ``strain_rate`` (bond strain 1/s) until first failure.

        ``stop_at_strain`` halts the ramp before failure (damage-clock studies):
        the run then reports ``failed = False`` and the bond damage reached.
        """
        if not (math.isfinite(strain_rate) and strain_rate > 0.0):
            raise ValueError("strain_rate must be finite and positive")
        cfg = self.cfg
        limit = cfg.max_strain if max_strain is None else float(max_strain)
        if not (math.isfinite(limit) and limit > cfg.breaking_strain):
            raise ValueError("max_strain must exceed breaking_strain")
        sim = self._build_sim()
        dt = sim.dt
        speed = strain_rate * self.gauge
        max_steps = int(math.ceil(limit * self.gauge / speed / dt)) + 2

        strains: list[float] = []
        loads: list[float] = []
        times: list[float] = []
        peak_force = 0.0
        peak_strain = 0.0
        axial_force_at_peak = 0.0
        max_kinetic = 0.0
        max_stored = 0.0
        max_nonuniform = 0.0
        max_spread = 0.0
        failed = False
        stopped_early = False
        step = 0
        while step < max_steps:
            extension = self._axial_extensions()
            mean_strain = float((extension / sim.rest_lengths)[self._axial_bonds].mean())
            load, axial_force = self._load()
            if abs(load) > abs(peak_force):
                peak_force = load
                peak_strain = mean_strain
                axial_force_at_peak = axial_force
            if step % cfg.sample_every == 0:
                kinetic, stored, nonuniform, spread = self._inertia()
                max_kinetic = max(max_kinetic, kinetic)
                max_stored = max(max_stored, stored)
                max_nonuniform = max(max_nonuniform, nonuniform)
                max_spread = max(max_spread, spread)
                strains.append(mean_strain)
                loads.append(load)
                times.append(float(sim.time))
            if not bool(sim.alive.all()):
                failed = True
                break
            if stop_at_strain is not None and mean_strain >= stop_at_strain:
                stopped_early = True
                break
            sim.step()
            # Kinematic boundary condition: overwrite the driven layer after the
            # solver has advanced everything else.
            sim.positions[self.top, 2] = self.drive_z0 + speed * sim.time
            sim.positions[self.top, :2] = self._top_xy
            sim.velocities[self.top] = 0.0
            sim.velocities[self.top, 2] = speed
            step += 1

        dead = torch.nonzero(~sim.alive, as_tuple=False).flatten()
        energy = sim.energy()
        record: dict = {
            "strain_rate": strain_rate,
            "n_layers": cfg.n_layers,
            "n_particles": cfg.n_particles,
            "n_bonds_total": int(len(sim.pairs)),
            "n_load_path_bonds": bond_count(cfg),
            "dt": dt,
            "steps": step,
            "gauge_length_m": self.gauge,
            "bond_rest_length_m": cfg.bond_rest_length,
            "bond_model": cfg.bond_model,
            "creep_exponent": cfg.bond_creep_exponent,
            "creep_time_s": cfg.bond_creep_time_s,
            "breaking_strain": cfg.breaking_strain,
            "steady_speed_m_s": speed,
            "failed": failed,
            "stopped_early": stopped_early,
            "final_strain": strains[-1] if strains else 0.0,
            "final_max_bond_damage": float(sim.bond_damage.max()) if sim.bond_damage.numel() else 0.0,
            "peak_force_N": peak_force,
            "axial_bond_force_N": axial_force_at_peak,
            "peak_strain": peak_strain,
            "max_kinetic_free_J": max_kinetic,
            "max_bond_energy_J": max_stored,
            "inertia_index": (max_kinetic / max_stored) if max_stored > 0.0 else 0.0,
            "max_strain_nonuniformity": max_nonuniform,
            "max_bond_force_spread": max_spread,
            "kinetic_energy_J": float(energy["kinetic_energy_J"]),
            "bond_energy_J": float(energy.get("bond_energy_J", 0.0)),
            "cumulative_bond_dissipation_J": float(sim.cumulative_bond_dissipation_J),
            "strain": strains,
            "load": loads,
            "time": times,
        }
        if len(dead):
            first = int(dead[int(torch.argmin(sim.failure_time_s[dead]).item())].item())
            record.update(
                failure_bond=first,
                failure_strain=float(sim.failure_extension_m[first] / sim.rest_lengths[first]),
                failure_time_s=float(sim.failure_time_s[first]),
                failure_mode=int(sim.failure_mode[first]),
                failure_extension_m=float(sim.failure_extension_m[first]),
                broken_bonds=int((~sim.alive).sum()),
            )
        else:
            record.update(failure_bond=-1, failure_strain=math.nan,
                          failure_time_s=math.nan, failure_mode=0,
                          failure_extension_m=math.nan, broken_bonds=0)
        return record
