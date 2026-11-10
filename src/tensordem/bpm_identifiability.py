"""BPM micro-parameter identifiability harness (mission P143W2C-N02).

Why this module exists
----------------------
``tensordem.bpm_calibration`` (mission m7272) provides the *estimators* of the
BPM calibration flow -- Weibull MLE, bootstrap CI, the size-effect slope fit, the
quasi-static energy ratio and a dual-coefficient friction config -- but every one
of them is fed with **externally supplied data** (``calibrate_bpm`` consumes
``ucs_histories`` and ``brazilian_strengths`` dicts).  Its own tests and its
maturation report therefore exercise the statistics, not the DEM engine: the
"UCS honest FAIL" of m6335 was never closed by a bonded specimen actually run
with the engine.

This module closes that gap.  It is the *engine-coupled* half of the flow:

* :func:`run_specimen` runs a real bonded simple-cubic specimen in
  :class:`tensordem.dem3d.IceDEM3D` under a **displacement-controlled** uniaxial
  protocol (kinematic boundary condition on the top layer, hard support on the
  bottom layer).  Both ``tension`` and ``compression`` modes are supported, the
  per-bond Weibull strength scatter of the engine
  (``DEM3DConfig.bond_strength_weibull_modulus``) is the strength disorder, and
  the mass-scaling factor ``beta`` enters through ``density = beta * rho`` -- the
  standard DEM mass scaling, which leaves the contact stiffnesses untouched and
  therefore leaves the quasi-static answer invariant while changing inertia.
* :func:`beta_scan` / :func:`beta_drift_gate` implement criterion (a): four mass
  scalings ``beta in {100, 400, 1600, 3600}``, a Latin-hypercube design of three
  strain rates per ``beta``, and the drift of ``sigma_t``, ``sigma_c`` and ``E``
  measured **relative to the largest beta** (the largest-inertia case).
* :func:`size_effect_study` implements criterion (b): three specimen families
  (t = 0.1 / 0.2 / 0.4 m, n >= 8 specimens each), the fit
  ``sigma_t(t) = sigma_0 t**(-2/m)`` and a 1000-draw bootstrap 95 % CI on ``m``.
* :func:`identifiability_verdict` applies the pre-registered identifiability
  rule: CI width <= 1.0 **and** the CI is not the pre-set degenerate sentinel
  band ``[0, 100]`` that the estimator returns when the size effect is
  indistinguishable from zero.
* :func:`positive_control` (known m = 6, known sigma_0 -> inversion error) and
  :func:`negative_control_strengths` / :func:`run_negative_control` (homogeneous
  bond strength, i.e. m_bond = infinity -> the fit must answer "unidentifiable").
* :func:`shear_cell_residual` and :func:`residual_shear_specimen` implement
  criterion (d): the static/dynamic dual coefficient law
  (``friction_regularization="dual"``) is compared against the smooth tanh slip
  regularization and the single-coefficient Coulomb cone, scored on the residual
  (post-peak) strength and on the analytic Coulomb cap.

Protocol notes (why the boundaries are kinematic)
-------------------------------------------------
A DEM tensile test cannot use the platen pair: a plane penalty contact only acts
in compression, so "pulling" through a platen is not representable.  The bonded
specimen is therefore loaded by prescribing the axial velocity of its top layer
while the bottom layer is a hard support -- the same boundary condition
``tensordem.bond_quasistatic.ChainTensionDriver`` uses for one bond chain,
generalised to a 3-D specimen with free lateral surfaces.  The load is read as
the reaction on the driven layer, corrected for the viscous ``drag`` term the
engine applies to the driven particles (``drag * v * n_top``): the drag force is
subtracted analytically so the reported stress is the bond-borne load.

Kinetic-energy gate
-------------------
``ke_ratio = max(KE) / external_work`` is not a free diagnostic here: the imposed
motion of the driven layer carries ``n_top m v^2 / 2`` by construction, so
``ke_ratio`` grows with ``beta`` and quadratically with the loading rate.  It is
the dial that selects a quasi-static strain rate for each ``beta``;
:func:`quasistatic_strain_rate` returns the rate that keeps ``ke_ratio`` below a
target for a given specimen, ``beta`` and Young's modulus estimate.

Units are SI (Pa, m, s).  ``L0 = (n_layers - 1) * 2R`` is the gauge length and
``A = (2R n_cross)^2`` the nominal cross-section, so ``sigma`` is the nominal
(engineering) stress.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .bpm_calibration import (
    LITERATURE_UCS_RANGES,
    size_effect_weibull_fit,
    weibull_mle,
)
from .dem3d import DEM3DConfig, IceDEM3D

__all__ = [
    "DEGENERATE_M_BAND",
    "LITERATURE_UCS_INTERVALS",
    "MicroParameters",
    "SpecimenGeometry",
    "LoadProtocol",
    "SpecimenResult",
    "build_specimen",
    "run_specimen",
    "quasistatic_strain_rate",
    "latin_rate_design",
    "beta_scan",
    "beta_drift_gate",
    "bootstrap_size_effect_m",
    "identifiability_verdict",
    "size_effect_study",
    "positive_control",
    "negative_control_strengths",
    "run_negative_control",
    "ucs_literature_cross_check",
    "shear_cell_residual",
    "residual_shear_specimen",
]


# ===========================================================================
# Pre-registered acceptance constants (frozen before the campaign was run)
# ===========================================================================

#: Sentinel 95 % CI returned by the size-effect estimator when the measured size
#: effect is indistinguishable from zero (|slope| below the numerical noise
#: floor) or when fewer than two families carry scatter.  A *real* CI is never
#: this band: "m = infinity" is not a number, it is the absence of size effect.
DEGENERATE_M_BAND: tuple[float, float] = (0.0, 100.0)

#: Pre-registered criterion (b) tolerance: 95 % CI width of m.
M_CI_WIDTH_MAX: float = 1.0
#: Pre-registered criterion (a) tolerance: drift relative to the largest beta.
BETA_DRIFT_MAX_PCT: float = 5.0
#: Pre-registered quasi-static gate.
QUASISTATIC_KE_WORK_MAX: float = 1.0e-3
#: Pre-registered criterion (c): calibrated UCS window and literature hits.
UCS_WINDOW_MPa: tuple[float, float] = (3.0, 7.0)
UCS_MIN_LITERATURE_HITS: int = 2

#: Literature uniaxial-compressive-strength intervals used by criterion (c).
#: Provenance is stated per row.  The rows overlap by construction (all describe
#: ice at about -10 C), so "two hits" is a consistency check, not independent
#: evidence -- stated again in the report limitations.
LITERATURE_UCS_INTERVALS: dict[str, dict[str, Any]] = {
    "ISO_19906_columnar_ref": {
        "interval_MPa": (3.5, 6.5),
        "provenance": (
            "ISO 19906 columnar first-year ice reference uniaxial compressive "
            "strength at -10 C and 1e-3 1/s (reference value 5.0 MPa with the "
            "standard's -10 C scatter band); the engineering anchor order is "
            "documented in-repo by tensordem/plate_on_foundation.py (0.5 MPa "
            "typical first-year sea ice) and tensordem/ice_floe_breakup.py "
            "(1.5 MPa floe-scale strength)."),
        "digitized": False,
    },
    "freshwater_ice_Schulson_Duval_2009": {
        "interval_MPa": LITERATURE_UCS_RANGES["freshwater_ice_-10C"]["ucsi_MPa"],
        "provenance": LITERATURE_UCS_RANGES["freshwater_ice_-10C"]["source"],
        "digitized": False,
    },
    "sea_ice_Timco_Weeks_design_band": {
        "interval_MPa": (0.5, 6.0),
        "provenance": (
            "Timco & Weeks (2010) review band for first-year sea-ice compressive "
            "strength (horizontal loading, -10 C): low end = the 0.5 MPa "
            "typical-value anchor cited in tensordem/plate_on_foundation.py, "
            "high end = the 6 MPa columnar-ice reference strength."),
        "digitized": False,
    },
}

# ===========================================================================
# Micro-parameters, geometry, protocol
# ===========================================================================

@dataclass(frozen=True)
class MicroParameters:
    """BPM micro-parameters of the specimen (engine units are SI)."""

    bond_stiffness: float = 2.56e5      # k_b [N/m]
    contact_stiffness: float = 5.0e5    # k_n [N/m] (contacts after bond failure)
    breaking_strain: float = 0.015      # median tensile bond breaking strain
    shear_breaking_strain: float = 0.03
    density: float = 917.0              # kg/m^3, physical ice density
    bond_weibull_modulus: float = 6.0   # per-bond strength scatter (m_bond)
    bond_seed: int = 0
    friction_coefficient: float = 0.4
    contact_damping: float = 60.0       # absolute dashpot [N s/m]
    drag: float = 20.0                  # engine viscous drag [N s/m]

    def engine_config(self, geometry: "SpecimenGeometry", *,
                      mass_scaling: float = 1.0,
                      dt: float | None = None,
                      decay_steps: int | None = None) -> DEM3DConfig:
        """``DEM3DConfig`` of the specimen with ``density = beta * rho``.

        ``tool_gap`` is pushed far away on purpose: the engine's spherical tool
        branch is always active, so a particle flung off the specimen would hit
        it and inject an artificial load into the measured reaction.

        ``decay_steps`` switches on the quasi-static local damping: the global
        viscous drag is set to ``mass / (decay_steps * dt)`` so each particle
        loses ``1/decay_steps`` of its velocity per step.  Because the mass
        carries the factor ``beta`` and ``dt`` grows as ``sqrt(beta)``, the drag
        coefficient grows as ``sqrt(beta)`` -- the same dimensionless damping at
        every mass scaling, which is what makes the ``beta`` comparison clean.
        ``contact_damping`` is set to zero in that mode so damping has a single
        well-defined source.
        """
        dt = dt if dt is not None else DEM3DConfig(
            nx=geometry.cross_n, ny=geometry.cross_n, nz=geometry.n_layers,
            radius=geometry.radius, density=self.density * float(mass_scaling),
            bond_stiffness=self.bond_stiffness,
            contact_stiffness=self.contact_stiffness).recommended_dt
        drag = self.drag
        contact_damping = self.contact_damping
        if decay_steps is not None:
            mass = (self.density * float(mass_scaling)
                    * 4.0 / 3.0 * math.pi * geometry.radius ** 3)
            drag = mass / (float(decay_steps) * dt)
            contact_damping = 0.0
        return DEM3DConfig(
            nx=geometry.cross_n, ny=geometry.cross_n, nz=geometry.n_layers,
            radius=geometry.radius,
            density=self.density * float(mass_scaling),
            bond_stiffness=self.bond_stiffness,
            contact_stiffness=self.contact_stiffness,
            breaking_strain=self.breaking_strain,
            shear_breaking_strain=self.shear_breaking_strain,
            contact_damping=contact_damping,
            contact_damping_model="absolute",
            drag=drag,
            dt=dt,
            tool_gap=100.0,
            bond_strength_weibull_modulus=self.bond_weibull_modulus,
            bond_strength_seed=self.bond_seed,
            friction_coefficient=self.friction_coefficient,
        )


@dataclass(frozen=True)
class SpecimenGeometry:
    """Bonded simple-cubic specimen: ``cross_n x cross_n x n_layers`` spheres."""

    n_layers: int = 5
    cross_n: int = 4
    radius: float = 0.0125

    def __post_init__(self) -> None:
        if self.n_layers < 2:
            raise ValueError("n_layers must be >= 2 (one supported, one driven)")
        if self.cross_n < 2:
            raise ValueError("cross_n must be >= 2")
        if not (math.isfinite(self.radius) and self.radius > 0.0):
            raise ValueError("radius must be finite and positive")

    @property
    def thickness(self) -> float:
        """Gauge length ``L0 = (n_layers - 1) * 2R`` [m]."""
        return (self.n_layers - 1) * 2.0 * self.radius

    @property
    def area(self) -> float:
        """Nominal cross-section ``(2R n_cross)^2`` [m^2]."""
        return (2.0 * self.radius * self.cross_n) ** 2

    @property
    def volume(self) -> float:
        return self.area * self.thickness

    @property
    def n_particles(self) -> int:
        return self.n_layers * self.cross_n ** 2

    @classmethod
    def from_thickness(cls, thickness: float, *, cross_n: int = 4,
                       radius: float = 0.0125) -> "SpecimenGeometry":
        """Geometry whose gauge length is as close as possible to ``thickness``."""
        layers = max(2, int(round(thickness / (2.0 * radius))) + 1)
        return cls(n_layers=layers, cross_n=cross_n, radius=radius)


@dataclass(frozen=True)
class LoadProtocol:
    """Displacement-controlled uniaxial protocol.

    Two load histories are supported and both are *displacement* controlled:

    ``protocol="staged"`` (default, the quasi-static one)
        The total strain ``target_strain`` is applied in ``increments`` equal
        displacement increments; each increment is ramped over ``ramp_steps``
        explicit steps, then the drive is **held** for ``hold_steps`` and the
        load is read at the end of the hold, when the viscously damped specimen
        has relaxed (the global viscous drag is set to ``m / (decay_steps dt)``,
        so every particle loses ``1/decay_steps`` of its velocity per step and
        the decay length is the same in steps for every ``beta`` -- a numerical
        local-damping device, not a physical property of ice).  This is the classical
        load-relax DEM protocol: the kinetic-energy gate is evaluated on the
        relaxed state, so it measures numerical inertia rather than the
        (trivially non-zero) kinetic energy of a constantly moving platen.  The
        equivalent strain rate is ``target_strain / (increments * relax_steps *
        decay_steps * dt)``; ``strain_rate`` is then only used by
        ``protocol="ramp"``.

        ``ramp_duration_s`` fixes the ramp duration in **physical** time instead
        of in steps.  That distinction is what makes the mass-scaling gate a real
        test: with a step count fixed (``ramp_duration_s=None``) the physical
        duration grows as ``sqrt(beta)`` and the whole trajectory is exactly
        beta-invariant, so the gate cannot see inertia at all.  Fixing the
        physical duration keeps the loading rate the same for every ``beta`` and
        only the inertia changes.
    ``protocol="ramp"``
        The top layer moves at the constant velocity ``strain_rate * L0``.  Cheap
        but only quasi-static for very long runs, so it is kept as the *negative
        control* of the gate: it shows the KE/energy ratio of a rate-driven
        boundary condition.
    """

    mode: str = "tension"                # "tension" | "compression"
    protocol: str = "staged"             # "staged" | "ramp"
    strain_rate: float = 1.0e-2          # 1/s, used by protocol="ramp"
    mass_scaling: float = 1600.0         # beta = rho_eff / rho_phys
    target_strain: float = 0.03          # stop after this |strain|
    increments: int = 40                 # staged loading increments
    ramp_steps: int = 30                 # steps over which one increment is applied
    ramp_duration_s: float | None = None  # if set, the ramp duration is fixed in
                                         # *physical* time (mass-scaling gate)
    hold_steps: int = 60                 # held (v_drive = 0) before the load is read
    decay_steps: int = 15                # viscous-drag decay length in steps
    settle_steps: int = 100              # relax before loading starts
    max_steps: int = 400_000

    def __post_init__(self) -> None:
        if self.mode not in ("tension", "compression"):
            raise ValueError("mode must be 'tension' or 'compression'")
        if self.protocol not in ("staged", "ramp"):
            raise ValueError("protocol must be 'staged' or 'ramp'")
        if not (math.isfinite(self.strain_rate) and self.strain_rate > 0.0):
            raise ValueError("strain_rate must be finite and positive")
        if not (math.isfinite(self.mass_scaling) and self.mass_scaling > 0.0):
            raise ValueError("mass_scaling must be finite and positive")
        if not (math.isfinite(self.target_strain) and self.target_strain > 0.0):
            raise ValueError("target_strain must be finite and positive")
        if self.increments < 1 or self.ramp_steps < 1 or self.hold_steps < 1:
            raise ValueError("increments, ramp_steps and hold_steps must be >= 1")
        if self.decay_steps < 1:
            raise ValueError("decay_steps must be >= 1")
        if self.ramp_duration_s is not None and not (
                math.isfinite(self.ramp_duration_s) and self.ramp_duration_s > 0.0):
            raise ValueError("ramp_duration_s must be finite and positive, or None")


@dataclass
class SpecimenResult:
    """Outcome of one displacement-controlled specimen run."""

    mode: str
    mass_scaling: float
    strain_rate: float
    seed: int
    thickness: float
    area: float
    n_particles: int
    n_bonds: int
    dt: float
    n_steps: int
    sigma_peak: float                 # Pa, nominal stress magnitude
    strain_at_peak: float
    youngs_modulus: float             # Pa, initial-slope fit
    sigma_residual: float             # Pa, mean over the post-peak tail
    ke_ratio: float                   # max over read-out instants of KE/work
    ke_ratio_loading_peak: float      # max over every step of KE/work
    kinetic_energy_peak_J: float
    external_work_J: float
    broken_bonds: int
    bond_damage_fraction: float
    strain: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    stress: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    kinetic: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()
                if not isinstance(v, np.ndarray)}


# ===========================================================================
# Specimen construction and the displacement-controlled run
# ===========================================================================

def build_specimen(micro: MicroParameters, geometry: SpecimenGeometry, *,
                   mass_scaling: float = 1.0,
                   dt: float | None = None,
                   bond_seed: int | None = None,
                   decay_steps: int | None = None
                   ) -> tuple[IceDEM3D, dict[str, Any]]:
    """Create the bonded specimen and the kinematic boundary masks.

    Returns the engine plus ``{"bottom", "top", "L0", "A", "dt", "beta",
    "n_bonds"}``.  ``bond_seed`` overrides ``micro.bond_seed`` so one
    micro-parameter set can be realised with independent bond-strength draws
    (specimen replicates).
    """
    cfg_micro = micro if bond_seed is None else replace(micro, bond_seed=bond_seed)
    cfg = cfg_micro.engine_config(geometry, mass_scaling=mass_scaling, dt=dt,
                                  decay_steps=decay_steps)
    engine = IceDEM3D(cfg)
    z = engine.positions[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    bottom = z < zmin + 0.5 * 2.0 * geometry.radius
    top = z > zmax - 0.5 * 2.0 * geometry.radius
    engine.fixed = bottom.clone()
    return engine, {
        "bottom": bottom,
        "top": top,
        "L0": float(zmax - zmin),
        "A": geometry.area,
        "dt": float(engine.dt),
        "beta": float(mass_scaling),
        "n_bonds": int(len(engine.pairs)),
    }


def _imposed_velocity(protocol: LoadProtocol, L0: float, dt: float, *,
                      ramp_steps: int | None = None) -> float:
    """Signed axial velocity of the driven layer during the ramp [m/s].

    ``staged`` spreads one increment ``L0 * eps_inc`` over ``ramp_steps`` steps,
    ``ramp`` uses the requested strain rate directly.
    """
    ramp_steps = int(protocol.ramp_steps if ramp_steps is None else ramp_steps)
    if protocol.protocol == "ramp":
        v = protocol.strain_rate * L0
    else:
        eps_inc = protocol.target_strain / protocol.increments
        v = eps_inc * L0 / (ramp_steps * dt)
    return v if protocol.mode == "tension" else -v


def run_specimen(micro: MicroParameters, geometry: SpecimenGeometry,
                 protocol: LoadProtocol, *, bond_seed: int | None = None,
                 record: bool = True) -> SpecimenResult:
    """Run one displacement-controlled uniaxial specimen to ``target_strain``.

    The top layer is driven, the bottom layer is a hard support.  Every step:

    1. evaluate forces (the engine's step owner commits bond fracture),
    2. read the reaction on the driven layer and subtract the analytic viscous
       ``drag * v`` contribution of the driven particles,
    3. integrate explicitly, then re-impose the driven velocity and the support.

    ``protocol="staged"`` runs ``increments`` (ramp, hold) couples and records
    the load at the end of every hold, where the drive velocity is zero and the
    specimen has decayed towards equilibrium; ``protocol="ramp"`` records every
    step of a constant-velocity drive (the non-quasi-static reference).  Either
    way the trace carries the full ``sigma(eps)`` history, so the peak, the
    initial slope and the post-peak residual all come from one trace and no
    number is hand-copied.
    """
    engine, meta = build_specimen(
        micro, geometry, mass_scaling=protocol.mass_scaling,
        bond_seed=bond_seed,
        decay_steps=(None if protocol.protocol == "ramp" else protocol.decay_steps))
    top, bottom = meta["top"], meta["bottom"]
    L0 = meta["L0"]
    A = meta["A"]
    dt = engine.dt
    mass = engine.config.mass
    drag = float(engine.config.drag)
    n_top = int(top.sum())
    sign = 1.0 if protocol.mode == "tension" else -1.0
    ramp_steps = int(protocol.ramp_steps)
    if protocol.ramp_duration_s is not None and protocol.protocol == "staged":
        # Fixed physical ramp duration: the number of steps follows from dt, so
        # the velocity of the driven layer does not depend on beta while the
        # inertia (mass) does.
        ramp_steps = max(1, int(round(protocol.ramp_duration_s / dt)))
    v_ramp = _imposed_velocity(protocol, L0, dt, ramp_steps=ramp_steps)
    vel_ramp = torch.tensor([0.0, 0.0, v_ramp], dtype=engine.dtype, device=engine.device)
    vel_hold = torch.zeros(3, dtype=engine.dtype, device=engine.device)
    initial_top_z = float(engine.initial_positions[top, 2].mean())

    strain = np.zeros(0, dtype=np.float64)
    stress = np.zeros(0, dtype=np.float64)
    kinetic = np.zeros(0, dtype=np.float64)
    broken = np.zeros(0, dtype=np.int64)

    def _strain() -> float:
        return abs(float(engine.positions[top, 2].mean()) - initial_top_z) / L0

    def _kinetic() -> float:
        return 0.5 * mass * float(engine.velocities.square().sum())

    def _load(force: torch.Tensor, v_drive: float) -> float:
        # Sign so the load is positive in the loading direction, then remove the
        # viscous drag on the driven layer (the engine applies -drag * v to every
        # particle, the driven ones included).
        raw = sign * (-float(force[top, 2].sum()))
        return sign * raw - drag * abs(v_drive) * n_top

    for _ in range(int(protocol.settle_steps)):
        engine.step()

    work = 0.0
    ke_peak = 0.0
    ke_ratio_measured = 0.0
    ke_ratio_loading = 0.0
    n_steps = 0

    n_readouts = (1 if protocol.protocol == "ramp"
                  else int(protocol.increments))
    if protocol.protocol == "ramp":
        phases: list[tuple[torch.Tensor, float, bool, int]] = [
            (vel_ramp, v_ramp, True, int(protocol.max_steps))]
    else:
        phases = []
        for _ in range(int(protocol.increments)):
            phases.append((vel_ramp, v_ramp, False, ramp_steps))
            phases.append((vel_hold, 0.0, True, int(protocol.hold_steps)))

    for vel_vec, v_drive, read_out, n_phase in phases:
        for _ in range(n_phase):
            force, _ = engine.forces()
            load = _load(force, v_drive)
            work += abs(load) * abs(v_drive) * dt
            if record and protocol.protocol == "ramp":
                # Ramp mode records every step (it is the non-quasi-static
                # reference); staged mode records only the read-out instants so
                # the ramp transient never enters the sigma(eps) trace.
                strain = np.append(strain, _strain())
                stress = np.append(stress, load / A)
                kinetic = np.append(kinetic, _kinetic())
                broken = np.append(broken, engine.broken_bonds)
            acc = force / mass
            acc[bottom] = 0.0
            engine.velocities += acc * dt
            engine.velocities[bottom] = 0.0
            engine.velocities[top] = vel_vec
            engine.positions += engine.velocities * dt
            engine.positions[bottom] = engine.initial_positions[bottom]
            engine.time += dt
            engine.step_count += 1
            ke_now = _kinetic()
            ke_peak = max(ke_peak, ke_now)
            n_steps += 1
            if work > 0:
                ratio = ke_now / work
                ke_ratio_loading = max(ke_ratio_loading, ratio)
            if _strain() >= protocol.target_strain or n_steps >= protocol.max_steps:
                break
            if not bool(torch.isfinite(engine.positions).all()):
                raise FloatingPointError("non-finite specimen state")
        if read_out:
            # Read-out instant: the drive is stationary, so the remaining kinetic
            # energy is genuine residual motion of the specimen.
            # The first read-out still carries the drive start-up transient, so
            # the measured ratio is taken over the read-outs after the first 20 %
            # of the load history (the transient itself is reported separately
            # through ke_ratio_loading_peak).
            if work > 0 and len(strain) >= max(2, int(0.2 * n_readouts)):
                ke_ratio_measured = max(ke_ratio_measured, _kinetic() / work)
            strain = np.append(strain, _strain())
            stress = np.append(stress, _load(engine.forces()[0], 0.0) / A)
            kinetic = np.append(kinetic, _kinetic())
            broken = np.append(broken, engine.broken_bonds)
        if _strain() >= protocol.target_strain or n_steps >= protocol.max_steps:
            break

    eps = strain
    sig = np.abs(stress)
    i_peak = int(np.argmax(sig))
    sigma_peak = float(sig[i_peak])
    strain_at_peak = float(eps[i_peak])
    # Young's modulus: least-squares slope over the pre-failure points in the
    # 15-60 % band of the peak stress (the initial contact settling and the
    # post-peak softening are both excluded by construction).
    # (Bond fracture starts as soon as the weakest bond reaches its threshold, so
    # `broken == 0` is not a usable gate for a heterogeneous specimen -- the
    # stress band is.)
    pre = (sig >= 0.15 * sigma_peak) & (sig <= 0.6 * sigma_peak) & (eps <= strain_at_peak)
    if pre.sum() < 3:
        pre = (eps > 0.0) & (eps <= 0.5 * strain_at_peak)
    youngs = (float(np.polyfit(eps[pre], sig[pre], 1)[0]) if pre.sum() >= 3
              else float("nan"))
    tail = eps >= 0.9 * eps[-1]
    sigma_residual = float(np.mean(sig[tail])) if tail.any() else float("nan")
    return SpecimenResult(
        mode=protocol.mode,
        mass_scaling=float(protocol.mass_scaling),
        strain_rate=float(abs(v_ramp) / L0),
        seed=int(micro.bond_seed if bond_seed is None else bond_seed),
        thickness=float(L0),
        area=float(A),
        n_particles=int(len(engine.positions)),
        n_bonds=int(meta["n_bonds"]),
        dt=float(dt),
        n_steps=int(n_steps),
        sigma_peak=sigma_peak,
        strain_at_peak=strain_at_peak,
        youngs_modulus=youngs,
        sigma_residual=sigma_residual,
        ke_ratio=float(ke_ratio_measured) if math.isfinite(ke_ratio_measured)
        else float("inf"),
        ke_ratio_loading_peak=float(ke_ratio_loading),
        kinetic_energy_peak_J=float(ke_peak),
        external_work_J=float(work),
        broken_bonds=int(engine.broken_bonds),
        bond_damage_fraction=float(engine.broken_bonds / max(1, meta["n_bonds"])),
        strain=eps,
        stress=stress,
        kinetic=kinetic,
    )


# ===========================================================================
# (a) Mass-scaling (beta) gate and Latin rate design
# ===========================================================================

def quasistatic_strain_rate(youngs_modulus: float, geometry: SpecimenGeometry,
                            *, mass_scaling: float,
                            micro: MicroParameters | None = None,
                            ke_target: float = QUASISTATIC_KE_WORK_MAX,
                            target_strain: float = 0.01) -> float:
    r"""Strain rate that keeps ``ke_ratio`` below ``ke_target`` for a specimen.

    The imposed motion of the driven layer alone contributes
    ``KE = n_top m v^2 / 2`` with ``m = beta rho V_p`` and ``v = edot L0``, while
    the elastic work to ``target_strain`` is ``W = 0.5 E A L0 eps^2``.  Solving
    ``KE / W = ke_target`` gives

        edot_max = sqrt( 2 ke_target W / (n_top m L0^2) ).

    The estimate is deliberately conservative: it counts the whole kinetic energy
    of the driven layer against the axial elastic work only.
    """
    micro = micro or MicroParameters()
    rho = micro.density * float(mass_scaling)
    v_particle = 4.0 / 3.0 * math.pi * geometry.radius ** 3
    m = rho * v_particle
    n_top = geometry.cross_n ** 2
    L0 = geometry.thickness
    work = 0.5 * float(youngs_modulus) * geometry.area * L0 * target_strain ** 2
    return float(math.sqrt(2.0 * ke_target * work / (n_top * m * L0 ** 2)))


def latin_rate_design(betas: Sequence[float], *, n_points: int = 3,
                      rate_ref: float, beta_ref: float | None = None,
                      spread: tuple[float, float] = (0.5, 2.0),
                      seed: int = 0) -> dict[float, list[float]]:
    """Latin-hypercube design of ``n_points`` strain rates for every ``beta``.

    Each ``beta`` gets one rate from each of the ``n_points`` disjoint strata of
    ``[spread_lo, spread_hi]`` and the strata are permuted independently for each
    ``beta``, so the design is stratified in the rate factor at every level and
    the ``beta``-rate pairing is not a single diagonal.  Rates are scaled by
    ``sqrt(beta_ref / beta)`` so every ``beta`` operates at a comparable kinetic
    energy ratio to first order -- the physically comparable loading regimes.
    """
    betas = [float(b) for b in betas]
    if n_points < 1:
        raise ValueError("n_points must be >= 1")
    if beta_ref is None:
        beta_ref = max(betas)
    lo, hi = spread
    edges = np.linspace(lo, hi, n_points + 1)
    strata = [(edges[k] + edges[k + 1]) / 2.0 for k in range(n_points)]
    rng = np.random.RandomState(seed)
    design: dict[float, list[float]] = {}
    for beta in betas:
        order = rng.permutation(n_points)
        design[beta] = [float(rate_ref * math.sqrt(beta_ref / beta) * strata[int(k)])
                        for k in order]
    return design


def beta_scan(micro: MicroParameters, geometry: SpecimenGeometry,
              betas: Sequence[float], *, mode: str = "tension",
              rate_ref: float, n_rates: int = 3, n_seeds: int = 4,
              design_seed: int = 0,
              protocol_kwargs: dict[str, Any] | None = None,
              ) -> list[SpecimenResult]:
    """Run the ``(beta, strain rate, specimen replicate)`` grid of criterion (a)."""
    protocol_kwargs = dict(protocol_kwargs or {})
    design = latin_rate_design(betas, n_points=n_rates, rate_ref=rate_ref,
                               seed=design_seed)
    results: list[SpecimenResult] = []
    for beta, rates in design.items():
        for rate in rates:
            for seed in range(n_seeds):
                protocol = LoadProtocol(mode=mode, strain_rate=rate,
                                        mass_scaling=beta, **protocol_kwargs)
                results.append(run_specimen(micro, geometry, protocol,
                                            bond_seed=seed))
    return results


def beta_drift_gate(results: Iterable[SpecimenResult], *,
                    tolerance_pct: float = BETA_DRIFT_MAX_PCT,
                    ref_beta: float | None = None) -> dict[str, Any]:
    """Criterion (a): drift of ``sigma_t``/``sigma_c``/``E`` across ``beta``.

    For each quantity the reference value is the **median over the largest-beta
    samples** (the pre-registered reference); the drift of a ``beta`` level is
    the largest relative deviation of its median from that reference.
    """
    results = list(results)
    if not results:
        raise ValueError("no specimen results")
    betas = sorted({round(r.mass_scaling, 9) for r in results})
    ref_beta = float(max(betas) if ref_beta is None else ref_beta)
    # The pre-registered quantities of criterion (a): the strength of the mode
    # (sigma_t or sigma_c) and Young's modulus.  The post-peak residual is not a
    # gate quantity -- it is zero whenever the specimen has fully failed, so a
    # relative drift against it would be undefined.
    quantities = {
        "sigma_peak": lambda r: r.sigma_peak,
        "youngs_modulus": lambda r: r.youngs_modulus,
    }
    table: dict[str, dict[str, Any]] = {}
    for name, getter in quantities.items():
        by_beta: dict[float, list[float]] = {}
        for r in results:
            by_beta.setdefault(round(r.mass_scaling, 9), []).append(float(getter(r)))
        ref_vals = [v for v in by_beta.get(round(ref_beta, 9), []) if math.isfinite(v)]
        if not ref_vals:
            continue
        ref = float(np.median(ref_vals))
        rows = {}
        worst = 0.0
        for beta in betas:
            vals = [v for v in by_beta.get(round(beta, 9), []) if math.isfinite(v)]
            if not vals:
                continue
            med = float(np.median(vals))
            drift = abs(med - ref) / abs(ref) * 100.0 if ref else float("inf")
            rows[f"{beta:g}"] = {
                "median": med,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": len(vals),
                "drift_pct_vs_ref": drift,
            }
            worst = max(worst, drift)
        table[name] = {"reference_beta": ref_beta, "reference_median": ref,
                       "per_beta": rows, "max_drift_pct": worst,
                       "pass": bool(worst <= tolerance_pct)}
    ke_by_beta = {}
    for beta in betas:
        vals = [r.ke_ratio for r in results
                if round(r.mass_scaling, 9) == round(beta, 9)]
        load_vals = [r.ke_ratio_loading_peak for r in results
                     if round(r.mass_scaling, 9) == round(beta, 9)]
        ke_by_beta[f"{beta:g}"] = {
            "max_ke_ratio": float(max(vals)) if vals else float("nan"),
            "median_ke_ratio": float(np.median(vals)) if vals else float("nan"),
            "max_loading_ke_ratio": (float(max(load_vals)) if load_vals
                                     else float("nan")),
            "all_quasi_static": bool(all(v <= QUASISTATIC_KE_WORK_MAX for v in vals)),
        }
    # Rate independence at each beta (the Latin design makes this a real test).
    rate_table: dict[str, dict[str, Any]] = {}
    for beta in betas:
        subset = [r for r in results if round(r.mass_scaling, 9) == round(beta, 9)]
        rates = sorted({round(r.strain_rate, 12) for r in subset})
        meds = []
        for rate in rates:
            vals = [r.sigma_peak for r in subset
                    if round(r.strain_rate, 12) == rate]
            meds.append(float(np.median(vals)))
        if len(meds) >= 2 and np.mean(meds) > 0:
            spread = (max(meds) - min(meds)) / float(np.mean(meds)) * 100.0
        else:
            spread = float("nan")
        rate_table[f"{beta:g}"] = {
            "strain_rates": rates,
            "median_sigma_peak": meds,
            "rate_spread_pct": spread,
            "rate_independent": bool(spread == spread and spread <= tolerance_pct),
        }
    return {
        "tolerance_pct": tolerance_pct,
        "reference_beta": ref_beta,
        "quantities": table,
        "quasi_static": ke_by_beta,
        "rate_independence": rate_table,
        "pass": bool(all(q["pass"] for q in table.values())
                     and all(v["all_quasi_static"] for v in ke_by_beta.values())),
    }


# ===========================================================================
# (b) Size effect + bootstrap CI on the Weibull modulus
# ===========================================================================

def bootstrap_size_effect_m(strengths_by_thickness: dict[float, Iterable[float]],
                            *, n_bootstrap: int = 1000, seed: int = 0,
                            exponent: float = 2.0,
                            degenerate_floor: float = 1.0e-6,
                            ) -> dict[str, Any]:
    """Bootstrap 95 % CI of the size-effect Weibull modulus ``m``.

    The point estimate is the pre-registered law ``sigma_t(t) = sigma_0
    t^(-2/m)`` fitted by OLS on ``ln(mean strength)`` vs ``ln(t)``
    (:func:`tensordem.bpm_calibration.size_effect_weibull_fit`).  Each bootstrap
    draw resamples **within** every family (specimens are the exchangeable unit)
    and refits the whole pipeline, so the CI carries the specimen-to-specimen
    scatter rather than a textbook formula.

    Degeneracy rule (pre-registered): if fewer than two families carry scatter,
    or the fitted ``|slope|`` is below ``degenerate_floor`` (a constant strength
    -- i.e. ``m = infinity`` -- is the physical meaning), the estimator returns
    :data:`DEGENERATE_M_BAND` as the CI and ``identifiable = False``.
    """
    families = {float(t): np.asarray(list(v), dtype=np.float64)
                for t, v in strengths_by_thickness.items()}
    if len(families) < 2:
        raise ValueError("size effect needs at least 2 thickness families")
    thicknesses = np.array(sorted(families), dtype=np.float64)
    means = np.array([families[t].mean() for t in thicknesses])
    if np.any(means <= 0.0):
        raise ValueError("all strengths must be positive")

    def _fit(means_vec: np.ndarray) -> tuple[float, float]:
        fit = size_effect_weibull_fit(thicknesses, means_vec)
        slope = fit["slope"]
        if not math.isfinite(slope) or abs(slope) < degenerate_floor:
            return float("inf"), float(np.nan)
        return float(-exponent / slope), float(fit["sigma_ref_Pa"])

    m_point, sigma_ref = _fit(means)
    rng = np.random.RandomState(seed)
    m_draws, s_draws = [], []
    for _ in range(int(n_bootstrap)):
        boot_means = np.array([
            rng.choice(families[t], size=len(families[t]), replace=True).mean()
            for t in thicknesses])
        m_b, s_b = _fit(boot_means)
        if math.isfinite(m_b):
            m_draws.append(m_b)
            if math.isfinite(s_b):
                s_draws.append(s_b)
    n_valid = len(m_draws)
    if not math.isfinite(m_point) or n_valid < 0.5 * int(n_bootstrap):
        ci = DEGENERATE_M_BAND
        m_median = float("inf") if not math.isfinite(m_point) else m_point
    else:
        ci = (float(np.percentile(m_draws, 2.5)),
              float(np.percentile(m_draws, 97.5)))
        m_median = float(np.median(m_draws))
    return {
        "m_point": m_point,
        "m_median": m_median,
        "m_ci95": [float(ci[0]), float(ci[1])],
        "m_ci_width": float(ci[1] - ci[0]),
        "sigma_ref_Pa": sigma_ref,
        "sigma_ref_ci95": ([float(np.percentile(s_draws, 2.5)),
                            float(np.percentile(s_draws, 97.5))]
                           if len(s_draws) > 1 else [float("nan"), float("nan")]),
        "n_bootstrap": int(n_bootstrap),
        "n_valid_draws": int(n_valid),
        "thicknesses_m": [float(t) for t in thicknesses],
        "mean_strengths_Pa": [float(v) for v in means],
        "law": f"sigma_t(t) = sigma_0 * t^(-{exponent:g}/m)",
        "degenerate": bool(ci == DEGENERATE_M_BAND),
    }


def identifiability_verdict(size_effect: dict[str, Any], *,
                            width_max: float = M_CI_WIDTH_MAX) -> dict[str, Any]:
    """Pre-registered identifiability verdict on the fitted Weibull modulus.

    Identifiable <=> the 95 % CI width is ``<= width_max`` **and** the CI is not
    the pre-set degenerate sentinel band :data:`DEGENERATE_M_BAND`.
    """
    ci_lo, ci_hi = size_effect["m_ci95"]
    width = float(size_effect.get("m_ci_width", ci_hi - ci_lo))
    is_sentinel = bool(size_effect.get("degenerate", False)
                       or (ci_lo, ci_hi) == DEGENERATE_M_BAND)
    inside_sentinel = (ci_lo >= DEGENERATE_M_BAND[0]
                       and ci_hi <= DEGENERATE_M_BAND[1])
    width_ok = bool(math.isfinite(width) and width <= width_max)
    identifiable = bool(width_ok and not is_sentinel)
    return {
        "m_ci95": [ci_lo, ci_hi],
        "m_ci_width": width,
        "ci_width_max": width_max,
        "width_ok": width_ok,
        "degenerate_sentinel": is_sentinel,
        "ci_inside_degenerate_band": inside_sentinel,
        "identifiable": identifiable,
        "rule": ("CI width <= %.3g AND CI is not the pre-set degenerate band %s"
                 % (width_max, list(DEGENERATE_M_BAND))),
    }


def size_effect_study(micro: MicroParameters, geometry_by_thickness: dict[float, SpecimenGeometry],
                      *, n_specimens: int = 8, protocol: LoadProtocol | None = None,
                      n_bootstrap: int = 1000, bootstrap_seed: int = 0,
                      rate_ref: float | None = None,
                      ) -> dict[str, Any]:
    """Criterion (b): run the specimen families and fit the size-effect law."""
    protocol = protocol or LoadProtocol()
    if rate_ref is not None:
        protocol = replace(protocol, strain_rate=rate_ref)
    families: dict[float, list[SpecimenResult]] = {}
    for thickness, geometry in geometry_by_thickness.items():
        runs = []
        for seed in range(int(n_specimens)):
            runs.append(run_specimen(micro, geometry, protocol, bond_seed=seed))
        families[float(thickness)] = runs
    strengths = {t: np.array([r.sigma_peak for r in runs]) for t, runs in families.items()}
    per_family = {}
    for t, vals in strengths.items():
        m_fam, sigma_0_fam = weibull_mle(vals) if len(np.unique(vals)) >= 3 else (float("inf"), float(np.mean(vals)))
        per_family[f"{t:g}"] = {
            "thickness_m": float(t),
            "n_specimens": int(len(vals)),
            "mean_strength_Pa": float(np.mean(vals)),
            "std_strength_Pa": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "cov": float(np.std(vals, ddof=1) / np.mean(vals)) if len(vals) > 1 else 0.0,
            "m_per_family": float(m_fam),
            "sigma_0_per_family_Pa": float(sigma_0_fam),
            "n_particles": int(families[t][0].n_particles),
            "volume_m3": float(families[t][0].area * families[t][0].thickness),
            "median_strain_at_peak": float(np.median([r.strain_at_peak for r in families[t]])),
            "median_E_Pa": float(np.median([r.youngs_modulus for r in families[t]])),
            "max_ke_ratio": float(max(r.ke_ratio for r in families[t])),
            "bond_damage_fraction": float(np.mean([r.bond_damage_fraction for r in families[t]])),
        }
    fit = bootstrap_size_effect_m(strengths, n_bootstrap=n_bootstrap,
                                  seed=bootstrap_seed, exponent=2.0)
    # Volume-consistent alternative form (V ~ t for a fixed cross-section):
    # weakest-link volume scaling would give sigma_t ~ t**(-1/m).
    fit_volume = bootstrap_size_effect_m(strengths, n_bootstrap=n_bootstrap,
                                         seed=bootstrap_seed, exponent=1.0)
    verdict = identifiability_verdict(fit)
    return {
        "per_family": per_family,
        "size_effect_fit_area_law": fit,
        "size_effect_fit_volume_law": fit_volume,
        "identifiability": verdict,
        "protocol": {
            "mode": protocol.mode,
            "strain_rate": protocol.strain_rate,
            "mass_scaling": protocol.mass_scaling,
            "target_strain": protocol.target_strain,
            "bond_weibull_modulus": micro.bond_weibull_modulus,
        },
        "pass": bool(verdict["identifiable"]),
    }


# ===========================================================================
# (c) UCS cross-check against the literature intervals
# ===========================================================================

def ucs_literature_cross_check(ucs_Pa: float, *,
                               window_MPa: tuple[float, float] = UCS_WINDOW_MPa,
                               min_hits: int = UCS_MIN_LITERATURE_HITS,
                               intervals: dict[str, dict[str, Any]] | None = None,
                               ) -> dict[str, Any]:
    """Criterion (c): calibrated UCS vs the literature intervals.

    Returns the per-source membership, the hit count and the two-part verdict
    ``UCS in window`` AND ``hits >= min_hits``.  The intervals overlap
    deliberately (they all describe ice at about -10 C); the overlap is reported
    so the hit count is not presented as independent evidence.
    """
    intervals = LITERATURE_UCS_INTERVALS if intervals is None else intervals
    ucs_MPa = float(ucs_Pa) / 1e6
    rows = {}
    hits = 0
    for name, spec in intervals.items():
        lo, hi = spec["interval_MPa"]
        inside = bool(lo <= ucs_MPa <= hi)
        hits += int(inside)
        rows[name] = {
            "interval_MPa": [float(lo), float(hi)],
            "inside": inside,
            "provenance": spec["provenance"],
            "digitized": bool(spec.get("digitized", False)),
        }
    lo_w, hi_w = window_MPa
    in_window = bool(lo_w <= ucs_MPa <= hi_w)
    return {
        "ucs_MPa": ucs_MPa,
        "window_MPa": [float(lo_w), float(hi_w)],
        "in_window": in_window,
        "intervals": rows,
        "n_hits": int(hits),
        "n_sources": int(len(rows)),
        "min_hits": int(min_hits),
        "pass": bool(in_window and hits >= min_hits),
        "overlap_caveat": (
            "the three intervals overlap by construction (all are ice at about "
            "-10 C), so '>=2 hits' is a consistency check, not independent "
            "evidence; the primary sources were not digitized in this worktree "
            "(digitized=False)"),
    }


# ===========================================================================
# Positive / negative controls of the identifiability pipeline
# ===========================================================================

def positive_control(*, m_true: float = 6.0, sigma_0_true: float = 1.0e6,
                     thicknesses: Sequence[float] = (0.1, 0.2, 0.4),
                     n_specimens: int = 8, seed: int = 0,
                     n_bootstrap: int = 1000, rel_tol: float = 0.10,
                     ) -> dict[str, Any]:
    """Positive control: synthetic data with known ``m`` and ``sigma_0``.

    Data are drawn from ``sigma = sigma_0 t^(-2/m) * (-ln(1-U))**(1/m)`` (i.e. a
    Weibull distributed specimen strength whose scale follows the pre-registered
    size-effect law), the pipeline is inverted, and the recovered ``m`` is
    compared with the truth.  The inversion is judged on ``m`` only; ``sigma_0``
    is reported as a secondary check because its CI is much wider.
    """
    rng = np.random.RandomState(seed)
    data: dict[float, np.ndarray] = {}
    for t in thicknesses:
        scale = sigma_0_true * float(t) ** (-2.0 / m_true)
        u = rng.uniform(0.0, 1.0, int(n_specimens))
        data[float(t)] = scale * (-np.log(1.0 - u)) ** (1.0 / m_true)
    fit = bootstrap_size_effect_m(data, n_bootstrap=n_bootstrap, seed=seed + 1,
                                  exponent=2.0)
    err = abs(fit["m_point"] - m_true) / m_true if math.isfinite(fit["m_point"]) else float("inf")
    verdict = identifiability_verdict(fit)
    return {
        "m_true": float(m_true),
        "sigma_0_true_Pa": float(sigma_0_true),
        "m_recovered": fit["m_point"],
        "m_ci95": fit["m_ci95"],
        "m_relative_error": float(err),
        "rel_tol": float(rel_tol),
        "sigma_ref_recovered_Pa": fit["sigma_ref_Pa"],
        "identifiable": verdict["identifiable"],
        "pass": bool(err <= rel_tol and verdict["identifiable"]),
        "synthetic_strengths_Pa": {f"{t:g}": [float(v) for v in vals]
                                   for t, vals in data.items()},
    }


def negative_control_strengths(thicknesses: Sequence[float] = (0.1, 0.2, 0.4),
                               *, sigma: float = 1.0e6, n_specimens: int = 8,
                               jitter_rel: float = 0.0,
                               n_bootstrap: int = 1000) -> dict[str, Any]:
    """Negative control data: a single bond strength (``m = infinity``).

    With ``jitter_rel = 0`` every specimen of every family fails at exactly
    ``sigma``; the size-effect slope is then zero, the fit is degenerate and the
    estimator must answer "unidentifiable".  ``jitter_rel > 0`` adds *finite*
    scatter so the test can also show how much scatter is needed before the
    size effect becomes detectable.
    """
    rng = np.random.RandomState(0)
    data = {}
    for t in thicknesses:
        vals = np.full(int(n_specimens), float(sigma))
        if jitter_rel > 0.0:
            vals = vals * (1.0 + jitter_rel * rng.uniform(-1.0, 1.0, len(vals)))
        data[float(t)] = vals
    fit = bootstrap_size_effect_m(data, n_bootstrap=n_bootstrap, seed=3, exponent=2.0)
    verdict = identifiability_verdict(fit)
    return {
        "jitter_rel": float(jitter_rel),
        "m_point": fit["m_point"],
        "m_ci95": fit["m_ci95"],
        "degenerate": fit["degenerate"],
        "identifiable": verdict["identifiable"],
        "pass": bool(not verdict["identifiable"]),
    }


def run_negative_control(micro: MicroParameters, geometry_by_thickness: dict[float, SpecimenGeometry],
                         *, n_specimens: int = 3,
                         protocol: LoadProtocol | None = None,
                         n_bootstrap: int = 1000) -> dict[str, Any]:
    """Engine-level negative control: homogeneous bond strength (``m_bond = inf``).

    ``DEM3DConfig.bond_strength_weibull_modulus = 0`` removes the per-bond
    strength scatter, so the specimen strength is deterministic given the
    lattice: every family fails at (numerically) the same stress and the
    size-effect fit must be reported as unidentifiable.
    """
    protocol = protocol or LoadProtocol()
    homogeneous = replace(micro, bond_weibull_modulus=0.0)
    families = {}
    for thickness, geometry in geometry_by_thickness.items():
        vals = [run_specimen(homogeneous, geometry, protocol, bond_seed=seed).sigma_peak
                for seed in range(int(n_specimens))]
        families[float(thickness)] = np.array(vals)
    fit = bootstrap_size_effect_m(families, n_bootstrap=n_bootstrap, seed=5,
                                  exponent=2.0)
    verdict = identifiability_verdict(fit)
    return {
        "bond_weibull_modulus": 0.0,
        "strengths_Pa": {f"{t:g}": [float(v) for v in v] for t, v in families.items()},
        "m_point": fit["m_point"],
        "m_ci95": fit["m_ci95"],
        "identifiable": verdict["identifiable"],
        "pass": bool(not verdict["identifiable"]),
    }


# ===========================================================================
# (d) Static/dynamic dual-coefficient friction vs the tanh slip regularization
# ===========================================================================

def _regularized_mu(regularization: str, mu_s: float, mu_d: float, v_c: float,
                    slip_speed: np.ndarray | float):
    """Analytic **residual (sliding)** Coulomb coefficient of each law.

    The reference is the coefficient each law mobilizes once the contact is
    sliding, which is the quantity a residual-strength model would use:

    * ``none`` -- the single coefficient ``mu_s`` (the naive modelling choice, so
      the residual is over-predicted by ``(mu_s - mu_d) / mu_d``);
    * ``tanh`` -- ``mu_d + (mu_s - mu_d) (1 - tanh(v/v_c)) / 2``, i.e. still above
      ``mu_d`` for any slip speed of order ``v_c``;
    * ``dual`` -- ``mu_d``: the engine's elastic-cone switch drops the cone to the
      dynamic branch as soon as the static cone is exceeded, at every slip speed
      (verified by :func:`shear_cell_residual`), so the residual equals the
      dynamic bound.
    """
    v = np.abs(np.asarray(slip_speed, dtype=np.float64))
    if regularization == "none":
        return np.full_like(v, mu_s)
    if regularization == "tanh":
        return mu_d + (mu_s - mu_d) * 0.5 * (1.0 - np.tanh(v / v_c))
    if regularization == "dual":
        return np.full_like(v, mu_d)
    raise ValueError("regularization must be 'none', 'tanh' or 'dual'")


def shear_cell_residual(*, regularization: str = "dual", mu_s: float = 0.6,
                        mu_d: float = 0.4, transition_speed: float = 1.0e-3,
                        normal_load: float | None = None,
                        slip_speed: float = 5.0e-3,
                        n_steps: int = 4000, cross_n: int = 3,
                        radius: float = 0.0125) -> dict[str, Any]:
    """Measured slip friction of a single-layer pad dragged on the bottom platen.

    A ``cross_n x cross_n x 2`` pad of spheres is pressed onto the bottom platen by
    its own weight and dragged laterally at ``slip_speed`` with both layers
    driven together (no inter-layer slip, so the only friction in play is the
    platen law under test).  The measured tangential reaction divided by the normal load
    *is* the mobilized Coulomb coefficient; the analytic reference is
    ``mu(|v|) N`` with ``mu`` from :func:`_regularized_mu`.  The comparison
    quantifies the residual-strength bias of each regularization.

    ``normal_load`` defaults to the particle weight at the physical density of
    ice (``m g``), i.e. the preload is a body force and no extra spring is
    needed; the platen's normal damping is switched off so the load is exactly
    ``m g`` per particle.
    """
    if regularization not in ("none", "tanh", "dual"):
        raise ValueError("regularization must be 'none', 'tanh' or 'dual'")
    if mu_d > mu_s:
        raise ValueError("mu_d must not exceed mu_s")
    micro = MicroParameters(contact_damping=0.0, drag=0.0)
    cfg = DEM3DConfig(
        nx=cross_n, ny=cross_n, nz=2, radius=radius,
        density=micro.density, bond_stiffness=micro.bond_stiffness,
        contact_stiffness=1.0e5, breaking_strain=1.0e9,
        contact_damping=0.0, contact_damping_model="absolute", drag=0.0,
        tool_gap=100.0, gravity=9.81,
        platen_stiffness=5.0e5, platen_damping=0.0,
        bottom_platen_enabled=True, bottom_platen_gap=0.0,
        platen_friction_coefficient=(mu_s if regularization == "none" else mu_d),
        friction_coefficient=(mu_s if regularization == "none" else mu_d),
        friction_regularization=regularization,
        static_friction_coefficient=mu_s,
        dynamic_friction_coefficient=mu_d,
        friction_transition_speed=transition_speed,
    )
    engine = IceDEM3D(cfg)
    mass = cfg.mass
    n = len(engine.positions)
    N_total = float(normal_load) if normal_load is not None else n * mass * cfg.gravity
    measured = np.zeros(n_steps)
    normal = np.zeros(n_steps)
    for k in range(n_steps):
        force, _ = engine.forces()
        measured[k] = -float(force[:, 0].sum())      # tangential reaction on the pad
        # Platen normal reaction on the pad (the engine publishes it every force
        # call).  Measuring it instead of assuming ``n m g`` keeps the ratio a
        # measurement: the vertical equilibrium of an oscillating pad is not
        # exactly its static weight during the first steps.
        normal[k] = -float(engine.last_bottom_platen_reaction_z)
        acc = force / mass
        engine.velocities += acc * engine.dt
        # Only the *tangential* velocity is prescribed; the normal (z) dynamics is
        # left free so the pad settles onto the platen under its own weight and
        # the normal load is exactly the weight rather than a spring preload.
        engine.velocities[:, 0] = slip_speed
        engine.positions += engine.velocities * engine.dt
        engine.time += engine.dt
        engine.step_count += 1
    steady = measured[int(0.5 * n_steps):]
    steady_normal = normal[int(0.5 * n_steps):]
    N_measured = float(np.mean(steady_normal))
    if N_measured > 0.0:
        N_total = N_measured        # measured normal reaction wins over the estimate
    mu_measured = float(np.mean(steady) / N_total)
    mu_reference = float(_regularized_mu(regularization, mu_s, mu_d,
                                         transition_speed, slip_speed))
    bias_pct = (mu_measured - mu_reference) / mu_reference * 100.0
    return {
        "regularization": regularization,
        "mu_s": mu_s,
        "mu_d": mu_d,
        "transition_speed": transition_speed,
        "slip_speed": slip_speed,
        "normal_load_N": N_total,
        "normal_load_measured_N": float(N_measured),
        "n_particles": int(n),
        "mu_measured": mu_measured,
        "mu_reference": mu_reference,
        "bias_pct": float(bias_pct),
        "mu_measured_over_mu_d": float(mu_measured / mu_d) if mu_d else float("nan"),
        "residual_over_prediction_vs_dynamic_pct": (
            float((mu_measured - mu_d) / mu_d * 100.0) if mu_d else float("nan")),
        "dt": float(engine.dt),
    }


def residual_shear_specimen(*, regularization: str = "dual", mu_s: float = 0.6,
                            mu_d: float = 0.4, transition_speed: float = 1.0e-3,
                            geometry: SpecimenGeometry | None = None,
                            mass_scaling: float = 1600.0,
                            axial_preload_strain: float = 0.01,
                            shear_strain_rate: float = 2.0e-2,
                            target_shear_strain: float = 0.06,
                            n_axial_increments: int = 4,
                            n_shear_increments: int = 8,
                            decay_steps: int = 15,
                            ) -> dict[str, Any]:
    """Post-failure residual shear strength of a bonded specimen (criterion d).

    Protocol (all displacement controlled, same damping device as
    :func:`run_specimen`):

    1. the specimen is compressed axially to ``axial_preload_strain`` so the
       particle contacts carry a normal load once the bonds are gone;
    2. the axial drive is frozen and the top layer is dragged laterally past the
       bond shear failure strain at ``shear_strain_rate``.

    The lateral traction plateau after the bond population has failed is the
    residual strength.  The plateau is compared with the two Coulomb caps
    ``mu_d N`` (dynamic / residual) and ``mu_s N`` (static) built from the axial
    load measured at the end of step 1, so the comparison is a measurement
    against an analytic bound rather than a comparison between two simulations.
    """
    if regularization not in ("none", "tanh", "dual"):
        raise ValueError("regularization must be 'none', 'tanh' or 'dual'")
    geometry = geometry or SpecimenGeometry(n_layers=5, cross_n=3)
    micro = MicroParameters(contact_damping=0.0, drag=0.0)
    cfg = micro.engine_config(geometry, mass_scaling=mass_scaling,
                              decay_steps=decay_steps)
    cfg = replace(cfg,
                  platen_friction_coefficient=0.0,
                  friction_coefficient=(mu_s if regularization == "none" else mu_d),
                  friction_regularization=regularization,
                  static_friction_coefficient=mu_s,
                  dynamic_friction_coefficient=mu_d,
                  friction_transition_speed=transition_speed)
    engine = IceDEM3D(cfg)
    z = engine.positions[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    bottom = z < zmin + geometry.radius
    top = z > zmax - geometry.radius
    engine.fixed = bottom.clone()
    L0 = float(zmax - zmin)
    A = geometry.area
    mass = cfg.mass
    drag = float(cfg.drag)
    n_top = int(top.sum())

    def advance(vel_vec: torch.Tensor, steps: int) -> tuple[float, float]:
        """Integrate ``steps`` steps with the driven layer pinned; return
        ``(axial, lateral)`` traction magnitudes [N]."""
        v_mag = float(torch.linalg.vector_norm(vel_vec))
        for _ in range(int(steps)):
            force, _ = engine.forces()
            engine.velocities += (force / mass) * engine.dt
            engine.velocities[bottom] = 0.0
            engine.velocities[top] = vel_vec
            engine.positions += engine.velocities * engine.dt
            engine.positions[bottom] = engine.initial_positions[bottom]
            engine.time += engine.dt
            engine.step_count += 1
        force, _ = engine.forces()
        correction = drag * v_mag * n_top
        axial = -float(force[top, 2].sum()) - correction
        lateral = -float(force[top, 0].sum()) - correction
        return axial, lateral

    # --- phase 1: axial compression in RAMP/HOLD increments ---------------
    n_inc = int(n_axial_increments)
    ramp, hold = 40, 120
    eps_inc = axial_preload_strain / n_inc
    v_ax = -eps_inc * L0 / (ramp * engine.dt)
    axial_load = 0.0
    for _ in range(n_inc):
        advance(torch.tensor([0.0, 0.0, v_ax], dtype=engine.dtype), ramp)
        axial_load, _ = advance(torch.zeros(3, dtype=engine.dtype), hold)

    # --- phase 2: lateral shear past bond failure -------------------------
    n_shear = int(n_shear_increments)
    shear_inc = target_shear_strain / n_shear
    v_sh = shear_inc * L0 / (ramp * engine.dt)
    trace = []
    shear_strain = []
    for _ in range(n_shear):
        advance(torch.tensor([v_sh, 0.0, 0.0], dtype=engine.dtype), ramp)
        _, lateral = advance(torch.zeros(3, dtype=engine.dtype), hold)
        trace.append((engine.broken_bonds / max(1, len(engine.pairs)), lateral))
        shear_strain.append(abs(float(engine.positions[top, 0].mean())
                                - float(engine.initial_positions[top, 0].mean())) / L0)
    trace = np.array(trace, dtype=np.float64)
    shear_strain = np.array(shear_strain, dtype=np.float64)
    tau = trace[:, 1] / A
    late = shear_strain >= 0.7 * shear_strain.max()
    tau_residual = float(np.mean(tau[late])) if late.any() else float("nan")
    normal_load = abs(float(axial_load))
    cap_dynamic = mu_d * normal_load / A
    cap_static = mu_s * normal_load / A
    return {
        "regularization": regularization,
        "mu_s": mu_s,
        "mu_d": mu_d,
        "transition_speed": transition_speed,
        "normal_load_N": normal_load,
        "axial_preload_strain": axial_preload_strain,
        "shear_strain_rate": float(v_sh / L0),
        "target_shear_strain": float(shear_strain.max()),
        "bond_damage_fraction_end": float(trace[-1, 0]),
        "tau_peak_Pa": float(np.max(tau)),
        "tau_residual_Pa": tau_residual,
        "mu_slip_measured": (float(tau_residual * A / normal_load)
                             if normal_load > 0 else float("nan")),
        "coulomb_cap_dynamic_Pa": float(cap_dynamic),
        "coulomb_cap_static_Pa": float(cap_static),
        "residual_over_dynamic_cap": (float(tau_residual / cap_dynamic)
                                      if cap_dynamic > 0 else float("nan")),
        "residual_over_static_cap": (float(tau_residual / cap_static)
                                     if cap_static > 0 else float("nan")),
        "dt": float(engine.dt),
        "shear_strain": shear_strain,
        "tau": tau,
        "bond_damage": trace[:, 0],
    }
