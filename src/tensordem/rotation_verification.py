"""Verification harness for the rotational-DOF path of :class:`tensordem.dem3d.IceDEM3D`.

Purpose
-------
``IceDEM3D`` integrates translation only unless ``DEM3DConfig.enable_rotation``
is set.  Without spin state the solver cannot represent rolling resistance as a
torque, so the legacy code path turns ``T_r = mu_r |F_n| r_eff`` into an extra
*tangential force* couple.  This module builds the minimal contact configurations
whose response has a closed-form solution, so the rotational path can be checked
against rigid-body mechanics rather than against itself:

``build_plane_case``
    One free sphere resting on the infinite bottom platen, body weight supplied
    as an external nodal load.  Used for sliding-to-rolling transition, pure
    rolling deceleration and no-slip kinematics.
``build_spin_down_case``
    One sphere held against a fixed support sphere by a constant external load.
    Because the load balances the normal contact force the normal force is
    constant, so each rolling-resistance model has an exact spin-down solution.

Analytical references
---------------------
``slip_to_roll_speed_ratio``
    For a homogeneous sphere released with sliding speed ``v0`` and no spin,
    angular momentum about the (fixed) contact point is conserved, so the final
    rolling speed is ``v0 / (1 + I/(m R^2))`` = ``5/7 v0`` for a solid sphere.
``rolling_deceleration``
    Once rolling without slip is established, a rolling-resistance torque
    ``mu_r N R`` gives ``a = mu_r g / (1 + I/(m R^2))`` (a factor
    ``1 + I/(m R^2)`` smaller than the legacy force-surrogate prediction
    ``mu_r g``).
``cdt_spin``, ``viscous_spin``, ``epsd_spin``
    Exact spin histories for the three laws in
    :mod:`tensordem.rolling_resistance` under a constant normal force.

All quantities are SI.  The harness deliberately disables bonds (contact-only
assembly), drag and the prescribed indenter, so no unmodelled force enters the
comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .dem3d import DEM3DConfig, IceDEM3D
from .rolling_resistance import MODELS as ROLLING_MODELS

__all__ = [
    "PlaneCaseConfig",
    "SpinDownCaseConfig",
    "build_plane_case",
    "build_spin_down_case",
    "cdt_spin",
    "epsd_clamp_peak",
    "epsd_stop_time",
    "epsd_spin",
    "gravity_loads",
    "legacy_rolling_deceleration",
    "plane_static_penetration",
    "rolling_deceleration",
    "slip_to_roll_speed_ratio",
    "viscous_spin",
    "viscous_time_constant",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlaneCaseConfig:
    """Parameters of the single-sphere-on-platen verification case."""

    radius: float = 0.025
    density: float = 917.0
    platen_stiffness: float = 5.0e4
    dt: float = 2.0e-5
    mu_platen: float = 0.9
    mu_r: float = 0.05
    rolling_model: str = "cdt"
    rolling_omega_ref: float = 1.0
    rolling_stiffness: float = 0.0
    gravity: float = 9.81
    enable_rotation: bool = True

    def __post_init__(self) -> None:
        for name in ("radius", "density", "platen_stiffness", "dt", "gravity"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("mu_platen", "mu_r"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.rolling_model not in ROLLING_MODELS:
            raise ValueError(f"rolling_model must be one of {ROLLING_MODELS}")
        if not isinstance(self.enable_rotation, bool):
            raise ValueError("enable_rotation must be a boolean")

    @property
    def mass(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius ** 3 * self.density

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2


@dataclass(frozen=True)
class SpinDownCaseConfig:
    """Parameters of the constant-normal-force spin-down case."""

    radius: float = 0.025
    density: float = 917.0
    contact_stiffness: float = 5.0e3
    overlap: float = 1.0e-4
    dt: float = 1.0e-4
    mu_r: float = 0.05
    rolling_model: str = "cdt"
    rolling_omega_ref: float = 1.0
    rolling_stiffness: float = 0.0
    enable_rotation: bool = True

    def __post_init__(self) -> None:
        for name in ("radius", "density", "contact_stiffness", "overlap", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.mu_r) or self.mu_r < 0.0:
            raise ValueError("mu_r must be finite and nonnegative")
        if self.rolling_model not in ROLLING_MODELS:
            raise ValueError(f"rolling_model must be one of {ROLLING_MODELS}")
        if not isinstance(self.enable_rotation, bool):
            raise ValueError("enable_rotation must be a boolean")

    @property
    def mass(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius ** 3 * self.density

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    @property
    def normal_force(self) -> float:
        return self.contact_stiffness * self.overlap

    @property
    def effective_radius(self) -> float:
        """Contact effective radius of two equal spheres, ``R/2``."""
        return 0.5 * self.radius


# ---------------------------------------------------------------------------
# Analytical references
# ---------------------------------------------------------------------------
def slip_to_roll_speed_ratio(inertia_factor: float = 0.4) -> float:
    """Final speed of a released sliding sphere divided by its initial speed."""
    return 1.0 / (1.0 + inertia_factor)


def rolling_deceleration(mu_r: float, gravity: float, inertia_factor: float = 0.4) -> float:
    """Magnitude of the linear deceleration of a sphere rolling without slip."""
    return mu_r * gravity / (1.0 + inertia_factor)


def legacy_rolling_deceleration(mu_r: float, gravity: float) -> float:
    """Deceleration predicted by the legacy translational force surrogate."""
    return mu_r * gravity


def plane_static_penetration(mass: float, gravity: float, stiffness: float) -> float:
    """Static penetration depth of a sphere resting on a spring plane."""
    return mass * gravity / stiffness


def cdt_spin(omega0: float, mu_r: float, normal_force: float, effective_radius: float,
             inertia: float, time: float) -> float:
    """Exact spin history for the constant-directional-torque law (before reversal)."""
    torque = mu_r * normal_force * effective_radius
    return omega0 - torque / inertia * time


def viscous_time_constant(mu_r: float, normal_force: float, effective_radius: float,
                          inertia: float, omega_ref: float) -> float:
    """Decay time constant of the viscous rolling law."""
    return inertia * omega_ref / (mu_r * normal_force * effective_radius)


def viscous_spin(omega0: float, mu_r: float, normal_force: float, effective_radius: float,
                 inertia: float, omega_ref: float, time: float) -> float:
    """Exact spin history for the viscous rolling law."""
    return omega0 * math.exp(-time / viscous_time_constant(
        mu_r, normal_force, effective_radius, inertia, omega_ref))


def epsd_spin(omega0: float, stiffness: float, inertia: float, time: float) -> float:
    """Elastic-phase spin history of the EPSD rolling spring (harmonic)."""
    return omega0 * math.cos(math.sqrt(stiffness / inertia) * time)


def epsd_clamp_peak(omega0: float, stiffness: float, inertia: float) -> float:
    """Peak elastic torque magnitude, ``I omega0 sqrt(k_r / I)``."""
    return inertia * omega0 * math.sqrt(stiffness / inertia)


def epsd_stop_time(omega0: float, stiffness: float, inertia: float) -> float:
    """Time at which the unclamped elastic branch would bring the spin to zero."""
    return 0.5 * math.pi / math.sqrt(stiffness / inertia)


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------
def _isolate(dem: IceDEM3D, keep: int = 0, far: float = 5.0) -> None:
    """Keep particle ``keep``; park the others far away, fixed, and bond-free.

    The lattice of :class:`IceDEM3D` is created from a grid, so a contact-only
    case has to strip the bonds (``alive = False``) and remove the unused
    particles from any possible contact.
    """
    dem.alive.zero_()
    dem.fixed.zero_()
    spacing = 6.0 * dem.config.radius
    for index in range(len(dem.positions)):
        if index == keep:
            continue
        position = torch.tensor(
            [far, far + index * spacing, far + 0.5 * index * spacing],
            dtype=dem.dtype, device=dem.device)
        dem.positions[index] = position
        dem.initial_positions[index] = position
        dem.fixed[index] = True
    dem.velocities.zero_()
    if dem.omega is not None:
        dem.omega.zero_()


def _base_config(**overrides) -> DEM3DConfig:
    """A quiet 2x2x2 box: no bonds forces, no drag, no indenter, no platens."""
    params = dict(
        nx=2, ny=2, nz=2, geometry="box", fix_x_edges=False, fix_bottom=False,
        drag=0.0, contact_damping=0.0, restitution_coefficient=1.0,
        tool_speed=1.0e-9,
    )
    params.update(overrides)
    config = DEM3DConfig(**params)
    return config


def gravity_loads(dem: IceDEM3D, gravity: float) -> torch.Tensor:
    """Body weight as an external nodal load, zero on fixed particles."""
    loads = torch.zeros_like(dem.positions)
    active = ~dem.fixed
    loads[active, 2] = -dem.config.mass * gravity
    return loads


def build_plane_case(config: PlaneCaseConfig = PlaneCaseConfig(),
                     *, plane_z: float = 0.0) -> tuple[IceDEM3D, torch.Tensor]:
    """Build a single free sphere resting on the bottom platen.

    Returns the solver and the constant body-weight load vector to pass to
    ``step``/``forces``.  The sphere starts at the static equilibrium depth
    ``m g / k_platen`` so no normal-contact transient is excited.
    """
    dem_config = _base_config(
        radius=config.radius,
        density=config.density,
        dt=config.dt,
        platen_stiffness=config.platen_stiffness,
        platen_damping=0.0,
        bottom_platen_enabled=True,
        bottom_platen_gap=0.0,
        enable_rotation=config.enable_rotation,
        platen_friction_coefficient=config.mu_platen,
        rolling_resistance_coefficient=config.mu_r,
        rolling_model=config.rolling_model,
        rolling_omega_ref=config.rolling_omega_ref,
        rolling_stiffness=config.rolling_stiffness,
    )
    dem = IceDEM3D(dem_config)
    _isolate(dem)
    dem.tool_start = torch.tensor([0.0, 0.0, 1.0], dtype=dem.dtype, device=dem.device)
    dem.tool_velocity = torch.zeros(3, dtype=dem.dtype, device=dem.device)
    dem.bottom_platen_start_z = plane_z
    penetration = plane_static_penetration(config.mass, config.gravity,
                                           config.platen_stiffness)
    dem.positions[0] = torch.tensor(
        [0.0, 0.0, plane_z + config.radius - penetration],
        dtype=dem.dtype, device=dem.device)
    dem.initial_positions[0] = dem.positions[0].clone()
    dem.velocities.zero_()
    if dem.omega is not None:
        dem.omega.zero_()
    return dem, gravity_loads(dem, config.gravity)


def build_spin_down_case(config: SpinDownCaseConfig = SpinDownCaseConfig(),
                         spin: float = 10.0) -> tuple[IceDEM3D, torch.Tensor]:
    """Build a sphere spinning against a fixed support sphere at constant load.

    Particle 0 is a fixed support at the origin, particle 1 is the spinning
    sphere offset along ``+x`` by ``2R - overlap`` so the normal is ``+x``.  The
    returned load vector cancels the normal contact force on particle 1, which
    keeps the overlap (and therefore ``F_n``) constant without freezing the spin.
    The initial spin is about ``+z``, i.e. perpendicular to the contact normal,
    so it is a pure rolling component.
    """
    dem_config = _base_config(
        radius=config.radius,
        density=config.density,
        dt=config.dt,
        contact_stiffness=config.contact_stiffness,
        friction_coefficient=0.0,
        rolling_resistance_coefficient=config.mu_r,
        rolling_model=config.rolling_model,
        rolling_omega_ref=config.rolling_omega_ref,
        rolling_stiffness=config.rolling_stiffness,
        enable_rotation=config.enable_rotation,
    )
    dem = IceDEM3D(dem_config)
    _isolate(dem, keep=0)  # particle 0 = fixed support
    # Particle 1 becomes the spinning sphere; particle 0 stays at the origin.
    dem.positions[0] = torch.zeros(3, dtype=dem.dtype, device=dem.device)
    dem.initial_positions[0] = dem.positions[0].clone()
    dem.fixed[0] = True
    offset = 2.0 * config.radius - config.overlap
    dem.positions[1] = torch.tensor([offset, 0.0, 0.0], dtype=dem.dtype, device=dem.device)
    dem.initial_positions[1] = dem.positions[1].clone()
    dem.fixed[1] = False
    dem.tool_start = torch.tensor([0.0, 0.0, 1.0], dtype=dem.dtype, device=dem.device)
    dem.tool_velocity = torch.zeros(3, dtype=dem.dtype, device=dem.device)
    dem.velocities.zero_()
    if dem.omega is not None:
        dem.omega.zero_()
        dem.omega[1, 2] = spin
    loads = torch.zeros_like(dem.positions)
    loads[1, 0] = -config.normal_force  # balance the +x normal force on particle 1
    return dem, loads