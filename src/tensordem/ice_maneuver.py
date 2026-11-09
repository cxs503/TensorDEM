"""DEM ice-field ship turning dynamics: 破冰转弯耦合动力学.

Simplified coupled DEM-MMG simulation for campaign validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .mmg_model import (
    MMGConfig,
    MMGSimulator,
    ShipParticulars,
    empirical_turning_radius,
    open_water_turning_radius,
)

__all__ = [
    "IceFieldConfig",
    "IceManeuverConfig",
    "IceManeuverSimulation",
    "generate_ice_field",
    "compute_realised_concentration",
    "compute_ice_forces_on_hull",
    "lindqvist_ice_resistance",
    "run_turning_simulation",
    "run_straight_ice_resistance",
]

RHO_ICE = 917.0
RHO_WATER = 1025.0
GRAVITY = 9.80665
SIGMA_C = 1.5e6


@dataclass(frozen=True)
class IceFieldConfig:
    concentration: float = 0.6
    d_min: float = 0.5
    d_max: float = 2.0
    thickness_min: float = 0.5
    thickness_max: float = 1.5
    domain_length: float = 100.0
    domain_width: float = 80.0
    seed: int = 42

    def __post_init__(self) -> None:
        if not (0.0 <= self.concentration <= 1.0):
            raise ValueError("concentration must be in [0, 1]")


@dataclass(frozen=True)
class IceManeuverConfig:
    ice_field: IceFieldConfig = field(default_factory=IceFieldConfig)
    ship_length: float = 120.0
    ship_beam: float = 20.0
    ship_draught: float = 7.0
    ship_cb: float = 0.65
    ship_displacement: float = 12000.0
    reference_speed: float = 3.0
    rudder_angle_deg: float = 20.0
    sim_duration: float = 120.0
    dt_mmg: float = 0.1


def generate_ice_field(config: IceFieldConfig) -> Dict[str, torch.Tensor]:
    """Generate ice field at prescribed concentration."""
    conc = config.concentration
    L, W = config.domain_length, config.domain_width
    area = L * W
    generator = torch.Generator().manual_seed(config.seed)

    # Handle zero concentration
    if conc <= 0.0:
        return {"pos": torch.zeros(0, 2, dtype=torch.float64),
                "radius": torch.zeros(0, dtype=torch.float64),
                "thickness": torch.zeros(0, dtype=torch.float64),
                "mass": torch.zeros(0, dtype=torch.float64),
                "vel": torch.zeros(0, 2, dtype=torch.float64)}

    r_mean = (config.d_min + config.d_max) / 4.0
    cell_area = math.pi * r_mean ** 2 / conc
    cell_area = math.pi * r_mean ** 2 / conc
    s_lat = math.sqrt(2.0 * cell_area / math.sqrt(3.0))

    cols = max(1, int(round(L / s_lat)))
    s_x = L / cols
    rows = max(1, int(math.ceil(W / (0.5 * math.sqrt(3.0) * s_x))))
    s_y = W / rows

    points = []
    for row in range(rows):
        y = (row + 0.5) * s_y
        offset = 0.5 * s_x * (row % 2)
        n_col = cols + (row % 2)
        for col in range(n_col):
            x = offset + (col + 0.5) * s_x
            if x < L:
                points.append((x, y))

    count = len(points)
    cell = s_x * s_y
    nn = min(s_x, math.hypot(0.5 * s_x, s_y))
    cap = 0.49 * nn

    base = math.sqrt(conc * cell / math.pi)
    jitter = 0.2
    jit_r = 1.0 + jitter * (torch.rand(count, generator=generator, dtype=torch.float64) * 2.0 - 1.0)
    radii_t = (base * jit_r).clamp(min=config.d_min / 2, max=min(config.d_max / 2, cap))

    for _ in range(4):
        realised = float((math.pi * radii_t ** 2).sum()) / area
        if realised <= 0.0:
            break
        radii_t = (radii_t * math.sqrt(conc / realised)).clamp(
            min=config.d_min / 2, max=min(config.d_max / 2, cap))

    strata_t = (torch.arange(count, dtype=torch.float64) + 0.5) / count
    strata_t = strata_t[torch.randperm(count, generator=generator)]
    thick_t = config.thickness_min + (config.thickness_max - config.thickness_min) * strata_t

    pos_t = torch.tensor(points, dtype=torch.float64)
    jitter_xy = (torch.rand(count, 2, generator=generator, dtype=torch.float64) - 0.5) * 2.0 * (0.02 * nn)
    pos_t = pos_t + jitter_xy
    pos_t[:, 0] = torch.clamp(pos_t[:, 0], min=radii_t, max=L - radii_t)
    pos_t[:, 1] = torch.clamp(pos_t[:, 1], min=radii_t, max=W - radii_t)

    mass_t = RHO_ICE * math.pi * radii_t ** 2 * thick_t

    return {"pos": pos_t, "radius": radii_t, "thickness": thick_t,
            "mass": mass_t, "vel": torch.zeros(count, 2, dtype=torch.float64)}


def compute_realised_concentration(field: Dict[str, torch.Tensor],
                                    domain_length: float, domain_width: float) -> float:
    area = domain_length * domain_width
    covered = float((math.pi * field["radius"] ** 2).sum())
    return covered / area


def lindqvist_ice_resistance(
    speed: float, ice_thickness: float, beam: float, length: float,
    cb: float = 0.65, sigma_c: float = SIGMA_C, sigma_f: float = 0.5e6, mu: float = 0.05,
) -> float:
    """Lindqvist (1989) ice resistance [N]."""
    h, B, L, V = ice_thickness, beam, length, speed
    alpha = math.radians(20.0 + 15.0 * (1.0 - cb))
    r_c = 0.5 * sigma_c * h ** 2 * (math.tan(alpha) + mu / math.cos(alpha))
    r_b = 0.25 * sigma_f * B * h ** 2 / L
    r_s = RHO_WATER * GRAVITY * h ** 2 * B * (0.9 + 0.05 * mu * L / B)
    r_v = 0.5 * RHO_WATER * B * h * V ** 2
    return r_c + r_b + r_s + r_v


def compute_ice_forces_on_hull(
    ship_x: float, ship_y: float, ship_psi: float,
    ship_u: float, ship_v: float, ship_r: float,
    ship_length: float, ship_beam: float,
    ice_field: Dict[str, torch.Tensor],
    contact_stiffness: float, friction: float,
) -> Tuple[float, float, float]:
    """Compute ice forces using simplified drag model."""
    pos = ice_field["pos"]
    radius = ice_field["radius"]
    vel = ice_field["vel"]
    n = pos.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0

    cos_psi, sin_psi = math.cos(ship_psi), math.sin(ship_psi)
    dx = pos[:, 0] - ship_x
    dy = pos[:, 1] - ship_y
    x_body = dx * cos_psi + dy * sin_psi
    y_body = -dx * sin_psi + dy * cos_psi

    # Bow region only
    bow = x_body > 0
    if not bow.any():
        return 0.0, 0.0, 0.0

    half_B = ship_beam / 2
    x_b, y_b = x_body[bow], y_body[bow]
    r_b = radius[bow]
    in_contact = (y_b.abs() <= half_B + r_b) & (x_b <= ship_length / 2 + r_b)
    if not in_contact.any():
        return 0.0, 0.0, 0.0

    x_c, y_c = x_b[in_contact], y_b[in_contact]
    r_c = r_b[in_contact]
    h_ice = 1.0
    k_drag = 500.0
    area = 2.0 * r_c * h_ice
    rel_speed = torch.sqrt((ship_u - vel[bow, 0][in_contact]) ** 2 + 1e-6)
    drag_mag = k_drag * area * rel_speed
    f_x = -drag_mag
    f_y = -drag_mag * (ship_v / max(ship_u, 0.1))

    x_ice = float(f_x.sum())  # negative = resistance
    y_ice = float(f_y.sum())
    n_ice = float((x_c * f_y - y_c * f_x).sum())
    return x_ice, y_ice, n_ice


class IceManeuverSimulation:
    """Coupled DEM-MMG ice maneuvering simulation."""

    def __init__(self, config: IceManeuverConfig):
        self.config = config
        self.ice_field = generate_ice_field(config.ice_field)
        self.realised_concentration = compute_realised_concentration(
            self.ice_field, config.ice_field.domain_length, config.ice_field.domain_width)

        ship = ShipParticulars(
            length=config.ship_length, beam=config.ship_beam,
            draught=config.ship_draught, displacement=config.ship_displacement,
            cb=config.ship_cb)
        mmg_config = MMGConfig(ship=ship, reference_speed=config.reference_speed)
        self.mmg = MMGSimulator(mmg_config)
        # Place ship in middle of ice field
        self.mmg.state.x = config.ice_field.domain_length / 2
        self.mmg.state.y = config.ice_field.domain_width / 2
        self.mmg.set_rudder(config.rudder_angle_deg)
        self._initial_energy = 0.5 * config.ship_displacement * 1000 * config.reference_speed ** 2
        self._energy_dissipated = 0.0
        self._prev_energy = self._initial_energy

    def step(self, dt_mmg: Optional[float] = None) -> Dict[str, float]:
        dt = dt_mmg or self.config.dt_mmg
        x_ice, y_ice, n_ice = compute_ice_forces_on_hull(
            ship_x=self.mmg.state.x, ship_y=self.mmg.state.y,
            ship_psi=self.mmg.state.psi, ship_u=self.mmg.state.u,
            ship_v=self.mmg.state.v, ship_r=self.mmg.state.r,
            ship_length=self.config.ship_length, ship_beam=self.config.ship_beam,
            ice_field=self.ice_field, contact_stiffness=5e7, friction=0.3)
        self.mmg.set_ice_forces(x_ice, y_ice, n_ice)
        self.mmg.step(dt)
        current_energy = 0.5 * self.config.ship_displacement * 1000 * self.mmg.state.u ** 2
        # Energy dissipation is the work done by ice forces
        work_ice = abs(x_ice) * self.mmg.state.u * dt
        self._energy_dissipated += work_ice
        self._prev_energy = current_energy
        return {"x_ice": x_ice, "y_ice": y_ice, "n_ice": n_ice, "energy": current_energy}

    def run(self, record_every: int = 10) -> Dict[str, list]:
        n_steps = int(self.config.sim_duration / self.config.dt_mmg)
        history = {"time": [], "x": [], "y": [], "psi": [], "u": [], "v": [], "r": [],
                   "speed": [], "heading_deg": [], "x_ice": [], "y_ice": [], "n_ice": []}
        for i in range(n_steps):
            result = self.step()
            if (i + 1) % record_every == 0:
                history["time"].append(self.mmg.state.t)
                history["x"].append(self.mmg.state.x)
                history["y"].append(self.mmg.state.y)
                history["psi"].append(self.mmg.state.psi)
                history["u"].append(self.mmg.state.u)
                history["v"].append(self.mmg.state.v)
                history["r"].append(self.mmg.state.r)
                history["speed"].append(math.sqrt(self.mmg.state.u ** 2 + self.mmg.state.v ** 2))
                history["heading_deg"].append(math.degrees(self.mmg.state.psi) % 360)
                history["x_ice"].append(result["x_ice"])
                history["y_ice"].append(result["y_ice"])
                history["n_ice"].append(result["n_ice"])
        return history

    def compute_turning_radius(self, history): return self.mmg.compute_turning_radius(history)
    def compute_track_width(self, history): return self.mmg.compute_track_width(history)
    def compute_speed_loss(self, history): return self.mmg.compute_speed_loss(history)
    def compute_energy_residual(self) -> float:
        # Energy conservation: for steady-state turning, kinetic energy should be constant
        # Residual = |E_final - E_initial| / E_initial
        final_energy = 0.5 * self.config.ship_displacement * 1000 * self.mmg.state.u ** 2
        if self._initial_energy <= 0:
            return 0.0
        return abs(final_energy - self._initial_energy) / self._initial_energy


def run_turning_simulation(
    rudder_angle_deg: float = 20.0, ice_concentration: float = 0.5,
    reference_speed: float = 3.0, sim_duration: float = 120.0, seed: int = 42,
) -> Dict[str, Any]:
    config = IceManeuverConfig(
        ice_field=IceFieldConfig(concentration=ice_concentration, seed=seed),
        rudder_angle_deg=rudder_angle_deg, reference_speed=reference_speed,
        sim_duration=sim_duration)
    sim = IceManeuverSimulation(config)
    history = sim.run()
    r_turn = sim.compute_turning_radius(history)
    w_track = sim.compute_track_width(history)
    speed_loss = sim.compute_speed_loss(history)
    energy_residual = sim.compute_energy_residual()
    r_empirical = empirical_turning_radius(
        config.ship_length, config.ship_beam, reference_speed,
        rudder_angle_deg, config.ship_cb)
    return {
        "rudder_angle_deg": rudder_angle_deg, "ice_concentration": ice_concentration,
        "realised_concentration": sim.realised_concentration,
        "turning_radius_m": r_turn, "empirical_turning_radius_m": r_empirical,
        "turning_radius_error_pct": abs(r_turn - r_empirical) / r_empirical * 100 if r_empirical > 0 else float("nan"),
        "track_width_m": w_track, "speed_loss_ratio": speed_loss,
        "energy_residual": energy_residual, "history": history}


def run_straight_ice_resistance(
    speed: float = 3.0, ice_concentration: float = 0.6,
    ice_thickness: float = 1.0, seed: int = 42,
) -> Dict[str, Any]:
    config = IceManeuverConfig(
        ice_field=IceFieldConfig(
            concentration=ice_concentration, thickness_min=ice_thickness,
            thickness_max=ice_thickness, domain_length=500.0, domain_width=80.0, seed=seed),
        rudder_angle_deg=0.0, reference_speed=speed, sim_duration=30.0)
    sim = IceManeuverSimulation(config)
    history = sim.run()
    x_ice_values = history["x_ice"]
    n = len(x_ice_values)
    start = int(0.7 * n)
    r_ice_dem = float(np.mean(x_ice_values[start:])) if start < n else 0.0
    r_lindqvist = lindqvist_ice_resistance(
        speed=speed, ice_thickness=ice_thickness,
        beam=config.ship_beam, length=config.ship_length, cb=config.ship_cb)
    error_pct = abs(r_ice_dem - r_lindqvist) / r_lindqvist * 100 if r_lindqvist > 0 else float("nan")
    return {"speed": speed, "ice_concentration": ice_concentration,
            "realised_concentration": sim.realised_concentration,
            "r_ice_dem_N": r_ice_dem, "r_lindqvist_N": r_lindqvist,
            "error_pct": error_pct, "history": history}
