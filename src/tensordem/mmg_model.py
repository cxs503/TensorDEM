"""MMG 4-DOF ship maneuvering model for ice-zone turning.

Simplified but physically-based model for campaign validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "ShipParticulars",
    "MMGConfig",
    "MMGState",
    "MMGSimulator",
    "empirical_turning_radius",
    "open_water_turning_radius",
]


@dataclass(frozen=True)
class ShipParticulars:
    length: float = 120.0
    beam: float = 20.0
    draught: float = 7.0
    displacement: float = 12000.0
    cb: float = 0.65

    @property
    def mass(self) -> float:
        return self.displacement * 1000.0


@dataclass(frozen=True)
class MMGConfig:
    ship: ShipParticulars = field(default_factory=ShipParticulars)
    water_density: float = 1025.0
    reference_speed: float = 3.0
    # Turning model parameters
    turning_rate_factor: float = 0.000156  # rad/s per degree of rudder at V_ref
    speed_loss_factor: float = 0.005     # speed loss per degree of rudder


def empirical_turning_radius(
    length: float, beam: float, speed: float,
    rudder_angle_deg: float, cb: float = 0.65,
) -> float:
    """Empirical turning radius [m]. R ≈ L / (C * sin(delta))."""
    delta_rad = math.radians(rudder_angle_deg)
    if delta_rad < 1e-6:
        return float("inf")
    C = 0.3 + 0.1 * cb
    return length / (C * math.sin(delta_rad))


def open_water_turning_radius(
    length: float, beam: float, speed: float,
    rudder_angle_deg: float, cb: float = 0.65,
) -> float:
    return empirical_turning_radius(length, beam, speed, rudder_angle_deg, cb)


@dataclass
class MMGState:
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    u: float = 3.0
    v: float = 0.0
    r: float = 0.0
    x_ice: float = 0.0
    y_ice: float = 0.0
    n_ice: float = 0.0
    delta: float = 0.0
    t: float = 0.0


class MMGSimulator:
    """Simplified MMG simulator for ice-zone turning validation."""

    def __init__(self, config: MMGConfig):
        self.config = config
        self.state = MMGState(u=config.reference_speed)
        self._m = config.ship.mass

    def set_ice_forces(self, x_ice: float, y_ice: float, n_ice: float) -> None:
        self.state.x_ice = float(x_ice)
        self.state.y_ice = float(y_ice)
        self.state.n_ice = float(n_ice)

    def set_rudder(self, delta_deg: float) -> None:
        self.state.delta = math.radians(float(delta_deg))

    def step(self, dt: float) -> None:
        dt = float(dt)
        cfg = self.config
        delta_deg = math.degrees(abs(self.state.delta))

        # Turning rate: proportional to rudder angle, reduced by ice forces
        ice_factor = max(0.3, 1.0 - abs(self.state.x_ice) / (cfg.ship.mass * 0.1))
        r_target = cfg.turning_rate_factor * delta_deg * ice_factor
        if self.state.delta < 0:
            r_target = -r_target

        # Smooth yaw rate response
        tau_r = 10.0  # time constant [s]
        self.state.r = self.state.r + dt / tau_r * (r_target - self.state.r)

        # Speed loss due to turning and ice
        turn_loss = cfg.speed_loss_factor * delta_deg * (1.0 - ice_factor)
        ice_resistance = abs(self.state.x_ice) / (cfg.ship.mass * 0.5)
        # Ice causes speed loss proportional to resistance
        ice_speed_loss = ice_resistance * 2.0
        u_target = cfg.reference_speed * (1.0 - turn_loss) - ice_speed_loss
        u_target = max(0.5, min(cfg.reference_speed, u_target))
        tau_u = 20.0
        self.state.u = self.state.u + dt / tau_u * (u_target - self.state.u)

        # Sway velocity (simplified)
        self.state.v = -self.state.u * math.sin(self.state.psi) * 0.1

        # Kinematics
        self.state.x += dt * self.state.u * math.cos(self.state.psi)
        self.state.y += dt * self.state.u * math.sin(self.state.psi)
        self.state.psi += dt * self.state.r
        self.state.t += dt

    def run(self, dt: float, n_steps: int, record_every: int = 1) -> Dict[str, list]:
        history = {"time": [], "x": [], "y": [], "psi": [], "u": [], "v": [], "r": [],
                   "speed": [], "heading_deg": []}
        for i in range(n_steps):
            self.step(dt)
            if (i + 1) % record_every == 0:
                history["time"].append(self.state.t)
                history["x"].append(self.state.x)
                history["y"].append(self.state.y)
                history["psi"].append(self.state.psi)
                history["u"].append(self.state.u)
                history["v"].append(self.state.v)
                history["r"].append(self.state.r)
                history["speed"].append(math.sqrt(self.state.u ** 2 + self.state.v ** 2))
                history["heading_deg"].append(math.degrees(self.state.psi) % 360)
        return history

    def compute_turning_radius(self, history: Dict[str, list]) -> float:
        x = np.array(history["x"])
        y = np.array(history["y"])
        n = len(x)
        start = int(0.5 * n)
        if start < 10:
            return float("nan")
        x_s, y_s = x[start:], y[start:]
        A = np.column_stack([x_s, y_s, np.ones(len(x_s))])
        b = -(x_s ** 2 + y_s ** 2)
        try:
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            xc, yc = -0.5 * sol[0], -0.5 * sol[1]
            radius = math.sqrt(max(0, xc ** 2 + yc ** 2 - sol[2]))
            return float(radius)
        except Exception:
            return float("nan")

    def compute_track_width(self, history: Dict[str, list]) -> float:
        y = np.array(history["y"])
        return float(np.max(y) - np.min(y))

    def compute_speed_loss(self, history: Dict[str, list]) -> float:
        speeds = np.array(history["speed"])
        if len(speeds) < 2:
            return 0.0
        v0 = self.config.reference_speed
        # Use minimum speed during the simulation (steady-state may recover)
        v_min = float(np.min(speeds))
        return max(0.0, (v0 - v_min) / v0)
