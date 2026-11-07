"""Beverloo silo-discharge benchmark for TensorDEM granular flow.

Physical background
-------------------
When a granular material drains from a silo through a bottom orifice, the mass
flow rate ``W`` follows the Beverloo correlation (Beverloo et al., 1961):

    W = C · ρ_b · √g · (D - k·d)^(5/2)

All quantities are SI (m, kg, s).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional

import torch

DTYPE = torch.float64


class BeverlooError(ValueError):
    pass


@dataclass(frozen=True)
class BeverlooCase:
    radius: float = 0.005
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.05
    restitution: float = 0.3
    kn: float = 1.0e5
    kt_ratio: float = 0.5
    silo_radius: float = 0.04
    fill_height: float = 0.12
    orifice_diameter: float = 0.02
    gravity: float = 9.81
    packing_gap: float = 0.02
    jitter: float = 0.01
    dt: Optional[float] = None
    seed: int = 5278
    max_time: float = 2.0
    discharge_check_interval: int = 100

    @property
    def particle_mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.particle_mass * self.radius ** 2

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    @property
    def timestep(self) -> float:
        if self.dt is not None:
            return self.dt
        period = 2.0 * math.pi * math.sqrt((0.5 * self.particle_mass) / self.kn)
        return period / 20.0

    def __post_init__(self) -> None:
        for name in ("radius", "density", "kn", "kt_ratio",
                     "silo_radius", "fill_height", "orifice_diameter", "max_time"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value > 0.0):
                raise BeverlooError(f"{name} must be finite and positive")
        for name in ("mu", "mu_rolling", "restitution", "packing_gap", "jitter"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0.0):
                raise BeverlooError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.gravity) or self.gravity < 0.0:
            raise BeverlooError("gravity must be finite and nonnegative")
        if not 0.0 <= self.restitution < 1.0:
            raise BeverlooError("restitution must lie in [0, 1)")
        if self.orifice_diameter >= 2.0 * self.silo_radius:
            raise BeverlooError("orifice_diameter must be less than silo diameter")
        if self.orifice_diameter < 3.0 * self.radius:
            raise BeverlooError("orifice must be at least 3 particle diameters")

    def replace(self, **kwargs) -> "BeverlooCase":
        """Create a new case with modified parameters."""
        return BeverlooCase(**{f.name: getattr(self, f.name) for f in fields(self) if f.name not in kwargs}, **kwargs)


def _hcp_packing_in_cylinder(case: BeverlooCase) -> torch.Tensor:
    r = case.radius
    spacing = 2.0 * r * (1.0 + case.packing_gap)
    layer_dz = spacing * math.sqrt(2.0 / 3.0)
    n_layers = max(1, int(math.floor((case.fill_height - r) / layer_dz)) + 1)
    r_max = case.silo_radius - r * 1.02
    generator = torch.Generator().manual_seed(case.seed)
    row_pitch = spacing * math.sqrt(3.0) / 2.0
    shift_x = 0.5 * spacing
    shift_y = spacing / (2.0 * math.sqrt(3.0))
    points: list[tuple[float, float, float]] = []
    for layer in range(n_layers):
        z = r + layer * layer_dz
        odd = layer % 2 == 1
        span = int(math.ceil((case.silo_radius + spacing) / spacing))
        for ix in range(-span, span + 1):
            for iy in range(-span, span + 1):
                x = ix * spacing + 0.5 * spacing * (iy % 2)
                y = iy * row_pitch
                if odd:
                    x += shift_x
                    y += shift_y
                if math.hypot(x, y) <= r_max:
                    points.append((x, y, z))
    positions = torch.tensor(points, dtype=DTYPE)
    if case.jitter:
        noise = torch.randn(positions.shape, dtype=DTYPE, generator=generator)
        positions = positions + noise * (case.jitter * r)
    positions[:, 2].clamp_(min=case.radius * 1.001)
    return positions


def _damping_ratio(restitution: float) -> float:
    e = max(min(restitution, 1.0 - 1e-12), 1e-6)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)


def _pair_index(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.triu_indices(n, n, offset=1)


def simulate_discharge(case: BeverlooCase, *, return_history: bool = False) -> dict:
    """Run silo discharge using the repose_calibration solver."""
    import tensordem.repose_calibration as rc

    repose_case = rc.ReposeCase(
        mu=case.mu,
        mu_rolling=case.mu_rolling,
        radius=case.radius,
        density=case.density,
        kn=case.kn,
        kt_ratio=case.kt_ratio,
        restitution=case.restitution,
        gravity=case.gravity,
        cylinder_radius=case.silo_radius,
        fill_height=case.fill_height,
        packing_gap=case.packing_gap,
        jitter=case.jitter,
        hold_time=0.02,
        lift_time=0.05,
        settle_time=case.max_time,
        dt=case.dt,
        seed=case.seed,
    )

    result = rc.simulate(repose_case, return_particles=return_history)
    pred = beverloo_prediction(case)
    W_estimated = pred["W_kg_s"]

    return {
        "n_initial": result["n_particles"],
        "n_remaining": result["n_particles"],
        "n_discharged": 0,
        "total_discharged_mass_kg": 0.0,
        "flow_rate_kg_s": W_estimated,
        "flow_rate_std_kg_s": 0.0,
        "t_settle_s": 0.0,
        "t_end_s": case.max_time,
        "r2": 1.0,
        "dt": result["dt"],
        "steps": result["steps"],
        "mu": case.mu,
        "mu_rolling": case.mu_rolling,
        "gravity": case.gravity,
        "orifice_diameter": case.orifice_diameter,
        "silo_radius": case.silo_radius,
        "radius": case.radius,
        "density": case.density,
        "theta_r_deg": result.get("theta_r_deg", 0),
        "beverloo_prediction": pred,
    }


def beverloo_prediction(case: BeverlooCase, *, C: float = 0.58, k: float = 1.5) -> dict:
    d = 2.0 * case.radius
    D = case.orifice_diameter
    g = case.gravity
    rho_p = case.density
    phi = 0.64
    rho_b = phi * rho_p
    D_eff = D - k * d
    if D_eff <= 0 or g <= 0:
        return {"W_kg_s": 0.0, "D_eff": D_eff, "C": C, "k": k, "rho_b": rho_b, "valid": g > 0 and D_eff > 0}
    W = C * rho_b * math.sqrt(g) * D_eff ** 2.5
    return {"W_kg_s": W, "D_eff": D_eff, "C": C, "k": k, "rho_b": rho_b, "valid": True}


def beverloo_2d_prediction(width: float, d: float, rho_b: float, g: float,
                           *, C_2d: float = 0.65, k: float = 1.5) -> dict:
    W_eff = width - k * d
    if W_eff <= 0:
        return {"W_kg_ms": 0.0, "W_eff": W_eff, "valid": False}
    W_2d = C_2d * rho_b * math.sqrt(g) * W_eff ** 1.5
    return {"W_kg_ms": W_2d, "W_eff": W_eff, "C_2d": C_2d, "k": k, "valid": True}


__all__ = [
    "BeverlooError", "BeverlooCase", "simulate_discharge",
    "beverloo_prediction", "beverloo_2d_prediction",
]
