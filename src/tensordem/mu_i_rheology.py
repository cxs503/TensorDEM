"""μ(I) rheology benchmark for TensorDEM granular flow.

Physical background
-------------------
Dense granular flows exhibit a rate-dependent friction law known as the μ(I)
rheology (Jop, Forterre & Chaucha, Nature 2006; GDR MiDi, JFM 2004):

    μ(I) = μ_s + (μ_2 - μ_s) / (I_0 / I + 1)

where:
    μ(I)  : effective friction coefficient (τ/P)
    I     : inertial number = γ̇ d / √(P/ρ_p)
    μ_s   : static friction coefficient (I → 0 limit)
    μ_2   : dynamic friction coefficient (I → ∞ limit)
    I_0   : characteristic inertial number (typically ~0.3)
    γ̇    : shear rate (1/s)
    d     : particle diameter (m)
    P     : confining pressure (Pa)
    ρ_p   : particle material density (kg/m³)

Implementation
--------------
A direct shear box is simulated: a rectangular cell filled with granular
particles, with periodic boundary conditions in the shear direction (x),
rigid walls in the transverse direction (y), and a moving top wall that
applies a constant shear rate γ̇ while maintaining a constant normal stress P.

All quantities are SI (m, kg, s, Pa).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

DTYPE = torch.float64


class MuIRheologyError(ValueError):
    """Base class for μ(I) rheology errors."""


@dataclass(frozen=True)
class MuICase:
    radius: float = 0.005
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.05
    restitution: float = 0.3
    kn: float = 1.0e5
    kt_ratio: float = 0.5
    box_width: float = 0.06
    box_depth: float = 0.03
    fill_height: float = 0.04
    shear_rate: float = 5.0
    normal_stress: float = 5000.0
    packing_gap: float = 0.02
    jitter: float = 0.01
    dt: Optional[float] = None
    seed: int = 5279
    max_time: float = 2.0
    sample_interval: int = 50

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

    @property
    def particle_diameter(self) -> float:
        return 2.0 * self.radius

    def __post_init__(self) -> None:
        for name in ("radius", "density", "kn", "kt_ratio",
                     "box_width", "box_depth", "fill_height",
                     "shear_rate", "normal_stress", "max_time"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value > 0.0):
                raise MuIRheologyError(f"{name} must be finite and positive")
        for name in ("mu", "mu_rolling", "restitution", "packing_gap", "jitter"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0.0):
                raise MuIRheologyError(f"{name} must be finite and nonnegative")
        if not 0.0 <= self.restitution < 1.0:
            raise MuIRheologyError("restitution must lie in [0, 1)")
        if self.dt is not None and (not math.isfinite(self.dt) or self.dt <= 0.0):
            raise MuIRheologyError("dt must be finite and positive")


def _hcp_packing_in_box(case: MuICase) -> torch.Tensor:
    r = case.radius
    spacing = 2.0 * r * (1.0 + case.packing_gap)
    layer_dz = spacing * math.sqrt(2.0 / 3.0)
    n_layers = max(1, int(math.floor((case.fill_height - r) / layer_dz)) + 1)
    generator = torch.Generator().manual_seed(case.seed)
    row_pitch = spacing * math.sqrt(3.0) / 2.0
    shift_x = 0.5 * spacing
    shift_y = spacing / (2.0 * math.sqrt(3.0))
    points: list[tuple[float, float, float]] = []
    x_min = r + 0.001
    x_max = case.box_width - r - 0.001
    y_min = r + 0.001
    y_max = case.box_depth - r - 0.001
    for layer in range(n_layers):
        z = r + layer * layer_dz
        odd = layer % 2 == 1
        span_x = int(math.ceil((case.box_width + spacing) / spacing))
        span_y = int(math.ceil((case.box_depth + spacing) / spacing))
        for ix in range(-span_x, span_x + 1):
            for iy in range(-span_y, span_y + 1):
                x = ix * spacing + 0.5 * spacing * (iy % 2)
                y = iy * row_pitch
                if odd:
                    x += shift_x
                    y += shift_y
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    points.append((x, y, z))
    positions = torch.tensor(points, dtype=DTYPE)
    if len(positions) < 8:
        raise MuIRheologyError("packing produced fewer than 8 particles")
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


def _safe_normalize(v: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    norm = torch.linalg.vector_norm(v, dim=1, keepdim=True)
    return torch.where(norm > eps, v / norm, torch.zeros_like(v))


def simulate_shear(case: MuICase, *, return_history: bool = False) -> dict:
    positions = _hcp_packing_in_box(case)
    n = len(positions)
    r = case.radius
    mass = case.particle_mass
    inertia = case.inertia
    dt = case.timestep
    kn, kt = case.kn, case.kt
    zeta = _damping_ratio(case.restitution)
    gamma_damp = 2.0 * zeta * math.sqrt(kn * 0.5 * mass)
    gamma_wall = 2.0 * zeta * math.sqrt(kn * mass)

    velocities = torch.zeros_like(positions)
    omega = torch.zeros_like(positions)

    box_w = case.box_width
    box_d = case.box_depth
    gamma_dot = case.shear_rate
    top_z = case.fill_height
    top_vx = gamma_dot * top_z

    steps = int(math.ceil(case.max_time / dt))
    history: list[tuple[float, float, float, float]] = []
    settle_steps = min(500, steps // 4)

    for step in range(steps):
        # Periodic BC in x
        positions[:, 0] = torch.remainder(positions[:, 0], box_w)

        force = torch.zeros_like(positions)
        torque = torch.zeros_like(positions)

        # Particle-particle contacts with periodic x
        # all distinct pairs (i < j), the upstream ordering of the pair forces
        aiu, aju = _pair_index(n)
        dvec = positions[aju] - positions[aiu]
        # Apply minimum image in x
        dvec[:, 0] = dvec[:, 0] - box_w * torch.round(dvec[:, 0] / box_w)
        dist = torch.linalg.vector_norm(dvec, dim=1)
        overlap = 2.0 * r - dist
        touching = overlap > 0.0

        if touching.any():
            t_idx = torch.nonzero(touching, as_tuple=False).flatten()
            n_vec = _safe_normalize(dvec[t_idx])
            v_rel = velocities[aju[t_idx]] - velocities[aiu[t_idx]]
            v_n = (v_rel * n_vec).sum(dim=1)
            fn = torch.clamp(kn * overlap[t_idx] - gamma_damp * v_n, min=0.0)

            v_t = v_rel - v_n[:, None] * n_vec
            if case.mu > 0:
                ft_mag = case.mu * fn
                ft_dir = _safe_normalize(-v_t)
                ft = ft_mag[:, None] * ft_dir
            else:
                ft = torch.zeros_like(v_t)

            f_total = fn[:, None] * n_vec + ft
            force.index_add_(0, aiu[t_idx], f_total)
            force.index_add_(0, aju[t_idx], -f_total)

            if case.mu_rolling > 0:
                omega_rel = omega[aju[t_idx]] - omega[aiu[t_idx]]
                tr_mag = case.mu_rolling * r * fn
                tr_dir = _safe_normalize(-omega_rel)
                tr = tr_mag[:, None] * tr_dir
                torque.index_add_(0, aiu[t_idx], tr)
                torque.index_add_(0, aju[t_idx], -tr)

        # Side walls (y direction)
        y_pen_left = r - positions[:, 1]
        y_pen_right = positions[:, 1] + r - box_d
        for pen, sign in [(y_pen_left, 1.0), (y_pen_right, -1.0)]:
            hit = pen > 0.0
            if hit.any():
                idx = torch.nonzero(hit, as_tuple=False).flatten()
                normal = torch.tensor([[0.0, sign, 0.0]], dtype=DTYPE).expand(len(idx), 3)
                v_n = (velocities[idx] * normal).sum(dim=1)
                fn_w = torch.clamp(kn * pen[idx] - gamma_wall * v_n, min=0.0)
                force.index_add_(0, idx, fn_w[:, None] * normal)
                if case.mu > 0:
                    v_t = velocities[idx].clone()
                    v_t[:, 1] = 0.0
                    ft_w = case.mu * fn_w
                    ft_dir = _safe_normalize(-v_t)
                    force.index_add_(0, idx, ft_w[:, None] * ft_dir)

        # Top wall
        top_pen = positions[:, 2] + r - top_z
        top_hit = top_pen > 0.0
        top_force_z = 0.0
        top_force_x = 0.0
        if top_hit.any():
            idx = torch.nonzero(top_hit, as_tuple=False).flatten()
            normal = torch.tensor([[0.0, 0.0, -1.0]], dtype=DTYPE).expand(len(idx), 3)
            v_n = (velocities[idx] * normal).sum(dim=1)
            fn_w = torch.clamp(kn * top_pen[idx] - gamma_wall * v_n, min=0.0)
            force.index_add_(0, idx, fn_w[:, None] * normal)
            top_force_z = float(fn_w.sum())
            if case.mu > 0:
                v_rel_x = velocities[idx, 0] - top_vx
                ft_w = case.mu * fn_w
                ft_dir_x = torch.where(v_rel_x > 0, -1.0, 1.0)
                fx = ft_w * ft_dir_x
                force.index_add_(0, idx, fx[:, None] * torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE))
                top_force_x = float(fx.sum())

        # Bottom wall
        bottom_pen = r - positions[:, 2]
        bottom_hit = bottom_pen > 0.0
        if bottom_hit.any():
            idx = torch.nonzero(bottom_hit, as_tuple=False).flatten()
            normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE).expand(len(idx), 3)
            v_n = velocities[idx, 2]
            fn_w = torch.clamp(kn * bottom_pen[idx] - gamma_wall * v_n, min=0.0)
            force.index_add_(0, idx, fn_w[:, None] * normal)
            if case.mu > 0:
                v_t = velocities[idx].clone()
                v_t[:, 2] = 0.0
                ft_w = case.mu * fn_w
                ft_dir = _safe_normalize(-v_t)
                force.index_add_(0, idx, ft_w[:, None] * ft_dir)

        # Gravity
        force[:, 2] -= mass * 9.81

        # Semi-implicit Euler
        velocities = velocities + force * (dt / mass)
        omega = omega + torque * (dt / inertia)
        positions = positions + velocities * dt

        # Sample stress
        if step % case.sample_interval == 0 and step > settle_steps:
            area = box_w * box_d
            tau = top_force_x / area if area > 0 else 0.0
            P = top_force_z / area if area > 0 else 0.0
            mu_eff = tau / P if P > 0 else 0.0
            history.append((t_now := step * dt, tau, P, mu_eff))

    # Compute averages
    if len(history) >= 3:
        n_steady = max(1, int(0.6 * len(history)))
        steady = history[-n_steady:]
        avg_tau = sum(h[1] for h in steady) / n_steady
        avg_P = sum(h[2] for h in steady) / n_steady
        avg_mu = sum(h[3] for h in steady) / n_steady
    else:
        avg_tau = avg_P = avg_mu = 0.0

    d = 2.0 * case.radius
    rho_p = case.density
    I = gamma_dot * d / math.sqrt(avg_P / rho_p) if avg_P > 0 else 0.0

    mu_s = case.mu
    mu_2 = 0.7
    I_0 = 0.3
    mu_theoretical = mu_s + (mu_2 - mu_s) / (I_0 / I + 1) if I > 0 else mu_s

    result = {
        "n_particles": n,
        "mu_effective": avg_mu,
        "shear_stress_Pa": avg_tau,
        "normal_stress_Pa": avg_P,
        "shear_rate_imposed": gamma_dot,
        "inertial_number": I,
        "mu_theoretical": mu_theoretical,
        "mu_s": mu_s, "mu_2": mu_2, "I_0": I_0,
        "dt": dt, "steps": steps,
        "mu": case.mu, "mu_rolling": case.mu_rolling,
        "radius": r, "density": case.density,
    }
    if return_history:
        result["history"] = history
    return result


def jop_mu_I(I: float, mu_s: float, mu_2: float, I_0: float) -> float:
    if I <= 0:
        return mu_s
    return mu_s + (mu_2 - mu_s) / (I_0 / I + 1)


def fit_jop_law(I_values: list[float], mu_values: list[float]) -> dict[str, float]:
    import numpy as np
    I_arr = np.array(I_values)
    mu_arr = np.array(mu_values)
    best_rmse = float("inf")
    best_params = {}
    for mu_s in np.arange(0.2, 0.5, 0.02):
        for mu_2 in np.arange(0.5, 0.9, 0.02):
            for I_0 in np.arange(0.1, 0.6, 0.05):
                mu_pred = np.array([jop_mu_I(I, mu_s, mu_2, I_0) for I in I_arr])
                rmse = float(np.sqrt(np.mean((mu_arr - mu_pred) ** 2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {"mu_s": mu_s, "mu_2": mu_2, "I_0": I_0, "rmse": rmse}
    return best_params


__all__ = [
    "MuIRheologyError", "MuICase", "simulate_shear",
    "jop_mu_I", "fit_jop_law",
]
