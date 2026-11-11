"""DEM-based Couette shear cell for ice-water slurry.

A simplified 2D periodic shear cell that measures the effective viscosity
of an ice-water slurry by imposing a shear rate and measuring the resulting
shear stress. The carrier fluid is modelled as a Stokes drag background
(one-way coupling).

Stress computation uses the Irving-Kirkwood (virial) formula from particle
interactions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "IceSlurryCouette",
    "run_couette_sweep",
]

RHO_ICE = 917.0
RHO_WATER = 1000.0
G = 9.81


@dataclass
class IceSlurryCouette:
    """2D periodic Couette shear cell for ice-water slurry."""
    H: float = 0.05
    L: float = 0.10
    d: float = 0.005
    phi: float = 0.20
    rho_p: float = RHO_ICE
    rho_f: float = RHO_WATER
    mu_f: float = 1.787e-3
    mu_contact: float = 0.3
    kn: float = 1.0e5
    kt_ratio: float = 0.5
    rolling_friction: float = 0.05
    dt: float = 1.0e-5
    seed: int = 42
    wall_velocity: float = 0.5

    pos: Optional[np.ndarray] = field(default=None, repr=False)
    vel: Optional[np.ndarray] = field(default=None, repr=False)
    ang_vel: Optional[np.ndarray] = field(default=None, repr=False)
    angle: Optional[np.ndarray] = field(default=None, repr=False)
    n_particles: int = field(default=0, repr=False)
    t: float = 0.0
    _tau_samples: List[float] = field(default_factory=list, repr=False)
    _mass_total: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.phi <= 0.0 or self.phi >= 0.75:
            raise ValueError(f"phi={self.phi} must be in (0, 0.75)")
        if self.d <= 0.0:
            raise ValueError("d must be positive")
        if self.H <= 0.0 or self.L <= 0.0:
            raise ValueError("H and L must be positive")

    @property
    def r(self) -> float:
        return self.d / 2.0

    @property
    def shear_rate(self) -> float:
        return self.wall_velocity / self.H

    @property
    def n_target(self) -> int:
        area_p = math.pi * self.r ** 2
        area_c = self.L * self.H
        return max(1, int(self.phi * area_c / area_p))

    def initialize(self) -> None:
        """Place particles on a hexagonal lattice with slight overlap."""
        n = self.n_target
        self.n_particles = n
        rng = np.random.RandomState(self.seed)

        # Hexagonal lattice with spacing = 0.95 * d (ensures contacts)
        spacing = 0.95 * self.d
        nx = max(1, int(self.L / spacing))
        ny = max(1, int(self.H / (spacing * math.sqrt(3) / 2)))

        positions = []
        for iy in range(ny):
            for ix in range(nx):
                if len(positions) >= n:
                    break
                x = (ix + 0.5 * (iy % 2)) * spacing
                y = iy * spacing * math.sqrt(3) / 2
                # Add small random perturbation
                x += rng.uniform(-0.02 * self.d, 0.02 * self.d)
                y += rng.uniform(-0.02 * self.d, 0.02 * self.d)
                x = max(self.r + 1e-9, min(self.L - self.r - 1e-9, x))
                y = max(self.r + 1e-9, min(self.H - self.r - 1e-9, y))
                positions.append([x, y])

        # Trim or pad to exact n
        positions = positions[:n]
        while len(positions) < n:
            x = rng.uniform(self.r, self.L - self.r)
            y = rng.uniform(self.r, self.H - self.r)
            positions.append([x, y])

        self.pos = np.array(positions[:n], dtype=np.float64)
        self.vel = np.zeros((n, 2), dtype=np.float64)
        self.ang_vel = np.zeros(n, dtype=np.float64)
        self.angle = np.zeros(n, dtype=np.float64)
        self.t = 0.0
        self._tau_samples = []
        self._mass_total = float(n) * self.rho_p * math.pi * self.r ** 2

    def _pair_forces(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute contact forces and virial shear stress."""
        n = self.n_particles
        fx = np.zeros(n, dtype=np.float64)
        fy = np.zeros(n, dtype=np.float64)
        torque = np.zeros(n, dtype=np.float64)
        virial_xy = 0.0
        volume = self.L * self.H

        kt = self.kn * self.kt_ratio
        mu = self.mu_contact
        mu_r = self.rolling_friction

        for i in range(n):
            for j in range(i + 1, n):
                dx = self.pos[j, 0] - self.pos[i, 0]
                dy = self.pos[j, 1] - self.pos[i, 1]
                if dx > self.L / 2:
                    dx -= self.L
                elif dx < -self.L / 2:
                    dx += self.L
                dist = math.sqrt(dx * dx + dy * dy)
                overlap = self.d - dist
                if overlap <= 0:
                    continue

                nx_c = dx / dist
                ny_c = dy / dist
                fn = self.kn * overlap

                v_rel_x = self.vel[j, 0] - self.vel[i, 0]
                v_rel_y = self.vel[j, 1] - self.vel[i, 1]
                v_tan = v_rel_x * (-ny_c) + v_rel_y * nx_c
                ft = kt * v_tan * self.dt
                ft_max = mu * fn
                ft = max(-ft_max, min(ft_max, ft))

                fxi = fn * nx_c - ft * ny_c
                fyi = fn * ny_c + ft * nx_c
                fx[i] += fxi
                fy[i] += fyi
                fx[j] -= fxi
                fy[j] -= fyi

                virial_xy -= (fxi * dy + fyi * (-dx)) / volume

                omega_rel = self.ang_vel[j] - self.ang_vel[i]
                tr = mu_r * fn * self.r
                t_val = min(tr, max(-tr, kt * omega_rel * self.dt * self.r))
                torque[i] -= t_val
                torque[j] += t_val

        return fx, fy, torque, virial_xy

    def _fluid_forces(self) -> Tuple[np.ndarray, np.ndarray]:
        """Stokes drag from background fluid with linear shear profile."""
        n = self.n_particles
        fx = np.zeros(n, dtype=np.float64)
        fy = np.zeros(n, dtype=np.float64)
        gamma_dot = self.shear_rate
        drag_coeff = 6.0 * math.pi * self.mu_f * self.r
        for i in range(n):
            u_fx = gamma_dot * self.pos[i, 1]
            fx[i] = drag_coeff * (u_fx - self.vel[i, 0])
            fy[i] = drag_coeff * (0.0 - self.vel[i, 1])
        return fx, fy

    def _wall_forces(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wall contact forces (penalty method)."""
        n = self.n_particles
        fx = np.zeros(n, dtype=np.float64)
        fy = np.zeros(n, dtype=np.float64)
        torque = np.zeros(n, dtype=np.float64)
        kt = self.kn * self.kt_ratio
        mu = self.mu_contact
        for i in range(n):
            dist_bottom = self.pos[i, 1] - self.r
            if dist_bottom < 0:
                overlap = -dist_bottom
                fn = self.kn * overlap
                fy[i] += fn
                v_tan = self.vel[i, 0] + self.ang_vel[i] * self.r
                ft = kt * v_tan * self.dt
                ft = max(-mu * fn, min(mu * fn, ft))
                fx[i] -= ft
                torque[i] -= ft * self.r
            dist_top = self.H - self.r - self.pos[i, 1]
            if dist_top < 0:
                overlap = -dist_top
                fn = self.kn * overlap
                fy[i] -= fn
                v_tan = (self.vel[i, 0] - self.wall_velocity) - self.ang_vel[i] * self.r
                ft = kt * v_tan * self.dt
                ft = max(-mu * fn, min(mu * fn, ft))
                fx[i] -= ft
                torque[i] -= ft * self.r
        return fx, fy, torque

    def step(self) -> None:
        n = self.n_particles
        dt = self.dt
        mass = self.rho_p * math.pi * self.r ** 2
        inertia = 0.5 * mass * self.r ** 2
        g_eff = G * (1.0 - self.rho_f / self.rho_p)

        cx, cy, ct, virial_xy = self._pair_forces()
        ffx, ffy = self._fluid_forces()
        wx, wy, wt = self._wall_forces()

        for i in range(n):
            ax = (cx[i] + ffx[i] + wx[i]) / mass
            ay = (cy[i] + ffy[i] + wy[i]) / mass - g_eff
            self.vel[i, 0] += ax * dt
            self.vel[i, 1] += ay * dt
            self.pos[i, 0] += self.vel[i, 0] * dt
            self.pos[i, 1] += self.vel[i, 1] * dt
            if self.pos[i, 0] < 0:
                self.pos[i, 0] += self.L
            elif self.pos[i, 0] >= self.L:
                self.pos[i, 0] -= self.L
            self.pos[i, 1] = max(self.r, min(self.H - self.r, self.pos[i, 1]))
            ang_acc = (ct[i] + wt[i]) / inertia
            self.ang_vel[i] += ang_acc * dt
            self.angle[i] += self.ang_vel[i] * dt

        self.t += dt
        self._tau_samples.append(virial_xy)

    def results(self) -> Dict[str, float]:
        if not self._tau_samples:
            return {"tau_Pa": 0.0, "gamma_dot_1_s": self.shear_rate,
                    "eta_eff_Pa_s": 0.0, "n_samples": 0, "mass_kg": self._mass_total}
        tau_mean = float(np.mean(self._tau_samples))
        gamma_dot = self.shear_rate
        eta_eff = abs(tau_mean) / gamma_dot if gamma_dot > 0 else 0.0
        return {
            "tau_Pa": abs(tau_mean),
            "gamma_dot_1_s": gamma_dot,
            "eta_eff_Pa_s": eta_eff,
            "n_samples": len(self._tau_samples),
            "mass_kg": self._mass_total,
            "phi": self.phi,
            "d_m": self.d,
        }

    def run(self, duration: float, sample_start: float = 0.0,
            sample_interval: float = 1e-3) -> Dict[str, float]:
        self.initialize()
        n_steps = int(duration / self.dt)
        self._tau_samples = []
        for _ in range(n_steps):
            self.step()
        return self.results()


def run_couette_sweep(
    phi_values: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    shear_rates: Sequence[float] = (5.0, 10.0, 20.0, 40.0, 80.0),
    d: float = 0.005,
    mu_f: float = 1.787e-3,
    mu_contact: float = 0.3,
    duration: float = 0.05,
    sample_start: float = 0.02,
    seed: int = 42,
) -> List[Dict[str, float]]:
    results = []
    for phi in phi_values:
        for gamma_dot in shear_rates:
            H = 0.05
            cell = IceSlurryCouette(
                H=H, L=0.10, d=d, phi=phi,
                mu_f=mu_f, mu_contact=mu_contact,
                wall_velocity=gamma_dot * H,
                dt=5e-6, seed=seed,
            )
            res = cell.run(duration, sample_start)
            res["phi"] = phi
            res["gamma_dot_target"] = gamma_dot
            results.append(res)
    return results
