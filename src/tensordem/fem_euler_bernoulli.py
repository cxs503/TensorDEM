"""Euler-Bernoulli beam FEM solver for DEM-FEM one-way coupling.

Why this module exists
----------------------
The DEM-FEM one-way coupling example needs a lightweight, analytically-verifiable
FEM beam solver.  This module provides a 2-node Hermite beam element assembly
with simply-supported or clamped boundary conditions, accepting arbitrary
time-varying nodal loads (supplied by the DEM side of the coupling).

All quantities are SI.  The beam axis is x, deflection is w (positive downward).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "EulerBernoulliBeam",
    "BeamConfig",
    "beam_center_deflection_analytic",
    "beam_center_deflection_static",
]


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def beam_center_deflection_analytic(
    load: float, length: float, young_modulus: float, inertia: float,
    *, bc: str = "simply_supported",
) -> float:
    """Analytical center deflection of an Euler-Bernoulli beam under central point load.

    Simply-supported:  delta = P L^3 / (48 E I)
    Clamped-clamped:   delta = P L^3 / (192 E I)
    Cantilever (load at tip): delta = P L^3 / (3 E I)
    """
    P = _finite("load", load)
    L = _finite("length", length, positive=True)
    E = _finite("young_modulus", young_modulus, positive=True)
    I = _finite("inertia", inertia, positive=True)
    EI = E * I
    if bc == "simply_supported":
        return P * L ** 3 / (48.0 * EI)
    elif bc == "clamped":
        return P * L ** 3 / (192.0 * EI)
    elif bc == "cantilever":
        return P * L ** 3 / (3.0 * EI)
    else:
        raise ValueError(f"unknown bc={bc!r}")


def beam_center_deflection_static(
    load: float, length: float, young_modulus: float, inertia: float,
    *, bc: str = "simply_supported",
) -> float:
    """Alias for :func:`beam_center_deflection_analytic`."""
    return beam_center_deflection_analytic(load, length, young_modulus, inertia, bc=bc)


@dataclass(frozen=True)
class BeamConfig:
    """Euler-Bernoulli beam parameters."""
    length: float = 1.0
    width: float = 0.1
    height: float = 0.01
    young_modulus: float = 2.1e11
    density: float = 7850.0
    n_elements: int = 20
    bc: str = "simply_supported"
    damping_ratio: float = 0.0

    def __post_init__(self) -> None:
        _finite("length", self.length, positive=True)
        _finite("width", self.width, positive=True)
        _finite("height", self.height, positive=True)
        _finite("young_modulus", self.young_modulus, positive=True)
        _finite("density", self.density, positive=True)
        if isinstance(self.n_elements, bool) or not isinstance(self.n_elements, int) or self.n_elements < 1:
            raise ValueError("n_elements must be an integer >= 1")
        if self.bc not in ("simply_supported", "clamped"):
            raise ValueError("bc must be 'simply_supported' or 'clamped'")
        if not (0.0 <= self.damping_ratio < 1.0):
            raise ValueError("damping_ratio must be in [0, 1)")

    @property
    def inertia(self) -> float:
        return self.width * self.height ** 3 / 12.0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def mass_per_length(self) -> float:
        return self.density * self.area


class EulerBernoulliBeam:
    """Dynamic Euler-Bernoulli beam FEM solver with Newmark time integration.

    Uses the constant average acceleration method (beta=1/4, gamma=1/2),
    which is unconditionally stable for linear systems.
    """

    def __init__(self, config: BeamConfig = BeamConfig(), *, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.float64

        n_elem = config.n_elements
        n_node = n_elem + 1
        n_dof = 2 * n_node

        h = config.length / n_elem
        EI = config.young_modulus * config.inertia
        rhoA = config.mass_per_length

        # Element stiffness
        ke = EI / h ** 3 * torch.tensor([
            [12.0, 6.0 * h, -12.0, 6.0 * h],
            [6.0 * h, 4.0 * h * h, -6.0 * h, 2.0 * h * h],
            [-12.0, -6.0 * h, 12.0, -6.0 * h],
            [6.0 * h, 2.0 * h * h, -6.0 * h, 4.0 * h * h],
        ], dtype=self.dtype, device=self.device)

        # Consistent mass
        me = rhoA * h / 420.0 * torch.tensor([
            [156.0, 22.0 * h, 54.0, -13.0 * h],
            [22.0 * h, 4.0 * h * h, 13.0 * h, -3.0 * h * h],
            [54.0, 13.0 * h, 156.0, -22.0 * h],
            [-13.0 * h, -3.0 * h * h, -22.0 * h, 4.0 * h * h],
        ], dtype=self.dtype, device=self.device)

        # Assemble
        K = torch.zeros(n_dof, n_dof, dtype=self.dtype, device=self.device)
        M = torch.zeros(n_dof, n_dof, dtype=self.dtype, device=self.device)

        for e in range(n_elem):
            dof_idx = torch.tensor([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3],
                                   dtype=torch.long, device=self.device)
            K[dof_idx[:, None], dof_idx[None, :]] += ke
            M[dof_idx[:, None], dof_idx[None, :]] += me

        # Boundary conditions
        self._fixed_dofs = torch.empty(0, dtype=torch.long, device=self.device)
        if config.bc == "simply_supported":
            self._fixed_dofs = torch.tensor([0, 2 * n_elem], dtype=torch.long, device=self.device)
        elif config.bc == "clamped":
            self._fixed_dofs = torch.tensor([0, 1, 2 * n_elem, 2 * n_elem + 1],
                                            dtype=torch.long, device=self.device)

        free_mask = torch.ones(n_dof, dtype=torch.bool, device=self.device)
        free_mask[self._fixed_dofs] = False
        self._free_idx = torch.arange(n_dof, device=self.device)[free_mask]

        self.K_free = K[self._free_idx[:, None], self._free_idx[None, :]]
        self.M_free = M[self._free_idx[:, None], self._free_idx[None, :]]

        # Rayleigh damping
        if config.damping_ratio > 0.0:
            # Use scipy-style generalized eigenvalue via Cholesky
            L = torch.linalg.cholesky(self.M_free)
            LinvK = torch.linalg.solve_triangular(L, self.K_free, upper=False)
            A = torch.linalg.solve_triangular(L.T, LinvK.T, upper=True).T
            eigvals = torch.linalg.eigvalsh(A)
            omega1 = torch.sqrt(eigvals[0])
            zeta = config.damping_ratio
            beta_damp = 2.0 * zeta / float(omega1)
            self.C_free = beta_damp * self.K_free
        else:
            self.C_free = torch.zeros_like(self.K_free)

        # Newmark: constant average acceleration
        self._beta = 0.25
        self._gamma = 0.5

        # Newmark coefficients (constant)
        self._a0 = 1.0 / (self._beta)  # will be divided by dt^2 at runtime
        self._a1 = self._gamma / self._beta  # will be divided by dt at runtime
        self._a2 = 1.0 / self._beta
        self._a3 = self._gamma / self._beta - 1.0
        self._a4 = self._gamma / (2.0 * self._beta) - 1.0

        # State
        self.n_dof = n_dof
        self.n_node = n_node
        self.h_elem = h
        self.disp = torch.zeros(n_dof, dtype=self.dtype, device=self.device)
        self.vel = torch.zeros(n_dof, dtype=self.dtype, device=self.device)
        self.acc = torch.zeros(n_dof, dtype=self.dtype, device=self.device)
        self.time = 0.0
        self.step_count = 0

        self._force_history: list[torch.Tensor] = []
        self.x_nodes = torch.linspace(0.0, config.length, n_node,
                                      dtype=self.dtype, device=self.device)

    def static_solve(self, force_vector: torch.Tensor) -> dict[str, Any]:
        """Solve K * d = F statically."""
        if force_vector.shape[0] != self.n_dof:
            raise ValueError(f"force_vector must have {self.n_dof} entries")
        f = force_vector.to(device=self.device, dtype=self.dtype)
        f_free = f[self._free_idx]
        d_free = torch.linalg.solve(self.K_free, f_free)
        d_full = torch.zeros(self.n_dof, dtype=self.dtype, device=self.device)
        d_full[self._free_idx] = d_free
        w = d_full[0::2]
        theta = d_full[1::2]
        return {
            "deflection": w.detach().cpu().numpy().tolist(),
            "rotation": theta.detach().cpu().numpy().tolist(),
            "center_deflection": float(w[self.n_node // 2]),
        }

    def apply_load(self, force_vector: torch.Tensor, dt: float) -> dict[str, Any]:
        """Apply nodal force and advance one Newmark step.

        Standard Newmark-beta formulation:
            K_eff * d_{n+1} = F_eff
        where
            K_eff = K + gamma/(beta*dt)*C + 1/(beta*dt^2)*M
            F_eff = F_{n+1} + M*(a0*d_n + a1*v_n + a2*a_n) + C*(a3*d_n + a4*v_n + a5*a_n)
        with
            a0 = 1/(beta*dt^2), a1 = 1/(beta*dt), a2 = 1/(2*beta) - 1
            a3 = gamma/(beta*dt), a4 = gamma/beta - 1, a5 = dt*(gamma/(2*beta) - 1)
        """
        if force_vector.shape[0] != self.n_dof:
            raise ValueError(f"force_vector must have {self.n_dof} entries")

        f = force_vector.to(device=self.device, dtype=self.dtype)
        f_free = f[self._free_idx]

        # Current state (at step n)
        d_n = self.disp[self._free_idx]
        v_n = self.vel[self._free_idx]
        a_n = self.acc[self._free_idx]

        # Newmark coefficients for this dt
        a0 = 1.0 / (self._beta * dt ** 2)
        a1 = 1.0 / (self._beta * dt)
        a2 = 1.0 / (2.0 * self._beta) - 1.0
        a3 = self._gamma / (self._beta * dt)
        a4 = self._gamma / self._beta - 1.0
        a5 = dt * (self._gamma / (2.0 * self._beta) - 1.0)

        # Effective stiffness
        K_eff = self.K_free + a3 * self.C_free + a0 * self.M_free

        # Effective force
        f_eff = (f_free
                 + self.M_free @ (a0 * d_n + a1 * v_n + a2 * a_n)
                 + self.C_free @ (a3 * d_n + a4 * v_n + a5 * a_n))

        # Solve for d_{n+1}
        d_new_free = torch.linalg.solve(K_eff, f_eff)

        # Acceleration at n+1
        a_new = a0 * (d_new_free - d_n) - a1 * v_n - a2 * a_n

        # Velocity at n+1
        v_new = v_n + dt * ((1.0 - self._gamma) * a_n + self._gamma * a_new)

        # Update state
        new_disp = self.disp.clone()
        new_disp[self._free_idx] = d_new_free
        new_vel = self.vel.clone()
        new_vel[self._free_idx] = v_new
        new_acc = self.acc.clone()
        new_acc[self._free_idx] = a_new

        self.disp = new_disp
        self.vel = new_vel
        self.acc = new_acc
        self.time += dt
        self.step_count += 1

        self._force_history.append(f.clone())

        return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        w = self.disp[0::2]
        theta = self.disp[1::2]
        v_w = self.vel[0::2]
        return {
            "time": float(self.time),
            "step": self.step_count,
            "deflection": w.detach().cpu().numpy().tolist(),
            "rotation": theta.detach().cpu().numpy().tolist(),
            "velocity": v_w.detach().cpu().numpy().tolist(),
            "center_deflection": float(w[self.n_node // 2]),
        }

    def center_deflection(self) -> float:
        return float(self.disp[0::2][self.n_node // 2].item())

    def center_velocity(self) -> float:
        return float(self.vel[0::2][self.n_node // 2].item())

    def kinetic_energy(self) -> float:
        v_full = self.vel.clone()
        return float(0.5 * (v_full @ self._full_mass() @ v_full).item())

    def strain_energy(self) -> float:
        d_full = self.disp.clone()
        return float(0.5 * (d_full @ self._full_stiffness() @ d_full).item())

    def _full_mass(self) -> torch.Tensor:
        M = torch.zeros(self.n_dof, self.n_dof, dtype=self.dtype, device=self.device)
        M[self._free_idx[:, None], self._free_idx[None, :]] = self.M_free
        return M

    def _full_stiffness(self) -> torch.Tensor:
        K = torch.zeros(self.n_dof, self.n_dof, dtype=self.dtype, device=self.device)
        K[self._free_idx[:, None], self._free_idx[None, :]] = self.K_free
        return K

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.strain_energy()

    def reset(self) -> None:
        self.disp.zero_()
        self.vel.zero_()
        self.acc.zero_()
        self.time = 0.0
        self.step_count = 0
        self._force_history.clear()
