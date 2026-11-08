"""Euler-Bernoulli beam FEM solver for DEM-FEM one-way coupling.

Why this module exists
----------------------
The DEM-FEM one-way coupling example needs a lightweight, analytically-verifiable
FEM beam solver.  This module provides a 2-node Hermite beam element assembly
with simply-supported or clamped boundary conditions, accepting arbitrary
time-varying nodal loads (supplied by the DEM side of the coupling).

All quantities are SI.  The beam axis is x, deflection is w (positive downward).
Nodal degrees of freedom are (w, theta) with theta = dw/dx in radians
(consistent with the cubic Hermite interpolation used by both the stiffness and
the consistent mass matrix).

Supported boundary conditions: ``simply_supported`` (w=0 at both ends, rotations
free), ``clamped`` (w=theta=0 at both ends) and ``cantilever`` (w=theta=0 at
x=0, both free at x=L).  The deflection field between nodes is recovered by
:meth:`EulerBernoulliBeam.deflection_at`, so meshes with an odd number of
elements (no node exactly at mid-span) still report the mid-span value.
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
    "beam_tip_deflection_analytic",
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
    """Analytical deflection of an Euler-Bernoulli beam under a central point load.

    Simply-supported:  delta = P L^3 / (48 E I)   (mid-span)
    Clamped-clamped:   delta = P L^3 / (192 E I)  (mid-span)

    ``bc="cantilever"`` is kept for backward compatibility but the cantilever is
    *not* loaded at mid-span: the returned value is the **tip** deflection of a
    cantilever under a tip point load, delta = P L^3 / (3 E I).  Use
    :func:`beam_tip_deflection_analytic` for that case.
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


def beam_tip_deflection_analytic(
    load: float, length: float, young_modulus: float, inertia: float,
    *, bc: str = "cantilever",
) -> float:
    """Analytical tip deflection of an Euler-Bernoulli beam under a tip point load.

    Cantilever (load at free end): delta = P L^3 / (3 E I)
    """
    P = _finite("load", load)
    L = _finite("length", length, positive=True)
    E = _finite("young_modulus", young_modulus, positive=True)
    I = _finite("inertia", inertia, positive=True)
    if bc != "cantilever":
        raise ValueError(f"tip-load analytic solution only defined for bc='cantilever', got {bc!r}")
    return P * L ** 3 / (3.0 * E * I)


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
        if self.bc not in ("simply_supported", "clamped", "cantilever"):
            raise ValueError(
                "bc must be 'simply_supported', 'clamped' or 'cantilever'")
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
        elif config.bc == "cantilever":
            # clamped at x=0, free at x=L
            self._fixed_dofs = torch.tensor([0, 1], dtype=torch.long,
                                            device=self.device)

        free_mask = torch.ones(n_dof, dtype=torch.bool, device=self.device)
        free_mask[self._fixed_dofs] = False
        self._free_idx = torch.arange(n_dof, device=self.device)[free_mask]

        self._K_full = K
        self._M_full = M
        self.K_free = K[self._free_idx[:, None], self._free_idx[None, :]]
        self.M_free = M[self._free_idx[:, None], self._free_idx[None, :]]

        # Generalized eigenproblem K phi = omega^2 M phi, reduced to the
        # symmetric standard form A = L^-1 K L^-T with M = L L^T, using a
        # Cholesky factorization of the (free) consistent mass matrix.
        # NOTE: the reduction must be A = L^-1 K L^-T (symmetric, positive
        # semi-definite); a spurious extra transpose produces L^-1 K L^-1 and
        # negative eigenvalues, which previously turned any damped run into NaN.
        L_chol = torch.linalg.cholesky(self.M_free)
        LinvK = torch.linalg.solve_triangular(L_chol, self.K_free, upper=False)
        self._A_eig = torch.linalg.solve_triangular(L_chol, LinvK.T, upper=False)
        self._eigvals = torch.linalg.eigvalsh(self._A_eig)
        self._omegas = torch.sqrt(torch.clamp(self._eigvals, min=0.0))

        # Rayleigh damping
        if config.damping_ratio > 0.0:
            omega1 = float(self._omegas[0])
            if not math.isfinite(omega1) or omega1 <= 0.0:
                raise ValueError(
                    "damping_ratio requires a positive fundamental frequency; "
                    "check mass/stiffness assembly and boundary conditions")
            zeta = config.damping_ratio
            # Stiffness-proportional Rayleigh term: C = (2 zeta / omega_1) K
            beta_damp = 2.0 * zeta / omega1
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

    def _interpolate_deflection(self, d: torch.Tensor, x: float) -> float:
        """Cubic Hermite interpolation of the deflection field at position x."""
        length = float(self.config.length)
        xf = _finite("x", x)
        tol = 1.0e-12 * length
        if xf < -tol or xf > length + tol:
            raise ValueError(f"x={xf!r} outside beam domain [0, {length}]")
        xf = min(max(xf, 0.0), length)
        n_elem = int(self.config.n_elements)
        h = self.h_elem
        e = min(int(xf // h), n_elem - 1)
        xi = (xf - e * h) / h
        n1 = 1.0 - 3.0 * xi ** 2 + 2.0 * xi ** 3
        n2 = xi - 2.0 * xi ** 2 + xi ** 3
        n3 = 3.0 * xi ** 2 - 2.0 * xi ** 3
        n4 = -xi ** 2 + xi ** 3
        w_l = float(d[2 * e])
        t_l = float(d[2 * e + 1])
        w_r = float(d[2 * e + 2])
        t_r = float(d[2 * e + 3])
        return n1 * w_l + h * n2 * t_l + n3 * w_r + h * n4 * t_r

    def deflection_at(self, x: float) -> float:
        """Current deflection at physical position x (cubic Hermite interpolation)."""
        return self._interpolate_deflection(self.disp, x)

    def consistent_point_load(self, x: float, load: float) -> torch.Tensor:
        """Equivalent nodal force vector for a transverse point load at position x.

        The load is distributed onto the two nodes of the element containing x
        with the cubic Hermite shape functions, which is the variationally
        consistent load vector for Euler-Bernoulli elements.  At a node the
        result reduces exactly to a single nodal force.
        """
        length = float(self.config.length)
        xf = _finite("x", x)
        P = _finite("load", load)
        tol = 1.0e-12 * length
        if xf < -tol or xf > length + tol:
            raise ValueError(f"x={xf!r} outside beam domain [0, {length}]")
        xf = min(max(xf, 0.0), length)
        n_elem = int(self.config.n_elements)
        h = self.h_elem
        e = min(int(xf // h), n_elem - 1)
        xi = (xf - e * h) / h
        n1 = 1.0 - 3.0 * xi ** 2 + 2.0 * xi ** 3
        n2 = xi - 2.0 * xi ** 2 + xi ** 3
        n3 = 3.0 * xi ** 2 - 2.0 * xi ** 3
        n4 = -xi ** 2 + xi ** 3
        f = torch.zeros(self.n_dof, dtype=self.dtype, device=self.device)
        f[2 * e] += P * n1
        f[2 * e + 1] += P * h * n2
        f[2 * e + 2] += P * n3
        f[2 * e + 3] += P * h * n4
        return f

    def static_solve(
        self, force_vector: torch.Tensor, *, commit: bool = False,
    ) -> dict[str, Any]:
        """Solve K * d = F statically.

        By default the result is returned without touching the dynamic state
        (this keeps :meth:`static_solve` free of side effects and lets callers
        use it as a pure probe).  Pass ``commit=True`` to load the static
        equilibrium into the dynamic state (velocity and acceleration reset to
        zero), which is the usual restart-from-equilibrium pattern.
        """
        if force_vector.shape[0] != self.n_dof:
            raise ValueError(f"force_vector must have {self.n_dof} entries")
        f = force_vector.to(device=self.device, dtype=self.dtype)
        f_free = f[self._free_idx]
        d_free = torch.linalg.solve(self.K_free, f_free)
        d_full = torch.zeros(self.n_dof, dtype=self.dtype, device=self.device)
        d_full[self._free_idx] = d_free
        w = d_full[0::2]
        theta = d_full[1::2]
        center = self._interpolate_deflection(d_full, 0.5 * float(self.config.length))
        if commit:
            self.disp = d_full.clone()
            self.vel.zero_()
            self.acc.zero_()
            self._static_load = f.clone()
        return {
            "deflection": w.detach().cpu().numpy().tolist(),
            "rotation": theta.detach().cpu().numpy().tolist(),
            "center_deflection": center,
            "reaction": self._reaction(f, d_full),
        }

    def _reaction(self, f: torch.Tensor, d_full: torch.Tensor) -> list[float]:
        """Support reactions R = K d - F at the constrained degrees of freedom."""
        K = self._full_stiffness()
        r = (K @ d_full) - f
        return [float(r[i]) for i in self._fixed_dofs.tolist()]

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
            "center_deflection": self.deflection_at(0.5 * float(self.config.length)),
        }

    def center_deflection(self) -> float:
        """Mid-span deflection (interpolated, valid for odd element counts too)."""
        return self.deflection_at(0.5 * float(self.config.length))

    def center_velocity(self) -> float:
        """Mid-span velocity (interpolated, valid for odd element counts too)."""
        return self._interpolate_deflection(self.vel, 0.5 * float(self.config.length))

    def kinetic_energy(self) -> float:
        v_full = self.vel.clone()
        return float(0.5 * (v_full @ self._full_mass() @ v_full).item())

    def strain_energy(self) -> float:
        d_full = self.disp.clone()
        return float(0.5 * (d_full @ self._full_stiffness() @ d_full).item())

    def _full_mass(self) -> torch.Tensor:
        """Complete (unconstrained) consistent mass matrix."""
        return self._M_full.clone()

    def _full_stiffness(self) -> torch.Tensor:
        """Complete (unconstrained) stiffness matrix, including constrained rows."""
        return self._K_full.clone()

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.strain_energy()

    def reset(self) -> None:
        self.disp.zero_()
        self.vel.zero_()
        self.acc.zero_()
        self.time = 0.0
        self.step_count = 0
        self._force_history.clear()
