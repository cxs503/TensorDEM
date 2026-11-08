"""Unsteady drag closure for a sphere in a viscous fluid.

Decomposes the total hydrodynamic force into four separable contributions::

    F_total = F_quasi-steady + F_added-mass + F_Basset + F_pressure-gradient

The pressure gradient (Froude-Krylov) force is essential for oscillatory flows.

References
----------
- Schiller & Naumann (1935): quasi-steady drag correlation.
- Stokes (1851): added-mass coefficient = 1/2 for a sphere; oscillatory drag.
- Landau & Lifshitz, Fluid Mechanics §24: oscillatory drag on a sphere.
- Maxey & Riley (1983): equation of motion for a small rigid sphere.
- van Hinsberg & Winckelmans (2010): O(N) recursive Basset kernel.

All quantities are SI unless noted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "schiller_naumann_drag",
    "added_mass_force",
    "pressure_gradient_force",
    "basset_direct_convolution",
    "basset_recursive",
    "BassetRecursionState",
    "unsteady_drag_force",
    "stokes_second_problem_velocity",
    "landau_lifshitz_oscillatory_drag",
    "landau_lifshitz_oscillatory_drag_amplitude_phase",
    "UnsteadyDragResult",
    "run_oscillatory_flow_simulation",
    "run_sudden_acceleration",
    "run_steady_uniform",
]


# ---------------------------------------------------------------------------
# 1. Quasi-steady drag (Schiller–Naumann)
# ---------------------------------------------------------------------------

def schiller_naumann_drag(
    mu: float,
    rho_f: float,
    d: float,
    u_slip: float,
) -> float:
    """Quasi-steady drag on a sphere using Schiller–Naumann correlation.

    Parameters
    ----------
    mu : dynamic viscosity [Pa·s]
    rho_f : fluid density [kg/m³]
    d : sphere diameter [m]
    u_slip : signed slip velocity u_f - u_p [m/s]

    Returns
    -------
    F_qs : drag force [N], signed (same sign as u_slip)
    """
    nu = mu / rho_f
    Re = abs(u_slip) * d / nu if nu > 0 else 0.0
    if Re < 1e-12:
        return 3.0 * math.pi * mu * d * u_slip
    Cd = 24.0 / Re * (1.0 + 0.15 * Re**0.687)
    A_proj = math.pi * d**2 / 4.0
    F = 0.5 * rho_f * u_slip * abs(u_slip) * A_proj * Cd
    return F


# ---------------------------------------------------------------------------
# 2. Added-mass force
# ---------------------------------------------------------------------------

def added_mass_force(
    rho_f: float,
    d: float,
    du_dt: float,
    C_m: float = 0.5,
) -> float:
    """Added-mass (virtual mass) force on a sphere.

    F_am = C_m · ρ_f · V · d(u_f − u_p)/dt

    For a sphere in potential flow C_m = 1/2 exactly (Stokes 1851, Lamb 1932).

    Parameters
    ----------
    rho_f : fluid density [kg/m³]
    d : sphere diameter [m]
    du_dt : relative acceleration d(u_f − u_p)/dt [m/s²]
    C_m : added-mass coefficient (default 0.5 for sphere)

    Returns
    -------
    F_am : added-mass force [N]
    """
    V = math.pi * d**3 / 6.0
    return C_m * rho_f * V * du_dt


# ---------------------------------------------------------------------------
# 3. Pressure gradient (Froude-Krylov) force
# ---------------------------------------------------------------------------

def pressure_gradient_force(
    rho_f: float,
    d: float,
    du_f_dt: float,
) -> float:
    """Pressure gradient (Froude-Krylov) force on a sphere.

    F_PG = ρ_f · V · Du_f/Dt

    For an oscillatory flow, this is the force due to the unsteady pressure
    field. In the Stokes limit, Du_f/Dt ≈ ∂u_f/∂t.

    Parameters
    ----------
    rho_f : fluid density [kg/m³]
    d : sphere diameter [m]
    du_f_dt : fluid acceleration ∂u_f/∂t [m/s²]

    Returns
    -------
    F_PG : pressure gradient force [N]
    """
    V = math.pi * d**3 / 6.0
    return rho_f * V * du_f_dt


# ---------------------------------------------------------------------------
# 4. Basset history force
# ---------------------------------------------------------------------------

def basset_direct_convolution(
    accel_history: np.ndarray,
    dt: float,
    mu: float,
    rho_f: float,
    d: float,
) -> np.ndarray:
    """Direct O(N²) evaluation of the Basset history force.

    Uses the Boussinesq–Basset form::

        F_B(t) = 6 d² √(π ρ_f μ) ∫₀ᵗ (d(u_f−u_p)/dτ) / √(t−τ) dτ

    Parameters
    ----------
    accel_history : relative acceleration d(u_f−u_p)/dt at each time step
    dt : time step [s]
    mu, rho_f, d : fluid/sphere parameters

    Returns
    -------
    F_basset : array of Basset force at each time step
    """
    N = len(accel_history)
    F = np.zeros(N)
    coeff = 1.5 * d**2 * math.sqrt(math.pi * rho_f * mu)
    for n in range(1, N):
        s = 0.0
        for k in range(n):
            tau = (n - k) * dt
            if tau > 0:
                s += accel_history[k] / math.sqrt(tau)
        F[n] = coeff * s * dt
    return F


@dataclass
class BassetRecursionState:
    """State for the O(N) recursive Basset scheme.

    The 1/√t kernel is approximated by a sum of exponentials::

        1/√t ≈ Σ_k  a_k · exp(−b_k · t)

    so the convolution becomes a set of first-order ODEs.
    """
    a: np.ndarray
    b: np.ndarray
    state: np.ndarray
    coeff: float

    @classmethod
    def create(
        cls,
        mu: float,
        rho_f: float,
        d: float,
        dt: float,
        n_exp: int = 12,
        t_max: Optional[float] = None,
    ) -> "BassetRecursionState":
        """Create recursion state with exponential approximation of 1/√t."""
        if t_max is None:
            t_max = max(10000.0 * dt, 10.0)

        n_sample = 500
        t_sample = np.logspace(np.log10(dt / 2), np.log10(t_max), n_sample)
        y_target = 1.0 / np.sqrt(t_sample)

        b_short = np.logspace(np.log10(1.0 / dt), np.log10(10.0 / dt), n_exp // 2)
        b_long = np.logspace(np.log10(1.0 / t_max), np.log10(1.0 / dt), n_exp - n_exp // 2)
        b = np.concatenate([b_short, b_long])
        b = np.sort(b)[::-1]

        A_mat = np.exp(-np.outer(b, t_sample))
        weights = 1.0 / np.sqrt(t_sample)
        A_mat_w = A_mat * weights[np.newaxis, :]
        y_target_w = y_target * weights
        a, _, _, _ = np.linalg.lstsq(A_mat_w.T, y_target_w, rcond=None)

        coeff = 1.5 * d**2 * math.sqrt(math.pi * rho_f * mu)

        return cls(a=a, b=b, state=np.zeros(n_exp), coeff=coeff)

    def step(self, accel: float, dt: float) -> float:
        """Advance one time step and return current Basset force."""
        for k in range(len(self.a)):
            exp_factor = math.exp(-self.b[k] * dt)
            self.state[k] = (
                self.state[k] * exp_factor
                + accel * (1.0 - exp_factor) / self.b[k]
            )
        return self.coeff * float(np.dot(self.a, self.state))

    def reset(self) -> None:
        """Reset all state variables to zero."""
        self.state[:] = 0.0


def basset_recursive(
    accel_history: np.ndarray,
    dt: float,
    mu: float,
    rho_f: float,
    d: float,
    n_exp: int = 12,
) -> np.ndarray:
    """O(N) recursive evaluation of Basset history force."""
    state = BassetRecursionState.create(mu, rho_f, d, dt, n_exp=n_exp)
    N = len(accel_history)
    F = np.zeros(N)
    for n in range(N):
        F[n] = state.step(accel_history[n], dt)
    return F


# ---------------------------------------------------------------------------
# 5. Combined unsteady drag
# ---------------------------------------------------------------------------

@dataclass
class UnsteadyDragResult:
    """Container for the force decomposition."""
    time: np.ndarray
    F_quasi_steady: np.ndarray
    F_added_mass: np.ndarray
    F_basset: np.ndarray
    F_pressure_gradient: np.ndarray
    F_total: np.ndarray

    @property
    def F_qs(self) -> np.ndarray:
        return self.F_quasi_steady

    @property
    def F_am(self) -> np.ndarray:
        return self.F_added_mass

    @property
    def F_pg(self) -> np.ndarray:
        return self.F_pressure_gradient


def unsteady_drag_force(
    u_f: np.ndarray,
    u_p: np.ndarray,
    dt: float,
    mu: float,
    rho_f: float,
    d: float,
    rho_p: float,
    method: str = "recursive",
    n_exp: int = 12,
) -> UnsteadyDragResult:
    """Compute the full unsteady drag decomposition over a time series.

    F_total = F_quasi-steady + F_added-mass + F_Basset + F_pressure-gradient

    Parameters
    ----------
    u_f : fluid velocity at particle location, shape (N,)
    u_p : particle velocity, shape (N,)
    dt : time step
    mu, rho_f, d : fluid/sphere parameters
    rho_p : particle density
    method : "direct" (O(N²)) or "recursive" (O(N))
    n_exp : number of exponentials for recursive method

    Returns
    -------
    UnsteadyDragResult with all components and total.
    """
    N = len(u_f)
    assert len(u_p) == N

    u_slip = u_f - u_p

    # 1. Quasi-steady drag (Schiller-Naumann)
    F_qs = np.array([schiller_naumann_drag(mu, rho_f, d, s) for s in u_slip])

    # 2. Relative acceleration (central differences)
    du_slip_dt = np.zeros(N)
    du_f_dt = np.zeros(N)
    if N >= 2:
        du_slip_dt[0] = (u_slip[1] - u_slip[0]) / dt
        du_slip_dt[-1] = (u_slip[-1] - u_slip[-2]) / dt
        du_f_dt[0] = (u_f[1] - u_f[0]) / dt
        du_f_dt[-1] = (u_f[-1] - u_f[-2]) / dt
        for n in range(1, N - 1):
            du_slip_dt[n] = (u_slip[n + 1] - u_slip[n - 1]) / (2.0 * dt)
            du_f_dt[n] = (u_f[n + 1] - u_f[n - 1]) / (2.0 * dt)

    # 3. Added-mass force
    V_p = math.pi * d**3 / 6.0
    F_am = 0.5 * rho_f * V_p * du_slip_dt

    # 4. Pressure gradient force
    F_pg = rho_f * V_p * du_f_dt

    # 5. Basset history force
    if method == "direct":
        F_basset = basset_direct_convolution(du_slip_dt, dt, mu, rho_f, d)
    elif method == "recursive":
        F_basset = basset_recursive(du_slip_dt, dt, mu, rho_f, d, n_exp=n_exp)
    else:
        raise ValueError(f"Unknown method: {method}")

    F_total = F_qs + F_am + F_basset + F_pg
    time = np.arange(N) * dt

    return UnsteadyDragResult(
        time=time,
        F_quasi_steady=F_qs,
        F_added_mass=F_am,
        F_basset=F_basset,
        F_pressure_gradient=F_pg,
        F_total=F_total,
    )


# ---------------------------------------------------------------------------
# 6. Analytical solutions for validation
# ---------------------------------------------------------------------------

def stokes_second_problem_velocity(
    y: np.ndarray,
    t: float,
    U0: float,
    omega: float,
    nu: float,
) -> np.ndarray:
    """Stokes second problem: oscillatory boundary layer.

    u(y, t) = U₀ · exp(−κ y) · cos(ω t − κ y)

    where κ = √(ω / (2ν)) is the Stokes layer wavenumber.
    """
    kappa = math.sqrt(omega / (2.0 * nu))
    return U0 * np.exp(-kappa * y) * np.cos(omega * t - kappa * y)


def landau_lifshitz_oscillatory_drag(
    omega: float,
    d: float,
    mu: float,
    rho_f: float,
    U0: float,
) -> complex:
    """Complex drag amplitude on a sphere in oscillatory flow.

    Uses the exact Stokes (1851) / Landau–Lifshitz §24 result.

    For a sphere of radius a in an oscillatory flow U∞ = U₀ e^{iωt}, the
    complex drag amplitude is::

        F̂ = 6πμa U₀ · H(λ)

    where λ = a √(iω/ν) = (1+i) α,  α = a √(ω/(2ν)), and::

        H(λ) = 1 + λ²/9 + λ(1 + λ²/9) / (1 + λ + λ²/9)

    Note: This is the drag on an oscillating sphere in fluid at rest.
    For a fixed sphere in oscillatory flow, add the pressure gradient force.
    """
    a = d / 2.0
    nu = mu / rho_f
    lam = a * math.sqrt(omega / nu) * (1.0 + 1.0j) / math.sqrt(2.0)

    lam2 = lam * lam
    H = 1.0 + lam2 / 9.0 + lam * (1.0 + lam2 / 9.0) / (1.0 + lam + lam2 / 9.0)

    F_stokes = 6.0 * math.pi * mu * a * U0
    return F_stokes * H


def landau_lifshitz_oscillatory_drag_amplitude_phase(
    omega: float,
    d: float,
    mu: float,
    rho_f: float,
    U0: float,
) -> Tuple[float, float]:
    """Return (amplitude, phase_deg) of TOTAL force in oscillatory flow.

    For a fixed sphere in oscillatory flow U = U₀ cos(ωt), the total force is:
    F_total = F_drag + F_added-mass + F_Basset + F_pressure-gradient

    The analytical result combines all contributions.
    """
    a = d / 2.0
    V = math.pi * d**3 / 6.0
    nu = mu / rho_f

    # Drag contribution (from Stokes 1851)
    lam = a * math.sqrt(omega / nu) * (1.0 + 1.0j) / math.sqrt(2.0)
    lam2 = lam * lam
    H = 1.0 + lam2 / 9.0 + lam * (1.0 + lam2 / 9.0) / (1.0 + lam + lam2 / 9.0)
    F_stokes = 6.0 * math.pi * mu * a * U0
    F_drag = F_stokes * H

    # Pressure gradient force: F_PG = ρ_f V du_f/dt
    # For u_f = U₀ e^{iωt}, du_f/dt = iω U₀ e^{iωt}
    F_pg = rho_f * V * 1j * omega * U0

    # Total force
    F_total = F_drag + F_pg

    amplitude = abs(F_total)
    phase_deg = math.degrees(math.atan2(F_total.imag, F_total.real))
    return amplitude, phase_deg


# ---------------------------------------------------------------------------
# 7. Simulation scenarios for validation
# ---------------------------------------------------------------------------

def run_sudden_acceleration(
    rho_f: float,
    d: float,
    mu: float,
    du_dt: float,
    t_end: float,
    dt: float,
) -> dict:
    """Simulate a sphere suddenly accelerated in a quiescent fluid."""
    N = int(t_end / dt) + 1
    time = np.arange(N) * dt

    u_slip = du_dt * time
    u_f = u_slip
    u_p = np.zeros(N)

    result = unsteady_drag_force(u_f, u_p, dt, mu, rho_f, d, rho_p=rho_f * 2.0)

    V = math.pi * d**3 / 6.0
    F_expected = 0.5 * rho_f * V * du_dt

    # Measure added-mass coefficient from the FIRST time step
    F_initial = result.F_total[1] if N > 1 else result.F_total[0]
    A_measured = F_initial / (rho_f * V * du_dt) if abs(du_dt) > 0 else 0.0

    return {
        "A_coeff": A_measured,
        "A_expected": 0.5,
        "F_expected": F_expected,
        "time": time,
        "F_total": result.F_total,
        "F_added_mass": result.F_added_mass,
        "F_basset": result.F_basset,
        "F_quasi_steady": result.F_quasi_steady,
        "F_pressure_gradient": result.F_pressure_gradient,
    }


def run_steady_uniform(
    rho_f: float,
    d: float,
    mu: float,
    rho_p: float,
    U_slip: float,
    t_end: float,
    dt: float,
) -> dict:
    """Steady uniform flow: only quasi-steady drag should survive."""
    N = int(t_end / dt) + 1
    time = np.arange(N) * dt

    u_f = np.full(N, U_slip)
    u_p = np.zeros(N)

    result = unsteady_drag_force(u_f, u_p, dt, mu, rho_f, d, rho_p)

    F_sn = schiller_naumann_drag(mu, rho_f, d, U_slip)

    return {
        "F_qs_mean": np.mean(result.F_quasi_steady[1:]),
        "F_am_max": np.max(np.abs(result.F_added_mass)),
        "F_basset_max": np.max(np.abs(result.F_basset)),
        "F_pg_max": np.max(np.abs(result.F_pressure_gradient)),
        "F_schiller_naumann": F_sn,
        "time": time,
        "F_total": result.F_total,
    }


def run_oscillatory_flow_simulation(
    rho_f: float,
    d: float,
    mu: float,
    rho_p: float,
    U0: float,
    omega: float,
    t_end: float,
    dt: float,
    method: str = "recursive",
) -> dict:
    """Simulate a fixed sphere in oscillatory flow u_f = U₀ cos(ωt)."""
    N = int(t_end / dt) + 1
    time = np.arange(N) * dt

    u_f = U0 * np.cos(omega * time)
    u_p = np.zeros(N)

    result = unsteady_drag_force(u_f, u_p, dt, mu, rho_f, d, rho_p, method=method)

    # Measure amplitude and phase from the last few periods
    period = 2.0 * math.pi / omega
    n_periods_skip = max(int(3.0 * period / dt), N // 3)
    t_meas = time[n_periods_skip:]
    F_meas = result.F_total[n_periods_skip:]

    # Fit: F(t) = a cos(ωt) + b sin(ωt)
    cos_wt = np.cos(omega * t_meas)
    sin_wt = np.sin(omega * t_meas)
    A_mat = np.column_stack([cos_wt, sin_wt])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, F_meas, rcond=None)
    a_cos, b_sin = coeffs

    amplitude_measured = math.sqrt(a_cos**2 + b_sin**2)
    phase_measured = math.degrees(math.atan2(b_sin, a_cos))

    # Analytical reference
    amplitude_analytical, phase_analytical = (
        landau_lifshitz_oscillatory_drag_amplitude_phase(omega, d, mu, rho_f, U0)
    )

    return {
        "amplitude_measured": amplitude_measured,
        "amplitude_analytical": amplitude_analytical,
        "phase_measured_deg": phase_measured,
        "phase_analytical_deg": phase_analytical,
        "time": time,
        "F_total": result.F_total,
        "F_quasi_steady": result.F_quasi_steady,
        "F_added_mass": result.F_added_mass,
        "F_basset": result.F_basset,
        "F_pressure_gradient": result.F_pressure_gradient,
        "u_f": u_f,
    }
