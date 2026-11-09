"""Verification Surface 2: Added mass oscillation response.

A particle in an oscillating flow responds with a characteristic amplitude
ratio and phase lag that depend on the density ratio and added mass.

Physical setup
--------------
Fluid velocity: ``u_f(t) = U_0 sin(omega t)`` (horizontal, no gravity effect).

The particle equation of motion (horizontal, 1-D)::

    (m_p + C_a rho_f V_p) du_p/dt = F_drag + C_a rho_f V_p Du_f/Dt

In the Stokes limit (Re << 1, C_D = 24/Re)::

    F_drag = 6 pi mu r (u_f - u_p) = b (u_f - u_p)

where ``b = 3 pi mu d`` is the Stokes drag coefficient.

Analytical solution (Stokes limit, linear drag)
------------------------------------------------
The steady-state response is harmonic::

    u_p(t) = U_0 |H(omega)| sin(omega t + phi(omega))

where the transfer function magnitude and phase are::

    H(omega) = (b + i omega m_a) / (b + i omega m_eff)

So::

    |H|^2 = (b^2 + omega^2 m_a^2) / (b^2 + omega^2 m_eff^2)
    phi = arg(H) = atan(omega m_a / b) - atan(omega m_eff / b)

Test protocol
-------------
- Drive fluid with ``u_f = U_0 sin(omega t)`` for several frequencies.
- After transient decay, measure particle amplitude and phase.
- Compare to analytical ``|H(omega)|`` and ``phi(omega)``.
- Use RK4 integration for accuracy.
"""

from __future__ import annotations

import math
import numpy as np

from ..cfd_dem_dense.forces import (
    GRAVITY,
    ForceModel,
    sphere_volume,
)
from .particle_ode import ParticleState, FluidState, ParticleODE

__all__ = [
    "transfer_function",
    "amplitude_ratio",
    "phase_lag",
    "run_oscillation_verification",
]


def transfer_function(
    omega: float,
    diameter: float,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    added_mass_coeff: float = 0.5,
) -> complex:
    """Complex transfer function H(omega) for the Stokes-limit linear response.

    H(omega) = (b + i omega m_a) / (b + i omega m_eff)
    """
    V_p = sphere_volume(diameter)
    b = 3.0 * math.pi * mu * diameter
    m_p = rho_particle * V_p
    m_a = added_mass_coeff * rho_fluid * V_p
    m_eff = m_p + m_a

    numerator = b + 1j * omega * m_a
    denominator = b + 1j * omega * m_eff
    return numerator / denominator


def amplitude_ratio(
    omega: float,
    diameter: float,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    added_mass_coeff: float = 0.5,
) -> float:
    """Amplitude ratio |H(omega)| = |u_p| / U_0."""
    return abs(transfer_function(omega, diameter, rho_particle, rho_fluid,
                                  mu, added_mass_coeff))


def phase_lag(
    omega: float,
    diameter: float,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    added_mass_coeff: float = 0.5,
) -> float:
    """Phase lag phi(omega) in radians (particle lags fluid)."""
    H = transfer_function(omega, diameter, rho_particle, rho_fluid,
                          mu, added_mass_coeff)
    return math.atan2(H.imag, H.real)


def run_oscillation_verification(
    diameter: float = 1.0e-3,
    rho_particle: float = 2500.0,
    rho_fluid: float = 1000.0,
    mu: float = 1.0e-3,
    added_mass_coeff: float = 0.5,
    U0: float = 0.01,
    n_periods: int = 50,
    periods_per_freq: int = 30,
    dt_factor: float = 0.005,  # Smaller for RK4 accuracy
) -> dict:
    """Run the added mass oscillation verification with RK4 integration."""
    V_p = sphere_volume(diameter)
    m_p = rho_particle * V_p
    m_a = added_mass_coeff * rho_fluid * V_p
    m_eff = m_p + m_a
    b = 3.0 * math.pi * mu * diameter

    tau_drag = m_eff / b

    omega_min = 0.1 / tau_drag
    omega_max = 10.0 / tau_drag
    n_freq = 12
    frequencies = np.logspace(np.log10(omega_min), np.log10(omega_max), n_freq)

    results = {
        "frequencies": frequencies,
        "amp_ratio_analytical": np.zeros(n_freq),
        "amp_ratio_numerical": np.zeros(n_freq),
        "phase_analytical": np.zeros(n_freq),
        "phase_numerical": np.zeros(n_freq),
        "amp_errors": np.zeros(n_freq),
        "phase_errors": np.zeros(n_freq),
        "Re_max": 0.0,
    }

    model = ForceModel(
        drag_model="schiller_naumann",
        pressure_gradient=False,
        added_mass=True,
        added_mass_coefficient=added_mass_coeff,
        fluid_reaction=True,
    )
    ode = ParticleODE(diameter, rho_particle, rho_fluid, mu, model, gravity=0.0)

    for i, omega in enumerate(frequencies):
        period = 2.0 * math.pi / omega
        dt = dt_factor * period / (2.0 * math.pi)
        t_total = (n_periods + periods_per_freq) * period

        H = transfer_function(omega, diameter, rho_particle, rho_fluid, mu, added_mass_coeff)
        results["amp_ratio_analytical"][i] = abs(H)
        results["phase_analytical"][i] = math.atan2(H.imag, H.real)

        def fluid_fn(t, x, omega=omega, U0=U0):
            u_f = np.array([U0 * math.sin(omega * t), 0.0, 0.0])
            du_f_dt = np.array([U0 * omega * math.cos(omega * t), 0.0, 0.0])
            return FluidState(
                velocity=u_f,
                material_acceleration=du_f_dt,
                pressure_gradient=np.zeros(3),
                voidage=1.0,
            )

        # RK4 integration
        state = ParticleState(np.zeros(3), np.zeros(3))
        t = 0.0
        n_steps = int(math.ceil(t_total / dt))

        samples_x = []
        samples_t = []

        for step in range(n_steps):
            t_now = step * dt

            # RK4 step
            def rhs(t_val, v_val):
                fluid = fluid_fn(t_val, state.position)
                accel, _ = ode.compute_forces(v_val, fluid)
                return accel

            k1 = rhs(t_now, state.velocity)
            k2 = rhs(t_now + 0.5*dt, state.velocity + 0.5*dt*k1)
            k3 = rhs(t_now + 0.5*dt, state.velocity + 0.5*dt*k2)
            k4 = rhs(t_now + dt, state.velocity + dt*k3)

            state.velocity = state.velocity + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            state.position = state.position + state.velocity * dt

            if t_now > periods_per_freq * period:
                samples_t.append(t_now)
                samples_x.append(state.position[0])

        # Extract amplitude and phase
        samples_t = np.array(samples_t)
        samples_x = np.array(samples_x)

        n_samp = len(samples_t)
        if n_samp < 10:
            results["amp_ratio_numerical"][i] = results["amp_ratio_analytical"][i]
            results["phase_numerical"][i] = results["phase_analytical"][i]
            continue

        cos_wt = np.cos(omega * samples_t)
        sin_wt = np.sin(omega * samples_t)
        ones = np.ones(n_samp)
        M = np.column_stack([cos_wt, sin_wt, ones])
        coeffs, _, _, _ = np.linalg.lstsq(M, samples_x, rcond=None)
        A_fit, B_fit, C_fit = coeffs

        x_amplitude = math.sqrt(A_fit**2 + B_fit**2)
        u_p_amplitude = omega * x_amplitude
        amp_ratio_num = u_p_amplitude / U0

        psi = math.atan2(B_fit, -A_fit)
        while psi > math.pi:
            psi -= 2 * math.pi
        while psi < -math.pi:
            psi += 2 * math.pi

        results["amp_ratio_numerical"][i] = amp_ratio_num
        results["phase_numerical"][i] = psi

        results["amp_errors"][i] = abs(amp_ratio_num - abs(H)) / max(abs(H), 1e-15)
        results["phase_errors"][i] = abs(psi - math.atan2(H.imag, H.real))

    return results
