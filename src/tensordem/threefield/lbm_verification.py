"""Analytic verification harness for the three-field D3Q19 LBM core.

``tensordem.threefield.wake_lbm`` ships a D3Q19 single-relaxation-time solver with
Smagorinsky subgrid closure and Guo forcing for the propeller actuator disk. Its
committed evidence compares the *solver against itself* (a Gaussian self-similar
race model, a momentum-flux bookkeeping audit).  Nothing in the repository compared
it with an exact solution of the Navier-Stokes equations.  This module supplies
three independent closed-form references and the plumbing to run them against the
shipped kernels:

1. **Decaying shear wave.**  In a box periodic in ``y`` the velocity field

       u_x(y, t) = U cos(k y) exp(-nu k^2 t),   k = 2 pi m / N_y

   is an exact solution of the incompressible Navier-Stokes equations (constant
   pressure, zero divergence) and, to O(k^4), of the discrete BGK lattice
   Boltzmann equation.  The decay rate of its cosine projection measures the
   kinematic viscosity the solver actually realises, which must equal
   ``nu = (tau0 - 0.5)/3``.

2. **Chapman-Enskog strain field.**  For a prescribed traceless strain rate ``S``
   the first-order non-equilibrium part of the distribution is closed form,

       f_i^(1) = -3 tau0 w_i rho (c_ia c_ib - c_s^2 delta_ab) S_ab,

   which reproduces the non-equilibrium stress ``Pi_ab = -2 c_s^2 tau0 rho S_ab``.
   Feeding that field to the shipped ``_effective_tau`` returns the eddy viscosity
   the solver applies; it is compared with the analytic Smagorinsky value

       nu_t = (sqrt(tau0^2 + 12 Cs^2 tau0 |S|) - tau0) / 6,
       |S|  = sqrt(2 S_ab S_ab).

3. **Momentum impulse identity.**  With Guo forcing the first moment of the
   populations must grow by exactly ``sum(F)`` per step.  A uniform prescribed
   body force therefore gives ``dP_x / dt = sum(F_x)`` to round-off.

Items 1 and 3 only work with periodic propagation, which the propeller-race solver
never uses; ``WakeLBM(periodic=True)`` (a verification switch) wraps every face
instead of applying the inlet / zero-gradient-outlet / free-slip wall set, so the
shipped collision, equilibrium and streaming kernels are exercised unmodified.  The
actuator-disk controller is switched off by passing a zero prescribed body force
``step(n, body_force=torch.zeros(3))``.

All quantities are in lattice units.  Nothing here is a physical validation.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import torch

from .wake_lbm import LBMConfig, WakeLBM

__all__ = [
    "analytic_eddy_viscosity",
    "boundary_audit",
    "chapman_enskog_field",
    "fit_decay_rate",
    "initialise_shear_wave",
    "momentum_impulse_audit",
    "run_shear_decay",
    "shear_decay_reference_1d",
    "shear_mode_number",
    "shear_profiles",
    "shear_solver",
]


# ---------------------------------------------------------------------------
# configuration helpers
# ---------------------------------------------------------------------------
def shear_solver(
    *,
    nx: int = 12,
    ny: int = 48,
    nz: int = 8,
    tau0: float = 0.8,
    smagorinsky_cs: float = 0.0,
    seed: int = 20261103,
) -> WakeLBM:
    """Periodic-box solver sized for the shear-wave and forcing audits.

    The actuator-disk parameters are inert here (no disk force is ever applied)
    but ``LBMConfig`` validates them, so they are kept small and quiescent.
    """
    if nx < 12:
        raise ValueError("nx must be at least 12 (disk_x = 8 requires nx >= 12)")
    if min(ny, nz) < 8:
        raise ValueError("ny and nz must be at least 8 (LBMConfig floor)")
    config = LBMConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        tau0=tau0,
        smagorinsky_cs=smagorinsky_cs,
        inflow_velocity=0.02,
        jet_velocity=0.04,
        disk_x=8,
        disk_radius=2.0,
        disk_thickness=1,
        inlet_perturbation=0.0,
        disk_perturbation=0.0,
        disk_profile="parabolic",
        quarter_domain=False,
        dtype="float64",
    )
    return WakeLBM(config, seed=seed, periodic=True)


def shear_mode_number(ny: int, mode: int = 1) -> float:
    """Wavenumber of the ``mode``-th periodic cosine in a box of ``ny`` nodes."""
    if ny < 4:
        raise ValueError("ny must be at least 4")
    if not 1 <= int(mode) < ny // 2:
        raise ValueError("mode must satisfy 1 <= mode < ny/2")
    return 2.0 * math.pi * float(mode) / float(ny)


def _first_moment(solver: WakeLBM) -> torch.Tensor:
    return torch.einsum("ia,...i->...a", solver.c, solver.f)


def _zero_force(solver: WakeLBM) -> torch.Tensor:
    return torch.zeros(3, dtype=solver.dtype)


# ---------------------------------------------------------------------------
# 1. decaying shear wave
# ---------------------------------------------------------------------------
def initialise_shear_wave(solver: WakeLBM, amplitude: float, mode: int = 1) -> float:
    """Set ``u_x(y) = amplitude * cos(k y)`` on a local-equilibrium distribution."""
    k = shear_mode_number(solver.ny, mode)
    y = torch.arange(solver.ny, dtype=solver.dtype).reshape(1, solver.ny, 1)
    ux = float(amplitude) * torch.cos(k * y)
    u = torch.zeros((solver.nx, solver.ny, solver.nz, 3), dtype=solver.dtype)
    u[:, :, :, 0] = ux
    rho = torch.ones((solver.nx, solver.ny, solver.nz), dtype=solver.dtype)
    solver.f = solver._equilibrium(rho.unsqueeze(-1), u)
    solver.rho, solver.u = rho, u
    solver.step_count = 0
    return k


def shear_profiles(solver: WakeLBM, mode: int = 1) -> dict[str, Any]:
    """Cosine-projected amplitude plus the raw ``u_x(y)`` profile and invariants."""
    k = shear_mode_number(solver.ny, mode)
    rho, u = solver._macroscopic(solver.f)
    profile = u[:, :, :, 0].mean(dim=(0, 2))
    y = torch.arange(solver.ny, dtype=solver.dtype)
    weight = torch.cos(k * y)
    amplitude = float((2.0 / solver.ny) * (profile * weight).sum().item())
    momentum = _first_moment(solver)
    return {
        "amplitude": amplitude,
        "profile": profile.detach().clone(),
        "mass": float(rho.sum().item()),
        "momentum_x": float(momentum[..., 0].sum().item()),
        "momentum_y": float(momentum[..., 1].sum().item()),
        "max_uy": float(u[:, :, :, 1].abs().max().item()),
    }


def fit_decay_rate(times: Sequence[float], amplitudes: Sequence[float]) -> dict[str, float]:
    """Least-squares fit of ``ln amplitude = ln A - rate * t``."""
    ts = [float(t) for t in times]
    ys = [float(a) for a in amplitudes]
    if len(ts) < 3:
        raise ValueError("need at least three samples for a decay fit")
    if any(a <= 0.0 for a in ys):
        raise ValueError("amplitudes must stay positive for a log fit")
    n = float(len(ts))
    mean_t = sum(ts) / n
    mean_y = sum(math.log(a) for a in ys) / n
    sxx = sum((t - mean_t) ** 2 for t in ts)
    if sxx == 0.0:
        raise ValueError("degenerate time samples")
    sxy = sum((t - mean_t) * (math.log(a) - mean_y) for t, a in zip(ts, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_t
    ss_tot = sum((math.log(a) - mean_y) ** 2 for a in ys)
    ss_res = sum(
        (math.log(a) - (intercept + slope * t)) ** 2 for t, a in zip(ts, ys)
    )
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {"rate": -slope, "log_amplitude": intercept, "r2": r2, "samples": len(ts)}


def run_shear_decay(
    *,
    tau0: float = 0.8,
    smagorinsky_cs: float = 0.0,
    ny: int = 48,
    nx: int = 12,
    nz: int = 8,
    mode: int = 1,
    amplitude: float = 0.02,
    steps: int | None = None,
    sample_every: int = 20,
) -> dict[str, Any]:
    """Run the shipped kernels on the exact shear-wave initial condition."""
    solver = shear_solver(
        nx=nx, ny=ny, nz=nz, tau0=tau0, smagorinsky_cs=smagorinsky_cs
    )
    k = initialise_shear_wave(solver, amplitude, mode)
    nu = (tau0 - 0.5) / 3.0
    if steps is None:
        target = 4.0 / (nu * k * k) if nu > 0.0 else 400.0
        steps = int(min(8000, max(400, round(target))))
    zero = _zero_force(solver)
    times: list[float] = [0.0]
    amps: list[float] = [float(amplitude)]
    masses: list[float] = []
    mom_x: list[float] = []
    max_uy: list[float] = []
    state = shear_profiles(solver, mode)
    masses.append(state["mass"])
    mom_x.append(state["momentum_x"])
    max_uy.append(state["max_uy"])
    for _ in range(int(steps) // int(sample_every)):
        solver.step(int(sample_every), body_force=zero)
        state = shear_profiles(solver, mode)
        times.append(float(solver.step_count))
        amps.append(state["amplitude"])
        masses.append(state["mass"])
        mom_x.append(state["momentum_x"])
        max_uy.append(state["max_uy"])

    # Skip the interval in which the O(u^2) initial transient is still radiating,
    # and stop once the amplitude is too small to separate from that transient.
    keep = [
        (t, a)
        for t, a in zip(times, amps)
        if t >= 0.08 * times[-1] and abs(a) >= 0.18 * abs(amplitude)
    ]
    if len(keep) < 4:  # short runs: fall back to the whole record
        keep = list(zip(times, amps))
    fit = fit_decay_rate([t for t, _ in keep], [a for _, a in keep])
    nu_measured = fit["rate"] / (k * k)
    final = shear_profiles(solver, mode)
    y = torch.arange(ny, dtype=solver.dtype)
    analytic_profile = final["amplitude"] * torch.cos(k * y)
    shape_error = float(
        (final["profile"] - analytic_profile).abs().max().item()
    ) / abs(float(amplitude))
    mass_drift = max(abs(m - masses[0]) for m in masses) / masses[0]
    mom_drift = max(abs(p - mom_x[0]) for p in mom_x) / max(
        abs(float(amplitude)) * float(nx * nz), 1e-30
    )
    return {
        "tau0": float(tau0),
        "smagorinsky_cs": float(smagorinsky_cs),
        "ny": int(ny),
        "mode": int(mode),
        "wavenumber": k,
        "amplitude": float(amplitude),
        "steps": int(solver.step_count),
        "sample_every": int(sample_every),
        "times": times,
        "amplitudes": amps,
        "mass": masses,
        "momentum_x": mom_x,
        "max_uy": max_uy,
        "decay_fit": fit,
        "nu_theory": nu,
        "nu_measured": nu_measured,
        "nu_relative_error": (nu_measured - nu) / nu if nu else float("nan"),
        "shape_error_rel_amplitude": shape_error,
        "mass_drift_rel": mass_drift,
        "momentum_drift_rel": mom_drift,
        "max_transverse_velocity": max(max_uy),
        "final_profile": final["profile"].tolist(),
    }


# ---------------------------------------------------------------------------
# 2. Chapman-Enskog strain field and the Smagorinsky closure
# ---------------------------------------------------------------------------
def analytic_eddy_viscosity(tau0: float, smagorinsky_cs: float, strain_norm: float) -> float:
    """Eddy viscosity the shipped closure must produce for strain magnitude |S|.

    ``|S| = sqrt(2 S_ab S_ab)``.  Derived from the shipped formula
    ``tau_eff = 0.5 (tau0 + sqrt(tau0^2 + 18 sqrt(2) Cs^2 |Pi| / rho))`` together
    with the Chapman-Enskog non-equilibrium stress, where the shipped ``_outer``
    contract lists the six unique components ``(xx, yy, zz, xy, xz, yz)`` once, so
    ``|Pi| = |Pi_xy| = (1/3) tau0 |S|`` for pure shear.  That gives
    ``tau_eff - tau0 = (sqrt(tau0^2 + 6 sqrt(2) Cs^2 tau0 |S|) - tau0) / 2`` and
    hence ``nu_t = (sqrt(tau0^2 + 6 sqrt(2) Cs^2 tau0 |S|) - tau0) / 6``.  For small
    eddy viscosity this reduces to ``nu_t -> Cs^2 |S| / sqrt(2)``, i.e. an effective
    coefficient ``Cs_eff = Cs / 2^(1/4) = 0.8409 Cs`` (the ``Pi``-norm convention
    differs from Hou et al. by ``sqrt(2)``).
    """
    if tau0 <= 0.5:
        raise ValueError("tau0 must exceed 0.5")
    if smagorinsky_cs < 0.0:
        raise ValueError("smagorinsky_cs must be non-negative")
    if strain_norm < 0.0:
        raise ValueError("strain_norm must be non-negative")
    inner = tau0 * tau0 + 6.0 * math.sqrt(2.0) * smagorinsky_cs ** 2 * tau0 * strain_norm
    return (math.sqrt(inner) - tau0) / 6.0


def chapman_enskog_field(solver: WakeLBM, strain_xy: float) -> dict[str, Any]:
    """Build ``f = f^eq(rho=1, u=0) + f^(1)`` for a pure shear strain rate."""
    rho = torch.ones((solver.nx, solver.ny, solver.nz), dtype=solver.dtype)
    u = torch.zeros((solver.nx, solver.ny, solver.nz, 3), dtype=solver.dtype)
    feq = solver._equilibrium(rho.unsqueeze(-1), u)
    strain = torch.zeros((3, 3), dtype=solver.dtype)
    strain[0, 1] = float(strain_xy)
    strain[1, 0] = float(strain_xy)
    c = solver.c
    cc = torch.einsum("ia,ib->iab", c, c)
    cs2 = torch.tensor(solver.cs2, dtype=solver.dtype)
    deviator = cc - cs2 * torch.eye(3, dtype=solver.dtype).unsqueeze(0)
    projection = torch.einsum("iab,ab->i", deviator, strain)
    noneq = (
        -3.0
        * solver.config.tau0
        * solver.w.view(1, 1, 1, -1)
        * rho.unsqueeze(-1)
        * projection.view(1, 1, 1, -1)
    )
    f = feq + noneq
    measured_rho, measured_u = solver._macroscopic(f)
    pi = torch.einsum("ik,...i->...k", solver._outer, f - feq)
    pi_norm = float(torch.sqrt((pi * pi).sum(dim=-1).clamp_min(0.0)).mean().item())
    strain_norm = abs(2.0 * float(strain_xy))
    tau_eff = solver._effective_tau(f, measured_rho, measured_u)
    nu_eff = float(((tau_eff - 0.5) / 3.0).mean().item())
    nu0 = (solver.config.tau0 - 0.5) / 3.0
    return {
        "field": f,
        "tau0": float(solver.config.tau0),
        "smagorinsky_cs": float(solver.config.smagorinsky_cs),
        "strain_xy": float(strain_xy),
        "strain_norm": strain_norm,
        "pi_norm": pi_norm,
        "pi_norm_over_rho_theory": solver.config.tau0 * strain_norm / 3.0,
        "rho_from_field": float(measured_rho.mean().item()),
        "max_velocity_from_field": float(measured_u.abs().max().item()),
        "nu_molecular": nu0,
        "nu_total_measured": nu_eff,
        "nu_eddy_measured": nu_eff - nu0,
        "nu_eddy_theory": analytic_eddy_viscosity(
            solver.config.tau0, solver.config.smagorinsky_cs, strain_norm
        ),
        "nu_eddy_leading_order": solver.config.smagorinsky_cs ** 2 * strain_norm / math.sqrt(2.0),
    }


# ---------------------------------------------------------------------------
# 3. momentum impulse identity of the Guo forcing
# ---------------------------------------------------------------------------
def momentum_impulse_audit(
    *,
    tau0: float = 0.5006,
    smagorinsky_cs: float = 0.0,
    steps: int = 5,
    force_x: float = 1e-5,
    nx: int = 12,
    ny: int = 8,
    nz: int = 8,
) -> dict[str, float]:
    """Check that the shipped forcing injects exactly ``sum(F)`` per step."""
    solver = shear_solver(
        nx=nx, ny=ny, nz=nz, tau0=tau0, smagorinsky_cs=smagorinsky_cs
    )
    solver.f = solver._equilibrium(
        torch.ones((nx, ny, nz, 1), dtype=solver.dtype),
        torch.zeros((nx, ny, nz, 3), dtype=solver.dtype),
    )
    force = torch.tensor([float(force_x), 0.0, 0.0], dtype=solver.dtype)
    before = float(_first_moment(solver)[..., 0].sum().item())
    solver.step(int(steps), body_force=force)
    after = float(_first_moment(solver)[..., 0].sum().item())
    injected = float(steps) * float(force_x) * nx * ny * nz
    booked = solver.thrust_impulse
    return {
        "tau0": float(tau0),
        "smagorinsky_cs": float(smagorinsky_cs),
        "steps": int(steps),
        "momentum_gain": after - before,
        "injected_expected": injected,
        "ratio_measured_over_expected": (after - before) / injected,
        "booked_impulse": booked,
        "ratio_booked_over_expected": booked / injected,
        "ratio_measured_over_booked": (after - before) / booked if booked else float("nan"),
        "unguarded_prefactor": 1.0 + 0.5 / float(tau0),
    }


# ---------------------------------------------------------------------------
# 4. boundary-set audit of the production (non-periodic) configuration
# ---------------------------------------------------------------------------
def boundary_audit(
    *,
    nx: int = 24,
    ny: int = 12,
    nz: int = 12,
    inflow_velocity: float = 0.04,
    tau0: float = 0.8,
) -> dict[str, float]:
    """Apply the shipped boundaries to an exactly uniform field.

    A uniform stream at the inlet velocity is a steady state of the
    inlet / zero-gradient-outlet / free-slip set: ``_apply_boundaries(_stream(f))``
    must reproduce ``f`` to round-off.  Any deviation is a spurious force injected
    by the boundary treatment rather than by physics (the defect class that was
    found and fixed in the platen boundaries of ``dem3d``).
    """
    config = LBMConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        tau0=tau0,
        smagorinsky_cs=0.0,
        inflow_velocity=inflow_velocity,
        jet_velocity=inflow_velocity + 0.02,
        disk_x=8,
        disk_radius=2.0,
        disk_thickness=1,
        inlet_perturbation=0.0,
        disk_perturbation=0.0,
        quarter_domain=False,
        dtype="float64",
    )
    solver = WakeLBM(config, periodic=False)
    rho = torch.ones((nx, ny, nz), dtype=solver.dtype)
    u = torch.zeros((nx, ny, nz, 3), dtype=solver.dtype)
    u[:, :, :, 0] = inflow_velocity
    f = solver._equilibrium(rho.unsqueeze(-1), u)
    streamed = solver._stream(f)
    bounded = solver._apply_boundaries(streamed.clone())
    rho_b, u_b = solver._macroscopic(bounded)
    scale = float(f.abs().max().item())
    return {
        "max_population_deviation": float((bounded - f).abs().max().item()),
        "max_population_deviation_rel": float((bounded - f).abs().max().item()) / scale,
        "max_density_deviation": float((rho_b - 1.0).abs().max().item()),
        "max_velocity_deviation": float((u_b - u).abs().max().item()),
        "streaming_max_deviation": float((streamed - f).abs().max().item()),
    }


# ---------------------------------------------------------------------------
# 5. nonlinear-diffusion reference for the dynamic Smagorinsky decay
# ---------------------------------------------------------------------------
def shear_decay_reference_1d(
    *,
    ny: int = 48,
    tau0: float = 0.8,
    smagorinsky_cs: float = 0.0,
    amplitude: float = 0.02,
    mode: int = 1,
    steps: int = 400,
    substeps: int = 40,
) -> list[float]:
    """Reference amplitude history of the same shear wave under the closure model.

    Integrates ``du/dt = d/dy [ (nu0 + Cs^2 |du/dy|) du/dy ]`` on the same
    ``ny``-point periodic grid with ``substeps`` explicit sub-steps per lattice
    step, i.e. the continuum reading of the Smagorinsky closure the solver
    implements.  Returns the cosine-projected amplitude at each lattice step.
    """
    k = shear_mode_number(ny, mode)
    y = torch.arange(ny, dtype=torch.float64)
    u = float(amplitude) * torch.cos(k * y)
    nu0 = (tau0 - 0.5) / 3.0
    dy = 1.0
    dt = 1.0 / float(substeps)
    amps = [float(amplitude)]
    for _ in range(int(steps)):
        for _ in range(int(substeps)):
            grad = (torch.roll(u, -1) - torch.roll(u, 1)) / (2.0 * dy)
            nu_t = smagorinsky_cs ** 2 * grad.abs()
            flux = (nu0 + nu_t) * grad
            dflux = (torch.roll(flux, -1) - torch.roll(flux, 1)) / (2.0 * dy)
            u = u + dt * dflux
        amps.append(float((2.0 / ny) * (u * torch.cos(k * y)).sum().item()))
    return amps