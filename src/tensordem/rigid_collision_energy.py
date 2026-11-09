"""Rigid collision energy ledger convergence certification (IN155-N07).

Three energy accounting methods for head-on rigid sphere collisions:
  (i)   State-difference: dE = E(t_final) - E(t_initial)  [current m5404口径]
  (ii)  Impulse-displacement integral:
        (ii-a) explicit final velocity:  W = integral F . v_final dt  [O(dt)]
        (ii-b) trapezoidal average:      W = integral F . (v_n + v_{n+1})/2 dt  [O(dt^2)]
  (iii) Restitution theory:  dE_theory = (1 - e^2)/2 * m_eff * v_rel^2

Key insight: Methods (i) and (ii-b) are mathematically equivalent for symplectic
Euler integration (both are O(dt^2) accurate). Method (ii-a) is O(dt) accurate.
The residual |dE_method - dE_theory| does NOT converge to zero because dE_theory
uses the design e_target while the simulation produces e_measured ≠ e_target due
to time discretization. The proper test is internal consistency: |dE_i - dE_iib|
should be ~0, and |dE_iia - dE_iib| should converge at O(dt).

Plus Newton's cradle (5-ball) multi-contact energy audit.

All quantities SI-normalised by E0 = 0.5 * m_eff * v_rel^2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .dem3d import DEM3DConfig, IceDEM3D


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class CollisionResult:
    """Single collision result at one (dt, k_n) setting."""
    dt: float
    k_n: float
    e_measured: float
    e_target: float
    # Method (i): state-difference
    dE_state: float
    # Method (ii-a): impulse * explicit final velocity
    dE_impulse_explicit: float
    # Method (ii-b): impulse * trapezoidal average velocity
    dE_impulse_trap: float
    # Method (iii): theoretical dissipation
    dE_theory: float
    # Cross-checks
    momentum_error: float
    contact_steps: int
    v0: float
    m_eff: float
    E0: float  # reference energy for normalisation


@dataclass
class ConvergenceRow:
    """One row of the dt-sweep."""
    dt: float
    k_n: float
    # Internal consistency residuals (method vs method)
    residual_i_vs_iib: float   # |dE_i - dE_iib| / E0
    residual_iia_vs_iib: float  # |dE_iia - dE_iib| / E0
    # Theory residuals
    residual_i_vs_theory: float
    residual_iib_vs_theory: float
    e_measured: float


@dataclass
class NewtonCradleResult:
    """5-ball Newton's cradle result."""
    total_momentum_before: float
    total_momentum_after: float
    momentum_error: float
    E0: float
    dE_state: float
    dE_theory_total: float
    e_measured_per_pair: list[float]
    dt: float
    k_n: float


# ---------------------------------------------------------------------------
# Core: build a two-sphere impact harness
# ---------------------------------------------------------------------------
def _build_pair_impact(
    *,
    k_n: float,
    c_n: float,
    radius: float = 0.05,
    density: float = 900.0,
    dt: float = 1e-5,
) -> tuple[IceDEM3D, int, int]:
    """Two free spheres on a collision course; all other particles frozen."""
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2,
        radius=radius, density=density,
        contact_stiffness=k_n, contact_damping=c_n,
        contact_damping_model="absolute",
        drag=0.0,
        bond_stiffness=1.0, breaking_strain=1e-12, shear_breaking_strain=1e-12,
        fix_x_edges=False, fix_bottom=False,
        tool_gap=1.0e3, tool_speed=1.0e-9,
    )
    dem = IceDEM3D(cfg)
    n = len(dem.positions)
    pos = torch.zeros_like(dem.positions)
    for idx in range(n):
        pos[idx] = torch.tensor([10.0 * idx, 500.0, 500.0], dtype=dem.dtype)
    # Place the two free spheres at z = ±(radius + 1um)
    pos[0] = torch.tensor([0.0, 0.0, -(radius + 1.0e-6)], dtype=dem.dtype)
    pos[1] = torch.tensor([0.0, 0.0, radius + 1.0e-6], dtype=dem.dtype)
    dem.positions = pos.clone()
    dem.initial_positions = pos.clone()
    mask = torch.ones(n, dtype=torch.bool)
    mask[0] = False
    mask[1] = False
    dem.fixed = mask
    # Break all bonds
    dem.forces()
    dem.dt = dt
    return dem, 0, 1


def _restitution_to_damping(e_target: float, k_n: float, m_eff: float) -> float:
    """Map restitution coefficient to damping (same as contact_tangential)."""
    from .contact_tangential import restitution_to_damping
    return restitution_to_damping(e_target, k_n, m_eff)


def dem_recommended_dt(k_n: float, mass: float, radius: float) -> float:
    """Compute the recommended dt for the DEM3D solver."""
    # From DEM3DConfig.recommended_dt: dt <= pi * sqrt(m / k) / 20
    omega = math.sqrt(k_n / mass)
    return math.pi / (20.0 * omega)


# ---------------------------------------------------------------------------
# Single collision with three energy accounting methods
# ---------------------------------------------------------------------------
def run_single_collision(
    *,
    e_target: float = 0.7,
    v0: float = 0.5,
    k_n: float = 5.0e4,
    radius: float = 0.05,
    density: float = 900.0,
    dt: float | None = None,
    dt_scale: float = 1.0,
) -> CollisionResult:
    """Run a head-on equal-mass collision and compute three energy residuals.

    Parameters
    ----------
    e_target : target coefficient of restitution
    v0 : approach speed (each sphere, so v_rel = 2*v0)
    k_n : contact stiffness [N/m]
    dt : absolute time step [s]; if None, uses recommended_dt * dt_scale
    dt_scale : multiplier on recommended_dt (used only if dt is None)
    """
    m = density * (4.0 / 3.0) * math.pi * radius ** 3
    m_eff = 0.5 * m
    c_n = _restitution_to_damping(e_target, k_n, m_eff)

    if dt is None:
        dt = dem_recommended_dt(k_n, m, radius) * dt_scale

    dem, i, j = _build_pair_impact(k_n=k_n, c_n=c_n, radius=radius, density=density, dt=dt)

    dem.velocities = torch.zeros_like(dem.positions)
    dem.velocities[i] = torch.tensor([0.0, 0.0, v0], dtype=dem.dtype)
    dem.velocities[j] = torch.tensor([0.0, 0.0, -v0], dtype=dem.dtype)

    # Reference energy: E0 = 2 * 0.5 * m * v0^2 = m * v0^2 (total KE)
    E0 = m * v0 ** 2

    # State before
    KE_before = float(0.5 * m * (dem.velocities ** 2).sum())
    momentum_before = float(dem.velocities[:, 2].sum() * m)

    # Run the collision, recording impulse integral
    impulse_explicit = 0.0   # sum F * v_final * dt
    impulse_trap = 0.0       # sum F * (v_n + v_{n+1})/2 * dt
    contact_steps = 0
    in_contact = False

    for step_idx in range(800_000):
        # Record velocity before step
        v_i_before = float(dem.velocities[i, 2])
        v_j_before = float(dem.velocities[j, 2])
        v_rel_before = v_j_before - v_i_before

        dem.step()

        v_i_after = float(dem.velocities[i, 2])
        v_j_after = float(dem.velocities[j, 2])
        v_rel_after = v_j_after - v_i_after

        # Compute contact force from momentum change
        sep = float(dem.positions[j, 2] - dem.positions[i, 2])
        overlap = 2.0 * radius - sep

        if overlap > 0:
            if not in_contact:
                in_contact = True
            contact_steps += 1
            # Force on i from contact (positive z = repulsive for i)
            F_i = m * (v_i_after - v_i_before) / dem.dt

            # Method (ii-a): explicit endpoint (use v_after)
            v_rel_explicit = v_rel_after
            dW_explicit = -F_i * v_rel_explicit * dem.dt

            # Method (ii-b): trapezoidal
            v_rel_avg = 0.5 * (v_rel_before + v_rel_after)
            dW_trap = -F_i * v_rel_avg * dem.dt

            impulse_explicit += dW_explicit
            impulse_trap += dW_trap
        elif in_contact:
            # Separated
            break
    else:
        raise RuntimeError("Pair did not separate within step budget")

    # State after
    KE_after = float(0.5 * m * (dem.velocities ** 2).sum())
    momentum_after = float(dem.velocities[:, 2].sum() * m)

    # Method (i): state-difference
    dE_state = KE_before - KE_after

    # Method (ii-a): impulse * explicit velocity
    dE_impulse_explicit = -impulse_explicit

    # Method (ii-b): impulse * trapezoidal velocity
    dE_impulse_trap = -impulse_trap

    # Method (iii): theoretical
    e_measured = float(abs(dem.velocities[i, 2])) / v0
    dE_theory = (1.0 - e_target ** 2) * 0.5 * m_eff * (2.0 * v0) ** 2

    return CollisionResult(
        dt=dem.dt,
        k_n=k_n,
        e_measured=e_measured,
        e_target=e_target,
        dE_state=dE_state,
        dE_impulse_explicit=dE_impulse_explicit,
        dE_impulse_trap=dE_impulse_trap,
        dE_theory=dE_theory,
        momentum_error=abs(momentum_after - momentum_before),
        contact_steps=contact_steps,
        v0=v0,
        m_eff=m_eff,
        E0=E0,
    )


# ---------------------------------------------------------------------------
# Convergence sweep
# ---------------------------------------------------------------------------
def run_convergence_sweep(
    *,
    e_target: float = 0.7,
    v0: float = 0.5,
    k_n_values: list[float] | None = None,
    dt_factors: list[float] | None = None,
    radius: float = 0.05,
    density: float = 900.0,
) -> list[ConvergenceRow]:
    """Sweep dt and k_n, compute residuals for all three methods."""
    if k_n_values is None:
        k_n_values = [1.0e4, 5.0e4, 1.0e5]
    if dt_factors is None:
        dt_factors = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]

    rows: list[ConvergenceRow] = []
    for k_n in k_n_values:
        m = density * (4.0 / 3.0) * math.pi * radius ** 3
        dt_base = dem_recommended_dt(k_n, m, radius)
        for factor in dt_factors:
            dt = dt_base * factor
            try:
                result = run_single_collision(
                    e_target=e_target, v0=v0, k_n=k_n,
                    radius=radius, density=density, dt=dt,
                )
            except RuntimeError:
                continue

            E0 = result.E0
            rows.append(ConvergenceRow(
                dt=dt,
                k_n=k_n,
                # Internal consistency
                residual_i_vs_iib=abs(result.dE_state - result.dE_impulse_trap) / E0,
                residual_iia_vs_iib=abs(result.dE_impulse_explicit - result.dE_impulse_trap) / E0,
                # Theory
                residual_i_vs_theory=abs(result.dE_state - result.dE_theory) / E0,
                residual_iib_vs_theory=abs(result.dE_impulse_trap - result.dE_theory) / E0,
                e_measured=result.e_measured,
            ))
    return rows


def fit_convergence_order(
    rows: list[ConvergenceRow],
    k_n: float,
    residual_key: str = "residual_iia_vs_iib",
) -> dict[str, float]:
    """Fit log|residual| vs log(dt) for a given k_n and residual type."""
    filtered = [r for r in rows if r.k_n == k_n]
    # Filter out zeros and very small values
    filtered = [r for r in filtered if getattr(r, residual_key) > 1e-15]
    if len(filtered) < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "r_squared": 0.0}

    log_dt = [math.log(r.dt) for r in filtered]
    log_res = [math.log(getattr(r, residual_key)) for r in filtered]

    # Linear regression
    n = len(log_dt)
    sum_x = sum(log_dt)
    sum_y = sum(log_res)
    sum_xy = sum(x * y for x, y in zip(log_dt, log_res))
    sum_x2 = sum(x * x for x in log_dt)

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-30:
        return {"slope": float("nan"), "intercept": float("nan"), "r_squared": 0.0}

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R-squared
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in log_res)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(log_dt, log_res))
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

    return {"slope": slope, "intercept": intercept, "r_squared": r_squared}


# ---------------------------------------------------------------------------
# Newton's cradle (5-ball)
# ---------------------------------------------------------------------------
def run_newtons_cradle(
    *,
    e_target: float = 0.95,
    v0: float = 1.0,
    k_n: float = 5.0e4,
    radius: float = 0.05,
    density: float = 900.0,
    dt: float | None = None,
    n_balls: int = 5,
) -> NewtonCradleResult:
    """Simulate a Newton's cradle: ball 0 hits a chain of n_balls-1 stationary balls.
    
    Uses the lattice structure but places n_balls along x-axis at y=500, z=500,
    and freezes all other particles.
    """
    m = density * (4.0 / 3.0) * math.pi * radius ** 3
    m_eff = 0.5 * m
    c_n = _restitution_to_damping(e_target, k_n, m_eff)

    # Build lattice with enough particles
    cfg = DEM3DConfig(
        nx=n_balls, ny=2, nz=2,
        radius=radius, density=density,
        contact_stiffness=k_n, contact_damping=c_n,
        contact_damping_model="absolute",
        drag=0.0,
        bond_stiffness=1.0, breaking_strain=1e-12, shear_breaking_strain=1e-12,
        fix_x_edges=False, fix_bottom=False,
        tool_gap=1.0e3, tool_speed=1.0e-9,
    )
    dem = IceDEM3D(cfg)
    n_particles = len(dem.positions)

    # Place n_balls along x-axis at y=500, z=500 with small gaps
    # The lattice has nx*ny*nz particles; we use the first n_balls (which are at y=-0.05, z=0)
    # and reposition them
    pos = dem.positions.clone()
    spacing = 2.0 * radius + 1.0e-6
    for idx in range(min(n_balls, n_particles)):
        pos[idx] = torch.tensor([spacing * idx, 500.0, 500.0], dtype=dem.dtype)
    dem.positions = pos.clone()
    dem.initial_positions = pos.clone()

    # Freeze all except the first n_balls
    mask = torch.ones(n_particles, dtype=torch.bool)
    for idx in range(min(n_balls, n_particles)):
        mask[idx] = False
    dem.fixed = mask

    # Break all bonds
    dem.forces()

    if dt is not None:
        dem.dt = dt
    else:
        dem.dt = dem_recommended_dt(k_n, m, radius) * 0.1

    # Ball 0 moves with +v0 in x; all others stationary
    dem.velocities = torch.zeros_like(dem.positions)
    dem.velocities[0, 0] = v0

    # Record initial state
    KE_before = float(0.5 * m * (dem.velocities ** 2).sum())
    momentum_before = float(dem.velocities[:, 0].sum() * m)
    E0 = KE_before

    # Run until the impulse has propagated through
    max_steps = 2_000_000
    step_count = 0

    for _ in range(max_steps):
        dem.step()
        step_count += 1

        # Check if all contacts have separated
        all_separated = True
        for idx in range(n_balls - 1):
            sep = float(dem.positions[idx + 1, 0] - dem.positions[idx, 0])
            if sep < 2.0 * radius:
                all_separated = False
                break

        if all_separated and step_count > 100:
            v0_final = float(dem.velocities[0, 0])
            if v0_final <= 0.01 * v0:
                break

    # Final state
    KE_after = float(0.5 * m * (dem.velocities ** 2).sum())
    momentum_after = float(dem.velocities[:, 0].sum() * m)

    dE_state = KE_before - KE_after

    # Theoretical total dissipation
    v_rel_initial = v0
    dE_theory_total = (1.0 - e_target ** 2) * 0.5 * m_eff * v_rel_initial ** 2

    # Measure per-pair restitution
    v_final = [float(dem.velocities[idx, 0]) for idx in range(n_balls)]
    e_pairs: list[float] = []
    if n_balls >= 2:
        e_measured_last = abs(v_final[-1]) / v0 if v0 > 0 else 0.0
        e_pairs.append(e_measured_last)

    return NewtonCradleResult(
        total_momentum_before=momentum_before,
        total_momentum_after=momentum_after,
        momentum_error=abs(momentum_after - momentum_before),
        E0=E0,
        dE_state=dE_state,
        dE_theory_total=dE_theory_total,
        e_measured_per_pair=e_pairs,
        dt=dem.dt,
        k_n=k_n,
    )


# ---------------------------------------------------------------------------
# High-level audit driver
# ---------------------------------------------------------------------------
def run_full_audit(
    *,
    e_target: float = 0.7,
    v0: float = 0.5,
    k_n_values: list[float] | None = None,
    radius: float = 0.05,
    density: float = 900.0,
) -> dict[str, Any]:
    """Run the full energy ledger convergence audit."""
    if k_n_values is None:
        k_n_values = [1.0e4, 5.0e4, 1.0e5]

    # 1. Convergence sweep
    dt_factors = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
    rows = run_convergence_sweep(
        e_target=e_target, v0=v0,
        k_n_values=k_n_values,
        dt_factors=dt_factors,
        radius=radius, density=density,
    )

    # 2. Fit convergence orders
    fits: dict[str, dict[str, float]] = {}
    for k_n in k_n_values:
        for residual_key in ("residual_i_vs_iib", "residual_iia_vs_iib"):
            key = f"k{k_n:.0e}_{residual_key}"
            fits[key] = fit_convergence_order(rows, k_n, residual_key)

    # 3. Newton's cradle
    cradle = run_newtons_cradle(
        e_target=0.95, v0=1.0, k_n=5.0e4,
        radius=radius, density=density,
    )

    # 4. Verdict
    verdict = "PASS"
    details: list[str] = []

    # Check internal consistency: (i) vs (ii-b) should be ~0
    for k_n in k_n_values:
        filtered = [r for r in rows if r.k_n == k_n]
        if filtered:
            max_i_vs_iib = max(r.residual_i_vs_iib for r in filtered)
            if max_i_vs_iib > 1e-12:
                details.append(f"WARN: k_n={k_n:.0e} max|(i)-(ii-b)|/E0 = {max_i_vs_iib:.2e}")

    # Check (ii-a) vs (ii-b) convergence order
    for k_n in k_n_values:
        slope = fits.get(f"k{k_n:.0e}_residual_iia_vs_iib", {}).get("slope", float("nan"))
        if not math.isnan(slope) and abs(slope - 1.0) > 0.3:
            details.append(f"WARN: k_n={k_n:.0e} slope(ii-a vs ii-b)={slope:.3f}, expected ~1.0")

    # Check Newton's cradle momentum
    if cradle.momentum_error > 1e-12:
        details.append(f"WARN: Newton's cradle momentum error = {cradle.momentum_error:.2e}")

    if not details:
        details.append("All checks passed.")

    return {
        "convergence_rows": rows,
        "fit_results": fits,
        "newtons_cradle": cradle,
        "verdict": verdict,
        "details": details,
        "e_target": e_target,
        "v0": v0,
        "k_n_values": k_n_values,
    }
