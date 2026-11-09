"""Rigid-collision energy-ledger convergence certification.

Why this module exists
----------------------
The DEM3D energy ledger (``EnergyAuditedIceDEM3D``) tracks work, dissipation,
and a scalar ``energy_balance_residual_J``.  For *rigid* contact collisions
(no bonds, no drag, no external loads) the physics is exactly solvable:

    E_kin_before = 0.5 * m * v0^2
    E_kin_after  = 0.5 * m * (e * v0)^2
    D_theory     = E_kin_before - E_kin_after = 0.5 * m * v0^2 * (1 - e^2)

The ledger residual should converge to zero at a rate determined by the
integration scheme (semi-implicit Euler → O(dt) for the explicit forward-Euler
part, O(dt^2) for the trapezoidal work integral).

This module defines **three independent energy-accounting口径 (criteria)**
for a single rigid collision and provides a dt-sweep harness that proves
convergence order.

Three口径
---------
A. **State-difference口径** (ΔE):
   ΔE = E_kin_after - E_kin_before + D_theory
   This is the simplest: compare the kinetic energy change with the theoretical
   dissipation.  The residual is O(dt) because semi-implicit Euler integrates
   the velocity with first-order accuracy.

B. **Impulse–displacement work口径** (W = ∫ F·v dt):
   Accumulate F · v · dt at each step during contact.  The net work done by
   the contact force equals the change in kinetic energy (since the spring
   energy is fully recovered).  Using the *explicit* (left-Riemann) quadrature
   gives O(dt); using the *trapezoidal* quadrature gives O(dt^2).

C. **Restitution-theory口径** (D_theory):
   D_theory = 0.5 * m * v0^2 * (1 - e^2), where e is the measured restitution
   coefficient.  This is the "ground truth" dissipation for a single impact.

Negative control
----------------
The explicit work integral (口径B-explicit) must converge at O(dt), while the
trapezoidal work integral (口径B-trapezoidal) must converge at O(dt^2).
This proves that the convergence order is a property of the quadrature, not
an artifact of the solver.

5-ball Newton cradle
--------------------
Five equal-mass spheres in a line, with the first sphere given an initial
velocity.  In the rigid-body limit (e=1, no dissipation), total momentum
and total energy are conserved.  With e<1, the energy loss per collision
is predictable.  This tests multi-contact energy accounting.

All quantities SI: kg, m, s, J, N.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from .dem3d import DEM3DConfig, IceDEM3D
from .dem3d_energy import EnergyAuditedIceDEM3D
from .restitution import unilateral_restitution

__all__ = [
    "CollisionConfig",
    "CollisionResult",
    "LedgerCriterion",
    "run_single_collision",
    "dt_sweep",
    "newton_cradle_result",
    "run_newton_cradle",
    "observed_order",
]


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class CollisionConfig:
    """Parameters for a single rigid collision test."""

    radius: float = 0.025
    density: float = 917.0
    stiffness: float = 5_000.0
    damping: float = 5.0
    impact_speed: float = 1.0       # m/s, towards the rigid plane
    dt: float | None = None
    device: str = "cpu"
    # Contact damping model: "absolute" uses contact_damping directly as gamma_n
    contact_damping_model: str = "absolute"

    @property
    def mass(self) -> float:
        return self.density * (4.0 / 3.0) * math.pi * self.radius ** 3

    @property
    def e_theory(self) -> float:
        """Theoretical restitution for the clamped unilateral law."""
        m_eff = self.mass  # rigid plane → effective mass = particle mass
        return unilateral_restitution(self.damping, self.stiffness, m_eff, self.impact_speed)

    @property
    def e_kin_before(self) -> float:
        return 0.5 * self.mass * self.impact_speed ** 2

    @property
    def d_theory(self) -> float:
        """Theoretical energy dissipation for one impact."""
        e = self.e_theory
        return self.e_kin_before * (1.0 - e ** 2)


@dataclass
class LedgerCriterion:
    """One energy-accounting口径 for a single collision."""

    name: str
    # State-difference口径
    delta_e_j: float = 0.0          # E_after - E_before
    # Work-integral口径 (F·v·dt, signed)
    work_explicit_j: float = 0.0    # left-Riemann sum of F·v·dt
    work_trapezoidal_j: float = 0.0 # trapezoidal sum of F·v·dt
    # Restitution-theory口径
    d_theory_j: float = 0.0         # 0.5*m*v0^2*(1-e^2)
    e_measured: float = 0.0
    # Residuals: |W - ΔE_kin| where ΔE_kin = E_after - E_before
    # The work done by contact force should equal ΔE_kin (spring energy recovered)
    residual_state_diff: float = 0.0   # |ΔE_kin + D_theory|
    residual_work_explicit: float = 0.0  # |W_explicit - ΔE_kin|
    residual_work_trap: float = 0.0      # |W_trapezoidal - ΔE_kin|
    # Metadata
    dt: float = 0.0
    contact_steps: int = 0
    max_overlap: float = 0.0


@dataclass
class CollisionResult:
    """Full result of a single collision run."""

    config: dict[str, Any]
    criterion: LedgerCriterion
    v_in: float
    v_out: float
    restitution_measured: float
    restitution_theory: float
    total_steps: int
    total_time: float


@dataclass
class NewtonCradleResult:
    """Result of the 5-ball Newton cradle test."""

    n_balls: int
    mass: float
    dt: float
    v0: float
    e_theory: float
    # Per-collision data
    collisions: list[dict[str, Any]]
    # Conservation checks
    initial_momentum: float
    final_momentum: float
    momentum_error_pct: float
    initial_energy: float
    final_energy: float
    energy_error_pct: float
    # Energy residual convergence
    energy_residual_j: float
    total_steps: int


# ===================================================================
# Single collision harness
# ===================================================================

def _build_single_collision(cfg: CollisionConfig) -> tuple[EnergyAuditedIceDEM3D, float]:
    """Build a single particle bouncing off a rigid bottom platen."""
    kwargs = dict(
        nx=2, ny=2, nz=2,
        radius=cfg.radius, density=cfg.density,
        contact_stiffness=cfg.stiffness, contact_damping=cfg.damping,
        platen_stiffness=cfg.stiffness, platen_damping=cfg.damping,
        drag=0.0, fix_x_edges=False, fix_bottom=False,
        bottom_platen_enabled=True, top_platen_enabled=False,
        tool_gap=100.0, tool_speed=1e-9,
        device=cfg.device,
        contact_damping_model=cfg.contact_damping_model,
    )
    base = DEM3DConfig(**kwargs)
    dt = cfg.dt if cfg.dt is not None else 0.5 * base.recommended_dt
    kwargs["dt"] = dt
    sim = EnergyAuditedIceDEM3D(DEM3DConfig(**kwargs))
    sim.alive[:] = False
    sim.failure_energy_J[:] = 0.0

    r = float(cfg.radius)
    bottom_z = 0.0
    sim.bottom_platen_start_z = bottom_z
    # Place particle above the plane, moving downward
    sim.positions[0] = torch.tensor((0.0, 0.0, bottom_z + r + 0.01),
                                    dtype=sim.dtype)
    sim.velocities[0] = torch.tensor((0.0, 0.0, -cfg.impact_speed),
                                     dtype=sim.dtype)
    # Park other particles far away
    z_mid = r + 0.05
    sites = ((-0.20, -0.20), (0.20, -0.20), (-0.20, 0.20),
             (0.20, 0.20), (0.0, 0.20), (0.0, -0.20), (-0.20, 0.0))
    with torch.no_grad():
        for idx in range(1, sim.positions.shape[0]):
            dx, dy = sites[(idx - 1) % len(sites)]
            sim.positions[idx] = torch.tensor((dx, dy, z_mid), dtype=sim.dtype)
            sim.velocities[idx] = torch.zeros(3, dtype=sim.dtype)

    return sim, bottom_z


def run_single_collision(cfg: CollisionConfig | None = None) -> CollisionResult:
    """Run a single rigid collision and compute all three energy口径."""
    cfg = cfg if cfg is not None else CollisionConfig()

    sim, bottom_z = _build_single_collision(cfg)
    r = float(cfg.radius)
    dt = float(sim.dt)
    mass = float(cfg.mass)

    v_in = cfg.impact_speed
    e_kin_before = 0.5 * mass * v_in ** 2

    # Track work integrals during contact: W = ∫ F · v dt
    # In semi-implicit Euler:
    #   F_n is computed at the beginning of step n (using positions)
    #   v_{n+1} = v_n + dt * F_n / m
    #   x_{n+1} = x_n + dt * v_{n+1}
    #
    # Explicit (left-Riemann): W = sum(F_n · v_n · dt)
    # Trapezoidal: W = sum(0.5 * (F_n · v_n + F_{n+1} · v_{n+1}) · dt)
    #   But F_{n+1} is not available until the next step, so we use:
    #   W = sum(0.5 * (F_n · v_n + F_n · v_{n+1}) · dt)
    #     = sum(F_n · (v_n + v_{n+1})/2 · dt)
    #   This is the trapezoidal rule using the force at the start of the step
    #   and the average velocity.

    work_explicit = 0.0
    work_trapezoidal = 0.0
    prev_vz = float(sim.velocities[0, 2])
    prev_force = 0.0
    in_contact = False
    contact_steps = 0
    max_overlap = 0.0
    steps = 0
    max_steps = 500_000

    v_out = 0.0
    separated_after_contact = False

    while steps < max_steps:
        sim.step()
        steps += 1

        z = float(sim.positions[0, 2])
        vz = float(sim.velocities[0, 2])  # This is v_{n+1} after the update
        overlap = max(0.0, bottom_z + r - z)
        reaction = abs(float(sim.last_bottom_platen_reaction_z))

        if overlap > 0:
            if not in_contact:
                in_contact = True
                prev_force = 0.0
            contact_steps += 1
            max_overlap = max(max_overlap, overlap)

            # Force is upward (+z), velocity vz is v_{n+1}
            # prev_vz is v_n (velocity at start of step)
            # reaction is F_n (force computed at start of step, used in this step)

            # Explicit: W = sum(F_n · v_n · dt)
            dW_explicit = prev_force * prev_vz * dt
            work_explicit += dW_explicit

            # Trapezoidal: W = sum(F_n · (v_n + v_{n+1})/2 · dt)
            dW_trap = prev_force * 0.5 * (prev_vz + vz) * dt
            work_trapezoidal += dW_trap

            prev_force = reaction
            prev_vz = vz
        else:
            if in_contact:
                # Just left contact
                if vz > 0:
                    v_out = vz
                    separated_after_contact = True
                in_contact = False
                prev_force = 0.0
                if separated_after_contact:
                    break

    if v_out == 0:
        v_out = abs(float(sim.velocities[0, 2]))

    e_measured = v_out / v_in if v_in > 0 else 0.0
    e_theory = cfg.e_theory
    d_theory = cfg.d_theory

    #口径A: State difference
    e_kin_after = 0.5 * mass * v_out ** 2
    delta_e = e_kin_after - e_kin_before
    residual_state = abs(delta_e + d_theory)

    #口径B: Work integrals vs kinetic energy change
    # The work done by contact force should equal ΔE_kin (since spring energy is recovered)
    residual_work_exp = abs(work_explicit - delta_e)
    residual_work_trap = abs(work_trapezoidal - delta_e)

    criterion = LedgerCriterion(
        name="rigid_collision",
        delta_e_j=delta_e,
        work_explicit_j=work_explicit,
        work_trapezoidal_j=work_trapezoidal,
        d_theory_j=d_theory,
        e_measured=e_measured,
        residual_state_diff=residual_state,
        residual_work_explicit=residual_work_exp,
        residual_work_trap=residual_work_trap,
        dt=dt,
        contact_steps=contact_steps,
        max_overlap=max_overlap,
    )

    return CollisionResult(
        config={"radius": cfg.radius, "density": cfg.density,
                "stiffness": cfg.stiffness, "damping": cfg.damping,
                "impact_speed": cfg.impact_speed},
        criterion=criterion,
        v_in=v_in,
        v_out=v_out,
        restitution_measured=e_measured,
        restitution_theory=e_theory,
        total_steps=steps,
        total_time=float(sim.time),
    )


# ===================================================================
# dt sweep
# ===================================================================

def dt_sweep(
    base_cfg: CollisionConfig | None = None,
    k_max: int = 6,
    stiffness_factors: tuple[float, ...] = (1.0,),
) -> list[dict[str, Any]]:
    """Run collision at dt/2^k for k=0..k_max, optionally with stiffness scan.

    Returns a list of rows suitable for CSV / convergence analysis.
    """
    base_cfg = base_cfg if base_cfg is not None else CollisionConfig()
    rows: list[dict[str, Any]] = []

    for k_factor in stiffness_factors:
        cfg = CollisionConfig(
            radius=base_cfg.radius,
            density=base_cfg.density,
            stiffness=base_cfg.stiffness * k_factor,
            damping=base_cfg.damping,
            impact_speed=base_cfg.impact_speed,
            device=base_cfg.device,
        )
        # Get base dt
        base_sim, _ = _build_single_collision(cfg)
        base_dt = float(base_sim.dt)

        for k in range(k_max + 1):
            dt_k = base_dt / (2 ** k)
            cfg_k = CollisionConfig(
                radius=cfg.radius,
                density=cfg.density,
                stiffness=cfg.stiffness,
                damping=cfg.damping,
                impact_speed=cfg.impact_speed,
                dt=dt_k,
                device=cfg.device,
            )
            result = run_single_collision(cfg_k)
            c = result.criterion
            rows.append({
                "k": k,
                "dt_s": dt_k,
                "stiffness_factor": k_factor,
                "contact_steps": c.contact_steps,
                "e_measured": c.e_measured,
                "e_theory": cfg.e_theory,
                "d_theory_j": c.d_theory_j,
                "residual_state_diff_j": c.residual_state_diff,
                "residual_work_explicit_j": c.residual_work_explicit,
                "residual_work_trap_j": c.residual_work_trap,
                "max_overlap_m": c.max_overlap,
                "v_in": result.v_in,
                "v_out": result.v_out,
            })

    return rows


def observed_order(values: list[float], dts: list[float]) -> float:
    """Compute observed convergence order from two or more data points.

    order = log(|e2/e1|) / log(dt2/dt1), averaged over consecutive pairs.
    """
    if len(values) < 2 or len(dts) < 2:
        return float("nan")
    orders = []
    for i in range(len(values) - 1):
        if values[i] > 0 and values[i + 1] > 0 and dts[i] > 0 and dts[i + 1] > 0:
            if dts[i] != dts[i + 1]:
                o = math.log(abs(values[i + 1] / values[i])) / math.log(dts[i + 1] / dts[i])
                orders.append(o)
    return sum(orders) / len(orders) if orders else float("nan")


# ===================================================================
# 5-ball Newton cradle
# ===================================================================

def _build_newton_cradle(
    n: int = 5,
    radius: float = 0.025,
    density: float = 917.0,
    stiffness: float = 5_000.0,
    damping: float = 5.0,
    v0: float = 1.0,
    dt: float | None = None,
    device: str = "cpu",
) -> tuple[EnergyAuditedIceDEM3D, float]:
    """Build a Newton cradle: n spheres in a line along x, first sphere moving."""
    mass = density * (4.0 / 3.0) * math.pi * radius ** 3

    kwargs = dict(
        nx=2, ny=2, nz=2,
        radius=radius, density=density,
        contact_stiffness=stiffness, contact_damping=damping,
        platen_stiffness=stiffness, platen_damping=damping,
        drag=0.0, fix_x_edges=False, fix_bottom=False,
        bottom_platen_enabled=False, top_platen_enabled=False,
        tool_gap=100.0, tool_speed=1e-9,
        device=device,
        contact_damping_model="absolute",
    )
    base = DEM3DConfig(**kwargs)
    dt_val = dt if dt is not None else 0.5 * base.recommended_dt
    kwargs["dt"] = dt_val
    sim = EnergyAuditedIceDEM3D(DEM3DConfig(**kwargs))
    sim.alive[:] = False
    sim.failure_energy_J[:] = 0.0

    # Place n balls in a line along x, just touching
    y0, z0 = 0.0, radius
    with torch.no_grad():
        for i in range(n):
            x = i * 2.0 * radius
            sim.positions[i] = torch.tensor((x, y0, z0), dtype=sim.dtype)
            sim.velocities[i] = torch.zeros(3, dtype=sim.dtype)
        # First ball moving right
        sim.velocities[0] = torch.tensor((v0, 0.0, 0.0), dtype=sim.dtype)
        # Park remaining particles far away
        for i in range(n, sim.positions.shape[0]):
            sim.positions[i] = torch.tensor((10.0 + i * 0.1, 0.0, z0), dtype=sim.dtype)

    return sim, mass


def run_newton_cradle(
    n: int = 5,
    radius: float = 0.025,
    density: float = 917.0,
    stiffness: float = 5_000.0,
    damping: float = 5.0,
    v0: float = 1.0,
    dt: float | None = None,
    max_steps: int = 200_000,
    device: str = "cpu",
) -> NewtonCradleResult:
    """Run the 5-ball Newton cradle and track momentum/energy conservation."""
    sim, mass = _build_newton_cradle(
        n=n, radius=radius, density=density,
        stiffness=stiffness, damping=damping, v0=v0, dt=dt, device=device,
    )

    # Initial state
    with torch.no_grad():
        v_init = sim.velocities[:n].clone()
    initial_momentum = float((mass * v_init).sum())
    initial_energy = float(0.5 * mass * (v_init ** 2).sum())

    # e_theory for pair collision
    m_eff = mass / 2.0  # two equal-mass mobile spheres
    from .restitution import unilateral_restitution
    e_theory = unilateral_restitution(damping, stiffness, m_eff, v0)

    collisions: list[dict[str, Any]] = []
    steps = 0
    in_contact = False
    contact_pair: tuple[int, int] | None = None
    v_before: dict[int, float] = {}

    while steps < max_steps:
        sim.step()
        steps += 1

        with torch.no_grad():
            vels = sim.velocities[:n].cpu().numpy()

        # Detect contacts
        for i in range(n - 1):
            for j in range(i + 1, n):
                dx = float(sim.positions[j, 0] - sim.positions[i, 0])
                dist = abs(dx)
                if dist < 2.0 * radius:
                    if not in_contact:
                        in_contact = True
                        contact_pair = (i, j)
                        v_before[i] = float(vels[i, 0])
                        v_before[j] = float(vels[j, 0])
                    break
            if in_contact:
                break
        else:
            if in_contact:
                # Contact ended
                vi_after = float(vels[contact_pair[0], 0])
                vj_after = float(vels[contact_pair[1], 0])
                vi_before = v_before[contact_pair[0]]
                vj_before = v_before[contact_pair[1]]

                # Relative speeds
                v_rel_before = vi_before - vj_before
                v_rel_after = vi_after - vj_after
                e_meas = abs(v_rel_after / v_rel_before) if abs(v_rel_before) > 1e-12 else 0.0

                collisions.append({
                    "index": len(collisions),
                    "pair": list(contact_pair),
                    "v_before": [vi_before, vj_before],
                    "v_after": [vi_after, vj_after],
                    "e_measured": e_meas,
                    "step": steps,
                })
                in_contact = False
                contact_pair = None

    # Final state
    with torch.no_grad():
        v_final = sim.velocities[:n].clone()
    final_momentum = float((mass * v_final).sum())
    final_energy = float(0.5 * mass * (v_final ** 2).sum())

    mom_error = abs(final_momentum - initial_momentum) / (abs(initial_momentum) + 1e-30) * 100
    eng_error = abs(final_energy - initial_energy) / (abs(initial_energy) + 1e-30) * 100

    return NewtonCradleResult(
        n_balls=n,
        mass=mass,
        dt=float(sim.dt),
        v0=v0,
        e_theory=e_theory,
        collisions=collisions,
        initial_momentum=initial_momentum,
        final_momentum=final_momentum,
        momentum_error_pct=mom_error,
        initial_energy=initial_energy,
        final_energy=final_energy,
        energy_error_pct=eng_error,
        energy_residual_j=abs(final_energy - initial_energy),
        total_steps=steps,
    )
