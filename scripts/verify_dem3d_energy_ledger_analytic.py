"""Closed-form verification of the DEM3D cumulative energy ledger.

Why this exists
---------------
``EnergyAuditedIceDEM3D`` (``src/tensordem/dem3d_energy.py``) accumulates
``tool_work``, ``external_work``, ``drag_dissipation``, contact-damping
dissipation and fracture release, and reports a scalar
``energy_balance_residual_J``. Until now that ledger had only been checked for
*self-consistency* (terms finite/non-negative, checkpoint reproducibilty,
repeatability of a benchmark run). No published check compared the accumulated
terms with a closed-form reference, so the residual magnitude and its
discretisation order were unquantified.

This script builds two exactly solvable subsystems out of the shipped solver
(no source modifications) and scores the ledger against the analytical
solutions:

Case A - exponential viscous decay (drag only)
    Single free particle, drag ``c``, mass ``m``, no bonds, no contacts::

        m dv/dt = -c v                 ->  v(t) = v0 exp(-t/tau),  tau = m/c
        D(t) = int_0^t c v^2 dt'       ->  D(t) = 0.5 m v0^2 (1 - exp(-2t/tau))

    The ledger's ``drag_dissipation_cumulative_J`` should equal ``D(t)`` and
    the residual should be bounded by the step-size truncation, not drift.

Case B - bonded harmonic oscillator (elasticity only)
    Two free particles joined by one elastic bond of stiffness ``k`` with
    ``m1 = m2 = m``.  For the relative stretch
    ``xi = |x1 - x0| - L0``::

        xi'' = -omega^2 xi,   omega = sqrt(2k/m)   ->  xi(t) = A cos(omega t)

    No drag, no external loads, no failure: the mechanical energy must be
    constant, so the ledger residual measures purely the integrator's
    energy behaviour (bounded oscillation, no secular drift).

Both are integrated with the solver's own step (semi-implicit Euler), so the
reported numbers quantify the ledger *as used*, not an idealised scheme.

Run
---
    PYTHONPATH=src python scripts/verify_dem3d_energy_ledger_analytic.py \
        --output results-dem3d-energy-ledger-analytic

Artifacts: ``energy_ledger_analytic.json``, ``energy_ledger_dt_sweep.csv`` and
``energy_ledger_analytic.png`` in the output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from tensordem.dem3d import DEM3DConfig
from tensordem.dem3d_energy import EnergyAuditedIceDEM3D

# Reference values fixed here (not silently re-derived inside the tests) so the
# verification can be reproduced independently.
RADIUS = 0.025
DENSITY = 917.0
DRAG = 2.0
BOND_STIFFNESS = 2_000.0
V0 = 0.5
OSC_AMPLITUDE = 2.5e-4
OSC_PERIODS = 3.0
DRAG_END_IN_TAU = 1.5

# Fracture is switched off for the oscillator so that the case isolates linear
# elasticity.  This is not a convenience: a two-particle (single-bond) cluster
# has a rank-deficient local reference configuration, and the fitted
# Green-Lagrange invariant of ``IceDEM3D._objective_shear_strain`` then returns
# 1/3 even for a rigid, undeformed bond (see ``probe_shear_invariant``), which
# instantly trips the default 3 % shear limit.  Thresholds are therefore lifted
# far above the artifact so that the dynamics stay purely elastic.
BREAKING_STRAIN_DISABLED = 10.0
SHEAR_STRAIN_DISABLED = 10.0

# The tool is parked far above the specimen and its velocity is effectively
# zero (DEM3DConfig rejects tool_speed <= 0), so neither case sees tool contact.
TOOL_SPEED_IDLE = 1e-9
TOOL_GAP_IDLE = 10.0


def base_config(dt: float | None = None, **overrides: Any) -> DEM3DConfig:
    params: dict[str, Any] = dict(
        nx=2,
        ny=2,
        nz=2,
        radius=RADIUS,
        density=DENSITY,
        drag=DRAG,
        tool_speed=TOOL_SPEED_IDLE,
        tool_gap=TOOL_GAP_IDLE,
        fix_x_edges=False,
        dt=dt,
    )
    params.update(overrides)
    return DEM3DConfig(**params)


def build_subsystem(
    config: DEM3DConfig,
    positions: list[list[float]],
    velocities: list[list[float]],
    bonds: list[tuple[int, int]],
    rest_lengths: list[float] | None = None,
) -> EnergyAuditedIceDEM3D:
    """Replace the lattice state of a fresh solver with an explicit subsystem.

    The shipped solver only builds lattice geometries (minimum 2x2x2 = 8
    particles), so the closed-form cases are obtained by overwriting the state
    arrays and then resetting the ledger reference energy.  No solver source is
    modified, and the integration path (``step``) is the production one.
    """
    solver = EnergyAuditedIceDEM3D(config)
    n = len(positions)
    pos = torch.tensor(positions, dtype=solver.dtype)
    vel = torch.tensor(velocities, dtype=solver.dtype)
    pairs = torch.tensor(bonds, dtype=torch.long).reshape(-1, 2)

    solver.positions = pos.clone()
    solver.initial_positions = pos.clone()
    solver.velocities = vel.clone()
    solver.fixed = torch.zeros(n, dtype=torch.bool)
    solver.pairs = pairs
    solver.bond_indices = pairs
    solver.bonded = torch.ones(pairs.shape[0], dtype=torch.bool)
    solver.alive = solver.bonded.clone()

    if pairs.numel():
        i, j = pairs.T
        reference = pos[j] - pos[i]
        if rest_lengths is None:
            lengths = torch.linalg.vector_norm(reference, dim=1)
        else:
            lengths = torch.tensor(rest_lengths, dtype=solver.dtype)
    else:
        reference = torch.zeros((0, 3), dtype=solver.dtype)
        lengths = torch.zeros(0, dtype=solver.dtype)
    solver.rest_lengths = lengths
    solver.reference_bond_vectors = reference.clone()
    solver.initial_normals = reference / lengths.clamp_min(torch.finfo(solver.dtype).eps)[:, None]
    solver.failure_mode = torch.zeros(pairs.shape[0], dtype=torch.uint8)
    solver.failure_time_s = torch.full((pairs.shape[0],), -1.0, dtype=solver.dtype)
    solver.failure_extension_m = torch.zeros(pairs.shape[0], dtype=solver.dtype)
    solver.failure_energy_J = torch.zeros(pairs.shape[0], dtype=solver.dtype)
    solver._bond_keys = pairs[:, 0] * n + pairs[:, 1]

    solver.time = 0.0
    solver.step_count = 0
    solver.cumulative_tool_work_J = 0.0
    solver.cumulative_external_work_J = 0.0
    solver.cumulative_drag_dissipation_J = 0.0
    solver.cumulative_particle_damping_J = 0.0
    solver.cumulative_tool_damping_J = 0.0
    solver.cumulative_friction_dissipation_J = 0.0
    solver.cumulative_fracture_release_J = 0.0
    solver.reference_mechanical_energy_J = solver._mechanical_energy()
    solver.energy_balance_residual_J = 0.0
    return solver


# ---------------------------------------------------------------------------
# Case A: exponential viscous decay
# ---------------------------------------------------------------------------

def run_drag_decay(dt: float, end_in_tau: float = DRAG_END_IN_TAU) -> dict[str, Any]:
    """Integrate a single free particle with linear drag and score the ledger."""
    config = base_config(dt=dt)
    solver = build_subsystem(
        config,
        positions=[[0.0, 0.0, 0.0]],
        velocities=[[V0, 0.0, 0.0]],
        bonds=[],
    )
    m = config.mass
    tau = m / DRAG
    t_end = end_in_tau * tau
    steps = int(round(t_end / solver.dt))

    history = []
    for _ in range(steps):
        solver.step()
    diag = solver.diagnostics()
    t = solver.time

    v_sim = float(solver.velocities[0, 0])
    v_ref = V0 * math.exp(-t / tau)
    d_sim = diag["drag_dissipation_cumulative_J"]
    d_ref = 0.5 * m * V0**2 * (1.0 - math.exp(-2.0 * t / tau))
    ke_sim = 0.5 * m * v_sim**2
    ke_ref = 0.5 * m * v_ref**2

    return {
        "case": "drag_decay",
        "dt_s": solver.dt,
        "steps": steps,
        "time_s": t,
        "tau_s": tau,
        "velocity_sim_m_s": v_sim,
        "velocity_reference_m_s": v_ref,
        "velocity_rel_error_pct": 100.0 * abs(v_sim - v_ref) / v_ref,
        "kinetic_energy_sim_J": ke_sim,
        "kinetic_energy_reference_J": ke_ref,
        "drag_dissipation_sim_J": d_sim,
        "drag_dissipation_reference_J": d_ref,
        "drag_dissipation_rel_error_pct": 100.0 * abs(d_sim - d_ref) / d_ref,
        "energy_balance_residual_J": diag["energy_balance_residual_J"],
        "energy_balance_residual_relative": diag["energy_balance_residual_relative"],
        # Closed-form first-order prediction of the ledger residual: the drag
        # rate is sampled at the pre-update velocity, so each step over-records
        # 0.5 * c * v^2 * (dt c / m) * dt, i.e. a fraction 0.5 dt c / m of the
        # dissipated energy.
        "residual_predicted_J": 0.5 * (DRAG / m) * solver.dt * d_ref,
        "residual_prediction_rel_error_pct": 100.0 * abs(
            diag["energy_balance_residual_J"] - 0.5 * (DRAG / m) * solver.dt * d_ref
        ) / (0.5 * (DRAG / m) * solver.dt * d_ref),
        "reference_mechanical_energy_J": solver.reference_mechanical_energy_J,
        "particle_damping_J": diag["particle_damping_cumulative_J"],
        "tool_work_J": diag["tool_work_cumulative_J"],
        "history": history,
    }


# ---------------------------------------------------------------------------
# Case B: two-particle bonded harmonic oscillator
# ---------------------------------------------------------------------------

def run_oscillator(dt: float, periods: float = OSC_PERIODS) -> dict[str, Any]:
    """Integrate one elastic bond between two free particles; compare to cos."""
    config = base_config(
        dt=dt,
        drag=0.0,
        breaking_strain=BREAKING_STRAIN_DISABLED,
        shear_breaking_strain=SHEAR_STRAIN_DISABLED,
    )
    rest = 2.0 * RADIUS
    gap = OSC_AMPLITUDE / 2.0
    solver = build_subsystem(
        config,
        positions=[[-rest / 2.0 - gap, 0.0, 0.0], [rest / 2.0 + gap, 0.0, 0.0]],
        velocities=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        bonds=[(0, 1)],
        rest_lengths=[rest],
    )
    m = config.mass
    omega = math.sqrt(2.0 * BOND_STIFFNESS / m)
    period = 2.0 * math.pi / omega
    steps_per_period = int(round(period / solver.dt))
    steps = int(round(periods * steps_per_period))

    energy_history = []
    xi_sim = []
    xi_ref = []
    for _ in range(steps):
        solver.step()
        energy_history.append(solver._mechanical_energy())
        # Every step is recorded: the period estimate is a least-squares fit over
        # all upward zero crossings, so sub-sampling would bias it.
        xi = float(solver.positions[1, 0] - solver.positions[0, 0]) - rest
        xi_sim.append((solver.time, xi))
        xi_ref.append((solver.time, OSC_AMPLITUDE * math.cos(omega * solver.time)))

    diag = solver.diagnostics()
    t = solver.time
    xi_final = float(solver.positions[1, 0] - solver.positions[0, 0]) - rest
    xi_final_ref = OSC_AMPLITUDE * math.cos(omega * t)

    # Measure the simulated period from upward zero crossings of the stretch.
    zero_crossings = []
    prev_t: float | None = None
    prev_xi: float | None = None
    for time_s, xi in xi_sim:
        if prev_xi is not None and prev_xi < 0.0 <= xi:
            frac = -prev_xi / (xi - prev_xi)
            zero_crossings.append(prev_t + frac * (time_s - prev_t))
        prev_t, prev_xi = time_s, xi
    period_sim = None
    if len(zero_crossings) >= 3:
        # Least-squares slope of crossing time vs crossing index.
        idx = list(range(len(zero_crossings)))
        n_c = len(idx)
        mean_i = sum(idx) / n_c
        mean_t = sum(zero_crossings) / n_c
        num = sum((i - mean_i) * (t - mean_t) for i, t in zip(idx, zero_crossings))
        den = sum((i - mean_i) ** 2 for i in idx)
        period_sim = num / den if den > 0 else None

    energies = torch.tensor(energy_history, dtype=solver.dtype)
    e_mean = float(energies.mean())
    e_spread = float(energies.max() - energies.min())
    max_shear = float(solver._objective_shear_strain().max()) if solver.pairs.numel() else 0.0

    return {
        "case": "bonded_oscillator",
        "dt_s": solver.dt,
        "steps": steps,
        "time_s": t,
        "periods": periods,
        "steps_per_period": steps_per_period,
        "omega_rad_s": omega,
        "period_reference_s": period,
        "period_sim_s": period_sim,
        "period_rel_error_pct": None if period_sim is None else 100.0 * abs(period_sim - period) / period,
        "amplitude_reference_m": OSC_AMPLITUDE,
        "stretch_sim_m": xi_final,
        "stretch_reference_m": xi_final_ref,
        "stretch_abs_error_m": abs(xi_final - xi_final_ref),
        "mechanical_energy_mean_J": e_mean,
        "mechanical_energy_spread_J": e_spread,
        "mechanical_energy_spread_relative": e_spread / max(abs(e_mean), 1e-30),
        # Symplectic Euler conserves a modified Hamiltonian, so the sampled
        # energy oscillates with relative amplitude dt*omega instead of drifting.
        "energy_spread_predicted_relative": solver.dt * omega,
        "energy_spread_prediction_rel_error_pct": 100.0 * abs(
            e_spread / max(abs(e_mean), 1e-30) - solver.dt * omega
        ) / (solver.dt * omega),
        "max_objective_shear": max_shear,
        "alive_bonds": int(solver.alive.sum()),
        "energy_balance_residual_J": diag["energy_balance_residual_J"],
        "energy_balance_residual_relative": diag["energy_balance_residual_relative"],
        "tool_work_J": diag["tool_work_cumulative_J"],
        "drag_dissipation_J": diag["drag_dissipation_cumulative_J"],
        "stretch_series": xi_sim,
        "stretch_reference_series": xi_ref,
    }


# ---------------------------------------------------------------------------
# Probe: is the objective shear invariant trustworthy on the tested geometry?
# ---------------------------------------------------------------------------

def probe_shear_invariant() -> dict[str, Any]:
    """Score ``IceDEM3D._objective_shear_strain`` against closed-form strains.

    A pure verification check of the fracture criterion's strain measure:

    P1 rank-deficient neighbourhood (one bond, undeformed): the least-squares
       deformation gradient must be the identity, so the invariant must vanish.
    P2 full cubic lattice, uniform uniaxial stretch ``F = diag(1+e, 1, 1)``:
       ``E = 0.5 (F^T F - I)``, invariant ``sqrt(2/3 dev(E):dev(E))`` has the
       closed form below.
    P3 full cubic lattice, rigid rotation: frame indifference requires 0.
    """
    from tensordem.dem3d import IceDEM3D

    # P1
    collinear_cfg = base_config(drag=0.0)
    collinear = build_subsystem(
        collinear_cfg,
        positions=[[-RADIUS, 0.0, 0.0], [RADIUS, 0.0, 0.0]],
        velocities=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        bonds=[(0, 1)],
        rest_lengths=[2.0 * RADIUS],
    )
    p1 = float(collinear._objective_shear_strain().abs().max())

    # P2 / P3
    lattice = IceDEM3D(DEM3DConfig())
    reference = lattice.positions.clone()
    lattice.positions = reference.clone()
    p2_undeformed = float(lattice._objective_shear_strain().abs().max())

    strain = 0.01
    lattice.positions = reference * torch.tensor([1.0 + strain, 1.0, 1.0], dtype=lattice.dtype)
    p2_code = float(lattice._objective_shear_strain().mean())
    principal = [0.5 * ((1.0 + strain) ** 2 - 1.0), 0.0, 0.0]
    mean_principal = sum(principal) / 3.0
    p2_analytic = math.sqrt((2.0 / 3.0) * sum((v - mean_principal) ** 2 for v in principal))

    angle = math.radians(30.0)
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle), 0.0],
         [math.sin(angle), math.cos(angle), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=lattice.dtype,
    )
    lattice.positions = reference @ rotation.T
    p3 = float(lattice._objective_shear_strain().abs().max())

    return {
        "case": "shear_invariant_probe",
        "collinear_undeformed_shear": p1,
        "collinear_degenerate": p1 > 1e-6,
        "lattice_undeformed_shear": p2_undeformed,
        "lattice_uniaxial_strain": strain,
        "lattice_uniaxial_shear_code": p2_code,
        "lattice_uniaxial_shear_analytic": p2_analytic,
        "lattice_uniaxial_rel_error_pct": 100.0 * abs(p2_code - p2_analytic) / p2_analytic,
        "lattice_rigid_rotation_shear": p3,
        "lattice_particles": int(len(reference)),
    }


# ---------------------------------------------------------------------------
# Step-size convergence of the ledger
# ---------------------------------------------------------------------------

def convergence_sweep(refinements: int = 5) -> list[dict[str, Any]]:
    """Refine dt and record the error of the ledger and the residual."""
    base = base_config()
    rows: list[dict[str, Any]] = []
    for level in range(refinements):
        dt = base.recommended_dt / (2**level)
        decay = run_drag_decay(dt)
        osc = run_oscillator(dt, periods=OSC_PERIODS)
        rows.append(
            {
                "level": level,
                "dt_s": decay["dt_s"],
                "drag_decay_steps": decay["steps"],
                "velocity_rel_error_pct": decay["velocity_rel_error_pct"],
                "drag_dissipation_rel_error_pct": decay["drag_dissipation_rel_error_pct"],
                "drag_residual_relative": decay["energy_balance_residual_relative"],
                "oscillator_steps": osc["steps"],
                "oscillator_period_rel_error_pct": osc["period_rel_error_pct"],
                "oscillator_energy_spread_relative": osc["mechanical_energy_spread_relative"],
                "oscillator_energy_spread_predicted": osc["energy_spread_predicted_relative"],
                "drag_residual_predicted": decay["residual_predicted_J"],
                "oscillator_residual_relative": abs(osc["energy_balance_residual_relative"]),
            }
        )
    return rows


def observed_order(values: list[float], dts: list[float]) -> float | None:
    """Least-squares slope of log(error) vs log(dt) over the refined levels."""
    pairs = [(math.log(d), math.log(v)) for d, v in zip(dts, values) if v > 0.0]
    if len(pairs) < 2:
        return None
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    den = sum((p[0] - mx) ** 2 for p in pairs)
    return num / den if den > 0 else None


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(decay: dict[str, Any], osc: dict[str, Any], rows: list[dict[str, Any]],
                probe: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    # (a) case A: closed-form viscous decay
    ax = axes[0][0]
    tau = decay["tau_s"]
    t_end = decay["time_s"]
    grid = [t_end * i / 200.0 for i in range(201)]
    ax.plot([t / tau for t in grid], [math.exp(-t / tau) for t in grid], "k-",
            label="analytic exp(-t/tau)")
    ax.plot([t_end / tau], [decay["velocity_sim_m_s"] / V0], "o", ms=7,
            label=f"DEM3D (err {decay['velocity_rel_error_pct']:.3f} %)")
    ax.set_xlabel("t / tau")
    ax.set_ylabel("v / v0")
    ax.set_title("A: viscous decay, drag only")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (b) case A: ledger dissipation vs closed form
    ax = axes[0][1]
    d_ref = decay["drag_dissipation_reference_J"]
    ax.plot([t / tau for t in grid],
            [0.5 * decay["kinetic_energy_sim_J"] * 0.0 + d_ref * (1.0 - math.exp(-2.0 * t / tau))
             / (1.0 - math.exp(-2.0 * t_end / tau)) for t in grid], "k-",
            label="analytic D(t)")
    ax.plot([t_end / tau], [decay["drag_dissipation_sim_J"]], "s", ms=7,
            label=f"ledger ({decay['drag_dissipation_rel_error_pct']:.3f} % off)")
    ax.set_xlabel("t / tau")
    ax.set_ylabel("cumulative dissipation [J]")
    ax.set_title("A: drag dissipation ledger vs analytic")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # (c) case B: harmonic oscillator
    ax = axes[1][0]
    period = osc["period_reference_s"]
    ax.plot([p[0] / period for p in osc["stretch_series"]],
            [p[1] / OSC_AMPLITUDE for p in osc["stretch_series"]], "-", lw=1.0,
            label="DEM3D bond stretch")
    ax.plot([p[0] / period for p in osc["stretch_reference_series"]],
            [p[1] / OSC_AMPLITUDE for p in osc["stretch_reference_series"]], "k--", lw=1.2,
            label="analytic cos(omega t)")
    ax.set_xlabel("t / T")
    ax.set_ylabel("xi / A")
    ax.set_title(f"B: bonded oscillator (period err {osc['period_rel_error_pct']:.2e} %)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (d) step-size convergence
    ax = axes[1][1]
    dts = [r["dt_s"] for r in rows]
    ax.loglog(dts, [r["velocity_rel_error_pct"] for r in rows], "o-", label="A: v error [%]")
    ax.loglog(dts, [r["drag_dissipation_rel_error_pct"] for r in rows], "s-",
              label="A: ledger D error [%]")
    ax.loglog(dts, [abs(r["drag_residual_relative"]) * 100.0 for r in rows], "^-",
              label="A: |residual| / E0 [%]")
    ax.loglog(dts, [r["oscillator_energy_spread_relative"] * 100.0 for r in rows], "d-",
              label="B: energy spread [%] (= dt*omega)")
    ax.loglog(dts, [r["oscillator_period_rel_error_pct"] for r in rows], "v-",
              label="B: period error [%]")
    ref_dt = dts[0]
    guide = [rows[0]["velocity_rel_error_pct"] * (d / ref_dt) for d in dts]
    ax.loglog(dts, guide, "k:", lw=1.0, label="slope 1 guide")
    guide2 = [rows[0]["oscillator_period_rel_error_pct"] * (d / ref_dt) ** 2 for d in dts]
    ax.loglog(dts, guide2, "k-.", lw=1.0, label="slope 2 guide")
    ax.set_xlabel("dt [s]")
    ax.set_ylabel("error [%]")
    ax.set_title("step-size convergence")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        "TensorDEM cumulative energy ledger: closed-form verification "
        f"(collinear shear artifact = {probe['collinear_undeformed_shear']:.3f})",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-energy-ledger-analytic"))
    parser.add_argument("--refinements", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    decay = run_drag_decay(base_config().recommended_dt)
    osc = run_oscillator(base_config().recommended_dt)
    probe = probe_shear_invariant()
    rows = convergence_sweep(args.refinements)

    dts = [r["dt_s"] for r in rows]
    orders = {
        "velocity_rel_error": observed_order([r["velocity_rel_error_pct"] for r in rows], dts),
        "drag_dissipation_rel_error": observed_order(
            [r["drag_dissipation_rel_error_pct"] for r in rows], dts),
        "drag_residual_relative": observed_order([r["drag_residual_relative"] for r in rows], dts),
        "oscillator_period_rel_error": observed_order(
            [r["oscillator_period_rel_error_pct"] for r in rows], dts),
    }

    payload = {
        "solver": "tensordem.dem3d_energy.EnergyAuditedIceDEM3D",
        "material": {"radius_m": RADIUS, "density_kg_m3": DENSITY, "mass_kg": base_config().mass},
        "drag_decay": {k: v for k, v in decay.items() if k not in ("history", "stretch_series")},
        "oscillator": {k: v for k, v in osc.items()
                       if k not in ("stretch_series", "stretch_reference_series")},
        "shear_invariant_probe": probe,
        "convergence": rows,
        "observed_order": orders,
    }
    json_path = args.output / "energy_ledger_analytic.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    csv_path = args.output / "energy_ledger_dt_sweep.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    make_figure(decay, osc, rows, probe, args.output / "energy_ledger_analytic.png")

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {args.output / 'energy_ledger_analytic.png'}")
    print(json.dumps({
        "velocity_rel_error_pct": decay["velocity_rel_error_pct"],
        "drag_dissipation_rel_error_pct": decay["drag_dissipation_rel_error_pct"],
        "drag_residual_relative": decay["energy_balance_residual_relative"],
        "oscillator_period_rel_error_pct": osc["period_rel_error_pct"],
        "oscillator_energy_spread_relative": osc["mechanical_energy_spread_relative"],
        "oscillator_residual_relative": osc["energy_balance_residual_relative"],
        "orders": orders,
        "probe": probe,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())