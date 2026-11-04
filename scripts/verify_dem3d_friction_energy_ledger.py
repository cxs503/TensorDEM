"""Closed-form verification of the friction channel in the DEM3D energy ledger.

Why this exists
---------------
``EnergyAuditedIceDEM3D`` (``src/tensordem/dem3d_energy.py``) accumulated drag,
normal contact damping, tool damping and fracture release, but **no tangential
friction term**.  With ``friction_coefficient > 0`` the Coulomb channel does
real work on the relative motion, so ``energy_balance_residual_J`` silently
absorbed the entire friction work as a "residual" (debt flagged in
``docs/dem3d_tangential_frame`` / report m3468, limitation 2).  This script
scores the restored term against an independent closed-form reference:

System
    Two spheres of radius ``R`` and mass ``m``; particle 0 is a hard support,
    particle 1 starts in contact with overlap ``delta_n`` and initial
    tangential velocity ``v_t0``.  A tracking external force
    ``F_ext = k_n * overlap * n`` on particle 1 cancels the normal contact
    force at every step, so the normal overlap (and hence ``F_n``) is held
    constant to O(dt^2) and the tangential subsystem is isolated.

Reference (independent of the solver)
    The tangential coordinate ``x`` obeys the 1-D law

        m x'' = -k_t x                      (stick:  k_t x < mu F_n)
        m x'' = -mu F_n(x)                  (slip:   k_t x >= mu F_n)

    with ``F_n(x) = k_n (2R - sqrt(d^2 + x^2))``, ``d = 2R - delta_n``.
    The reference trajectory is integrated with RK4 at 200 000 steps over the
    run (``dt_ref << dt``) and the friction work is Simpson-quadred on that
    trajectory.  The stick phase additionally has the exact solution
    ``x(t) = (v_t0/omega) sin(omega t)``, ``omega = sqrt(k_t/m)``, which is
    used as a second, fully analytic anchor.

Cases
    S  stick-only (mu large, T < t_s):  D_f(T) = 0.5 k_t x(T)^2 exactly;
    L  stick-slip (mu = 0.3, T = 0.9 t_stop): D_f(T) vs the RK4 reference;
    B  balance closure: the mu > 0 residual must sit at the same
       discretisation floor as the mu = 0 control, while the pre-fix
       (friction-less) residual must equal the friction work;
    C  dt convergence of D_f(T) (first-order scheme -> ~2x per halving).

Run
---
    PYTHONPATH=src python scripts/verify_dem3d_friction_energy_ledger.py \
        --output results-dem3d-friction-energy-ledger

Artifacts: ``friction_energy_ledger.json``, ``friction_energy_dt_sweep.csv``
and ``friction_energy_ledger.png`` in the output directory.
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

# ---------------------------------------------------------------------------
# Fixed reference parameters (SI).  Chosen so that the friction work is
# comparable to the initial kinetic energy (good signal) while the geometric
# overlap drift over the whole run stays below 0.4 % of delta_n.
# ---------------------------------------------------------------------------
RADIUS = 0.05
DENSITY = 917.0
K_CONTACT = 5_000.0          # k_n = k_t (linear law, k_t = k_n in the solver)
DELTA_N = 2.0e-3
MU_SLIP = 0.3
V_T0 = 0.1
DT_BASE = 1.0e-4
TOOL_SPEED_IDLE = 1e-9
TOOL_GAP_IDLE = 10.0


def mass() -> float:
    return DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3


def normal_force(x: float) -> float:
    """Exact normal spring force for tangential travel x (geometry included)."""
    d = 2.0 * RADIUS - DELTA_N
    return K_CONTACT * (2.0 * RADIUS - math.hypot(d, x))


def friction_force_law(x: float, v: float, mu: float) -> float:
    """Tangential force on the free particle (negative = opposing motion)."""
    f_raw = -K_CONTACT * x
    f_limit = mu * normal_force(x)
    if abs(f_raw) > f_limit:
        return math.copysign(f_limit, f_raw)
    return f_raw


def reference_trajectory(mu: float, t_end: float, n_ref: int = 200_000) -> dict[str, list[float]]:
    """RK4 integration of the 1-D stick-slip ODE (independent reference)."""
    m = mass()
    dt = t_end / n_ref
    x, v = 0.0, V_T0
    ts = [0.0]
    xs = [0.0]
    vs = [V_T0]
    ws = [0.0]
    w = 0.0

    def accel(x_: float) -> float:
        return friction_force_law(x_, 0.0, mu) / m

    for _ in range(n_ref):
        k1v, k1x = accel(x), v
        k2v, k2x = accel(x + 0.5 * dt * k1x), v + 0.5 * dt * k1v
        k3v, k3x = accel(x + 0.5 * dt * k2x), v + 0.5 * dt * k2v
        k4v, k4x = accel(x + dt * k3x), v + dt * k3v
        x_new = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        v_new = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        # ledger-equivalent power: work of the force pair on the relative
        # motion = -(force on free particle) * v  (positive in both regimes)
        w += 0.5 * (-friction_force_law(x, v, mu) * v
                    - friction_force_law(x_new, v_new, mu) * v_new) * dt
        x, v = x_new, v_new
        ts.append(ts[-1] + dt)
        xs.append(x)
        vs.append(v)
        ws.append(w)
    return {"t": ts, "x": xs, "v": vs, "w": ws}


def stick_analytic(t: float) -> tuple[float, float]:
    """Exact stick-phase solution: harmonic oscillator about x = 0."""
    omega = math.sqrt(K_CONTACT / mass())
    x = (V_T0 / omega) * math.sin(omega * t)
    v = V_T0 * math.cos(omega * t)
    return x, v


def slip_onset(mu: float) -> tuple[float, float, float]:
    """(t_s, x_s, v_s): slip onset from k_t x_s = mu F_n(x_s)."""
    omega = math.sqrt(K_CONTACT / mass())
    # fixed point on x_s (F_n varies by < 0.1 % over the stick phase)
    x_s = mu * normal_force(0.0) / K_CONTACT
    for _ in range(50):
        x_new = mu * normal_force(x_s) / K_CONTACT
        if abs(x_new - x_s) < 1e-15:
            x_s = x_new
            break
        x_s = x_new
    t_s = math.asin(min(1.0, mu * normal_force(x_s) * omega / (K_CONTACT * V_T0))) / omega
    _, v_s = stick_analytic(t_s)
    return t_s, x_s, v_s


def build_friction_subsystem(config: DEM3DConfig, mu: float) -> EnergyAuditedIceDEM3D:
    """Two-particle contact subsystem driven through the production step()."""
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2,
        radius=RADIUS, density=DENSITY,
        bond_stiffness=2_000.0,
        contact_stiffness=K_CONTACT,
        contact_damping=0.0,
        restitution_coefficient=1.0,
        drag=0.0,
        friction_coefficient=mu,
        tool_speed=TOOL_SPEED_IDLE,
        tool_gap=TOOL_GAP_IDLE,
        fix_x_edges=False,
        dt=config.dt,
    )
    solver = EnergyAuditedIceDEM3D(cfg)
    d = 2.0 * RADIUS - DELTA_N
    pos = torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]], dtype=solver.dtype)
    vel = torch.tensor([[0.0, 0.0, 0.0], [0.0, V_T0, 0.0]], dtype=solver.dtype)
    pairs = torch.zeros((0, 2), dtype=torch.long)

    solver.positions = pos.clone()
    solver.initial_positions = pos.clone()
    solver.velocities = vel.clone()
    solver.fixed = torch.tensor([True, False])
    solver.pairs = pairs
    solver.bond_indices = pairs
    solver.bonded = torch.zeros(0, dtype=torch.bool)
    solver.alive = torch.zeros(0, dtype=torch.bool)
    solver.rest_lengths = torch.zeros(0, dtype=solver.dtype)
    solver.reference_bond_vectors = torch.zeros((0, 3), dtype=solver.dtype)
    solver.initial_normals = torch.zeros((0, 3), dtype=solver.dtype)
    solver.failure_mode = torch.zeros(0, dtype=torch.uint8)
    solver.failure_time_s = torch.zeros(0, dtype=solver.dtype)
    solver.failure_extension_m = torch.zeros(0, dtype=solver.dtype)
    solver.failure_energy_J = torch.zeros(0, dtype=solver.dtype)
    solver._bond_keys = torch.zeros(0, dtype=torch.long)
    solver.time = 0.0
    solver.step_count = 0
    solver.reference_mechanical_energy_J = solver._mechanical_energy()
    solver.energy_balance_residual_J = 0.0
    return solver


def tracking_external_force(solver: EnergyAuditedIceDEM3D) -> torch.Tensor:
    """External force that cancels the normal contact force on particle 1.

    The repulsive contact force on particle 1 is ``+k_n * overlap * n`` with
    ``n`` pointing from particle 0 to particle 1, so the cancelling load is
    ``-k_n * overlap * n``.
    """
    delta = solver.positions[1] - solver.positions[0]
    dist = torch.linalg.vector_norm(delta)
    n = delta / dist
    overlap = (2.0 * RADIUS - dist).clamp_min(0.0)
    f = torch.zeros_like(solver.positions)
    f[1] = -K_CONTACT * overlap * n
    return f


def run_case(mu: float, dt: float, t_end: float,
              record: bool = False) -> dict[str, Any]:
    """Integrate the subsystem and score the ledger against the reference."""
    solver = build_friction_subsystem(DEM3DConfig(dt=dt), mu)
    ref = reference_trajectory(mu, t_end)
    n_steps = int(round(t_end / dt))
    hist_t: list[float] = []
    hist_d: list[float] = []
    hist_p: list[float] = []
    hist_res: list[float] = []
    for _ in range(n_steps):
        solver.step(tracking_external_force(solver))
        if record:
            diag = solver.diagnostics()
            hist_t.append(solver.time)
            hist_d.append(diag["friction_dissipation_cumulative_J"])
            hist_p.append(diag["friction_power_instantaneous_W"])
            hist_res.append(diag["energy_balance_residual_J"])
    diag = solver.diagnostics()
    t = solver.time
    d_sim = diag["friction_dissipation_cumulative_J"]
    d_ref = ref["w"][-1]
    d_rel = abs(d_sim - d_ref) / d_ref if d_ref > 0.0 else abs(d_sim)
    # tangential travel from the stored Cundall-Strack history
    key = solver._pair_key(0, 1)
    tang = solver._tangential_state.get(key)
    x_sim = float(tang[0].norm()) if tang is not None else 0.0
    x_ref = ref["x"][-1]
    # exact stick-phase anchor: D_f(T) = 0.5 k_t x(T)^2 with the analytic
    # harmonic solution (valid whenever no slip occurred before T)
    x_a, _ = stick_analytic(t)
    d_analytic = 0.5 * K_CONTACT * x_a**2
    return {
        "mu": mu,
        "dt": dt,
        "time_s": t,
        "friction_dissipation_J": d_sim,
        "friction_dissipation_ref_J": d_ref,
        "friction_rel_error": d_rel,
        "tangential_travel_m": x_sim,
        "tangential_travel_ref_m": x_ref,
        "travel_rel_error": abs(x_sim - x_ref) / x_ref,
        "residual_J": diag["energy_balance_residual_J"],
        "residual_relative": diag["energy_balance_residual_relative"],
        "external_work_J": diag["external_work_cumulative_J"],
        "drag_dissipation_J": diag["drag_dissipation_cumulative_J"],
        "particle_damping_J": diag["particle_damping_cumulative_J"],
        "history": ({"t": hist_t, "d": hist_d, "p": hist_p, "res": hist_res}
                    if record else None),
        "stick_analytic_dissipation_J": d_analytic,
        "stick_analytic_rel_error": (
            abs(d_sim - d_analytic) / d_analytic if d_analytic > 0.0 else abs(d_sim - d_analytic)),
    }


def run_balance(dt: float) -> dict[str, Any]:
    """Balance closure: mu > 0 residual vs mu = 0 control and pre-fix residual."""
    t_s, x_s, v_s = slip_onset(MU_SLIP)
    m = mass()
    t_stop = t_s + v_s * m / (MU_SLIP * normal_force(x_s))
    t_end = 0.9 * t_stop

    slip = run_case(MU_SLIP, dt, t_end)
    ctrl = run_case(0.0, dt, t_end)
    # Pre-fix bookkeeping: the same run without the friction term in D.
    # residual_old = residual_new - D_f  (D_f enters with + sign in the balance)
    residual_old = slip["residual_J"] - slip["friction_dissipation_J"]
    # The pre-fix residual must be dominated by the missing friction work
    # (it appears with a minus sign: friction removes mechanical energy).
    residual_old = -residual_old
    return {
        "case": "balance_closure",
        "dt": dt,
        "time_s": t_end,
        "residual_new_J": slip["residual_J"],
        "residual_new_relative": slip["residual_relative"],
        "residual_control_mu0_J": ctrl["residual_J"],
        "residual_control_relative": ctrl["residual_relative"],
        "residual_prefix_J": residual_old,
        "friction_work_J": slip["friction_dissipation_J"],
        "prefix_residual_explains_friction_work":
            abs(abs(residual_old) - slip["friction_dissipation_J"])
            <= 0.10 * slip["friction_dissipation_J"]
            + 5.0 * abs(ctrl["residual_J"]),
    }


def convergence_sweep(base_dt: float = DT_BASE, n: int = 3) -> list[dict[str, Any]]:
    t_s, x_s, v_s = slip_onset(MU_SLIP)
    t_end = 0.9 * (t_s + v_s * mass() / (MU_SLIP * normal_force(x_s)))
    rows = []
    for i in range(n):
        dt = base_dt / (2.0 ** i)
        case = run_case(MU_SLIP, dt, t_end)
        rows.append({
            "dt": dt,
            "steps": int(round(t_end / dt)),
            "friction_dissipation_J": case["friction_dissipation_J"],
            "friction_rel_error": case["friction_rel_error"],
            "travel_rel_error": case["travel_rel_error"],
            "residual_relative": case["residual_relative"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results-dem3d-friction-energy-ledger")
    args = parser.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    m = mass()
    omega = math.sqrt(K_CONTACT / m)
    t_s, x_s, v_s = slip_onset(MU_SLIP)
    t_stop = t_s + v_s * m / (MU_SLIP * normal_force(x_s))
    t_slip_end = 0.9 * t_stop
    t_stick_end = 0.25 * (2.0 * math.pi / omega)

    stick = run_case(10.0, DT_BASE, t_stick_end, record=True)
    slip = run_case(MU_SLIP, DT_BASE, t_slip_end, record=True)
    balance = run_balance(DT_BASE)
    sweep = convergence_sweep()

    report = {
        "parameters": {
            "radius_m": RADIUS, "mass_kg": m, "k_n_k_t_N_m": K_CONTACT,
            "delta_n_m": DELTA_N, "F_n0_N": K_CONTACT * DELTA_N,
            "mu_slip": MU_SLIP, "v_t0_m_s": V_T0,
            "omega_s": omega, "t_s_s": t_s, "x_s_m": x_s,
            "v_s_m_s": v_s, "t_stop_s": t_stop,
            "dt_base_s": DT_BASE,
        },
        "stick": stick,
        "slip": slip,
        "balance": balance,
        "sweep": sweep,
    }
    (outdir / "friction_energy_ledger.json").write_text(
        json.dumps(report, indent=2))
    with (outdir / "friction_energy_dt_sweep.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
        ax = axes[0]
        ref = reference_trajectory(MU_SLIP, t_slip_end)
        ax.plot(ref["t"], [friction_force_law(x, v, MU_SLIP) * v
                          for x, v in zip(ref["x"], ref["v"])],
                 "k-", lw=1.2, label="reference $F_t v_t$")
        ax.plot(slip["history"]["t"], slip["history"]["p"], "r.", ms=3,
                label="ledger integrand")
        ax.axvline(t_s, color="gray", ls=":", lw=1)
        ax.text(t_s, ax.get_ylim()[1] * 0.9, " slip", rotation=90, fontsize=8)
        ax.set_xlabel("t [s]"); ax.set_ylabel("$F_t v_t$ [W]")
        ax.set_title("friction power"); ax.legend(fontsize=8)
        ax = axes[1]
        ax.plot(ref["t"], ref["w"], "k-", lw=1.2, label="reference $\\int F_t v_t$")
        ax.plot(slip["history"]["t"], slip["history"]["d"], "r.", ms=3,
                label="ledger $D_f$")
        ax.set_xlabel("t [s]"); ax.set_ylabel("$D_f$ [J]")
        ax.set_title("cumulative friction work"); ax.legend(fontsize=8)
        ax = axes[2]
        ax.plot(slip["history"]["t"], slip["history"]["res"], "r-", lw=1.2,
                label="$\\mu=0.3$, ledger with $D_f$")
        ax.plot(slip["history"]["t"],
                [r - d for r, d in zip(slip["history"]["res"], slip["history"]["d"])],
                "b--", lw=1.0, label="$\\mu=0.3$, pre-fix bookkeeping")
        ax.set_xlabel("t [s]"); ax.set_ylabel("balance residual [J]")
        ax.set_title("energy balance closure"); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "friction_energy_ledger.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # figures are optional for the numeric verdict
        report["figure_error"] = repr(exc)

    print(json.dumps({
        "stick_rel_error": stick["friction_rel_error"],
        "stick_analytic_rel_error": stick["stick_analytic_rel_error"],
        "slip_rel_error": slip["friction_rel_error"],
        "balance": {k: balance[k] for k in
                    ("residual_new_relative", "residual_control_relative",
                     "residual_prefix_J", "friction_work_J",
                     "prefix_residual_explains_friction_work")},
        "sweep_errors": [row["friction_rel_error"] for row in sweep],
    }, indent=2))


if __name__ == "__main__":
    main()
