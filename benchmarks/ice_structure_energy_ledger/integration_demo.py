"""Integration demo: four-ledger audit driven by the DEM main solver.

The audit module certifies *itself* against independent references.  This script
certifies the *integration*: it attaches the audit to
``tensordem.dem3d.IceDEM3D`` (the production bonded-sphere ice solver) and asks
whether the main solver's own energy bookkeeping closes the four ledgers.

Cases (all: 24-particle free floe, 1 m/s impact on a static spherical tool,
no gravity, no drag, no friction, no platens, ``restitution = 1``):

``closure_undamped``   ``contact_damping = 0`` -- the only channels left are the
                       elastic springs and the fracture ledger, so the residual
                       is the solver's own explicit-Euler drift.  Run at three
                       time steps to show the residual halves with ``dt``.
``production_damped``  solver defaults (``contact_damping = 5`` N s/m) with the
                       tool/bow dashpot *not* accounted for: the residual is the
                       audit flagging an unaccounted dissipative channel.
``production_damped_accounted``
                       same run, but the caller integrates the tool dashpot
                       dissipation from public state and feeds it into ledger 4:
                       closure must come back below 1 %, which proves the
                       residual was the missing channel and not numerical noise.

``trajectory_neutrality``  an audited and an unaudited run must be bit-for-bit
                       identical, i.e. auditing does not change the physics.

Usage:  python benchmarks/ice_structure_energy_ledger/integration_demo.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.dem3d import DEM3DConfig, IceDEM3D  # noqa: E402
from tensordem.ice_structure_energy_audit_integration import (  # noqa: E402
    IceDEMEnergyAuditAdapter,
    attach_energy_audit,
)

OUT = ROOT / "docs" / "artifacts" / "ice_structure_energy_ledger"
OUT.mkdir(parents=True, exist_ok=True)

IMPACT_VELOCITY = 1.0      # m/s, +z towards the static tool
STEPS = 400
GRID = (4, 3, 2)


def _config(*, dt: float | None = None, contact_damping: float = 0.0) -> DEM3DConfig:
    return DEM3DConfig(
        nx=GRID[0], ny=GRID[1], nz=GRID[2],
        radius=0.025, density=917.0,
        bond_stiffness=2000.0, contact_stiffness=5000.0,
        breaking_strain=0.05, shear_breaking_strain=0.10,
        contact_damping=contact_damping,
        drag=0.0, restitution_coefficient=1.0,
        friction_coefficient=0.0,
        tool_radius=0.1, tool_speed=1.0e-12, tool_gap=0.002,
        fix_x_edges=False, fix_bottom=False, gravity=0.0,
        dt=dt,
    )


def _build(cfg: DEM3DConfig) -> IceDEM3D:
    sim = IceDEM3D(cfg)
    sim.velocities[:, 2] = IMPACT_VELOCITY
    return sim


def tool_dashpot_rate(sim: IceDEM3D) -> float:
    """Dissipation rate of the tool dashpot, from public solver state only."""
    cfg = sim.config
    delta = sim.positions - sim.tool_position
    dist = torch.linalg.vector_norm(delta, dim=1)
    normal = delta / dist.clamp_min(1e-30)[:, None]
    overlap = (cfg.radius + cfg.tool_radius - dist).clamp_min(0.0)
    vn = ((sim.velocities - sim.tool_velocity) * normal).sum(dim=1)
    mask = (overlap > 0.0) & (vn < 0.0)
    return float((cfg.contact_damping * vn * vn * mask).sum())


def run_case(*, dt: float | None = None, contact_damping: float,
             account_tool_dashpot: bool, steps: int = STEPS):
    cfg = _config(dt=dt, contact_damping=contact_damping)
    sim = _build(cfg)
    adapter = IceDEMEnergyAuditAdapter()
    attach_energy_audit(sim, adapter)
    for _ in range(steps):
        if account_tool_dashpot:
            # left-endpoint quadrature on the pre-step state, the same
            # convention the audit module itself uses for ledger 4
            adapter.extra_viscous_dissipation_J += tool_dashpot_rate(sim) * sim.dt
        sim.step()
    return sim, adapter


def trajectory_neutrality(steps: int = 60) -> dict[str, float]:
    """Audited vs unaudited run from identical initial data."""
    plain = _build(_config())
    for _ in range(steps):
        plain.step()
    audited = _build(_config())
    adapter = IceDEMEnergyAuditAdapter()
    attach_energy_audit(audited, adapter)
    for _ in range(steps):
        audited.step()
    dpos = float((plain.positions - audited.positions).abs().max())
    dvel = float((plain.velocities - audited.velocities).abs().max())
    return {
        "steps": steps,
        "max_abs_position_difference_m": dpos,
        "max_abs_velocity_difference_m_s": dvel,
        "bitwise_identical": bool(dpos == 0.0 and dvel == 0.0),
        "audit_rows": len(adapter.rows),
    }


def _finite(x):
    return None if (isinstance(x, float) and not np.isfinite(x)) else x


def summarise(name: str, sim: IceDEM3D, adapter: IceDEMEnergyAuditAdapter,
              extra: dict | None = None) -> dict:
    s = adapter.summary()
    out = {
        "case": name,
        "dt_s": float(sim.dt),
        "n_particles": int(sim.positions.shape[0]),
        "n_bonds": int(sim.pairs.shape[0]),
        "n_broken": int(sim.broken_bonds),
        "residual_relative_max": s["residual_relative_max"],
        "closed_within_1pct": s["closed_within_1pct"],
        "ledger_final": {k: s[k] for k in ("E_k", "E_s", "E_f", "E_v", "E_ext",
                                           "E_total", "E_total_0")},
        "kinetic_cross_check_max": s["kinetic_cross_check_max"],
        "extra_viscous_dissipation_J": s["extra_viscous_dissipation_J"],
        "external_work_source": s["external_work_source"],
    }
    if extra:
        out.update(extra)
    return out


def make_figure(cases: dict[str, dict], neutrality: dict, path: Path,
                histories: dict[str, list[dict]]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    ax = axes[0]
    colors = {"closure_undamped": "C0", "production_damped": "C3",
              "production_damped_accounted": "C2"}
    for key, rows in histories.items():
        t = np.array([r["t"] for r in rows]) * 1.0e3
        res = np.array([r["residual_relative"] for r in rows]) * 100.0
        ax.plot(t, res, color=colors.get(key, "C7"), lw=1.6,
                label=key.replace("_", " "))
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="1 % acceptance bar")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(r"four-ledger residual $\varepsilon_E$ [%]")
    ax.set_title("main solver (dem3d) closure, audited in the step loop")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    dts = [c["dt_s"] for c in cases.values() if c["case"].startswith("closure_undamped")]
    res = [c["residual_relative_max"] * 100.0 for c in cases.values()
           if c["case"].startswith("closure_undamped")]
    order = np.argsort(dts)
    dts = np.array(dts)[order]
    res = np.array(res)[order]
    ax.loglog(dts, res, "o-", label="undamped closure run")
    if len(res) > 1:
        ref = res[0] * (dts / dts[0])
        ax.loglog(dts, ref, "k--", lw=1.0, label=r"slope 1 ($\propto \Delta t$)")
    ax.set_xlabel(r"time step $\Delta t$ [s]")
    ax.set_ylabel(r"$\varepsilon_E^{\max}$ [%]")
    ax.set_title("explicit-Euler drift is first order in dt")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def main() -> dict:
    cases: dict[str, dict] = {}
    histories: dict[str, list[dict]] = {}

    sim, adapter = run_case(contact_damping=0.0, account_tool_dashpot=False)
    cases["closure_undamped"] = summarise("closure_undamped", sim, adapter)
    histories["closure_undamped"] = adapter.rows

    base_dt = float(sim.dt)
    for factor in (2, 4):
        key = f"closure_undamped_dt_over_{factor}"
        s2, a2 = run_case(dt=base_dt / factor, contact_damping=0.0,
                          account_tool_dashpot=False, steps=STEPS * factor)
        cases[key] = summarise(key, s2, a2)
        histories[key] = a2.rows

    sim3, a3 = run_case(contact_damping=5.0, account_tool_dashpot=False)
    cases["production_damped"] = summarise("production_damped", sim3, a3)
    histories["production_damped"] = a3.rows

    sim4, a4 = run_case(contact_damping=5.0, account_tool_dashpot=True)
    cases["production_damped_accounted"] = summarise(
        "production_damped_accounted", sim4, a4)
    histories["production_damped_accounted"] = a4.rows

    neutrality = trajectory_neutrality()

    fig = make_figure(cases, neutrality,
                      OUT / "fig6_main_solver_integration.png", histories)

    undamped = [c for k, c in cases.items() if k.startswith("closure_undamped")]
    undamped.sort(key=lambda c: c["dt_s"])
    drift_halving = {
        "dts": [c["dt_s"] for c in undamped],
        "residuals": [c["residual_relative_max"] for c in undamped],
        "ratios": [b["residual_relative_max"] / a["residual_relative_max"]
                   for a, b in zip(undamped, undamped[1:])],
        "first_order": all(1.5 <= r <= 2.5 for r in
                           [b["residual_relative_max"] / a["residual_relative_max"]
                            for a, b in zip(undamped, undamped[1:])]),
    }

    criteria = {
        "i_audit_called_by_main_solver": {
            "value": len(a4.rows),
            "expected": STEPS + 1,
            "pass": len(a4.rows) == STEPS + 1,
        },
        "j_audit_does_not_change_trajectory": {
            "value": neutrality["max_abs_position_difference_m"],
            "threshold": 0.0,
            "pass": neutrality["bitwise_identical"],
        },
        "k_closure_undamped_main_solver_lt_1pct": {
            "value": cases["closure_undamped"]["residual_relative_max"],
            "threshold": 1.0e-2,
            "pass": cases["closure_undamped"]["residual_relative_max"] < 1.0e-2,
        },
        "l_missing_channel_detected_and_attributed": {
            "unaccounted_residual": cases["production_damped"]["residual_relative_max"],
            "accounted_residual": cases["production_damped_accounted"]["residual_relative_max"],
            "pass": (cases["production_damped"]["residual_relative_max"] > 1.0e-2
                     and cases["production_damped_accounted"]["residual_relative_max"] < 1.0e-2),
        },
        "m_kinetic_ledger_matches_solver_lt_0.1pct": {
            "value": cases["closure_undamped"]["kinetic_cross_check_max"],
            "threshold": 1.0e-3,
            "pass": cases["closure_undamped"]["kinetic_cross_check_max"] < 1.0e-3,
        },
        "n_explicit_euler_drift_is_first_order": {
            "detail": drift_halving,
            "pass": drift_halving["first_order"],
        },
    }

    report = {
        "node": "P304W2-N03",
        "capability": "ice-structure contact-collision four-ledger energy audit",
        "integration_target": "tensordem.dem3d.IceDEM3D.step (production DEM main loop)",
        "hook": "sim.energy_audit (None by default; set by attach_energy_audit)",
        "scenario": {
            "grid": list(GRID),
            "n_particles": cases["closure_undamped"]["n_particles"],
            "n_bonds": cases["closure_undamped"]["n_bonds"],
            "impact_velocity_m_s": IMPACT_VELOCITY,
            "steps": STEPS,
            "dt_auto_s": base_dt,
        },
        "cases": cases,
        "criteria": criteria,
        "all_criteria_pass": all(c["pass"] for c in criteria.values()),
        "trajectory_neutrality": neutrality,
        "figure": fig,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    text = json.dumps(report, indent=2, default=_finite)
    path = OUT / "integration_demo.json"
    path.write_text(text)
    report["integration_json"] = str(path.relative_to(ROOT))
    report["integration_json_sha256"] = hashlib.sha256(text.encode()).hexdigest()

    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk in ("value", "threshold", "pass")}
                      for k, v in criteria.items()}, indent=2, default=_finite))
    print("\nall_criteria_pass:", report["all_criteria_pass"])
    print("integration:", report["integration_json"])
    return report


if __name__ == "__main__":
    main()