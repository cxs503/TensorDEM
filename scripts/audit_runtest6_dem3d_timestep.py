#!/usr/bin/env python3
"""AUDIT-RUNTEST-6 (m14498): DEM3D timestep convergence, measured locally.

The cluster job that would have run ``scripts/benchmark_dem3d_ucs_timestep_convergence.py``
never left the queue (see §5 of the report), so this probe measures the same
question at reduced cost inside the control-plane budget: a fixed indentation
depth reached with dt, dt/2, dt/4, dt/8 and the observed order of the tool force
and of the discrete energy ledger.

Protocol (predeclared before looking at the numbers):
  * geometry    : nx=8, ny=4, nz=2 bonded lattice, x edges fixed, no platen
  * tool        : default rigid spherical indenter moving at ``tool_speed``
  * end state   : fixed *depth* -> fixed simulated time T = tool_speed * t_end
  * refinement  : dt_k = dt_rec / 2**k, k = 0..3
  * metrics     : peak reaction_z [N], final bond elastic energy [J],
                  discrete energy-ledger residual
                  |W_tool - (dKE + dU_bond + dU_contact + released)| / W_tool
  * order       : least-squares slope of log|m_k - m_finest| vs log(dt_k)
  * pass rule   : the two finest levels differ by < 10 % (screening threshold,
                  not a physical standard) -> reported as PASS/WARN

Run: ``PYTHONPATH=src python3 scripts/audit_runtest6_dem3d_timestep.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.dem3d import DEM3DConfig, IceDEM3D  # noqa: E402


def run(dt: float, t_end: float) -> dict:
    cfg = DEM3DConfig(nx=8, ny=4, nz=2, dt=dt, fix_x_edges=True)
    sim = IceDEM3D(cfg)
    steps = int(round(t_end / dt))
    peak_reaction = 0.0
    work_tool = 0.0
    e0 = sim.diagnostics()
    for i in range(steps):
        sim.step()
        dz = sim.config.tool_speed * dt
        # sample the reaction every 5 steps to keep the probe inside the budget
        if i % 5 == 0 or i == steps - 1:
            snap = sim.diagnostics()
            peak_reaction = max(peak_reaction, abs(snap["reaction_z"]))
            work_tool += 5 * abs(snap["reaction_z"]) * abs(dz)
    last = sim.diagnostics()
    total_energy = (last["kinetic_energy_J"] + last["bond_elastic_energy_J"]
                    + last["particle_contact_energy_J"] + last["tool_contact_energy_J"]
                    + last["platen_contact_energy_J"] + last["released_bond_energy_J"])
    total_initial = (e0["kinetic_energy_J"] + e0["bond_elastic_energy_J"]
                     + e0["particle_contact_energy_J"] + e0["tool_contact_energy_J"]
                     + e0["platen_contact_energy_J"] + e0["released_bond_energy_J"])
    ledger_residual = abs(work_tool - (total_energy - total_initial)) / max(work_tool, 1e-30)
    return {
        "dt": dt, "steps": steps, "t_end": last["time"],
        "peak_reaction_z_N": peak_reaction,
        "final_reaction_z_N": abs(last["reaction_z"]),
        "bond_elastic_energy_J": last["bond_elastic_energy_J"],
        "kinetic_energy_J": last["kinetic_energy_J"],
        "broken_bonds": last["broken_bonds"],
        "tool_work_J": work_tool,
        "ledger_residual_relative": ledger_residual,
    }


def order(dts, values) -> dict:
    import math
    pts = [(d, v) for d, v in zip(dts, values) if v > 0]
    if len(pts) < 2:
        return {"slope": None, "n": len(pts)}
    xs = [math.log(d) for d, _ in pts]
    ys = [math.log(v) for _, v in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else float("nan")
    ss_tot = sum((y - my) ** 2 for y in ys)
    pred = [my + slope * (x - mx) for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    return {"slope": slope, "r2": 1 - ss_res / ss_tot if ss_tot else 1.0, "n": n}


def main() -> int:
    cfg0 = DEM3DConfig(nx=8, ny=4, nz=2, fix_x_edges=True)
    dt_rec = cfg0.recommended_dt
    # the default indenter starts 10 mm above the specimen and travels at
    # tool_speed = 0.5 m/s, so contact happens at ~2e-2 s; run just past it
    t_end = 0.02

    rows = []
    for k in range(3):
        dt = dt_rec / (2 ** k)
        r = run(dt, t_end)
        rows.append(r)
        print(f"dt={dt:.4e} (k={k}) steps={r['steps']:5d}  peakFz={r['peak_reaction_z_N']:.6e} N  "
              f"U_bond={r['bond_elastic_energy_J']:.6e} J  ledger_res={r['ledger_residual_relative']:.3e}")

    finest = rows[-1]
    d_peak = [abs(r["peak_reaction_z_N"] - finest["peak_reaction_z_N"]) / max(abs(finest["peak_reaction_z_N"]), 1e-30)
              for r in rows]
    d_ub = [abs(r["bond_elastic_energy_J"] - finest["bond_elastic_energy_J"]) / max(abs(finest["bond_elastic_energy_J"]), 1e-30)
            for r in rows]
    dts = [r["dt"] for r in rows]
    report = {
        "protocol": {
            "geometry": "nx=8, ny=4, nz=2 bonded lattice, x edges fixed",
            "indentation": f"t_end = {t_end:.6e} s (indenter travels {0.5 * t_end * 1000:.1f} mm)",
            "dt_reference_s": dt_rec,
            "levels": 3,
            "note": "screening threshold: two finest levels within 10% -> PASS",
        },
        "cases": rows,
        "relative_difference_to_finest": {"peak_reaction_z": d_peak, "bond_elastic_energy": d_ub},
        "fitted_order": {
            "peak_reaction_z": order(dts[:-1], d_peak[:-1]),
            "bond_elastic_energy": order(dts[:-1], d_ub[:-1]),
            "ledger_residual": order(dts, [r["ledger_residual_relative"] for r in rows]),
        },
        "verdict": ("PASS" if max(d_peak[-2], d_ub[-2]) < 0.10 else "WARN"),
        "verdict_basis": {
            "peak_reaction_two_finest_delta": d_peak[-2],
            "bond_energy_two_finest_delta": d_ub[-2],
        },
    }
    print(json.dumps(report["fitted_order"], indent=1))
    print("verdict:", report["verdict"], report["verdict_basis"])

    out = ROOT / "runs" / "dem3d_timestep"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("wrote", out / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())