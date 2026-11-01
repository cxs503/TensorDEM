"""Evidence generator for mission 1504 (TensorDEM main CI red).

Produces, in docs/:
  * dem3d_ci_regression_metrics.json      -- programmatic metrics (nothing hand-copied)
  * dem3d_ci_regression_load_balance.png  -- reaction/deflection histories + peak bars

Three solver states are replayed through the identical three-point-bending
fixture of benchmarks/dem3d/three_point_bending_cases.json:
  PRE_REGRESSION : dem3d.py at a68fa8a (last green main before PR #73)
  BROKEN         : dem3d.py at a40e29a (first bad commit located by git bisect)
  FIXED          : dem3d.py in this worktree
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
torch.set_num_threads(1)

SOURCES = {
    "PRE_REGRESSION": "/tmp/oldver/dem3d_a68.py",
    "BROKEN": "/tmp/oldver/dem3d_a40.py",
}


def load_variant(name: str):
    if name == "FIXED":
        from tensordem.dem3d import DEM3DConfig, IceDEM3D

        return DEM3DConfig, IceDEM3D
    spec = importlib.util.spec_from_file_location(f"dem3d_{name}", SOURCES[name])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DEM3DConfig, module.IceDEM3D


def run_fixture(name: str, steps: int = 24) -> dict:
    cfg_cls, sim_cls = load_variant(name)
    manifest = json.loads(
        (ROOT / "benchmarks/dem3d/three_point_bending_cases.json").read_text(encoding="utf-8")
    )
    case = manifest["cases"][0]
    p = case["config"]
    sim = sim_cls(cfg_cls(
        nx=int(p["nx"]), ny=int(p["ny"]), nz=int(p["nz"]), radius=float(p["radius_m"]),
        density=float(p["density_kg_m3"]), bond_stiffness=float(p["bond_stiffness_N_m"]),
        contact_stiffness=float(p["contact_stiffness_N_m"]),
        breaking_strain=float(p["breaking_strain"]),
        shear_breaking_strain=float(p["shear_breaking_strain"]),
        contact_damping=float(p["contact_damping_Ns_m"]), drag=float(p["drag"]),
        fix_x_edges=False, fix_bottom=False, tool_gap=1.0, tool_radius=0.01, tool_speed=0.01,
    ))
    sim.tool_start[:] = torch.tensor([1e6, 1e6, 1e6], dtype=sim.dtype)
    sim.tool_velocity.zero_()
    x = sim.initial_positions[:, 0]
    z = sim.initial_positions[:, 2]
    xv = torch.unique(x)
    zb, zt = z.min(), z.max()
    lx = xv[max(1, len(xv) // 4)]
    rx = xv[min(len(xv) - 2, (3 * len(xv)) // 4)]
    supports = (z == zb) & ((x == lx) | (x == rx))
    loaded = (z == zt) & (x == xv[len(xv) // 2])
    sim.fixed[:] = supports
    sim.positions[sim.fixed] = sim.initial_positions[sim.fixed]
    load_max = float(p["peak_load_N"])
    nload = int(loaded.sum())
    rows = []
    for step in range(steps + 1):
        applied = load_max * step / steps
        loads = torch.zeros_like(sim.positions)
        loads[loaded, 2] = -applied / nload
        if step:
            sim.step(external_forces=loads)
        force, _ = sim.forces(loads, update_fracture=False)
        reaction = -force[sim.fixed].sum(dim=0)
        rows.append({
            "step": step,
            "applied_load_N": applied,
            "support_reaction_z_N": float(reaction[2]),
            "midspan_deflection_m": float(
                (sim.positions[loaded, 2] - sim.initial_positions[loaded, 2]).mean()
            ),
        })
    peak_reaction = max(abs(r["support_reaction_z_N"]) for r in rows)
    peak_deflection = max(abs(r["midspan_deflection_m"]) for r in rows)
    return {
        "state": name,
        "dt_s": float(sim.dt),
        "steps": steps,
        "particle_count": int(len(sim.positions)),
        "bond_count": int(len(sim.pairs)),
        "support_particles": int(supports.sum()),
        "loaded_particles": nload,
        "peak_support_reaction_N": peak_reaction,
        "peak_midspan_deflection_m": peak_deflection,
        "min_peak_support_reaction_N": float(case["acceptance"]["minimum_peak_support_reaction_N"]),
        "min_peak_deflection_m": float(case["acceptance"]["minimum_peak_deflection_m"]),
        "acceptance_pass": bool(
            peak_reaction > float(case["acceptance"]["minimum_peak_support_reaction_N"])
            and peak_deflection > float(case["acceptance"]["minimum_peak_deflection_m"])
        ),
        "rows": rows,
    }


def campaign_metrics() -> dict:
    """Post-fix load-convergence campaign numbers (the workload of the red test)."""
    sys.path.insert(0, str(ROOT))
    from scripts.benchmark_dem3d_bending_load_convergence import run_bending_load_convergence

    with tempfile.TemporaryDirectory() as tmp:
        report = run_bending_load_convergence(
            Path(tmp) / "campaign", base_steps=8, levels=2, peak_load_N=0.01
        )
    return {
        "verdict": report["verdict"],
        "step_counts": report["step_counts"],
        "runs": [
            {
                "steps": r["steps"],
                "dt_s": r["dt_s"],
                "ramp_duration_s": r["ramp_duration_s"],
                "peak_support_reaction_N": r["peak_support_reaction_abs_N"],
                "peak_midspan_deflection_m": r["peak_midspan_deflection_abs_m"],
            }
            for r in report["runs"]
        ],
        "comparisons": report["comparisons"],
    }


def platen_damper_probe() -> dict:
    """Damped bottom-platen response: sign of the damping term, broken vs fixed."""
    out = {}
    for name in ("BROKEN", "FIXED"):
        cfg_cls, sim_cls = load_variant(name)
        sim = sim_cls(cfg_cls(nx=5, ny=2, nz=2, radius=0.01, density=917.0, drag=0.0,
                              fix_x_edges=False, bottom_platen_enabled=True,
                              platen_stiffness=4_000.0, platen_damping=50.0))
        sim.tool_start[:] = torch.tensor([1e6, 1e6, 1e6], dtype=sim.dtype)
        sim.tool_velocity.zero_()
        bottom = sim.positions[:, 2] == sim.positions[:, 2].min()
        sim.positions[bottom, 2] -= 0.002
        count = int(bottom.sum())
        measured = {}
        for label, vz in (("static", 0.0), ("approaching", -0.03), ("separating", 0.03)):
            sim.velocities.zero_()
            sim.velocities[bottom, 2] = vz
            force, _ = sim.forces(update_fracture=False)
            measured[label] = float(force[:, 2].sum())
        out[name] = {
            "bottom_particles": count,
            "total_fz_static_N": measured["static"],
            "total_fz_approaching_N": measured["approaching"],
            "total_fz_separating_N": measured["separating"],
            "expected_damping_delta_N": 50.0 * 0.03 * count,
            "damper_delta_approaching_N": measured["approaching"] - measured["static"],
            "damper_delta_separating_N": measured["separating"] - measured["static"],
        }
    return out


def figure(metrics: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "PRE_REGRESSION": dict(color="#1f77b4", ls="-", label="PRE_REGRESSION a68fa8a (last green)"),
        "BROKEN": dict(color="#d62728", ls="--", label="BROKEN a40e29a (bisect first-bad)"),
        "FIXED": dict(color="#2ca02c", ls=":", lw=2.4, label="FIXED this worktree"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for state in metrics["states"]:
        rows = state["rows"]
        ax.plot([r["step"] for r in rows], [abs(r["support_reaction_z_N"]) for r in rows],
                **styles[state["state"]])
    ax.set_xlabel("step")
    ax.set_ylabel(r"$|R_z|$ at supports (N)")
    ax.set_title("(a) support reaction history")
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    for state in metrics["states"]:
        rows = state["rows"]
        ax.plot([r["step"] for r in rows], [abs(r["midspan_deflection_m"]) for r in rows],
                **styles[state["state"]])
    ax.set_xlabel("step")
    ax.set_ylabel(r"$|w_{mid}|$ (m)")
    ax.set_title("(b) midspan deflection history")
    ax.set_yscale("symlog", linthresh=1e-13)
    ax.grid(alpha=0.3)

    ax = axes[2]
    labels = [s["state"] for s in metrics["states"]]
    peaks = [s["peak_support_reaction_N"] for s in metrics["states"]]
    bars = ax.bar(labels, peaks, color=[styles[s]["color"] for s in labels])
    ax.set_ylabel(r"peak $|R_z|$ (N)")
    ax.set_title("(c) peak support reaction vs acceptance floor")
    ax.axhline(metrics["states"][0]["min_peak_support_reaction_N"], color="k", ls=":",
               label="acceptance floor 1e-8 N")
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    for bar, value in zip(bars, peaks):
        ax.annotate(f"{value:.3e}", (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    fig.suptitle("TensorDEM three-point-bending fixture: external-load regression and fix "
                 "(24 steps, dt = 1.8059e-05 s)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"figure -> {path}")


def main() -> int:
    metrics = {
        "fixture": "benchmarks/dem3d/three_point_bending_cases.json",
        "states": [run_fixture(name) for name in ("PRE_REGRESSION", "BROKEN", "FIXED")],
        "load_convergence_campaign_fixed": campaign_metrics(),
        "bottom_platen_damper": platen_damper_probe(),
    }
    by_state = {s["state"]: s for s in metrics["states"]}
    pre, fix = by_state["PRE_REGRESSION"], by_state["FIXED"]
    metrics["fixed_vs_pre_regression"] = {
        "peak_reaction_relative_error": abs(
            fix["peak_support_reaction_N"] - pre["peak_support_reaction_N"]
        ) / abs(pre["peak_support_reaction_N"]),
        "peak_deflection_relative_error": abs(
            fix["peak_midspan_deflection_m"] - pre["peak_midspan_deflection_m"]
        ) / abs(pre["peak_midspan_deflection_m"]),
        "history_max_abs_diff_N": max(
            abs(a["support_reaction_z_N"] - b["support_reaction_z_N"])
            for a, b in zip(fix["rows"], pre["rows"])
        ),
    }
    out = ROOT / "docs" / "dem3d_ci_regression_metrics.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    figure(metrics, ROOT / "docs" / "dem3d_ci_regression_load_balance.png")
    print(json.dumps({
        "peaks": {s["state"]: [s["peak_support_reaction_N"], s["peak_midspan_deflection_m"],
                               s["acceptance_pass"]] for s in metrics["states"]},
        "fixed_vs_pre_regression": metrics["fixed_vs_pre_regression"],
        "campaign_comparisons": metrics["load_convergence_campaign_fixed"]["comparisons"],
        "damper": metrics["bottom_platen_damper"],
        "written": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
