"""Restitution anchor for the ``polyhedron_floe3d`` contact law (h1 kernel).

The kernel declares ``f_n = k_n delta - c_n v_n`` with ``c_n = 2 zeta sqrt(k_n m_eff)``
(``ContactLaw.normal_damping``).  For that linear spring-dashpot the coefficient of
restitution is not a free parameter, it follows from ``zeta``:

    e(zeta) = exp(-zeta pi / sqrt(1 - zeta^2))

This script drops a flat-faced cube (square prism) onto a fixed floor and measures
``e = |v_out| / |v_in|`` for a sweep of ``zeta``; it also repeats one case at four
time steps so the measurement can be separated from integration error.

    PYTHONPATH=src python benchmarks/polyhedron_floe3d_restitution_anchor.py
    PYTHONPATH=src python benchmarks/polyhedron_floe3d_restitution_anchor.py --figures-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from tensordem.polyhedron_floe3d.dynamics import DEMSystem3D, FloeBody  # noqa: E402
from tensordem.polyhedron_floe3d.contact import ContactLaw  # noqa: E402
from tensordem.polyhedron_floe3d.geometry import extrude_polygon, regular_polygon  # noqa: E402

OUT_DIR = os.path.join(ROOT, "artifacts", "polyhedron_floe3d_restitution")
SIDE = 0.10          # cube edge [m]
DENSITY = 917.0      # sea ice [kg/m3]
V_IN = 0.5           # impact speed [m/s]
ZETA_GRID = (0.05, 0.1, 0.2, 0.4, 0.7)
DT_GRID = (2e-4, 1e-4, 5e-5, 2.5e-5)


def analytic_restitution(zeta: float) -> float:
    if zeta >= 1.0:
        return 0.0
    return math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta * zeta))


def cube(edge: float = SIDE):
    return extrude_polygon(regular_polygon(edge * edge, 4), edge)


def drop_case(zeta: float, dt: float, kn: float = 2.0e5, v_in: float = V_IN,
              clearance: float = 2.0e-3) -> dict:
    """Drop a flat cube onto the floor; return the measured restitution."""
    shape = cube()
    body = FloeBody(shape, DENSITY, position=[0.0, 0.0, 0.0],
                    velocity=[0.0, 0.0, -v_in], label="cube")
    # place the *lowest vertex* just above the floor: ``principal=True`` may leave the
    # shape in an arbitrary body frame (the inertia of a cube is isotropic, so the
    # principal axes are not unique), so the half-height is read off the world frame
    # instead of assumed to be ``SIDE / 2``.
    body.position = np.array([0.0, 0.0, clearance - float(body.world_vertices()[:, 2].min())])
    system = DEMSystem3D([body], law=ContactLaw(kn=kn, zeta=zeta, mu=0.3,
                                                mu_rolling=0.0, kt_ratio=0.25),
                         gravity=0.0, dt=dt)
    system.add_wall([0.0, 0.0, 1.0], 0.0, label="floor")
    system.record_initial_energy()
    e0 = system.energy()
    touched = False
    v_out = float("nan")
    steps = 0
    max_steps = int(0.5 / dt)
    for step in range(1, max_steps + 1):
        system.step()
        steps = step
        if system.n_wall_contacts > 0:
            touched = True
        elif touched:
            v_out = float(body.velocity[2])
            break
    return {
        "half_height_m": float(0.5 * (body.world_vertices()[:, 2].max()
                                      - body.world_vertices()[:, 2].min())),
        "zeta": float(zeta), "dt": float(dt), "kn": float(kn), "v_in": float(v_in),
        "v_out": v_out, "e_measured": abs(v_out) / v_in if v_out == v_out else float("nan"),
        "e_analytic": analytic_restitution(zeta),
        "touched": bool(touched), "steps": int(steps),
        "sim_time_s": float(system.time),
        "energy_ratio": float(system.energy() / e0),
        "energy_residual": float(system.energy_residual()),
        "mass_kg": float(body.mass),
    }


def run(out_dir: str = OUT_DIR) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    sweep = []
    for zeta in ZETA_GRID:
        row = drop_case(zeta, 1e-4)
        row["rel_err"] = (abs(row["e_measured"] - row["e_analytic"]) / row["e_analytic"]
                          if row["e_analytic"] > 0 else float("nan"))
        sweep.append(row)
    convergence = []
    for dt in DT_GRID:
        row = drop_case(0.2, dt)
        row["rel_err"] = abs(row["e_measured"] - row["e_analytic"]) / row["e_analytic"]
        convergence.append(row)
    summary = {
        "case": "cube-on-floor restitution vs the linear spring-dashpot law",
        "geometry": {"shape": "square prism (4-gon extrusion)", "edge_m": SIDE,
                     "density_kg_m3": DENSITY},
        "kn_N_m": 2.0e5, "v_in_m_s": V_IN,
        "zeta_grid": list(ZETA_GRID), "dt_grid_s": list(DT_GRID),
        "sweep": sweep, "dt_convergence_zeta_0.2": convergence,
        "max_rel_err": max(r["rel_err"] for r in sweep),
        "max_rel_err_in_band": max(r["rel_err"] for r in sweep
                                   if r["zeta"] <= 0.4),
    }
    path = os.path.join(out_dir, "results.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1)
    summary["_path"] = path
    return summary


def figures(path: str = OUT_DIR) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.load(open(os.path.join(path, "results.json")))
    zs = np.linspace(0.0, 0.9, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.plot(zs, [analytic_restitution(z) for z in zs], "k-",
            label=r"analytic $e=\exp(-\zeta\pi/\sqrt{1-\zeta^2})$")
    sx = [r["zeta"] for r in data["sweep"]]
    sy = [r["e_measured"] for r in data["sweep"]]
    ax.plot(sx, sy, "o", ms=7, label="measured (polyhedron_floe3d)")
    for r in data["sweep"]:
        ax.annotate(f"{r['rel_err']*100:.1f}%", (r["zeta"], r["e_measured"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel(r"declared damping ratio $\zeta$")
    ax.set_ylabel("coefficient of restitution $e$")
    ax.set_title("flat-face cube drop, $v_{in}=0.5$ m/s, $k_n=2\\times10^5$ N/m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    conv = data["dt_convergence_zeta_0.2"]
    dts = [r["dt"] for r in conv]
    ax.loglog(dts, [r["e_measured"] for r in conv], "s-", label="measured $e$")
    ax.axhline(conv[0]["e_analytic"], color="k", ls="--",
               label=f"analytic {conv[0]['e_analytic']:.4f}")
    ax.set_xlabel("time step $\\Delta t$ [s]")
    ax.set_ylabel("coefficient of restitution $e$")
    ax.set_title(r"time-step convergence ($\zeta=0.2$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(path, "restitution_anchor.png")
    fig.savefig(out, dpi=140)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()
    if not args.figures_only:
        summary = run()
        print(json.dumps({k: v for k, v in summary.items() if k != "sweep"}, indent=1))
        for r in summary["sweep"]:
            print(f"zeta={r['zeta']:<5} e_meas={r['e_measured']:.4f} "
                  f"e_analytic={r['e_analytic']:.4f} err={r['rel_err']*100:.2f}% "
                  f"energy_ratio={r['energy_ratio']:.6f} steps={r['steps']}")
        for r in summary["dt_convergence_zeta_0.2"]:
            print(f"dt={r['dt']:<8} e_meas={r['e_measured']:.4f} err={r['rel_err']*100:.2f}%")
    print("figure:", figures())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
