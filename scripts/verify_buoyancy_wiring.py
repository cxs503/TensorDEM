"""Produce the machine-generated evidence record + figures for the DEM3D buoyancy wiring.

Usage
-----
    PYTHONPATH=src python scripts/verify_buoyancy_wiring.py [--steps 4000]

Writes:
  docs/buoyancy-wiring/buoyancy_wiring_verification.json   (all numbers)
  docs/buoyancy-wiring/fig1_force_balance.png              (F_z(z) solver vs analytic)
  docs/buoyancy-wiring/fig2_archimedes_settling.png        (settling on the analytic draft)

The doc `docs/dem3d_buoyancy_wiring.md` quotes only numbers read back from that
JSON, so no value in it is hand-transcribed.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.buoyancy_wet3d import (  # noqa: E402
    equilibrium_centre_height, submerged_volume_sphere)
from tensordem.buoyancy_wiring_verification import (  # noqa: E402
    ISOLATED_BASE, isolated_particle, relaxation_case, run_all)

OUT = ROOT / "docs" / "buoyancy-wiring"


def force_curve(g: float = 9.81, h_max: float = 1.5, samples: int = 61):
    """Net vertical force on the free particle vs centre height (solver + analytic)."""
    radius = ISOLATED_BASE["radius"]
    hs = [h_max * (2.0 * k / (samples - 1) - 1.0) for k in range(samples)]
    dem, cfg = isolated_particle(z0=0.0)
    measured, analytic = [], []
    for h in hs:
        z = h * radius
        dem.positions[0, 2] = z
        measured.append(float(dem.body_forces()[0, 2]))
        analytic.append(-cfg.mass * cfg.gravity
                        + cfg.water_density * cfg.gravity
                        * submerged_volume_sphere(z, radius))
    return [h * radius for h in hs], measured, analytic, cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000,
                        help="explicit steps per relaxation case")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    record = run_all(n_steps=args.steps)
    zs, measured, analytic, cfg = force_curve()
    record["force_curve"] = {"z_m": zs, "measured_N": measured, "analytic_N": analytic}
    record["force_curve_max_abs_error_N"] = max(
        abs(a - b) for a, b in zip(measured, analytic))
    # Independent draft anchors for the figure legend / doc table.
    record["draft_anchors"] = {
        str(rho): equilibrium_centre_height(cfg.radius, rho, cfg.water_density)
        for rho in (800.0, 850.0, 917.0, 950.0)
    }
    json_path = OUT / "buoyancy_wiring_verification.json"
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot([z / cfg.radius for z in zs], analytic, "-", lw=3.0,
            color="0.55", label="analytic $-mg+\\rho_w g V_{sub}(h)$")
    ax.plot([z / cfg.radius for z in zs], measured, "--", lw=1.2,
            color="crimson", label="IceDEM3D body force (solver)")
    ax.axhline(0.0, color="k", lw=0.6)
    for rho, colour in ((850.0, "tab:blue"), (917.0, "tab:green")):
        h_eq = equilibrium_centre_height(cfg.radius, rho, cfg.water_density)
        ax.axvline(h_eq / cfg.radius, color=colour, lw=0.9, ls=":")
        ax.annotate(f"$h_{{eq}}$($\\rho_p$={rho:.0f})", (h_eq / cfg.radius, 0.25),
                    rotation=90, fontsize=8, color=colour, ha="right", va="bottom")
    ax.set_xlabel("dimensionless centre height $h/R$  ($h>0$ above the free surface)")
    ax.set_ylabel("net vertical force on the particle  [N]")
    ax.set_title("Gravity + Archimedes buoyancy wired into IceDEM3D\n"
                 f"max |solver - analytic| = {record['force_curve_max_abs_error_N']:.2e} N")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_force_balance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colours = ("tab:blue", "tab:green", "tab:red")
    labels = []
    for case, colour in zip(record["relaxation"], colours):
        trace = case["trace"]
        label = (f"$\\rho_p$={case['density_kg_m3']:.0f}, dt={case['dt_s']:.3e} s")
        ax.plot([t for _, t in trace], [z / cfg.radius for _, z in trace],
                "-o", ms=2.5, lw=1.1, color=colour, label=label)
        ax.axhline(case["h_eq_analytic_m"] / cfg.radius, color=colour, lw=0.9, ls=":")
        labels.append(case)
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("centre height $z/R$")
    ax.set_title("Single free ice sphere settles on the analytic Archimedes draft\n"
                 f"max |draft error| = {record['relaxation_max_rel_error_draft']*100:.3f} %")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_archimedes_settling.png", dpi=150)
    plt.close(fig)

    print(json.dumps({k: v for k, v in record.items()
                      if k not in ("static_force", "relaxation", "force_curve")},
                     indent=2, sort_keys=True)[:4000])
    print("wrote", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())