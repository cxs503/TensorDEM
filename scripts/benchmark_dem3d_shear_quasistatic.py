"""Quasi-static, finite-size-converged modulus of the DEM3D direct-shear block.

Closes the audit item that ``docs/dem3d_shear_specimen.md`` reported as WITHHELD
("the ramp is 20 % from rate convergence") and that
``docs/dem3d_material_calibration_roadmap.md`` lists as *"quasi-static apparent
modulus of a driven block"*:

1. **rate** -- seven ramp speeds (0.4 -> 0.00625 m/s, 0.6 -> 36.5 shear-wave
   transits), fixed-strain fit window, zero-rate extrapolation ``G0 + c v``;
2. **finite size** -- a width series (nx = 5..17) and a height series
   (nz = 5, 7, 9, each driven at its own strictest admissible rate), fitted
   jointly as ``G/C44 = c0 - alpha a/Lx - beta a/h``;
3. **mechanism** -- the drag series shows the converged modulus is drag
   insensitive, so the fast-ramp bias is a transient in the fit window (which the
   window-spread diagnostic measures) and not a viscous stress bias.

Outputs: ``results-dem3d-shear-quasistatic/`` (JSON + CSV + PNG) and the
generated document ``docs/dem3d_shear_specimen_quasistatic.md``.

Usage::

    python scripts/benchmark_dem3d_shear_quasistatic.py --jobs 8
    python -m pytest tests/test_dem3d_shear_quasistatic.py -q
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from tensordem.dem3d import DEM3DConfig
from tensordem.shear_quasistatic import (
    apparent_modulus_at_rate,
    drag_series,
    fit_infinite_width,
    fit_two_way_finite_size,
    fit_zero_rate_limit,
    height_series,
    lattice_wave_speed,
    quasi_static_config,
    quasi_static_velocity,
    shear_diffusion_time,
    shear_wave_transit,
    width_series,
)

DEFAULT_OUTPUT = ROOT / "results-dem3d-shear-quasistatic"
DOCS_FIGURES = ROOT / "docs" / "dem3d-shear-quasistatic"
PROTOCOL = "tensordem-dem3d-shear-quasistatic-v2"

GAMMA_MAX = 0.025
RATE_VELOCITIES = (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625)
CONVERGED_VELOCITY = 0.025
WIDTHS = (5, 9, 13, 17)
DRAGS = (2.0, 5.0, 10.0, 20.0)
HEIGHT_NZ = (5, 7, 9)
HEIGHT_DRAG = 5.0
CROSS_NX = (5,)          # extra widths measured at nz = 9


def _worker(arg: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    kind, payload = arg
    if kind == "rate":
        row = apparent_modulus_at_rate(quasi_static_config(),
                                       velocity=float(payload["velocity"]),
                                       gamma_max=GAMMA_MAX, keep_history=True)
    elif kind == "width":
        row = width_series(quasi_static_config(), nx_values=[int(payload["nx"])],
                           velocity=CONVERGED_VELOCITY, gamma_max=GAMMA_MAX,
                           keep_history=True)[0]
    elif kind == "drag":
        row = drag_series(quasi_static_config(), drags=[float(payload["drag"])],
                          velocity=CONVERGED_VELOCITY, gamma_max=GAMMA_MAX,
                          keep_history=True)[0]
    else:  # height / cross: auto-selected admissible rate for that height
        config = quasi_static_config(nx=int(payload.get("nx", 9)), drag=HEIGHT_DRAG,
                                     nz=int(payload["nz"]))
        row = height_series(config, nz_values=[int(payload["nz"])],
                            gamma_max=GAMMA_MAX, keep_history=True)[0]
    row["case"] = kind
    row["case_label"] = payload["label"]
    return row


def campaign(jobs: int) -> dict[str, list[dict[str, Any]]]:
    tasks: list[tuple[str, dict[str, Any]]] = []
    for velocity in RATE_VELOCITIES:
        tasks.append(("rate", {"velocity": velocity, "label": f"v={velocity:g}"}))
    for nx in WIDTHS:
        tasks.append(("width", {"nx": nx, "label": f"nx={nx},nz=5"}))
    for drag in DRAGS:
        tasks.append(("drag", {"drag": drag, "label": f"drag={drag:g}"}))
    for nz in HEIGHT_NZ:
        tasks.append(("height", {"nz": nz, "label": f"nx=9,nz={nz}"}))
    for nx in CROSS_NX:
        tasks.append(("cross", {"nx": nx, "nz": 9, "label": f"nx={nx},nz=9"}))
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(min(jobs, len(tasks))) as pool:
            rows = pool.map(_worker, tasks)
    else:
        rows = [_worker(task) for task in tasks]
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in
                                            ("rate", "width", "drag", "height", "cross")}
    for row in rows:
        out[row["case"]].append(row)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_traces(directory: Path, rows: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = row["case_label"].replace("=", "").replace(",", "_").replace(".", "p")
        with (directory / f"trace_{row['case']}_{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["gamma", "tau_Pa", "time_s", "kinetic_energy_J",
                             "bond_elastic_energy_J", "max_bond_invariant"])
            for point in row["history"] or []:
                writer.writerow([point["gamma"], point["tau_Pa"], point["time_s"],
                                 point["kinetic_energy_J"],
                                 point["bond_elastic_energy_J"],
                                 point["max_bond_invariant"]])


def _checks(rate, width, drag, height, joint, zero_rate, infinite_width,
            consistency) -> list[dict[str, Any]]:
    slowest = rate[-1]
    drag_core = [r for r in drag if r["drag_Ns_per_m"] >= 10.0]
    drag_spread = (max(r["ratio_to_C44"] for r in drag_core)
                   - min(r["ratio_to_C44"] for r in drag_core))
    return [
        {"name": "ramp speeds span two decades (0.4 -> 0.00625 m/s)",
         "value": rate[0]["velocity_m_per_s"] / slowest["velocity_m_per_s"],
         "reference": 50.0, "passed": rate[0]["velocity_m_per_s"] / slowest["velocity_m_per_s"] >= 50.0,
         "note": "previous audit used three rates inside one decade"},
        {"name": "fastest ramp is demonstrably not quasi-static",
         "value": rate[0]["margins"]["transits"], "reference": 20.0,
         "passed": rate[0]["margins"]["transits"] < 20.0,
         "note": "0.6 wave transits at v = 0.4 m/s"},
        {"name": "slowest ramp satisfies the quasi-static criterion",
         "value": slowest["margins"]["transits"], "reference": 20.0,
         "passed": bool(slowest["margins"]["quasi_static"]),
         "note": "transits >= 20 and diffusion times >= 5"},
        {"name": "zero-rate extrapolation residual",
         "value": zero_rate["relative_residual"], "reference": 5.0e-3,
         "passed": zero_rate["relative_residual"] < 5.0e-3,
         "note": "linear G0 + c v fit over the three slowest rates"},
        {"name": "zero-rate extrapolation stability",
         "value": zero_rate.get("stability_relative_change", math.inf),
         "reference": 1.0e-2,
         "passed": zero_rate.get("stability_relative_change", math.inf) < 1.0e-2,
         "note": "dropping the fastest fitted rate must not move G0 by 1 %"},
        {"name": "slowest measured rate is inside 2 % of G0",
         "value": abs(zero_rate["slowest_ratio_to_G0"] - 1.0), "reference": 2.0e-2,
         "passed": abs(zero_rate["slowest_ratio_to_G0"] - 1.0) < 2.0e-2,
         "note": "a measured point, not only the fit"},
        {"name": "fit window does not bias the converged slope",
         "value": slowest["window_spread_relative"], "reference": 2.0e-2,
         "passed": slowest["window_spread_relative"] < 2.0e-2,
         "note": "three strain windows on the same history"},
        {"name": "converged modulus is drag insensitive (drag >= 10)",
         "value": drag_spread, "reference": 1.0e-3, "passed": drag_spread < 1.0e-3,
         "note": "rules out a viscous stress bias in the converged limit"},
        {"name": "height series reproduces the width series at nx = 9, nz = 5",
         "value": consistency, "reference": 2.0e-2, "passed": consistency < 2.0e-2,
         "note": "auto-rate height run vs the converged-rate width run"},
        {"name": "width extrapolation is linear in a/Lx",
         "value": infinite_width["relative_residual"], "reference": 5.0e-2,
         "passed": infinite_width["relative_residual"] < 5.0e-2,
         "note": "four widths at nz = 5"},
        {"name": "joint finite-size fit residual",
         "value": joint["max_abs_residual_over_C44"], "reference": 2.0e-2,
         "passed": joint["max_abs_residual_over_C44"] < 2.0e-2,
         "note": "G/C44 = c0 - alpha a/Lx - beta a/h over seven (Lx, h) points"},
        {"name": "doubly extrapolated modulus is within 3 % of C44",
         "value": joint["relative_error_vs_C44"], "reference": 3.0e-2,
         "passed": joint["relative_error_vs_C44"] < 3.0e-2,
         "note": "the only comparison that is both physical and 3D-converged"},
    ]


def _figures(summary: dict[str, Any], directory: Path) -> list[dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 8.5})
    rate = sorted(summary["rate_series"], key=lambda r: -r["velocity_m_per_s"])
    zero = summary["zero_rate_limit"]
    joint = summary["joint_fit"]
    c44 = rate[0]["C44_Pa"]
    figures: list[dict[str, str]] = []

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax = axes[0]
    for row in rate:
        history = row["history"] or []
        ax.plot([p["gamma"] for p in history], [p["tau_Pa"] for p in history],
                lw=1.0,
                label=f"v={row['velocity_m_per_s']:g} m/s, {row['margins']['transits']:.1f} transits")
    ax.axvspan(*rate[-1]["window"], color="0.85", zorder=0, label="fit window")
    ax.set_xlabel(r"engineering shear strain $\gamma=\Delta x/h$")
    ax.set_ylabel(r"bottom-reaction traction $\tau$ [Pa]")
    ax.set_title("(a) driven-shear response vs ramp speed")
    ax.legend(fontsize=6, loc="upper left", ncol=1)
    ax.grid(alpha=0.3)

    ax = axes[1]
    v = [r["velocity_m_per_s"] for r in rate]
    g = [r["ratio_to_C44"] for r in rate]
    err = [r["window_spread_relative"] * r["ratio_to_C44"] for r in rate]
    ax.errorbar(v, g, yerr=err, marker="o", lw=1.2, capsize=3,
                label=r"$G_{\rm app}/C_{44}$ (window spread)")
    ax.axhline(zero["G0_over_C44"], color="C3", ls="--",
               label=fr"$v\to0$: {zero['G0_over_C44']:.4f} $C_{{44}}$")
    ax.axhline(1.0, color="0.4", ls=":", label=r"affine $C_{44}$")
    ax.set_xscale("log")
    ax.set_xlabel("ramp speed v [m/s]")
    ax.set_ylabel(r"$G_{\rm app}/C_{44}$")
    ax.set_title("(b) rate convergence, nx=9, nz=5")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(directory / "rate_convergence.png")
    plt.close(fig)
    figures.append({"name": "rate_convergence.png",
                    "caption": "(a) bottom-reaction traction vs engineering shear strain for "
                               "seven ramp speeds, common fit window shaded; (b) apparent modulus "
                               "normalised by the affine constant vs ramp speed with the "
                               "zero-rate extrapolation."})

    width5 = sorted(summary["width_series"], key=lambda r: r["nx"])
    height9 = sorted([r for r in summary["height_series"] + summary["cross_series"]],
                     key=lambda r: r["nz"])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax = axes[0]
    ax.plot([r["inverse_width_a_over_Lx"] for r in width5],
            [r["ratio_to_C44"] for r in width5], "o-",
            label=r"nz = 5 (nx = " + ", ".join(str(r["nx"]) for r in width5) + ")")
    tall = [r for r in height9 if r["nz"] == 9]
    ax.plot([r["inverse_width_a_over_Lx"] for r in tall],
            [r["ratio_to_C44"] for r in tall], "s--",
            label=r"nz = 9 (nx = " + ", ".join(str(r["nx"]) for r in tall) + ")")
    xx = [0.0, max(r["inverse_width_a_over_Lx"] for r in width5)]
    for nz, style in ((5, "-"), (9, "--")):
        a_over_h = 0.05 / ((nz - 1) * 0.05)
        ax.plot(xx, [joint["c0_G_over_C44_at_infinite_size"] - joint["beta_height"] * a_over_h
                     - joint["alpha_lateral"] * x for x in xx], style, color="C3", lw=1.0)
    ax.axhline(1.0, color="0.4", ls=":", label=r"affine $C_{44}$")
    ax.axhline(joint["c0_G_over_C44_at_infinite_size"], color="C3", lw=0.8, ls=":",
               label=fr"$c_0$ = {joint['c0_G_over_C44_at_infinite_size']:.4f}")
    ax.set_xlabel(r"inverse aspect ratio $a/L_x$")
    ax.set_ylabel(r"$G/C_{44}$ at converged rate")
    ax.set_title("(a) lateral free-surface relaxation")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot([r["inverse_height_a_over_h"] for r in height9],
            [r["ratio_to_C44"] for r in height9], "o-",
            label=r"nx = " + ", ".join(str(r["nx"]) for r in height9))
    hh = [0.0, max(r["inverse_height_a_over_h"] for r in height9)]
    a_over_Lx = 0.05 / (9 * 0.05)
    ax.plot(hh, [joint["c0_G_over_C44_at_infinite_size"] - joint["alpha_lateral"] * a_over_Lx
                 - joint["beta_height"] * x for x in hh], "--", color="C3",
            label="joint fit, $a/h\\to0$")
    ax.axhline(1.0, color="0.4", ls=":")
    ax.set_xlabel(r"inverse height $a/h$")
    ax.set_ylabel(r"$G/C_{44}$ at converged rate")
    ax.set_title("(b) clamped-base / platen height effect")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(directory / "finite_size.png")
    plt.close(fig)
    figures.append({"name": "finite_size.png",
                    "caption": "(a) converged apparent modulus vs inverse width at two heights, "
                               "with the joint-fit lines and the extrapolated c0; (b) converged "
                               "apparent modulus vs inverse height at nx = 9 with the joint fit "
                               "extrapolated to a/h -> 0."})
    return figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--skip-doc", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.time()
    rows = campaign(args.jobs)
    wall = time.time() - started

    rate = sorted(rows["rate"], key=lambda r: -r["velocity_m_per_s"])
    width = sorted(rows["width"], key=lambda r: r["nx"])
    drag = sorted(rows["drag"], key=lambda r: r["drag_Ns_per_m"])
    height = sorted(rows["height"], key=lambda r: r["nz"])
    cross = sorted(rows["cross"], key=lambda r: r["nx"])

    zero_rate = fit_zero_rate_limit(rate, points=3)
    infinite_width = fit_infinite_width(width)
    reference_height = [r for r in height if r["nz"] == 5]
    joint_points = width + [r for r in height if r["nz"] != 5] + cross
    joint = fit_two_way_finite_size(joint_points)
    consistency = (abs(reference_height[0]["apparent_shear_modulus_Pa"]
                       - [r for r in width if r["nx"] == 9][0]["apparent_shear_modulus_Pa"])
                   / [r for r in width if r["nx"] == 9][0]["apparent_shear_modulus_Pa"]
                   if reference_height else math.inf)
    drag_core = [r for r in drag if r["drag_Ns_per_m"] >= 10.0]
    drag_core_spread = (max(r["ratio_to_C44"] for r in drag_core)
                        - min(r["ratio_to_C44"] for r in drag_core))
    drag_all_spread = (max(r["ratio_to_C44"] for r in drag)
                       - min(r["ratio_to_C44"] for r in drag))
    checks = _checks(rate, width, drag, height, joint, zero_rate, infinite_width,
                     consistency)

    config = quasi_static_config()
    summary: dict[str, Any] = {
        "protocol": PROTOCOL,
        "config": {"nx": config.nx, "ny": config.ny, "nz": config.nz,
                   "radius_m": config.radius, "density_kg_per_m3": config.density,
                   "mass_kg": config.mass,
                   "bond_stiffness_N_per_m": config.bond_stiffness,
                   "breaking_strain": config.breaking_strain,
                   "shear_breaking_strain": config.shear_breaking_strain,
                   "drag_Ns_per_m": config.drag},
        "lattice_spacing_m": rate[0]["lattice_spacing_m"],
        "specimen_height_m": rate[0]["specimen_height_m"],
        "specimen_area_m2": rate[0]["specimen_area_m2"],
        "particle_count": rate[0]["particle_count"],
        "wave_speed_m_per_s": lattice_wave_speed(config),
        "wave_transit_s": shear_wave_transit(config),
        "diffusion_time_s": shear_diffusion_time(config),
        "max_bond_invariant": max(p["max_bond_invariant"] for r in rows["rate"]
                                  for p in (r["history"] or [])),
        "rate_series": rate, "width_series": width, "drag_series": drag,
        "height_series": height, "cross_series": cross,
        "zero_rate_limit": zero_rate, "infinite_width_fit": infinite_width,
        "joint_fit": joint, "consistency_relative": consistency,
        "drag_core_spread": drag_core_spread, "drag_all_spread": drag_all_spread,
        "checks": checks,
        "total_steps": sum(int(r["steps"]) for group in rows.values() for r in group),
        "wall_seconds": wall, "jobs": args.jobs,
        "all_checks_passed": all(c["passed"] for c in checks),
    }

    _write_csv(args.output / "rate_series.csv", rate,
               ["velocity_m_per_s", "steps", "apparent_shear_modulus_Pa", "ratio_to_C44",
                "window_spread_relative", "fit_relative_residual", "broken_bonds"])
    _write_csv(args.output / "finite_size_series.csv", width + height + cross,
               ["case", "nx", "ny", "nz", "velocity_m_per_s", "specimen_height_m",
                "inverse_width_a_over_Lx", "inverse_height_a_over_h",
                "apparent_shear_modulus_Pa", "ratio_to_C44", "window_spread_relative",
                "steps"])
    _write_csv(args.output / "drag_series.csv", drag,
               ["drag_Ns_per_m", "apparent_shear_modulus_Pa", "ratio_to_C44",
                "intercept_Pa", "window_spread_relative"])
    _write_traces(args.output / "traces", rate + width + height + cross + drag)

    figures = _figures(summary, args.output)
    DOCS_FIGURES.mkdir(parents=True, exist_ok=True)
    for name in ("rate_convergence.png", "finite_size.png"):
        (DOCS_FIGURES / name).write_bytes((args.output / name).read_bytes())
    # The published document docs/dem3d_shear_specimen_quasistatic.md is
    # maintained from summary.json (tables injected programmatically); the
    # script only produces the machine-readable record and the figures.

    stored = {k: v for k, v in summary.items() if not k.endswith("_series")}
    for key in ("rate_series", "width_series", "drag_series", "height_series",
                "cross_series"):
        stored[key] = [{k: v for k, v in r.items() if k != "history"} for r in summary[key]]
    stored["figures"] = figures
    with (args.output / "summary.json").open("w") as handle:
        json.dump(stored, handle, indent=1)

    for check in checks:
        print(("[PASS] " if check["passed"] else "[FAIL] ") + check["name"]
              + f"  value={check['value']:.4e} criterion={check['reference']:.1e}")
    print(f"G0(nx=9,nz=5) = {zero_rate['G0_Pa']:.1f} Pa = {zero_rate['G0_over_C44']:.4f} C44")
    print(f"G(infinite size) = {joint['G_infinite_Pa']:.1f} Pa = "
          f"{joint['c0_G_over_C44_at_infinite_size']:.4f} C44 "
          f"({100 * joint['relative_error_vs_C44']:.2f} % from C44)")
    print(f"wall {wall:.0f} s, {summary['total_steps']} steps, jobs={args.jobs}")
    return 0 if summary["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())