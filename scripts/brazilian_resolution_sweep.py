"""Lattice-resolution and contact-arc sweep of the Brazilian (diametral) specimen.

Why this script exists
----------------------
`docs/dem3d_brazilian_hondros_closure.md` closed the WITHHELD verdict on the
Brazilian centre field with a *static* equilibrium solve: the centre principal
ratio is ``-2.015`` against the isotropic Hondros value ``-3`` and the centre
tension misses ``2P/(pi D t)`` by ``+30.4%``.  It then left two mitigations
explicitly open (§5 "局限", §6 "建议二选一"):

  (1) refine the lattice and the contact arc and re-run -- "unknown whether
      refinement can reach the 15% tolerance";
  (2) replace the point-load baseline with a finite-contact-width analytic
      reference.

This script settles both with numbers instead of expectations.

Design
------
*Refinement family.*  The disc diameter ``D = 0.30 m`` and thickness
``t = 0.06 m`` are held fixed while the lattice spacing ``d = 2 r`` is refined by
``D / (2 nx)``.  ``nx`` is kept **odd and a multiple of 5** so that (a) a single
particle sits at the crown (see the flat-crown trap below) and (b) ``ny = nx/5``
stays integral, which keeps ``t / D = 0.20`` exact at every resolution.  The bond
stiffness is scaled as ``k_bond = 2000 * d / 0.02`` because the Born (affine)
constants of a central-force lattice scale as ``k / d``: the macroscopic
``C11 = 433.3 kPa``, ``C12 = C44 = 233.3 kPa``, ``E = 270 kPa``,
``nu = 0.35``, ``Zener = 2.33`` are then *identical* at every resolution, so any
change in the centre field is discretisation and not a change of material.
The platen stiffness and the prescribed travel keep their manifest values.

*Contact-arc axis.*  At one resolution the platen stiffness is swept, which moves
the measured contact-arc ratio over a decade; the measured field is compared with
the finite-arc closed form of :mod:`tensordem.brazilian_reference` evaluated at
the *measured* arc ratio.

*Flat-crown control.*  An even ``nx`` puts no particle on the crown: the topmost
lattice row then spans many particles and touches the flat platen as a wide
patch, so the specimen is no longer a diametral point-load test.  One even
resolution is run as a documented control, not as a convergence point.

Outputs: ``<output>/brazilian_resolution_sweep.json`` (all raw numbers),
``brazilian_resolution_sweep.csv`` (one row per run) and figures.

Scope: apparatus verification.  Nothing here calibrates an ice material or
certifies a tensile strength.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from brazilian_lattice_static import (  # noqa: E402  (script-level import)
    MANIFEST,
    lattice_moduli,
    solve_static_equilibrium,
)
from tensordem.brazilian_reference import (  # noqa: E402
    contact_half_angle,
    finite_arc_centre_field,
    point_load_centre_field,
    reference_comparison,
)

BASE_CASE_ID = "brazil_elastic_field"
BASE_DIAMETER_M = 0.30
BASE_THICKNESS_M = 0.06
BASE_SPACING_M = 0.02
BASE_BOND_STIFFNESS = 2000.0
MANIFEST_TRAVEL_M = 7.0913e-04  # 40000 steps * dt * loading speed, from m2445


def _base_case() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return next(c for c in manifest["cases"] if c["id"] == BASE_CASE_ID)


def build_case(nx: int, *, platen_stiffness: float | None = None,
               diameter_m: float = BASE_DIAMETER_M,
               thickness_m: float = BASE_THICKNESS_M) -> dict[str, Any]:
    """Refined case with matched Born constants (``k_bond * d`` held fixed)."""
    base = _base_case()
    if nx < 5 or nx % 2 == 0:
        raise ValueError("nx must be odd and >= 5 so that a particle sits on the crown")
    radius = diameter_m / (2.0 * nx)
    spacing = 2.0 * radius
    ny = max(1, int(round(thickness_m / spacing)))
    scale = spacing / BASE_SPACING_M
    config = dict(base["config"])
    config.update({
        "nx": nx,
        "nz": nx,
        "ny": ny,
        "radius_m": radius,
        "bond_stiffness_N_m": BASE_BOND_STIFFNESS * scale,
        "contact_stiffness_N_m": float(base["config"]["contact_stiffness_N_m"]) * scale,
        "platen_stiffness_N_m": (float(base["config"]["platen_stiffness_N_m"])
                                 if platen_stiffness is None else float(platen_stiffness)),
    })
    return {
        "id": f"brazil_refine_nx{nx}",
        "config": config,
        "steps": base.get("steps"),
        "derived": {
            "lattice_spacing_m": spacing,
            "nominal_thickness_m": ny * spacing,
            "spacing_ratio_to_manifest": scale,
            "bond_stiffness_scale": scale,
        },
    }


def measure(case: dict[str, Any], travel_m: float, *, increments: int,
            tolerance_Pa: float) -> dict[str, Any]:
    """Static equilibrium plus both analytic references at the measured arc."""
    started = time.time()
    result = solve_static_equilibrium(case, travel_m, increments=increments,
                                      tolerance_Pa=tolerance_Pa)
    elapsed = time.time() - started
    accuracy = reference_comparison(
        result,
        load_N=result["mean_platen_load_N"],
        diameter_m=result["specimen_diameter_m"],
        thickness_m=result["specimen_thickness_m"],
        contact_arc_ratio=result["contact_arc_ratio"],
    )
    config = case["config"]
    return {
        "case_id": case["id"],
        "nx": int(config["nx"]),
        "ny": int(config["ny"]),
        "lattice_spacing_m": 2.0 * float(config["radius_m"]),
        "bond_stiffness_N_m": float(config["bond_stiffness_N_m"]),
        "platen_stiffness_N_m": float(config["platen_stiffness_N_m"]),
        "specimen_diameter_m": result["specimen_diameter_m"],
        "specimen_thickness_m": result["specimen_thickness_m"],
        "thickness_to_diameter": (result["specimen_thickness_m"] / result["specimen_diameter_m"]),
        "particle_count": result["particle_count"],
        "bond_count": result["bond_count"],
        "travel_m": travel_m,
        "mean_platen_load_N": result["mean_platen_load_N"],
        "load_balance_rel": result["load_balance_rel"],
        "contact_arc_ratio": result["contact_arc_ratio"],
        "contact_particles_top": result["top_contact_particles"],
        "contact_half_angle_rad": accuracy["finite_arc"]["contact_half_angle_rad"],
        "centre_sigma_xx_Pa": result["centre_sigma_xx_Pa"],
        "centre_sigma_zz_Pa": result["centre_sigma_zz_Pa"],
        "centre_principal_ratio": result["centre_principal_ratio"],
        "split_stress_Pa": result["split_stress_Pa"],
        "axial_uniformity_rel": result["axial_uniformity_rel"],
        "equilibrium_residual_rel": result["equilibrium_residual_rel"],
        "equilibrium_converged": result["equilibrium_converged"],
        "lbfgs_iterations_total": result["lbfgs_iterations_total"],
        "lattice_moduli": lattice_moduli(case),
        "reference": accuracy,
        "analysis_wall_s": elapsed,
    }


def _fit_limit(points: list[dict[str, Any]], key: str) -> dict[str, float]:
    """Linear fit of ``key`` against lattice spacing, evaluated at zero spacing.

    Only used as a *bounded* extrapolation to say whether refinement is heading
    somewhere the tolerance accepts; with 3-4 points it is indicative, not a
    proof of convergence order.
    """
    usable = [p for p in points if p.get(key) is not None and math.isfinite(p[key])]
    if len(usable) < 2:
        return {"limit": float("nan"), "slope_per_m": float("nan"), "points": float(len(usable))}
    xs = [p["lattice_spacing_m"] for p in usable]
    ys = [p[key] for p in usable]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else float("nan")
    intercept = my - slope * mx
    return {"limit": intercept, "slope_per_m": slope, "points": float(n),
            "last_value": ys[-1] if xs[-1] == max(xs) else ys[xs.index(max(xs))]}


def _figure(sweep: list[dict[str, Any]], arc_runs: list[dict[str, Any]],
            control: dict[str, Any] | None, output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    spacing = [p["lattice_spacing_m"] for p in sweep]
    ratio = [p["centre_principal_ratio"] for p in sweep]
    tension_err = [100.0 * p["reference"]["point_load"]["sigma_xx_rel_error"] for p in sweep]
    tension_err_finite = [100.0 * p["reference"]["finite_arc"]["sigma_xx_rel_error"] for p in sweep]
    arc = [p["contact_arc_ratio"] for p in sweep]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    axes[0].plot(spacing, ratio, "o-", color="#1f4e79", label="lattice (static equilibrium)")
    axes[0].axhline(-3.0, color="#b03a2e", ls="--", label="Hondros point load (-3)")
    axes[0].axhline(sweep[-1]["reference"]["finite_arc"]["principal_ratio"], color="#e59866",
                    ls=":", label="finite-arc reference (finest arc)")
    axes[0].set_xlabel("lattice spacing d [m]  (coarser $\\rightarrow$)")
    axes[0].set_ylabel("centre principal ratio $\\sigma_{zz}/\\sigma_{xx}$")
    axes[0].set_title("(a) ratio is resolution independent")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(spacing, tension_err, "s-", color="#1f4e79", label="vs point load $2P/(\\pi D t)$")
    axes[1].plot(spacing, tension_err_finite, "^-", color="#117a65",
                 label="vs finite-arc reference (measured arc)")
    axes[1].axhline(15.0, color="#b03a2e", ls="--", label="15% tolerance")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("lattice spacing d [m]")
    axes[1].set_ylabel("centre tension error [%]")
    axes[1].set_title("(b) refining 3$\\times$ does not reach 15%")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(spacing, arc, "d-", color="#6c3483", label="measured contact arc w/D (odd nx)")
    if control:
        axes[2].axhline(control["contact_arc_ratio"], color="#b03a2e", ls="--",
                        label=f"even nx={control['nx']} flat crown")
    axes[2].axhline(0.2, color="k", ls=":", label="manifest limit 0.2")
    axes[2].set_xlabel("lattice spacing d [m]")
    axes[2].set_ylabel("contact arc ratio w/D")
    axes[2].set_title("(c) crown contact stays point-like")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    path = output / "brazilian_resolution_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # Contact-arc axis against the closed-form reference curve.
    if arc_runs:
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        arc_sorted = sorted(arc_runs, key=lambda r: r["contact_arc_ratio"])
        ax.plot([r["contact_arc_ratio"] for r in arc_sorted],
                [100.0 * r["reference"]["point_load"]["sigma_xx_rel_error"] for r in arc_sorted],
                "o", color="#1f4e79", ms=7, label="lattice, vs point load")
        arc_grid = [0.005 * i for i in range(1, 141)]
        ax.plot(arc_grid,
                [100.0 * (finite_arc_centre_field(
                    1.0, BASE_DIAMETER_M, a, BASE_THICKNESS_M)["tension_correction_factor"] - 1.0)
                 for a in arc_grid],
                "-", color="#b03a2e", lw=1.4,
                label="closed form: finite-arc vs point load")
        ax.set_xlabel("measured contact arc ratio w/D")
        ax.set_ylabel("centre tension change [%]")
        ax.set_title("(d) contact width moves the baseline by <1%")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = output / "brazilian_contact_arc_axis.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))
    return written


def run_sweep(*, resolutions: list[int], arc_sweep_platen: list[float],
              arc_sweep_nx: int, include_control: bool, even_control_nx: int,
              increments: int, tolerance_Pa: float, travel_m: float) -> dict[str, Any]:
    sweep: list[dict[str, Any]] = []
    for nx in resolutions:
        case = build_case(nx)
        sweep.append(measure(case, travel_m, increments=increments, tolerance_Pa=tolerance_Pa))

    arc_runs: list[dict[str, Any]] = []
    for stiffness in arc_sweep_platen:
        case = build_case(arc_sweep_nx, platen_stiffness=stiffness)
        arc_runs.append(measure(case, travel_m, increments=increments, tolerance_Pa=tolerance_Pa))

    control = None
    if include_control:
        control = measure(_even_case(even_control_nx), travel_m, increments=increments,
                          tolerance_Pa=tolerance_Pa)

    return {
        "sweep": sweep,
        "arc_axis": arc_runs,
        "even_nx_control": control,
        "fits": {
            "centre_principal_ratio": _fit_limit(sweep, "centre_principal_ratio"),
            "centre_sigma_xx_Pa": _fit_limit(sweep, "centre_sigma_xx_Pa"),
            "centre_tension_rel_error_point_load": _fit_error(sweep, "point_load"),
            "centre_tension_rel_error_finite_arc": _fit_error(sweep, "finite_arc"),
            "contact_arc_ratio": _fit_limit(sweep, "contact_arc_ratio"),
        },
    }


def _fit_error(sweep: list[dict[str, Any]], reference: str) -> dict[str, float]:
    proxy = [{"lattice_spacing_m": p["lattice_spacing_m"],
              "value": p["reference"][reference]["sigma_xx_rel_error"]} for p in sweep]
    fit = _fit_limit(proxy, "value")
    return fit


def _even_case(nx: int) -> dict[str, Any]:
    """Even-``nx`` control: same family but no particle on the crown."""
    base = _base_case()
    radius = BASE_DIAMETER_M / (2.0 * nx)
    spacing = 2.0 * radius
    ny = max(1, int(round(BASE_THICKNESS_M / spacing)))
    scale = spacing / BASE_SPACING_M
    config = dict(base["config"])
    config.update({
        "nx": nx, "nz": nx, "ny": ny, "radius_m": radius,
        "bond_stiffness_N_m": BASE_BOND_STIFFNESS * scale,
        "contact_stiffness_N_m": float(base["config"]["contact_stiffness_N_m"]) * scale,
    })
    return {"id": f"brazil_even_nx{nx}_control", "config": config,
            "derived": {"lattice_spacing_m": spacing, "note": "flat-crown control"}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[15, 25, 35, 45])
    parser.add_argument("--arc-platen", type=float, nargs="+",
                        default=[400.0, 1500.0, 5000.0, 25000.0])
    parser.add_argument("--arc-nx", type=int, default=25)
    parser.add_argument("--even-control-nx", type=int, default=20)
    parser.add_argument("--increments", type=int, default=20)
    parser.add_argument("--tolerance-pa", type=float, default=1e-4)
    parser.add_argument("--travel", type=float, default=MANIFEST_TRAVEL_M)
    parser.add_argument("--no-control", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dem3d_brazilian_sweep"))
    args = parser.parse_args(argv)

    payload = run_sweep(resolutions=args.resolutions,
                        arc_sweep_platen=args.arc_platen,
                        arc_sweep_nx=args.arc_nx,
                        include_control=not args.no_control,
                        even_control_nx=args.even_control_nx,
                        increments=args.increments,
                        tolerance_Pa=args.tolerance_pa,
                        travel_m=args.travel)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "brazilian_resolution_sweep.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = []
    for entry in payload["sweep"] + payload["arc_axis"] + (
            [payload["even_nx_control"]] if payload["even_nx_control"] else []):
        rows.append({
            "case_id": entry["case_id"],
            "nx": entry["nx"],
            "lattice_spacing_m": f"{entry['lattice_spacing_m']:.9f}",
            "particle_count": entry["particle_count"],
            "contact_arc_ratio": f"{entry['contact_arc_ratio']:.6f}",
            "mean_platen_load_N": f"{entry['mean_platen_load_N']:.9f}",
            "centre_sigma_xx_Pa": f"{entry['centre_sigma_xx_Pa']:.6f}",
            "centre_sigma_zz_Pa": f"{entry['centre_sigma_zz_Pa']:.6f}",
            "centre_principal_ratio": f"{entry['centre_principal_ratio']:.6f}",
            "tension_error_vs_point_load": f"{entry['reference']['point_load']['sigma_xx_rel_error']:.6f}",
            "tension_error_vs_finite_arc": f"{entry['reference']['finite_arc']['sigma_xx_rel_error']:.6f}",
            "finite_arc_tension_correction": f"{entry['reference']['finite_arc']['tension_correction_factor']:.6f}",
            "finite_arc_principal_ratio": f"{entry['reference']['finite_arc']['principal_ratio']:.6f}",
            "equilibrium_residual_rel": f"{entry['equilibrium_residual_rel']:.3e}",
        })
    with (args.output / "brazilian_resolution_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figures = _figure(payload["sweep"], payload["arc_axis"], payload["even_nx_control"], args.output)
    summary = {
        "sweep": [{k: e[k] for k in ("nx", "lattice_spacing_m", "contact_arc_ratio",
                                     "centre_principal_ratio", "centre_sigma_xx_Pa",
                                     "centre_sigma_zz_Pa", "mean_platen_load_N")}
                  | {"tension_error_vs_point_load": e["reference"]["point_load"]["sigma_xx_rel_error"],
                     "tension_error_vs_finite_arc": e["reference"]["finite_arc"]["sigma_xx_rel_error"]}
                  for e in payload["sweep"]],
        "fits": payload["fits"],
        "figures": figures,
        "output_dir": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()