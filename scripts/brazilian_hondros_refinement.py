"""Lattice-resolution refinement scan of the Brazilian (Hondros) centre field.

Why this script exists
----------------------
``docs/dem3d_brazilian_hondros_closure.md`` settles the *statics* of the
Brazilian disc and reports an **honest FAIL**: the relaxed centre field sits at
``sigma_zz / sigma_xx = -2.015`` against the Hondros point-load value ``-3``
and the centre tension is ``+30.5%`` above ``2P / (pi D t)``.  It then names two
root causes and, crucially, leaves one question explicitly unanswered:

    "未做晶格分辨/接触弧加密扫描（recipe 已写进 §6），故『加密后能否进 15%』未知。"

This script runs that scan.  The design is a *geometric similarity* study: the
specimen diameter ``D = 0.30 m`` and thickness ``t = 0.06 m`` are held fixed
exactly, while the number of grains across the diameter ``N`` grows, so the
grain radius ``r = 0.15 / N`` shrinks and ``ny = t / (2 r) = 0.2 N`` keeps the
thickness at five grain diameters.  Only the discretisation changes; the
specimen does not.

Two competing predictions
-------------------------
``H1 (discretisation-limited)``
    The miss is finite-grain error: a ``+-D/10`` centre window that holds only
    ~1.5 lattice spacings, and a load introduced through a single grain column,
    both blur the field.  Refinement should drive the error towards zero.

``H2 (structure-limited)``
    The miss is the central-force lattice itself.  The same bond set has Born
    (Affine) constants ``C11 = 433333 Pa``, ``C12 = C44 = 233333 Pa``, i.e. a
    Cauchy solid with Zener anisotropy ``A = C44 / ((C11 - C12)/2) = 2.33`` and
    ``nu = 0.35``.  A cubic-anisotropic lattice with ``A = 2.33`` is a *material*
    property of the bond network, **scale invariant**: refining the spacing does
    not change it, so the error must plateau at a non-zero limit.  The Hondros
    solution assumes an isotropic continuum, which this network never is.

The scan reports, per resolution: the centre field, its error against both the
point-load reference and the disc's own anisotropy, the axial-profile spread,
and the Born constants -- the last of which is the arithmetic proof that H2's
cause cannot be refined away.

Scope: numerical verification of the apparatus.  Nothing here calibrates an ice
material or certifies a tensile strength.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from benchmark_dem3d_brazilian import MANIFEST  # noqa: E402
from brazilian_lattice_static import (  # noqa: E402
    lattice_moduli,
    solve_static_equilibrium,
)

#: Held fixed across every resolution so the study is pure discretisation.
DIAMETER_M = 0.30
THICKNESS_M = 0.06
#: Dimensionless platen travel of the reference (N=15) closure run:
#: 7.091300e-04 m at D = 0.30 m.
TRAVEL_RATIO = 7.0913e-04 / DIAMETER_M


def case_at_resolution(base: dict[str, Any], resolution: int) -> dict[str, Any]:
    """Rescale the manifest case onto ``resolution`` grains across the diameter.

    ``D`` and ``t`` are preserved exactly: ``radius_m = (D/2)/N`` and
    ``ny = t/(2 radius_m) = 0.2 N``.  For ``N = 15`` this reproduces the
    manifest case (radius 0.01 m, ny 3) bit for bit, which is the control that
    keeps the scan anchored to the published closure run.
    """
    if resolution < 4:
        raise ValueError("resolution must be at least 4 grains across the diameter")
    ny = THICKNESS_M / (DIAMETER_M / resolution)
    ny_rounded = int(round(ny))
    if abs(ny - ny_rounded) > 1e-9:
        raise ValueError(
            f"resolution {resolution} does not tile the thickness exactly "
            f"(ny = {ny}); pick a resolution that is a multiple of 5")
    if ny_rounded < 1:
        raise ValueError("thickness must span at least one grain")

    case = json.loads(json.dumps(base))  # deep copy, JSON-safe types only
    radius = (DIAMETER_M / 2.0) / resolution
    case["config"]["nx"] = resolution
    case["config"]["nz"] = resolution
    case["config"]["ny"] = ny_rounded
    case["config"]["radius_m"] = radius
    return case


def anisotropy_metrics(moduli: dict[str, float]) -> dict[str, float]:
    """Zener anisotropy and the Cauchy-solid indicator of the bond network.

    ``A = C44 / ((C11 - C12)/2)`` is 1 for an isotropic cubic crystal.  A
    central-force (normal-only) bond network cannot reach ``A = 1`` because it
    carries no tangential stiffness, and -- unlike a discretisation error -- ``A``
    is built from stiffness ratios only, so it is independent of grain size.
    """
    c11 = float(moduli["C11_Pa"])
    c12 = float(moduli["C12_Pa"])
    c44 = float(moduli["C44_Pa"])
    shear = 0.5 * (c11 - c12)
    return {
        "C11_Pa": c11,
        "C12_Pa": c12,
        "C44_Pa": c44,
        "zener_anisotropy": (c44 / shear) if shear else float("nan"),
        "cauchy_residual_rel": (abs(c12 - c44) / c44) if c44 else float("nan"),
        "shear_modulus_Pa": shear,
    }


def run_resolution(base: dict[str, Any], resolution: int, *,
                   increments: int = 20, max_iterations: int = 4000,
                   tolerance_Pa: float = 1e-4) -> dict[str, Any]:
    """Static equilibrium of the specimen at one lattice resolution."""
    case = case_at_resolution(base, resolution)
    travel = TRAVEL_RATIO * DIAMETER_M
    started = time.perf_counter()
    result = solve_static_equilibrium(
        case, travel, increments=increments,
        max_iterations=max_iterations, tolerance_Pa=tolerance_Pa,
    )
    elapsed = time.perf_counter() - started

    moduli = lattice_moduli(case)
    result["refinement"] = {
        "resolution_across_diameter": resolution,
        "grain_radius_m": float(case["config"]["radius_m"]),
        "grains_across_thickness": int(case["config"]["ny"]),
        "specimen_diameter_m": DIAMETER_M,
        "specimen_thickness_m": THICKNESS_M,
        "travel_m": travel,
        "travel_over_diameter": TRAVEL_RATIO,
        "wall_clock_s": elapsed,
        **anisotropy_metrics(moduli),
        "poisson_ratio": moduli.get("poisson_ratio"),
        "youngs_modulus_Pa": moduli.get("youngs_modulus_Pa"),
    }
    return result


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank resolutions on the two quantities the closure doc left open."""
    ordered = sorted(rows, key=lambda r: r["refinement"]["resolution_across_diameter"])
    table = []
    for r in ordered:
        ref = r["refinement"]
        ratio = r["centre_principal_ratio"]
        table.append({
            "N": ref["resolution_across_diameter"],
            "particles": r["particle_count"],
            "bonds": r["bond_count"],
            "centre_particles": r.get("centre_particle_count"),
            "contact_arc_ratio": r["contact_arc_ratio"],
            "contact_particles": r["top_contact_particles"],
            "centre_sigma_xx_Pa": r["centre_sigma_xx_Pa"],
            "centre_sigma_zz_Pa": r["centre_sigma_zz_Pa"],
            "split_stress_Pa": r["split_stress_Pa"],
            "centre_principal_ratio": ratio,
            "principal_ratio_rel_error": abs(ratio + 3.0) / 3.0,
            "centre_tension_rel_error": r["centre_tension_rel_error"],
            "axial_uniformity_rel": r["axial_uniformity_rel"],
            "equilibrium_residual_rel": r["equilibrium_residual_rel"],
            "load_balance_rel": r["load_balance_rel"],
            "converged": r["equilibrium_converged"],
            "zener_anisotropy": ref["zener_anisotropy"],
        })

    errors = [row["centre_tension_rel_error"] for row in table]
    ratios = [row["principal_ratio_rel_error"] for row in table]
    passes = [e < 0.15 for e in errors]
    tail = table[-1]
    return {
        "rows": table,
        "monotone_decreasing_tension_error": all(
            b < a for a, b in zip(errors, errors[1:])),
        "achieved_below_15pct": bool(tail["centre_tension_rel_error"] < 0.15),
        "any_below_15pct": any(passes),
        "finest_resolution": tail["N"],
        "finest_tension_error": tail["centre_tension_rel_error"],
        "finest_principal_ratio_error": tail["principal_ratio_rel_error"],
        "tension_error_span": max(errors) - min(errors),
        "principal_ratio_error_span": max(ratios) - min(ratios),
        "verdict": _verdict(table),
    }


def _fmt(row: dict[str, Any]) -> str:
    return "N=%d:%+.1f%%" % (row["N"], 100.0 * row["centre_tension_rel_error"])


def _verdict(table: list[dict[str, Any]]) -> str:
    """Answer the closure doc's open question in one sentence."""
    if not table:
        return "no resolutions computed"
    finest = table[-1]
    err = finest["centre_tension_rel_error"]
    if err < 0.15:
        monotone = all(b["centre_tension_rel_error"] < a["centre_tension_rel_error"]
                       for a, b in zip(table, table[1:]))
        if not monotone:
            return (f"NON-MONOTONE: the finest resolution N={finest['N']} lands inside the "
                    f"tolerance ({err:.2%}) but the sequence is not monotone "
                    f"({', '.join(_fmt(r) for r in table)}); "
                    f"a single in-band point is not convergence.")
        return (f"DISCRETISATION-LIMITED: tension error falls monotonically to {err:.2%} at N="
                f"{finest['N']} (< 15% threshold); refinement closes the manifest tolerance.")
    if len(table) >= 2:
        first = table[0]["centre_tension_rel_error"]
        if err < first:
            return (f"PLATEAU: error falls from {first:.2%} (N={table[0]['N']}) to {err:.2%} "
                    f"(N={finest['N']}) but stays above the 15% threshold; the residual is "
                    f"material (Zener anisotropy {finest['zener_anisotropy']:.2f}, "
                    f"scale invariant), not a grain-size artefact.")
        return (f"NO IMPROVEMENT: error {err:.2%} at N={finest['N']} does not beat "
                f"{first:.2%} at N={table[0]['N']}; refinement does not address the cause.")
    return f"single resolution: tension error {err:.2%}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="brazil_elastic_field")
    parser.add_argument("--resolutions", type=int, nargs="+",
                        default=[15, 20, 25, 30],
                        help="grains across the diameter; multiples of 5 tile t exactly")
    parser.add_argument("--increments", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=4000)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/dem3d_brazilian_refinement"))
    args = parser.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    base = next(c for c in manifest["cases"] if c["id"] == args.case)

    rows = []
    for resolution in args.resolutions:
        result = run_resolution(
            base, resolution, increments=args.increments,
            max_iterations=args.max_iterations,
        )
        rows.append(result)
        ref = result["refinement"]
        print(f"N={resolution:3d} particles={result['particle_count']:6d} "
              f"window={result.get('centre_particle_count'):3d} "
              f"ratio={result['centre_principal_ratio']:+.4f} "
              f"tension_err={result['centre_tension_rel_error']:+.4%} "
              f"spread={result['axial_uniformity_rel']:.3f} "
              f"resid={result['equilibrium_residual_rel']:.2e} "
              f"[{ref['wall_clock_s']:.1f}s]", flush=True)

    summary = summarise(rows)
    summary["reference"] = manifest["reference"]
    summary["acceptance"] = manifest["acceptance"]
    summary["method"] = ("static elastic equilibrium (L-BFGS on the bond+platen potential, "
                         "20-step load continuation) of manifest case '%s' rescaled to N "
                         "grains across a fixed D=%.2f m / t=%.2f m specimen"
                         % (args.case, DIAMETER_M, THICKNESS_M))
    summary["cases"] = rows

    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "brazilian_refinement.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "verdict", "achieved_below_15pct", "finest_resolution",
        "finest_tension_error", "monotone_decreasing_tension_error",
    )}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()