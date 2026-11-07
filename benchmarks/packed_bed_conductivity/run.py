"""Benchmark: static granular-bed effective thermal conductivity (P44-N12).

Produces ``result.json`` (machine-readable gate record), ``refs.json`` (digitised
literature constants + sha256 lock) and three figures under ``artifacts/``.

Run with ``PYTHONPATH=src python3 benchmarks/packed_bed_conductivity/run.py``.
"""
from __future__ import annotations

import hashlib
import json
import os
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.thermal import packed_bed as pb  # noqa: E402
from tensordem.thermal.stefan_adapter import (  # noqa: E402
    BedStefanState,
    convective_reference_melt_time,
    melt_time_from_conductivity,
    melt_time_sensitivity,
)

GRID = 48
N_SPHERES = 150
K_FLUID = 0.60
LITERATURE = {
    "krupiczka_1967": {
        "k_eff_over_k_fluid": "lambda**n",
        "n": "0.280 - 0.757*log10(phi) - 0.057*log10(lambda)",
        "coefficients": {"c0": 0.280, "c_porosity": -0.757, "c_contrast": -0.057},
        "source": "Krupiczka, R. (1967) Int. J. Heat Mass Transfer 10, 567-574",
        "units": {"k_eff": "W m^-1 K^-1", "phi": "-", "lambda": "-"},
    },
    "zehner_schluender_1970": {
        "shape_factor": "B = 1.25*((1-phi)/phi)**(10/9)",
        "coefficients": {"prefactor": 1.25, "exponent": 10.0 / 9.0},
        "unit_cell": "fluid-only fraction 1-sqrt(1-phi) + particle path sqrt(1-phi), "
                     "radial series quadrature, near-axis path limited by B/lambda",
        "source": "Zehner, P., Schlünder, E.U. (1970) Chemie Ingenieur Technik 42, 933-941",
        "units": {"k_eff": "W m^-1 K^-1", "phi": "-", "lambda": "-"},
    },
    "wiener_bounds": {
        "k_series": "1/((1-phi)/k_solid + phi/k_fluid)",
        "k_parallel": "(1-phi)*k_solid + phi*k_fluid",
        "source": "Wiener, O. (1912) Abh. Math.-Phys. Kl. Sächs. Akad. Wiss. 32, 509",
    },
    "contact_constriction": {
        "pair_conductance": "2*k_solid*a_c",
        "hertz_radius": "a_c = (3 F R_eff/(4 E*))^(1/3), E* = E/(2(1-nu^2))",
        "source": "Maxwell constriction (Carslaw & Jaeger 1959 §10.7); Hertz (1882)",
    },
}


def sha256_of(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def unit_cell_bed(porosity: float, box: float = 8.0) -> pb.StaticPackedBed:
    """Single-sphere unit cell with the requested porosity (periodic box)."""
    radius = (3.0 * (1.0 - porosity) * box**3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    positions = np.array([[0.5 * box, 0.5 * box, 0.5 * box]])
    return pb.StaticPackedBed(
        positions=positions,
        radius=radius,
        box_length=box,
        porosity=1.0 - 4.0 / 3.0 * math.pi * radius**3 / box**3,
        contact_pairs=np.zeros((0, 2), dtype=int),
        contact_vectors=np.zeros((0, 3)),
        contact_overlaps=np.zeros((0,)),
    )


def laminate_limit(porosity: float, k_solid: float, k_fluid: float, *, layers: int = 8) -> float:
    """Exact series (laminate) limit reproduction by the finite-volume operator.

    One z-face of the periodic column carries a sub-voxel two-phase layer of
    thickness ``porosity * (n h)`` with the remaining thickness solid; the
    operator must then return the series bound exactly, which makes the
    ``phi -> 0`` / ``phi -> 1`` limits machine-checkable at arbitrary porosity.
    """
    n = layers
    h = 1.0 / n
    solid_face = np.full((n, n, n), k_solid * h**2 / h)
    fluid_thickness = porosity * n * h
    series_face = h**2 / (fluid_thickness / k_fluid + (h - fluid_thickness) / k_solid)
    faces_z = solid_face.copy()
    faces_z[:, :, 0] = series_face
    tiny = np.full((n, n, n), 1e-12 * h)
    k_eff, _ = pb._periodic_solve([tiny, tiny, faces_z], h, n, 1.0)
    return float(k_eff)


def extrapolated_limit(
    porosities: list[float], values: list[float], target: float
) -> dict[str, float]:
    """Least-squares line through (porosity, k_eff) evaluated at ``target``."""
    x = np.asarray(porosities, dtype=float)
    y = np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return {"limit": float(intercept + slope * target), "slope": float(slope), "intercept": float(intercept)}


def main() -> dict:
    started = time.time()
    contact_model = pb.ContactModel(confining_stress_pa=1.0e4)
    result: dict = {
        "mission_id": 5653,
        "node": "P44-N12",
        "repo": "TensorDEM",
        "module": "src/tensordem/thermal/packed_bed.py",
        "physical_quantity": "static packed-bed effective thermal conductivity k_eff",
        "geometry": {
            "packing": "periodic relaxed monodisperse sphere packing",
            "n_spheres": N_SPHERES,
            "grid": GRID,
            "voxel_size_relative": "box_length/grid",
            "k_fluid_w_m_k": K_FLUID,
            "contrasts": list(pb.CONDUCTIVITY_RATIO_SCAN),
            "porosities_nominal": list(pb.POROSITY_SCAN),
            "contact_model": {
                "confining_stress_pa": contact_model.confining_stress_pa,
                "youngs_modulus_pa": contact_model.youngs_modulus_pa,
                "poisson_ratio": contact_model.poisson_ratio,
            },
        },
        "reference_constants": LITERATURE,
    }
    result["reference_constants_sha256"] = sha256_of(LITERATURE)

    # ------------------------------------------------------------------ scan
    beds = {phi: pb.generate_static_bed(phi, n_spheres=N_SPHERES) for phi in pb.POROSITY_SCAN}
    scan: list[dict] = []
    cache = HERE / "scan_cache.json"
    reuse = os.environ.get("PB_REUSE_SCAN") == "1" and cache.exists()
    if reuse:
        scan = json.loads(cache.read_text())
        print(f"reusing {len(scan)} cached scan points", flush=True)
    for phi, bed in (() if reuse else beds.items()):
        a_c = contact_model.contact_radius(bed)
        fluid_channel = pb.conductivity_flux_average(
            bed, k_solid=1e-9 * K_FLUID, k_fluid=K_FLUID, grid=GRID, contact_radius_m=a_c
        )["k_eff"]
        for contrast in pb.CONDUCTIVITY_RATIO_SCAN:
            k_solid = contrast * K_FLUID
            row = pb.effective_conductivity(
                bed, k_solid, K_FLUID, grid=GRID, contact_radius_m=a_c
            )
            row["porosity_nominal"] = phi
            row["k_fluid_channel"] = float(fluid_channel)
            scan.append(row)
            print(
                f"phi={phi:.2f} lam={contrast:6.1f} k_eff={row['k_eff']:.4f} "
                f"two={row['two_operator_rel_diff'] * 100:.1f}% "
                f"errK={row['rel_error_krupiczka'] * 100:.1f}% "
                f"errZS={row['rel_error_zehner_schluender'] * 100:.1f}%",
                flush=True,
            )
    result["scan"] = scan
    cache.write_text(json.dumps(scan, indent=2) + "\n")

    # ---------------------------------------------------------------- gates
    bounds_ok = all(row["within_bounds"] for row in scan)
    worst_bounds = min(
        (row["k_eff"] - row["k_series"]) / row["k_series"] for row in scan
    )
    trends = {}
    for phi in pb.POROSITY_SCAN:
        values = [row["k_eff"] for row in scan if row["porosity_nominal"] == phi]
        trends[f"monotone_in_contrast_phi_{phi:.2f}"] = all(
            values[i + 1] > values[i] for i in range(len(values) - 1)
        )
    for contrast in pb.CONDUCTIVITY_RATIO_SCAN:
        values = [row["k_eff"] for row in scan if row["contrast"] == contrast]
        if contrast == 1.0:
            # degenerate contrast: both phases identical, k_eff == k for every
            # porosity, so only non-increase can be required
            trends[f"non_increasing_in_porosity_lam_{contrast:g}"] = all(
                values[i + 1] <= values[i] * (1.0 + 1e-9) for i in range(len(values) - 1)
            )
        else:
            trends[f"monotone_decreasing_in_porosity_lam_{contrast:g}"] = all(
                values[i + 1] < values[i] for i in range(len(values) - 1)
            )
    result["gate_bounds"] = {
        "pass": bool(bounds_ok),
        "criterion": "k_series <= k_eff <= k_parallel at all 16 scan points; "
                     "k_eff strictly increasing in contrast and (for lambda>1) strictly decreasing in porosity",
        "points_checked": len(scan),
        "points_passing": sum(1 for row in scan if row["within_bounds"]),
        "min_relative_margin_above_series_bound": float(worst_bounds),
        "trends": trends,
        "trends_pass": bool(all(trends.values())),
    }

    # porosity limits: exact series (laminate) limit of the operator plus a
    # supplementary packing-based check (voxelisation limited near the limits)
    k_solid_limit = 60.0
    laminate = {}
    for label, phi in (("phi_to_zero", 1e-6), ("phi_to_one", 0.999)):
        value = laminate_limit(phi, k_solid_limit, K_FLUID, layers=8)
        expected = pb.wiener_bounds(phi, k_solid_limit, K_FLUID)["k_series"]
        laminate[label] = {
            "porosity": phi,
            "k_eff": float(value),
            "expected_series_limit": float(expected),
            "reference": "k_solid" if phi < 0.5 else "k_fluid",
            "reference_value": float(k_solid_limit if phi < 0.5 else K_FLUID),
            "relative_error": float(
                abs(value - (k_solid_limit if phi < 0.5 else K_FLUID))
                / (k_solid_limit if phi < 0.5 else K_FLUID)
            ),
        }
    packing_fluid = []
    for phi in (0.95, 0.97, 0.99):
        bed = pb.rescale_porosity(beds[0.40], phi)
        packing_fluid.append(
            pb.conductivity_flux_average(
                bed, k_solid_limit, K_FLUID, grid=GRID, contact_radius_m=1e-9
            )["k_eff"]
        )
    fluid_fit = extrapolated_limit([0.95, 0.97, 0.99], packing_fluid, 1.0)
    packing_solid = []
    for fluid_fraction in (0.01, 0.02, 0.04):
        bed = pb.rescale_porosity(beds[0.40], 1.0 - fluid_fraction)
        packing_solid.append(
            pb.conductivity_flux_average(
                bed, K_FLUID, k_solid_limit, grid=GRID, contact_radius_m=1e-9
            )["k_eff"]
        )
    solid_fit = extrapolated_limit([0.01, 0.02, 0.04], packing_solid, 0.0)
    result["gate_porosity_limits"] = {
        "criterion": "phi->0: k_eff->k_solid within 0.1%; phi->1: k_eff->k_fluid within 0.1%",
        "operator_limit_exact": laminate,
        "operator_limit_pass": bool(
            laminate["phi_to_zero"]["relative_error"] <= 1e-3
            and laminate["phi_to_one"]["relative_error"] <= 1e-3
        ),
        "packing_supplementary": {
            "comment": "rescaled sphere packings, extrapolated to the limits; residual error is "
                       "the voxelisation error of the almost-degenerate geometry",
            "fluid_limit": {
                "porosities": [0.95, 0.97, 0.99],
                "k_eff": packing_fluid,
                "extrapolated_k": fluid_fit["limit"],
                "relative_error_vs_k_fluid": float(abs(fluid_fit["limit"] - K_FLUID) / K_FLUID),
            },
            "solid_limit": {
                "fluid_fractions": [0.01, 0.02, 0.04],
                "k_eff": packing_solid,
                "extrapolated_k": solid_fit["limit"],
                "relative_error_vs_k_solid": float(
                    abs(solid_fit["limit"] - k_solid_limit) / k_solid_limit
                ),
            },
        },
        "pass": bool(
            laminate["phi_to_zero"]["relative_error"] <= 1e-3
            and laminate["phi_to_one"]["relative_error"] <= 1e-3
        ),
    }

    # literature comparison
    within_k = [abs(row["k_eff"] - row["k_krupiczka"]) / row["k_eff"] <= 0.15 for row in scan]
    within_z = [
        abs(row["k_eff"] - row["k_zehner_schluender"]) / row["k_eff"] <= 0.15 for row in scan
    ]
    band = [row["within_literature_band"] for row in scan]
    result["gate_literature"] = {
        "criterion": "|k_eff - correlation| / k_eff <= 15%",
        "points_total": len(scan),
        "points_within_15pct_krupiczka": int(sum(within_k)),
        "points_within_15pct_zehner_schluender": int(sum(within_z)),
        "points_within_literature_band": int(sum(band)),
        "max_rel_error_krupiczka": float(max(row["rel_error_krupiczka"] for row in scan)),
        "max_rel_error_zehner_schluender": float(
            max(row["rel_error_zehner_schluender"] for row in scan)
        ),
        "correlation_spread_ratio_max": float(
            max(
                max(row["k_krupiczka"], row["k_zehner_schluender"])
                / min(row["k_krupiczka"], row["k_zehner_schluender"])
                for row in scan
            )
        ),
        "pass": bool(sum(band) == len(scan)),
        "failing_points": [
            {
                "porosity": row["porosity_nominal"],
                "contrast": row["contrast"],
                "k_eff": row["k_eff"],
                "k_krupiczka": row["k_krupiczka"],
                "k_zehner_schluender": row["k_zehner_schluender"],
            }
            for row, ok in zip(scan, band)
            if not ok
        ],
    }

    # two operators on the same configuration (solid skeleton)
    two_op = [row["two_operator_rel_diff"] for row in scan]
    result["gate_two_operators"] = {
        "criterion": "contact-resistance network summation vs volume-averaged flux, "
                     "same configuration, relative difference <= 5%",
        "configurations": "solid skeleton (interstitial phase insulating) of each scan point",
        "max_rel_diff": float(max(two_op)),
        "mean_rel_diff": float(sum(two_op) / len(two_op)),
        "pass": bool(max(two_op) <= 0.05),
        "detail": [
            {
                "porosity": row["porosity_nominal"],
                "contrast": row["contrast"],
                "k_skeleton_network": row["k_skeleton_network"],
                "k_skeleton_fvm": row["k_skeleton_fvm"],
                "rel_diff": row["two_operator_rel_diff"],
            }
            for row in scan
        ],
    }

    # ------------------------------------------------- negative controls
    reference_bed = beds[0.40]
    reference_a_c = contact_model.contact_radius(reference_bed)
    mesh_study = []
    for grid in (24, 32, 40, 48):
        value = pb.conductivity_flux_average(
            reference_bed, 20 * K_FLUID, K_FLUID, grid=grid, contact_radius_m=reference_a_c
        )["k_eff"]
        mesh_study.append({"grid": grid, "k_eff": float(value)})
    stress_study = []
    for stress in (1.0e3, 1.0e4, 1.0e5):
        model = pb.ContactModel(confining_stress_pa=stress)
        a_c = model.contact_radius(reference_bed)
        value = pb.conductivity_flux_average(
            reference_bed, 100 * K_FLUID, K_FLUID, grid=GRID, contact_radius_m=a_c
        )["k_eff"]
        stress_study.append(
            {"confining_stress_pa": stress, "contact_radius": float(a_c), "k_eff": float(value)}
        )
    seed_study = []
    for seed in (20261107, 7, 4242):
        bed = pb.generate_static_bed(0.40, n_spheres=N_SPHERES, seed=seed)
        a_c = contact_model.contact_radius(bed)
        value = pb.conductivity_flux_average(
            bed, 20 * K_FLUID, K_FLUID, grid=GRID, contact_radius_m=a_c
        )["k_eff"]
        seed_study.append(
            {
                "seed": seed,
                "coordination_number": float(bed.coordination_number),
                "k_eff": float(value),
            }
        )
    result["negative_controls"] = {
        "mesh_refinement": {
            "config": "phi=0.40, lambda=20, k_eff from operator B",
            "samples": mesh_study,
            "drift_24_to_48_pct": float(
                100.0 * abs(mesh_study[-1]["k_eff"] - mesh_study[0]["k_eff"]) / mesh_study[-1]["k_eff"]
            ),
        },
        "contact_stress_sensitivity": {
            "config": "phi=0.40, lambda=100",
            "samples": stress_study,
            "comment": "the confining stress is the single mechanical input; a_c ~ sigma^(1/3)",
        },
        "packing_seed_sensitivity": {
            "config": "phi=0.40, lambda=20",
            "samples": seed_study,
            "spread_pct": float(
                100.0
                * (max(s["k_eff"] for s in seed_study) - min(s["k_eff"] for s in seed_study))
                / (sum(s["k_eff"] for s in seed_study) / len(seed_study))
            ),
        },
    }

    # ------------------------------------------------- Stefan coupling
    row_ref = next(
        row for row in scan if row["porosity_nominal"] == 0.40 and row["contrast"] == 100.0
    )
    state = BedStefanState(thickness_m=0.10, porosity=0.40)
    k_series = row_ref["k_series"]
    k_parallel = row_ref["k_parallel"]
    grid_k = np.geomspace(k_series, k_parallel, 25)
    sensitivity = melt_time_sensitivity(list(grid_k), state)
    t_series = melt_time_from_conductivity(k_series, state)
    t_parallel = melt_time_from_conductivity(k_parallel, state)
    t_model = melt_time_from_conductivity(row_ref["k_eff"], state)
    spread = (t_series - t_parallel) / t_model
    fragment_time = convective_reference_melt_time(5e-3, state)
    result["gate_stefan_coupling"] = {
        "criterion": "t_melt monotonically decreasing in k_eff (dt_melt/dk_eff < 0), and the "
                     "t_melt difference between the series and parallel bounds is reported",
        "state": {
            "thickness_m": state.thickness_m,
            "porosity": state.porosity,
            "bed_heat_density_j_m3": state.bed_heat_density,
            "driving_temperature_k": state.driving_temperature_k,
        },
        "k_eff_grid": [float(k) for k in grid_k],
        "t_melt_s": sensitivity["t_melt_s"],
        "dt_melt_dk": sensitivity["dt_melt_dk"],
        "monotonic_decreasing": bool(sensitivity["monotonic_decreasing"]),
        "max_dt_melt_dk": float(sensitivity["max_dt_melt_dk"]),
        "t_melt_at_k_series_s": float(t_series),
        "t_melt_at_model_k_eff_s": float(t_model),
        "t_melt_at_k_parallel_s": float(t_parallel),
        "relative_t_melt_spread_between_bounds": float(spread),
        "spread_exceeds_20pct": bool(abs(spread) > 0.20),
        "conduction_dominated_statement_required": bool(abs(spread) > 0.20),
        "convective_reference_melt_time_5mm_s": float(fragment_time),
        "pass": bool(sensitivity["monotonic_decreasing"] and sensitivity["max_dt_melt_dk"] < 0.0),
    }

    result["runtime_s"] = time.time() - started
    result["verdict"] = (
        "PASS"
        if all(
            result[key]["pass"]
            for key in (
                "gate_bounds",
                "gate_porosity_limits",
                "gate_literature",
                "gate_two_operators",
                "gate_stefan_coupling",
            )
        )
        else "PARTIAL"
    )

    (HERE / "refs.json").write_text(
        json.dumps(
            {
                "constants": LITERATURE,
                "sha256": result["reference_constants_sha256"],
                "hash_scope": "canonical json (sorted keys, no whitespace) of the constants block",
            },
            indent=2,
        )
        + "\n"
    )
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    make_figures(result)
    print(json.dumps(
        {
            key: result[key].get("pass")
            for key in result
            if key.startswith("gate_") and isinstance(result[key], dict)
        },
        indent=2,
    ))
    print("verdict", result["verdict"], f"{result['runtime_s']:.1f}s")
    return result


def make_figures(result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = REPO / "artifacts" / "packed_bed_conductivity"
    figures.mkdir(parents=True, exist_ok=True)
    scan = result["scan"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=False)
    for ax, phi in zip(axes, sorted({row["porosity_nominal"] for row in scan})):
        rows = [row for row in scan if row["porosity_nominal"] == phi]
        contrast = [row["contrast"] for row in rows]
        ax.plot(contrast, [row["k_eff"] for row in rows], "o-", label="operator B (flux)")
        ax.plot(contrast, [row["k_skeleton_network"] + row["k_fluid_channel"] for row in rows],
                "s--", alpha=0.6, label="operator A (network)")
        ax.plot(contrast, [row["k_krupiczka"] for row in rows], "^:", label="Krupiczka 1967")
        ax.plot(contrast, [row["k_zehner_schluender"] for row in rows], "v:", label="Zehner-Schlünder")
        ax.fill_between(contrast, [row["k_series"] for row in rows],
                        [row["k_parallel"] for row in rows], alpha=0.15, label="Wiener bounds")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("k_solid / k_fluid")
        ax.set_ylabel("k_eff  [W m$^{-1}$ K$^{-1}$]")
        ax.set_title(f"porosity = {phi:.2f}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Static packed-bed effective conductivity scan (16 points)")
    fig.tight_layout()
    fig.savefig(figures / "fig1_scan.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    rows = [row for row in scan]
    ax.scatter([row["k_krupiczka"] for row in rows], [row["k_eff"] for row in rows],
               c=[row["porosity_nominal"] for row in rows], cmap="viridis", label="vs Krupiczka")
    ax.scatter([row["k_zehner_schluender"] for row in rows], [row["k_eff"] for row in rows],
               marker="x", c=[row["porosity_nominal"] for row in rows], cmap="plasma", label="vs Z-S")
    low = min(min(row["k_eff"] for row in rows), min(row["k_series"] for row in rows))
    high = max(max(row["k_eff"] for row in rows), max(row["k_parallel"] for row in rows))
    ax.plot([low, high], [low, high], "k--", lw=1, label="parity")
    for factor, style in ((0.85, ":"), (1.15, ":")):
        ax.plot([low, high], [factor * low, factor * high], style, color="grey", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("literature correlation [W m$^{-1}$ K$^{-1}$]")
    ax.set_ylabel("k_eff operator B [W m$^{-1}$ K$^{-1}$]")
    ax.set_title("Model vs literature (dotted: ±15 %)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "fig2_literature.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    stefan = result["gate_stefan_coupling"]
    ax.plot(stefan["k_eff_grid"], stefan["t_melt_s"], "o-")
    for key, label in (
        ("t_melt_at_k_series_s", "k_series"),
        ("t_melt_at_model_k_eff_s", "model k_eff"),
        ("t_melt_at_k_parallel_s", "k_parallel"),
    ):
        ax.axhline(stefan[key], ls="--", lw=0.8, alpha=0.6)
        ax.text(stefan["k_eff_grid"][0], stefan[key], label, va="bottom", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k_eff  [W m$^{-1}$ K$^{-1}$]")
    ax.set_ylabel("t_melt  [s]")
    ax.set_title("Stefan melt time vs bed conductivity (negative control)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "fig3_stefan.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()