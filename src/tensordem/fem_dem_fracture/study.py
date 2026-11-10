"""End-to-end FEM -> DEM fracture handoff study (campaign p302w1-n03).

The chain, in the order it runs
-------------------------------
1. **FEM:** :func:`tensordem.fem_dem_fracture.plate_fem.solve_plate` solves the
   buoyancy-supported (Winkler) ice plate under a non-uniform contact pressure
   field and returns the moment field; the flexural stress is
   ``sigma = 6 |M| / t**2``.
2. **Flexural failure load:** because the problem is linear, the pressure that
   puts ``max sigma`` on ``sigma_flexure`` follows from a single solve
   (``scale_to_failure``).  This is criterion C1: at that state ``sigma_max =
   sigma_flexure`` and the artefact is written.
3. **Crack network + floe partition:** :func:`trace_crack_network` places
   nucleation sites on the Weibull strength field and grows mode-I cracks along
   mesh edges; :func:`find_splitting_state` ramps the frozen field until the
   sheet is actually split (no piece keeps more than 80 % of the area), which is
   the state a DEM needs.
4. **Artefact:** ``failure_faces.json`` -- crack polylines, fracture-element
   polygons, floe polygons and the closed mass/area/boundary ledger.
5. **DEM side:** the file is re-read, the ledger recomputed from the file alone,
   and DEM body initial conditions emitted (``dem_initial_conditions.json``).

Everything the study produces is written under one directory as JSON + PNG, and
every number quoted in the report is read back from those files (no hand
transcription).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time

import numpy as np

from .crack_network import CrackTracerConfig, trace_crack_network, find_splitting_state
from .handoff import (
    build_failure_faces,
    dem_initial_conditions,
    read_failure_faces,
)
from .plate_fem import (
    GRAVITY,
    WATER_DENSITY,
    PlateMaterial,
    PlateMesh,
    PressurePatch,
    clamped_square_deflection,
    cylindrical_bending_moment,
    hetenyi_point_load_deflection,
    navier_ssss_square_deflection,
    navier_ssss_square_moment,
    solve_plate,
)

__all__ = [
    "IcePlateCase",
    "DEFAULT_PATCHES",
    "solve_case_field",
    "run_handoff",
    "fem_benchmark_table",
    "mesh_convergence",
    "sensitivity_sweep",
    "write_figures",
]

FOUNDATION = WATER_DENSITY * GRAVITY

#: Non-uniform contact pressure of a level sheet against a wide structure face:
#: two contact zones with different intensities and sizes (a uniform global
#: pressure would only put the sheet in rigid-body translation on its Winkler
#: foundation and would never bend it).
DEFAULT_PATCHES: tuple[PressurePatch, ...] = (
    PressurePatch(center=(4.60, 4.40), size=(2.00, 2.00), pressure=1.0),
    PressurePatch(center=(6.60, 5.70), size=(1.20, 1.60), pressure=0.65),
)


@dataclass(frozen=True)
class IcePlateCase:
    """The physical case handed to the chain."""

    material: PlateMaterial = PlateMaterial()
    nx: int = 40
    ny: int = 40
    length_x: float = 10.0
    length_y: float = 10.0
    patches: tuple[PressurePatch, ...] = DEFAULT_PATCHES
    foundation_modulus: float = FOUNDATION
    tracer: CrackTracerConfig = CrackTracerConfig()

    def mesh(self) -> PlateMesh:
        return PlateMesh(nx=self.nx, ny=self.ny,
                         length_x=self.length_x, length_y=self.length_y)

    def total_patch_load(self, pressure_scale: float = 1.0) -> float:
        """Total contact force [N] at the given pressure scale."""
        return float(sum(p.pressure * p.size[0] * p.size[1] for p in self.patches)
                     * pressure_scale)

    def as_meta(self, pressure_scale: float, load_factor: float) -> dict:
        return {
            "case": "level ice sheet on a Winkler foundation, non-uniform contact load",
            "load": {
                "patches": [
                    {"center_xy_m": list(p.center), "size_m": list(p.size),
                     "pressure_pa": p.pressure * pressure_scale * load_factor}
                    for p in self.patches
                ],
                "pressure_scale_pa": pressure_scale,
                "load_factor": load_factor,
                "total_contact_force_n": self.total_patch_load(pressure_scale)
                * load_factor,
            },
        }


def solve_case_field(case: IcePlateCase):
    """Solve the plate for a unit pressure scale (``sigma`` per Pa of load)."""
    return solve_plate(
        case.mesh(),
        case.material,
        case.patches,
        foundation_modulus=case.foundation_modulus,
        load_amplitude=1.0,
        boundary="free",
    )


# --------------------------------------------------------------------------
# the production run
# --------------------------------------------------------------------------
def run_handoff(case: IcePlateCase, out_dir: str, *, tag: str = "production") -> dict:
    """Run the whole chain and write the artefacts; return the acceptance numbers."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    unit = solve_case_field(case)
    q_first = unit.scale_to_failure()          # pressure scale at first cracking [Pa]
    first_state = unit.scaled(q_first)         # sigma_max == sigma_flexure  (C1)
    sigma_f = case.material.sigma_flexure

    # First-cracking pattern (reported alongside, not the handoff state).
    first_net = trace_crack_network(
        first_state.scaled(1.0 * first_state.load_amplitude), case.tracer
    )

    split = find_splitting_state(first_state, case.tracer)
    collapse_state = first_state.scaled(split.load_factor * first_state.load_amplitude)
    network = split.network

    # Floe-size distribution: power-law fit from the package's existing estimator.
    diameters = network.diameters()
    fsd = {
        "n_floe": int(len(diameters)),
        "diameters_m": [float(d) for d in sorted(diameters, reverse=True)],
        "mean_diameter_m": float(np.mean(diameters)),
        "median_diameter_m": float(np.median(diameters)),
        "largest_floe_fraction": split.largest_floe_fraction,
        "source": "floe equivalent diameters sqrt(4A/pi)",
    }
    # A power-law exponent needs a few hundred fragments; with O(10) floes the fit
    # is reported as *insufficient* instead of being quoted as a result.
    if len(diameters) >= 12:
        try:
            from tensordem.broken_ice_micro.fsd import fit_power_law

            fit = fit_power_law(diameters, min_tail=8)
            fsd["power_law_fit"] = fit.as_dict()
        except Exception as exc:                # pragma: no cover - reported, not hidden
            fsd["power_law_fit"] = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        fsd["power_law_fit"] = {
            "insufficient_sample": True,
            "reason": (f"{len(diameters)} floes: a power-law exponent needs O(100) "
                       "fragments; the FSD exponent is estimated in "
                       "tensordem.broken_ice_micro instead"),
        }

    meta = case.as_meta(pressure_scale=q_first, load_factor=split.load_factor)
    meta["ramp"] = {"first_cracking_load_factor": 1.0,
                    "splitting_load_factor": split.load_factor,
                    "evaluations": split.evaluations,
                    "history": split.history,
                    "first_cracking_n_floe": first_net.n_floe,
                    "first_cracking_largest_fraction": float(
                        np.max(first_net.areas()) / case.mesh().plate_area)}

    faces = build_failure_faces(network, collapse_state, case_meta=meta, fsd=fsd)
    faces_path = os.path.join(out_dir, "failure_faces.json")
    faces.write(faces_path)

    # --- DEM side: read the file back, rebuild the ledger, make bodies -------
    reloaded = read_failure_faces(faces_path)
    bodies = dem_initial_conditions(reloaded)
    bodies_path = os.path.join(out_dir, "dem_initial_conditions.json")
    with open(bodies_path, "w", encoding="utf-8") as handle:
        json.dump(bodies.as_payload(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    ledger = reloaded.conservation
    plate_area = case.length_x * case.length_y
    plate_mass = plate_area * case.material.thickness * case.material.density

    criteria = {
        "C1_fracture_declared": bool(reloaded.meta["stress_state"]["fracture_declared"]),
        "C1_sigma_max_pa": float(reloaded.meta["stress_state"]["sigma_max_pa"]),
        "C1_sigma_flexure_pa": sigma_f,
        "C1_sigma_over_strength": float(reloaded.meta["stress_state"]["overstress_ratio"]),
        "C1_n_fractured_elements": int(reloaded.meta["stress_state"]["n_fractured_elements"]),
        "C2_plate_mass_kg": plate_mass,
        "C2_floes_mass_kg": ledger.floes_mass_sum,
        "C2_mass_residual_kg": ledger.mass_residual,
        "C2_pass": bool(abs(ledger.mass_residual) < 1e-6),
        "C3_plate_area_m2": plate_area,
        "C3_floes_area_m2": ledger.floes_area_sum,
        "C3_area_residual_m2": ledger.area_residual,
        "C3_pass": bool(abs(ledger.area_residual) < 1e-3),
        "C3_rim_length_from_polygons_m": ledger.rim_length_from_polygons,
        "C3_rim_residual_m": ledger.rim_residual,
        "C3_internal_face_length_m": ledger.internal_face_length,
        "C3_n_shared_segments_over_two": ledger.n_segments_used_three_times,
        "C4_n_floe": int(ledger.n_floe),
        "C4_pass": bool(3 <= ledger.n_floe <= 30),
        "C5_roundtrip_bodies": int(len(bodies)),
        "C5_roundtrip_mass_kg": float(bodies.total_mass),
        "C5_pass": bool(abs(bodies.total_mass - plate_mass) < 1e-6
                        and len(bodies) == ledger.n_floe),
        "first_cracking_pressure_pa": q_first,
        "splitting_pressure_pa": q_first * split.load_factor,
        "max_deflection_m": float(collapse_state.max_deflection),
        "bending_deflection_m": float(
            np.max(collapse_state.w) - np.min(collapse_state.w)),
        "max_rotation_rad": float(max(np.max(np.abs(collapse_state.theta_x)),
                                      np.max(np.abs(collapse_state.theta_y)))),
        "max_curvature_times_thickness": float(
            np.max(np.abs(np.stack([
                np.abs(collapse_state.moment_x),
                np.abs(collapse_state.moment_y),
            ]))) / case.material.flexural_rigidity * case.material.thickness),
        "total_crack_length_m": network.total_crack_length,
        "n_cracks": len(network.cracks),
        "n_seed_sites": len(network.seed_nodes),
        "fsd_alpha": fsd.get("power_law_fit", {}).get("alpha"),
        "wall_time_s": round(time.time() - t0, 2),
    }

    summary = {
        "tag": tag,
        "case": {
            "nx": case.nx, "ny": case.ny,
            "length_x": case.length_x, "length_y": case.length_y,
            "thickness": case.material.thickness,
            "characteristic_length_m": case.material.characteristic_length(
                case.foundation_modulus),
        },
        "criteria": criteria,
        "artifacts": {"failure_faces": faces_path,
                      "dem_initial_conditions": bodies_path},
        "ramp": meta["ramp"],
        "fsd": fsd,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


# --------------------------------------------------------------------------
# FEM verification against closed forms
# --------------------------------------------------------------------------
def fem_benchmark_table() -> dict:
    """Mesh-converged comparison of the FEM against four closed forms."""
    mat = PlateMaterial()
    k = FOUNDATION
    rows: list[dict] = []

    # (a) cylindrical bending of a narrow strip -> beam with EI = E t^3 / 12
    q, span, width = 1000.0, 5.0, 1.0
    ei = mat.young_modulus * mat.thickness ** 3 / 12.0
    w_beam = 5.0 * q * span ** 4 / (384.0 * ei)
    m_beam = cylindrical_bending_moment(q, span)
    mesh = PlateMesh(nx=40, ny=8, length_x=span, length_y=width)
    sol = solve_plate(mesh, mat, [PressurePatch((span / 2, width / 2), (span, width), q)],
                      boundary="simply_supported", constrained_edges=("x0", "x1"))
    rows.append({
        "benchmark": "strip cylindrical bending (beam limit)",
        "quantity": "w_max", "value": sol.max_deflection, "reference": w_beam,
        "rel_error": abs(sol.max_deflection - w_beam) / w_beam,
        "note": "EI = E t^3/12 (plane stress, free longitudinal edges)",
    })
    rows.append({
        "benchmark": "strip cylindrical bending (beam limit)",
        "quantity": "M_max", "value": float(np.max(np.abs(sol.moment_x))),
        "reference": m_beam,
        "rel_error": abs(np.max(np.abs(sol.moment_x)) - m_beam) / m_beam,
        "note": "q a^2 / 8",
    })

    # (b) simply supported square plate, uniform load: Navier double series
    side, q2 = 10.0, 1000.0
    mesh = PlateMesh(nx=40, ny=40, length_x=side, length_y=side)
    sol = solve_plate(mesh, mat, [PressurePatch((side / 2, side / 2), (side, side), q2)],
                      boundary="simply_supported")
    w_ref = navier_ssss_square_deflection(q2, side, mat)
    m_ref = navier_ssss_square_moment(q2, side, mat)
    centre = (mesh.ny // 2) * mesh.nx + mesh.nx // 2
    rows.append({"benchmark": "SSSS square plate (Navier series)", "quantity": "w_max",
                 "value": sol.max_deflection, "reference": w_ref,
                 "rel_error": abs(sol.max_deflection - w_ref) / w_ref,
                 "note": "0.00406 q a^4 / D"})
    rows.append({"benchmark": "SSSS square plate (Navier series)", "quantity": "M_x(centre)",
                 "value": float(abs(sol.moment_x[centre])), "reference": m_ref,
                 "rel_error": abs(abs(sol.moment_x[centre]) - m_ref) / m_ref,
                 "note": "0.0479 q a^2 at nu = 0.3"})

    # (c) clamped square plate
    sol = solve_plate(mesh, mat, [PressurePatch((side / 2, side / 2), (side, side), q2)],
                      boundary="clamped")
    w_ref = clamped_square_deflection(q2, side, mat)
    rows.append({"benchmark": "clamped square plate", "quantity": "w_max",
                 "value": sol.max_deflection, "reference": w_ref,
                 "rel_error": abs(sol.max_deflection - w_ref) / w_ref,
                 "note": "0.00126 q a^4 / D"})

    # (d) Winkler foundation plate under a local load: Hetenyi (the ice benchmark)
    lc = mat.characteristic_length(k)
    domain = 10.0 * lc
    n = 109
    p_load, patch_side = 1.0e5, 0.5
    mesh = PlateMesh(nx=n, ny=n, length_x=domain, length_y=domain)
    sol = solve_plate(mesh, mat,
                      [PressurePatch((domain / 2, domain / 2),
                                     (patch_side, patch_side),
                                     p_load / patch_side ** 2)],
                      foundation_modulus=k, boundary="free")
    w_ref = hetenyi_point_load_deflection(p_load, mat, k)
    rows.append({"benchmark": "Winkler plate, local load (Hetenyi)", "quantity": "w_max",
                 "value": sol.max_deflection, "reference": w_ref,
                 "rel_error": abs(sol.max_deflection - w_ref) / w_ref,
                 "note": f"P l_c^2/(8D), domain {domain/lc:.0f} l_c, h = {domain/n:.2f} m"})

    table = {
        "material": {"E": mat.young_modulus, "nu": mat.poisson_ratio,
                     "t": mat.thickness, "D": mat.flexural_rigidity},
        "characteristic_length_m": lc,
        "rows": rows,
        "max_rel_error": max(r["rel_error"] for r in rows),
        "all_within_3pct": bool(all(r["rel_error"] < 0.03 for r in rows)),
    }
    return table


# --------------------------------------------------------------------------
# convergence and sensitivity
# --------------------------------------------------------------------------
def mesh_convergence(case: IcePlateCase, resolutions=(20, 30, 40, 50), *, max_evals=None) -> dict:
    """Floe census and stress vs mesh refinement (the pattern is mesh-sensitive)."""
    rows = []
    base_pressure = None
    for n in resolutions:
        variant = IcePlateCase(
            material=case.material, nx=n, ny=n,
            length_x=case.length_x, length_y=case.length_y,
            patches=case.patches, foundation_modulus=case.foundation_modulus,
            tracer=case.tracer,
        )
        unit = solve_case_field(variant)
        q_first = unit.scale_to_failure()
        if base_pressure is None:
            base_pressure = q_first
        first = unit.scaled(q_first)
        split = find_splitting_state(first, case.tracer)
        diameters = split.network.diameters()
        rows.append({
            "nx": n, "cell_size_m": case.length_x / n,
            "first_cracking_pressure_pa": q_first,
            "splitting_load_factor": split.load_factor,
            "n_floe": split.network.n_floe,
            "largest_floe_fraction": split.largest_floe_fraction,
            "mean_diameter_m": float(np.mean(diameters)),
            "max_diameter_m": float(np.max(diameters)),
            "total_crack_length_m": split.network.total_crack_length,
            "sigma_max_over_pressure": float(unit.max_sigma),
        })
    return {
        "rows": rows,
        "pressure_spread_rel": float(
            (max(r["first_cracking_pressure_pa"] for r in rows)
             - min(r["first_cracking_pressure_pa"] for r in rows)) / base_pressure),
        "n_floe_range": [min(r["n_floe"] for r in rows), max(r["n_floe"] for r in rows)],
    }


def sensitivity_sweep(
    case: IcePlateCase,
    *,
    separations=(1.0, 1.25, 1.5, 2.0),
    weibull=(3.0, 4.0, 6.0),
    seeds=(20261110, 7, 12345),
) -> dict:
    """One-at-a-time sensitivity of the floe census to the three model knobs."""
    unit = solve_case_field(case)
    q_first = unit.scale_to_failure()
    first = unit.scaled(q_first)

    def run(tracer: CrackTracerConfig) -> dict:
        split = find_splitting_state(first, tracer)
        d = split.network.diameters()
        return {
            "n_floe": split.network.n_floe,
            "largest_floe_fraction": split.largest_floe_fraction,
            "splitting_load_factor": split.load_factor,
            "mean_diameter_m": float(np.mean(d)),
            "median_diameter_m": float(np.median(d)),
            "total_crack_length_m": split.network.total_crack_length,
        }

    def variant(**kwargs) -> CrackTracerConfig:
        base = case.tracer
        return CrackTracerConfig(
            weibull_modulus=kwargs.get("weibull_modulus", base.weibull_modulus),
            min_seed_separation=kwargs.get("min_seed_separation", base.min_seed_separation),
            arrest_fraction=base.arrest_fraction,
            max_steps_per_tip=base.max_steps_per_tip,
            seed=kwargs.get("seed", base.seed),
        )

    return {
        "first_cracking_pressure_pa": q_first,
        "separation": {str(s): run(variant(min_seed_separation=s)) for s in separations},
        "weibull_modulus": {str(m): run(variant(weibull_modulus=m)) for m in weibull},
        "rng_seed": {str(s): run(variant(seed=s)) for s in seeds},
    }


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def write_figures(case: IcePlateCase, out_dir: str, *, summary: dict,
                  convergence: dict | None = None,
                  sensitivity: dict | None = None,
                  benchmarks: dict | None = None) -> list[str]:
    """Render the report figures (PNG) into ``out_dir``; returns their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []

    unit = solve_case_field(case)
    q_first = unit.scale_to_failure()
    first = unit.scaled(q_first)
    split = find_splitting_state(first, case.tracer)
    state = first.scaled(split.load_factor * first.load_amplitude)
    mesh = case.mesh()

    # --- figure 1: stress field + crack network + floe centroids -----------
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=150)
    sigma = state.sigma_max.reshape(mesh.ny, mesh.nx) / 1e6
    im = ax.imshow(sigma, origin="lower", extent=[0, mesh.length_x, 0, mesh.length_y],
                   cmap="inferno", aspect="equal")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(r"flexural stress $\sigma=6|M|/t^2$ [MPa]")
    for crack in split.network.cracks:
        ax.plot(crack.points[:, 0], crack.points[:, 1], "-", color="cyan", lw=1.1)
    for floe in split.network.floes:
        ring = np.vstack([floe.outer, floe.outer[:1]])
        ax.plot(ring[:, 0], ring[:, 1], "-", color="white", lw=0.6, alpha=0.9)
        ax.plot(floe.centroid[0], floe.centroid[1], "o", ms=2.5, color="lime")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"FEM moment field at the splitting load "
        f"(lambda = {split.load_factor:.2f}, {split.network.n_floe} floes)\n"
        f"10 m x 10 m ice sheet, t = {case.material.thickness} m, "
        f"max sigma = {state.max_sigma/1e6:.3f} MPa")
    path = os.path.join(out_dir, "fig1_stress_and_cracks.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    # --- figure 2: floe size distribution + power-law fit ------------------
    diameters = np.sort(split.network.diameters())
    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=150)
    tail = [np.sum(diameters >= d) for d in diameters]
    ax.loglog(diameters, tail, "o-", label="FEM->DEM floe census")
    fit = summary.get("fsd", {}).get("power_law_fit", {})
    if "alpha" in fit and "binned_x" in fit and len(fit["binned_x"]) > 1:
        ax.loglog(fit["binned_x"], fit["binned_y"], "s--",
                  label=(f"log-binned fit: alpha = {fit['alpha']:.2f} "
                         f"+- {fit['alpha_stderr']:.2f}, R2 = {fit['r2']:.2f}"))
    ax.set_xlabel("floe equivalent diameter D = sqrt(4A/pi) [m]")
    ax.set_ylabel("N(D' >= D)")
    ax.set_title(f"Floe size distribution ({split.network.n_floe} floes)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    path = os.path.join(out_dir, "fig2_floe_size_distribution.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    # --- figure 3: load-ramp response --------------------------------------
    history = summary.get("ramp", {}).get("history", [])
    if history:
        lam = np.array([h["load_factor"] for h in history])
        frac = np.array([h["largest_floe_fraction"] for h in history])
        nfl = np.array([h["n_floe"] for h in history])
        order = np.argsort(lam)
        fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
        ax.plot(lam[order], frac[order], "o-", color="C0",
                label="largest floe area fraction")
        ax.axhline(0.8, color="C0", ls=":", lw=0.9)
        ax.set_xlabel("load factor lambda (x first-cracking pressure)")
        ax.set_ylabel("largest floe / plate area")
        ax2 = ax.twinx()
        ax2.plot(lam[order], nfl[order], "s--", color="C3", label="number of floes")
        ax2.set_ylabel("number of floes")
        ax.set_title("Frozen-field load ramp: sheet splits at lambda_c")
        lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
        ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")
        ax.grid(True, alpha=0.3)
        path = os.path.join(out_dir, "fig3_load_ramp.png")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    # --- figure 4: verification against closed forms -----------------------
    if benchmarks:
        labels = [f"{r['benchmark'].split('(')[0].strip()}\n{r['quantity']}"
                  for r in benchmarks["rows"]]
        errs = [100.0 * r["rel_error"] for r in benchmarks["rows"]]
        fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
        bars = ax.bar(range(len(errs)), errs, color="C2")
        ax.axhline(3.0, color="C3", ls="--", lw=1.0, label="3 % acceptance line")
        ax.set_xticks(range(len(errs)))
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylabel("relative error vs closed form [%]")
        ax.set_title("FEM plate solver verification (mesh-converged)")
        for b, e in zip(bars, errs):
            ax.text(b.get_x() + b.get_width() / 2, e, f"{e:.2f}", ha="center",
                    va="bottom", fontsize=7)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        path = os.path.join(out_dir, "fig4_fem_verification.png")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    # --- figure 5: convergence / sensitivity -------------------------------
    if convergence:
        rows = convergence["rows"]
        n = [r["nx"] for r in rows]
        fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=150)
        ax.plot(n, [r["n_floe"] for r in rows], "o-", color="C0", label="number of floes")
        ax.plot(n, [r["largest_floe_fraction"] * 10 for r in rows], "s--", color="C1",
                label="largest floe fraction x10")
        ax.set_xlabel("mesh resolution n (n x n cells on 10 m x 10 m)")
        ax.set_ylabel("count / scaled fraction")
        ax.set_title("Mesh sensitivity of the floe census")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        path = os.path.join(out_dir, "fig5_mesh_sensitivity.png")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths