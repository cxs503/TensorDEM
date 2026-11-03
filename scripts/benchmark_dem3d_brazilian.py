"""Diametral (Brazilian) compression benchmark for the 3-D bonded-particle DEM.

The case builds a bonded-sphere cylinder whose axis lies along *y* (engine
geometry ``cylinder_xz``) and closes two opposed infinite platens along *z*, so
the specimen is squeezed across a diameter. That is the classical Brazilian
(indirect tension) configuration: a line load ``P`` on the crown of the disc is
converted into a transverse tensile stress

    sigma_t = 2 P / (pi D t)

with ``D`` the disc diameter and ``t`` its thickness (ASTM D3967; Hondros 1959).
The analytical elastic field at the disc centre is

    sigma_xx(0, 0) = +2 P / (pi D t)      (tension, perpendicular to the load)
    sigma_zz(0, 0) = -6 P / (pi D t)      (compression, along the load)

so the centre principal-stress ratio ``sigma_zz / sigma_xx`` is exactly ``-3``
for a point line load, independently of Young's modulus and Poisson's ratio.
This script measures the lattice counterpart of those three quantities from the
virial (bond-force) stress tensor, together with the platen contact-arc width,
the axial uniformity of the tensile stress and the crack pattern.

Scope: numerical verification of the *apparatus*. The bond model is a
central-force spring network, so agreement with a continuum solution is an
apparatus diagnostic and never a calibrated ice tensile strength.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tensordem.dem3d import DEM3DConfig, IceDEM3D

MANIFEST = ROOT / "benchmarks" / "dem3d" / "brazilian_cases.json"
DEFAULT_CASE = "brazil_elastic_field"


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def specimen_geometry(sim: IceDEM3D, radius: float) -> dict[str, float]:
    """Outer diameter and thickness of the particle assembly (contact envelope)."""
    x = sim.positions[:, 0]
    y = sim.positions[:, 1]
    z = sim.positions[:, 2]
    diameter = float(z.max() - z.min()) + 2.0 * radius
    width = float(x.max() - x.min()) + 2.0 * radius
    thickness = float(y.max() - y.min()) + 2.0 * radius
    if abs(width - diameter) > 4.0 * radius:
        raise ValueError("cylinder_xz geometry must be circular in the x-z plane")
    return {
        "diameter_m": diameter,
        "width_x_m": width,
        "thickness_m": thickness,
        "particle_count": int(len(sim.positions)),
        "bond_count": int(len(sim.pairs)),
    }


def virial_stress(sim: IceDEM3D, radius: float) -> torch.Tensor:
    """Per-particle bond virial stress, tension positive, shape (N, 3, 3).

    sigma_i = (1 / 2 V_i) * sum_j r_ij (x) f_ij with f_ij the force on i from j
    along the bond, so a stretched bond contributes tension. Bond forces are
    central and the current bond vector is used (small-strain convention).
    """
    if len(sim.pairs) == 0:
        return torch.zeros((len(sim.positions), 3, 3), dtype=sim.dtype, device=sim.device)
    i, j = sim.pairs.T
    delta = sim.positions[j] - sim.positions[i]
    distance = torch.linalg.vector_norm(delta, dim=1).clamp_min(torch.finfo(sim.dtype).eps)
    normal = delta / distance[:, None]
    extension = distance - sim.rest_lengths
    magnitude = sim.config.bond_stiffness * extension * sim.alive.to(sim.dtype)
    force = normal * magnitude[:, None]
    volume = 2.0 * (2.0 * radius) ** 3
    stress = torch.zeros((len(sim.positions), 3, 3), dtype=sim.dtype, device=sim.device)
    contribution = delta[:, :, None] * force[:, None, :]
    stress.index_add_(0, i, contribution)
    stress.index_add_(0, j, contribution)
    return stress / volume


def centre_field(sim: IceDEM3D, stress: torch.Tensor, diameter: float) -> dict[str, float]:
    """Mean stress components in the central window |x|,|z| <= D/10."""
    window = 0.1 * diameter
    mask = (sim.positions[:, 0].abs() <= window) & (sim.positions[:, 2].abs() <= window)
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError("no particles inside the centre window")
    return {
        "centre_particle_count": count,
        "centre_sigma_xx_Pa": float(stress[mask, 0, 0].mean()),
        "centre_sigma_yy_Pa": float(stress[mask, 1, 1].mean()),
        "centre_sigma_zz_Pa": float(stress[mask, 2, 2].mean()),
    }


def axial_profile(sim: IceDEM3D, stress: torch.Tensor, diameter: float,
                  bins: int = 8) -> list[dict[str, float]]:
    """Tensile stress sigma_xx along the loaded (z) diameter, |x| <= D/10."""
    window = 0.1 * diameter
    mask = (sim.positions[:, 0].abs() <= window)
    z = sim.positions[mask, 2]
    sxx = stress[mask, 0, 0]
    edges = torch.linspace(-0.5 * diameter, 0.5 * diameter, bins + 1, dtype=sim.dtype)
    profile: list[dict[str, float]] = []
    for k in range(bins):
        in_bin = (z >= edges[k]) & (z < edges[k + 1])
        if int(in_bin.sum().item()) == 0:
            continue
        profile.append({
            "z_over_D": float(0.5 * (edges[k] + edges[k + 1]) / diameter),
            "sigma_xx_Pa": float(sxx[in_bin].mean()),
            "particles": int(in_bin.sum().item()),
        })
    return profile


def contact_arc(sim: IceDEM3D, radius: float, diameter: float, platen_z: float,
                top: bool = True) -> dict[str, float]:
    """Width of the platen contact patch relative to the disc diameter."""
    limit = platen_z - radius if top else platen_z + radius
    touching = ((sim.positions[:, 2] + radius >= platen_z) if top
                else (sim.positions[:, 2] - radius <= platen_z))
    count = int(touching.sum().item())
    if count == 0:
        return {"contact_particles": 0, "contact_arc_m": 0.0, "contact_arc_ratio": 0.0}
    xs = sim.positions[touching, 0]
    width = float(xs.max() - xs.min()) + 2.0 * radius
    del limit, diameter
    return {
        "contact_particles": count,
        "contact_arc_m": width,
        "contact_arc_ratio": width / (float(sim.positions[:, 2].max()
                                            - sim.positions[:, 2].min()) + 2.0 * radius),
    }


def crack_map(sim: IceDEM3D, radius: float) -> dict[str, Any]:
    """Location and failure mode of broken bonds (tensile vs shear)."""
    broken = ~sim.alive
    count = int(broken.sum().item())
    if count == 0:
        return {"broken_bonds": 0, "tensile_fraction": 0.0,
                "mean_abs_x_over_R": 0.0, "crack_band_fraction": 0.0, "midpoints": []}
    i, j = sim.bond_indices.T
    mid = 0.5 * (sim.positions[i] + sim.positions[j])
    mid_broken = mid[broken]
    radius_disc = 0.5 * (float(sim.positions[:, 2].max() - sim.positions[:, 2].min()) + 2.0 * radius)
    abs_x = mid_broken[:, 0].abs()
    modes = sim.failure_mode[broken]
    return {
        "broken_bonds": count,
        "tensile_fraction": float((modes == 1).sum().item()) / count,
        "shear_fraction": float((modes == 2).sum().item()) / count,
        "mean_abs_x_over_R": float((abs_x / radius_disc).mean()),
        "crack_band_fraction": float((abs_x <= 0.25 * radius_disc).to(torch.float64).mean()),
        "first_failure_time_s": float(sim.failure_time_s[broken].min()),
        "midpoints": [[float(v) for v in row] for row in mid_broken.tolist()],
    }


# ---------------------------------------------------------------------------
# Case driver
# ---------------------------------------------------------------------------

def _build(case: dict[str, Any]) -> tuple[IceDEM3D, dict[str, float]]:
    p = case["config"]
    speed = float(p["loading_speed_m_s"])
    base = DEM3DConfig(
        nx=int(p["nx"]), ny=int(p["ny"]), nz=int(p["nz"]),
        geometry=str(p.get("geometry", "cylinder_xz")),
        radius=float(p["radius_m"]),
        density=float(p["density_kg_m3"]),
        bond_stiffness=float(p["bond_stiffness_N_m"]),
        contact_stiffness=float(p["contact_stiffness_N_m"]),
        breaking_strain=float(p["breaking_strain"]),
        platen_stiffness=float(p["platen_stiffness_N_m"]),
        platen_damping=float(p["platen_damping_Ns_m"]),
        top_platen_enabled=True,
        bottom_platen_enabled=True,
        top_platen_velocity=-speed,
        bottom_platen_velocity=speed,
        drag=float(p["drag_Ns_m"]),
        contact_damping=float(p["contact_damping_Ns_m"]),
        fix_x_edges=False,
        fix_bottom=False,
    )
    factor = float(p.get("dt_factor", 0.5))
    if not math.isfinite(factor) or not 0.0 < factor <= 1.0:
        raise ValueError("dt_factor must be finite and in (0, 1]")
    sim = IceDEM3D(DEM3DConfig(**{**base.__dict__, "dt": base.recommended_dt * factor}))
    # Isolate platen loading: park the spherical tool far outside the specimen.
    sim.tool_start[:] = torch.tensor([1.0e6, 1.0e6, 1.0e6], dtype=sim.dtype)
    sim.tool_velocity.zero_()
    geometry = specimen_geometry(sim, base.radius)
    geometry["dt_s"] = float(sim.dt)
    geometry["recommended_dt_s"] = float(base.recommended_dt)
    return sim, geometry


def _run_once(case: dict[str, Any], steps: int) -> tuple[list[dict[str, float]],
                                                         str, dict[str, Any]]:
    sim, geometry = _build(case)
    p = case["config"]
    radius = float(p["radius_m"])
    speed = float(p["loading_speed_m_s"])
    probe_every = max(int(p.get("probe_every", 100)), 1)
    diameter = geometry["diameter_m"]
    thickness = geometry["thickness_m"]
    digest = hashlib.sha256()
    rows: list[dict[str, float]] = []
    peak = {"load_N": 0.0, "stress_Pa": 0.0, "travel_m": 0.0}

    def sample(step: int) -> None:
        diag = sim.diagnostics()
        stress = virial_stress(sim, radius)
        centre = centre_field(sim, stress, diameter)
        top_z = sim.top_platen_start_z + sim.time * sim.config.top_platen_velocity
        arc = contact_arc(sim, radius, diameter, top_z)
        # Reactions are reported along +z for both platens, so the bottom reaction
        # carries the opposite sign; the compressive load is their half difference.
        top_load = float(diag["top_platen_reaction_z_N"])
        bottom_load = -float(diag["bottom_platen_reaction_z_N"])
        load = 0.5 * (top_load + bottom_load)
        split = 2.0 * load / (math.pi * diameter * thickness)
        row = {
            "step": step,
            "time_s": float(sim.time),
            "platen_travel_m": speed * float(sim.time),
            "mean_platen_load_N": load,
            "top_platen_load_N": top_load,
            "bottom_platen_load_N": bottom_load,
            "split_stress_Pa": split,
            "analytic_centre_sigma_xx_Pa": split,
            "broken_bonds": int(diag["broken_bonds"]),
            "kinetic_energy_J": float(diag["kinetic_energy_J"]),
            "platen_energy_J": float(diag["platen_contact_energy_J"]),
            "bond_energy_J": float(diag["bond_elastic_energy_J"]),
            "contact_arc_ratio": arc["contact_arc_ratio"],
            "contact_particles": arc["contact_particles"],
        }
        row.update(centre)
        ratio = row["centre_sigma_zz_Pa"] / row["centre_sigma_xx_Pa"] if row["centre_sigma_xx_Pa"] else float("nan")
        row["centre_principal_ratio"] = ratio
        row["centre_tension_rel_error"] = (
            (row["centre_sigma_xx_Pa"] - split) / split if split else float("nan")
        )
        derived = ("centre_principal_ratio", "centre_tension_rel_error")
        if not all(math.isfinite(float(v)) for key, v in row.items() if key not in derived):
            raise FloatingPointError(f"non-finite history value at step {step}")
        if load > peak["load_N"]:
            peak.update(load_N=load, stress_Pa=split, travel_m=speed * float(sim.time))
        rows.append(row)
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    sample(0)
    for step in range(1, steps + 1):
        sim.step()
        if step % probe_every == 0 or step == steps:
            sample(step)

    stress = virial_stress(sim, radius)
    profile = axial_profile(sim, stress, diameter)
    cracks = crack_map(sim, radius)
    prof_values = [entry["sigma_xx_Pa"] for entry in profile]
    uniformity = 0.0
    if prof_values:
        mean = sum(prof_values) / len(prof_values)
        spread = math.sqrt(sum((v - mean) ** 2 for v in prof_values) / len(prof_values))
        uniformity = spread / abs(mean) if mean else float("inf")
    top_z = sim.top_platen_start_z + sim.time * sim.config.top_platen_velocity
    arc = contact_arc(sim, radius, diameter, top_z)
    last = rows[-1]
    metrics: dict[str, Any] = {
        **geometry,
        "case_id": case["id"],
        "steps": steps,
        "specimen_diameter_m": diameter,
        "specimen_thickness_m": thickness,
        "peak_load_N": peak["load_N"],
        "peak_split_stress_Pa": peak["stress_Pa"],
        "platen_travel_at_peak_m": peak["travel_m"],
        "final_broken_bonds": int(cracks["broken_bonds"]),
        "final_contact_arc_ratio": arc["contact_arc_ratio"],
        "final_contact_particles": arc["contact_particles"],
        "final_centre_sigma_xx_Pa": last["centre_sigma_xx_Pa"],
        "final_centre_sigma_zz_Pa": last["centre_sigma_zz_Pa"],
        "final_centre_principal_ratio": last.get("centre_principal_ratio"),
        "final_centre_tension_rel_error": last["centre_tension_rel_error"],
        "axial_uniformity_rel": uniformity,
        "axial_profile": profile,
        "crack_map": {k: v for k, v in cracks.items() if k != "midpoints"},
        "crack_midpoints": cracks["midpoints"],
        "signature_sha256": digest.hexdigest(),
    }
    return rows, digest.hexdigest(), metrics


def run_brazilian_benchmark(output: Path, *, steps: int | None = None,
                            case_id: str = DEFAULT_CASE) -> dict[str, Any]:
    """Run the named Brazilian case twice and persist JSON/CSV artifacts."""
    torch.set_num_threads(1)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "tensordem-dem3d-brazilian-v1":
        raise ValueError("unsupported Brazilian benchmark protocol")
    case = next((item for item in manifest["cases"] if item["id"] == case_id), None)
    if case is None:
        raise ValueError(f"manifest has no case {case_id!r}")
    nsteps = int(case["steps"] if steps is None else steps)
    if nsteps < 1:
        raise ValueError("steps must be positive")
    rows, signature, metrics = _run_once(case, nsteps)
    repeat_rows, repeat_signature, repeat_metrics = _run_once(case, nsteps)
    acceptance = manifest["acceptance"]

    # Compare loads on a scale set by the peak load: a relative test on values that
    # are numerically zero only measures round-off, not a physical asymmetry.
    load_floor = 1e-9 * max(metrics["peak_load_N"], 1e-30)

    def rel(a: float, b: float) -> float:
        scale = max(abs(a), abs(b), load_floor)
        return abs(a - b) / scale

    max_asymmetry = max(
        rel(float(row["top_platen_load_N"]), float(row["bottom_platen_load_N"])) for row in rows
    )
    formula_residual = max(
        rel(float(row["split_stress_Pa"]),
            2.0 * float(row["mean_platen_load_N"])
            / (math.pi * metrics["specimen_diameter_m"] * metrics["specimen_thickness_m"]))
        for row in rows
    )
    # Quasi-static index is only meaningful once the specimen stores energy; rows
    # within round-off of zero are excluded through an energy floor.
    stored = [float(row["platen_energy_J"]) + float(row["bond_energy_J"]) for row in rows]
    energy_floor = 1e-6 * max(max(stored), 1e-30)
    energy_ratio = max(
        (float(row["kinetic_energy_J"]) / max(store, 1e-30), store)
        for row, store in zip(rows, stored) if store >= energy_floor
    )[0] if any(store >= energy_floor for store in stored) else 0.0
    checks = {
        "reaction_symmetry": max_asymmetry <= float(acceptance["reaction_symmetry_rel_tol"]),
        "split_stress_definition_consistent": formula_residual <= float(acceptance["split_stress_rel_tol"]),
        "finite_history": all(math.isfinite(float(v)) for row in rows
                              for k, v in row.items()
                              if k not in ("centre_principal_ratio", "centre_tension_rel_error")),
        "repeatable_history": (signature == repeat_signature
                              and metrics["peak_load_N"] == repeat_metrics["peak_load_N"]),
        "quasi_static": energy_ratio <= float(acceptance["quasi_static_energy_ratio_max"]),
        "specimen_populated": metrics["particle_count"] >= int(acceptance["minimum_particles"]),
        "positive_load": metrics["peak_load_N"] > float(acceptance["minimum_peak_load_N"]),
    }
    ratio = metrics.get("final_centre_principal_ratio")
    physics = {
        "centre_principal_ratio": ratio,
        "centre_principal_ratio_target": float(manifest["reference"]["centre_principal_stress_ratio"]),
        "centre_principal_ratio_rel_error": (
            abs(ratio - float(manifest["reference"]["centre_principal_stress_ratio"]))
            / abs(float(manifest["reference"]["centre_principal_stress_ratio"]))
            if ratio is not None and math.isfinite(ratio) else None
        ),
        "centre_tension_rel_error": (
            metrics["final_centre_tension_rel_error"]
            if metrics["final_centre_tension_rel_error"] is not None
            and math.isfinite(metrics["final_centre_tension_rel_error"]) else None
        ),
        "contact_arc_ratio": metrics["final_contact_arc_ratio"],
        "axial_uniformity_rel": metrics["axial_uniformity_rel"],
    }
    physics_checks = {
        "centre_principal_ratio_within_tol": (
            physics["centre_principal_ratio_rel_error"] is not None
            and physics["centre_principal_ratio_rel_error"] <= float(acceptance["centre_principal_ratio_tol"])
        ),
        "centre_tension_within_tol": (
            physics["centre_tension_rel_error"] is not None
            and abs(physics["centre_tension_rel_error"]) <= float(acceptance["centre_tension_rel_tol"])
        ),
        "diametral_contact_arc": (
            physics["contact_arc_ratio"] <= float(acceptance["contact_arc_ratio_max"])
        ),
        "axial_uniformity_within_tol": (
            physics["axial_uniformity_rel"] <= float(acceptance["axial_uniformity_rel_tol"])
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    report: dict[str, Any] = {
        "protocol": manifest["protocol"],
        "case_id": case["id"],
        "purpose": case["purpose"],
        "scope": manifest["scope"],
        "reference": manifest["reference"],
        "verdict": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "physics_checks": physics_checks,
        "physics_diagnostics": physics,
        "quasi_static_energy_ratio_max": energy_ratio,
        "reaction_asymmetry_max": max_asymmetry,
        "split_stress_formula_residual_max": formula_residual,
        **metrics,
        "repeat_signature_sha256": repeat_signature,
        "rows": rows,
        "interpretation_limits": manifest["interpretation_limits"],
        "interpretation": (
            "PASS certifies the protocol invariants (reaction symmetry, split-stress definition, "
            "repeatability, quasi-static energy ratio). The physics_diagnostics block reports how far "
            "the central-force lattice centre field is from the Hondros solution; it is an apparatus "
            "diagnostic, not an ice tensile-strength calibration."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{case['id']}_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / f"{case['id']}_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-brazilian"))
    args = parser.parse_args(argv)
    report = run_brazilian_benchmark(args.output, steps=args.steps, case_id=args.case)
    print(json.dumps({
        "case_id": report["case_id"],
        "verdict": report["verdict"],
        "checks": report["checks"],
        "physics_checks": report["physics_checks"],
        "physics_diagnostics": report["physics_diagnostics"],
        "steps": report["steps"],
        "peak_split_stress_Pa": report["peak_split_stress_Pa"],
        "final_broken_bonds": report["final_broken_bonds"],
        "signature_sha256": report["signature_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()