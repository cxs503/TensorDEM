"""Programmatic evidence for the two engine-closure restorations (m2779).

Restoration A -- platen contact mask
    ``IceDEM3D`` applied the infinite-platen damping term to *every* particle.
    The elastic term vanishes outside contact, so a platen touching nothing
    still reported the fictitious reaction ``n * gamma_p * |v_z - v_p|`` and
    dragged the whole specimen.  The fixed kernel masks both terms with
    ``overlap > 0`` (the convention already used by the spherical tool contact).

Restoration B -- pair dashpot convention in the impact harnesses
    ``measure_dem3d_impact`` (``tensordem.contact_restitution``) and
    ``_build_pair`` (``tensordem.contact_impact``) installed ``contact_damping``
    but left ``DEM3DConfig.contact_damping_model`` at its default
    ``"restitution"``.  In that convention the pair damper is derived from
    ``restitution_coefficient`` (default 1.0), so ``contact_damping`` never
    enters the pair law and every measured impact was perfectly elastic
    (``e = 1``) regardless of ``gamma_n`` -- which silently invalidated the
    restitution calibration built on top of it.

Run with::

    PYTHONPATH=src python scripts/verify_engine_closure_restoration.py

Writes ``artifacts/engine_closure_restoration/{evidence.json,restitution.csv,engine_closure.png}``.
"""
from __future__ import annotations

import csv
import json
import math
import os

import torch

from tensordem.contact_restitution import (
    clamped_restitution_ratio,
    classic_restitution_ratio,
    damping_for_target_restitution,
    measure_dem3d_impact,
)
from tensordem.contact_tangential import restitution_to_damping
from tensordem.dem3d import DEM3DConfig, IceDEM3D

OUT = os.path.join("artifacts", "engine_closure_restoration")
RADIUS = 0.025
DENSITY = 917.0
K_N = 5_000.0
MASS = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3


def platen_mask_evidence() -> dict:
    """Reaction of a platen that touches nothing, vs the pre-fix closed form."""
    cfg = DEM3DConfig(
        nx=8, ny=6, nz=4, radius=RADIUS, density=DENSITY,
        top_platen_enabled=True, bottom_platen_enabled=True,
        platen_damping=15.0, top_platen_gap=0.5, bottom_platen_gap=0.5,
        drag=0.0, fix_x_edges=False, tool_gap=1.0,
    )
    sim = IceDEM3D(cfg)
    sim.velocities[:, 2] = 0.05          # uniform rigid-body drift, zero contact
    sim.forces(update_fracture=False)
    row = sim.diagnostics()
    n = int(sim.positions.shape[0])
    # Pre-fix kernel: force = clamp(k*0 + gamma*(v_z - v_p), 0) for every particle.
    fictitious_top = max(0.0, float(cfg.platen_damping * 0.05)) * n
    fictitious_bottom = max(0.0, float(-cfg.platen_damping * 0.05)) * n
    return {
        "particles": n,
        "platen_damping_Ns_m": cfg.platen_damping,
        "rigid_body_v_z_m_s": 0.05,
        "crossing_particles": 0,
        "between_platen_reaction_z_N": float(row["top_platen_reaction_z_N"]),
        "bottom_platen_reaction_z_N": float(row["bottom_platen_reaction_z_N"]),
        "prefix_fictitious_reaction_top_N": fictitious_top,
        "prefix_fictitious_reaction_bottom_N": fictitious_bottom,
        "masked_reaction_is_zero": abs(float(row["top_platen_reaction_z_N"])) == 0.0
        and abs(float(row["bottom_platen_reaction_z_N"])) == 0.0,
    }


def restitution_evidence() -> dict:
    """Solver restitution vs the clipped-law reference, and the calibration loop."""
    rows = []
    worst = {"gamma": None, "rel": 0.0}
    for gamma in (1.0, 3.0, 5.0, 10.0, 20.0):
        ref = clamped_restitution_ratio(gamma, K_N, MASS)
        run = measure_dem3d_impact(gamma, k_n=K_N, radius=RADIUS, density=DENSITY)
        rel = abs(run["e_measured"] - ref["e_reference"]) / ref["e_reference"]
        classic = classic_restitution_ratio(run["zeta"])
        rows.append({
            "gamma_n_Ns_m": gamma,
            "zeta": run["zeta"],
            "e_solver": run["e_measured"],
            "e_clamped_reference": ref["e_reference"],
            "e_classic_unclamped": classic,
            "rel_error_clamped": rel,
            "rel_error_classic": abs(classic - ref["e_reference"]) / ref["e_reference"],
            "contact_time_s": run["contact_time_s"],
        })
        if rel > worst["rel"]:
            worst = {"gamma": gamma, "rel": rel}

    target = 0.6
    calibrated = damping_for_target_restitution(target, K_N, MASS)
    loop = measure_dem3d_impact(calibrated["gamma_n"], k_n=K_N, radius=RADIUS, density=DENSITY)
    classic_gamma = restitution_to_damping(target, K_N, MASS)
    classic_loop = measure_dem3d_impact(classic_gamma, k_n=K_N, radius=RADIUS, density=DENSITY)
    return {
        "rows": rows,
        "worst_clamped_rel_error": worst["rel"],
        "worst_gamma": worst["gamma"],
        "calibration_target": target,
        "calibration_gamma_n": calibrated["gamma_n"],
        "calibration_e_solver": loop["e_measured"],
        "calibration_rel_error": abs(loop["e_measured"] - target) / target,
        "classic_mapping_gamma_n": classic_gamma,
        "classic_mapping_e_solver": classic_loop["e_measured"],
        "classic_mapping_rel_error": abs(classic_loop["e_measured"] - target) / target,
    }


def elastic_negative_control() -> dict:
    """gamma = 0 must still give e = 1 (the mask/convention must not add damping)."""
    run = measure_dem3d_impact(0.0, k_n=K_N, radius=RADIUS, density=DENSITY)
    return {"e_solver": run["e_measured"], "rel_error_vs_one": abs(run["e_measured"] - 1.0)}


def main() -> dict:
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    payload = {
        "protocol": "tensordem-engine-closure-restoration-v1",
        "platen_mask": platen_mask_evidence(),
        "restitution": restitution_evidence(),
        "elastic_negative_control": elastic_negative_control(),
    }
    with open(os.path.join(OUT, "evidence.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    with open(os.path.join(OUT, "restitution.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(payload["restitution"]["rows"][0]))
        writer.writeheader()
        writer.writerows(payload["restitution"]["rows"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = payload["restitution"]["rows"]
        gammas = [r["gamma_n_Ns_m"] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
        axes[0].plot(gammas, [r["e_solver"] for r in rows], "o-", label="DEM3D solver")
        axes[0].plot(gammas, [r["e_clamped_reference"] for r in rows], "s--",
                     label="clipped-law reference (RK4)")
        axes[0].plot(gammas, [r["e_classic_unclamped"] for r in rows], "^:",
                     label="classic unclamped mapping")
        axes[0].set_xlabel(r"dashpot $\gamma_n$ [N s/m]")
        axes[0].set_ylabel("coefficient of restitution $e$")
        axes[0].set_title("pair impact: solver vs reference")
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.3)

        axes[1].bar([str(g) for g in gammas],
                    [100.0 * r["rel_error_clamped"] for r in rows], width=0.5)
        axes[1].axhline(3.0, color="k", ls="--", lw=1, label="3 % acceptance band")
        axes[1].set_xlabel(r"$\gamma_n$ [N s/m]")
        axes[1].set_ylabel("|e$_{solver}$ - e$_{ref}$| / e$_{ref}$  [%]")
        axes[1].set_title("closure error after restoration")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "engine_closure.png"), dpi=140)
        payload["figure"] = os.path.join(OUT, "engine_closure.png")
    except Exception as exc:  # pragma: no cover - plotting is optional
        payload["figure_error"] = repr(exc)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()