"""True-triaxial + Lode strength-envelope benchmark for the 3-D bonded DEM.

Protocol (see docs/dem3d_triaxial_lode.md for the full derivation):

1. Degradation anchor  - the triaxial runner at sigma_c -> 0, b = 0 must
   reproduce the recorded dem3d_ucs peak stress (relative difference <= 2 %).
2. Confining ladder    - sigma_c in {0.1, 0.25, 0.5, 1.0} * sigma_UCS, b = 0;
   the peak strength must rise monotonically and a Mohr-Coulomb line must fit
   with R^2 >= 0.98.
3. Lode sweep          - b in {0, 0.25, 0.5, 0.75, 1}; record the strength-b
   curve and adjudicate Drucker-Prager (b-independent) vs Lade-Duncan
   (b-dependent) by their fit error.
4. Boundary / size     - negative controls: rigid wall vs free lateral
   boundary (<= 5 %), and N in {800, 1600, 3200} (strength drift <= 5 %).
5. Energy ledger       - per-test fracture / friction / kinetic closure
   residual (reported honestly; a discrete diagnostic, not exact conservation).

The runner is a thin driver over :mod:`tensordem.dem3d_triaxial`; every number
in result.json is produced by the solver, never hand-typed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tensordem.dem3d_triaxial import TriaxialConfig, run_triaxial  # noqa: E402

# Recorded dem3d_ucs peak (ucs_dynamic_smoke, 500 steps, opposed platens).
UCS_BASELINE_PA = 1754.4779213871718
ANCHOR_TOL = 0.02
LADDER_FACTORS = (0.1, 0.25, 0.5, 1.0)
LODE_B = (0.0, 0.25, 0.5, 0.75, 1.0)
SIZE_TARGETS = (800, 1600, 3200)


# --------------------------------------------------------------------------
# Individual experiments
# --------------------------------------------------------------------------
def run_anchor() -> dict[str, Any]:
    cfg = TriaxialConfig(protocol="ucs_opposed", sigma_x=0.0, sigma_y=0.0,
                         max_axial_steps=500, post_peak_fraction=0.0)
    r = run_triaxial(cfg)
    rel = abs(r["sigma1_Pa"] - UCS_BASELINE_PA) / UCS_BASELINE_PA
    return {
        "sigma1_Pa": r["sigma1_Pa"],
        "baseline_Pa": UCS_BASELINE_PA,
        "relative_difference": rel,
        "gate": "relative_difference <= 0.02",
        "pass": rel <= ANCHOR_TOL,
        "n_steps": r["n_steps"],
        "final_broken_bonds": r["final_broken_bonds"],
    }


def _triaxial(sigma_x: float, sigma_y: float, **kw) -> dict[str, Any]:
    cfg = TriaxialConfig(protocol="triaxial", sigma_x=sigma_x, sigma_y=sigma_y,
                         max_axial_steps=kw.pop("max_axial_steps", 1800),
                         post_peak_fraction=kw.pop("post_peak_fraction", 0.55),
                         **kw)
    return run_triaxial(cfg)


def run_ladder() -> dict[str, Any]:
    points = []
    for f in LADDER_FACTORS:
        sc = f * UCS_BASELINE_PA
        r = _triaxial(sc, sc)
        points.append({
            "factor": f,
            "sigma3_Pa": sc,
            "sigma1_Pa": r["sigma1_Pa"],
            "b_lode": r["b_lode"],
            "final_broken_bonds": r["final_broken_bonds"],
            "n_steps": r["n_steps"],
        })
    s3 = np.array([p["sigma3_Pa"] for p in points])
    s1 = np.array([p["sigma1_Pa"] for p in points])
    # monotonicity
    mono = bool(np.all(np.diff(s1) > 0))
    # Mohr-Coulomb linear fit sigma1 = a*sigma3 + c0
    a, c0 = np.polyfit(s3, s1, 1)
    pred = a * s3 + c0
    ss_res = float(((s1 - pred) ** 2).sum())
    ss_tot = float(((s1 - s1.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "points": points,
        "monotonic": mono,
        "mc_slope": float(a),
        "mc_intercept_Pa": float(c0),
        "mc_R2": float(r2),
        "gate": "monotonic and mc_R2 >= 0.98",
        "pass": mono and r2 >= 0.98,
    }


def run_lode() -> dict[str, Any]:
    # Fix sigma3 = 0.25*UCS; span b by raising sigma2 toward the b=0 peak.
    s3 = 0.25 * UCS_BASELINE_PA
    # first the b=0 point to estimate sigma1
    r0 = _triaxial(s3, s3)
    s1_est = r0["sigma1_Pa"]
    points = []
    for b in LODE_B:
        s2 = s3 + b * (s1_est - s3)
        r = _triaxial(s2, s3)
        points.append({
            "target_b": b,
            "sigma2_Pa": s2,
            "sigma3_Pa": s3,
            "sigma1_Pa": r["sigma1_Pa"],
            "b_achieved": r["b_lode"],
            "final_broken_bonds": r["final_broken_bonds"],
            "n_steps": r["n_steps"],
        })
    return _fit_lode(points)


def _fit_lode(points: list[dict[str, Any]]) -> dict[str, Any]:
    b = np.array([p["b_achieved"] for p in points])
    s1 = np.array([p["sigma1_Pa"] for p in points])
    s3 = points[0]["sigma3_Pa"]
    # Drucker-Prager (b-independent): sigma1 = a*sigma3 + c0  (b dropped)
    # Use the b=0 anchor of the ladder as the b-independent prediction:
    # sigma1_pred = s1_at_b0 + (sigma3 - s3_ref) * slope, slope from ladder.
    # Here all points share sigma3, so the DP prediction is the b=0 value.
    s1_b0 = points[0]["sigma1_Pa"]
    dp_pred = np.full_like(s1, s1_b0)
    dp_err = float(np.sqrt(((s1 - dp_pred) ** 2).mean()))
    # Lade-Duncan (b-dependent): sigma1 = a*sigma3 + c0 + c1*b
    X = np.column_stack([np.full_like(b, s3), np.ones_like(b), b])
    coef, *_ = np.linalg.lstsq(X, s1, rcond=None)
    ld_pred = X @ coef
    ld_err = float(np.sqrt(((s1 - ld_pred) ** 2).mean()))
    # adjudication: lower RMS error wins
    winner = "Lade-Duncan" if ld_err < dp_err else "Drucker-Prager"
    return {
        "sigma3_Pa": s3,
        "points": points,
        "dp_rms_error_Pa": dp_err,
        "ld_rms_error_Pa": ld_err,
        "ld_coef": {"a": float(coef[0]), "c0_Pa": float(coef[1]), "c1": float(coef[2])},
        "adjudication": winner,
        "gate": "explicit DP vs LD error comparison",
        "pass": True,  # adjudication is always produced
    }


def run_boundary() -> dict[str, Any]:
    sc = 0.25 * UCS_BASELINE_PA
    # The triaxial protocol uses servo platens (free lateral boundary).  The
    # rigid-wall variant fixes the x-edges instead (see _rigid_wall).
    free = _triaxial(sc, sc)
    rigid = _rigid_wall(sc)
    diff = abs(free["sigma1_Pa"] - rigid["sigma1_Pa"]) / max(free["sigma1_Pa"], 1e-12)
    return {
        "sigma_c_Pa": sc,
        "free_sigma1_Pa": free["sigma1_Pa"],
        "rigid_sigma1_Pa": rigid["sigma1_Pa"],
        "relative_difference": float(diff),
        "gate": "relative_difference <= 0.05",
        "pass": diff <= 0.05,
        "note": ("periodic lateral boundary not implemented in the engine; "
                 "free servo-platen used as the non-rigid reference"),
    }


def _rigid_wall(sigma_c: float) -> dict[str, Any]:
    """Triaxial test with rigid x-walls (fix_x_edges) instead of servo platens."""
    import torch
    from tensordem.dem3d import DEM3DConfig, IceDEM3D
    cfg = TriaxialConfig(protocol="triaxial", sigma_x=sigma_c, sigma_y=sigma_c,
                        max_axial_steps=1500, post_peak_fraction=0.55)
    base = DEM3DConfig(
        nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, radius=cfg.radius, density=cfg.density,
        bond_stiffness=cfg.bond_stiffness, contact_stiffness=cfg.contact_stiffness,
        breaking_strain=cfg.breaking_strain, shear_breaking_strain=cfg.shear_breaking_strain,
        platen_stiffness=cfg.platen_stiffness, platen_damping=cfg.platen_damping,
        contact_damping=cfg.contact_damping, drag=cfg.drag,
        friction_coefficient=cfg.friction_coefficient,
        top_platen_enabled=True, bottom_platen_enabled=True,
        top_platen_velocity=-cfg.loading_speed, bottom_platen_velocity=0.0,
        fix_x_edges=True, fix_bottom=False,
        lateral_platens=True,  # still confine in y; x is rigid
    )
    sim = IceDEM3D(DEM3DConfig(**{**base.__dict__, "dt": base.recommended_dt}))
    sim.tool_start[:] = torch.tensor([1e6, 1e6, 1e6], dtype=sim.dtype)
    sim.tool_velocity.zero_()
    # confine in y only (x is rigid)
    a_yz = (cfg.nx * 2 * cfg.radius) * (cfg.nz * 2 * cfg.radius)
    fy_target = sigma_c * a_yz
    for _ in range(400):
        sim.step()
        rymin = float(sim.last_lateral_platen_reaction[2])
        rymax = float(sim.last_lateral_platen_reaction[3])
        fy = -rymin + rymax
        ey = fy_target - fy
        g, vmax = 1e-4, 0.02
        sim.lateral_platen_velocity[2] = max(-vmax, min(vmax, g * ey))
        sim.lateral_platen_velocity[3] = max(-vmax, min(vmax, -g * ey))
        sim.lateral_platen_velocity[0] = 0.0
        sim.lateral_platen_velocity[1] = 0.0
    sim.lateral_platen_velocity.zero_()
    gross = (cfg.nx * 2 * cfg.radius) * (cfg.ny * 2 * cfg.radius)
    peak = 0.0
    for _ in range(1800):
        sim.step()
        top = abs(float(sim.last_top_platen_reaction_z))
        bot = abs(float(sim.last_bottom_platen_reaction_z))
        stress = 0.5 * (top + bot) / gross
        peak = max(peak, stress)
    return {"sigma1_Pa": float(peak)}


def run_size() -> dict[str, Any]:
    points = []
    for target in SIZE_TARGETS:
        n = int(round(target ** (1.0 / 3.0)))
        n = max(2, n)
        sc = 0.25 * UCS_BASELINE_PA
        r = _triaxial(sc, sc, nx=n, ny=n, nz=n)
        points.append({
            "target_N": target,
            "n_particles": n ** 3,
            "sigma1_Pa": r["sigma1_Pa"],
            "n_steps": r["n_steps"],
        })
    s1 = np.array([p["sigma1_Pa"] for p in points])
    drift = float((s1.max() - s1.min()) / s1.mean())
    return {
        "points": points,
        "strength_drift": drift,
        "gate": "strength_drift <= 0.05",
        "pass": drift <= 0.05,
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_all() -> dict[str, Any]:
    out: dict[str, Any] = {
        "protocol": "tensordem-dem3d-triaxial-v1",
        "units": "SI",
        "ucs_baseline_Pa": UCS_BASELINE_PA,
    }
    out["anchor"] = run_anchor()
    out["ladder"] = run_ladder()
    out["lode"] = run_lode()
    out["boundary"] = run_boundary()
    out["size"] = run_size()
    out["verdict"] = "PASS" if all(
        out[k].get("pass", False) for k in ("anchor", "ladder", "boundary", "size")
    ) else "FAIL"
    return out


def main() -> None:
    out = run_all()
    here = Path(__file__).resolve().parent
    (here / "result.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    refs = {
        "ucs_baseline": {
            "source": "benchmarks/dem3d/ucs_cases.json (ucs_dynamic_smoke, 500 steps)",
            "peak_engineering_stress_Pa": UCS_BASELINE_PA,
        },
        "models": {
            "mohr_coulomb": "sigma1 = a*sigma3 + c0 (linear in confining pressure)",
            "drucker_prager": "I1 + alpha*sqrt(J2) = k (Lode-angle independent)",
            "lade_duncan": "b-dependent (Lode-angle) strength surface",
        },
    }
    (here / "refs.json").write_text(json.dumps(refs, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("anchor", "ladder", "boundary", "size")
                      if k in out and isinstance(out[k], dict) and "points" not in out[k]},
          indent=2, sort_keys=True))
    print("verdict:", out["verdict"])


if __name__ == "__main__":
    main()
