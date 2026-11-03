#!/usr/bin/env python3
"""Audit the restitution -> normal-damping mapping used by the 3-D DEM solver.

Question
--------
``DEM3DConfig.restitution_coefficient`` is a user-facing target, but the solver
integrates a *compressive-only* normal law ``F_n = max(k_n*overlap - c_n*v_n, 0)``.
The historical mapping (``restitution_mapping="classic"``) inverts the free
spring-dashpot identity ``e = exp(-pi*zeta/sqrt(1-zeta^2))``, which is exact only
when the force is allowed to become adhesive. The truncation removes the
retarding force late in the contact, so the realised rebound is systematically
*higher* than requested.

This script measures the realised coefficient of restitution with the shipped
solver (:func:`tensordem.contact_restitution.measure_dem3d_impact`) for both
mappings, over a target grid and two time steps, and writes machine-readable
artefacts. Nothing here is transcribed by hand.

Run:  PYTHONPATH=src python3 scripts/audit_dem3d_restitution_mapping.py
Out:  artifacts/dem3d_restitution_mapping/{results.json,results.csv,figure.png}
"""
from __future__ import annotations

import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordem.contact_impact import clamped_restitution, impact_analytic, restitution_to_damping_clamped
from tensordem.contact_restitution import (
    clamped_restitution_ratio,
    classic_restitution_ratio,
    measure_dem3d_impact,
)
from tensordem.dem3d import DEM3DConfig

OUT = pathlib.Path("artifacts/dem3d_restitution_mapping")
E_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
K_N = 5_000.0
RADIUS = 0.025
DENSITY = 917.0


def run_case(e_target: float, mapping: str, dt_scale: float = 1.0,
             both_free: bool = True) -> dict:
    """One closed-loop measurement; equal-mass pair matches the mass/2 convention."""
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY,
        contact_stiffness=K_N, contact_damping=0.0,
        restitution_coefficient=e_target, restitution_mapping=mapping,
        drag=0.0, fix_x_edges=False,
    )
    measured = measure_dem3d_impact(
        restitution_coefficient=e_target,
        restitution_mapping=mapping,
        k_n=K_N, radius=RADIUS, density=DENSITY,
        drag=0.0, dt_scale=dt_scale, both_free=both_free,
    )
    m_eff = cfg.mass / 2.0
    gamma = cfg.pair_normal_damping
    zeta = gamma / (2.0 * (K_N * m_eff) ** 0.5)
    analytic = impact_analytic(K_N, gamma, m_eff)
    return {
        "e_target": e_target,
        "mapping": mapping,
        "dt_scale": dt_scale,
        "both_free": both_free,
        "gamma_n": gamma,
        "zeta": zeta,
        "e_analytic_clamped": analytic.e_clamped,
        "e_analytic_unclamped": analytic.e_unclamped,
        "e_measured": measured["e_measured"],
        "rel_error_vs_target": measured["e_measured"] / e_target - 1.0,
        "abs_rel_error_vs_target": abs(measured["e_measured"] / e_target - 1.0),
        "within_3pct": abs(measured["e_measured"] / e_target - 1.0) < 0.03,
        "contact_time_s": measured["contact_time_s"],
        "dt_s": measured["dt_scale"] * cfg.recommended_dt,
        "steps": measured["steps"],
        "recommended_dt_s": cfg.recommended_dt,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for mapping in ("classic", "clamped"):
        for e in E_GRID:
            rows.append(run_case(e, mapping))
    # Time-step refinement of the corrected mapping (the realised value must not
    # be an artefact of the default step size).
    for scale in (1.0, 0.5, 0.25):
        for e in (0.3, 0.6):
            rows.append(run_case(e, "clamped", dt_scale=scale))
    # Convention check: with one sphere frozen (an infinite-mass wall) the solver
    # still derives damping for m_eff = mass/2, so the realised rebound is higher.
    fixed_wall = [run_case(e, "clamped", both_free=False) for e in (0.3, 0.6)]

    # Analytic sanity: the closed-form clamped restitution used by the inverse
    # mapping must agree with the independent RK4 integration of the same law.
    analytic_cross = []
    for e in E_GRID:
        gamma = restitution_to_damping_clamped(e, K_N, DEM3DConfig(
            nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY).mass / 2.0)
        m_eff = DEM3DConfig(nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY).mass / 2.0
        zeta = gamma / (2.0 * (K_N * m_eff) ** 0.5)
        closed = clamped_restitution(zeta) if zeta < 0.70710678 else None
        rk4 = clamped_restitution_ratio(gamma, K_N, m_eff, steps=4_000)["e_reference"]
        analytic_cross.append({
            "e_target": e, "zeta": zeta,
            "e_clamped_closed_form": closed,
            "e_clamped_rk4": rk4,
            "closed_form_applicable": closed is not None,
        })

    summary = {
        "grid": list(E_GRID),
        "k_n_N_per_m": K_N,
        "radius_m": RADIUS,
        "density_kg_per_m3": DENSITY,
        "criterion_relative_error": 0.03,
        "clamped_max_abs_rel_error": max(r["abs_rel_error_vs_target"] for r in rows
                                         if r["mapping"] == "clamped" and r["dt_scale"] == 1.0),
        "classic_max_abs_rel_error": max(r["abs_rel_error_vs_target"] for r in rows
                                         if r["mapping"] == "classic"),
        "classic_rel_error_at_e0.3": next(r["rel_error_vs_target"] for r in rows
                                          if r["mapping"] == "classic" and r["e_target"] == 0.3),
        "clamped_rel_error_at_e0.3": next(r["rel_error_vs_target"] for r in rows
                                          if r["mapping"] == "clamped" and r["e_target"] == 0.3
                                          and r["dt_scale"] == 1.0),
        "clamped_validity_band": [0.2, 1.0],
        "clamped_worst_in_band_abs_rel_error": max(
            r["abs_rel_error_vs_target"] for r in rows
            if r["mapping"] == "clamped" and r["e_target"] >= 0.2),
        "clamped_all_in_band_within_3pct": all(
            r["within_3pct"] for r in rows
            if r["mapping"] == "clamped" and r["e_target"] >= 0.2),
        "clamped_all_within_3pct": all(r["within_3pct"] for r in rows if r["mapping"] == "clamped"),
        "classic_all_within_3pct": all(r["within_3pct"] for r in rows if r["mapping"] == "classic"),
    }

    payload = {"summary": summary, "cases": rows, "fixed_wall_cases": fixed_wall,
               "analytic_cross_check": analytic_cross}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))

    with (OUT / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- figure: realised e vs target, and relative error on a log axis -------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for mapping, style in (("classic", "o--"), ("clamped", "s-")):
        sel = [r for r in rows if r["mapping"] == mapping and r["dt_scale"] == 1.0]
        ax1.plot([r["e_target"] for r in sel], [r["e_measured"] for r in sel],
                 style, label=f"solver, {mapping} mapping")
        ax2.semilogy([r["e_target"] for r in sel],
                     [max(r["abs_rel_error_vs_target"], 1e-6) for r in sel],
                     style, label=f"{mapping}")
    ax1.plot(E_GRID, E_GRID, "k:", lw=1, label="target e (identity)")
    ax1.set_xlabel("requested restitution e")
    ax1.set_ylabel("realised restitution")
    ax1.set_title("DEM3D pair impact: realised vs requested")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.axhline(0.03, color="r", ls="-.", lw=1, label="3 % criterion")
    ax2.set_xlabel("requested restitution e")
    ax2.set_ylabel("|realised/requested - 1|")
    ax2.set_title("Mapping error (both free, default dt)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "figure.png", dpi=150)

    print(json.dumps(summary, indent=2))
    print(f"artefacts -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())