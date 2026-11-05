#!/usr/bin/env python3
"""Re-verify the DEM3D contact-impact harness after the dashpot-convention drift.

The merged validation harness ``tensordem.contact_impact.pair_impact`` integrates
the clipped (clamped) law

    F_n = max(0, k_n * delta - c_n * v_n)                                (1)

and compares the measured restitution and contact duration with the closed form
for that same law.  When the engine gained ``DEM3DConfig.contact_damping_model``
(whose default ``"restitution"`` derives the pair dashpot from
``restitution_coefficient`` and ignores ``contact_damping``), the harness kept
injecting ``c_n`` through ``contact_damping`` without requesting the absolute
convention -- so the simulated pair was undamped (``e -> 1``) while the reference
was damped, and five of the harness' own tests went red on trunk.

This script produces the evidence for the repair:

* ``cases``        -- per-target restitution / contact-duration closure against
                      the clamped closed form (the acceptance band is 3 %);
* ``convention``   -- negative control: the *same* engine state probed under both
                      conventions, proving the flag is load-bearing;
* ``convergence``  -- time-step refinement, to separate physics from
                      discretisation in the contact-duration residual.

Outputs (``artifacts/dem3d_contact_impact_convention/``):
``verification.json``, ``cases.csv``, ``convergence.csv``, ``fig1_restitution.png``,
``fig2_duration_convergence.png``.

Usage
-----
    PYTHONPATH=src python scripts/verify_dem3d_contact_impact_convention.py
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from dataclasses import replace  # noqa: E402

from tensordem.contact_impact import (  # noqa: E402
    _build_pair,
    impact_analytic,
    pair_impact,
)
from tensordem.contact_tangential import restitution_to_damping  # noqa: E402

OUT = pathlib.Path("artifacts/dem3d_contact_impact_convention")
BAND = 0.03  # 3 % engineering acceptance band (repo standard)
RADIUS, DENSITY, K_N, V0 = 0.05, 900.0, 5.0e4, 0.5
E_TARGETS = (0.9, 0.7, 0.5, 0.3)
DT_SCALES = (1.0, 0.5, 0.25)


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cases() -> list[dict]:
    rows = []
    for e_target in E_TARGETS:
        r = pair_impact(e_target=e_target)
        rows.append(
            {
                "e_target": e_target,
                "c_n": reast_damping(e_target),
                "e_measured": r.e_measured,
                "e_clamped_analytic": r.e_clamped,
                "e_unclamped_analytic": r.e_unclamped,
                "e_err_vs_clamped_pct": 100.0 * r.e_error_vs_clamped,
                "e_err_vs_unclamped_pct": 100.0 * r.e_error_vs_unclamped,
                "force_time_analytic_s": r.contact_time_force_analytic,
                "force_time_measured_s": r.contact_time_force,
                "force_time_err_pct": 100.0 * r.contact_time_force_error,
                "geom_time_analytic_s": r.contact_time_analytic,
                "geom_time_measured_s": r.contact_time,
                "geom_time_err_pct": 100.0 * r.contact_time_error,
                "force_law_residual": r.force_law_residual,
                "momentum_error": r.momentum_error,
                "steps_per_contact": r.steps_per_contact,
                "dt_s": r.dt,
            }
        )
    return rows


def reast_damping(e_target: float) -> float:
    """Damping the harness installs for ``e_target`` (absolute convention)."""
    from tensordem.contact_impact import restitution_to_damping_clamped

    m = DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3
    return restitution_to_damping_clamped(e_target, K_N, 0.5 * m)


def _pair_with_convention(convention: str, c_n: float):
    """The harness pair, but with the pair dashpot convention overridden.

    ``DEM3DConfig`` is frozen, so the variant is built from a
    ``dataclasses.replace`` copy and then re-run through the same bond-breaking
    step the harness performs, leaving geometry, stiffness and velocities
    identical and only the convention different.
    """
    demo, i, j = _build_pair(K_N, c_n, RADIUS, DENSITY, 1.0)
    if demo.config.contact_damping_model == convention:
        return demo, i, j
    dem = demo.__class__(replace(demo.config, contact_damping_model=convention))
    positions = demo.initial_positions.clone()
    dem.positions = positions.clone()
    dem.initial_positions = positions.clone()
    dem.fixed = demo.fixed.clone()
    dem.forces()
    if int(dem.alive.sum().item()) != 0:
        raise RuntimeError("bonded lattice did not fully break in convention control")
    return dem, i, j


def convention_control() -> dict:
    """Probe one engine state under both dashpot conventions.

    Geometry, stiffness, installed dashpot and velocities are identical between
    the two engine instances, so any force difference comes from the convention
    alone.  ``contact_damping_model="absolute"`` must reproduce Eq. (1);
    ``"restitution"`` (the engine default) must reproduce the *undamped* law for
    ``restitution_coefficient = 1`` -- that is what silently detached the harness.
    """
    m = DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3
    m_eff = 0.5 * m
    e_target = 0.5
    c_n = reast_damping(e_target)

    pairs = {}
    for convention in ("absolute", "restitution"):
        dem, i, j = _pair_with_convention(convention, c_n)
        dem.velocities = torch.zeros_like(dem.positions)
        dem.velocities[i] = torch.tensor([0.0, 0.0, V0], dtype=dem.dtype)
        dem.velocities[j] = torch.tensor([0.0, 0.0, -V0], dtype=dem.dtype)
        pairs[convention] = (dem, i, j)

    vn = -2.0 * V0  # closing: both particles approach at V0
    rows = []
    for delta in (1.0e-4, 5.0e-4, 2.0e-3, 5.0e-3):
        reference = max(0.0, K_N * delta - c_n * vn)
        measured = {}
        for convention, (dem, i, j) in pairs.items():
            probe = dem.positions.clone()
            probe[i] = torch.tensor([0.0, 0.0, -(RADIUS - delta / 2)], dtype=dem.dtype)
            probe[j] = torch.tensor([0.0, 0.0, RADIUS - delta / 2], dtype=dem.dtype)
            saved = dem.positions.clone()
            dem.positions = probe
            f = dem.forces(update_fracture=False)[0]
            dem.positions = saved
            measured[convention] = abs(float(f[i, 2]))
        rows.append(
            {
                "overlap_m": delta,
                "closing_speed_m_s": abs(vn),
                "analytic_F_n_N": reference,
                "engine_F_n_absolute_N": measured["absolute"],
                "engine_F_n_restitution_N": measured["restitution"],
                "abs_rel_err_pct": 100.0 * (measured["absolute"] - reference)
                / max(reference, 1e-30),
                "restitution_rel_err_pct": 100.0 * (measured["restitution"] - reference)
                / max(reference, 1e-30),
            }
        )
    return {
        "e_target": e_target,
        "c_n_installed": c_n,
        "m_eff_kg": m_eff,
        "probe_rows": rows,
        "absolute_matches_law": all(abs(r["abs_rel_err_pct"]) < 1e-9 for r in rows),
        "restitution_ignores_dashpot": all(
            abs(r["restitution_rel_err_pct"]) > 1.0 for r in rows
        ),
    }


def convergence() -> list[dict]:
    rows = []
    for dt_scale in DT_SCALES:
        r = pair_impact(e_target=0.5, dt_scale=dt_scale)
        rows.append(
            {
                "dt_scale": dt_scale,
                "dt_s": r.dt,
                "steps_per_contact": r.steps_per_contact,
                "e_err_vs_clamped_pct": 100.0 * r.e_error_vs_clamped,
                "force_time_err_pct": 100.0 * r.contact_time_force_error,
                "geom_time_err_pct": 100.0 * r.contact_time_error,
            }
        )
    return rows


def observed_order(rows: list[dict], key: str) -> float:
    """Log-log slope of |error| against dt over consecutive refinements."""
    pts = [(r["dt_s"], abs(r[key])) for r in rows if r[key] != 0.0]
    if len(pts) < 2:
        return float("nan")
    slopes = [
        math.log(pts[i + 1][1] / pts[i][1]) / math.log(pts[i + 1][0] / pts[i][0])
        for i in range(len(pts) - 1)
    ]
    return sum(slopes) / len(slopes)


def force_law_figure(control: dict) -> None:
    rows = control["probe_rows"]
    deltas = [r["overlap_m"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=160)
    ax.plot(deltas, [r["analytic_F_n_N"] for r in rows], "k-", lw=1.2,
            label="clipped law $\\max(0, k\\delta - c v_n)$")
    ax.plot(deltas, [r["engine_F_n_absolute_N"] for r in rows], "o", ms=7,
            mfc="none", mew=1.5, color="#d62728",
            label="engine, $contact\\_damping\\_model$='absolute'")
    ax.plot(deltas, [r["engine_F_n_restitution_N"] for r in rows], "s", ms=5,
            color="#1f77b4", label="engine, default 'restitution' ($e=1$)")
    ax.set_xscale("log")
    ax.set_xlabel("overlap $\\delta$ [m]")
    ax.set_ylabel("normal force $F_n$ [N]")
    ax.set_title("Same engine state, two dashpot conventions ($c_n=55.4$ N s/m)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_force_law.png")
    plt.close(fig)


def render_tables(summary: dict) -> str:
    """Markdown tables rendered straight from the verification records."""
    case_rows = summary["cases"]
    conv_rows = summary["convergence"]
    control = summary["convention_control"]
    order = summary["observed_orders"]
    lines = []
    lines.append("### T1  restitution / contact-duration closure (absolute dashpot)")
    lines.append("")
    lines.append("| $e_{target}$ | $c_n$ [N s/m] | $e$ solver | $e$ clamped | err [%] | "
                 "$t^{force}$ solver [ms] | $t^{force}$ clamped [ms] | err [%] | $t^{geom}$ err [%] |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in case_rows:
        lines.append(
            f"| {r['e_target']:.2f} | {r['c_n']:.4f} | {r['e_measured']:.5f} | "
            f"{r['e_clamped_analytic']:.5f} | {r['e_err_vs_clamped_pct']:+.3f} | "
            f"{r['force_time_measured_s']*1e3:.4f} | {r['force_time_analytic_s']*1e3:.4f} | "
            f"{r['force_time_err_pct']:+.3f} | {r['geom_time_err_pct']:+.3f} |"
        )
    lines.append("")
    lines.append("### T2  convention negative control (identical state, two conventions)")
    lines.append("")
    lines.append("| $\delta$ [m] | clipped law $F_n$ [N] | engine 'absolute' [N] | err [%] | "
                 "engine 'restitution' [N] | err [%] |")
    lines.append("|---|---|---|---|---|---|")
    for r in control["probe_rows"]:
        lines.append(
            f"| {r['overlap_m']:.1e} | {r['analytic_F_n_N']:.4f} | "
            f"{r['engine_F_n_absolute_N']:.4f} | {r['abs_rel_err_pct']:+.2e} | "
            f"{r['engine_F_n_restitution_N']:.4f} | {r['restitution_rel_err_pct']:+.2f} |"
        )
    lines.append("")
    lines.append("### T3  time-step refinement at $e_{target}=0.5$")
    lines.append("")
    lines.append("| $\Delta t$ [s] | steps/contact | $e$ err [%] | $t^{force}$ err [%] | "
                 "$t^{geom}$ err [%] |")
    lines.append("|---|---|---|---|---|")
    for r in conv_rows:
        lines.append(
            f"| {r['dt_s']:.3e} | {r['steps_per_contact']:.0f} | "
            f"{r['e_err_vs_clamped_pct']:+.3f} | {r['force_time_err_pct']:+.3f} | "
            f"{r['geom_time_err_pct']:+.3f} |"
        )
    lines.append("")
    lines.append(
        "Observed log-log orders over the refinement (dt to dt/4): "
        f"restitution {order['e_err_vs_clamped_pct']:.2f}, "
        f"force duration {order['force_time_err_pct']:.2f}, "
        f"geometric duration {order['geom_time_err_pct']:.2f} "
        "(first order = semi-implicit Euler; the duration residual is discretisation, "
        "not a force-law defect)."
    )
    lines.append("")
    lines.append("### T4  unclamped-mapping bias reproduced in the solver")
    lines.append("")
    lines.append("| $e_{target}$ | 0.90 | 0.70 | 0.50 | 0.30 |")
    lines.append("|---|---|---|---|---|")
    bias = summary["verdicts"]["unclamped_bias_pct_by_target"]
    lines.append("| bias [%] | " + " | ".join(
        f"{bias[f'{e:.2f}']:.2f}" for e in (0.9, 0.7, 0.5, 0.3)) + " |")
    lines.append("")
    return "\n".join(lines)


def figures(case_rows: list[dict], conv_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=160)
    e = [r["e_target"] for r in case_rows]
    ax.plot(e, [r["e_clamped_analytic"] for r in case_rows], "k-", lw=1.2,
            label="clamped closed form (reference)")
    ax.plot(e, [r["e_unclamped_analytic"] for r in case_rows], "b--", lw=1.0,
            label="unclamped closed form (legacy mapping)")
    ax.plot(e, [r["e_measured"] for r in case_rows], "o", ms=6, mfc="none", mew=1.4,
            color="#d62728", label="DEM3D solver (absolute dashpot)")
    ax.set_xlabel("target restitution $e_{target}$")
    ax.set_ylabel("realised restitution $e$")
    ax.set_title("DEM3D pair impact vs clamped closed form")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_restitution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=160)
    dt = [r["dt_s"] for r in conv_rows]
    for key, style, label in (
        ("force_time_err_pct", "o-", "force-contact duration"),
        ("geom_time_err_pct", "s-", "geometric contact duration"),
        ("e_err_vs_clamped_pct", "^-", "restitution"),
    ):
        ax.loglog(dt, [abs(r[key]) for r in conv_rows], style, ms=5, label=label)
    ax.set_xlabel("time step $\\Delta t$ [s]")
    ax.set_ylabel("|relative error| [%]")
    ax.set_title("Mesh refinement of the impact harness ($e_{target}=0.5$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_duration_convergence.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    case_rows = cases()
    control = convention_control()
    conv_rows = convergence()
    figures(case_rows, conv_rows)
    force_law_figure(control)

    orders = {
        key: observed_order(conv_rows, key)
        for key in ("force_time_err_pct", "geom_time_err_pct", "e_err_vs_clamped_pct")
    }
    summary = {
        "protocol": "tensordem-dem3d-contact-impact-convention-v1",
        "acceptance_band_pct": 100.0 * BAND,
        "cases": case_rows,
        "convention_control": control,
        "convergence": conv_rows,
        "observed_orders": orders,
        "verdicts": {
            "restitution_max_abs_pct": max(abs(r["e_err_vs_clamped_pct"]) for r in case_rows),
            "restitution_within_band": all(
                abs(r["e_err_vs_clamped_pct"]) < 100.0 * BAND for r in case_rows
            ),
            "force_duration_max_abs_pct": max(
                abs(r["force_time_err_pct"]) for r in case_rows
            ),
            "force_duration_within_band": all(
                abs(r["force_time_err_pct"]) < 100.0 * BAND for r in case_rows
            ),
            "force_law_residual_max": max(r["force_law_residual"] for r in case_rows),
            "momentum_error_max": max(abs(r["momentum_error"]) for r in case_rows),
            "unclamped_bias_pct_by_target": {
                f"{r['e_target']:.2f}": abs(r["e_err_vs_unclamped_pct"]) for r in case_rows
            },
            "unclamped_bias_exceeds_band_below_e07": all(
                abs(r["e_err_vs_unclamped_pct"]) > 100.0 * BAND
                for r in case_rows
                if r["e_target"] <= 0.7
            ),
            "convention_flag_load_bearing": control["restitution_ignores_dashpot"],
        },
    }
    _write_csv(OUT / "cases.csv", case_rows)
    _write_csv(OUT / "convergence.csv", conv_rows)
    (OUT / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "tables.md").write_text(render_tables(summary) + "\n")

    print("== cases ==")
    for r in case_rows:
        print(
            f"e_target={r['e_target']:.2f}  c_n={r['c_n']:.4f}  "
            f"e={r['e_measured']:.5f} (clamped {r['e_clamped_analytic']:.5f}, "
            f"{r['e_err_vs_clamped_pct']:+.3f} %)  "
            f"t_force {r['force_time_measured_s']*1e3:.4f} vs "
            f"{r['force_time_analytic_s']*1e3:.4f} ms ({r['force_time_err_pct']:+.3f} %)"
        )
    print("== convention control ==")
    for r in control["probe_rows"]:
        print(
            f"delta={r['overlap_m']:.1e} m  F(abs)={r['engine_F_n_absolute_N']:.4f} N  "
            f"F(restitution)={r['engine_F_n_restitution_N']:.4f} N  "
            f"analytic={r['analytic_F_n_N']:.4f} N"
        )
    print("== convergence ==")
    for r in conv_rows:
        print(
            f"dt={r['dt_s']:.3e} ({r['steps_per_contact']:.1f} steps/contact)  "
            f"e {r['e_err_vs_clamped_pct']:+.3f} %  "
            f"t_force {r['force_time_err_pct']:+.3f} %  "
            f"t_geom {r['geom_time_err_pct']:+.3f} %"
        )
    print("== observed orders ==", {k: round(v, 3) for k, v in orders.items()})
    print("== tables ->", OUT / "tables.md", "==")
    print("== verdicts ==")
    for k, v in summary["verdicts"].items():
        print(f"  {k}: {v}")
    return 0 if all(
        [
            summary["verdicts"]["restitution_within_band"],
            summary["verdicts"]["force_duration_within_band"],
            summary["verdicts"]["convention_flag_load_bearing"],
        ]
    ) else 1


if __name__ == "__main__":
    sys.exit(main())