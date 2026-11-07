"""Reproduce every number in ``docs/nonspherical_contact_timestep_layer.md``.

Usage::

    python scripts/verify_nonspherical_contact_and_timestep.py [--quick]

Writes ``artifacts/nonspherical-contact-timestep/`` (JSON ledgers + markdown
tables) and ``docs/nonspherical-contact-timestep/*.png``.  ``--quick`` shrinks the
energy-ledger step count (for smoke runs); the paper numbers use the defaults.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensordem.convex_contact import (  # noqa: E402
    Box, Clump, Cone, Polytope, Sphere, analytic_box_box_sat, analytic_clump_box_gap,
    analytic_clump_clump_gap, analytic_polytope_sat, analytic_sphere_box,
    analytic_sphere_sphere, clump_contact, contact,
)
from tensordem.contact_timestep_ledger import (  # noqa: E402
    cundall_critical_timestep, gas_energy_ledger, hertz_contact_time_weber_stillinger,
    hookean_contact_time, hunt_crossley_semi_analytic_restitution,
    measure_critical_timestep, measure_linear_contact_time, measure_restitution_sweep,
)

ARTIFACT_DIR = ROOT / "artifacts" / "nonspherical-contact-timestep"
FIGURE_DIR = ROOT / "docs" / "nonspherical-contact-timestep"
REL_TOL = 1.0e-10
ANGLE_TOL = 1.0e-8
ROT45 = np.array([[math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0],
                  [math.sin(math.pi / 4), math.cos(math.pi / 4), 0.0],
                  [0.0, 0.0, 1.0]])


def angle_between(a, b) -> float:
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    return float(np.linalg.norm(np.cross(a, b)))


# ---------------------------------------------------------------------------
# 1. contact geometry: eight analytic families
# ---------------------------------------------------------------------------
def contact_geometry_campaign() -> list[dict]:
    rows = []

    def add(family, solver, oracle_gap, oracle_normal, extra=None):
        if oracle_gap >= 0.0:
            kind = "separated"
            value = solver.distance
            rel = abs(solver.distance - oracle_gap) / max(oracle_gap, 1e-300)
        else:
            kind = "penetrating"
            value = solver.depth
            rel = abs(solver.depth + oracle_gap) / max(-oracle_gap, 1e-300)
        rows.append({"family": family, "configuration": kind, "value": float(value),
                     "oracle": float(abs(oracle_gap)), "relative_error": float(rel),
                     "normal_angle_rad": angle_between(solver.normal, oracle_normal),
                     "gjk_iterations": solver.gjk_iterations,
                     "epa_iterations": solver.epa_iterations,
                     "epa_faces": solver.epa_faces, **(extra or {})})

    a = Sphere([0.0, 0.0, 0.0], 1.0)
    for centre in ([2.7, 0.3, 0.2], [1.5, 0.2, 0.1]):
        b = Sphere(centre, 0.8)
        gap, normal, _, _ = analytic_sphere_sphere(a.center, a.radius, b.center, b.radius)
        add("sphere-sphere", contact(a, b), gap, normal)

    box = Box([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
    for centre, radius in (([0.95, 0.85, 0.9], 0.25), ([0.62, 0.55, 0.58], 0.25)):
        sphere = Sphere(centre, radius)
        gap, normal, _, _ = analytic_sphere_box(sphere.center, sphere.radius,
                                                box.center, box.half_extents)
        add("sphere-box", contact(sphere, box), gap, normal)

    b1 = Box([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
    for centre in ([1.23, 0.1, 0.05], [0.87, 0.1, 0.05]):
        b2 = Box(centre, [0.5, 0.5, 0.5])
        gap, normal, _, _ = analytic_box_box_sat(b1.center, b1.half_extents, None,
                                                 b2.center, b2.half_extents, None)
        add("box-box-aligned", contact(b1, b2), gap, normal)

    for centre in ([1.35, 0.0, 0.05], [1.10, 0.0, 0.05]):
        b2 = Box(centre, [0.5, 0.5, 0.5], ROT45)
        gap, normal, _, _ = analytic_box_box_sat(b1.center, b1.half_extents, None,
                                                 b2.center, b2.half_extents, ROT45)
        add("box-box-rotated-45deg", contact(b1, b2), gap, normal)

    tetra = Polytope([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    for centre in ([1.25, 0.15, 0.15], [0.85, 0.15, 0.15]):
        b5 = Box(centre, [0.2, 0.2, 0.2])
        gap, normal = analytic_polytope_sat(tetra.vertices, b5.vertices())
        add("tetra-box", contact(tetra, b5), gap, normal)

    floor = Box([0.0, 0.0, -0.5], [2.0, 2.0, 0.5])
    for apex_z in (0.06, -0.04):
        cone = Cone([0.1, 0.05, apex_z], [0.0, 0.0, 1.0], 0.3, 0.12)
        gap, normal, _, _ = analytic_sphere_box(cone.apex, 0.0, floor.center,
                                                floor.half_extents)
        add("cone-floor", contact(cone, floor), gap, normal)

    clump = Clump([Sphere([0.0, 0.0, 0.0], 0.2), Sphere([0.35, 0.0, 0.0], 0.2),
                   Sphere([0.17, 0.3, 0.0], 0.2)])
    for centre_z in (-0.45, -0.22):
        b7 = Box([0.17, 0.1, centre_z], [0.5, 0.5, 0.2])
        gap, index = analytic_clump_box_gap(clump, b7.center, b7.half_extents,
                                            b7.orientation)
        member = clump.spheres[index]
        _, normal, _, _ = analytic_sphere_box(member.center, member.radius, b7.center,
                                              b7.half_extents, b7.orientation)
        add("clump-box", clump_contact(clump, b7)[0], gap, normal,
            {"extremal_member": index})

    for offset_y in (0.75, 0.35):
        other = Clump([Sphere([0.0, offset_y, 0.0], 0.2),
                       Sphere([0.35, offset_y, 0.0], 0.2),
                       Sphere([0.17, offset_y, 0.5], 0.2)])
        gap, pair, normal = analytic_clump_clump_gap(clump, other)
        add("clump-clump", clump_contact(clump, other)[0], gap, normal,
            {"extremal_pair": list(pair)})

    return rows


# ---------------------------------------------------------------------------
# 2. time-integration layer
# ---------------------------------------------------------------------------
def contact_time_campaign() -> dict:
    linear = [measure_linear_contact_time(0.5, 1.0, k) for k in (500.0, 2000.0, 8000.0)]
    hertz = [measure_linear_contact_time(v, 1.0, kh, law="hertz")
             for v, kh in ((0.10, 4.0e4), (0.20, 9.0e4))]
    cross_check = [{"k_hertz": kh, "closed_form": hertz_contact_time_weber_stillinger(v, 1.0, kh),
                    "measured": r["measured_tc"], "relative_error": r["relative_error"]}
                   for (v, kh), r in zip(((0.10, 4.0e4), (0.20, 9.0e4)), hertz)]
    return {"linear": linear, "hertz": hertz, "hertz_cross_check": cross_check,
            "hookean_closed_form_note": "t_c = pi sqrt(m/k) = pi sqrt(m/(2 k_n)) in the "
                                        "per-body stiffness convention k = 2 k_n"}


def restitution_campaign() -> dict:
    v_imp = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
    sweep = measure_restitution_sweep(v_imp, 1.0, 1.0e4, 0.002)
    return sweep


def timestep_campaign() -> dict:
    measured = measure_critical_timestep(1.0, 2000.0)
    # stability map: growth factor versus dt/dt_cr
    ratios = np.array([0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5])
    from tensordem.contact_timestep_ledger import SpringOscillator
    growth = []
    for ratio in ratios:
        probe = SpringOscillator(mass=1.0, stiffness=2000.0, displacement=0.05)
        growth.append(probe.run(ratio * measured["theory_dt_cr"], 400))
    return {"bisection": measured, "stability_map": {"dt_over_cr": ratios.tolist(),
                                                     "growth_factor": [float(g) for g in growth]}}


def ledger_campaign(quick: bool) -> dict:
    steps = 20_000 if quick else 100_000
    undamped = gas_energy_ledger(count=64, steps=steps, check_every=max(1, steps // 20),
                                 safety=0.01)
    damped = gas_energy_ledger(count=64, steps=steps, check_every=max(1, steps // 20),
                               safety=1.0e-4, damping=5.0)
    return {"undamped": undamped, "damped": damped, "steps": steps}


def negative_controls(quick: bool) -> dict:
    """Three fail-closed controls: the harness must flag each of them."""
    steps = 4_000 if quick else 20_000
    dt_cr = cundall_critical_timestep(1.0, 2.0e4)
    # (1) dt = 1.5 dt_cr must diverge
    from tensordem.contact_timestep_ledger import SpringOscillator
    over = SpringOscillator(mass=1.0, stiffness=2.0e4, displacement=0.05)
    growth_over = over.run(1.5 * dt_cr, 400)
    safe = SpringOscillator(mass=1.0, stiffness=2.0e4, displacement=0.05)
    growth_safe = safe.run(0.5 * dt_cr, 400)
    # (2) negative damping must add energy
    gas = gas_energy_ledger(count=64, steps=steps, safety=0.01, damping=-2.0,
                            check_every=max(1, steps // 4))
    # (3) torn ledger: dropping the dissipative term from the accounting must be
    #     detected (this is the residual of a deliberately wrong ledger)
    gas_ok = gas_energy_ledger(count=64, steps=steps, safety=0.01, damping=2.0,
                               check_every=max(1, steps // 4))
    torn = abs(gas_ok["energy_drift_relative"])
    return {"timestep_over_dt_cr": {"dt_over_cr": 1.5, "growth_factor": growth_over,
                                    "diverged": bool(growth_over > 4.0),
                                    "reference_dt_over_cr": 0.5,
                                    "reference_growth": growth_safe},
            "negative_damping": {"damping": -2.0,
                                 "energy_gain_relative": gas["energy_drift_relative"],
                                 "energy_increased": bool(gas["energy_drift_relative"] > 0.0)},
            "torn_ledger": {"damping": 2.0, "energy_drift_relative": gas_ok["energy_drift_relative"],
                            "ledger_residual_relative": gas_ok["ledger_residual_relative"],
                            "unaccounted_relative": torn,
                            "detected": bool(torn > 1.0e-4)}}


# ---------------------------------------------------------------------------
# 3. figures
# ---------------------------------------------------------------------------
def make_figures(payload: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    rows = payload["contact_geometry"]["rows"]
    families = sorted({r["family"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    width = 0.38
    for axis, key, title, tol in ((axes[0], "relative_error", "relative error on gap / depth", REL_TOL),
                                  (axes[1], "normal_angle_rad", "normal angle error [rad]", ANGLE_TOL)):
        for i, family in enumerate(families):
            for j, configuration in enumerate(("separated", "penetrating")):
                values = [r[key] for r in rows
                          if r["family"] == family and r["configuration"] == configuration]
                value = max(values) if values else np.nan
                axis.bar(i + (j - 0.5) * width, max(value, 1e-18), width,
                         label=configuration if i == 0 else None,
                         color=("#1f77b4", "#d62728")[j])
        axis.set_yscale("log")
        axis.axhline(tol, color="k", ls="--", lw=1.0, label=f"criterion {tol:g}")
        axis.set_xticks(range(len(families)))
        axis.set_xticklabels(families, rotation=25, ha="right", fontsize=8)
        axis.set_title(title, fontsize=10)
        axis.grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path = FIGURE_DIR / "contact_geometry_accuracy.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    sweep = payload["restitution"]
    v = [p["v_imp"] for p in sweep["points"]]
    e = [p["e"] for p in sweep["points"]]
    semi = [p["e_semi_analytic"] for p in sweep["points"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    axes[0].plot(v, e, "o-", label="measured (velocity Verlet, H-C dashpot)")
    axes[0].plot(v, semi, "s--", label="semi-analytic first-order balance")
    axes[0].set_xlabel("impact speed v_imp [m/s]")
    axes[0].set_ylabel("coefficient of restitution e")
    axes[0].set_title("Hertz + Hunt-Crossley restitution", fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    loss = np.maximum(1.0 - np.array(e), 1e-16)
    axes[1].loglog(v, loss, "o-", label="measured 1 - e")
    fit_v = np.geomspace(min(v), max(v), 50)
    axes[1].loglog(fit_v, sweep["fit_amplitude"] * fit_v ** sweep["fit_exponent"], "k--",
                   label=f"fit p = {sweep['fit_exponent']:.3f} (R^2 = {sweep['fit_r2']:.5f})")
    axes[1].set_xlabel("impact speed v_imp [m/s]")
    axes[1].set_ylabel("1 - e")
    axes[1].set_title("power-law signature of the Hertz contact", fontsize=10)
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = FIGURE_DIR / "restitution_vs_impact_speed.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    stability = payload["timestep"]["stability_map"]
    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.semilogy(stability["dt_over_cr"], stability["growth_factor"], "o-")
    axis.axvline(1.0, color="k", ls="--", lw=1.0, label="Cundall-Strack dt_cr")
    axis.set_xlabel("dt / dt_cr")
    axis.set_ylabel("amplitude growth factor (400 steps)")
    axis.set_title("velocity-Verlet stability boundary", fontsize=10)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURE_DIR / "timestep_stability_map.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for key, label, colour in (("undamped", "no damping", "#1f77b4"),
                               ("damped", "dashpot c = 5 N s/m", "#d62728")):
        series = payload["ledger"][key]["series"]
        axes[0].plot([s["step"] for s in series], [s["energy"] for s in series], "o-",
                     color=colour, label=label)
        axes[1].semilogy([s["step"] for s in series],
                         [max(abs(s["residual"]), 1e-16) for s in series], "o-",
                         color=colour, label=label)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("mechanical energy [J]")
    axes[0].set_title("64-sphere gas: mechanical energy", fontsize=10)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("|energy ledger residual| / E0")
    axes[1].axhline(1e-6, color="k", ls=":", lw=1.0, label="criterion 1e-6")
    axes[1].set_title("ledger closure", fontsize=10)
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURE_DIR / "energy_ledger.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def _strip_series(payload: dict) -> dict:
    """JSON-safe copy with the long time series removed."""
    out = json.loads(json.dumps(payload, default=float))
    for key in ("undamped", "damped"):
        out["ledger"][key].pop("series", None)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    t_start = time.time()
    payload: dict = {"environment": {"python": platform.python_version(),
                                     "numpy": np.__version__,
                                     "platform": platform.platform()}}

    print("[1/6] contact geometry", flush=True)
    payload["contact_geometry"] = {"rows": contact_geometry_campaign(),
                                   "rel_tol": REL_TOL, "angle_tol": ANGLE_TOL}
    print("[2/6] contact time", flush=True)
    payload["contact_time"] = contact_time_campaign()
    print("[3/6] restitution", flush=True)
    payload["restitution"] = restitution_campaign()
    print("[4/6] timestep", flush=True)
    payload["timestep"] = timestep_campaign()
    print("[5/6] energy ledger", flush=True)
    payload["ledger"] = ledger_campaign(args.quick)
    print("[6/6] negative controls", flush=True)
    payload["negative_controls"] = negative_controls(args.quick)

    rows = payload["contact_geometry"]["rows"]
    worst_rel = max(r["relative_error"] for r in rows)
    worst_angle = max(r["normal_angle_rad"] for r in rows)
    linear = payload["contact_time"]["linear"]
    hertz = payload["contact_time"]["hertz"]
    sweep = payload["restitution"]
    bisection = payload["timestep"]["bisection"]
    undamped = payload["ledger"]["undamped"]
    damped = payload["ledger"]["damped"]
    controls = payload["negative_controls"]

    criteria = {
        "C1_contact_geometry_8_families_rel_error_le_1e-10": worst_rel <= REL_TOL,
        "C1_worst_relative_error": worst_rel,
        "C1_worst_normal_angle_rad": worst_angle,
        "C1_normal_angle_le_1e-8": worst_angle <= ANGLE_TOL,
        "C2_contact_time_le_2pct_three_stiffness": max(abs(r["relative_error"]) for r in linear) <= 0.02,
        "C2_worst_contact_time_error": max(abs(r["relative_error"]) for r in linear),
        "C2_hertz_contact_time_le_2pct": max(abs(r["relative_error"]) for r in hertz) <= 0.02,
        "C3_restitution_monotone": sweep["monotone"],
        "C3_restitution_max_deviation": sweep["max_relative_deviation"],
        "C3_restitution_le_5pct": sweep["max_relative_deviation"] <= 0.05,
        "C3_restitution_r2": sweep["fit_r2"],
        "C3_restitution_r2_ge_0.99": sweep["fit_r2"] >= 0.99,
        "C4_dtcr_ratio": bisection["ratio"],
        "C4_dtcr_ratio_in_0.9_1.1": 0.9 <= bisection["ratio"] <= 1.1,
        "C5_undamped_drift": undamped["energy_drift_relative"],
        "C5_undamped_drift_lt_0.1pct": abs(undamped["energy_drift_relative"]) < 1.0e-3,
        "C5_damped_residual": damped["ledger_residual_relative"],
        "C5_damped_residual_lt_1e-6": abs(damped["ledger_residual_relative"]) < 1.0e-6,
        "C6_negative_controls": {
            "timestep_1.5dtcr_diverges": controls["timestep_over_dt_cr"]["diverged"],
            "negative_damping_gains_energy": controls["negative_damping"]["energy_increased"],
            "torn_ledger_detected": controls["torn_ledger"]["detected"],
        },
        "C6_all_negative_controls_pass": all([
            controls["timestep_over_dt_cr"]["diverged"],
            controls["negative_damping"]["energy_increased"],
            controls["torn_ledger"]["detected"],
        ]),
    }
    payload["criteria"] = criteria
    payload["runtime_seconds"] = time.time() - t_start

    figures = make_figures(payload)
    payload["figures"] = [str(p.relative_to(ROOT)) for p in figures]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "verification.json").write_text(
        json.dumps(_strip_series(payload), indent=1, sort_keys=True), encoding="utf-8")
    table = ["| family | config | value | oracle | rel. error | normal angle [rad] |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        table.append(f"| {row['family']} | {row['configuration']} | {row['value']:.12g} | "
                     f"{row['oracle']:.12g} | {row['relative_error']:.2e} | "
                     f"{row['normal_angle_rad']:.2e} |")
    (ARTIFACT_DIR / "contact_geometry_table.md").write_text("\n".join(table) + "\n",
                                                            encoding="utf-8")
    print(json.dumps(criteria, indent=1, default=float))
    print(f"artifacts: {ARTIFACT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())