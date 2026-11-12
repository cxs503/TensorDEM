"""CASE-DEM-PLANE-SHEAR-MUIDI round 5: constant-stress validity gate.

Why this script exists
----------------------
``tensordem.mu_i_rheology.simulate_servo_shear`` is documented as a
*constant-stress* servo shear cell, and the mu(I)/phi(I) case built on it
(rounds 1-3) reads ``mu = tau_wall / p_wall`` with ``p_wall`` the platen
normal stress.  A reproduction of the stored round-3 artifacts with the
current code (identical mu to 1e-15, so the artifacts were not stale) shows
that the platen **does not hold the applied stress at high I**:

    I_target   arm       p_meas/p_applied - 1
    0.3        baseline  -16.5 %
    0.3        thick     +57.8 %

so a point with I >~ 0.1 was never a constant-stress measurement, and the
reported ``mu`` is a ratio of two drifting wall tractions.

Round 5 adds a validity gate to the device (``pressure_control_tol`` /
``pressure_recover_window`` / ``max_recover_steps``): when the requested
strain budget is spent but the servo has not held the applied stress, the
shear phase continues (same imposed shear rate, same contact law, no physics
change) until the servo error is inside the tolerance for a sustained window.
This script measures the effect: the same (arm, I) points are run with the
gate off (``tol=1e9`` reproduces rounds 1-3 bit-for-bit) and on.

Usage
-----
    python servo_validity_fix.py --arm baseline --I 0.3 --gate on
    python servo_validity_fix.py --aggregate      # tables + figure

Artifacts: ``runs/servo_validity_gate.json`` (raw) and
``runs/servo_validity_gate_tables.md`` + ``runs/fig_servo_validity.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_sweep as rs  # noqa: E402  (device configuration of rounds 1-3)
from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear  # noqa: E402

RUNS = HERE / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
RAW = RUNS / "servo_validity_gate.json"

ARMS = {
    # arm -> (box_depth, fill_height, layers, box_depth_d)
    "baseline": (rs.D, rs.H, 6.0, 5.0),
    "stiff": (rs.D, rs.H, 6.0, 5.0),      # kn arm handled by run_point below
    "thick": (0.08, 0.09, 9.0, 8.0),
}
STORED = {"baseline": "sweep_baseline_round2.json",
          "stiff": "sweep_stiff.json",
          "thick": "sweep_thick.json"}


def load_stored() -> dict:
    """Round-3 artifact rows keyed by (arm, I_target) -> measurement."""
    out = {}
    for arm, fname in STORED.items():
        path = RUNS / fname
        if not path.exists():
            continue
        for row in json.loads(path.read_text())["rows"]:
            out[(arm, round(float(row["I_target"]), 6))] = {
                "mu": row["mu_measured"],
                "phi": row["phi_bed_steady"],
                "p_meas": row.get("normal_stress_pa"),
                "mu_i": row["I_measured"],
                "pressure_control_error": (
                    row["normal_stress_pa"] / rs.P - 1.0
                    if row.get("normal_stress_pa") else None),
            }
    return out


def run_point(arm: str, I: float, gate: bool) -> dict:
    box_depth, fill_height = ARMS[arm][0], ARMS[arm][1]
    kn = rs.KN if arm != "stiff" else 10.0 * rs.KN
    scale = rs.stiffness_step_scale(kn)
    kwargs = (dict(pressure_control_tol=0.05, pressure_recover_window=200,
                   max_recover_steps=20000) if gate
              else dict(max_recover_steps=0))   # gate off == rounds 1-3 protocol
    case = ServoShearCase(
        inertial_number=I, mu=rs.MU_CONTACT, mu_rolling=rs.MU_ROLLING,
        restitution=rs.E, kn=kn, normal_stress=rs.P,
        shear_strain=rs.SHEAR_STRAIN,
        settle_steps=int(round(rs.SETTLE_STEPS_BASE * scale)),
        min_shear_steps=int(round(rs.MIN_SHEAR_STEPS_BASE * scale)),
        packing_gap=0.02, jitter=0.01, seed=rs.SEED, radius=rs.R,
        density=rs.RHO, box_width=rs.W, box_depth=box_depth,
        fill_height=fill_height, **kwargs)
    t0 = time.perf_counter()
    res = simulate_servo_shear(case)
    wall = time.perf_counter() - t0
    return {
        "arm": arm, "I_target": I, "gate": bool(gate), "kn": kn,
        "box_depth": box_depth, "fill_height": fill_height,
        "n_particles": res["n_particles"],
        "I_measured": res["inertial_number"],
        "mu": res["mu_effective"],
        "tau": res["shear_stress_pa"],
        "p_meas": res["normal_stress_measured_pa"],
        "p_applied": rs.P,
        "pressure_control_error": res["pressure_control_error"],
        "pressure_control_error_final": res["pressure_control_error_final"],
        "mu_final": res["mu_final"],
        "phi_final": res["phi_bed_final"],
        "servo_valid": res["servo_valid"],
        "recover_steps": res["recover_steps"],
        "shear_steps_executed": res["shear_steps_executed"],
        "phi": res["phi_bed_steady"],
        "phi_device": res["solid_fraction"],
        "dilatancy": res["dilatancy"],
        "coordination": res["coordination_number"],
        "strain_achieved": res["shear_strain_achieved"],
        "ref_mu": rs.ref_mu(res["inertial_number"]),
        "ref_phi": rs.ref_phi(res["inertial_number"]),
        "wall_clock_s": wall,
    }


def _merge(rec: dict) -> None:
    rows = []
    if RAW.exists():
        rows = json.loads(RAW.read_text()).get("rows", [])
    rows = [r for r in rows
            if not (r["arm"] == rec["arm"] and r["I_target"] == rec["I_target"]
                    and r["gate"] == rec["gate"])]
    rows.append(rec)
    rows.sort(key=lambda r: (r["arm"], r["I_target"], not r["gate"]))
    RAW.write_text(json.dumps({"case": "CASE-DEM-PLANE-SHEAR-MUIDI",
                               "round": 5, "rows": rows}, indent=1))


def aggregate() -> None:
    rows = json.loads(RAW.read_text())["rows"]
    stored = load_stored()
    lines = ["### Table A - stored round-3 points vs constant-stress validity "
             "(gate off = rounds 1-3 protocol)",
             "",
             "| arm | I_target | p_meas (Pa) | p error | mu | phi_bed_steady | valid (<5%) |",
             "|---|---|---|---|---|---|---|"]
    for arm in ("baseline", "stiff", "thick"):
        for I in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
            s = stored.get((arm, round(I, 6)))
            if not s or s["pressure_control_error"] is None:
                continue
            err = s["pressure_control_error"]
            lines.append(
                f"| {arm} | {I:g} | {s['p_meas']:.0f} | {err*100:+.1f}% | "
                f"{s['mu']:.4f} | {s['phi']:.4f} | "
                f"{'YES' if abs(err) <= 0.05 else '**NO**'} |")

    lines += ["", "### Table B - gate off vs gate on, same (arm, I) point "
              "(mu/phi on the **final 20 % window**, which is the window the "
              "gate settles on)", "",
              "| arm | I_target | gate | p error (tail) | p error (final) | "
              "servo_valid | recover steps | mu final | phi final | "
              "mu err vs ref | wall clock (s) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    by = {(r["arm"], r["I_target"], r["gate"]): r for r in rows}
    for arm in ("baseline", "thick"):
        for I in sorted({r["I_target"] for r in rows if r["arm"] == arm}):
            for tag, gate in (("off", False), ("on", True)):
                r = by.get((arm, I, gate))
                if not r:
                    lines.append(f"| {arm} | {I:g} | {tag} | (not run) | | | | | | | |")
                    continue
                lines.append(
                    f"| {arm} | {I:g} | {tag} | "
                    f"{r['pressure_control_error']*100:+.1f}% | "
                    f"{r['pressure_control_error_final']*100:+.1f}% | "
                    f"{'yes' if r['servo_valid'] else 'no'} | "
                    f"{r['recover_steps']} | {r['mu_final']:.4f} | "
                    f"{r['phi_final']:.4f} | "
                    f"{(r['mu_final']/rs.ref_mu(r['I_measured'])-1)*100:+.1f}% | "
                    f"{r['wall_clock_s']:.0f} |")
    lines += ["", "### Table C - phi(I) reference sign check", "",
              "| I | ref_phi (in-repo law) | d(ref_phi)/dI | measured phi "
              "(thick, gate on) |", "|---|---|---|---|"]
    refs = [(I, rs.ref_phi(I)) for I in (0.001, 0.01, 0.1, 0.3)]
    for I, rp in refs:
        m = next((r["phi"] for r in rows
                  if r["arm"] == "thick" and r["gate"] and abs(r["I_target"]-I) < 1e-9),
                 None)
        lines.append(f"| {I:g} | {rp:.4f} | {'+' if I > 0.001 else '-'} | "
                     f"{'%.4f' % m if m else 'n/a'} |")
    out = RUNS / "servo_validity_gate_tables.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- figure ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    Iline = np.logspace(-3, math.log10(0.35), 200)
    ax[0].plot(Iline, [rs.ref_mu(i) for i in Iline], "k-", lw=1.4,
               label="da Cruz 2005 mu(I) ref")
    ax[1].plot(Iline, [rs.ref_phi(i) for i in Iline], "k--", lw=1.4,
               label="in-repo phi(I) ref (sign contradicted)")
    colours = {"baseline": "tab:blue", "thick": "tab:orange"}
    for arm, mk in (("baseline", "o"), ("thick", "s")):
        for gate, fill in ((False, "white"), (True, colours[arm])):
            pts = sorted([r for r in rows if r["arm"] == arm and r["gate"] == gate],
                         key=lambda r: r["I_measured"])
            if not pts:
                continue
            lbl = f"{arm} gate={'on' if gate else 'off'}"
            ax[0].plot([p["I_measured"] for p in pts], [p["mu_final"] for p in pts],
                       marker=mk, ls="-" if gate else "--",
                       color=colours[arm], mfc=fill, label=lbl)
            ax[1].plot([p["I_measured"] for p in pts], [p["phi_final"] for p in pts],
                       marker=mk, ls="-" if gate else "--",
                       color=colours[arm], mfc=fill, label=lbl)
    ax[0].set_xscale("log"); ax[1].set_xscale("log")
    ax[0].set_xlabel("I"); ax[0].set_ylabel(r"$\mu$")
    ax[1].set_xlabel("I"); ax[1].set_ylabel(r"$\phi$")
    ax[0].set_title("wall-traction friction coefficient")
    ax[1].set_title("bed solid fraction")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RUNS / "fig_servo_validity.png", dpi=140)
    print(f"wrote {out} and {RUNS/'fig_servo_validity.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--I", type=float)
    ap.add_argument("--gate", choices=("on", "off"))
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate(); return
    if not (a.arm and a.I is not None and a.gate):
        ap.error("need --arm --I --gate, or --aggregate")
    rec = run_point(a.arm, a.I, a.gate == "on")
    _merge(rec)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()