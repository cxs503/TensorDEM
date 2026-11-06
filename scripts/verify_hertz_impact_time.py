"""Evidence generator for the Hertz impact-duration verification (m4747).

Re-runs the time-domain verification of the DEM3D Hertz contact law against the
classical Johnson (1985, Sec. 4.2) closed form and dumps every number into
``artifacts/hertz_impact_time_m4747/`` (JSON + CSV + PNG).  Every table in the
companion document is injected from this JSON -- nothing is hand-copied.

Anchors measured here
---------------------
A. prefactor: ``C = 2 int_0^1 du/sqrt(1-u^(5/2)) = 2.9427`` vs the textbook 2.94
B. duration:   ``t_c`` from the shipped solver vs ``C (5m/4k_h)^(2/5) v^(-1/5)``
C. exponent:   log-log slope of ``t_c`` vs ``v0`` = -1/5 (Hertz signature)
D. negative control: linear contact law -> ``t_c = pi sqrt(m/k)``, slope 0
E. peak overlap: ``delta_max`` vs ``(5 m v0^2 / 4 k_h)^(2/5)``, slope 4/5
F. dt convergence: the raw bias is exactly one step (O(dt)) and vanishes
G. trajectory: solver overlap(t) vs an independent RK4 integration of the
   Hertz oscillator (time-domain comparison, the figure's left panel)

Run:  python3 scripts/verify_hertz_impact_time.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordem.hertz_impact_time import (
    JOHNSON_PREFACTOR,
    fit_loglog_exponent,
    hertz_contact_time,
    hertz_contact_time_integral,
    hertz_peak_overlap,
    linear_contact_time,
    measure_platen_impact,
    rigid_plane_hertz_stiffness,
    sweep_speeds,
)

RADIUS = 0.025
DENSITY = 917.0
YOUNG = 1.0e9
POISSON = 0.3
PLATEN_K = 5_000.0
STEPS_PER_CONTACT = 600
TEXTBOOK_PREFACTOR = 2.94

MASS = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
K_H = rigid_plane_hertz_stiffness(YOUNG, POISSON, RADIUS)

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "hertz_impact_time_m4747"


def rk4_reference_trajectory(v_in: float, samples: int = 1200) -> list[tuple[float, float]]:
    """Independent RK4 integration of ``m delta'' = -k_h delta^(3/2)``.

    Written from the textbook ODE, not from the solver; used only to overlay a
    reference curve on the measured overlap history.
    """
    t_c = hertz_contact_time(v_in, MASS, K_H)
    dt = t_c / samples
    delta, rate, t = 0.0, v_in, 0.0
    out = [(0.0, 0.0)]
    for _ in range(4 * samples):
        def acc(d: float) -> float:
            return -K_H * max(d, 0.0) ** 1.5 / MASS

        a1 = acc(delta)
        d2, r2 = delta + 0.5 * dt * rate, rate + 0.5 * dt * a1
        a2 = acc(d2)
        d3, r3 = delta + 0.5 * dt * r2, rate + 0.5 * dt * a2
        a3 = acc(d3)
        d4, r4 = delta + dt * r3, rate + dt * a3
        a4 = acc(d4)
        new_delta = delta + dt * (rate + 2.0 * r2 + 2.0 * r3 + r4) / 6.0
        new_rate = rate + dt * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0
        t += dt
        delta, rate = new_delta, new_rate
        out.append((t, delta))
        if delta <= 0.0 and rate < 0.0 and t > dt:
            break
    return out


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    evidence: dict = {
        "protocol": "hertz-impact-duration-johnson-v1",
        "configuration": {
            "radius_m": RADIUS, "density_kg_m3": DENSITY,
            "young_modulus_Pa": YOUNG, "poisson_ratio": POISSON,
            "platen_stiffness_N_m": PLATEN_K,
            "particle_mass_kg": MASS,
            "rigid_plane_hertz_stiffness_N_m15": K_H,
            "steps_per_contact": STEPS_PER_CONTACT,
            "contact_damping": 0.0, "drag": 0.0, "platen_damping": 0.0,
            "gravity": 0.0, "friction": 0.0,
            "grid": "2x2x2 particles, lowest-inner particle driven downward",
        },
        "prefactor": {
            "integral_I": hertz_contact_time_integral(),
            "C_beta_function": JOHNSON_PREFACTOR,
            "C_textbook": TEXTBOOK_PREFACTOR,
            "rel_error_vs_textbook": JOHNSON_PREFACTOR / TEXTBOOK_PREFACTOR - 1.0,
        },
    }

    speeds = [0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
    hertz_rows = sweep_speeds(speeds, contact_model="hertz",
                              steps_per_contact=STEPS_PER_CONTACT)
    linear_rows = sweep_speeds(speeds[:4], contact_model="linear",
                               steps_per_contact=STEPS_PER_CONTACT)

    def summarise(rows, ref_key):
        durations = [r["measured_duration_s"] for r in rows]
        refs = [r[ref_key] for r in rows]
        worst = max(abs(r["duration_rel_error"]) for r in rows)
        return {
            "rows": rows,
            "worst_abs_duration_rel_error": worst,
            "exponent_fit": fit_loglog_exponent(
                [r["v_in"] for r in rows], durations),
            "worst_abs_overlap_rel_error": max(
                abs(r["peak_overlap_rel_error"]) for r in rows),
            "overlap_exponent_fit": fit_loglog_exponent(
                [r["v_in"] for r in rows], [r["peak_overlap_m"] for r in rows]),
            "reference_values": refs,
        }

    evidence["hertz"] = summarise(hertz_rows, "duration_reference_s")
    evidence["linear_control"] = summarise(linear_rows, "duration_reference_s")

    # F: dt convergence of the raw bias at v0 = 1 m/s (bias must be O(dt)).
    convergence = []
    t_c_ref = hertz_contact_time(1.0, MASS, K_H)
    for steps in (300, 600, 1200, 2400):
        run = measure_platen_impact(v_in=1.0, dt=t_c_ref / steps,
                                    steps_per_contact=steps)
        convergence.append({
            "steps_per_contact": steps,
            "dt": run["dt"],
            "measured_duration_s": run["measured_duration_s"],
            "duration_rel_error": run["measured_duration_s"] / t_c_ref - 1.0,
        })
    evidence["dt_convergence"] = {
        "v_in": 1.0,
        "reference_duration_s": t_c_ref,
        "runs": convergence,
    }

    # G: trajectory comparison at v0 = 1 m/s.
    traj = measure_platen_impact(v_in=1.0, dt=t_c_ref / STEPS_PER_CONTACT,
                                 steps_per_contact=STEPS_PER_CONTACT,
                                 record_history=True)
    history = traj["history"]
    reference = rk4_reference_trajectory(1.0)
    evidence["trajectory"] = {
        "v_in": 1.0,
        "solver_samples": len(history),
        "reference_samples": len(reference),
        "solver_peak_overlap_m": max(abs(h[1]) for h in history),
        "reference_peak_overlap_m": max(abs(r[1]) for r in reference),
        "reference_duration_s": reference[-1][0],
        "solver_duration_s": traj["measured_duration_s"],
        "max_abs_overlap_difference_m": None,
    }
    # Overlap difference on the common time window (linear interpolation of the
    # solver history, which is the coarser of the two grids).
    def interp(t: float) -> float:
        for i in range(1, len(history)):
            t0, o0, _ = history[i - 1]
            t1, o1, _ = history[i]
            if t <= t1:
                w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return o0 + w * (o1 - o0)
        return history[-1][1]

    window = min(history[-1][0], reference[-1][0])
    diffs = [abs(interp(t) - o) for t, o in reference if t <= window]
    evidence["trajectory"]["max_abs_overlap_difference_m"] = max(diffs)
    evidence["trajectory"]["window_s"] = window

    # ---------------------------------------------------------------- CSV
    csv_path = OUT / "duration_scaling.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "contact_model", "v_in_m_s", "measured_duration_s",
            "reference_duration_s", "duration_rel_error",
            "measured_peak_overlap_m", "reference_peak_overlap_m",
            "restitution_measured", "dt_s", "contact_steps",
        ])
        for row in hertz_rows + linear_rows:
            writer.writerow([
                row["contact_model"], f"{row['v_in']:.6g}",
                f"{row['measured_duration_s']:.10e}",
                f"{row['duration_reference_s']:.10e}",
                f"{row['duration_rel_error']:+.6e}",
                f"{row['peak_overlap_m']:.10e}",
                f"{row['peak_overlap_reference_m']:.10e}",
                f"{row['restitution_measured']:.12f}",
                f"{row['dt']:.6e}", row["contact_steps"],
            ])
    evidence["csv"] = str(csv_path.relative_to(OUT.parents[1]))

    # --------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.6))

    ax = axes[0]
    t_solver = [h[0] * 1e3 for h in history]
    o_solver = [h[1] * 1e3 for h in history]
    ax.plot([t for t, _ in reference], [o * 1e3 for _, o in reference],
            "-", color="#1f4e79", lw=1.8, label="RK4 Hertz reference (textbook ODE)")
    ax.plot(t_solver, o_solver, "o", ms=3.0, color="#c00000", alpha=0.6,
            label="DEM3D solver (dt = t_c/600)")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("platen overlap $\\delta$ [mm]")
    ax.set_title("A. one elastic impact, $v_0$ = 1 m/s\n"
                 f"$\\delta_{{max}}$: solver {max(abs(h[1]) for h in history)*1e3:.4f} mm, "
                 f"theory {max(abs(r[1]) for r in reference)*1e3:.4f} mm")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    vh = [r["v_in"] for r in hertz_rows]
    th = [r["measured_duration_s"] * 1e3 for r in hertz_rows]
    refh = [r["duration_reference_s"] * 1e3 for r in hertz_rows]
    vl = [r["v_in"] for r in linear_rows]
    tl = [r["measured_duration_s"] * 1e3 for r in linear_rows]
    refl = [r["duration_reference_s"] * 1e3 for r in linear_rows]
    grid_v = [0.2 * (1.5 ** i) for i in range(14)]
    ax.loglog(grid_v, [hertz_contact_time(v, MASS, K_H) * 1e3 for v in grid_v],
              "-", color="#1f4e79", lw=1.6,
              label=f"Johnson: $t_c\\propto v_0^{{-1/5}}$ (C={JOHNSON_PREFACTOR:.4f})")
    ax.loglog(vh, th, "o", ms=6, color="#c00000",
              label=f"DEM3D Hertz (slope {evidence['hertz']['exponent_fit']:+.4f})")
    ax.loglog(grid_v, [linear_contact_time(MASS, PLATEN_K) * 1e3 for _ in grid_v],
              "--", color="#548235", lw=1.4,
              label="linear law: $\\pi\\sqrt{m/k}$ (v-independent)")
    ax.loglog(vl, tl, "s", ms=5, color="#548235", alpha=0.8,
              label=f"DEM3D linear control (slope "
                    f"{evidence['linear_control']['exponent_fit']:+.4f})")
    ax.set_xlabel("impact speed $v_0$ [m/s]")
    ax.set_ylabel("contact duration $t_c$ [ms]")
    ax.set_title("B. duration scaling: Hertz $v_0^{-1/5}$ vs linear $v_0^0$")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    steps = [c["steps_per_contact"] for c in convergence]
    err = [c["duration_rel_error"] * 100.0 for c in convergence]
    ax.plot(steps, err, "o-", color="#c00000", label="raw bias (measured)")
    ax.plot([s for s in steps], [-100.0 / s for s in steps], ":", color="#404040",
            label="$-1/N$ (one-step quantisation, O(dt))")
    ax.set_xscale("log")
    ax.set_xlabel("integration steps per contact $N = t_c/\\Delta t$")
    ax.set_ylabel("relative duration error [%]")
    ax.set_title("C. fidelity: bias is exactly one step and vanishes with $\\Delta t$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Hertz contact duration vs Johnson closed form — TensorDEM m4747", y=1.01)
    fig.tight_layout()
    fig_path = OUT / "hertz_impact_duration.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    evidence["figure"] = str(fig_path.relative_to(OUT.parents[1]))

    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2))

    # ------------------------------------------------------------ console
    print(f"prefactor C = {JOHNSON_PREFACTOR:.6f} (textbook {TEXTBOOK_PREFACTOR}) "
          f"-> {evidence['prefactor']['rel_error_vs_textbook']*100:+.4f} %")
    print(f"hertz : worst |dt error| = {evidence['hertz']['worst_abs_duration_rel_error']*100:.4f} %"
          f"   slope = {evidence['hertz']['exponent_fit']:+.4f}"
          f"   overlap slope = {evidence['hertz']['overlap_exponent_fit']:+.4f}")
    print(f"linear: worst |dt error| = "
          f"{evidence['linear_control']['worst_abs_duration_rel_error']*100:.4f} %"
          f"   slope = {evidence['linear_control']['exponent_fit']:+.4f}")
    print("dt convergence:", [(c["steps_per_contact"],
                               round(c["duration_rel_error"] * 100, 4))
                              for c in convergence])
    print(f"trajectory: max |overlap difference| = "
          f"{evidence['trajectory']['max_abs_overlap_difference_m']*1e6:.3f} um")
    print("artifacts ->", OUT)
    return evidence


if __name__ == "__main__":
    main()