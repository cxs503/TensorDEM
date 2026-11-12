"""Artifact generator: tilt-plane rolling resistance vs the closed form.

Run: PYTHONPATH=src python benchmarks/tilt_plane_rolling_resistance.py
Writes artifacts/tilt_plane_rolling/{result.json,fig_rolling_resistance.png}.
Every number in the report is read from result.json (no hand typing).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensordem import tilt_plane_friction as tp  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "tilt_plane_rolling"
OUT.mkdir(parents=True, exist_ok=True)
BASE = dict(shape="sphere", rotation_locked=False, mu_s=0.6, size=0.006,
            density=1500.0, t_end=0.25, samples=120)
THETA = 15.0
MU_R = (0.0, 0.02, 0.05, 0.10)


def main() -> dict:
    rows = []
    for mu_r in MU_R:
        case = tp.TiltPlaneCase(mu_rolling=mu_r, **BASE)
        r = tp.simulate(case, THETA)
        anchor = r["rolling_anchor"]
        rows.append({
            "mu_rolling": mu_r,
            "theta_deg": THETA,
            "a_measured_m_s2": r["acceleration_m_s2"],
            "a_analytic_m_s2": anchor["a_m_s2"],
            "a_analytic_no_resistance_m_s2": anchor["a_no_rolling_resistance_m_s2"],
            "rel_err_pct": abs(r["acceleration_m_s2"] / anchor["a_m_s2"] - 1.0) * 100.0,
            "slide_m": r["slide_m"],
            "slip_speed_m_s": r["slip_speed_m_s"],
            "com_speed_m_s": r["com_speed_m_s"],
            "r2": r["acceleration_r2"],
            "steps": r["steps"],
            "rolling_moment_on": r["rolling_moment_on"],
        })
    hold_case = tp.TiltPlaneCase(mu_rolling=0.25, **BASE)
    hold = tp.simulate(hold_case, 10.0)
    theta25 = tp.TiltPlaneCase(mu_rolling=0.03, **BASE)
    r25 = tp.simulate(theta25, 25.0)
    payload = {
        "case": {**BASE, "theta_deg": THETA, "mu_rolling_values": MU_R},
        "rows": rows,
        "second_angle": {
            "theta_deg": 25.0, "mu_rolling": 0.03,
            "a_measured_m_s2": r25["acceleration_m_s2"],
            "a_analytic_m_s2": r25["rolling_anchor"]["a_m_s2"],
            "rel_err_pct": abs(r25["acceleration_m_s2"]
                               / r25["rolling_anchor"]["a_m_s2"] - 1.0) * 100.0,
        },
        "hold_case": {
            "theta_deg": 10.0, "mu_rolling": 0.25,
            "tan_theta": math.tan(math.radians(10.0)),
            "a_analytic_m_s2": hold["rolling_anchor"]["a_m_s2"],
            "moved": hold["moved"],
            "slide_m": hold["slide_m"],
            "slide_tolerance_m": hold["slide_tolerance_m"],
        },
        "formula": "a = g (sin(theta) - mu_r cos(theta)) / (1 + 2/5)  [solid sphere]",
        "reference": ("Ai, Chen, Rotter & Ooi (2011) Powder Technol. 206:72 "
                      "capped EPSD rolling moment M_r = mu_r R F_n"),
    }
    (OUT / "result.json").write_text(json.dumps(payload, indent=1, sort_keys=True))

    # --- figure: measured vs closed form, and the hold boundary -------------
    mu_grid = np.linspace(0.0, 0.12, 200)
    a_grid = [tp.rolling_acceleration_analytic(
        tp.TiltPlaneCase(mu_rolling=float(m), **BASE), THETA)["a_m_s2"]
        for m in mu_grid]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(mu_grid, a_grid, "-", color="0.25", lw=2,
            label=r"$a=\frac{5}{7}g(\sin\theta-\mu_r\cos\theta)$")
    ax.plot([r["mu_rolling"] for r in rows], [r["a_measured_m_s2"] for r in rows],
            "o", ms=7, mfc="tab:red", mec="k", label="DEM measured (fitted)")
    for r in rows:
        ax.annotate(f"{r['rel_err_pct']:.2e}%", (r["mu_rolling"], r["a_measured_m_s2"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.axhline(0.0, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel(r"rolling resistance coefficient $\mu_r$")
    ax.set_ylabel(r"down-slope acceleration $a$ (m/s$^2$)")
    ax.set_title(f"Tilt plane, sphere, $\\theta={THETA:g}^\\circ$: DEM vs closed form")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    ax2 = axes[1]
    th_grid = np.linspace(1.0, 40.0, 400)
    a_frac = []
    for t in th_grid:
        c = tp.TiltPlaneCase(mu_rolling=0.05, **BASE)
        a_frac.append(tp.rolling_acceleration_analytic(c, float(t))["a_m_s2"])
    ax2.plot(th_grid, a_frac, "-", color="tab:blue", lw=2, label=r"$\mu_r=0.05$")
    ax2.plot(np.degrees(np.arctan(mu_grid)), mu_grid, "--", color="0.6", lw=1,
             label=r"rolling limit $\tan\theta=\mu_r$")
    hold_deg = math.degrees(math.atan(0.25))
    ax2.axvline(hold_deg, color="tab:red", lw=1, ls=":")
    ax2.annotate(r"hold for $\mu_r=0.25$: $\theta<%.1f^\circ$" % hold_deg,
                 (hold_deg + 0.6, 1.2), fontsize=8, color="tab:red")
    ax2.axhline(0.0, color="0.7", lw=0.8, ls=":")
    ax2.set_xlabel(r"incline angle $\theta$ (deg)")
    ax2.set_ylabel(r"$a$ (m/s$^2$)")
    ax2.set_title("Rolling-resistance hold boundary")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rolling_resistance.png", dpi=150)
    print(json.dumps({"worst_rel_err_pct": max(r["rel_err_pct"] for r in rows),
                      "rows": rows}, indent=1))
    return payload


if __name__ == "__main__":
    main()
