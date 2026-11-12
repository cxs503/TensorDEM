#!/usr/bin/env python3
"""Case-pool benchmark: DEM-ROLLING-SINGLE-PARTICLE (TensorDEM).

Anchor / pool item
------------------
``DEM-ROLLING-SINGLE-PARTICLE`` — "单颗粒斜面纯滚动加速度" (ref: classical rigid-body
dynamics ``a = g sin(theta) / (1 + I/(m R^2))``; rolling-resistance branch after
Ai et al. 2011 Powder Technol. 206:269, DOI 10.1016/j.powtec.2010.09.030;
acceptance ``<2%`` on the centre-of-mass acceleration and the angular
acceleration at ``theta = 15 deg`` without slip).

Physics
-------
A homogeneous sphere (I = 2/5 m R^2) released from rest on an incline of angle
theta rolls without slipping when ``tan(theta) < 3.5 mu_s`` (the no-slip limit
follows from f = (2/7) m g sin(theta) <= mu_s m g cos(theta)).  Then

    a = 5/7 g sin(theta),          alpha = a / R.

If a constant directional rolling-resistance torque ``M_r = mu_r |F_n| R`` is
active (Ai et al. 2011, "cdt" family) the same force/moment balance gives the
*same* slope-reduced form

    a = 5/7 g (sin(theta) - mu_r cos(theta)),        (mu_r < tan(theta))

so ``mu_r`` acts as an effective reduction of the driving slope.  This script
measures the DEM centre-of-mass acceleration with the repository's polyhedron /
tilt-plane engine (``tensordem.tilt_plane_friction``) and compares it with the
analytic law, both as an absolute error at the acceptance angle and as an
``a(theta)`` curve.

Scope (honest)
--------------
The tilt-plane engine (``TiltPlaneCase``) exposes ``mu_s``/``mu_k`` but **no**
rolling-resistance coefficient, so the ``mu_r`` branch is *not* exercised by the
DEM here: it is reported as the analytic reference line and flagged
``NOT_SUPPORTED`` on this code path (the rolling laws themselves live in
``tensordem.rolling_resistance`` and are verified separately).  What this case
does certify is the no-slip rolling law ``a = 5/7 g sin(theta)``, the slip onset
at ``atan(3.5 mu_s)`` and the angular acceleration ``alpha = a/R``.

Usage
-----
    PYTHONPATH=src python benchmarks/dem_rolling_single_particle.py
    PYTHONPATH=src python benchmarks/dem_rolling_single_particle.py --fast

Artifacts: ``artifacts/dem_rolling_single_particle/results.json`` + ``figure.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

import tensordem.tilt_plane_friction as tp

GRAVITY = 9.81
ACCEPT_ANGLE = 15.0          # deg, pool-item acceptance angle
SPHERE_RADIUS = 0.002        # m (fast case: 15 s per measurement)

#: Angles of the a(theta) curve (all below the no-slip limit for mu_s = 0.4,
#: i.e. atan(3.5 * 0.4) = 54.5 deg, so every point must roll without slip).
CURVE_ANGLES = (10.0, 15.0, 20.0, 30.0)

#: mu_s values for the slip-onset scan (limit angle atan(3.5 mu_s)).
ONSET_MU = (0.20, 0.40)

ARTIFACT_DIR = pathlib.Path("artifacts/dem_rolling_single_particle")


def analytic_acceleration(theta_deg: float, mu_r: float = 0.0,
                          gravity: float = GRAVITY) -> float:
    """No-slip rolling acceleration ``5/7 g (sin - mu_r cos)``, m/s^2."""
    th = math.radians(theta_deg)
    return 5.0 / 7.0 * gravity * (math.sin(th) - mu_r * math.cos(th))


def analytic_alpha(theta_deg: float, radius: float = SPHERE_RADIUS,
                   mu_r: float = 0.0, gravity: float = GRAVITY) -> float:
    """Angular acceleration ``alpha = a / R``, rad/s^2."""
    return analytic_acceleration(theta_deg, mu_r, gravity) / radius


def _case(**kwargs) -> tp.TiltPlaneCase:
    base = dict(shape="sphere", rotation_locked=False, mu_s=0.4, size=SPHERE_RADIUS,
                density=1500.0, t_end=0.06)
    base.update(kwargs)
    return tp.TiltPlaneCase(**base)


def measure(theta_deg: float, *, case: tp.TiltPlaneCase | None = None) -> dict:
    """Run one DEM tilt-plane release and return the fitted kinematics."""
    case = case or _case()
    res = tp.dynamic_friction(case, theta_deg)
    a_dem = float(res["acceleration_m_s2"])
    a_ref = analytic_acceleration(theta_deg)
    return {
        "theta_deg": theta_deg,
        "mu_s": case.mu_s,
        "a_dem": a_dem,
        "a_analytic": a_ref,
        "rel_error": abs(a_dem - a_ref) / a_ref,
        "alpha_dem": a_dem / SPHERE_RADIUS,
        "alpha_analytic": analytic_alpha(theta_deg),
        "slip_speed_m_s": float(res.get("slip_speed_m_s", float("nan"))),
        "mu_measured": float(res.get("mu_measured", float("nan"))),
    }


def run_acceptance(*, fast: bool = False) -> dict:
    """Acceptance point (theta = 15 deg) + a(theta) curve + slip-onset scan."""
    angles = (ACCEPT_ANGLE,) if fast else CURVE_ANGLES
    curve = [measure(th) for th in angles]
    acceptance = next(r for r in curve if r["theta_deg"] == ACCEPT_ANGLE)

    onset = []
    for mu in (() if fast else ONSET_MU):
        limit = math.degrees(math.atan(3.5 * mu))
        below = max(1.0, limit - 3.0)
        above = min(70.0, limit + 3.0)
        rec_below = measure(below, case=_case(mu_s=mu, t_end=0.06))
        rec_above = measure(above, case=_case(mu_s=mu, t_end=0.06))
        onset.append({
            "mu_s": mu,
            "limit_deg": limit,
            "below_deg": below,
            "above_deg": above,
            "roll_error_below": rec_below["rel_error"],
            "signed_slip_ratio_above": rec_above["a_dem"] / rec_above["a_analytic"],
            "slip_speed_below": rec_below["slip_speed_m_s"],
            "slip_speed_above": rec_above["slip_speed_m_s"],
        })

    max_err = max(r["rel_error"] for r in curve)
    return {
        "pool_case": "DEM-ROLLING-SINGLE-PARTICLE",
        "engine": "tensordem.tilt_plane_friction.dynamic_friction",
        "acceptance_angle_deg": ACCEPT_ANGLE,
        "acceptance_tol": 0.02,
        "acceptance": acceptance,
        "acceptance_pass": acceptance["rel_error"] < 0.02,
        "curve": curve,
        "curve_max_rel_error": max_err,
        "slip_onset_scan": onset,
        "rolling_resistance_branch": {
            "status": "NOT_SUPPORTED on this code path",
            "reason": "TiltPlaneCase exposes mu_s/mu_k only; no rolling-resistance "
                      "coefficient is wired into the polyhedron tilt-plane engine.",
            "analytic_reference": {
                f"mu_r={mu_r}": analytic_acceleration(ACCEPT_ANGLE, mu_r)
                for mu_r in (0.0, 0.01, 0.02, 0.05)
            },
        },
        "sphere_radius_m": SPHERE_RADIUS,
    }


def make_figure(payload: dict, path: pathlib.Path) -> None:
    """a(theta) DEM points vs analytic 5/7 g sin(theta) line + mu_r references."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=140)
    th = np.linspace(0.0, 55.0, 400)
    ax.plot(th, [analytic_acceleration(t) for t in th], "-", color="tab:blue",
            lw=2.0, label=r"analytic $a=\frac{5}{7}g\sin\theta$")
    for mu_r, style in ((0.02, "--"), (0.05, ":")):
        ax.plot(th, [analytic_acceleration(t, mu_r) for t in th], style,
                color="tab:green", lw=1.4,
                label=rf"analytic with $\mu_r={mu_r}$ ($\mu_r$ branch NOT_SUPPORTED)")
    pts = payload["curve"]
    ax.plot([p["theta_deg"] for p in pts], [p["a_dem"] for p in pts], "o",
            color="tab:red", ms=6, label="DEM (TensorDEM tilt plane)")
    acc = payload["acceptance"]
    ax.axvline(ACCEPT_ANGLE, color="0.6", lw=0.8)
    ax.annotate(rf"$\theta=15^\circ$: err $={100 * acc['rel_error']:.2f}\%$",
                xy=(ACCEPT_ANGLE, acc["a_dem"]), xytext=(17.0, 0.9),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel(r"incline angle $\theta$ [deg]")
    ax.set_ylabel(r"centre-of-mass acceleration $a$ [m/s$^2$]")
    ax.set_title("DEM-ROLLING-SINGLE-PARTICLE: no-slip rolling on an incline")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="acceptance point only")
    ap.add_argument("--out-dir", default=str(ARTIFACT_DIR))
    args = ap.parse_args()

    payload = run_acceptance(fast=args.fast)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    acc = payload["acceptance"]
    print(f"theta={acc['theta_deg']:.1f} deg  a_DEM={acc['a_dem']:.6f}  "
          f"a_analytic={acc['a_analytic']:.6f}  rel_err={100 * acc['rel_error']:.3f}%")
    print(f"curve max rel_err = {100 * payload['curve_max_rel_error']:.3f}%  "
          f"acceptance(<2%) = {payload['acceptance_pass']}")
    for rec in payload["slip_onset_scan"]:
        print(f"mu_s={rec['mu_s']:.2f} limit={rec['limit_deg']:.1f} deg  "
              f"roll_err_below={100 * rec['roll_error_below']:.2f}%  "
              f"a_DEM/a_analytic above={rec['signed_slip_ratio_above']:.3f}")

    if not args.fast:
        make_figure(payload, out / "figure.png")
        print(f"figure -> {out / 'figure.png'}")
    print(f"results -> {out / 'results.json'}")
    return 0 if payload["acceptance_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())